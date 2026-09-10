#!/usr/bin/env python3
"""Evaluate a sealed P1-03 1k LUMIA cache under the frozen outer-OOF protocol."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import platform
import time

import numpy as np
import pandas as pd

from poisoned_chalice.lumia_hidden_state_cache import OUTPUT_BASELINES, sha256_file
from poisoned_chalice.lumia_hidden_state_probe import (
    HIDDEN_SCORE_NAMES,
    LumiaProbeConfig,
    join_public_labels,
    load_sealed_cache,
    paired_bootstrap_delta,
    run_outer_oof,
    summarize_evaluation,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL_CONFIG_PATH = ROOT / "configs/p1_03_lumia_hidden_state_evaluation_v1_20260910.json"
SCIENCE_CONFIG_PATH = ROOT / "configs/p1_03_lumia_hidden_state_pilot_v1_20260909.json"
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")


def load_public_train_labels() -> pd.DataFrame:
    parts = []
    for language in LANGUAGES:
        path = ROOT / f"data/public/{language}/train-00000-of-00001.parquet"
        frame = pd.read_parquet(path, columns=["sample_id", "membership"])
        frame["language"] = language
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    if len(result) != 49040 or result.sample_id.duplicated().any():
        raise RuntimeError("public train label table identity changed")
    return result


def helper_language_bootstrap(
    frame: pd.DataFrame,
    hidden_scores: dict[str, np.ndarray],
    config: LumiaProbeConfig,
) -> dict[str, dict]:
    helper = hidden_scores["lumia_helper_language_aware_top5_ensemble"]
    caller = hidden_scores["lumia_caller_literal_top5_ensemble"]
    result = {}
    for language_index, language in enumerate(LANGUAGES):
        index = frame.index[frame.language == language].to_numpy()
        subset = frame.loc[index].reset_index(drop=True)
        result[language] = paired_bootstrap_delta(
            subset,
            helper[index],
            caller[index],
            replicates=config.bootstrap_replicates,
            seed=config.seed + 70000 + language_index,
        )
    return result


def evaluate_gate(summary: dict, helper_language: dict) -> dict:
    metrics = summary["metrics"]
    overlap = summary["overlap_vs_zsigmoid"]
    caller_name = "lumia_caller_literal_top5_ensemble"
    reference_name = "zsigmoid_single_sequence"
    caller = metrics[caller_name]
    reference = metrics[reference_name]
    conditions = {
        "caller_auc_plus_0_015": bool(caller["auc"] >= reference["auc"] + 0.015),
        "caller_tpr_01_plus_0_01_absolute": bool(
            caller["tpr_at_0.01_fpr"] >= reference["tpr_at_0.01_fpr"] + 0.01
        ),
        "caller_unique_tp_at_least_10": bool(
            overlap[caller_name]["unique_vs_zsigmoid"] >= 10
        ),
        "helper_language_auc_bootstrap_lower_bound_positive": bool(
            any(
                payload["auc_delta"]["lower_95"] > 0.0
                for payload in helper_language.values()
            )
        ),
    }
    return {
        "conditions": conditions,
        "scale_to_2000_or_more": any(conditions.values()),
        "caller_auc_delta_vs_zsigmoid": float(caller["auc"] - reference["auc"]),
        "caller_tpr_01_delta_vs_zsigmoid": float(
            caller["tpr_at_0.01_fpr"] - reference["tpr_at_0.01_fpr"]
        ),
        "caller_unique_tp_vs_zsigmoid": int(overlap[caller_name]["unique_vs_zsigmoid"]),
        "helper_language_gate_definition": "helper top5 minus caller top5 within-language AUC paired-bootstrap lower_95 > 0 for at least one language",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Cache integrity is validated before public membership labels are loaded.
    arrays, metadata, cache_manifest = load_sealed_cache(args.cache_dir)
    public_labels = load_public_train_labels()
    frame = join_public_labels(metadata, public_labels)
    del public_labels

    import torch
    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA evaluation device is unavailable")

    config = LumiaProbeConfig()
    started = time.perf_counter()
    hidden_scores, fold_records = run_outer_oof(arrays, frame, config, device=device)
    evaluation = summarize_evaluation(frame, hidden_scores, config)
    helper_language = helper_language_bootstrap(frame, hidden_scores, config)
    gate = evaluate_gate(evaluation, helper_language)
    wall = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics_payload = {
        "schema_version": 1,
        "task_id": "P1-03-lumia-hidden-state-evaluation-1000-v1",
        "probe_config": asdict(config),
        "device": device,
        "wall_seconds": wall,
        **evaluation,
        "helper_language_paired_bootstrap_vs_caller": helper_language,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "fold_selection.json").write_text(
        json.dumps(fold_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    decision = {
        "schema_version": 1,
        "task_id": "P1-03-lumia-hidden-state-evaluation-1000-v1",
        "gate": gate,
        "anti_posthoc": {
            "outer_holdout_used_for_layer_selection": False,
            "visible_validation_used": False,
            "public_lb_used": False,
            "result_derived_formula_added": False,
        },
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    predictions = frame[["sample_id", "language", "label", "seq_len", "at_max_length"]].copy()
    for baseline in OUTPUT_BASELINES:
        predictions[baseline] = frame[baseline].to_numpy(float)
    for name in HIDDEN_SCORE_NAMES:
        predictions[name] = hidden_scores[name]
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    run_manifest = {
        "schema_version": 1,
        "status": "complete",
        "task_id": "P1-03-lumia-hidden-state-evaluation-1000-v1",
        "cache_manifest_sha256": sha256_file(args.cache_dir / "cache_manifest.json"),
        "evaluation_config_sha256": sha256_file(EVAL_CONFIG_PATH),
        "scientific_config_sha256": sha256_file(SCIENCE_CONFIG_PATH),
        "rows": len(frame),
        "outer_oof_coverage": bool(all(np.isfinite(hidden_scores[name]).all() for name in HIDDEN_SCORE_NAMES)),
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "persistent_outputs": [
            "metrics.json", "fold_selection.json", "decision.json", "oof_predictions.csv", "run_manifest.json"
        ],
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate": gate, "wall_seconds": wall}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
