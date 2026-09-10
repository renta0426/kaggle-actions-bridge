#!/usr/bin/env python3
"""Evaluate the sealed P1-03a 5k hidden-state cache under the frozen scale protocol."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import platform
import time

import numpy as np
import pandas as pd

from poisoned_chalice.lumia_hidden_state_cache import OUTPUT_BASELINES, POOLING_VARIANTS, sha256_file
from poisoned_chalice.lumia_hidden_state_probe import (
    HIDDEN_SCORE_NAMES,
    LumiaProbeConfig,
    paired_bootstrap_delta,
    summarize_evaluation,
)
from poisoned_chalice.lumia_hidden_state_scale import (
    LANGUAGES,
    EXPECTED_PER_LANGUAGE_LABEL,
    EXPECTED_ROWS,
    paired_increment_bootstraps,
    run_outer_oof_5000,
)

ROOT = Path(__file__).resolve().parents[1]
SCALE_CONFIG_PATH = ROOT / "configs/p1_03_lumia_hidden_state_scale_5000_v1_20260910.json"
EXPECTED_CACHE_MANIFEST_SHA256 = "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa"
KNOWN_LEGACY_CACHE_TASK_ID = "P1-03-lumia-hidden-state-cache-1000-v1"
TASK_ID = "P1-03a-lumia-hidden-state-evaluation-5000-v1"
MODEL_ID = "bigcode/starcoder2-3b"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
DATASET_ID = "Poisoned-Chalice/ICSE-2027-public"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def load_sealed_cache_5000(root: str | Path) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    """Load the exact completed 5k artifact and fail closed on any identity drift.

    The completed cache has one known metadata-only defect: cache_manifest.task_id
    remained the frozen core's 1k literal.  This loader accepts that exact legacy
    value only; all scientific identities are checked independently.
    """
    root = Path(root)
    manifest_path = root / "cache_manifest.json"
    observed_manifest_sha = sha256_file(manifest_path)
    if observed_manifest_sha != EXPECTED_CACHE_MANIFEST_SHA256:
        raise ValueError(
            f"5k cache manifest identity changed: {observed_manifest_sha} != {EXPECTED_CACHE_MANIFEST_SHA256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config") or {}
    exact_config = {
        "rows": EXPECTED_ROWS,
        "rows_per_language_label": EXPECTED_PER_LANGUAGE_LABEL,
        "seed": 20260909,
        "max_length": 8192,
        "shard_size": 250,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dtype": "float32",
        "compression": "npz_compressed",
    }
    for key, value in exact_config.items():
        if config.get(key) != value:
            raise ValueError(f"5k cache config changed: {key}={config.get(key)!r}, expected {value!r}")
    if manifest.get("task_id") != KNOWN_LEGACY_CACHE_TASK_ID:
        raise ValueError("known 5k cache task_id compatibility marker changed")
    if manifest.get("rows") != EXPECTED_ROWS:
        raise ValueError("5k cache row count changed")
    if manifest.get("num_layers") != 30 or manifest.get("hidden_size") != 3072:
        raise ValueError("5k cache hidden dimensions changed")
    if manifest.get("labels_present_during_model_scoring") is not False:
        raise ValueError("5k cache label boundary changed")
    if manifest.get("performance_metrics_computed") is not False:
        raise ValueError("5k cache unexpectedly contains performance evaluation")
    if manifest.get("raw_token_layer_activations_persisted") is not False:
        raise ValueError("5k cache raw token-layer persistence boundary changed")
    if manifest.get("logits_persisted") is not False:
        raise ValueError("5k cache logits persistence boundary changed")
    if tuple(manifest.get("pooling_variants") or []) != POOLING_VARIANTS:
        raise ValueError("5k cache pooling variants changed")
    if tuple(manifest.get("output_baselines") or []) != OUTPUT_BASELINES:
        raise ValueError("5k cache output baselines changed")

    metadata_info = manifest.get("metadata") or {}
    metadata_path = root / str(metadata_info.get("sample_metadata"))
    cohort_path = root / str(metadata_info.get("cohort_manifest"))
    if sha256_file(metadata_path) != metadata_info.get("sample_metadata_sha256"):
        raise ValueError("5k sample metadata hash mismatch")
    if sha256_file(cohort_path) != metadata_info.get("cohort_manifest_sha256"):
        raise ValueError("5k cohort manifest hash mismatch")
    metadata = _read_jsonl(metadata_path)
    cohort = _read_jsonl(cohort_path)
    if len(metadata) != EXPECTED_ROWS or len(cohort) != EXPECTED_ROWS:
        raise ValueError("5k cache metadata coverage changed")
    expected_execution = np.arange(EXPECTED_ROWS)
    if not np.array_equal(metadata.execution_index.to_numpy(), expected_execution):
        raise ValueError("5k sample metadata execution order changed")
    if not np.array_equal(cohort.execution_index.to_numpy(), expected_execution):
        raise ValueError("5k cohort execution order changed")
    if metadata.sample_id.tolist() != cohort.sample_id.tolist():
        raise ValueError("5k metadata/cohort sample identity changed")
    if metadata.language.tolist() != cohort.language.tolist():
        raise ValueError("5k metadata/cohort language identity changed")
    if metadata.sample_id.duplicated().any():
        raise ValueError("5k cache contains duplicate sample IDs")
    language_counts = metadata.groupby("language").size().to_dict()
    if language_counts != {language: 1000 for language in LANGUAGES}:
        raise ValueError(f"5k cache language coverage changed: {language_counts}")

    shards = manifest.get("shards") or []
    if len(shards) != 20:
        raise ValueError("5k cache shard count changed")
    buffers = {name: [] for name in POOLING_VARIANTS}
    expected_start = 0
    total_bytes = 0
    for expected_index, shard in enumerate(shards):
        if int(shard.get("shard_index", -1)) != expected_index:
            raise ValueError("5k cache shard index changed")
        if int(shard.get("start_index", -1)) != expected_start:
            raise ValueError("5k cache shard coverage is not contiguous")
        if int(shard.get("rows", -1)) != 250:
            raise ValueError("5k cache shard row count changed")
        if list(shard.get("shape_per_variant") or []) != [250, 30, 3072]:
            raise ValueError("5k cache shard declared shape changed")
        if shard.get("dtype") != "float32":
            raise ValueError("5k cache shard declared dtype changed")
        path = root / str(shard.get("path"))
        if not path.is_file():
            raise ValueError(f"5k cache shard missing: {path}")
        if path.stat().st_size != int(shard.get("bytes", -1)):
            raise ValueError("5k cache shard byte size changed")
        if sha256_file(path) != shard.get("sha256"):
            raise ValueError("5k cache shard SHA-256 mismatch")
        total_bytes += path.stat().st_size
        with np.load(path, allow_pickle=False) as archive:
            if tuple(sorted(archive.files)) != tuple(sorted(POOLING_VARIANTS)):
                raise ValueError("5k cache shard keys changed")
            for name in POOLING_VARIANTS:
                values = np.asarray(archive[name], dtype=np.float32)
                if values.shape != (250, 30, 3072):
                    raise ValueError(f"5k cache shard hidden dimensions changed: {name} {values.shape}")
                if not np.isfinite(values).all():
                    raise ValueError("5k cache shard contains nonfinite values")
                buffers[name].append(values)
        expected_start = int(shard.get("stop_index_exclusive", -1))
    if expected_start != EXPECTED_ROWS:
        raise ValueError("5k cache shard coverage incomplete")
    if total_bytes != 4512897483:
        raise ValueError(f"5k cache total shard bytes changed: {total_bytes}")

    arrays = {name: np.concatenate(buffers[name], axis=0) for name in POOLING_VARIANTS}
    for name, values in arrays.items():
        if values.shape != (EXPECTED_ROWS, 30, 3072) or values.dtype != np.float32:
            raise ValueError(f"5k sealed cache array changed: {name} {values.shape} {values.dtype}")
        if not np.isfinite(values).all():
            raise ValueError(f"5k sealed cache array nonfinite: {name}")
    return arrays, metadata, manifest


def load_public_train_labels() -> pd.DataFrame:
    parts = []
    expected_rows = {"Go": 10000, "Java": 10000, "Python": 10000, "Ruby": 10000, "Rust": 9040}
    for language in LANGUAGES:
        path = ROOT / f"data/public/{language}/train-00000-of-00001.parquet"
        frame = pd.read_parquet(path, columns=["sample_id", "membership"])
        if len(frame) != expected_rows[language]:
            raise RuntimeError(f"public train label row count changed for {language}: {len(frame)}")
        frame["language"] = language
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    if len(result) != 49040 or result.sample_id.duplicated().any():
        raise RuntimeError("public train label table identity changed")
    return result


def join_public_labels_5000(metadata: pd.DataFrame, public_train: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "language", "membership"}
    if not required.issubset(public_train.columns):
        raise ValueError("public train label table columns changed")
    labels = public_train[["sample_id", "language", "membership"]].copy()
    normalized = labels.membership.astype("string").str.strip().str.lower().str.replace("_", "-", regex=False)
    labels["label"] = normalized.map({"member": 1, "non-member": 0, "nonmember": 0})
    if labels.label.isna().any() or labels.sample_id.duplicated().any():
        raise ValueError("public train label identity invalid")
    merged = metadata.merge(
        labels[["sample_id", "language", "label"]],
        on="sample_id",
        how="left",
        suffixes=("", "_public"),
        validate="one_to_one",
        sort=False,
    )
    if merged.label.isna().any():
        raise ValueError("5k sealed cache sample missing public label")
    if not (merged.language == merged.language_public).all():
        raise ValueError("5k sealed cache/public language mismatch")
    merged = merged.drop(columns=["language_public"]).reset_index(drop=True)
    merged["label"] = merged.label.astype(int)
    if not np.array_equal(merged.execution_index.to_numpy(), np.arange(EXPECTED_ROWS)):
        raise ValueError("5k public-label join changed execution order")
    counts = merged.groupby(["language", "label"]).size().to_dict()
    expected = {(language, label): EXPECTED_PER_LANGUAGE_LABEL for language in LANGUAGES for label in (0, 1)}
    if counts != expected:
        raise ValueError(f"unexpected frozen 5k language-label balance: {counts}")
    return merged


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


def evaluate_predeclared_decision(
    summary: dict,
    increment_bootstrap: dict,
) -> dict:
    metrics = summary["metrics"]
    bootstrap = summary["paired_bootstrap_vs_zsigmoid"]
    caller_name = "lumia_caller_literal_top5_ensemble"
    mean_name = "mean_only_top5_ensemble"
    helper_name = "lumia_helper_language_aware_top5_ensemble"
    reference_name = "zsigmoid_single_sequence"
    caller = metrics[caller_name]
    reference = metrics[reference_name]

    signal_conditions = {
        "caller_top5_auc_at_least_0_62": bool(caller["auc"] >= 0.62),
        "caller_vs_zsigmoid_auc_bootstrap_lower_95_positive": bool(
            bootstrap[caller_name]["auc_delta"]["lower_95"] > 0.0
        ),
        "caller_top5_all_language_auc_above_0_55": bool(
            all(value > 0.55 for value in caller["per_language_auc"].values())
        ),
    }
    weighting_conditions = {
        "caller_top5_vs_mean_top5_auc_bootstrap_lower_95_positive": bool(
            increment_bootstrap["caller_top5_minus_mean_top5"]["auc_delta"]["lower_95"] > 0.0
        ),
        "helper_top5_vs_mean_top5_auc_bootstrap_lower_95_positive": bool(
            increment_bootstrap["helper_top5_minus_mean_top5"]["auc_delta"]["lower_95"] > 0.0
        ),
    }
    low_fpr_by_anchor = {}
    for name in HIDDEN_SCORE_NAMES:
        point_delta = float(metrics[name]["tpr_at_0.01_fpr"] - reference["tpr_at_0.01_fpr"])
        lower = float(bootstrap[name]["tpr_at_0.01_fpr_delta"]["lower_95"])
        low_fpr_by_anchor[name] = {
            "point_tpr_delta_vs_zsigmoid": point_delta,
            "bootstrap_lower_95": lower,
            "passes": bool(point_delta >= 0.01 and lower >= 0.0),
        }
    return {
        "same_model_signal_confirmed": bool(all(signal_conditions.values())),
        "same_model_signal_conditions": signal_conditions,
        "weighting_increment_supported": bool(any(weighting_conditions.values())),
        "weighting_increment_conditions": weighting_conditions,
        "low_fpr_confirmed": bool(any(item["passes"] for item in low_fpr_by_anchor.values())),
        "low_fpr_by_frozen_anchor": low_fpr_by_anchor,
        "tail_failure_does_not_negate_auc_signal": True,
        "no_automatic_cross_model_promotion": True,
        "caller_auc_delta_vs_zsigmoid": float(caller["auc"] - reference["auc"]),
        "caller_top5_auc": float(caller["auc"]),
        "mean_top5_auc": float(metrics[mean_name]["auc"]),
        "helper_top5_auc": float(metrics[helper_name]["auc"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Validate the entire label-free artifact before public membership labels are loaded.
    cache_validation_started = time.perf_counter()
    arrays, metadata, cache_manifest = load_sealed_cache_5000(args.cache_dir)
    cache_validation_seconds = time.perf_counter() - cache_validation_started
    public_labels = load_public_train_labels()
    frame = join_public_labels_5000(metadata, public_labels)
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
    hidden_scores, fold_records = run_outer_oof_5000(arrays, frame, config, device=device)
    evaluation = summarize_evaluation(frame, hidden_scores, config)
    helper_language = helper_language_bootstrap(frame, hidden_scores, config)
    increment_bootstrap = paired_increment_bootstraps(frame, hidden_scores, config)
    decision_payload = evaluate_predeclared_decision(evaluation, increment_bootstrap)
    evaluation_wall = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics_payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "probe_config": asdict(config),
        "device": device,
        "evaluation_wall_seconds": evaluation_wall,
        "cache_validation_seconds": cache_validation_seconds,
        **evaluation,
        "helper_language_paired_bootstrap_vs_caller": helper_language,
        "paired_increment_bootstrap": increment_bootstrap,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "fold_selection.json").write_text(
        json.dumps(fold_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    decision = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "predeclared_decision": decision_payload,
        "anti_posthoc": {
            "outer_holdout_used_for_layer_selection": False,
            "hyperparameters_changed_after_labels_joined": False,
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
        "task_id": TASK_ID,
        "cache_manifest_sha256": sha256_file(args.cache_dir / "cache_manifest.json"),
        "expected_cache_manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
        "scale_config_sha256": sha256_file(SCALE_CONFIG_PATH),
        "rows": len(frame),
        "outer_oof_coverage": bool(all(np.isfinite(hidden_scores[name]).all() for name in HIDDEN_SCORE_NAMES)),
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "known_cache_manifest_task_id_bug_observed": cache_manifest.get("task_id") == KNOWN_LEGACY_CACHE_TASK_ID,
        "persistent_outputs": [
            "metrics.json", "fold_selection.json", "decision.json", "oof_predictions.csv", "run_manifest.json"
        ],
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "decision": decision_payload,
        "cache_validation_seconds": cache_validation_seconds,
        "evaluation_wall_seconds": evaluation_wall,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
