#!/usr/bin/env python3
"""Credential-free full regression for CMI-Flu E12b bridge/runtime/sanitizer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

REQUEST_ID = "20260911-cmi-flu-strategy-e12b-challenge-freeze-001"
TARGET = "renta0426/cmi-flu-e12b-challenge-freeze-20260911-001"
SCIENCE_COMMIT = "a411bf85a4a79fec2e9a2b9c2bc8cbe186adee5f"
E12B_BLOB = "af5df92ca81abc34a7f7046dba5bc284c98302d4"
E12B_SYNTH_BLOB = "6b36d94299ee3656f9faed623a2ac030a97a714d"
E12B_CONTRACT_BLOB = "7e844b2d56bd6739793888bf6306946a0028d966"
E12A_V2_BLOB = "0c4c970c8bacfed61bfbb9587e0a0bfdec7903d9"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
EXPECTED_PORTFOLIO = {
    "Task1.1": "b21_pls_2", "Task1.2": "task12_anchor_residual_et_d5_l5_sqrt_lambda0.5",
    "Task1.3": "strict_asc_anchor", "Task1.4": "raw_pre_vacc_conserved_anchor",
    "Task2.1": "b21_et_subtype_d3_l5", "Task2.2": "b21_et_subtype_d5_l10", "Task2.3": "b21_ridge_exact_a100",
}
TASKS = tuple(EXPECTED_PORTFOLIO)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture() -> dict:
    h = "0" * 64
    summaries = {}
    for task in TASKS:
        unique = 36 if task == "Task1.3" else 40
        summaries[task] = {
            "task": task, "rows": 40, "prediction_min": 0.1, "prediction_max": 1.0,
            "prediction_unique": unique, "tie_fraction": 1.0 - unique / 40.0,
            "prediction_sha256": h,
        }
    comp12 = {"task": "Task1.2", "rank_spearman": 0.9, "rank_spearman_status": "ok", "changed_rank_count": 20, "mean_absolute_percentile_shift": 0.05, "maximum_absolute_percentile_shift": 0.2}
    comp13 = {"task": "Task1.3", "rank_spearman": 0.8, "rank_spearman_status": "ok", "changed_rank_count": 30, "mean_absolute_percentile_shift": 0.1, "maximum_absolute_percentile_shift": 0.4}
    return {
        "schema_version": 1,
        "experiment": "strategy_v2_e12b_challenge_prediction_freeze",
        "final_portfolio": dict(EXPECTED_PORTFOLIO),
        "portfolio_task_count": 7,
        "challenge_rows": 40,
        "submission_validation": {
            "rows": 40,
            "columns": ["participant_id", *TASKS],
            "task_unique_counts": {task: summaries[task]["prediction_unique"] for task in TASKS},
            "minus99_tasks": [],
        },
        "fingerprint_contract": {
            "semantic_hash_version": "cmi-flu-e12b-semantic-v1",
            "semantic_submission_sha256": h,
            "canonical_csv_float_format": "%.17g",
            "canonical_csv_lineterminator": "LF",
            "canonical_csv_sha256": h,
            "canonical_csv_bytes": 4096,
        },
        "task_prediction_summaries": summaries,
        "structural_controls": {
            "b21_portfolio": {}, "task12_only_portfolio": {},
            "final_changed_tasks_vs_regenerated_b21": ["Task1.2", "Task1.3"],
            "final_changed_tasks_vs_regenerated_task12_only": ["Task1.3"],
            "regenerated_b21_semantic_sha256": h,
            "regenerated_task12_only_semantic_sha256": h,
            "historical_public_0_218_is_not_assumed_byte_identical": True,
            "task12_final_vs_b21": comp12,
            "task13_final_vs_b21": comp13,
            "task13_final_vs_task12_only": comp13,
        },
        "task13_reconciliation": {
            "final_incumbent": "strict_asc_anchor",
            "predictor_feature": "flow_rank__Antibody-secreting_cells_(ASC)",
            "challenge_unique_values": 36,
            "e12a_v2_reproduced": True,
        },
        "new_model_selection_performed": False,
        "leaderboard_used_for_selection": False,
        "public_probe_performed": False,
        "competition_submission_attempted": False,
        "competition_submission_authorized": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
        "row_level_candidate_persisted": False,
        "next_step": "ready_for_separate_submission_authorization",
    }


def load_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("cmi_flu_e12b_runtime_ci", path)
    if spec is None or spec.loader is None:
        raise SystemExit("E12b CI could not load generated runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    work = args.workdir.expanduser().resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    runtime = work / "runtime.py"
    subprocess.run(
        [sys.executable, str(root / "scripts/cmi_flu_strategy_e12b_prepare_v2.py"), "--repository-root", str(root), "--output", str(runtime)],
        check=True,
    )
    text = runtime.read_text(encoding="utf-8")
    for token in (
        f'REQUEST_ID = "{REQUEST_ID}"', f'TARGET_KERNEL = "{TARGET}"',
        f'SCIENCE_COMMIT = "{SCIENCE_COMMIT}"', f'E12B_BLOB = "{E12B_BLOB}"',
        '"Task1.3": "strict_asc_anchor"', '"row_level_candidate_persisted": False',
        "CMI_FLU_E12B_COMPLETE",
    ):
        if token not in text:
            raise SystemExit("E12b CI generated runtime identity mismatch")
    lowered = text.casefold()
    if "kaggle competitions submit" in lowered or "competition_submit(" in lowered:
        raise SystemExit("E12b CI found submission path")
    module = load_runtime(runtime)
    metrics = fixture()
    module.validate_result(metrics)
    summary = module.render_summary(metrics)
    if "participant_id" not in metrics["submission_validation"]["columns"]:
        raise SystemExit("E12b CI schema column regression fixture invalid")

    valid = work / "valid"
    valid.mkdir()
    metrics_path = valid / "metrics.json"
    summary_path = valid / "summary.md"
    bridge_path = valid / "bridge-result.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")
    bridge = {
        "schema_version": 1, "request_id": REQUEST_ID,
        "competition": "cmi-flu-first-prediction-challenge", "target_kernel": TARGET,
        "science_commit": SCIENCE_COMMIT, "strategy_e12b_blob_sha": E12B_BLOB,
        "strategy_e12b_contract_blob_sha": E12B_CONTRACT_BLOB,
        "strategy_e12a_v2_blob_sha": E12A_V2_BLOB, "config_blob_sha": CONFIG_BLOB,
        "package_sha256": "1" * 64, "python_version": "3.12.0", "md5_verified_count": 28,
        "metrics_sha256": sha256(metrics_path), "summary_sha256": sha256(summary_path),
        "frozen_portfolio": dict(EXPECTED_PORTFOLIO),
        "semantic_submission_sha256": metrics["fingerprint_contract"]["semantic_submission_sha256"],
        "canonical_csv_sha256": metrics["fingerprint_contract"]["canonical_csv_sha256"],
        "next_step": "ready_for_separate_submission_authorization",
        "row_level_candidate_persisted": False, "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False, "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
    bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(root / "scripts/cmi_flu_strategy_e12b_sanitize.py"), "--input-dir", str(valid)], check=True)

    negative = work / "negative-row-key"
    shutil.copytree(valid, negative)
    bad = json.loads((negative / "metrics.json").read_text(encoding="utf-8"))
    bad["leak_test"] = {"participant_id": "synthetic-only"}
    (negative / "metrics.json").write_text(json.dumps(bad, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    attempt = subprocess.run(
        [sys.executable, str(root / "scripts/cmi_flu_strategy_e12b_sanitize.py"), "--input-dir", str(negative)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if attempt.returncode == 0:
        raise SystemExit("E12b sanitizer failed to reject row-level identifier key")

    print(
        "CMI_FLU_E12B_CI PASS exact_runtime=true schema_column_allowed=true row_key_rejected=true "
        "row_persisted=false auth=false write=false compute=false submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
