#!/usr/bin/env python3
"""Validate E12a-v2 aggregate outputs and corrected final-portfolio presentation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260911-cmi-flu-strategy-e12a-v2-portfolio-reconciliation-001"
TARGET = "renta0426/cmi-flu-e12a-v2-portfolio-reconcile-20260911-001"
SCIENCE_COMMIT = "e6e05e578f172ce710bc0f1da55acdf290e83dd9"
E12A_V2_BLOB = "0c4c970c8bacfed61bfbb9587e0a0bfdec7903d9"
E12A_V2_CONTRACT_BLOB = "6b40ac35a3b43c3bafcdd08fcc49fc9fe22960c1"
HEADING = "# CMI-Flu Strategy-v2 E12a-v2 final portfolio reconciliation"
EXPECTED_PORTFOLIO = {
    "Task1.1": "b21_pls_2",
    "Task1.2": "task12_anchor_residual_et_d5_l5_sqrt_lambda0.5",
    "Task1.3": "strict_asc_anchor",
    "Task1.4": "raw_pre_vacc_conserved_anchor",
    "Task2.1": "b21_et_subtype_d3_l5",
    "Task2.2": "b21_et_subtype_d5_l10",
    "Task2.3": "b21_ridge_exact_a100",
}
EXPECTED_SCORES = {
    "Task1.1": 0.09970718035376519,
    "Task1.2": 0.5271033295423541,
    "Task1.3": 0.38158872734833993,
    "Task2.1": 0.623449034547404,
    "Task2.2": 0.576505972340535,
    "Task2.3": 0.6957305642219944,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.expanduser().resolve()
    expected_files = {"bridge-result.json", "metrics.json", "summary.md"}
    files = {path.name for path in root.iterdir() if path.is_file()}
    if files != expected_files:
        raise SystemExit(f"E12a-v2 output file set mismatch:{sorted(files)}")
    if any(path.is_symlink() for path in root.iterdir()):
        raise SystemExit("E12a-v2 symlink output forbidden")

    bridge = json.loads((root / "bridge-result.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    summary = (root / "summary.md").read_text(encoding="utf-8")

    if bridge.get("request_id") != REQUEST_ID or bridge.get("target_kernel") != TARGET:
        raise SystemExit("E12a-v2 bridge identity mismatch")
    if bridge.get("science_commit") != SCIENCE_COMMIT:
        raise SystemExit("E12a-v2 bridge science commit mismatch")
    if bridge.get("strategy_e12a_v2_blob_sha") != E12A_V2_BLOB:
        raise SystemExit("E12a-v2 bridge science blob mismatch")
    if bridge.get("strategy_e12a_v2_contract_blob_sha") != E12A_V2_CONTRACT_BLOB:
        raise SystemExit("E12a-v2 bridge contract blob mismatch")
    if bridge.get("metrics_sha256") != sha256(root / "metrics.json"):
        raise SystemExit("E12a-v2 metrics hash link mismatch")
    if bridge.get("summary_sha256") != sha256(root / "summary.md"):
        raise SystemExit("E12a-v2 summary hash link mismatch")
    if bridge.get("frozen_portfolio") != EXPECTED_PORTFOLIO:
        raise SystemExit("E12a-v2 bridge final portfolio mismatch")
    if bridge.get("portfolio_manifest_corrected") is not True or bridge.get("corrected_task") != "Task1.3":
        raise SystemExit("E12a-v2 bridge reconciliation marker mismatch")
    for key in (
        "competition_submission_attempted",
        "leaderboard_used_for_selection",
        "contains_participant_identifiers",
        "contains_row_level_predictions",
    ):
        if bridge.get(key) is not False:
            raise SystemExit(f"E12a-v2 bridge boundary mismatch:{key}")

    if metrics.get("experiment") != "strategy_v2_e12a_v2_portfolio_reconciliation":
        raise SystemExit("E12a-v2 metrics identity mismatch")
    if metrics.get("comparison_contract") != "paired_subject_purged_v2":
        raise SystemExit("E12a-v2 comparison contract mismatch")
    if metrics.get("source_e12a_v1_disposition") != "reproduction_harness_passed_final_portfolio_invalid_stale_task13_identity":
        raise SystemExit("E12a-v2 parent disposition mismatch")
    if metrics.get("portfolio_manifest_corrected") is not True or metrics.get("corrected_task") != "Task1.3":
        raise SystemExit("E12a-v2 metrics reconciliation marker mismatch")
    if metrics.get("frozen_portfolio") != EXPECTED_PORTFOLIO or metrics.get("portfolio_task_count") != 7:
        raise SystemExit("E12a-v2 metrics final portfolio mismatch")
    if metrics.get("e11_task11_candidates_carried") != []:
        raise SystemExit("E12a-v2 E11 carry mismatch")
    for key in (
        "new_model_selection_performed",
        "competition_incumbent_changed",
        "public_probe_authorized",
        "competition_submission_authorized",
        "leaderboard_used_for_selection",
        "competition_submission_attempted",
        "contains_participant_identifiers",
        "contains_row_level_predictions",
    ):
        if metrics.get(key) is not False:
            raise SystemExit(f"E12a-v2 metrics boundary mismatch:{key}")

    stale = metrics.get("stale_task13_e01_control") or {}
    if stale.get("incumbent") != "b21_pls_1" or stale.get("final_portfolio_member") is not False:
        raise SystemExit("E12a-v2 stale Task1.3 control boundary mismatch")
    stale_score = float(stale.get("study_equal_spearman"))
    if not math.isfinite(stale_score) or abs(stale_score - 0.10630410834652516) > 2e-6:
        raise SystemExit("E12a-v2 stale Task1.3 control reproduction mismatch")
    gain = float(metrics.get("task13_strict_anchor_gain_vs_e01_control"))
    if not math.isfinite(gain) or abs(gain - (0.38158872734833993 - stale_score)) > 1e-12 or gain <= 0.27:
        raise SystemExit("E12a-v2 Task1.3 reconciliation gain mismatch")

    rows = metrics.get("supervised_reproduction") or []
    if len(rows) != 6:
        raise SystemExit("E12a-v2 supervised reproduction count mismatch")
    by_task = {row.get("task"): row for row in rows}
    if set(by_task) != set(EXPECTED_SCORES):
        raise SystemExit("E12a-v2 supervised task set mismatch")
    for task, expected_score in EXPECTED_SCORES.items():
        row = by_task[task]
        if row.get("incumbent") != EXPECTED_PORTFOLIO[task]:
            raise SystemExit(f"E12a-v2 incumbent identity mismatch:{task}")
        exp = float(row.get("expected_study_equal_spearman"))
        obs = float(row.get("observed_study_equal_spearman"))
        dev = float(row.get("deviation"))
        absolute = float(row.get("absolute_deviation"))
        if not all(math.isfinite(value) for value in (exp, obs, dev, absolute)):
            raise SystemExit("E12a-v2 nonfinite reproduction value")
        if abs(exp - expected_score) > 1e-12:
            raise SystemExit(f"E12a-v2 expected reference drift:{task}")
        if abs((obs - exp) - dev) > 1e-12 or abs(abs(dev) - absolute) > 1e-12:
            raise SystemExit(f"E12a-v2 reproduction arithmetic mismatch:{task}")
        if bool(row.get("reproduced")) is not (absolute <= 2e-6):
            raise SystemExit(f"E12a-v2 tolerance mismatch:{task}")

    task13 = by_task["Task1.3"]
    if task13.get("predictor_feature") != "flow_rank__Antibody-secreting_cells_(ASC)":
        raise SystemExit("E12a-v2 Task1.3 anchor feature mismatch")
    if int(task13.get("challenge_rows", -1)) != 40 or task13.get("challenge_anchor_complete") is not True:
        raise SystemExit("E12a-v2 Task1.3 Challenge contract mismatch")
    if int(task13.get("challenge_anchor_unique_values", 0)) < 2:
        raise SystemExit("E12a-v2 Task1.3 Challenge anchor constant")

    all_pass = all(bool(by_task[task].get("reproduced")) for task in EXPECTED_SCORES)
    if metrics.get("all_supervised_tasks_reproduced") is not all_pass:
        raise SystemExit("E12a-v2 all-pass arithmetic mismatch")
    expected_next = (
        "proceed_to_E12b_challenge_prediction_freeze"
        if all_pass
        else "investigate_reconciliation_contract_no_model_selection"
    )
    if metrics.get("next_step") != expected_next or bridge.get("next_step") != expected_next:
        raise SystemExit("E12a-v2 next-step mismatch")
    if bridge.get("all_supervised_tasks_reproduced") is not all_pass:
        raise SystemExit("E12a-v2 bridge all-pass mismatch")

    task14 = metrics.get("task14_contract") or {}
    if (
        task14.get("incumbent") != "raw_pre_vacc_conserved_anchor"
        or task14.get("supervised_cv_available") is not False
        or task14.get("outcomes_accessed") is not False
        or int(task14.get("challenge_subjects_expected", -1)) != 40
    ):
        raise SystemExit("E12a-v2 Task1.4 contract mismatch")

    if not summary.startswith(HEADING + "\n"):
        raise SystemExit("E12a-v2 summary heading mismatch")
    if "None" in summary or "# CMI-Flu strategy E01" in summary or "# CMI-Flu Strategy-v2 E12a final-system reproduction audit" in summary:
        raise SystemExit("E12a-v2 stale summary presentation detected")
    required_summary = (
        f"- corrected task: `Task1.3`",
        f"- all supervised final components reproduced: `{str(all_pass).lower()}`",
        f"- next step: `{expected_next}`",
        "- final incumbent: `strict_asc_anchor`",
        "- final predictor feature: `flow_rank__Antibody-secreting_cells_(ASC)`",
        "- stale E01 control: `b21_pls_1`; final portfolio member=`false`",
        "- Task1.3: `strict_asc_anchor`",
        "- incumbent: `raw_pre_vacc_conserved_anchor`",
    )
    if any(token not in summary for token in required_summary):
        raise SystemExit("E12a-v2 summary/metrics mismatch")
    if "- Task1.3: `b21_pls_1`" in summary:
        raise SystemExit("E12a-v2 stale Task1.3 shown as final portfolio member")
    for task, row in by_task.items():
        token = (
            f"- {task}: incumbent `{row['incumbent']}`, "
            f"expected={float(row['expected_study_equal_spearman']):.12f}, "
            f"observed={float(row['observed_study_equal_spearman']):.12f}, "
            f"abs_delta={float(row['absolute_deviation']):.9g}, "
            f"reproduced={str(bool(row['reproduced'])).lower()}"
        )
        if token not in summary:
            raise SystemExit(f"E12a-v2 summary task mismatch:{task}")

    raw = "\n".join((root / name).read_text(errors="replace") for name in sorted(expected_files))
    for banned in (
        '"participant_id"',
        '"subject_group"',
        '"row_index"',
        '"oof_predictions"',
        '"challenge_predictions"',
        "KGAT_",
        "KAGGLE_API_TOKEN",
    ):
        if banned in raw:
            raise SystemExit("E12a-v2 sensitive/row-level token in persistent output")

    print(
        "CMI_FLU_E12A_V2_SANITIZE PASS "
        f"aggregate_only=true final_task13=strict_asc_anchor all_reproduced={str(all_pass).lower()} "
        f"next_step={expected_next} submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
