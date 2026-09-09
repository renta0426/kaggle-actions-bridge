#!/usr/bin/env python3
"""Validate aggregate-only E06c D28 Task2.1/Task2.2 calibration outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260909-cmi-flu-strategy-e06c-d28-calibration-001"
SCIENCE_COMMIT = "044d28e3c7e9ab23fa531f57479b03e2437bb1cb"
E06_BLOB = "dd5b5ed12e7667e527c97a99e86b8821e68e6442"
E06_V2_BLOB = "270a9616c4ab4726cac72e220f86c071161765dc"
E06C_BLOB = "90c6e3e2791cc2089d8666f4c8672a47953f5e12"
BANNED = (
    '"participant_id"',
    '"subject_group"',
    '"row_index"',
    '"oof_predictions"',
    '"challenge_predictions"',
    '"predictions"',
)
PROMOTION_CHECKS = {
    "strictly_positive_outer_slopes",
    "held_study_rank_preserved",
    "challenge_rank_preserved",
    "equal_study_mean_rmse_reduction_at_least_2pct",
    "median_study_rmse_improves",
    "no_study_rmse_worse_than_5pct",
}
ALLOWED_SELECTED = {
    None,
    "b21_reference__positive_affine",
    "b21_reference__log2_affine",
    "phase_a_target_domain__positive_affine",
    "phase_a_target_domain__log2_affine",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    root = p.parse_args().input_dir.resolve()
    files = sorted(path.name for path in root.iterdir() if path.is_file())
    if files != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit(f"E06c safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E06c aggregate privacy contract failed:{name}")

    bridge = json.loads(texts["bridge-result.json"])
    metrics = json.loads(texts["metrics.json"])
    expected_bridge = {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e06_blob_sha": E06_BLOB,
        "strategy_e06_v2_blob_sha": E06_V2_BLOB,
        "strategy_e06c_blob_sha": E06C_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
    for key, expected in expected_bridge.items():
        if bridge.get(key) != expected:
            raise SystemExit(f"E06c bridge contract mismatch:{key}:{bridge.get(key)!r}")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json"):
        raise SystemExit("E06c metrics hash mismatch")
    if bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E06c summary hash mismatch")

    if metrics.get("experiment") != "strategy_v2_e06c_d28_rank_preserving_calibration" or int(metrics.get("target_day", -1)) != 28:
        raise SystemExit("E06c experiment/day identity mismatch")
    if metrics.get("raw_conditions") != ["b21_reference", "phase_a_target_domain"] or metrics.get("model_name") != "ridge_exact_a100":
        raise SystemExit("E06c raw condition/model identity mismatch")
    if metrics.get("competition_submission_attempted") is not False or metrics.get("leaderboard_used_for_selection") is not False:
        raise SystemExit("E06c selection/submission contract mismatch")
    if metrics.get("incumbent_changed") is not False or metrics.get("public_probe_authorized") is not False or int(metrics.get("automatic_compute_retries", -1)) != 0:
        raise SystemExit("E06c incumbent/probe/retry contract mismatch")

    contract = metrics.get("calibration_contract") or {}
    expected_contract = {
        "calibration_kinds": ["positive_affine", "log2_affine"],
        "fit_unit": "task_panel_geometric_mean_donor",
        "calibration_weighting": "equal_study_total_weight",
        "outer_evaluation": "subject_purged_leave_one_study_out",
        "calibrator_training": "inner_subject_purged_study_out_oof_from_outer_training_only",
        "outer_fold_panel_overlap_filter": "fixed_panel_strain_availability_only_before_outcome_access",
        "minimum_calibrator_panel_overlap_studies": 2,
        "strict_monotonicity_required": True,
        "held_study_outcomes_used_for_calibrator": False,
        "challenge_outcomes_available": False,
        "public_leaderboard_used_for_selection": False,
        "e05_rank_changing_models_included": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise SystemExit(f"E06c calibration contract mismatch:{key}")

    thresholds = metrics.get("promotion_thresholds") or {}
    if float(thresholds.get("minimum_equal_study_mean_rmse_reduction", -1)) != 0.02:
        raise SystemExit("E06c mean RMSE threshold mismatch")
    if float(thresholds.get("minimum_median_study_rmse_reduction_strictly_greater_than", -1)) != 0.0:
        raise SystemExit("E06c median RMSE threshold mismatch")
    if float(thresholds.get("maximum_worst_study_rmse_ratio", -1)) != 1.05:
        raise SystemExit("E06c worst-study threshold mismatch")
    if not finite(thresholds.get("rank_tolerance")) or float(thresholds["rank_tolerance"]) > 1e-12:
        raise SystemExit("E06c rank tolerance mismatch")

    tasks = metrics.get("tasks") or {}
    if set(tasks) != {"Task2.1", "Task2.2"}:
        raise SystemExit("E06c task set mismatch")
    expected_panels = {"Task2.1": 3, "Task2.2": 12}
    aggregate_summary = {}
    allowed_skip_reasons = {"empty_outer_validation", "zero_fixed_panel_overlap", "fewer_than_two_calibration_panel_studies", "fewer_than_two_inner_study_folds"}
    for task, task_payload in tasks.items():
        if int(task_payload.get("panel_size", -1)) != expected_panels[task]:
            raise SystemExit(f"E06c panel size mismatch:{task}")
        conditions = task_payload.get("conditions") or {}
        if set(conditions) != {"b21_reference", "phase_a_target_domain"}:
            raise SystemExit(f"E06c condition set mismatch:{task}")
        task_summary = {"selected": task_payload.get("selected_promoted_condition"), "conditions": {}}
        for condition, payload in conditions.items():
            model = payload.get("model") or {}
            if model.get("name") != "ridge_exact_a100" or model.get("family") != "ridge" or float((model.get("params") or {}).get("alpha", -1)) != 100.0:
                raise SystemExit(f"E06c frozen model mismatch:{task}:{condition}")
            nested = payload.get("historical_nested_study_out") or {}
            candidate_count = int(nested.get("outer_candidate_split_count", -1))
            scored_count = int(nested.get("outer_split_count", -1))
            skipped_count = int(nested.get("skipped_outer_split_count", -1))
            skipped = nested.get("skipped_outer_folds") or []
            if scored_count < 3 or candidate_count != scored_count + skipped_count or skipped_count != len(skipped):
                raise SystemExit(f"E06c outer split accounting mismatch:{task}:{condition}")
            if any((item or {}).get("reason") not in allowed_skip_reasons for item in skipped):
                raise SystemExit(f"E06c skip reason mismatch:{task}:{condition}")
            raw = (nested.get("raw_metrics") or {}).get("equal_study_summary") or {}
            raw_rmse = raw.get("rmse_mean")
            if not finite(raw_rmse):
                raise SystemExit(f"E06c raw RMSE missing:{task}:{condition}")
            calibrated = nested.get("calibrated") or {}
            promotion = payload.get("promotion") or {}
            challenge = payload.get("challenge_rank_preservation") or {}
            if set(calibrated) != {"positive_affine", "log2_affine"} or set(promotion) != {"positive_affine", "log2_affine"}:
                raise SystemExit(f"E06c candidate set mismatch:{task}:{condition}")
            condition_summary = {"raw_rmse_mean": float(raw_rmse), "candidates": {}}
            for kind in ("positive_affine", "log2_affine"):
                item = calibrated.get(kind) or {}
                diag = item.get("rmse_diagnostics") or {}
                mean_reduction = diag.get("equal_study_mean_relative_reduction")
                median_reduction = diag.get("median_study_relative_reduction")
                worst_ratio = diag.get("worst_study_rmse_ratio")
                if not all(finite(value) for value in (mean_reduction, median_reduction, worst_ratio)):
                    raise SystemExit(f"E06c non-finite diagnostic:{task}:{condition}:{kind}")
                held_rank = (item.get("rank_preservation") or {}).get("preserved_within_tolerance")
                challenge_rank = (challenge.get(kind) or {}).get("rank_preserved_within_tolerance")
                if not isinstance(held_rank, bool) or not isinstance(challenge_rank, bool):
                    raise SystemExit(f"E06c rank boolean missing:{task}:{condition}:{kind}")
                calibrators = item.get("outer_fold_calibrators") or []
                if len(calibrators) != scored_count or any(int(entry.get("training_studies", 0)) < 2 for entry in calibrators):
                    raise SystemExit(f"E06c calibrator study contract:{task}:{condition}:{kind}")
                decision = promotion.get(kind) or {}
                passed = decision.get("passed")
                checks = decision.get("checks") or {}
                if not isinstance(passed, bool) or set(checks) != PROMOTION_CHECKS or any(not isinstance(value, bool) for value in checks.values()):
                    raise SystemExit(f"E06c promotion contract:{task}:{condition}:{kind}")
                if checks.get("held_study_rank_preserved") is not held_rank or checks.get("challenge_rank_preserved") is not challenge_rank:
                    raise SystemExit(f"E06c rank/promotion mismatch:{task}:{condition}:{kind}")
                if passed is not bool(all(checks.values())) or ((not held_rank or not challenge_rank) and passed):
                    raise SystemExit(f"E06c promotion boolean mismatch:{task}:{condition}:{kind}")
                condition_summary["candidates"][kind] = {
                    "mean_rmse_reduction": float(mean_reduction),
                    "median_study_reduction": float(median_reduction),
                    "worst_rmse_ratio": float(worst_ratio),
                    "held_rank_preserved": held_rank,
                    "challenge_rank_preserved": challenge_rank,
                    "promotion": passed,
                }
            task_summary["conditions"][condition] = condition_summary
        selected = task_payload.get("selected_promoted_condition")
        if selected not in ALLOWED_SELECTED:
            raise SystemExit(f"E06c selected condition invalid:{task}")
        if selected is not None:
            condition, kind = selected.split("__", 1)
            if ((conditions.get(condition) or {}).get("promotion") or {}).get(kind, {}).get("passed") is not True:
                raise SystemExit(f"E06c selected candidate not promoted:{task}")
        aggregate_summary[task] = task_summary

    print("CMI_FLU_E06C_AGGREGATE_SUMMARY " + json.dumps(aggregate_summary, sort_keys=True, separators=(",", ":")))
    print("CMI_FLU_E06C_RESULT PASS aggregate_only=true submission=false science_commit=" + SCIENCE_COMMIT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
