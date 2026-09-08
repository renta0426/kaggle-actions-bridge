#!/usr/bin/env python3
"""Validate aggregate E06a 003 results without requiring every candidate to preserve rank."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260908-cmi-flu-strategy-e06a-task23-calibration-003"
SCIENCE_COMMIT = "c968d00eb209e55e53b88643d99382ad6915c87e"
E06_BLOB = "dd5b5ed12e7667e527c97a99e86b8821e68e6442"
E06_V2_BLOB = "270a9616c4ab4726cac72e220f86c071161765dc"
BANNED = (
    '"participant_id"',
    '"subject_group"',
    '"row_index"',
    '"oof_predictions"',
    '"challenge_predictions"',
)
PROMOTION_CHECKS = {
    "strictly_positive_outer_slopes",
    "held_study_rank_preserved",
    "challenge_rank_preserved",
    "equal_study_mean_rmse_reduction_at_least_2pct",
    "median_study_rmse_improves",
    "no_study_rmse_worse_than_5pct",
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
        raise SystemExit(f"E06a v3 safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E06a v3 aggregate privacy contract failed:{name}")

    bridge = json.loads(texts["bridge-result.json"])
    metrics = json.loads(texts["metrics.json"])
    expected_bridge = {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e06_blob_sha": E06_BLOB,
        "strategy_e06_v2_blob_sha": E06_V2_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
    for key, expected in expected_bridge.items():
        if bridge.get(key) != expected:
            raise SystemExit(f"E06a v3 bridge contract mismatch:{key}:{bridge.get(key)!r}")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json"):
        raise SystemExit("E06a v3 metrics hash mismatch")
    if bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E06a v3 summary hash mismatch")

    if metrics.get("experiment") != "strategy_v2_e06a_task23_rank_preserving_calibration":
        raise SystemExit("E06a v3 experiment identity mismatch")
    if metrics.get("task") != "Task2.3" or int(metrics.get("target_day", -1)) != 365:
        raise SystemExit("E06a v3 task identity mismatch")
    if metrics.get("competition_submission_attempted") is not False or metrics.get("leaderboard_used_for_selection") is not False:
        raise SystemExit("E06a v3 selection/submission contract mismatch")
    contract = metrics.get("calibration_contract") or {}
    if contract.get("outer_fold_panel_overlap_filter") != "fixed_panel_strain_availability_only_before_outcome_access":
        raise SystemExit("E06a v3 panel-overlap filter contract mismatch")
    if contract.get("zero_panel_overlap_studies_scored_as_task23_proxy") is not False:
        raise SystemExit("E06a v3 zero-overlap scoring contract mismatch")
    if int(contract.get("minimum_calibrator_panel_overlap_studies", -1)) != 2:
        raise SystemExit("E06a v3 cross-study calibrator contract mismatch")

    coverage = metrics.get("historical_panel_proxy_coverage") or {}
    if int(coverage.get("requested_strains", -1)) != 12:
        raise SystemExit("E06a v3 requested panel size mismatch")
    if int(coverage.get("complete_requested_panel_donors", -1)) != 0:
        raise SystemExit("E06a v3 historical D365 panel unexpectedly complete")

    conditions = metrics.get("conditions") or {}
    if set(conditions) != {"b21_reference", "phase_a_target_domain"}:
        raise SystemExit("E06a v3 condition set mismatch")
    for condition, payload in conditions.items():
        nested = payload.get("historical_nested_study_out") or {}
        candidate_count = int(nested.get("outer_candidate_split_count", -1))
        scored_count = int(nested.get("outer_split_count", -1))
        skipped_count = int(nested.get("skipped_outer_split_count", -1))
        skipped = nested.get("skipped_outer_folds") or []
        if scored_count < 3 or candidate_count != scored_count + skipped_count or skipped_count != len(skipped):
            raise SystemExit(f"E06a v3 outer split accounting mismatch:{condition}")
        if not any((item or {}).get("reason") == "zero_fixed_panel_overlap" for item in skipped):
            raise SystemExit(f"E06a v3 expected zero-panel-overlap evidence missing:{condition}")

        raw = (nested.get("raw_metrics") or {}).get("equal_study_summary") or {}
        raw_rmse = raw.get("rmse_mean")
        if not finite(raw_rmse):
            raise SystemExit(f"E06a v3 raw RMSE missing:{condition}")
        print(
            f"{condition} outer_candidate={candidate_count} outer_scored={scored_count} "
            f"outer_skipped={skipped_count} raw_equal_study_rmse_mean={raw_rmse} "
            f"raw_equal_study_spearman_mean={raw.get('spearman_mean')}"
        )

        calibrated = nested.get("calibrated") or {}
        promotion = payload.get("promotion") or {}
        challenge = payload.get("challenge_rank_preservation") or {}
        for kind in ("positive_affine", "log2_affine"):
            item = calibrated.get(kind) or {}
            diag = item.get("rmse_diagnostics") or {}
            mean_reduction = diag.get("equal_study_mean_relative_reduction")
            median_reduction = diag.get("median_study_relative_reduction")
            worst_ratio = diag.get("worst_study_rmse_ratio")
            if not all(finite(value) for value in (mean_reduction, median_reduction, worst_ratio)):
                raise SystemExit(f"E06a v3 non-finite calibration diagnostic:{condition}:{kind}")
            held_rank = (item.get("rank_preservation") or {}).get("preserved_within_tolerance")
            challenge_rank = (challenge.get(kind) or {}).get("rank_preserved_within_tolerance")
            if not isinstance(held_rank, bool) or not isinstance(challenge_rank, bool):
                raise SystemExit(f"E06a v3 rank boolean missing:{condition}:{kind}")
            calibrators = item.get("outer_fold_calibrators") or []
            if len(calibrators) != scored_count or any(int(entry.get("training_studies", 0)) < 2 for entry in calibrators):
                raise SystemExit(f"E06a v3 calibrator study contract:{condition}:{kind}")
            decision = promotion.get(kind) or {}
            passed = decision.get("passed")
            checks = decision.get("checks") or {}
            if not isinstance(passed, bool) or set(checks) != PROMOTION_CHECKS or any(not isinstance(value, bool) for value in checks.values()):
                raise SystemExit(f"E06a v3 promotion contract:{condition}:{kind}")
            if checks.get("held_study_rank_preserved") is not held_rank or checks.get("challenge_rank_preserved") is not challenge_rank:
                raise SystemExit(f"E06a v3 rank/promotion mismatch:{condition}:{kind}")
            if passed is not bool(all(checks.values())):
                raise SystemExit(f"E06a v3 promotion boolean mismatch:{condition}:{kind}")
            if (not held_rank or not challenge_rank) and passed:
                raise SystemExit(f"E06a v3 non-preserving candidate promoted:{condition}:{kind}")
            print(
                f"{condition}/{kind} mean_rmse_reduction={mean_reduction} "
                f"median_study_reduction={median_reduction} worst_rmse_ratio={worst_ratio} "
                f"held_rank_preserved={str(held_rank).lower()} challenge_rank_preserved={str(challenge_rank).lower()} "
                f"promotion={passed}"
            )

    selected = metrics.get("selected_promoted_condition")
    allowed = {None, "b21_reference__positive_affine", "b21_reference__log2_affine", "phase_a_target_domain__positive_affine", "phase_a_target_domain__log2_affine"}
    if selected not in allowed:
        raise SystemExit("E06a v3 selected condition invalid")
    if selected is not None:
        condition, kind = selected.split("__", 1)
        if ((conditions.get(condition) or {}).get("promotion") or {}).get(kind, {}).get("passed") is not True:
            raise SystemExit("E06a v3 selected candidate was not promoted")
    print(
        "CMI_FLU_E06A_V3_RESULT PASS aggregate_only=true submission=false "
        f"selected={selected} science_commit={SCIENCE_COMMIT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
