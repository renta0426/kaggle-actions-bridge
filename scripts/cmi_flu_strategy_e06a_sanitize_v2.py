#!/usr/bin/env python3
"""Validate and print only aggregate repaired E06a 002 results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260908-cmi-flu-strategy-e06a-task23-calibration-002"
SCIENCE_COMMIT = "6974d1f3c6f2e9b52bc8f1076e46051c71ea62c3"
E06_BLOB = "dd5b5ed12e7667e527c97a99e86b8821e68e6442"
E06_V2_BLOB = "270a9616c4ab4726cac72e220f86c071161765dc"
BANNED = (
    '"participant_id"',
    '"subject_group"',
    '"row_index"',
    '"oof_predictions"',
    '"challenge_predictions"',
)


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
        raise SystemExit(f"E06a v2 safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E06a v2 aggregate privacy contract failed:{name}")

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
            raise SystemExit(f"E06a v2 bridge contract mismatch:{key}:{bridge.get(key)!r}")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json"):
        raise SystemExit("E06a v2 metrics hash mismatch")
    if bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E06a v2 summary hash mismatch")

    if metrics.get("experiment") != "strategy_v2_e06a_task23_rank_preserving_calibration":
        raise SystemExit("E06a v2 experiment identity mismatch")
    if metrics.get("task") != "Task2.3" or int(metrics.get("target_day", -1)) != 365:
        raise SystemExit("E06a v2 task identity mismatch")
    if metrics.get("competition_submission_attempted") is not False or metrics.get("leaderboard_used_for_selection") is not False:
        raise SystemExit("E06a v2 selection/submission contract mismatch")
    contract = metrics.get("calibration_contract") or {}
    if contract.get("outer_fold_panel_overlap_filter") != "fixed_panel_strain_availability_only_before_outcome_access":
        raise SystemExit("E06a v2 panel-overlap filter contract mismatch")
    if contract.get("zero_panel_overlap_studies_scored_as_task23_proxy") is not False:
        raise SystemExit("E06a v2 zero-overlap scoring contract mismatch")
    if int(contract.get("minimum_calibrator_panel_overlap_studies", -1)) != 2:
        raise SystemExit("E06a v2 cross-study calibrator contract mismatch")

    coverage = metrics.get("historical_panel_proxy_coverage") or {}
    if int(coverage.get("requested_strains", -1)) != 12:
        raise SystemExit("E06a v2 requested panel size mismatch")
    if int(coverage.get("complete_requested_panel_donors", -1)) != 0:
        raise SystemExit("E06a v2 historical D365 panel unexpectedly complete")

    conditions = metrics.get("conditions") or {}
    if set(conditions) != {"b21_reference", "phase_a_target_domain"}:
        raise SystemExit("E06a v2 condition set mismatch")
    for condition, payload in conditions.items():
        nested = payload.get("historical_nested_study_out") or {}
        candidate_count = int(nested.get("outer_candidate_split_count", -1))
        scored_count = int(nested.get("outer_split_count", -1))
        skipped_count = int(nested.get("skipped_outer_split_count", -1))
        skipped = nested.get("skipped_outer_folds") or []
        if scored_count < 3 or candidate_count != scored_count + skipped_count or skipped_count != len(skipped):
            raise SystemExit(f"E06a v2 outer split accounting mismatch:{condition}")
        if not any((item or {}).get("reason") == "zero_fixed_panel_overlap" for item in skipped):
            raise SystemExit(f"E06a v2 expected real zero-panel-overlap evidence missing:{condition}")

        raw = (nested.get("raw_metrics") or {}).get("equal_study_summary") or {}
        raw_rmse = raw.get("rmse_mean")
        if not finite(raw_rmse):
            raise SystemExit(f"E06a v2 raw RMSE missing:{condition}")
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
                raise SystemExit(f"E06a v2 non-finite calibration diagnostic:{condition}:{kind}")
            if (item.get("rank_preservation") or {}).get("preserved_within_tolerance") is not True:
                raise SystemExit(f"E06a v2 held-study rank changed:{condition}:{kind}")
            if (challenge.get(kind) or {}).get("rank_preserved_within_tolerance") is not True:
                raise SystemExit(f"E06a v2 challenge rank changed:{condition}:{kind}")
            calibrators = item.get("outer_fold_calibrators") or []
            if len(calibrators) != scored_count or any(int(entry.get("training_studies", 0)) < 2 for entry in calibrators):
                raise SystemExit(f"E06a v2 calibrator study contract:{condition}:{kind}")
            print(
                f"{condition}/{kind} mean_rmse_reduction={mean_reduction} "
                f"median_study_reduction={median_reduction} worst_rmse_ratio={worst_ratio} "
                f"promotion={(promotion.get(kind) or {}).get('passed')} rank_preserved=true"
            )
    print(
        "CMI_FLU_E06A_V2_RESULT PASS aggregate_only=true submission=false "
        f"selected={metrics.get('selected_promoted_condition')} science_commit={SCIENCE_COMMIT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
