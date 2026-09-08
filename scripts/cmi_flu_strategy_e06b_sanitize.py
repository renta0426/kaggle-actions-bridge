#!/usr/bin/env python3
"""Validate and print only aggregate E06b Task2.3 two-head results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260908-cmi-flu-strategy-e06b-task23-two-head-001"
SCIENCE_COMMIT = "d172adc2778f3e4fea7692d2a2129db402a994c4"
E06B_BLOB = "5b27abb1bb515e9388e63e7dd4c350aebea90574"
E06B_V2_BLOB = "fe05fd9497f5a80b56b96fbf5f2971a9d43f4e10"
BANNED = (
    '"participant_id"',
    '"subject_group"',
    '"row_index"',
    '"oof_predictions"',
    '"challenge_predictions"',
)
CHECKS = {
    "equal_study_mean_rmse_reduction_at_least_2pct",
    "median_study_rmse_improves",
    "no_study_rmse_worse_than_5pct",
    "equal_study_spearman_not_materially_worse",
    "no_large_study_spearman_decline_below_minus_0.10",
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
        raise SystemExit(f"E06b safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E06b aggregate privacy contract failed:{name}")

    bridge = json.loads(texts["bridge-result.json"])
    metrics = json.loads(texts["metrics.json"])
    expected_bridge = {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e06b_blob_sha": E06B_BLOB,
        "strategy_e06b_v2_blob_sha": E06B_V2_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
    for key, expected in expected_bridge.items():
        if bridge.get(key) != expected:
            raise SystemExit(f"E06b bridge contract mismatch:{key}:{bridge.get(key)!r}")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json"):
        raise SystemExit("E06b metrics hash mismatch")
    if bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E06b summary hash mismatch")

    if metrics.get("experiment") != "strategy_v2_e06b_task23_d28_d365_two_head":
        raise SystemExit("E06b experiment identity mismatch")
    if metrics.get("task") != "Task2.3" or int(metrics.get("primary_day", -1)) != 365 or int(metrics.get("auxiliary_day", -1)) != 28:
        raise SystemExit("E06b task/day identity mismatch")
    if metrics.get("competition_submission_attempted") is not False or metrics.get("leaderboard_used_for_selection") is not False:
        raise SystemExit("E06b selection/submission contract mismatch")
    model_contract = metrics.get("model_contract") or {}
    if model_contract.get("observed_d28_used_as_d365_feature") is not False:
        raise SystemExit("E06b D28 leakage contract mismatch")
    if model_contract.get("two_head_fit_adapter") != "strategy_e06b_v2_ci_typo_fix":
        raise SystemExit("E06b two-head adapter contract mismatch")

    conditions = metrics.get("conditions") or {}
    if set(conditions) != {"b21_reference", "phase_a_target_domain"}:
        raise SystemExit("E06b condition set mismatch")
    for condition, payload in conditions.items():
        historical = payload.get("historical_study_out") or {}
        candidate_count = int(historical.get("outer_candidate_split_count", -1))
        scored_count = int(historical.get("outer_split_count", -1))
        skipped_count = int(historical.get("skipped_outer_split_count", -1))
        if scored_count < 3 or candidate_count != scored_count + skipped_count:
            raise SystemExit(f"E06b outer split accounting:{condition}")
        independent = (historical.get("independent_d365") or {}).get("equal_study_summary") or {}
        shared = (historical.get("shared_two_head") or {}).get("equal_study_summary") or {}
        if not all(finite(v) for v in (independent.get("rmse_mean"), independent.get("spearman_mean"), shared.get("rmse_mean"), shared.get("spearman_mean"))):
            raise SystemExit(f"E06b nonfinite study summary:{condition}")
        paired = payload.get("paired_diagnostics") or {}
        rmse = paired.get("rmse") or {}; spearman = paired.get("spearman") or {}
        mean_rmse = rmse.get("equal_study_mean_relative_reduction")
        median_rmse = rmse.get("median_study_relative_reduction")
        worst_ratio = rmse.get("worst_study_rmse_ratio")
        spearman_delta = spearman.get("equal_study_mean_delta")
        if not all(finite(v) for v in (mean_rmse, median_rmse, worst_ratio, spearman_delta)):
            raise SystemExit(f"E06b nonfinite paired diagnostic:{condition}")
        decision = payload.get("promotion") or {}; checks = decision.get("checks") or {}; passed = decision.get("passed")
        if set(checks) != CHECKS or any(not isinstance(v, bool) for v in checks.values()) or not isinstance(passed, bool) or passed is not bool(all(checks.values())):
            raise SystemExit(f"E06b promotion contract:{condition}")
        challenge = payload.get("challenge_shared_vs_independent_rank") or {}
        rho = (challenge.get("rank_spearman") or {}).get("spearman")
        if int(challenge.get("donor_count", 0)) != 40 or not finite(rho):
            raise SystemExit(f"E06b Challenge rank contract:{condition}")
        print(
            f"{condition} outer_candidate={candidate_count} outer_scored={scored_count} outer_skipped={skipped_count} "
            f"independent_rmse={independent.get('rmse_mean')} shared_rmse={shared.get('rmse_mean')} "
            f"mean_rmse_reduction={mean_rmse} median_rmse_reduction={median_rmse} worst_rmse_ratio={worst_ratio} "
            f"spearman_delta={spearman_delta} challenge_rank_spearman={rho} promotion={passed}"
        )

    selected = metrics.get("selected_promoted_condition")
    if selected not in {None, "b21_reference", "phase_a_target_domain"}:
        raise SystemExit("E06b selected condition invalid")
    if selected is not None and ((conditions.get(selected) or {}).get("promotion") or {}).get("passed") is not True:
        raise SystemExit("E06b selected condition not promoted")
    print(
        "CMI_FLU_E06B_RESULT PASS aggregate_only=true submission=false "
        f"selected={selected} science_commit={SCIENCE_COMMIT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
