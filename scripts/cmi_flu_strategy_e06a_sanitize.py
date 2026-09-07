#!/usr/bin/env python3
"""Validate and print only aggregate E06a Task2.3 calibration results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260908-cmi-flu-strategy-e06a-task23-calibration-001"
SCIENCE_COMMIT = "9fd643e01b4aae4a8f7815006960022947eaa4e3"
E06_BLOB = "dd5b5ed12e7667e527c97a99e86b8821e68e6442"
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
        raise SystemExit(f"E06a safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E06a aggregate privacy contract failed:{name}")

    bridge = json.loads(texts["bridge-result.json"])
    metrics = json.loads(texts["metrics.json"])
    expected_bridge = {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e06_blob_sha": E06_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
    for key, expected in expected_bridge.items():
        if bridge.get(key) != expected:
            raise SystemExit(f"E06a bridge contract mismatch:{key}:{bridge.get(key)!r}")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json"):
        raise SystemExit("E06a metrics hash mismatch")
    if bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E06a summary hash mismatch")

    if metrics.get("experiment") != "strategy_v2_e06a_task23_rank_preserving_calibration":
        raise SystemExit("E06a experiment identity mismatch")
    if metrics.get("task") != "Task2.3" or int(metrics.get("target_day", -1)) != 365:
        raise SystemExit("E06a task identity mismatch")
    if metrics.get("competition_submission_attempted") is not False or metrics.get("leaderboard_used_for_selection") is not False:
        raise SystemExit("E06a selection/submission contract mismatch")
    coverage = metrics.get("historical_panel_proxy_coverage") or {}
    if int(coverage.get("requested_strains", -1)) != 12:
        raise SystemExit("E06a requested panel size mismatch")
    if int(coverage.get("complete_requested_panel_donors", -1)) != 0:
        raise SystemExit("E06a historical D365 panel unexpectedly complete")

    conditions = metrics.get("conditions") or {}
    if set(conditions) != {"b21_reference", "phase_a_target_domain"}:
        raise SystemExit("E06a condition set mismatch")
    for condition, payload in conditions.items():
        nested = payload.get("historical_nested_study_out") or {}
        raw = (nested.get("raw_metrics") or {}).get("equal_study_summary") or {}
        raw_rmse = raw.get("rmse_mean")
        if not finite(raw_rmse):
            raise SystemExit(f"E06a raw RMSE missing:{condition}")
        print(
            f"{condition} raw_equal_study_rmse_mean={raw_rmse} "
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
                raise SystemExit(f"E06a non-finite calibration diagnostic:{condition}:{kind}")
            if (item.get("rank_preservation") or {}).get("preserved_within_tolerance") is not True:
                raise SystemExit(f"E06a held-study rank changed:{condition}:{kind}")
            if (challenge.get(kind) or {}).get("rank_preserved_within_tolerance") is not True:
                raise SystemExit(f"E06a challenge rank changed:{condition}:{kind}")
            print(
                f"{condition}/{kind} mean_rmse_reduction={mean_reduction} "
                f"median_study_reduction={median_reduction} worst_rmse_ratio={worst_ratio} "
                f"promotion={(promotion.get(kind) or {}).get('passed')} rank_preserved=true"
            )
    print(
        "CMI_FLU_E06A_RESULT PASS aggregate_only=true submission=false "
        f"selected={metrics.get('selected_promoted_condition')} science_commit={SCIENCE_COMMIT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
