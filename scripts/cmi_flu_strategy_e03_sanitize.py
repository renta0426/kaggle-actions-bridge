#!/usr/bin/env python3
"""Validate downloaded E03 outputs and render only aggregate-safe diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260908-cmi-flu-strategy-e03-task11-optional-view-fusion-001"
SCIENCE_COMMIT = "2eaf8fac9a632c67f32c4a0ed37b64eeac571387"
E03_BLOB = "e38fe01606dc5653ad3d1542f722726c1374b278"
TASK11_PRIOR_BLOB = "50d9a43604d2b75479b8f873a86a8daf9d5bd7a9"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
PACKAGE_SHA256 = "48ff4ed2eadec8b059b8d37fb677af1f249b6486bccfbd55dca2274cfc6f3dc3"
ALLOWED = {"bridge-result.json", "metrics.json", "summary.md"}
CONDITIONS = (
    "b21",
    "hai_raw_fusion_w0.25",
    "hai_rank_fusion_w0.25",
    "hai_crossfit_residual_w0.25",
    "innate_flow_rank_fusion_w0.25",
    "innate_flow_crossfit_residual_w0.25",
)
FORBIDDEN_KEYS = {
    "participant_id", "participant_ids", "row_prediction", "row_predictions",
    "prediction_rows_raw", "predicted_values", "submission", "submission_rows_raw",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan(value, path="root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).strip().casefold() in FORBIDDEN_KEYS:
                raise SystemExit(f"forbidden E03 output key: {path}.{key}")
            scan(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            scan(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise SystemExit(f"non-finite E03 output value: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.expanduser().resolve()
    names = {path.name for path in root.iterdir() if path.is_file() or path.is_symlink()}
    if names != ALLOWED:
        raise SystemExit(f"E03 output allowlist mismatch: {sorted(names)}")
    if any((root / name).is_symlink() for name in ALLOWED):
        raise SystemExit("E03 output contains symlink")
    if (root / "bridge-result.json").stat().st_size > 1_048_576:
        raise SystemExit("E03 bridge result exceeds size budget")
    if (root / "metrics.json").stat().st_size > 10_485_760:
        raise SystemExit("E03 metrics exceeds size budget")
    if (root / "summary.md").stat().st_size > 1_048_576:
        raise SystemExit("E03 summary exceeds size budget")

    bridge = json.loads((root / "bridge-result.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    scan(bridge)
    scan(metrics)
    expected_bridge = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e03_blob_sha": E03_BLOB,
        "task11_prior_immunity_blob_sha": TASK11_PRIOR_BLOB,
        "config_blob_sha": CONFIG_BLOB,
        "frozen_b21_package_sha256": PACKAGE_SHA256,
        "status": "complete",
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
    }
    for key, value in expected_bridge.items():
        if bridge.get(key) != value:
            raise SystemExit(f"E03 bridge result mismatch: {key}")
    if bridge.get("metrics_sha256") != digest(root / "metrics.json"):
        raise SystemExit("E03 metrics hash mismatch")
    if bridge.get("summary_sha256") != digest(root / "summary.md"):
        raise SystemExit("E03 summary hash mismatch")

    if metrics.get("schema_version") != 1:
        raise SystemExit("E03 science schema mismatch")
    if metrics.get("experiment") != "strategy_v2_e03_task11_optional_view_fusion":
        raise SystemExit("E03 experiment identity mismatch")
    if metrics.get("task") != "Task1.1" or tuple(metrics.get("conditions", ())) != CONDITIONS:
        raise SystemExit("E03 condition contract mismatch")
    if metrics.get("competition_submission_attempted") is not False:
        raise SystemExit("E03 unexpectedly attempted competition submission")
    if metrics.get("leaderboard_used_for_selection") is not False:
        raise SystemExit("E03 unexpectedly used leaderboard for selection")
    if metrics.get("output_policy") != "aggregate_only_no_participant_ids_or_row_level_predictions":
        raise SystemExit("E03 output policy mismatch")
    challenge = metrics.get("challenge") or {}
    if int(challenge.get("rows", -1)) != 40:
        raise SystemExit("E03 Challenge snapshot mismatch")
    folds = metrics.get("folds") or []
    if len(folds) != 4:
        raise SystemExit("E03 held-study fold count mismatch")

    paired = metrics.get("paired_vs_b21") or {}
    safe = {
        "request_id": REQUEST_ID,
        "experiment": metrics.get("experiment"),
        "selected_promoted_condition": metrics.get("selected_promoted_condition"),
        "flow_panel": {
            key: (metrics.get("flow_panel") or {}).get(key)
            for key in (
                "state", "selected_count", "selected_subjects", "selected_studies",
                "broad_cross_study_replacement_claim_permitted",
            )
        },
        "study_equal_weight_spearman": {
            name: (metrics.get("summary") or {}).get(name, {}).get("study_equal_weight_spearman_mean_strict")
            for name in CONDITIONS
        },
        "paired_delta_vs_b21": {
            name: (paired.get(name) or {}).get("study_mean_delta_strict")
            for name in CONDITIONS if name != "b21"
        },
        "passes_candidate_delta_heuristic": {
            name: (paired.get(name) or {}).get("passes_candidate_delta_heuristic")
            for name in CONDITIONS if name != "b21"
        },
        "passes_guardrail": {
            name: (paired.get(name) or {}).get("passes_guardrail")
            for name in CONDITIONS if name != "b21"
        },
        "large_studies_requiring_review": {
            name: (paired.get(name) or {}).get("large_studies_requiring_review")
            for name in CONDITIONS if name != "b21"
        },
        "challenge_coverage": {
            "rows": challenge.get("rows"),
            "hai_rows": challenge.get("hai_rows"),
            "flow_rows": challenge.get("flow_rows"),
        },
        "metrics_sha256": digest(root / "metrics.json"),
        "summary_sha256": digest(root / "summary.md"),
    }
    scan(safe)
    print("CMI_FLU_E03_SANITIZE PASS")
    print("E03_SAFE_RESULT=" + json.dumps(safe, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
