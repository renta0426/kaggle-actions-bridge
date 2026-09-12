#!/usr/bin/env python3
"""Sanitize aggregate-only V3-06 result files for public runner logs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SAFE_NAMES = ["v3_v06_summary.json", "v3_v06_bank_manifest.json"]
PRIVATE_NAMES = ["v3_v06_oof_bank.csv", "v3_v06_challenge_bank.csv"]
FORBIDDEN_KEYS = {
    "participant_id",
    "subject",
    "subject_group",
    "row_index",
    "target",
    "targets",
    "prediction",
    "predictions",
    "raw_et_prediction",
    "calibrated_prediction",
    "oof_predictions",
    "challenge_predictions",
}


def walk(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in FORBIDDEN_KEYS:
                raise RuntimeError(f"V3-06 sanitizer forbidden key:{key}")
            walk(item)
    elif isinstance(value, list):
        for item in value:
            walk(item)


def validate(summary: dict, manifest: dict) -> None:
    if summary.get("stage") != "V3-06":
        raise RuntimeError("V3-06 sanitizer stage identity changed")
    if summary.get("new_candidate_conditions") != [
        "task21_et_log_affine", "task22_et_log_affine"
    ]:
        raise RuntimeError("V3-06 sanitizer condition set changed")
    if int(summary.get("new_candidate_condition_count", -1)) != 2:
        raise RuntimeError("V3-06 sanitizer condition count changed")
    if int(summary.get("fit_count", 999999)) > 256:
        raise RuntimeError("V3-06 sanitizer fit cap violated")
    if summary.get("aggregate_contains_row_level_values") is not False:
        raise RuntimeError("V3-06 sanitizer aggregate privacy flag changed")
    if summary.get("private_bank_contains_row_level_values") is not True:
        raise RuntimeError("V3-06 sanitizer private-bank flag changed")
    if summary.get("competition_submission_attempted") is not False:
        raise RuntimeError("V3-06 sanitizer submission boundary changed")
    if summary.get("public_leaderboard_used") is not False:
        raise RuntimeError("V3-06 sanitizer Public boundary changed")
    contract = summary.get("calibration_contract") or {}
    if contract.get("kind") != "log2_affine" or contract.get("inner_study_folds") != 3:
        raise RuntimeError("V3-06 sanitizer calibration contract changed")
    if contract.get("affine_or_power_grid_reopened") is not False:
        raise RuntimeError("V3-06 sanitizer forbidden grid reopened")
    runtime = summary.get("runtime") or {}
    if runtime.get("runtime_terminal_marker") != "CMI_FLU_V3_V06_RUNTIME_PASS":
        raise RuntimeError("V3-06 sanitizer runtime marker missing")
    files = manifest.get("files") or []
    if [item.get("filename") for item in files] != PRIVATE_NAMES:
        raise RuntimeError("V3-06 sanitizer manifest private file set changed")
    if manifest.get("row_level_contents_must_not_be_publicly_emitted") is not True:
        raise RuntimeError("V3-06 sanitizer manifest privacy flag changed")
    if manifest.get("competition_submission_attempted") is not False:
        raise RuntimeError("V3-06 sanitizer manifest submission boundary changed")
    walk(summary)
    walk(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    actual = sorted(p.name for p in root.iterdir() if p.is_file())
    if actual != sorted(SAFE_NAMES):
        raise SystemExit("V3-06 safe result directory allowlist mismatch")
    summary = json.loads((root / SAFE_NAMES[0]).read_text())
    manifest = json.loads((root / SAFE_NAMES[1]).read_text())
    validate(summary, manifest)
    for name, data in ((SAFE_NAMES[0], summary), (SAFE_NAMES[1], manifest)):
        raw = (root / name).read_bytes()
        encoded = json.dumps(data, separators=(",", ":"), sort_keys=True, allow_nan=False)
        print(
            f"CMI_FLU_V3_V06_SAFE_JSON name={name} bytes={len(raw)} "
            f"sha256={hashlib.sha256(raw).hexdigest()} json={encoded}"
        )
    print("CMI_FLU_V3_V06_SANITIZE PASS files=2 row_level=false competition_submit=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
