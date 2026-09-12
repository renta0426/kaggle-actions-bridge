#!/usr/bin/env python3
"""Sanitize aggregate-only V3-05 result files for runner logs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SAFE_NAMES = ["v3_v05_task13_summary.json", "v3_v05_task13_bank_manifest.json"]
PRIVATE_NAMES = ["v3_v05_task13_oof_bank.csv", "v3_v05_task13_challenge_bank.csv"]
FORBIDDEN_KEYS = {
    "participant_id", "subject", "subject_group", "row_index", "split",
    "strict_ASC_D7_target", "strict_ASC_raw_baseline", "prediction", "predictions",
    "target", "targets", "oof_predictions", "challenge_predictions",
}


def walk(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in FORBIDDEN_KEYS:
                raise RuntimeError(f"V3-05 sanitizer forbidden key:{key}")
            walk(item, path + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            walk(item, path + f"[{index}]")


def validate(summary: dict, manifest: dict) -> None:
    if summary.get("stage") != "V3-05" or summary.get("task") != "Task1.3":
        raise RuntimeError("V3-05 sanitizer stage identity changed")
    if summary.get("new_candidate_conditions") != ["S1", "S2", "S3", "S4"]:
        raise RuntimeError("V3-05 sanitizer condition set changed")
    if int(summary.get("new_candidate_condition_count", -1)) != 4:
        raise RuntimeError("V3-05 sanitizer condition count changed")
    if int(summary.get("fit_count", 999999)) > 96:
        raise RuntimeError("V3-05 sanitizer fit cap violated")
    if (summary.get("Task1.2") or {}).get("state") != "not_executed_data_limited":
        raise RuntimeError("V3-05 sanitizer Task1.2 boundary changed")
    if summary.get("aggregate_contains_row_level_values") is not False:
        raise RuntimeError("V3-05 sanitizer aggregate privacy flag changed")
    if summary.get("private_bank_contains_row_level_values") is not True:
        raise RuntimeError("V3-05 sanitizer private-bank flag changed")
    for key in (
        "competition_submission_attempted", "public_leaderboard_used",
        "raw_unit_conversion_performed", "pseudocount_added",
    ):
        if summary.get(key) is not False:
            raise RuntimeError(f"V3-05 sanitizer boundary changed:{key}")
    runtime = summary.get("runtime") or {}
    if runtime.get("runtime_terminal_marker") != "CMI_FLU_V3_V05_RUNTIME_PASS":
        raise RuntimeError("V3-05 sanitizer runtime marker missing")
    files = manifest.get("files") or []
    if [item.get("filename") for item in files] != PRIVATE_NAMES:
        raise RuntimeError("V3-05 sanitizer manifest private file set changed")
    if manifest.get("row_level_contents_must_not_be_publicly_emitted") is not True:
        raise RuntimeError("V3-05 sanitizer manifest privacy flag changed")
    if manifest.get("competition_submission_attempted") is not False:
        raise RuntimeError("V3-05 sanitizer manifest submission boundary changed")
    walk(summary)
    walk(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    actual = sorted(p.name for p in root.iterdir() if p.is_file())
    if actual != sorted(SAFE_NAMES):
        raise SystemExit("V3-05 safe result directory allowlist mismatch")
    summary = json.loads((root / SAFE_NAMES[0]).read_text())
    manifest = json.loads((root / SAFE_NAMES[1]).read_text())
    validate(summary, manifest)
    for name, data in ((SAFE_NAMES[0], summary), (SAFE_NAMES[1], manifest)):
        raw = (root / name).read_bytes()
        encoded = json.dumps(data, separators=(",", ":"), sort_keys=True, allow_nan=False)
        print(
            f"CMI_FLU_V3_V05_SAFE_JSON name={name} bytes={len(raw)} "
            f"sha256={hashlib.sha256(raw).hexdigest()} json={encoded}"
        )
    print("CMI_FLU_V3_V05_SANITIZE PASS files=2 row_level=false competition_submit=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
