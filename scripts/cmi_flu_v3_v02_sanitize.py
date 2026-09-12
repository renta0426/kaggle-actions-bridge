#!/usr/bin/env python3
"""Emit only aggregate-safe V3-02 JSON; fail closed on row-level fields/files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SAFE_NAMES = ("v3_v02_summary.json", "v3_v02_bank_manifest.json")
FORBIDDEN_KEYS = {
    "participant_id", "subject_group", "row_index", "target", "prediction",
    "post_hai", "post_prediction", "target_log2_fold", "log2_pre_hai",
}
FORBIDDEN_FILENAMES = {"v3_v02_oof_bank.csv", "v3_v02_challenge_bank.csv"}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_KEYS:
                raise SystemExit(f"V3-02 sanitizer rejected row-level key:{path}.{key}")
            scan(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        if len(value) > 10000:
            raise SystemExit("V3-02 sanitizer rejected oversized list")
        for index, child in enumerate(value):
            scan(child, path=f"{path}[{index}]")
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise SystemExit("V3-02 sanitizer rejected nonfinite JSON value")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args(); root = args.input_dir.resolve()
    actual = sorted(p.name for p in root.iterdir() if p.is_file())
    if actual != sorted(SAFE_NAMES):
        raise SystemExit("V3-02 sanitizer safe directory allowlist mismatch")
    if any((root / name).exists() for name in FORBIDDEN_FILENAMES):
        raise SystemExit("V3-02 sanitizer found private row-level CSV")
    values = {}
    for name in SAFE_NAMES:
        path = root / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 4 * 1024 * 1024:
            raise SystemExit("V3-02 sanitizer file contract failed")
        value = json.loads(path.read_text(encoding="utf-8")); scan(value); values[name] = value
    summary = values["v3_v02_summary.json"]
    if summary.get("stage") != "V3-02" or summary.get("new_candidate_conditions") != 0:
        raise SystemExit("V3-02 sanitizer science identity mismatch")
    if summary.get("aggregate_contains_row_level_values") is not False or summary.get("public_leaderboard_used") is not False or summary.get("competition_submission_attempted") is not False:
        raise SystemExit("V3-02 sanitizer boundary mismatch")
    manifest = values["v3_v02_bank_manifest.json"]
    if manifest.get("row_level_contents_must_not_be_publicly_emitted") is not True:
        raise SystemExit("V3-02 sanitizer manifest privacy flag missing")
    files = manifest.get("files") or []
    if [x.get("filename") for x in files] != ["v3_v02_oof_bank.csv", "v3_v02_challenge_bank.csv"]:
        raise SystemExit("V3-02 sanitizer private bank file identity mismatch")
    for name in SAFE_NAMES:
        compact = json.dumps(values[name], sort_keys=True, separators=(",", ":"), allow_nan=False)
        print(f"CMI_FLU_V3_V02_SAFE_JSON name={name} sha256={sha256_path(root / name)} json={compact}")
    print("CMI_FLU_V3_V02_SANITIZE PASS aggregate_files=2 private_bank_files=2 private_contents_exposed=false competition_submit=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
