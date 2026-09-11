#!/usr/bin/env python3
"""Validate and emit only aggregate-safe CMI-Flu V3 batch-1 results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SAFE_NAMES = [
    "teacher_ledger.json", "measurement_contracts.json", "split_support.json",
    "auxiliary_label_coverage.json", "source_alignment_audit.json",
    "diagnostic_manifest.json", "runtime_receipt.json",
]
FORBIDDEN_KEYS = {"participant_id", "subject_id", "subject_group", "prediction", "target_value", "teacher_value"}
SOURCE_B_SHA256 = "0f9df53c3aa8c6e4ac693f6a42dbd2633b4b61d1462df798a9bd767c220be3a5"


def walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args(); root = args.input_dir.resolve()
    actual = sorted(path.name for path in root.iterdir() if path.is_file())
    if actual != sorted(SAFE_NAMES):
        raise SystemExit(f"V3 batch1 recovered-safe allowlist mismatch:{actual}")
    payloads = {}
    for name in SAFE_NAMES:
        path = root / name
        if path.stat().st_size > 2097152:
            raise SystemExit(f"V3 batch1 safe output too large:{name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        bad = sorted(set(walk_keys(payload)) & FORBIDDEN_KEYS)
        if bad:
            raise SystemExit(f"V3 batch1 unsafe row-level keys:{name}:{bad}")
        payloads[name] = payload
    manifest = payloads["diagnostic_manifest.json"]
    files = manifest.get("files") or []
    if len(files) != 6 or manifest.get("diagnostic_not_final") is not True:
        raise SystemExit("V3 batch1 diagnostic manifest invalid")
    if any(item.get("source_csv_sha256") != SOURCE_B_SHA256 for item in files):
        raise SystemExit("V3 batch1 diagnostic source hash mismatch")
    receipt = payloads["runtime_receipt.json"]
    if receipt.get("model_fit_count") != 0 or receipt.get("competition_submission_attempted") is not False:
        raise SystemExit("V3 batch1 receipt no-fit/submit boundary invalid")
    if receipt.get("runtime_terminal_marker") != "CMI_FLU_V3_BATCH1_RUNTIME_PASS":
        raise SystemExit("V3 batch1 runtime success marker absent")
    for name in SAFE_NAMES:
        encoded = json.dumps(payloads[name], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        print(f"CMI_FLU_V3_BATCH1_SAFE_JSON name={name} sha256={hashlib.sha256((root / name).read_bytes()).hexdigest()} json={encoded}")
    print(f"CMI_FLU_V3_BATCH1_SANITIZE PASS aggregate_files={len(SAFE_NAMES)} diagnostic_files=6 diagnostic_not_final=true model_fit_count=0 competition_submit=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
