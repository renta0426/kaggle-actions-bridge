#!/usr/bin/env python3
"""Kaggle-side copier/validator for the fixed CMI-Flu 006 Public-probe family.

This script never submits to the competition. It attaches the successful private 006
Notebook as a kernel source, verifies the exact four frozen CSV hashes against the
Competition sample-submission contract, and writes manually submittable copies.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

REQUEST_ID = "20260906-cmi-flu-manual-probe-files-001"
SOURCE_KERNEL = "renta0426/cmi-flu-public-probes-20260906-006"
SOURCE_EXPECTED_VERSION = 1
SAMPLE_FILENAME = "sample_submission_part1.csv"
EXPECTED_HEADER = [
    "participant_id",
    "Task1.1",
    "Task1.2",
    "Task1.3",
    "Task1.4",
    "Task2.1",
    "Task2.2",
    "Task2.3",
]
SOURCE_FILES = {
    "control-b21-regenerated.csv": "33467e28f68f5b8f2731f3547000edc39af932cac04a233b549b6df67854b037",
    "probe-task13.csv": "6d1bac6f35ccbea549c4b23d71d6d4ddc652bf6577a853019855ea968295a717",
    "probe-task12.csv": "365607d59cd530656b929a1c1c57412cc6d375265a8d1ba10d304c64e012f387",
    "probe-task12-task13.csv": "0d6450453a5c82086372373d1f425229c8b4c3c05764b71f7c98265df4afbfc2",
}
OUTPUT_MAP = {
    "control-b21-regenerated.csv": "submission-control-b21.csv",
    "probe-task13.csv": "submission-task13-only.csv",
    "probe-task12.csv": "submission-task12-only.csv",
    "probe-task12-task13.csv": "submission-task12-task13.csv",
}
EXPECTED_ROWS = 40


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    return header, rows


def find_unique_by_hash(root: Path, basename: str, expected_hash: str) -> Path:
    candidates = [p for p in root.rglob(basename) if p.is_file()]
    matches = [p for p in candidates if sha256_path(p) == expected_hash]
    if len(matches) != 1:
        raise RuntimeError(
            f"source file contract failed for {basename}: candidates={len(candidates)} hash_matches={len(matches)}"
        )
    return matches[0]


def find_sample(root: Path) -> tuple[Path, list[dict[str, str]]]:
    matches: list[tuple[Path, list[dict[str, str]]]] = []
    for path in root.rglob(SAMPLE_FILENAME):
        if not path.is_file():
            continue
        header, rows = read_csv(path)
        if header == EXPECTED_HEADER and len(rows) == EXPECTED_ROWS:
            matches.append((path, rows))
    if len(matches) != 1:
        raise RuntimeError(f"sample-submission contract failed: valid_matches={len(matches)}")
    return matches[0]


def validate_submission_rows(
    *,
    name: str,
    rows: list[dict[str, str]],
    sample_rows: list[dict[str, str]],
) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"{name}: expected {EXPECTED_ROWS} rows, found {len(rows)}")
    ids = [row["participant_id"] for row in rows]
    sample_ids = [row["participant_id"] for row in sample_rows]
    if ids != sample_ids:
        raise RuntimeError(f"{name}: participant_id order/content differs from sample submission")
    if len(set(ids)) != EXPECTED_ROWS or any(not value for value in ids):
        raise RuntimeError(f"{name}: participant_id uniqueness/nonempty contract failed")
    for task in EXPECTED_HEADER[1:]:
        values: list[float] = []
        for row in rows:
            value = float(row[task])
            if not math.isfinite(value):
                raise RuntimeError(f"{name}: non-finite value in {task}")
            if value == -99.0:
                raise RuntimeError(f"{name}: forbidden -99 placeholder in {task}")
            values.append(value)
        if len(set(values)) <= 1:
            raise RuntimeError(f"{name}: constant prediction column {task}")


def main() -> int:
    input_root = Path("/kaggle/input")
    working_root = Path("/kaggle/working")
    if not input_root.is_dir() or not working_root.is_dir():
        raise RuntimeError("expected Kaggle input/working directories are unavailable")

    _, sample_rows = find_sample(input_root)
    output_hashes: dict[str, str] = {}

    for source_name, expected_hash in SOURCE_FILES.items():
        source_path = find_unique_by_hash(input_root, source_name, expected_hash)
        header, rows = read_csv(source_path)
        if header != EXPECTED_HEADER:
            raise RuntimeError(f"{source_name}: header differs from Competition sample submission")
        validate_submission_rows(name=source_name, rows=rows, sample_rows=sample_rows)

        output_name = OUTPUT_MAP[source_name]
        output_path = working_root / output_name
        output_path.write_bytes(source_path.read_bytes())
        found_hash = sha256_path(output_path)
        if found_hash != expected_hash:
            raise RuntimeError(f"{output_name}: byte-preserving copy hash mismatch")
        output_hashes[output_name] = found_hash

    manifest = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "source_kernel": SOURCE_KERNEL,
        "source_expected_version": SOURCE_EXPECTED_VERSION,
        "submission_rows": EXPECTED_ROWS,
        "submission_header": EXPECTED_HEADER,
        "sample_submission_validated": True,
        "participant_order_matches_sample": True,
        "all_task_values_finite": True,
        "minus_99_absent": True,
        "all_task_columns_nonconstant": True,
        "output_sha256": output_hashes,
        "competition_submission_attempted": False,
    }
    (working_root / "manual-submission-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "CMI_FLU_MANUAL_PROBE_FILES_001 PASS "
        f"rows={EXPECTED_ROWS} files={len(output_hashes)} "
        + " ".join(f"{name}={digest}" for name, digest in sorted(output_hashes.items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
