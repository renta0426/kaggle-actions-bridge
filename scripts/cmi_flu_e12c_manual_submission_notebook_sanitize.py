#!/usr/bin/env python3
"""Validate E12c manual-submission Notebook outputs without exposing row-level values."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260911-cmi-flu-e12c-manual-submission-notebook-001"
SCIENCE_COMMIT = "bb6f41ed81b0bbd4ecdc397c6abb9df671c6ebf8"
E12C_BLOB = "a495584a7e461478bd1a41d01d2536c4435a8f7f"
FINAL_CSV_SHA256 = "983aaf097d04477c4ccf7bf817fdf66e552937e69bceaa48cfb260cb84413f1b"
FINAL_CSV_BYTES = 5933
HISTORICAL_SHA256 = "365607d59cd530656b929a1c1c57412cc6d375265a8d1ba10d304c64e012f387"
HISTORICAL_SOURCE = "renta0426/cmi-flu-manual-probe-files-20260906-001"
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.expanduser().resolve()

    expected_files = {"submission.csv", "manual-submission-manifest.json"}
    found = {path.name for path in root.iterdir() if path.is_file()}
    if found != expected_files:
        raise SystemExit(f"E12c output file set mismatch:{sorted(found)}")
    if any(path.is_symlink() for path in root.iterdir()):
        raise SystemExit("E12c output symlink forbidden")

    submission = root / "submission.csv"
    manifest_path = root / "manual-submission-manifest.json"
    if submission.stat().st_size != FINAL_CSV_BYTES:
        raise SystemExit("E12c submission byte count mismatch")
    if sha256(submission) != FINAL_CSV_SHA256:
        raise SystemExit("E12c submission hash mismatch")

    with submission.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    if header != EXPECTED_HEADER or len(rows) != 40:
        raise SystemExit("E12c submission shape/header mismatch")
    ids = [row["participant_id"] for row in rows]
    if len(set(ids)) != 40 or any(not value for value in ids):
        raise SystemExit("E12c participant identity structure mismatch")
    for task in EXPECTED_HEADER[1:]:
        values = [float(row[task]) for row in rows]
        if not all(math.isfinite(value) and value != -99.0 for value in values):
            raise SystemExit(f"E12c invalid task value:{task}")
        if len(set(values)) < 2:
            raise SystemExit(f"E12c constant task column:{task}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "experiment": "strategy_v2_e12c_manual_submission_generator",
        "science_commit": SCIENCE_COMMIT,
        "strategy_e12c_blob_sha": E12C_BLOB,
        "parent_e12b_semantic_sha256": "5faf93bee4b68aba23c53e09adbb251bdd07d287dd48fb6d94d6fce53ef9a60f",
        "submission_filename": "submission.csv",
        "submission_sha256": FINAL_CSV_SHA256,
        "submission_bytes": FINAL_CSV_BYTES,
        "submission_rows": 40,
        "submission_columns": EXPECTED_HEADER,
        "historical_source_kernel": HISTORICAL_SOURCE,
        "historical_source_version": 1,
        "historical_task12_only_sha256": HISTORICAL_SHA256,
        "changed_tasks_vs_historical_task12_only": ["Task1.3"],
        "task13_only_intervention_proven": True,
        "manual_submission_ready": True,
        "manual_operator_submission_only": True,
        "competition_submission_attempted": False,
        "competition_submission_authorized": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise SystemExit(f"E12c manifest mismatch:{key}")
    md5_count = manifest.get("md5_verified_count")
    if isinstance(md5_count, bool) or not isinstance(md5_count, int) or md5_count <= 0:
        raise SystemExit("E12c manifest MD5 verification count invalid")

    # Never echo participant IDs or predictions. The only public result is the
    # fixed fingerprint and aggregate contract state.
    print(
        "CMI_FLU_E12C_MANUAL_SANITIZE PASS "
        f"rows=40 bytes={FINAL_CSV_BYTES} submission_sha256={FINAL_CSV_SHA256} "
        f"historical_sha256={HISTORICAL_SHA256} changed=Task1.3 "
        "submission_api=false manual_operator_submission_only=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
