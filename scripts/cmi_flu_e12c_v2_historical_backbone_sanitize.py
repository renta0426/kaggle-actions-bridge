#!/usr/bin/env python3
"""Validate E12c-v2 output without exposing participant-level values."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct

REQUEST_ID = "20260911-cmi-flu-e12c-v2-historical-backbone-002"
SCIENCE_COMMIT = "9916c04a3d5510eead8f361960d73b5d94201892"
E12C_V2_BLOB = "238d3ad67984fdd8c375bb0aa263bb4717029027"
HISTORICAL_SHA256 = "365607d59cd530656b929a1c1c57412cc6d375265a8d1ba10d304c64e012f387"
TASK13_SHA256 = "de8bc3b6bbd3e3aad4099b83eb63a1ebbd81c4f4eeb60f77808858f4674be5da"
SEMANTIC_VERSION = "cmi-flu-e12b-semantic-v1"
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
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_sha(rows: list[dict[str, str]], task: str) -> str:
    digest = hashlib.sha256()
    digest.update((SEMANTIC_VERSION + ":" + task + "\n").encode("utf-8"))
    for row in rows:
        participant = str(row["participant_id"]).encode("utf-8")
        digest.update(struct.pack(">I", len(participant)))
        digest.update(participant)
        digest.update(struct.pack(">d", float(row[task])))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.expanduser().resolve()

    expected_files = {"submission.csv", "manual-submission-manifest.json"}
    found = {path.name for path in root.iterdir() if path.is_file()}
    if found != expected_files:
        raise SystemExit(f"E12c-v2 output file set mismatch:{sorted(found)}")
    if any(path.is_symlink() for path in root.iterdir()):
        raise SystemExit("E12c-v2 output symlink forbidden")

    submission = root / "submission.csv"
    manifest_path = root / "manual-submission-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    fixed = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "experiment": "strategy_v2_e12c_v2_historical_backbone_manual_submission",
        "science_commit": SCIENCE_COMMIT,
        "strategy_e12c_v2_blob_sha": E12C_V2_BLOB,
        "historical_backbone_sha256": HISTORICAL_SHA256,
        "historical_public_score": 0.218,
        "changed_tasks_vs_historical_0_218": ["Task1.3"],
        "task13_prediction_sha256": TASK13_SHA256,
        "task13_unique_values": 36,
        "task13_historical_reproduced": True,
        "submission_filename": "submission.csv",
        "submission_rows": 40,
        "manual_submission_ready": True,
        "manual_operator_submission_only": True,
        "competition_submission_attempted": False,
        "competition_submission_authorized": False,
        "leaderboard_used_for_selection": False,
        "refit_unchanged_tasks": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
    for key, value in fixed.items():
        if manifest.get(key) != value:
            raise SystemExit(f"E12c-v2 manifest mismatch:{key}")

    submission_sha = manifest.get("submission_sha256")
    semantic_sha = manifest.get("semantic_submission_sha256")
    if not isinstance(submission_sha, str) or not HEX64.fullmatch(submission_sha):
        raise SystemExit("E12c-v2 manifest submission hash invalid")
    if not isinstance(semantic_sha, str) or not HEX64.fullmatch(semantic_sha):
        raise SystemExit("E12c-v2 manifest semantic hash invalid")
    submission_bytes = manifest.get("submission_bytes")
    if isinstance(submission_bytes, bool) or not isinstance(submission_bytes, int):
        raise SystemExit("E12c-v2 manifest submission byte count invalid")
    if not 1000 <= submission_bytes <= 262144:
        raise SystemExit("E12c-v2 manifest submission byte count out of bounds")
    if submission.stat().st_size != submission_bytes:
        raise SystemExit("E12c-v2 submission byte count mismatch")
    if sha256(submission) != submission_sha:
        raise SystemExit("E12c-v2 submission hash mismatch")

    with submission.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    if header != EXPECTED_HEADER or len(rows) != 40:
        raise SystemExit("E12c-v2 submission shape/header mismatch")
    ids = [row["participant_id"] for row in rows]
    if len(set(ids)) != 40 or any(not value for value in ids):
        raise SystemExit("E12c-v2 participant identity structure mismatch")
    for task in EXPECTED_HEADER[1:]:
        values = [float(row[task]) for row in rows]
        if not all(math.isfinite(value) and value != -99.0 for value in values):
            raise SystemExit(f"E12c-v2 invalid task value:{task}")
        if len(set(values)) < 2:
            raise SystemExit(f"E12c-v2 constant task column:{task}")
    if len(set(float(row["Task1.3"]) for row in rows)) != 36:
        raise SystemExit("E12c-v2 Task1.3 unique-value contract mismatch")
    if task_sha(rows, "Task1.3") != TASK13_SHA256:
        raise SystemExit("E12c-v2 Task1.3 task fingerprint mismatch")

    md5_count = manifest.get("md5_verified_count")
    if isinstance(md5_count, bool) or not isinstance(md5_count, int) or md5_count <= 0:
        raise SystemExit("E12c-v2 manifest MD5 verification count invalid")

    print(
        "CMI_FLU_E12C_V2_SANITIZE PASS "
        f"rows=40 bytes={submission_bytes} submission_sha256={submission_sha} "
        f"semantic_sha256={semantic_sha} historical_sha256={HISTORICAL_SHA256} "
        f"task13_sha256={TASK13_SHA256} changed=Task1.3 refit_unchanged=false "
        "submission_api=false manual_operator_submission_only=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
