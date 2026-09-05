#!/usr/bin/env python3
"""Read only the exact current-v1 B2.1 submission from the legacy Notebook.

B2.1 predates the newer output-hygiene/discoverability contract: its private title
contains ``B2.1`` while its slug contains ``b21``, and its successful working tree
retained runtime scratch.  Therefore the generic current-output helper is not a
valid reader for this one frozen Notebook.

This helper is intentionally non-general.  It accepts exactly one hard-coded
private kernel/version, proves identity through direct Kaggle metadata and the
saved bridge manifest, downloads current output into a temporary directory with
captured diagnostics, and exports only ``submission.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

KERNEL = "renta0426/cmi-flu-b21-robust-cv-20260903-001"
EXPECTED_VERSION = 1
EXPECTED_REQUEST_ID = "20260903-cmi-flu-b21-001"
EXPECTED_STAGE = "b21_taskwise_robust"
EXPECTED_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"
EXPECTED_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
EXPECTED_COLUMNS = (
    "participant_id",
    "Task1.1",
    "Task1.2",
    "Task1.3",
    "Task1.4",
    "Task2.1",
    "Task2.2",
    "Task2.3",
)
ACTIVE = ("RUNNING", "QUEUED", "PENDING")
MAX_TOTAL_DOWNLOAD_BYTES = 128 * 1024 * 1024
MAX_SUBMISSION_BYTES = 262144
MAX_MANIFEST_BYTES = 1048576


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def direct_verify(api: KaggleApi) -> str:
    status = str(getattr(api.kernels_status(KERNEL), "status", "")).upper()
    if not status or any(token in status for token in ACTIVE):
        raise RuntimeError(f"legacy B2.1 kernel is not terminal: status={status or 'UNKNOWN'}")

    owner, slug = KERNEL.split("/", 1)
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name = owner
        request.kernel_slug = slug
        metadata = client.kernels.kernels_api_client.get_kernel(request).metadata

    if str(getattr(metadata, "ref", "")) != KERNEL:
        raise RuntimeError("legacy B2.1 direct metadata identity mismatch")
    if not bool(getattr(metadata, "is_private", False)):
        raise RuntimeError("legacy B2.1 kernel is unexpectedly public")
    observed = int(getattr(metadata, "current_version_number", 0) or 0)
    if observed != EXPECTED_VERSION:
        raise RuntimeError(
            f"legacy B2.1 current version mismatch observed={observed} expected={EXPECTED_VERSION}; "
            "historical-version substitution is forbidden"
        )
    return status


def validate_submission_bytes(data: bytes) -> None:
    if not data or len(data) > MAX_SUBMISSION_BYTES:
        raise RuntimeError("legacy B2.1 submission byte contract violated")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("legacy B2.1 submission is not UTF-8") from error
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or tuple(rows[0]) != EXPECTED_COLUMNS:
        raise RuntimeError("legacy B2.1 submission header mismatch")
    if len(rows) != 41:
        raise RuntimeError(f"legacy B2.1 submission expected 40 data rows, found {len(rows) - 1}")
    if any(len(row) != len(EXPECTED_COLUMNS) for row in rows):
        raise RuntimeError("legacy B2.1 submission row width mismatch")
    ids = [row[0] for row in rows[1:]]
    if any(not value for value in ids) or len(set(ids)) != 40:
        raise RuntimeError("legacy B2.1 submission participant IDs are missing or duplicated")
    for row in rows[1:]:
        for value in row[1:]:
            number = float(value)
            if not (-float("inf") < number < float("inf")):
                raise RuntimeError("legacy B2.1 submission contains non-finite prediction")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"legacy B2.1 output already exists: {output}")

    api = KaggleApi()
    api.authenticate()
    status = direct_verify(api)
    kaggle_cli = shutil.which("kaggle")
    if not kaggle_cli:
        raise RuntimeError("official Kaggle CLI is not on PATH")

    with tempfile.TemporaryDirectory(prefix="cmi-flu-b21-current-v1-") as tmp:
        download = Path(tmp) / "download"
        download.mkdir()
        completed = subprocess.run(
            [kaggle_cli, "kernels", "output", KERNEL, "-p", str(download)],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            raise RuntimeError(f"legacy B2.1 kaggle kernels output failed rc={completed.returncode}")

        files = [path for path in download.rglob("*") if path.is_file()]
        total_bytes = sum(path.stat().st_size for path in files)
        if total_bytes <= 0 or total_bytes > MAX_TOTAL_DOWNLOAD_BYTES:
            raise RuntimeError(f"legacy B2.1 download byte contract violated: bytes={total_bytes}")

        submissions = [path for path in files if path.name == "submission.csv"]
        manifests = [path for path in files if path.name == "bridge-result.json"]
        if len(submissions) != 1 or len(manifests) != 1:
            raise RuntimeError(
                "legacy B2.1 required saved outputs missing/ambiguous: "
                f"submission_count={len(submissions)} manifest_count={len(manifests)}"
            )
        submission = submissions[0]
        manifest_path = manifests[0]
        if manifest_path.stat().st_size <= 0 or manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise RuntimeError("legacy B2.1 bridge manifest byte contract violated")
        submission_bytes = submission.read_bytes()
        validate_submission_bytes(submission_bytes)
        digest = sha256_bytes(submission_bytes)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("request_id") != EXPECTED_REQUEST_ID:
            raise RuntimeError("legacy B2.1 bridge request identity mismatch")
        if manifest.get("kernel_stage") != EXPECTED_STAGE:
            raise RuntimeError("legacy B2.1 bridge stage mismatch")
        if manifest.get("source_commit") != EXPECTED_SOURCE_COMMIT:
            raise RuntimeError("legacy B2.1 bridge science provenance mismatch")
        if manifest.get("runtime_adapter_blob_sha") != EXPECTED_ADAPTER_BLOB:
            raise RuntimeError("legacy B2.1 bridge adapter provenance mismatch")
        if int(manifest.get("submission_rows", 0) or 0) != 40:
            raise RuntimeError("legacy B2.1 bridge submission-row contract mismatch")
        if manifest.get("submission_sha256") != digest:
            raise RuntimeError("legacy B2.1 bridge submission SHA-256 mismatch")

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(submission_bytes)
        print(
            "CMI_FLU_B21_LEGACY_CURRENT_SUBMISSION_READ PASS "
            f"status={status} version={EXPECTED_VERSION} rows=40 bytes={len(submission_bytes)} "
            f"sha256={digest} download_bytes={total_bytes}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
