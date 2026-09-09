#!/usr/bin/env python3
"""Recover the existing E08 current-version output without launching Kaggle compute."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

REQUEST_ID = "20260909-cmi-flu-strategy-e08-result-recovery-001"
TARGET = "renta0426/cmi-flu-e08-task13-applicability-20260909-001"
EXPECTED_VERSION = 1
EXPECTED_HASHES = {
    "bridge-result.json": "ef532b131033cd49fa263402af319215ac7ccb4a4e91ae549cc19884de045eea",
    "metrics.json": "8f0ac5adf9534fdfa82a76e902365eefdb33f238a75f50c28b963da214053212",
    "summary.md": "187ff62489c6f266aeff5b11bb74e8e9b4d4212e525252b9e70789f59dbebd4f",
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def kernel_meta(api: KaggleApi):
    owner, slug = TARGET.split("/", 1)
    with api.build_kaggle_client() as client:
        q = ApiGetKernelRequest()
        q.user_name = owner
        q.kernel_slug = slug
        return client.kernels.kernels_api_client.get_kernel(q).metadata


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    a = args()
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_"):
        raise SystemExit("KAGGLE_API_TOKEN contract failed")
    api = KaggleApi()
    api.authenticate()
    meta = kernel_meta(api)
    if str(getattr(meta, "ref", "")) != TARGET:
        raise SystemExit("E08 recovery target mismatch")
    if not bool(getattr(meta, "is_private", False)):
        raise SystemExit("E08 recovery target is not private")
    if int(getattr(meta, "current_version_number", 0) or 0) != EXPECTED_VERSION:
        raise SystemExit("E08 recovery current version changed")
    status = str(getattr(api.kernels_status(TARGET), "status", "")).upper()
    if "COMPLETE" not in status:
        raise SystemExit(f"E08 recovery target not complete:{status}")

    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reader = a.bridge_root.resolve() / "scripts" / "kaggle_current_output_read.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(reader),
            "--kernel",
            TARGET,
            "--expected-version",
            str(EXPECTED_VERSION),
            "--allow-file",
            "bridge-result.json:1048576",
            "--allow-file",
            "metrics.json:16777216",
            "--allow-file",
            "summary.md:1048576",
            "--output-dir",
            str(out),
        ],
        check=False,
        timeout=240,
    )
    if completed.returncode != 0:
        raise SystemExit("E08 current output read failed")
    found = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    if found != EXPECTED_HASHES:
        raise SystemExit("E08 recovered output hashes changed")
    if int(getattr(kernel_meta(api), "current_version_number", 0) or 0) != EXPECTED_VERSION:
        raise SystemExit("E08 recovery current version changed after read")
    print(
        "CMI_FLU_E08_RECOVERY PASS "
        + json.dumps(
            {
                "request_id": REQUEST_ID,
                "target_sha256": hashlib.sha256(TARGET.encode()).hexdigest(),
                "expected_version": EXPECTED_VERSION,
                "output_hashes": EXPECTED_HASHES,
                "compute_started": False,
                "write_attempted": False,
                "competition_submission_attempted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
