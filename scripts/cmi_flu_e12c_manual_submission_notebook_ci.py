#!/usr/bin/env python3
"""Credential-free regression for the E12c manual-submission Kaggle Notebook."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

E12C_BLOB = "a495584a7e461478bd1a41d01d2536c4435a8f7f"
PREPARE = "scripts/cmi_flu_e12c_manual_submission_notebook_prepare.py"
PAYLOAD = "payloads/cmi-flu-e12c-manual-submission-notebook-001/strategy_e12c.py"
REQUEST = "requests/cmi-flu-e12c-manual-submission-notebook-001.json"
FINAL_SHA = "983aaf097d04477c4ccf7bf817fdf66e552937e69bceaa48cfb260cb84413f1b"
HISTORICAL_SHA = "365607d59cd530656b929a1c1c57412cc6d375265a8d1ba10d304c64e012f387"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    work = args.workdir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    payload = (root / PAYLOAD).read_bytes()
    if git_blob(payload) != E12C_BLOB:
        raise SystemExit("E12c CI exact science relay mismatch")
    compile(payload.decode("utf-8"), "cmi_flu/strategy_e12c.py", "exec")

    runtime = work / "runtime.py"
    subprocess.run(
        [
            sys.executable,
            str(root / PREPARE),
            "--repository-root",
            str(root),
            "--output",
            str(runtime),
        ],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "py_compile", str(runtime)], check=True)
    text = runtime.read_text(encoding="utf-8")
    required = (
        'REQUEST_ID = "20260911-cmi-flu-e12c-manual-submission-notebook-001"',
        'TARGET_KERNEL = "renta0426/cmi-flu-e12c-manual-submission-20260911-001"',
        'SCIENCE_COMMIT = "bb6f41ed81b0bbd4ecdc397c6abb9df671c6ebf8"',
        f'FINAL_CSV_SHA256 = "{FINAL_SHA}"',
        f'HISTORICAL_TASK12_ONLY_SHA256 = "{HISTORICAL_SHA}"',
        'submission_path = output_dir / "submission.csv"',
        'manifest_path = output_dir / "manual-submission-manifest.json"',
        'Path("/tmp") / "cmi-flu-e12c-manual-runtime"',
        'Path("/kaggle/input")',
        'changed_tasks_vs_historical_task12_only',
        '"Task1.3"',
        'CMI_FLU_E12C_MANUAL_SUBMISSION_READY',
        'submission_path.unlink(missing_ok=True)',
        'manifest_path.unlink(missing_ok=True)',
    )
    for token in required:
        if token not in text:
            raise SystemExit(f"E12c CI runtime marker missing:{token}")
    lowered = text.casefold()
    forbidden = (
        "competition_submit(",
        "kaggle competitions submit",
        "quota_view(",
        "max_active_runs",
        "cpu admission",
        "gpu admission",
        "tpu admission",
    )
    for token in forbidden:
        if token in lowered:
            raise SystemExit(f"E12c CI forbidden runtime capability:{token}")

    request_text = (root / REQUEST).read_text(encoding="utf-8")
    if '"execution_policy": "kaggle_native_capacity_v2"' not in request_text:
        raise SystemExit("E12c CI execution policy v2 marker missing")
    if '"competition_submission_authorized": false' not in request_text:
        raise SystemExit("E12c CI submission authorization boundary missing")

    print(
        "CMI_FLU_E12C_MANUAL_CI PASS "
        f"runtime_bytes={runtime.stat().st_size} science_blob={E12C_BLOB} "
        f"submission_sha256={FINAL_SHA} auth=false write=false compute=false submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
