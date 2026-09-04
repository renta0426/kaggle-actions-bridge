#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import py_compile
import re
import subprocess
import sys
from typing import Any

REQUEST_ID = "20260904-cmi-flu-phase-a-003"
SCIENCE_COMMIT = "0282262e049683135ec01a56f71b44a46356b194"
PHASE_A_BLOB = "bd9351a2857bc0964309ca117f60c766354066a0"
PHASE_A_PACKAGE_SHA256 = "e656f6cd355d4065c8340007e5493d80d38e6fca0e300cfd2b781096dfa6b1fd"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
BASE_SHA256 = "7894a9232e7372950f70a36d35de41bcc8bda90ec4d0320187e82c3b8c76f2db"
BASE_SIZE = 93363
TARGET_COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-b1-vs-b2-1-20260904-003"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    request_path = root / "requests/cmi-flu-phase-a-launch-v3.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    allowed = {
        "schema_version", "request_id", "competition", "operation", "target", "purpose",
        "science_source_commit", "phase_a_source_blob_sha", "phase_a_package_zip_sha256",
        "b21_base_request_id", "failed_request_id", "failed_request_error", "resource",
        "api_budget", "side_effects", "competition_submission_attempted",
        "automatic_compute_retries", "enable_internet", "rules_checked_at_utc",
    }
    if set(request) != allowed:
        raise SystemExit(f"manifest fields mismatch: {sorted(set(request) ^ allowed)}")
    if request["schema_version"] != 1 or request["request_id"] != REQUEST_ID:
        raise SystemExit("request identity mismatch")
    if request["competition"] != TARGET_COMPETITION or request["target"] != TARGET_KERNEL or request["operation"] != "kernel_run":
        raise SystemExit("target/operation mismatch")
    if request["science_source_commit"] != SCIENCE_COMMIT:
        raise SystemExit("science provenance mismatch")
    if request["phase_a_source_blob_sha"] != PHASE_A_BLOB or request["phase_a_package_zip_sha256"] != PHASE_A_PACKAGE_SHA256:
        raise SystemExit("Phase A source/package provenance mismatch")
    if request["b21_base_request_id"] != "20260903-cmi-flu-b21-001" or request["failed_request_id"] != "20260904-cmi-flu-phase-a-002":
        raise SystemExit("retry lineage mismatch")
    if request["competition_submission_attempted"] is not False or request["automatic_compute_retries"] != 0 or request["enable_internet"] is not False:
        raise SystemExit("safety contract mismatch")
    if request["resource"] != {"accelerator": "cpu", "expected_runtime_minutes": 120, "hard_timeout_minutes": 240, "max_active_runs": 1}:
        raise SystemExit("resource contract mismatch")
    if request["api_budget"] != {"max_calls": 20, "poll_interval_seconds": 900, "max_pages": 2}:
        raise SystemExit("API budget mismatch")

    adapter_path = root / "scripts/cmi_flu_b21_runtime_adapter.py"
    adapter = adapter_path.read_bytes()
    if git_blob_sha(adapter) != B21_ADAPTER_BLOB:
        raise SystemExit("B2.1 adapter blob mismatch")

    phase_dir = root / "payloads/cmi-flu-phase-a-001"
    phase = (
        (phase_dir / "part-00").read_bytes().rstrip(b"\n")
        + b"\n"
        + (phase_dir / "part-01").read_bytes().lstrip(b"\n").rstrip(b"\n")
        + b"\n"
    )
    if git_blob_sha(phase) != PHASE_A_BLOB:
        raise SystemExit("staged Phase A source blob mismatch")

    base_dir = root / "payloads/cmi-flu-b2-broad-004"
    base = b"".join((base_dir / f"part-{index:02d}").read_bytes() for index in range(20))
    if len(base) != BASE_SIZE or hashlib.sha256(base).hexdigest() != BASE_SHA256:
        raise SystemExit("B2 004 base payload mismatch")

    work = output.parent
    base_source = work / "base-source.py"
    base004 = work / "base004.py"
    b21 = work / "b21.py"
    base_source.write_bytes(base)
    run(sys.executable, str(root / "scripts/cmi_flu_b2_patch_v4.py"), "--source", str(base_source), "--request", str(root / "requests/cmi-flu-b2-launch-v4.json"), "--output", str(base004))
    run(sys.executable, str(root / "scripts/cmi_flu_b21_patch.py"), "--source", str(base004), "--adapter", str(adapter_path), "--request", str(root / "requests/cmi-flu-b21-launch-v1.json"), "--output", str(b21))
    run(sys.executable, str(root / "scripts/cmi_flu_phase_a_patch_v3.py"), "--source", str(b21), "--parts-dir", str(phase_dir), "--request", str(request_path), "--output", str(output))

    for path in (
        root / "scripts/cmi_flu_phase_a_patch.py",
        root / "scripts/cmi_flu_phase_a_patch_v2.py",
        root / "scripts/cmi_flu_phase_a_patch_v3.py",
        output,
    ):
        py_compile.compile(str(path), doraise=True)

    text = output.read_text(encoding="utf-8")
    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        "models.summarize_metric_frame = metric_summary",
        "def json_safe(value: Any) -> Any:",
        'TARGET_STAGE = "phase_a_b1_vs_b21_same_contract"',
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"generated runtime missing tokens: {missing}")

    namespace: dict[str, Any] = {"__name__": "phase_a_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    helper = namespace.get("json_safe")
    if not callable(helper):
        raise SystemExit("json_safe missing after module load")
    import numpy as np
    probe = helper({"nan": float("nan"), "integer": np.int64(7), "values": (np.float64(1.5),)})
    if probe != {"nan": None, "integer": 7, "values": [1.5]}:
        raise SystemExit(f"json_safe smoke mismatch: {probe!r}")

    run(sys.executable, str(output), "--self-test")
    print("CMI_FLU_PHASE_A_003_PREPARE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
