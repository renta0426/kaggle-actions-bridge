#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import py_compile
import subprocess
import sys
from typing import Any

REQUEST_ID = "20260906-cmi-flu-public-probes-002"
PARENT_REQUEST_ID = "20260906-cmi-flu-public-probes-001"
FAILED_WORKFLOW_RUN_ID = 33979598196
SCIENCE_COMMIT = "7a85abcbf8282bc8bfe047b7db02a72f35222caa"
SCIENCE_BLOB = "497a8875d440e1b8a03d4ef938bc3a834213b33d"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
BACKBONE_KERNEL = "renta0426/cmi-flu-b21-robust-cv-20260903-001"
BACKBONE_VERSION = 1
BACKBONE_READER = "scripts/cmi_flu_b21_legacy_current_submission_read.py"
BACKBONE_READER_BLOB = "8da7916ed71c280d775e54d96f3f156a56bab629"
TARGET_KERNEL = "renta0426/cmi-flu-public-probes-20260906-002"
OLD_OUTPUT = "/kaggle/working/cmi-flu-public-probes-001"
NEW_OUTPUT = "/kaggle/working/cmi-flu-public-probes-002"
PROBE_NAMES = ["task13_only", "task12_only", "task12_task13"]


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def validate_request(root: pathlib.Path) -> None:
    request = json.loads(
        (root / "requests/cmi-flu-public-probes-002.json").read_text(encoding="utf-8")
    )
    expected = {
        "schema_version": 2,
        "request_id": REQUEST_ID,
        "parent_request_id": PARENT_REQUEST_ID,
        "failed_workflow_run_id": FAILED_WORKFLOW_RUN_ID,
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": "agent_relay_exact_blob",
        "public_probes_blob_sha": SCIENCE_BLOB,
        "backbone_kernel": BACKBONE_KERNEL,
        "backbone_expected_version": BACKBONE_VERSION,
        "backbone_read_mode": "legacy_exact_current_v1_direct_metadata_manifest_verified",
        "backbone_reader_path": BACKBONE_READER,
        "backbone_reader_blob_sha": BACKBONE_READER_BLOB,
        "b21_base_request_id": "20260903-cmi-flu-b21-001",
        "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB,
        "expected_kernel_version": 1,
        "probe_family": PROBE_NAMES,
        "competition_submission_attempted": False,
        "public_scores_used_to_define_family": False,
        "automatic_compute_retries": 0,
        "enable_internet": False,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"controlled-probe repair request mismatch: {key}")
    repair = request.get("repair") or {}
    if repair != {
        "failed_stage": "read_b21_current_v1_backbone",
        "established_root_cause": "generic current-output helper requires kernels_list search discoverability and new output hygiene that the frozen legacy B2.1 Notebook does not satisfy",
        "scientific_contract_changed": False,
        "generator_science_blob_changed": False,
        "probe_family_changed": False,
    }:
        raise SystemExit("controlled-probe repair rationale mismatch")
    if request.get("resource") != {
        "accelerator": "cpu", "expected_runtime_minutes": 30,
        "hard_timeout_minutes": 90, "max_active_runs": 1,
    }:
        raise SystemExit("controlled-probe repair resource contract mismatch")
    reader = (root / BACKBONE_READER).read_bytes()
    if git_blob_sha(reader) != BACKBONE_READER_BLOB:
        raise SystemExit("legacy B2.1 reader blob mismatch")
    source = reader.decode("utf-8")
    compile(source, BACKBONE_READER, "exec")
    required_reader = (
        f'KERNEL = "{BACKBONE_KERNEL}"',
        "EXPECTED_VERSION = 1",
        'EXPECTED_REQUEST_ID = "20260903-cmi-flu-b21-001"',
        'EXPECTED_STAGE = "b21_taskwise_robust"',
        'EXPECTED_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"',
        f'EXPECTED_ADAPTER_BLOB = "{B21_ADAPTER_BLOB}"',
        "get_kernel(request).metadata",
        '"submission_sha256"',
    )
    missing = [token for token in required_reader if token not in source]
    if missing:
        raise SystemExit(f"legacy B2.1 reader contract tokens missing: {missing}")
    if "kernels_list(" in source:
        raise SystemExit("legacy B2.1 reader must not depend on list-search discoverability")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    validate_request(root)
    base = output.parent / "public-probes-v1-template.py"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/cmi_flu_public_probes_prepare.py"),
            "--repository-root", str(root),
            "--output", str(base),
        ],
        check=True,
    )
    text = base.read_text(encoding="utf-8")
    text = replace_once(
        text,
        f'REQUEST_ID = "{PARENT_REQUEST_ID}"',
        f'REQUEST_ID = "{REQUEST_ID}"',
        label="repair request id",
    )
    text = replace_once(text, OLD_OUTPUT, NEW_OUTPUT, label="repair output directory")
    if PARENT_REQUEST_ID in text or OLD_OUTPUT in text:
        raise SystemExit("controlled-probe repair runtime retains parent identity")
    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'SOURCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'PUBLIC_PROBES_SOURCE_BLOB_SHA = "{SCIENCE_BLOB}"',
        '__CMI_FLU_BACKBONE_B64__',
        '__CMI_FLU_BACKBONE_SHA256__',
        '"task13_only"', '"task12_only"', '"task12_task13"',
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"controlled-probe repair runtime missing tokens: {missing}")
    if "competition_submit" in text or "kaggle competitions submit" in text:
        raise SystemExit("controlled-probe repair runtime contains submission path")

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)
    namespace: dict[str, Any] = {"__name__": "public_probes_v2_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("controlled-probe repair runtime request mismatch")
    if namespace.get("PUBLIC_PROBES_SOURCE_BLOB_SHA") != SCIENCE_BLOB:
        raise SystemExit("controlled-probe repair runtime science blob mismatch")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_PUBLIC_PROBES_PREPARE_V2 PASS "
        f"request_id={REQUEST_ID} science_blob={SCIENCE_BLOB} "
        f"backbone_reader_blob={BACKBONE_READER_BLOB} repair=legacy_current_v1_reader_only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
