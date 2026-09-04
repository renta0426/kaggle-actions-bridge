#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import py_compile
import subprocess
import sys
from typing import Any

REQUEST_ID = "20260904-cmi-flu-hai-transfer-004"
PARENT_REQUEST_ID = "20260904-cmi-flu-hai-transfer-003"
FAILED_WORKFLOW_RUN_ID = 33876159784
FAILED_STAGE = "run_hai_transfer"
FAILED_ERROR_CODE = "74bf25be40ca17429601"
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_COMMIT = "d1ebf13dc0dc7e5d5a2798b29c288265cbf56618"
SCIENCE_PATH = "src/cmi_flu/hai_transfer.py"
SCIENCE_BLOB = "b671d8bf7f10bebbd65aca2a5bad42e267ee78d5"
B21_BASE_REQUEST_ID = "20260903-cmi-flu-b21-001"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
COMPAT_BLOB = "e2973831077ed8893ab1673c9633eb42df7b26de"
SEQUENCE_SHA256 = "63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887"
VACCINE_SHA256 = "8f6c7116f37f29df0bb21d6049d82fa28b4e42b2d10ed9394a1ae6f926bd9f35"
RAW_SEQUENCE_COLUMNS = ("Virus", "Sequence", "Status_of_sequence")
CANONICAL_SEQUENCE_COLUMNS = ("virus_strain", "sequence", "sequence_status")
SEQUENCE_ROWS = 352
TARGET_COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-hai-strain-transfer-20260904-004"
EXPECTED_KERNEL_VERSION = 1
ALLOWED_OUTPUT_PATHS = [
    "cmi-flu-hai-transfer-004/bridge-result.json",
    "cmi-flu-hai-transfer-004/metrics.json",
    "cmi-flu-hai-transfer-004/summary.md",
]


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def validate_sequence_reference(path: pathlib.Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise SystemExit("locked strain sequence reference is empty") from error
        rows = list(reader)
    if header != RAW_SEQUENCE_COLUMNS:
        raise SystemExit(f"locked strain sequence raw header mismatch: {header!r}")
    if len(rows) != SEQUENCE_ROWS:
        raise SystemExit(f"locked strain sequence row count mismatch: {len(rows)}")
    widths = {len(row) for row in rows}
    if widths != {len(RAW_SEQUENCE_COLUMNS)}:
        raise SystemExit(f"locked strain sequence row-width mismatch: {sorted(widths)}")
    for index, name in enumerate(RAW_SEQUENCE_COLUMNS):
        if any(not row[index].strip() for row in rows):
            raise SystemExit(f"locked strain sequence column contains empty values: {name}")


def validate_request(root: pathlib.Path) -> dict[str, Any]:
    request = json.loads(
        (root / "requests/cmi-flu-hai-transfer-004.json").read_text(encoding="utf-8")
    )
    expected_scalar = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "parent_request_id": PARENT_REQUEST_ID,
        "failed_workflow_run_id": FAILED_WORKFLOW_RUN_ID,
        "failed_stage": FAILED_STAGE,
        "failed_error_code": FAILED_ERROR_CODE,
        "competition": TARGET_COMPETITION,
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_repository": SCIENCE_REPOSITORY,
        "science_source_commit": SCIENCE_COMMIT,
        "science_source_path": SCIENCE_PATH,
        "hai_transfer_blob_sha": SCIENCE_BLOB,
        "b21_base_request_id": B21_BASE_REQUEST_ID,
        "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB,
        "hai_b21_compatibility_blob_sha": COMPAT_BLOB,
        "sequence_reference_sha256": SEQUENCE_SHA256,
        "vaccine_reference_sha256": VACCINE_SHA256,
        "expected_kernel_version": EXPECTED_KERNEL_VERSION,
        "competition_submission_attempted": False,
        "automatic_compute_retries": 0,
        "enable_internet": False,
    }
    for key, expected in expected_scalar.items():
        if request.get(key) != expected:
            raise SystemExit(f"HAI repair 004 manifest mismatch: {key}")
    if tuple(request.get("sequence_reference_raw_columns") or ()) != RAW_SEQUENCE_COLUMNS:
        raise SystemExit("HAI repair 004 raw sequence schema mismatch")
    if tuple(request.get("sequence_reference_canonical_columns") or ()) != CANONICAL_SEQUENCE_COLUMNS:
        raise SystemExit("HAI repair 004 canonical sequence schema mismatch")
    if request.get("allowed_output_paths") != ALLOWED_OUTPUT_PATHS:
        raise SystemExit("HAI repair 004 output allowlist mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 120,
        "hard_timeout_minutes": 210,
        "max_active_runs": 1,
    }:
        raise SystemExit("HAI repair 004 resource contract mismatch")
    if request.get("api_budget") != {
        "max_calls": 24,
        "initial_poll_schedule_seconds": [30, 60, 120, 300],
        "steady_poll_interval_seconds": 900,
        "max_pages": 2,
    }:
        raise SystemExit("HAI repair 004 API budget mismatch")
    return request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=pathlib.Path, required=True)
    parser.add_argument("--sequence-reference", type=pathlib.Path, required=True)
    parser.add_argument("--vaccine-reference", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    sequence_reference = args.sequence_reference.expanduser().resolve()
    vaccine_reference = args.vaccine_reference.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    validate_request(root)
    validate_sequence_reference(sequence_reference)

    old_output = output.parent / "hai-transfer-003-base.py"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/cmi_flu_hai_transfer_prepare_v3.py"),
            "--repository-root",
            str(root),
            "--sequence-reference",
            str(sequence_reference),
            "--vaccine-reference",
            str(vaccine_reference),
            "--output",
            str(old_output),
        ],
        check=True,
    )
    text = old_output.read_text(encoding="utf-8")

    # Repair 004 changes only request/output identity plus the exact raw->canonical
    # organizer sequence-reference column mapping. Science/model/CV logic stays exact.
    text = replace_once(
        text,
        f'REQUEST_ID = "{PARENT_REQUEST_ID}"',
        f'REQUEST_ID = "{REQUEST_ID}"',
        label="request id overlay",
    )
    text = replace_once(
        text,
        'default=Path("/kaggle/working/cmi-flu-hai-transfer-003")',
        'default=Path("/kaggle/working/cmi-flu-hai-transfer-004")',
        label="output directory overlay",
    )

    old_reference_load = '''        sequence_reference = pd.read_csv(sequence_path)\n        vaccine_reference = load_vaccine_reference(vaccine_path)\n'''
    new_reference_load = '''        sequence_reference = pd.read_csv(sequence_path)\n        expected_raw_sequence_columns = ["Virus", "Sequence", "Status_of_sequence"]\n        if list(sequence_reference.columns) != expected_raw_sequence_columns:\n            raise BundleContractError(\n                f"locked strain sequence raw schema mismatch: {list(sequence_reference.columns)}"\n            )\n        sequence_reference = sequence_reference.rename(\n            columns={\n                "Virus": "virus_strain",\n                "Sequence": "sequence",\n                "Status_of_sequence": "sequence_status",\n            }\n        )\n        if list(sequence_reference.columns) != [\n            "virus_strain", "sequence", "sequence_status"\n        ]:\n            raise BundleContractError("canonical strain sequence schema normalization failed")\n        vaccine_reference = load_vaccine_reference(vaccine_path)\n'''
    text = replace_once(
        text,
        old_reference_load,
        new_reference_load,
        label="organizer sequence schema normalization",
    )

    old_manifest = '''            "sequence_reference_sha256": SEQUENCE_REFERENCE_SHA256,\n            "vaccine_reference_sha256": VACCINE_REFERENCE_SHA256,\n'''
    new_manifest = '''            "sequence_reference_sha256": SEQUENCE_REFERENCE_SHA256,\n            "sequence_reference_raw_columns": ["Virus", "Sequence", "Status_of_sequence"],\n            "sequence_reference_canonical_columns": ["virus_strain", "sequence", "sequence_status"],\n            "vaccine_reference_sha256": VACCINE_REFERENCE_SHA256,\n'''
    text = replace_once(
        text,
        old_manifest,
        new_manifest,
        label="sequence schema provenance manifest",
    )

    if PARENT_REQUEST_ID in text:
        raise SystemExit("request 003 identity remained in repair 004 runtime")
    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'SOURCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'HAI_TRANSFER_SOURCE_BLOB_SHA = "{SCIENCE_BLOB}"',
        f'HAI_B21_COMPATIBILITY_SOURCE_BLOB_SHA = "{COMPAT_BLOB}"',
        f'B21_ADAPTER_BLOB_SHA = "{B21_ADAPTER_BLOB}"',
        'stage = "install_hai_b21_compatibility"',
        "install_hai_compat()",
        'default=Path("/kaggle/working/cmi-flu-hai-transfer-004")',
        'expected_raw_sequence_columns = ["Virus", "Sequence", "Status_of_sequence"]',
        '"Virus": "virus_strain"',
        '"Sequence": "sequence"',
        '"Status_of_sequence": "sequence_status"',
        '"sequence_reference_raw_columns": ["Virus", "Sequence", "Status_of_sequence"]',
        '"sequence_reference_canonical_columns": ["virus_strain", "sequence", "sequence_status"]',
        "selection_policy=selection_policy",
        "CMI_FLU_HAI_TRANSFER_COMPLETE",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"HAI repair 004 runtime missing tokens: {missing}")
    forbidden = ("kaggle competitions submit", "api.competition_submit")
    present = [token for token in forbidden if token in text]
    if present:
        raise SystemExit(f"HAI repair 004 runtime contains submission tokens: {present}")

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)
    namespace: dict[str, Any] = {"__name__": "hai_transfer_004_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("generated repair 004 runtime request mismatch")
    if namespace.get("SOURCE_COMMIT") != SCIENCE_COMMIT:
        raise SystemExit("generated repair 004 runtime science commit mismatch")
    if namespace.get("HAI_TRANSFER_SOURCE_BLOB_SHA") != SCIENCE_BLOB:
        raise SystemExit("generated repair 004 runtime science blob mismatch")
    if namespace.get("HAI_B21_COMPATIBILITY_SOURCE_BLOB_SHA") != COMPAT_BLOB:
        raise SystemExit("generated repair 004 runtime compatibility blob mismatch")

    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_HAI_TRANSFER_004_PREPARE PASS "
        f"science_commit={SCIENCE_COMMIT} science_blob={SCIENCE_BLOB} "
        f"compat_blob={COMPAT_BLOB} b21_adapter_blob={B21_ADAPTER_BLOB} "
        f"raw_sequence_columns={','.join(RAW_SEQUENCE_COLUMNS)} "
        f"target_kernel={TARGET_KERNEL} expected_version=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
