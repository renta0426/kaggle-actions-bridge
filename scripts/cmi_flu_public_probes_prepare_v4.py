#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import py_compile
import subprocess
import sys
from typing import Any

REQUEST_ID = "20260906-cmi-flu-public-probes-004"
PARENT_REQUEST_ID = "20260906-cmi-flu-public-probes-003"
FAILED_WORKFLOW_RUN_ID = 33999234203
TARGET_KERNEL = "renta0426/cmi-flu-public-probes-20260906-004"
TARGET_STAGE = "controlled_public_probes_regenerated_frozen_b21"
OUTPUT_DIR = "/kaggle/working/cmi-flu-public-probes-004"
SCIENCE_COMMIT = "7a85abcbf8282bc8bfe047b7db02a72f35222caa"
SCIENCE_BLOB = "497a8875d440e1b8a03d4ef938bc3a834213b33d"
B21_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
B21_PACKAGE_SHA256 = "48ff4ed2eadec8b059b8d37fb677af1f249b6486bccfbd55dca2274cfc6f3dc3"
PROBE_NAMES = ("task13_only", "task12_only", "task12_task13")


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def validate_request(root: pathlib.Path) -> dict[str, Any]:
    request = json.loads(
        (root / "requests/cmi-flu-public-probes-004.json").read_text(encoding="utf-8")
    )
    expected = {
        "schema_version": 4,
        "request_id": REQUEST_ID,
        "parent_request_id": PARENT_REQUEST_ID,
        "failed_workflow_run_id": FAILED_WORKFLOW_RUN_ID,
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_repository": "renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit": SCIENCE_COMMIT,
        "science_source_path": "src/cmi_flu/public_probes.py",
        "science_transport": "agent_relay_exact_blob",
        "public_probes_blob_sha": SCIENCE_BLOB,
        "relayed_science_path": "payloads/cmi-flu-public-probes-001/public_probes.py",
        "backbone_mode": "regenerate_frozen_b21_in_same_notebook",
        "b21_base_request_id": "20260903-cmi-flu-b21-001",
        "b21_source_commit": B21_SOURCE_COMMIT,
        "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB,
        "b21_base_package_zip_sha256": B21_PACKAGE_SHA256,
        "b21_verify_competition_md5": True,
        "expected_kernel_version": 1,
        "probe_family": list(PROBE_NAMES),
        "competition_submission_attempted": False,
        "public_scores_used_to_define_family": False,
        "automatic_compute_retries": 0,
        "enable_internet": False,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"Public-probe request 004 mismatch: {key}")
    expected_paths = [
        "cmi-flu-public-probes-004/probe-task13.csv",
        "cmi-flu-public-probes-004/probe-task12.csv",
        "cmi-flu-public-probes-004/probe-task12-task13.csv",
        "cmi-flu-public-probes-004/bridge-result.json",
        "cmi-flu-public-probes-004/summary.md",
    ]
    if request.get("allowed_output_paths") != expected_paths:
        raise SystemExit("Public-probe request 004 output allowlist mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 150,
        "hard_timeout_minutes": 300,
        "max_active_runs": 1,
    }:
        raise SystemExit("Public-probe request 004 resource contract mismatch")
    if request.get("api_budget") != {
        "max_calls": 36,
        "initial_poll_schedule_seconds": [30, 60, 120, 300],
        "steady_poll_interval_seconds": 600,
        "max_pages": 2,
    }:
        raise SystemExit("Public-probe request 004 API budget mismatch")
    repair = request.get("repair", {})
    expected_repair = {
        "failed_stage": "materialize_frozen_b21_package",
        "scientific_contract_changed": False,
        "generator_science_blob_changed": False,
        "probe_family_changed": False,
        "backbone_contract_changed": False,
        "resource_contract_changed": False,
    }
    for key, value in expected_repair.items():
        if repair.get(key) != value:
            raise SystemExit(f"Public-probe request 004 repair mismatch: {key}")
    root_cause = str(repair.get("established_root_cause") or "")
    mechanism = str(repair.get("mechanism_change") or "")
    if "sha256_bytes(bundle)" not in root_cause:
        raise SystemExit("request 004 root cause must identify the missing runtime symbol")
    if "hashlib.sha256(bundle).hexdigest()" not in mechanism:
        raise SystemExit("request 004 mechanism must name the direct runtime hash expression")
    return request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    validate_request(root)

    v3_output = output.parent / "public-probes-v3-runtime.py"
    run(
        sys.executable,
        str(root / "scripts/cmi_flu_public_probes_prepare_v3.py"),
        "--repository-root",
        str(root),
        "--output",
        str(v3_output),
    )
    text = v3_output.read_text(encoding="utf-8")

    text = replace_once(
        text,
        'REQUEST_ID = "20260906-cmi-flu-public-probes-003"',
        f'REQUEST_ID = "{REQUEST_ID}"',
        label="request id",
    )
    text = replace_once(
        text,
        'default=Path("/kaggle/working/cmi-flu-public-probes-003")',
        f'default=Path("{OUTPUT_DIR}")',
        label="output directory",
    )
    text = replace_once(
        text,
        'if sha256_bytes(bundle) != PACKAGE_ZIP_SHA256:',
        'if hashlib.sha256(bundle).hexdigest() != PACKAGE_ZIP_SHA256:',
        label="frozen-package SHA-256 runtime boundary",
    )
    text = replace_once(
        text,
        "# Controlled Public probe family 003\\n\\n",
        "# Controlled Public probe family 004\\n\\n",
        label="summary identity",
    )
    text = replace_once(
        text,
        "CMI_FLU_PUBLIC_PROBES_003_COMPLETE ",
        "CMI_FLU_PUBLIC_PROBES_004_COMPLETE ",
        label="completion marker",
    )

    if "sha256_bytes(" in text:
        raise SystemExit("request 004 generated runtime still references bridge-side sha256_bytes")
    direct_hash_line = "if hashlib.sha256(bundle).hexdigest() != PACKAGE_ZIP_SHA256:"
    if text.count(direct_hash_line) != 1:
        raise SystemExit("request 004 direct frozen-package hash expression mismatch")
    if "__CMI_FLU_BACKBONE_" in text:
        raise SystemExit("request 004 runtime retains a legacy backbone placeholder")
    forbidden = (
        "kernels_status(",
        "kernels_list(",
        "kaggle kernels output",
        "competition_submit",
        "kaggle competitions submit",
    )
    present = [token for token in forbidden if token in text]
    if present:
        raise SystemExit(f"request 004 runtime contains forbidden remote-read/submit tokens: {present}")
    for token in (
        'B21_SOURCE_COMMIT_FOR_REGEN = "33030746bc7bad02ad2c1e670ac319cc943c524d"',
        '"backbone_mode": "regenerate_frozen_b21_in_same_notebook"',
        'verify_md5=True',
        '"Task2.1": "et_subtype_d3_l5"',
        '"Task2.2": "et_subtype_d3_l5"',
        '"Task2.3": "pls_exact_5"',
        '"Task1.4") != "raw_pre_vacc_conserved_anchor"',
        'len(verified) != 28',
        'set(skipped or []) != {"md5sum"}',
    ):
        if token not in text:
            raise SystemExit(f"request 004 regenerated-B2.1 contract missing: {token}")

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)

    namespace: dict[str, Any] = {"__name__": "public_probes_v4_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("request 004 generated runtime identity mismatch")
    if namespace.get("TARGET_STAGE") != TARGET_STAGE:
        raise SystemExit("request 004 generated runtime stage mismatch")
    if namespace.get("PUBLIC_PROBES_SOURCE_BLOB_SHA") != SCIENCE_BLOB:
        raise SystemExit("request 004 generated runtime science blob mismatch")
    if namespace.get("PACKAGE_ZIP_SHA256") != B21_PACKAGE_SHA256:
        raise SystemExit("request 004 frozen B2.1 package hash mismatch")
    if namespace.get("B21_ADAPTER_BLOB_SHA") != B21_ADAPTER_BLOB:
        raise SystemExit("request 004 frozen B2.1 adapter blob mismatch")
    if namespace.get("B21_SOURCE_COMMIT_FOR_REGEN") != B21_SOURCE_COMMIT:
        raise SystemExit("request 004 frozen B2.1 source provenance mismatch")

    package_bytes = namespace.get("package_bytes")
    hashlib_module = namespace.get("hashlib")
    if not callable(package_bytes) or not callable(getattr(hashlib_module, "sha256", None)):
        raise SystemExit("request 004 generated runtime lacks package/hashlib boundary")
    bundle = package_bytes()
    digest = hashlib_module.sha256(bundle).hexdigest()
    if digest != B21_PACKAGE_SHA256:
        raise SystemExit(f"request 004 credential-free frozen-package hash mismatch: {digest}")

    run(sys.executable, str(output), "--self-test")
    print(
        "CMI_FLU_PUBLIC_PROBES_PREPARE_V4 PASS "
        f"request_id={REQUEST_ID} target={TARGET_KERNEL} "
        f"science_blob={SCIENCE_BLOB} b21_package={B21_PACKAGE_SHA256} "
        f"b21_adapter_blob={B21_ADAPTER_BLOB} backbone_mode=regenerated "
        f"runtime_hash_boundary={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
