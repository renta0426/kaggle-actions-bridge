#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import py_compile
import subprocess
import sys
from typing import Any

REQUEST_ID = "20260906-cmi-flu-public-probes-005"
PARENT_REQUEST_ID = "20260906-cmi-flu-public-probes-004"
FAILED_WORKFLOW_RUN_ID = 34001421161
FAILED_ERROR_CODE = "2daf3d1f7f807ac8f8b4"
TARGET_KERNEL = "renta0426/cmi-flu-public-probes-20260906-005"
TARGET_STAGE = "controlled_public_probes_regenerated_frozen_b21"
OUTPUT_DIR = "/kaggle/working/cmi-flu-public-probes-005"
SCIENCE_COMMIT = "7a85abcbf8282bc8bfe047b7db02a72f35222caa"
SCIENCE_BLOB = "497a8875d440e1b8a03d4ef938bc3a834213b33d"
B21_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
B21_PACKAGE_SHA256 = "48ff4ed2eadec8b059b8d37fb677af1f249b6486bccfbd55dca2274cfc6f3dc3"
B21_ORIGINAL_SUBMISSION_SHA256 = "46f187ba85957ef1815f8b89d6f7aec53fa0b935d37225f05d140e309105dd38"
B21_ORIGINAL_METRICS_SHA256 = "64ec0a5a4f53189b32135f53ada6608e6e9be24e08a63674a475226099881b86"
B21_REPORT_PATH = "reports/b2-1-results.md"
EXPECTED_SELECTED_MODELS = {
    "Task1.1": "pls_2",
    "Task1.2": "enet_a0.001_l0.5",
    "Task1.3": "pls_1",
    "Task1.4": "raw_pre_vacc_conserved_anchor",
    "Task2.1": "et_subtype_d3_l5",
    "Task2.2": "et_subtype_d5_l10",
    "Task2.3": "ridge_exact_a100",
}
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
        (root / "requests/cmi-flu-public-probes-005.json").read_text(encoding="utf-8")
    )
    expected = {
        "schema_version": 5,
        "request_id": REQUEST_ID,
        "parent_request_id": PARENT_REQUEST_ID,
        "failed_workflow_run_id": FAILED_WORKFLOW_RUN_ID,
        "failed_error_code": FAILED_ERROR_CODE,
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
        "b21_authoritative_report_path": B21_REPORT_PATH,
        "b21_original_submission_sha256": B21_ORIGINAL_SUBMISSION_SHA256,
        "b21_original_metrics_sha256": B21_ORIGINAL_METRICS_SHA256,
        "b21_expected_selected_models": EXPECTED_SELECTED_MODELS,
        "expected_kernel_version": 1,
        "probe_family": list(PROBE_NAMES),
        "competition_submission_attempted": False,
        "public_scores_used_to_define_family": False,
        "automatic_compute_retries": 0,
        "enable_internet": False,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"Public-probe request 005 mismatch: {key}")
    expected_paths = [
        "cmi-flu-public-probes-005/probe-task13.csv",
        "cmi-flu-public-probes-005/probe-task12.csv",
        "cmi-flu-public-probes-005/probe-task12-task13.csv",
        "cmi-flu-public-probes-005/bridge-result.json",
        "cmi-flu-public-probes-005/summary.md",
    ]
    if request.get("allowed_output_paths") != expected_paths:
        raise SystemExit("Public-probe request 005 output allowlist mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 150,
        "hard_timeout_minutes": 300,
        "max_active_runs": 1,
    }:
        raise SystemExit("Public-probe request 005 resource contract mismatch")
    if request.get("api_budget") != {
        "max_calls": 36,
        "initial_poll_schedule_seconds": [30, 60, 120, 300],
        "steady_poll_interval_seconds": 600,
        "max_pages": 2,
    }:
        raise SystemExit("Public-probe request 005 API budget mismatch")
    repair = request.get("repair", {})
    expected_repair = {
        "failed_stage": "regenerate_frozen_b21",
        "scientific_contract_changed": False,
        "generator_science_blob_changed": False,
        "probe_family_changed": False,
        "backbone_scientific_contract_changed": False,
        "provenance_guard_changed": True,
        "resource_contract_changed": False,
    }
    for key, value in expected_repair.items():
        if repair.get(key) != value:
            raise SystemExit(f"Public-probe request 005 repair mismatch: {key}")
    cause = str(repair.get("established_root_cause") or "")
    mechanism = str(repair.get("mechanism_change") or "")
    if "Task2.2=et_subtype_d5_l10" not in cause or "Task2.3=ridge_exact_a100" not in cause:
        raise SystemExit("request 005 root cause must name authoritative B2.1 HAI models")
    if "submission SHA-256" not in mechanism or "seven-task" not in mechanism:
        raise SystemExit("request 005 mechanism must lock full model map and submission hash")
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

    v4_output = output.parent / "public-probes-v4-runtime.py"
    run(
        sys.executable,
        str(root / "scripts/cmi_flu_public_probes_prepare_v4.py"),
        "--repository-root",
        str(root),
        "--output",
        str(v4_output),
    )
    text = v4_output.read_text(encoding="utf-8")

    text = replace_once(
        text,
        'REQUEST_ID = "20260906-cmi-flu-public-probes-004"',
        f'REQUEST_ID = "{REQUEST_ID}"',
        label="request id",
    )
    text = replace_once(
        text,
        'default=Path("/kaggle/working/cmi-flu-public-probes-004")',
        f'default=Path("{OUTPUT_DIR}")',
        label="output directory",
    )
    text = replace_once(
        text,
        "# Controlled Public probe family 004\\n\\n",
        "# Controlled Public probe family 005\\n\\n",
        label="summary identity",
    )
    text = replace_once(
        text,
        "CMI_FLU_PUBLIC_PROBES_004_COMPLETE ",
        "CMI_FLU_PUBLIC_PROBES_005_COMPLETE ",
        label="completion marker",
    )

    provenance_anchor = (
        f'B21_SOURCE_COMMIT_FOR_REGEN = "{B21_SOURCE_COMMIT}"\n'
    )
    provenance_insert = (
        provenance_anchor
        + f'B21_ORIGINAL_SUBMISSION_SHA256 = "{B21_ORIGINAL_SUBMISSION_SHA256}"\n'
        + f'B21_AUTHORITATIVE_REPORT_PATH = "{B21_REPORT_PATH}"\n'
    )
    text = replace_once(
        text,
        provenance_anchor,
        provenance_insert,
        label="authoritative B2.1 provenance insertion",
    )

    old_model_block = '''        expected_hai = {
            "Task2.1": "et_subtype_d3_l5",
            "Task2.2": "et_subtype_d3_l5",
            "Task2.3": "pls_exact_5",
        }
        for task, expected_model in expected_hai.items():
            if selected_models.get(task) != expected_model:
                raise BundleContractError(
                    f"regenerated B2.1 model mismatch for {task}: {selected_models.get(task)}"
                )
        if selected_models.get("Task1.4") != "raw_pre_vacc_conserved_anchor":
            raise BundleContractError("regenerated B2.1 Task1.4 anchor mismatch")
'''
    new_model_block = '''        expected_models = {
            "Task1.1": "pls_2",
            "Task1.2": "enet_a0.001_l0.5",
            "Task1.3": "pls_1",
            "Task1.4": "raw_pre_vacc_conserved_anchor",
            "Task2.1": "et_subtype_d3_l5",
            "Task2.2": "et_subtype_d5_l10",
            "Task2.3": "ridge_exact_a100",
        }
        if selected_models != expected_models:
            mismatches = {
                task: {"expected": expected_models.get(task), "actual": selected_models.get(task)}
                for task in sorted(expected_models)
                if selected_models.get(task) != expected_models.get(task)
            }
            raise BundleContractError(
                f"regenerated B2.1 selected-model map mismatch: {mismatches}"
            )
'''
    text = replace_once(
        text,
        old_model_block,
        new_model_block,
        label="authoritative selected-model map",
    )

    hash_anchor = '''        b21_submission_sha = sha256_file(b21_submission_path)
        b21_metrics_sha = sha256_file(b21_metrics_path)
'''
    hash_insert = '''        b21_submission_sha = sha256_file(b21_submission_path)
        if b21_submission_sha != B21_ORIGINAL_SUBMISSION_SHA256:
            raise BundleContractError(
                "regenerated B2.1 submission SHA-256 does not exactly match the original B2.1 submission"
            )
        b21_metrics_sha = sha256_file(b21_metrics_path)
'''
    text = replace_once(
        text,
        hash_anchor,
        hash_insert,
        label="original B2.1 submission hash gate",
    )

    text = replace_once(
        text,
        '            "schema_version": 3,\n            "request_id": REQUEST_ID,',
        '            "schema_version": 5,\n            "request_id": REQUEST_ID,',
        label="result schema version",
    )
    payload_anchor = '            "regenerated_b21_submission_sha256": b21_submission_sha,\n'
    payload_insert = (
        payload_anchor
        + '            "original_b21_submission_sha256": B21_ORIGINAL_SUBMISSION_SHA256,\n'
        + '            "regenerated_b21_matches_original_submission": True,\n'
        + '            "b21_authoritative_report_path": B21_AUTHORITATIVE_REPORT_PATH,\n'
    )
    text = replace_once(
        text,
        payload_anchor,
        payload_insert,
        label="result B2.1 exact-match provenance",
    )

    if "sha256_bytes(" in text:
        raise SystemExit("request 005 generated runtime regressed to bridge-side sha256_bytes")
    if '"Task2.2": "et_subtype_d3_l5"' in new_model_block:
        raise SystemExit("request 005 selected-model map contains stale Task2.2 model")
    if '"Task2.3": "pls_exact_5"' in new_model_block:
        raise SystemExit("request 005 selected-model map contains stale Task2.3 model")
    for token in (
        '"Task1.1": "pls_2"',
        '"Task1.2": "enet_a0.001_l0.5"',
        '"Task1.3": "pls_1"',
        '"Task1.4": "raw_pre_vacc_conserved_anchor"',
        '"Task2.1": "et_subtype_d3_l5"',
        '"Task2.2": "et_subtype_d5_l10"',
        '"Task2.3": "ridge_exact_a100"',
        f'B21_ORIGINAL_SUBMISSION_SHA256 = "{B21_ORIGINAL_SUBMISSION_SHA256}"',
        'if b21_submission_sha != B21_ORIGINAL_SUBMISSION_SHA256:',
        '"regenerated_b21_matches_original_submission": True',
    ):
        if token not in text:
            raise SystemExit(f"request 005 authoritative B2.1 contract missing: {token}")
    if "__CMI_FLU_BACKBONE_" in text:
        raise SystemExit("request 005 runtime retains a legacy backbone placeholder")
    forbidden = (
        "kernels_status(",
        "kernels_list(",
        "kaggle kernels output",
        "competition_submit",
        "kaggle competitions submit",
    )
    present = [token for token in forbidden if token in text]
    if present:
        raise SystemExit(f"request 005 runtime contains forbidden remote-read/submit tokens: {present}")

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)

    namespace: dict[str, Any] = {"__name__": "public_probes_v5_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("request 005 generated runtime identity mismatch")
    if namespace.get("TARGET_STAGE") != TARGET_STAGE:
        raise SystemExit("request 005 generated runtime stage mismatch")
    if namespace.get("PUBLIC_PROBES_SOURCE_BLOB_SHA") != SCIENCE_BLOB:
        raise SystemExit("request 005 generated runtime science blob mismatch")
    if namespace.get("PACKAGE_ZIP_SHA256") != B21_PACKAGE_SHA256:
        raise SystemExit("request 005 frozen B2.1 package hash mismatch")
    if namespace.get("B21_ADAPTER_BLOB_SHA") != B21_ADAPTER_BLOB:
        raise SystemExit("request 005 frozen B2.1 adapter blob mismatch")
    if namespace.get("B21_SOURCE_COMMIT_FOR_REGEN") != B21_SOURCE_COMMIT:
        raise SystemExit("request 005 frozen B2.1 source provenance mismatch")
    if namespace.get("B21_ORIGINAL_SUBMISSION_SHA256") != B21_ORIGINAL_SUBMISSION_SHA256:
        raise SystemExit("request 005 original B2.1 submission hash provenance mismatch")

    package_bytes = namespace.get("package_bytes")
    hashlib_module = namespace.get("hashlib")
    if not callable(package_bytes) or not callable(getattr(hashlib_module, "sha256", None)):
        raise SystemExit("request 005 generated runtime lacks package/hashlib boundary")
    bundle = package_bytes()
    digest = hashlib_module.sha256(bundle).hexdigest()
    if digest != B21_PACKAGE_SHA256:
        raise SystemExit(f"request 005 credential-free frozen-package hash mismatch: {digest}")

    run(sys.executable, str(output), "--self-test")
    print(
        "CMI_FLU_PUBLIC_PROBES_PREPARE_V5 PASS "
        f"request_id={REQUEST_ID} target={TARGET_KERNEL} "
        f"science_blob={SCIENCE_BLOB} b21_package={B21_PACKAGE_SHA256} "
        f"b21_adapter_blob={B21_ADAPTER_BLOB} backbone_mode=regenerated "
        f"original_b21_submission_sha256={B21_ORIGINAL_SUBMISSION_SHA256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
