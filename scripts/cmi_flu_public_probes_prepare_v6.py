#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import py_compile
import subprocess
import sys
from typing import Any

REQUEST_ID = "20260906-cmi-flu-public-probes-006"
PARENT_REQUEST_ID = "20260906-cmi-flu-public-probes-005"
FAILED_WORKFLOW_RUN_ID = 34008782099
FAILED_ERROR_CODE = "63b150fd446c17af259e"
TARGET_KERNEL = "renta0426/cmi-flu-public-probes-20260906-006"
TARGET_STAGE = "controlled_public_probes_regenerated_frozen_b21"
OUTPUT_DIR = "/kaggle/working/cmi-flu-public-probes-006"
CONTROL_FILENAME = "control-b21-regenerated.csv"
SCIENCE_COMMIT = "7a85abcbf8282bc8bfe047b7db02a72f35222caa"
SCIENCE_BLOB = "497a8875d440e1b8a03d4ef938bc3a834213b33d"
B21_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
B21_PACKAGE_SHA256 = "48ff4ed2eadec8b059b8d37fb677af1f249b6486bccfbd55dca2274cfc6f3dc3"
B21_ORIGINAL_SUBMISSION_SHA256 = "46f187ba85957ef1815f8b89d6f7aec53fa0b935d37225f05d140e309105dd38"
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
        (root / "requests/cmi-flu-public-probes-006.json").read_text(encoding="utf-8")
    )
    expected = {
        "schema_version": 6,
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
        "control_output_name": CONTROL_FILENAME,
        "b21_base_request_id": "20260903-cmi-flu-b21-001",
        "b21_source_commit": B21_SOURCE_COMMIT,
        "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB,
        "b21_base_package_zip_sha256": B21_PACKAGE_SHA256,
        "b21_verify_competition_md5": True,
        "b21_authoritative_report_path": B21_REPORT_PATH,
        "b21_original_submission_sha256": B21_ORIGINAL_SUBMISSION_SHA256,
        "b21_expected_selected_models": EXPECTED_SELECTED_MODELS,
        "require_byte_match_to_historical_submission": False,
        "expected_kernel_version": 1,
        "probe_family": list(PROBE_NAMES),
        "competition_submission_attempted": False,
        "public_scores_used_to_define_family": False,
        "automatic_compute_retries": 0,
        "enable_internet": False,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"Public-probe request 006 mismatch: {key}")
    expected_paths = [
        "cmi-flu-public-probes-006/control-b21-regenerated.csv",
        "cmi-flu-public-probes-006/probe-task13.csv",
        "cmi-flu-public-probes-006/probe-task12.csv",
        "cmi-flu-public-probes-006/probe-task12-task13.csv",
        "cmi-flu-public-probes-006/bridge-result.json",
        "cmi-flu-public-probes-006/summary.md",
    ]
    if request.get("allowed_output_paths") != expected_paths:
        raise SystemExit("Public-probe request 006 output allowlist mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 150,
        "hard_timeout_minutes": 300,
        "max_active_runs": 1,
    }:
        raise SystemExit("Public-probe request 006 resource contract mismatch")
    repair = request.get("repair", {})
    expected_repair = {
        "failed_stage": "regenerate_frozen_b21",
        "scientific_contract_changed": False,
        "generator_science_blob_changed": False,
        "probe_family_changed": False,
        "backbone_scientific_contract_changed": False,
        "control_artifact_added": True,
        "historical_byte_identity_claimed": False,
        "resource_contract_changed": False,
    }
    for key, value in expected_repair.items():
        if repair.get(key) != value:
            raise SystemExit(f"Public-probe request 006 repair mismatch: {key}")
    cause = str(repair.get("established_root_cause") or "")
    mechanism = str(repair.get("mechanism_change") or "")
    if "byte-for-byte equality" not in cause:
        raise SystemExit("request 006 root cause must identify over-strong historical byte gate")
    if "control CSV" not in mechanism or "non-target column" not in mechanism:
        raise SystemExit("request 006 mechanism must define same-run control and invariance")
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

    v5_output = output.parent / "public-probes-v5-runtime.py"
    run(
        sys.executable,
        str(root / "scripts/cmi_flu_public_probes_prepare_v5.py"),
        "--repository-root",
        str(root),
        "--output",
        str(v5_output),
    )
    text = v5_output.read_text(encoding="utf-8")

    text = replace_once(
        text,
        'REQUEST_ID = "20260906-cmi-flu-public-probes-005"',
        f'REQUEST_ID = "{REQUEST_ID}"',
        label="request id",
    )
    text = replace_once(
        text,
        'default=Path("/kaggle/working/cmi-flu-public-probes-005")',
        f'default=Path("{OUTPUT_DIR}")',
        label="output directory",
    )
    text = replace_once(
        text,
        "# Controlled Public probe family 005\\n\\n",
        "# Controlled Public probe family 006\\n\\n",
        label="summary identity",
    )
    text = replace_once(
        text,
        "CMI_FLU_PUBLIC_PROBES_005_COMPLETE ",
        "CMI_FLU_PUBLIC_PROBES_006_COMPLETE ",
        label="completion marker",
    )

    hash_gate = '''        b21_submission_sha = sha256_file(b21_submission_path)
        if b21_submission_sha != B21_ORIGINAL_SUBMISSION_SHA256:
            raise BundleContractError(
                "regenerated B2.1 submission SHA-256 does not exactly match the original B2.1 submission"
            )
        b21_metrics_sha = sha256_file(b21_metrics_path)
'''
    control_block = f'''        b21_submission_sha = sha256_file(b21_submission_path)
        b21_matches_historical_bytes = b21_submission_sha == B21_ORIGINAL_SUBMISSION_SHA256
        control_path = output_dir / "{CONTROL_FILENAME}"
        shutil.copyfile(b21_submission_path, control_path)
        control_sha = sha256_file(control_path)
        if control_sha != b21_submission_sha:
            raise BundleContractError("same-run regenerated B2.1 control copy SHA-256 mismatch")
        b21_metrics_sha = sha256_file(b21_metrics_path)
'''
    text = replace_once(
        text,
        hash_gate,
        control_block,
        label="replace historical byte gate with same-run control",
    )

    text = replace_once(
        text,
        '            "schema_version": 5,\n            "request_id": REQUEST_ID,',
        '            "schema_version": 6,\n            "request_id": REQUEST_ID,',
        label="result schema version",
    )
    text = replace_once(
        text,
        '            "regenerated_b21_matches_original_submission": True,\n',
        '            "regenerated_b21_matches_original_submission": b21_matches_historical_bytes,\n'
        '            "regenerated_b21_control_filename": "control-b21-regenerated.csv",\n'
        '            "regenerated_b21_control_sha256": control_sha,\n'
        '            "historical_byte_identity_claimed": False,\n',
        label="control provenance payload",
    )
    summary_anchor = '            "- Competition submission attempted by generator: `false`\\n\\n"\n'
    summary_insert = (
        '            f"- regenerated B2.1 control: `control-b21-regenerated.csv` SHA-256 `{control_sha}`\\n"\n'
        '            f"- historical B2.1 byte match: `{str(b21_matches_historical_bytes).lower()}`\\n"\n'
        + summary_anchor
    )
    text = replace_once(
        text,
        summary_anchor,
        summary_insert,
        label="summary control provenance",
    )

    if 'if b21_submission_sha != B21_ORIGINAL_SUBMISSION_SHA256:' in text:
        raise SystemExit("request 006 runtime still contains historical byte hard gate")
    if 'regenerated_b21_matches_original_submission": True' in text:
        raise SystemExit("request 006 runtime still asserts historical byte identity")
    for token in (
        '"Task1.1": "pls_2"',
        '"Task1.2": "enet_a0.001_l0.5"',
        '"Task1.3": "pls_1"',
        '"Task1.4": "raw_pre_vacc_conserved_anchor"',
        '"Task2.1": "et_subtype_d3_l5"',
        '"Task2.2": "et_subtype_d5_l10"',
        '"Task2.3": "ridge_exact_a100"',
        'control_path = output_dir / "control-b21-regenerated.csv"',
        'if control_sha != b21_submission_sha:',
        '"historical_byte_identity_claimed": False',
    ):
        if token not in text:
            raise SystemExit(f"request 006 control contract missing: {token}")
    if "__CMI_FLU_BACKBONE_" in text:
        raise SystemExit("request 006 runtime retains legacy backbone placeholder")
    forbidden = (
        "kernels_status(",
        "kernels_list(",
        "kaggle kernels output",
        "competition_submit",
        "kaggle competitions submit",
    )
    present = [token for token in forbidden if token in text]
    if present:
        raise SystemExit(f"request 006 runtime contains forbidden remote-read/submit tokens: {present}")

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)

    namespace: dict[str, Any] = {"__name__": "public_probes_v6_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("request 006 generated runtime identity mismatch")
    if namespace.get("TARGET_STAGE") != TARGET_STAGE:
        raise SystemExit("request 006 generated runtime stage mismatch")
    if namespace.get("PUBLIC_PROBES_SOURCE_BLOB_SHA") != SCIENCE_BLOB:
        raise SystemExit("request 006 generated runtime science blob mismatch")
    if namespace.get("PACKAGE_ZIP_SHA256") != B21_PACKAGE_SHA256:
        raise SystemExit("request 006 frozen B2.1 package hash mismatch")
    if namespace.get("B21_ADAPTER_BLOB_SHA") != B21_ADAPTER_BLOB:
        raise SystemExit("request 006 frozen B2.1 adapter mismatch")
    if namespace.get("B21_ORIGINAL_SUBMISSION_SHA256") != B21_ORIGINAL_SUBMISSION_SHA256:
        raise SystemExit("request 006 historical B2.1 hash reference mismatch")

    run(sys.executable, str(output), "--self-test")
    print(
        "CMI_FLU_PUBLIC_PROBES_PREPARE_V6 PASS "
        f"request_id={REQUEST_ID} target={TARGET_KERNEL} science_blob={SCIENCE_BLOB} "
        f"b21_package={B21_PACKAGE_SHA256} b21_adapter_blob={B21_ADAPTER_BLOB} "
        f"control={CONTROL_FILENAME} historical_byte_identity_claimed=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
