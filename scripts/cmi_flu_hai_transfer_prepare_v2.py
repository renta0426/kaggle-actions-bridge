#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import py_compile
import subprocess
import sys
from typing import Any

REQUEST_ID = "20260904-cmi-flu-hai-transfer-002"
PARENT_REQUEST_ID = "20260904-cmi-flu-hai-transfer-001"
FAILED_WORKFLOW_RUN_ID = 33871307603
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_COMMIT = "d1ebf13dc0dc7e5d5a2798b29c288265cbf56618"
SCIENCE_PATH = "src/cmi_flu/hai_transfer.py"
SCIENCE_BLOB = "b671d8bf7f10bebbd65aca2a5bad42e267ee78d5"
B21_BASE_REQUEST_ID = "20260903-cmi-flu-b21-001"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
COMPAT_PATH = "scripts/cmi_flu_hai_transfer_b21_compat.py"
COMPAT_BLOB = "e2973831077ed8893ab1673c9633eb42df7b26de"
SEQUENCE_SHA256 = "63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887"
VACCINE_SHA256 = "8f6c7116f37f29df0bb21d6049d82fa28b4e42b2d10ed9394a1ae6f926bd9f35"
TARGET_COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-hai-strain-transfer-20260904-002"
TARGET_STAGE = "phase_a_hai_strain_transfer"
EXPECTED_KERNEL_VERSION = 1
ALLOWED_OUTPUT_PATHS = [
    "cmi-flu-hai-transfer-002/bridge-result.json",
    "cmi-flu-hai-transfer-002/metrics.json",
    "cmi-flu-hai-transfer-002/summary.md",
]


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def load_compat(root: pathlib.Path) -> tuple[str, str]:
    data = (root / COMPAT_PATH).read_bytes()
    found = git_blob_sha(data)
    if found != COMPAT_BLOB:
        raise SystemExit(f"HAI B2.1 compatibility blob mismatch: {found}")
    source = data.decode("utf-8")
    tree = ast.parse(source, filename=COMPAT_PATH)
    names = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    compatible_hai = names.get("compatible_hai")
    if compatible_hai is None:
        raise SystemExit("compatibility shim lacks compatible_hai")
    keyword_names = {item.arg for item in compatible_hai.args.kwonlyargs}
    if "selection_policy" not in keyword_names:
        raise SystemExit("compatibility shim does not accept selection_policy")
    required = (
        'selection_policy != "robust_v1"',
        "return robust_hai(",
        "evaluation.run_hai_compact_for_panels = compatible_hai",
        "evaluation.evaluate_hai_spec = compatible_evaluate_hai_spec",
        "result.panel_proxy_fold_metrics =",
        "result.panel_proxy_fold_summary =",
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise SystemExit(f"compatibility shim contract tokens missing: {missing}")
    return source, hashlib.sha256(data).hexdigest()


def validate_request(root: pathlib.Path) -> dict[str, Any]:
    request = json.loads(
        (root / "requests/cmi-flu-hai-transfer-002.json").read_text(encoding="utf-8")
    )
    if request.get("schema_version") != 1 or request.get("request_id") != REQUEST_ID:
        raise SystemExit("HAI repair request identity mismatch")
    if request.get("parent_request_id") != PARENT_REQUEST_ID:
        raise SystemExit("HAI repair parent request mismatch")
    if request.get("failed_workflow_run_id") != FAILED_WORKFLOW_RUN_ID:
        raise SystemExit("HAI repair failed-run lineage mismatch")
    if request.get("competition") != TARGET_COMPETITION:
        raise SystemExit("HAI repair competition mismatch")
    if request.get("target") != TARGET_KERNEL:
        raise SystemExit("HAI repair target mismatch")
    if request.get("operation") != "kernel_run_and_current_output_read":
        raise SystemExit("HAI repair operation mismatch")
    if request.get("science_repository") != SCIENCE_REPOSITORY:
        raise SystemExit("HAI repair science repository mismatch")
    if request.get("science_source_commit") != SCIENCE_COMMIT:
        raise SystemExit("HAI repair science commit changed")
    if request.get("science_source_path") != SCIENCE_PATH:
        raise SystemExit("HAI repair science path changed")
    if request.get("hai_transfer_blob_sha") != SCIENCE_BLOB:
        raise SystemExit("HAI repair science blob changed")
    if request.get("b21_base_request_id") != B21_BASE_REQUEST_ID:
        raise SystemExit("HAI repair B2.1 base lineage mismatch")
    if request.get("b21_runtime_adapter_blob_sha") != B21_ADAPTER_BLOB:
        raise SystemExit("HAI repair B2.1 adapter changed")
    if request.get("hai_b21_compatibility_blob_sha") != COMPAT_BLOB:
        raise SystemExit("HAI repair compatibility shim mismatch")
    if request.get("sequence_reference_sha256") != SEQUENCE_SHA256:
        raise SystemExit("HAI repair sequence reference changed")
    if request.get("vaccine_reference_sha256") != VACCINE_SHA256:
        raise SystemExit("HAI repair vaccine reference changed")
    if request.get("expected_kernel_version") != EXPECTED_KERNEL_VERSION:
        raise SystemExit("HAI repair expected version mismatch")
    if request.get("allowed_output_paths") != ALLOWED_OUTPUT_PATHS:
        raise SystemExit("HAI repair output allowlist mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 120,
        "hard_timeout_minutes": 210,
        "max_active_runs": 1,
    }:
        raise SystemExit("HAI repair resource contract mismatch")
    if request.get("api_budget") != {
        "max_calls": 24,
        "initial_poll_schedule_seconds": [30, 60, 120, 300],
        "steady_poll_interval_seconds": 900,
        "max_pages": 2,
    }:
        raise SystemExit("HAI repair API budget mismatch")
    if request.get("competition_submission_attempted") is not False:
        raise SystemExit("HAI repair must not submit")
    if request.get("automatic_compute_retries") != 0:
        raise SystemExit("HAI repair automatic retry must remain zero")
    if request.get("enable_internet") is not False:
        raise SystemExit("HAI repair Kaggle internet must remain disabled")
    return request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=pathlib.Path, required=True)
    parser.add_argument("--sequence-reference", type=pathlib.Path, required=True)
    parser.add_argument("--vaccine-reference", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    validate_request(root)
    compat_source, compat_sha256 = load_compat(root)

    old_output = output.parent / "hai-transfer-001-base.py"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/cmi_flu_hai_transfer_prepare.py"),
            "--repository-root",
            str(root),
            "--sequence-reference",
            str(args.sequence_reference.expanduser().resolve()),
            "--vaccine-reference",
            str(args.vaccine_reference.expanduser().resolve()),
            "--output",
            str(old_output),
        ],
        check=True,
    )
    text = old_output.read_text(encoding="utf-8")

    # Request 002 is a runtime-compatibility overlay only. Science source/provenance stay exact.
    text = replace_once(
        text,
        f'REQUEST_ID = "{PARENT_REQUEST_ID}"',
        f'REQUEST_ID = "{REQUEST_ID}"',
        label="request id overlay",
    )
    text = replace_once(
        text,
        'default=Path("/kaggle/working/cmi-flu-hai-transfer")',
        'default=Path("/kaggle/working/cmi-flu-hai-transfer-002")',
        label="output directory overlay",
    )

    execute_marker = "\ndef execute(input_dir: Path, output_dir: Path) -> Mapping[str, Any]:\n"
    compat_constants = (
        f'\nHAI_B21_COMPATIBILITY_SOURCE_BLOB_SHA = "{COMPAT_BLOB}"\n'
        f'HAI_B21_COMPATIBILITY_SOURCE_SHA256 = "{compat_sha256}"\n'
        f'HAI_B21_COMPATIBILITY_SOURCE = {compat_source!r}\n'
    )
    text = replace_once(
        text,
        execute_marker,
        compat_constants + execute_marker,
        label="compatibility source insertion",
    )

    old_install = '''        install_adapter()\n\n        stage = "load_hai_transfer"\n'''
    new_install = '''        install_adapter()\n\n        stage = "install_hai_b21_compatibility"\n        compat_namespace: dict[str, Any] = {}\n        exec(\n            compile(\n                HAI_B21_COMPATIBILITY_SOURCE,\n                "<cmi_flu_hai_transfer_b21_compat>",\n                "exec",\n            ),\n            compat_namespace,\n            compat_namespace,\n        )\n        install_hai_compat = compat_namespace.get("install")\n        if not callable(install_hai_compat):\n            raise BundleContractError("HAI B2.1 compatibility shim lacks install()")\n        install_hai_compat()\n\n        stage = "load_hai_transfer"\n'''
    text = replace_once(
        text,
        old_install,
        new_install,
        label="B2.1 compatibility installation",
    )

    old_manifest = '''            "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB_SHA,\n            "sequence_reference_sha256": SEQUENCE_REFERENCE_SHA256,\n'''
    new_manifest = '''            "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB_SHA,\n            "hai_b21_compatibility_blob_sha": HAI_B21_COMPATIBILITY_SOURCE_BLOB_SHA,\n            "hai_b21_compatibility_sha256": HAI_B21_COMPATIBILITY_SOURCE_SHA256,\n            "sequence_reference_sha256": SEQUENCE_REFERENCE_SHA256,\n'''
    text = replace_once(
        text,
        old_manifest,
        new_manifest,
        label="compatibility provenance manifest",
    )

    if PARENT_REQUEST_ID in text:
        raise SystemExit("request 001 identity remained in repair runtime")
    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'SOURCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'HAI_TRANSFER_SOURCE_BLOB_SHA = "{SCIENCE_BLOB}"',
        f'HAI_B21_COMPATIBILITY_SOURCE_BLOB_SHA = "{COMPAT_BLOB}"',
        'stage = "install_hai_b21_compatibility"',
        "install_hai_compat()",
        'default=Path("/kaggle/working/cmi-flu-hai-transfer-002")',
        '"hai_b21_compatibility_blob_sha": HAI_B21_COMPATIBILITY_SOURCE_BLOB_SHA',
        "selection_policy=selection_policy",
        "CMI_FLU_HAI_TRANSFER_COMPLETE",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"HAI repair runtime missing tokens: {missing}")
    forbidden = ("kaggle competitions submit", "api.competition_submit")
    present = [token for token in forbidden if token in text]
    if present:
        raise SystemExit(f"HAI repair runtime contains submission tokens: {present}")

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)

    namespace: dict[str, Any] = {"__name__": "hai_transfer_002_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("generated repair runtime request mismatch")
    if namespace.get("SOURCE_COMMIT") != SCIENCE_COMMIT:
        raise SystemExit("generated repair runtime science commit mismatch")
    if namespace.get("HAI_TRANSFER_SOURCE_BLOB_SHA") != SCIENCE_BLOB:
        raise SystemExit("generated repair runtime science blob mismatch")
    if namespace.get("HAI_B21_COMPATIBILITY_SOURCE_BLOB_SHA") != COMPAT_BLOB:
        raise SystemExit("generated repair runtime compatibility blob mismatch")

    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_HAI_TRANSFER_002_PREPARE PASS "
        f"science_commit={SCIENCE_COMMIT} science_blob={SCIENCE_BLOB} "
        f"compat_blob={COMPAT_BLOB} b21_adapter_blob={B21_ADAPTER_BLOB} "
        f"target_kernel={TARGET_KERNEL} expected_version=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
