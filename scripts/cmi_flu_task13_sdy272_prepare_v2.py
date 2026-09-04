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

REQUEST_ID = "20260904-cmi-flu-task13-sdy272-002"
PARENT_REQUEST_ID = "20260904-cmi-flu-task13-sdy272-001"
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_COMMIT = "67b48da985c2f4727a635d07acea6a4f5745b8ab"
SCIENCE_PATH = "src/cmi_flu/task13_harmonization.py"
SCIENCE_TRANSPORT = "agent_relay_exact_blob"
PAYLOAD_PATH = "payloads/cmi-flu-task13-sdy272-002/task13_harmonization.py"
TASK13_BLOB = "5c6725dc757a5ba9dd21289b1c4f09997e1afdb8"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
TARGET_COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-task1-3-sdy272-20260904-002"
TARGET_STAGE = "phase_a_task13_sdy272_harmonization"
EXPECTED_KERNEL_VERSION = 1
OLD_REQUEST_ID = "20260904-cmi-flu-task13-sdy272-001"
OLD_SCIENCE_COMMIT = "baae9fb40329057a316935d0d1285adc64c948b0"
OLD_TASK13_BLOB = "0f1e728fe2e5ea0f3713c1442c3beeba21b8d347"
ALLOWED_OUTPUT_PATHS = [
    "cmi-flu-task13-sdy272/bridge-result.json",
    "cmi-flu-task13-sdy272/metrics.json",
    "cmi-flu-task13-sdy272/summary.md",
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


def replace_assignment_line(text: str, name: str, value_repr: str) -> str:
    prefix = f"{name} = "
    matches = [line for line in text.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise SystemExit(f"{name}: expected one assignment, found {len(matches)}")
    return text.replace(matches[0], f"{prefix}{value_repr}", 1)


def load_relayed_science_source(root: pathlib.Path) -> tuple[str, str]:
    path = root / PAYLOAD_PATH
    data = path.read_bytes()
    if len(data) <= 0 or len(data) > 65536:
        raise SystemExit("Task1.3 repaired science source byte budget mismatch")
    found = git_blob_sha(data)
    if found != TASK13_BLOB:
        raise SystemExit(f"Task1.3 repaired science source blob mismatch: {found}")
    source = data.decode("utf-8")
    compile(source, SCIENCE_PATH, "exec")
    if "aggregate_repeats=" in source:
        raise SystemExit("Task1.3 repaired source still depends on post-B2 aggregate_repeats keyword")
    return source, hashlib.sha256(data).hexdigest()


def validate_request(root: pathlib.Path) -> dict[str, Any]:
    request = json.loads(
        (root / "requests/cmi-flu-task13-sdy272-002.json").read_text(encoding="utf-8")
    )
    allowed = {
        "schema_version",
        "request_id",
        "parent_request_id",
        "competition",
        "operation",
        "target",
        "purpose",
        "science_repository",
        "science_source_commit",
        "science_source_path",
        "science_transport",
        "task13_harmonization_blob_sha",
        "b21_base_request_id",
        "b21_runtime_adapter_blob_sha",
        "expected_kernel_version",
        "allowed_output_paths",
        "resource",
        "api_budget",
        "side_effects",
        "competition_submission_attempted",
        "automatic_compute_retries",
        "enable_internet",
        "rules_checked_at_utc",
    }
    if set(request) != allowed:
        raise SystemExit(f"manifest fields mismatch: {sorted(set(request) ^ allowed)}")
    if request["schema_version"] != 1 or request["request_id"] != REQUEST_ID:
        raise SystemExit("request identity mismatch")
    if request["parent_request_id"] != PARENT_REQUEST_ID:
        raise SystemExit("parent request mismatch")
    if (
        request["competition"] != TARGET_COMPETITION
        or request["target"] != TARGET_KERNEL
        or request["operation"] != "kernel_run_and_current_output_read"
    ):
        raise SystemExit("target/operation mismatch")
    if (
        request["science_repository"] != SCIENCE_REPOSITORY
        or request["science_source_commit"] != SCIENCE_COMMIT
        or request["science_source_path"] != SCIENCE_PATH
        or request["science_transport"] != SCIENCE_TRANSPORT
        or request["task13_harmonization_blob_sha"] != TASK13_BLOB
    ):
        raise SystemExit("science provenance mismatch")
    if request["b21_base_request_id"] != "20260903-cmi-flu-b21-001":
        raise SystemExit("B2.1 lineage mismatch")
    if request["b21_runtime_adapter_blob_sha"] != B21_ADAPTER_BLOB:
        raise SystemExit("B2.1 adapter provenance mismatch")
    if request["expected_kernel_version"] != EXPECTED_KERNEL_VERSION:
        raise SystemExit("expected kernel version mismatch")
    if request["allowed_output_paths"] != ALLOWED_OUTPUT_PATHS:
        raise SystemExit("allowed output path contract mismatch")
    if request["resource"] != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 90,
        "hard_timeout_minutes": 180,
        "max_active_runs": 1,
    }:
        raise SystemExit("resource contract mismatch")
    if request["api_budget"] != {
        "max_calls": 30,
        "poll_interval_seconds": 900,
        "max_pages": 2,
    }:
        raise SystemExit("API budget mismatch")
    if request["side_effects"] != [
        "create one new private Notebook and start one CPU run",
        "after successful completion read only current version 1 aggregate outputs bridge-result.json metrics.json summary.md",
    ]:
        raise SystemExit("side-effect contract mismatch")
    if (
        request["competition_submission_attempted"] is not False
        or request["automatic_compute_retries"] != 0
        or request["enable_internet"] is not False
    ):
        raise SystemExit("safety contract mismatch")
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
    repaired_source, repaired_sha256 = load_relayed_science_source(root)

    old_output = output.parent / "task13-001-base.py"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/cmi_flu_task13_sdy272_prepare.py"),
            "--repository-root",
            str(root),
            "--output",
            str(old_output),
        ],
        check=True,
    )
    text = old_output.read_text(encoding="utf-8")
    text = replace_once(
        text,
        f'REQUEST_ID = "{OLD_REQUEST_ID}"',
        f'REQUEST_ID = "{REQUEST_ID}"',
        label="request id overlay",
    )
    text = replace_once(
        text,
        f'SOURCE_COMMIT = "{OLD_SCIENCE_COMMIT}"',
        f'SOURCE_COMMIT = "{SCIENCE_COMMIT}"',
        label="science commit overlay",
    )
    text = replace_assignment_line(
        text,
        "TASK13_HARMONIZATION_SOURCE_BLOB_SHA",
        repr(TASK13_BLOB),
    )
    text = replace_assignment_line(
        text,
        "TASK13_HARMONIZATION_SOURCE_SHA256",
        repr(repaired_sha256),
    )
    text = replace_assignment_line(
        text,
        "TASK13_HARMONIZATION_SOURCE",
        repr(repaired_source),
    )
    if OLD_TASK13_BLOB in text or OLD_SCIENCE_COMMIT in text or OLD_REQUEST_ID in text:
        raise SystemExit("request 001 provenance remained in repaired runtime")
    if "aggregate_repeats=" in repaired_source:
        raise SystemExit("post-B2-only Task1.3 API keyword remained")

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)

    namespace: dict[str, Any] = {"__name__": "task13_sdy272_002_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("generated runtime request identity mismatch")
    if namespace.get("SOURCE_COMMIT") != SCIENCE_COMMIT:
        raise SystemExit("generated runtime science commit mismatch")
    if namespace.get("TARGET_STAGE") != TARGET_STAGE:
        raise SystemExit("generated runtime stage identity mismatch")
    if namespace.get("TASK13_HARMONIZATION_SOURCE_BLOB_SHA") != TASK13_BLOB:
        raise SystemExit("generated runtime Task1.3 source mismatch")
    if namespace.get("B21_ADAPTER_BLOB_SHA") != B21_ADAPTER_BLOB:
        raise SystemExit("generated runtime B2.1 adapter mismatch")
    helper = namespace.get("task13_json_safe")
    if not callable(helper):
        raise SystemExit("generated runtime JSON safety helper missing")
    probe = helper({"nan": float("nan"), "values": (1.5, 2)})
    if probe != {"nan": None, "values": [1.5, 2]}:
        raise SystemExit(f"task13_json_safe smoke mismatch: {probe!r}")

    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_TASK13_SDY272_002_PREPARE PASS "
        f"science_commit={SCIENCE_COMMIT} task13_blob={TASK13_BLOB} "
        f"b21_adapter_blob={B21_ADAPTER_BLOB} repair=drop_post_b2_aggregate_repeats_keyword "
        f"target_kernel={TARGET_KERNEL} expected_version=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
