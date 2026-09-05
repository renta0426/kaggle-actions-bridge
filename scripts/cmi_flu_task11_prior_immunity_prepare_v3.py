#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import py_compile
import subprocess
import sys
from typing import Any

REQUEST_ID = "20260906-cmi-flu-task11-prior-immunity-002"
PARENT_REQUEST_ID = "20260905-cmi-flu-task11-prior-immunity-001"
SCIENCE_COMMIT = "06c85e6263b59cd6ac97b7087779e7a9fb1cbdae"
SCIENCE_BLOB = "50d9a43604d2b75479b8f873a86a8daf9d5bd7a9"
TARGET_KERNEL = "renta0426/cmi-flu-phase-b-t11-prior-immunity-20260906-002"
OUTPUT_DIR_OLD = "/kaggle/working/cmi-flu-task11-prior-immunity-001"
OUTPUT_DIR_NEW = "/kaggle/working/cmi-flu-task11-prior-immunity-002"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def validate_request(root: pathlib.Path) -> None:
    request_path = root / "requests/cmi-flu-task11-prior-immunity-002.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "parent_request_id": PARENT_REQUEST_ID,
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_repository": "renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit": SCIENCE_COMMIT,
        "science_source_path": "src/cmi_flu/task11_prior_immunity.py",
        "science_transport": "agent_relay_exact_blob",
        "relayed_science_path": "payloads/cmi-flu-task11-prior-immunity-001/task11_prior_immunity.py",
        "task11_prior_immunity_blob_sha": SCIENCE_BLOB,
        "b21_base_request_id": "20260903-cmi-flu-b21-001",
        "b21_runtime_adapter_blob_sha": "de71dea0e335bdcd79325c0de926bf8848d0979f",
        "expected_kernel_version": 1,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
        "enable_internet": False,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"Task1.1 prior-immunity repair request mismatch: {key}")
    if request.get("allowed_output_paths") != [
        "cmi-flu-task11-prior-immunity-002/bridge-result.json",
        "cmi-flu-task11-prior-immunity-002/metrics.json",
        "cmi-flu-task11-prior-immunity-002/summary.md",
    ]:
        raise SystemExit("Task1.1 prior-immunity repair output allowlist mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 60,
        "hard_timeout_minutes": 120,
        "max_active_runs": 1,
    }:
        raise SystemExit("Task1.1 prior-immunity repair resource contract mismatch")
    if request.get("api_budget") != {
        "max_calls": 18,
        "initial_poll_schedule_seconds": [30, 60, 120, 300],
        "steady_poll_interval_seconds": 600,
        "max_pages": 2,
    }:
        raise SystemExit("Task1.1 prior-immunity repair API budget mismatch")
    if request.get("repair") != {
        "parent_push_status": "bad_request",
        "parent_title_length": 51,
        "parent_slug_length": 51,
        "repair_scope": "shorten_title_and_slug_to_kaggle_50_character_limit_only",
    }:
        raise SystemExit("Task1.1 prior-immunity repair provenance mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=pathlib.Path, required=True)
    parser.add_argument("--science-source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    science = args.science_source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    validate_request(root)

    base_output = output.parent / "task11-prior-immunity-001-runtime.py"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/cmi_flu_task11_prior_immunity_prepare_v2.py"),
            "--repository-root",
            str(root),
            "--science-source",
            str(science),
            "--output",
            str(base_output),
        ],
        check=True,
    )
    text = base_output.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'REQUEST_ID = "20260905-cmi-flu-task11-prior-immunity-001"',
        f'REQUEST_ID = "{REQUEST_ID}"',
        label="request id",
    )
    text = replace_once(text, OUTPUT_DIR_OLD, OUTPUT_DIR_NEW, label="output directory")
    text = text.replace("TASK11_PRIOR_IMMUNITY_001", "TASK11_PRIOR_IMMUNITY_002")

    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'SOURCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'TASK11_PRIOR_IMMUNITY_SOURCE_BLOB_SHA = "{SCIENCE_BLOB}"',
        OUTPUT_DIR_NEW,
        "ANCHOR_RESIDUAL_LAMBDA = 0.25",
        "FUSION_WEIGHTS = (0.25, 0.5)",
        "BASE_CONDITIONS = ('b1', 'b21', 'anchor_residual')",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"Task1.1 prior-immunity repair runtime missing tokens: {missing}")
    if PARENT_REQUEST_ID in text:
        raise SystemExit("Task1.1 prior-immunity repair runtime retained parent request id")
    if OUTPUT_DIR_OLD in text:
        raise SystemExit("Task1.1 prior-immunity repair runtime retained parent output directory")
    if "kaggle competitions submit" in text or "api.competition_submit" in text:
        raise SystemExit("Task1.1 prior-immunity repair runtime contains submission path")

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)
    namespace: dict[str, Any] = {"__name__": "task11_prior_immunity_repair_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("Task1.1 prior-immunity repair runtime request identity mismatch")
    if namespace.get("SOURCE_COMMIT") != SCIENCE_COMMIT:
        raise SystemExit("Task1.1 prior-immunity repair science commit mismatch")
    if namespace.get("TASK11_PRIOR_IMMUNITY_SOURCE_BLOB_SHA") != SCIENCE_BLOB:
        raise SystemExit("Task1.1 prior-immunity repair science blob mismatch")

    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_TASK11_PRIOR_IMMUNITY_PREPARE_V3 PASS "
        f"request_id={REQUEST_ID} science_commit={SCIENCE_COMMIT} science_blob={SCIENCE_BLOB} "
        "repair=title_slug_length_only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
