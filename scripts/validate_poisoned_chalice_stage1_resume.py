"""Validate the immutable CPU-only Stage 1 cached-continuation request."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


REQUEST_ID = "20260903-poisoned-chalice-stage1-resume-001"
TARGET = "renta0426/stage1-raw-fim-resume-v1"
SOURCE_DATASET = "renta0426/stage1-raw-fim-submission-v1-output"
SOURCE_KERNEL = "renta0426/stage1-raw-fim-submission-v1"
BUILDER_PATH = "scripts/poisoned_chalice_stage1_resume_builder.py"
BUILDER_BLOB = "d4be3a7c523558ca3a99feefb6a9bcbb0a0a86ef"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def expected_request() -> dict:
    return {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "parent_failed_request_id": "20260902-poisoned-chalice-stage1-raw-fim-005",
        "competition": "poisoned-chalice-icse27",
        "operation": "kernel_run",
        "target": TARGET,
        "source_kernel": SOURCE_KERNEL,
        "source_kernel_version": 2,
        "source_kernel_expected_status": "error",
        "source_failure": "Expected 113 base features, got 111",
        "source_dataset": SOURCE_DATASET,
        "source_dataset_version": 1,
        "builder_path": BUILDER_PATH,
        "builder_blob_sha": BUILDER_BLOB,
        "resource": {
            "accelerator": "cpu",
            "expected_runtime_minutes": 5,
            "hard_timeout_minutes": 30,
            "max_active_runs": 1,
        },
        "api_budget": {
            "max_calls": 20,
            "poll_interval_seconds": 300,
            "max_pages": 2,
        },
        "side_effects": [
            "create one private notebook version and start one CPU run"
        ],
        "automatic_compute_retries": 0,
        "enable_internet": False,
        "competition_submission": False,
    }


def validate_request(request_path: Path, builder_path: Path) -> dict:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request != expected_request():
        raise RuntimeError("Stage1 resume request differs from the exact contract")
    builder = builder_path.read_bytes()
    if git_blob_sha(builder) != BUILDER_BLOB:
        raise RuntimeError("Stage1 resume builder Git blob mismatch")
    compile(builder, BUILDER_PATH, "exec")
    return request


def validate_notebook(root: Path) -> dict:
    files = sorted(path.name for path in root.iterdir() if path.is_file())
    if files != ["kernel-metadata.json", "stage1-raw-fim-resume-v1.ipynb"]:
        raise RuntimeError(f"unexpected kernel files: {files}")
    metadata = json.loads((root / "kernel-metadata.json").read_text(encoding="utf-8"))
    expected_metadata = {
        "id": TARGET,
        "title": "Stage1 Raw FIM Resume V1",
        "code_file": "stage1-raw-fim-resume-v1.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [SOURCE_DATASET],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "keywords": ["cpu", "membership-inference", "resume", "cached-features"],
    }
    if metadata != expected_metadata:
        raise RuntimeError("Stage1 resume kernel metadata changed")

    notebook = json.loads(
        (root / "stage1-raw-fim-resume-v1.ipynb").read_text(encoding="utf-8")
    )
    if notebook.get("nbformat") != 4 or len(notebook.get("cells", [])) != 2:
        raise RuntimeError("Stage1 resume notebook structure changed")
    if [cell.get("id") for cell in notebook["cells"]] != [
        "pc-stage1-resume-00",
        "pc-stage1-resume-01",
    ]:
        raise RuntimeError("stable cell IDs are missing")
    code = "".join(notebook["cells"][1].get("source", []))
    tree = ast.parse(code, filename="stage1-raw-fim-resume-v1.ipynb")

    required = (
        'SOURCE_DATASET = "renta0426/stage1-raw-fim-submission-v1-output"',
        'SOURCE_KERNEL = "renta0426/stage1-raw-fim-submission-v1"',
        'SOURCE_FAILURE = "Expected 113 base features, got 111"',
        '"validation_order"',
        '"token_count", "window_count"',
        "EXPECTED_TRAIN_SHARDS = 40",
        "EXPECTED_VALIDATION_SHARDS = 20",
        "EXPECTED_BASE_FEATURES = 113",
        "EXPECTED_STRUCTURE_FEATURES = 50",
        "EXPECTED_FIM_FEATURES = 11",
        "EXPECTED_OOF_AUC = 0.664524",
        "OOF_AUC_TOLERANCE = 0.002",
        '"source_extraction_reused": True',
        '"gpu_forward_passes": 0',
        '"hidden_validation_labels_used": False',
        '"public_leaderboard_tuning_used": False',
        '"validation_labels_used_for_fit_or_feature_selection": False',
        '"submission_created": True',
    )
    for marker in required:
        if marker not in code:
            raise RuntimeError(f"resume notebook invariant missing: {marker}")

    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden_imports = {
        "torch",
        "transformers",
        "datasets",
        "requests",
        "urllib",
        "subprocess",
        "socket",
    }
    if imported.intersection(forbidden_imports):
        raise RuntimeError(
            f"resume notebook gained forbidden imports: {sorted(imported & forbidden_imports)}"
        )
    for forbidden in (
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
        "competitions submit",
        "kernels push",
        "load_dataset",
        "AutoModel",
        "AutoTokenizer",
    ):
        if forbidden in code:
            raise RuntimeError(f"resume notebook gained forbidden capability: {forbidden}")
    return {
        "files": files,
        "cells": len(notebook["cells"]),
        "accelerator": "cpu",
        "internet": False,
        "source_dataset": SOURCE_DATASET,
    }


def validate_launch_workflow(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    for marker in (
        "permissions: {}",
        "group: kaggle-resource-global",
        "cancel-in-progress: false",
        "environment: kaggle-readonry",
        "REQUESTED_ACCELERATOR: cpu",
        f"TARGET_KERNEL: {TARGET}",
        f"SOURCE_DATASET: {SOURCE_DATASET}",
        f"SOURCE_KERNEL: {SOURCE_KERNEL}",
        'EXPECTED_SOURCE_VERSION: "2"',
        "RESUME_LIVE_PREFLIGHT PASS",
        "SOURCE_DATASET_SCHEMA PASS feature_shards=60",
        "STAGE1_RESUME_KERNEL_BUILD PASS",
        "RESUME_KERNEL_LAUNCH PASS",
        "automatic_retries=0 submission=false",
        "STAGE1_RESUME_RUNNER_LOCAL_MATERIAL_REMOVED",
    ):
        if marker not in source:
            raise RuntimeError(f"launch workflow invariant missing: {marker}")
    if source.count('"${WORKDIR}/venv/bin/kaggle" kernels push') != 1:
        raise RuntimeError("launch workflow must contain exactly one kernels push")
    if source.count("${{ secrets.KAGGLE_API_TOKEN }}") != 3:
        raise RuntimeError("Kaggle secret step count changed")
    for forbidden in (
        "workflow_dispatch",
        "continue-on-error: true",
        "actions/checkout",
        "actions/upload-artifact",
        "actions/download-artifact",
        "actions/cache",
        "competitions submit",
        "kernels delete",
        "kernels cancel",
        "datasets create",
        "datasets version",
    ):
        if forbidden in source:
            raise RuntimeError(f"launch workflow gained forbidden operation: {forbidden}")
    if re.search(r"^\s*sleep\s+", source, flags=re.MULTILINE):
        raise RuntimeError("launch workflow must not poll")


def validate(
    request_path: Path,
    builder_path: Path,
    launch_workflow: Path | None = None,
) -> dict:
    request = validate_request(request_path, builder_path)
    with tempfile.TemporaryDirectory(prefix="stage1-resume-static-") as temporary:
        output = Path(temporary) / "kernel"
        subprocess.run(
            [
                sys.executable,
                str(builder_path),
                "--output-dir",
                str(output),
                "--builder-blob-sha",
                request["builder_blob_sha"],
            ],
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        notebook = validate_notebook(output)
    if launch_workflow is not None:
        validate_launch_workflow(launch_workflow)
    result = {
        "status": "pass",
        "request_id": request["request_id"],
        "target": request["target"],
        "builder_blob_sha": request["builder_blob_sha"],
        "source_dataset": request["source_dataset"],
        "source_kernel_version": request["source_kernel_version"],
        "resource": request["resource"],
        "notebook": notebook,
        "competition_submission": False,
        "automatic_compute_retries": 0,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--launch-workflow", type=Path)
    args = parser.parse_args()
    validate(args.request, args.builder, args.launch_workflow)


if __name__ == "__main__":
    main()
