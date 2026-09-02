"""Validate cached Stage 1 continuation request 003 without credentials."""

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


REQUEST_ID = "20260903-poisoned-chalice-stage1-resume-003"
TARGET = "renta0426/stage1-raw-fim-resume-v1"
SOURCE_DATASET = "renta0426/stage1-raw-fim-submission-v1-output"
SOURCE_KERNEL = "renta0426/stage1-raw-fim-submission-v1"
BUILDER_PATH = "scripts/poisoned_chalice_stage1_resume_builder.py"
BUILDER_BLOB = "d4be3a7c523558ca3a99feefb6a9bcbb0a0a86ef"
PATCHER_PATH = "scripts/poisoned_chalice_stage1_resume_v3_patch.py"
PATCHER_BLOB = "a424e66b0a0efce5edb42547bb6dd961013aba1c"
SHARD_VALIDATOR_PATH = "scripts/poisoned_chalice_stage1_dataset_shards.py"
SHARD_VALIDATOR_BLOB = "5ff8ddc8913c2ad8ae1c94cb5aab6c8c9baa7220"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def expected_request() -> dict[str, object]:
    return {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "parent_failed_request_id": "20260903-poisoned-chalice-stage1-resume-002",
        "source_launch_request_id": "20260902-poisoned-chalice-stage1-raw-fim-005",
        "competition": "poisoned-chalice-icse27",
        "operation": "kernel_run",
        "target": TARGET,
        "source_kernel": SOURCE_KERNEL,
        "source_kernel_version": 3,
        "source_kernel_expected_status": "error",
        "source_failure": "Expected 113 base features, got 111",
        "source_dataset": SOURCE_DATASET,
        "source_dataset_version": 1,
        "previous_preflight_failure": (
            "feature shard basenames were deduplicated across train and validation"
        ),
        "builder_path": BUILDER_PATH,
        "builder_blob_sha": BUILDER_BLOB,
        "provenance_patcher_path": PATCHER_PATH,
        "provenance_patcher_blob_sha": PATCHER_BLOB,
        "shard_validator_path": SHARD_VALIDATOR_PATH,
        "shard_validator_blob_sha": SHARD_VALIDATOR_BLOB,
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


def literal_assignments(tree: ast.Module) -> dict[str, object]:
    result: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = value
    return result


def imported_modules(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".", 1)[0])
    return modules


def validate_sources(
    request_path: Path,
    builder_path: Path,
    patcher_path: Path,
    shard_validator_path: Path,
) -> dict[str, object]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request != expected_request():
        raise RuntimeError("resume-v3 request differs from exact contract")
    expected = {
        builder_path: (BUILDER_PATH, BUILDER_BLOB),
        patcher_path: (PATCHER_PATH, PATCHER_BLOB),
        shard_validator_path: (SHARD_VALIDATOR_PATH, SHARD_VALIDATOR_BLOB),
    }
    for path, (label, blob) in expected.items():
        data = path.read_bytes()
        if git_blob_sha(data) != blob:
            raise RuntimeError(f"Git blob mismatch: {label}")
        compile(data, label, "exec")
    subprocess.run(
        [sys.executable, str(shard_validator_path), "--self-test"],
        check=True,
        timeout=30,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return request


def validate_notebook(root: Path) -> dict[str, object]:
    files = sorted(path.name for path in root.iterdir() if path.is_file())
    if files != ["kernel-metadata.json", "stage1-raw-fim-resume-v1.ipynb"]:
        raise RuntimeError(f"unexpected resume-v3 kernel files: {files}")
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
        raise RuntimeError("resume-v3 kernel metadata changed")
    notebook = json.loads((root / metadata["code_file"]).read_text(encoding="utf-8"))
    cells = notebook.get("cells") or []
    if notebook.get("nbformat") != 4 or len(cells) != 2:
        raise RuntimeError("resume-v3 notebook structure changed")
    if [cell.get("id") for cell in cells] != [
        "pc-stage1-resume-00",
        "pc-stage1-resume-01",
    ]:
        raise RuntimeError("resume-v3 stable cell IDs missing")
    raw = cells[1].get("source", "")
    code = "".join(raw) if isinstance(raw, list) else str(raw)
    tree = ast.parse(code, filename=metadata["code_file"])
    constants = literal_assignments(tree)
    expected_constants = {
        "REQUEST_ID": REQUEST_ID,
        "SOURCE_DATASET": SOURCE_DATASET,
        "SOURCE_DATASET_VERSION": 1,
        "SOURCE_KERNEL": SOURCE_KERNEL,
        "SOURCE_KERNEL_VERSION": 3,
        "SOURCE_FAILURE": "Expected 113 base features, got 111",
        "BRIDGE_BUILDER_BLOB_SHA": BUILDER_BLOB,
        "EXPECTED_TRAIN_ROWS": 10_000,
        "EXPECTED_VALIDATION_ROWS": 5_000,
        "EXPECTED_TRAIN_SHARDS": 40,
        "EXPECTED_VALIDATION_SHARDS": 20,
        "EXPECTED_BASE_FEATURES": 113,
        "EXPECTED_STRUCTURE_FEATURES": 50,
        "EXPECTED_FIM_FEATURES": 11,
        "EXPECTED_OOF_AUC": 0.664524,
        "OOF_AUC_TOLERANCE": 0.002,
    }
    for name, value in expected_constants.items():
        if constants.get(name) != value:
            raise RuntimeError(f"resume-v3 notebook constant changed: {name}")
    for old_request in (
        "20260903-poisoned-chalice-stage1-resume-001",
        "20260903-poisoned-chalice-stage1-resume-002",
    ):
        if old_request in code:
            raise RuntimeError(f"old request ID remains in notebook: {old_request}")
    for marker in (
        '"token_count", "window_count"',
        '"source_extraction_reused": True',
        '"gpu_forward_passes": 0',
        '"hidden_validation_labels_used": False',
        '"public_leaderboard_tuning_used": False',
        '"validation_labels_used_for_fit_or_feature_selection": False',
        '"submission_created": True',
    ):
        if marker not in code:
            raise RuntimeError(f"resume-v3 notebook invariant missing: {marker}")
    forbidden_imports = {
        "torch", "transformers", "datasets", "requests", "urllib",
        "subprocess", "socket",
    }
    imports = imported_modules(tree)
    if imports & forbidden_imports:
        raise RuntimeError(
            f"resume-v3 notebook gained forbidden imports: {sorted(imports & forbidden_imports)}"
        )
    for marker in (
        "KAGGLE_API_TOKEN", "KAGGLE_KEY", "competitions submit", "kernels push",
        "load_dataset", "AutoModel", "AutoTokenizer",
    ):
        if marker in code:
            raise RuntimeError(f"resume-v3 notebook gained forbidden capability: {marker}")
    return {
        "request_id": REQUEST_ID,
        "source_kernel_version": 3,
        "accelerator": "cpu",
        "internet": False,
        "algorithm_changed_by_patch": False,
    }


def validate_launch(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    required = (
        "permissions: {}",
        "group: kaggle-resource-global",
        "cancel-in-progress: false",
        "environment: kaggle-readonry",
        "requests/poisoned-chalice-stage1-resume-v3.json",
        "REQUESTED_ACCELERATOR: cpu",
        f"TARGET_KERNEL: {TARGET}",
        f"SOURCE_DATASET: {SOURCE_DATASET}",
        f"SOURCE_KERNEL: {SOURCE_KERNEL}",
        'EXPECTED_SOURCE_VERSION: "3"',
        "poisoned_chalice_stage1_resume_v3_patch.py",
        "poisoned_chalice_stage1_dataset_shards.py",
        "RESUME_V3_LIVE_PREFLIGHT PASS",
        "SOURCE_DATASET_V3_SCHEMA PASS full_paths=60 train=40 validation=20",
        "STAGE1_RESUME_V3_KERNEL_BUILD PASS",
        "RESUME_V3_KERNEL_LAUNCH PASS",
        "automatic_retries=0 submission=false",
        "STAGE1_RESUME_V3_RUNNER_LOCAL_MATERIAL_REMOVED",
    )
    for marker in required:
        if marker not in source:
            raise RuntimeError(f"resume-v3 launch invariant missing: {marker}")
    if source.count('"${WORKDIR}/venv/bin/kaggle" kernels push') != 1:
        raise RuntimeError("resume-v3 launch must contain exactly one kernels push")
    secret_marker = "$" + "{{ secrets.KAGGLE_API_TOKEN }}"
    if source.count(secret_marker) != 3:
        raise RuntimeError("resume-v3 Kaggle secret step count changed")
    for forbidden in (
        "workflow_dispatch", "continue-on-error: true", "actions/checkout",
        "actions/upload-artifact", "actions/download-artifact", "actions/cache",
        "competitions submit", "kernels delete", "kernels cancel",
        "datasets create", "datasets version",
    ):
        if forbidden in source:
            raise RuntimeError(f"resume-v3 launch gained forbidden operation: {forbidden}")
    if re.search(r"^\s*sleep\s+", source, flags=re.MULTILINE):
        raise RuntimeError("resume-v3 launch must not poll")


def validate(
    request_path: Path,
    builder_path: Path,
    patcher_path: Path,
    shard_validator_path: Path,
    launch_workflow: Path | None,
) -> dict[str, object]:
    request = validate_sources(
        request_path, builder_path, patcher_path, shard_validator_path
    )
    with tempfile.TemporaryDirectory(prefix="stage1-resume-v3-") as temporary:
        root = Path(temporary) / "kernel"
        subprocess.run(
            [
                sys.executable, str(builder_path), "--output-dir", str(root),
                "--builder-blob-sha", request["builder_blob_sha"],
            ],
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                sys.executable, str(patcher_path), "--notebook",
                str(root / "stage1-raw-fim-resume-v1.ipynb"),
            ],
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        notebook = validate_notebook(root)
    if launch_workflow is not None:
        validate_launch(launch_workflow)
    result = {
        "status": "pass",
        "request_id": request["request_id"],
        "parent_failed_request_id": request["parent_failed_request_id"],
        "source_kernel_version": request["source_kernel_version"],
        "builder_blob": request["builder_blob_sha"],
        "patcher_blob": request["provenance_patcher_blob_sha"],
        "shard_validator_blob": request["shard_validator_blob_sha"],
        "notebook": notebook,
        "full_path_inventory_validation": True,
        "competition_submission": False,
        "automatic_compute_retries": 0,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--patcher", type=Path, required=True)
    parser.add_argument("--shard-validator", type=Path, required=True)
    parser.add_argument("--launch-workflow", type=Path)
    args = parser.parse_args()
    validate(
        args.request,
        args.builder,
        args.patcher,
        args.shard_validator,
        args.launch_workflow,
    )


if __name__ == "__main__":
    main()
