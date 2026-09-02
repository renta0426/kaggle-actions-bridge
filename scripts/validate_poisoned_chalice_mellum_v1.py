"""Credential-free static validation for the frozen Mellum transfer launch."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re


REQUEST_ID = "20260903-poisoned-chalice-mellum-transfer-v1-001"
RESEARCH_REPOSITORY = "renta0426/The-Poisoned-Chalice-of-LLM-Evaluation"
RESEARCH_COMMIT = "9f6be68c27dc1f3326d68bed4e2abf80db893748"
MATERIALIZER_PATH = "scripts/materialize_poisoned_chalice_mellum_v1.py"
MATERIALIZER_BLOB = "fd9c645c9ff9a128b8f3a8dfdadd7c937e6d62a1"
PRIVATE_LOADER_PATH = "scripts/materialize_poisoned_chalice_mellum_v1_private.py"
PRIVATE_LOADER_BLOB = "425ca4378e64de9f959ec7c241184deb79c70dbd"
SOURCE_FILES = {
    "scripts/build_mellum_transfer_notebook.py": (
        "095725f10279431a0982e7ebb38faa5b03a7754e",
        131_072,
    ),
    "src/poisoned_chalice/stage2.py": (
        "9c2086f3c73ec5998ac3b50d7a4e166f6b1b4443",
        131_072,
    ),
    "configs/mellum_transfer_v1.json": (
        "2fa4b6cbe346283c9047b9228f6764bb52cecdd7",
        32_768,
    ),
    "experiments/pseudo-stage2-transfer-v1/transfer_sample_manifest.parquet": (
        "dedfd34d43e53c158398ae3cc99ed508cbe37f66",
        2_097_152,
    ),
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def assigned_constants(tree: ast.Module) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                result[target.id] = value
    return result


def literal_constant(constants: dict[str, ast.AST], name: str):  # noqa: ANN201
    if name not in constants:
        raise RuntimeError(f"required constant missing: {name}")
    return ast.literal_eval(constants[name])


def validate_request(request: dict, materializer: bytes, loader: bytes) -> None:
    expected = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": "poisoned-chalice-icse27",
        "operation": "kernel_run",
        "target": "renta0426/mellum-transfer-v1",
        "prerequisite_kernel": "renta0426/stage1-raw-fim-submission-v1",
        "materializer_path": MATERIALIZER_PATH,
        "materializer_blob_sha": MATERIALIZER_BLOB,
        "private_source_loader_path": PRIVATE_LOADER_PATH,
        "private_source_loader_blob_sha": PRIVATE_LOADER_BLOB,
        "automatic_compute_retries": 0,
        "enable_internet": True,
        "competition_submission": False,
        "runner_local_private_material_retention_days": 0,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise RuntimeError(f"request contract mismatch: {key}")
    if git_blob_sha(materializer) != MATERIALIZER_BLOB:
        raise RuntimeError("public materializer Git blob differs from request pin")
    if git_blob_sha(loader) != PRIVATE_LOADER_BLOB:
        raise RuntimeError("private-source loader Git blob differs from request pin")
    if request.get("resource") != {
        "accelerator": "gpu",
        "machine_shape": "NvidiaTeslaT4",
        "expected_runtime_minutes": 60,
        "hard_timeout_minutes": 180,
        "max_active_runs": 1,
        "min_remaining_quota_hours": 4.0,
    }:
        raise RuntimeError("resource contract changed")
    if request.get("api_budget") != {
        "max_calls": 20,
        "poll_interval_seconds": 300,
        "max_pages": 2,
    }:
        raise RuntimeError("API budget changed")
    if request.get("side_effects") != [
        "read four files from one private research commit",
        "create one private notebook version and start one T4 GPU run",
    ]:
        raise RuntimeError("side-effect allowlist changed")
    source = request.get("research_source") or {}
    if source.get("repository") != RESEARCH_REPOSITORY:
        raise RuntimeError("research repository changed")
    if source.get("commit") != RESEARCH_COMMIT:
        raise RuntimeError("research commit changed")
    observed = {
        str(item.get("path")): (
            str(item.get("git_blob_sha")),
            int(item.get("max_bytes")),
        )
        for item in source.get("files", [])
    }
    if observed != SOURCE_FILES:
        raise RuntimeError("research file allowlist or pin changed")
    allowed_keys = set(expected) | {
        "research_source",
        "resource",
        "api_budget",
        "side_effects",
    }
    unknown = set(request).difference(allowed_keys)
    if unknown:
        raise RuntimeError(f"unknown request fields: {sorted(unknown)}")


def validate_materializer(source: str) -> None:
    tree = ast.parse(source, filename=MATERIALIZER_PATH)
    constants = assigned_constants(tree)
    if literal_constant(constants, "EXPECTED_REQUEST_ID") != REQUEST_ID:
        raise RuntimeError("materializer request ID changed")
    if literal_constant(constants, "EXPECTED_TARGET") != "renta0426/mellum-transfer-v1":
        raise RuntimeError("materializer target changed")
    if literal_constant(constants, "EXPECTED_RESEARCH_REPOSITORY") != RESEARCH_REPOSITORY:
        raise RuntimeError("materializer research repository changed")
    if literal_constant(constants, "EXPECTED_RESEARCH_COMMIT") != RESEARCH_COMMIT:
        raise RuntimeError("materializer research commit changed")
    if literal_constant(constants, "EXPECTED_MODEL_ID") != "JetBrains/Mellum-4b-base":
        raise RuntimeError("materializer model changed")
    if literal_constant(constants, "EXPECTED_MODEL_REVISION") != (
        "83cce2605fbdf6a3868627e9b0a5924e0072b94d"
    ):
        raise RuntimeError("materializer model revision changed")
    if literal_constant(constants, "EXPECTED_ROWS") != 2_000:
        raise RuntimeError("materializer row count changed")
    if literal_constant(constants, "FORBIDDEN_RECORD_KEYS") != {
        "label", "membership", "is_member", "lumia_score"
    }:
        raise RuntimeError("materializer label guard changed")
    for marker in (
        "actual_blob = _git_blob_sha(data)",
        "if actual_blob != expected_blob:",
        "records = _prediction_records(notebook)",
        '"target_labels_embedded": False',
        '"competition_submission": False',
        '"target_labels_embedded_in_gpu_notebook": False',
        '"target_labels_used_for_training_or_normalization": False',
        '"previous_model_scores_used": False',
        '"submission_created": False',
        '"PATH": os.environ.get("PATH", "")',
        '"PYTHONHASHSEED": "0"',
    ):
        if marker not in source:
            raise RuntimeError(f"materializer invariant missing: {marker}")
    for forbidden in (
        "RESEARCH_REPO_READ_TOKEN",
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
        "competitions submit",
        "kernels push",
    ):
        if forbidden in source:
            raise RuntimeError(f"materializer gained forbidden capability: {forbidden}")


def validate_private_loader(source: str) -> None:
    tree = ast.parse(source, filename=PRIVATE_LOADER_PATH)
    constants = assigned_constants(tree)
    if literal_constant(constants, "TOKEN_ENV") != "RESEARCH_REPO_READ_TOKEN":
        raise RuntimeError("private loader token variable changed")
    if literal_constant(constants, "EXPECTED_REQUEST_ID") != REQUEST_ID:
        raise RuntimeError("private loader request ID changed")
    if literal_constant(constants, "EXPECTED_REPOSITORY") != RESEARCH_REPOSITORY:
        raise RuntimeError("private loader repository changed")
    if literal_constant(constants, "EXPECTED_COMMIT") != RESEARCH_COMMIT:
        raise RuntimeError("private loader commit changed")
    if literal_constant(constants, "EXPECTED_MATERIALIZER_PATH") != MATERIALIZER_PATH:
        raise RuntimeError("private loader materializer path changed")
    if literal_constant(constants, "EXPECTED_MATERIALIZER_BLOB") != MATERIALIZER_BLOB:
        raise RuntimeError("private loader materializer blob changed")
    if literal_constant(constants, "EXPECTED_LOADER_PATH") != PRIVATE_LOADER_PATH:
        raise RuntimeError("private loader self path changed")
    expected_paths = literal_constant(constants, "EXPECTED_FILES")
    if expected_paths != SOURCE_FILES:
        raise RuntimeError("private loader source allowlist or pin changed")

    required = (
        'token = os.environ.pop(TOKEN_ENV, "")',
        'or parsed.hostname != "raw.githubusercontent.com"',
        'parsed.hostname != "api.github.com"',
        '"Authorization": f"Bearer {token}"',
        '"X-GitHub-Api-Version": "2022-11-28"',
        'payload.get("encoding") != "base64"',
        "base64.b64decode(encoded, validate=True)",
        'payload.get("sha") != expected_blob',
        "module._validate_request = _validate_request_manifest",
        "module._download_raw = _authenticated_downloader(token, request_manifest)",
        '"repository_token_in_child_environment": False',
    )
    for marker in required:
        if marker not in source:
            raise RuntimeError(f"private loader invariant missing: {marker}")
    forbidden = (
        "subprocess.",
        "api.github.com/user",
        "api.github.com/orgs",
        "git clone",
        "print(token",
        "repr(token",
        "str(token",
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
        "competitions submit",
        "kernels push",
    )
    for marker in forbidden:
        if marker in source:
            raise RuntimeError(f"private loader gained forbidden capability: {marker}")


def step_blocks(job_source: str) -> list[str]:
    return re.split(r"(?m)^      - name: ", job_source)[1:]


def validate_launch(source: str) -> None:
    required = (
        "permissions: {}",
        "group: kaggle-resource-global",
        "cancel-in-progress: false",
        "environment: kaggle-readonry",
        "TARGET_KERNEL: renta0426/mellum-transfer-v1",
        "PREREQUISITE_KERNEL: renta0426/stage1-raw-fim-submission-v1",
        "REQUESTED_ACCELERATOR: gpu",
        "Read four private research files and materialize audited kernel",
        "RESEARCH_REPO_READ_TOKEN: ${{ secrets.RESEARCH_REPO_READ_TOKEN }}",
        "KAGGLE_API_TOKEN: ${{ secrets.KAGGLE_API_TOKEN }}",
        "--materializer \"${WORKDIR}/materializer.py\"",
        "MELLUM_PRIVATE_SOURCE_MATERIALIZED PASS",
        "MELLUM_RUNNER_LOCAL_BUNDLE PASS",
        "MELLUM_LIVE_PREFLIGHT PASS",
        "MELLUM_LAUNCH_DEFERRED",
        "MELLUM_KERNEL_LAUNCH PASS",
        "MELLUM_RUNNER_LOCAL_PRIVATE_MATERIAL_REMOVED",
        "automatic_retries=0 submission=false",
        "labels_embedded=false",
    )
    for marker in required:
        if marker not in source:
            raise RuntimeError(f"protected launch invariant missing: {marker}")

    if source.count('"${WORKDIR}/kaggle-venv/bin/kaggle" kernels push') != 1:
        raise RuntimeError("protected launch must contain one executable kernels push")
    if source.count("${{ secrets.RESEARCH_REPO_READ_TOKEN }}") != 1:
        raise RuntimeError("private repository token use count changed")
    if source.count("${{ secrets.KAGGLE_API_TOKEN }}") != 2:
        raise RuntimeError("Kaggle token use count changed")
    if source.count("environment: kaggle-readonry") != 1:
        raise RuntimeError("protected Environment use count changed")

    launch_marker = "\n  launch:\n"
    if launch_marker not in source:
        raise RuntimeError("protected launch job missing")
    before_launch, protected_job = source.split(launch_marker, 1)
    if "${{ secrets." in before_launch:
        raise RuntimeError("secret reference appears outside protected job")
    blocks = step_blocks(protected_job)
    research_blocks = [
        block for block in blocks if "secrets.RESEARCH_REPO_READ_TOKEN" in block
    ]
    kaggle_blocks = [block for block in blocks if "secrets.KAGGLE_API_TOKEN" in block]
    if len(research_blocks) != 1 or len(kaggle_blocks) != 2:
        raise RuntimeError("secret step partition changed")
    if "secrets.KAGGLE_API_TOKEN" in research_blocks[0]:
        raise RuntimeError("private source and Kaggle credentials share a step")
    if any("secrets.RESEARCH_REPO_READ_TOKEN" in block for block in kaggle_blocks):
        raise RuntimeError("Kaggle and private-source credentials share a step")
    if "private-loader.py" not in research_blocks[0]:
        raise RuntimeError("private-source token is not bound to the loader step")
    if any("materializer.py" in block for block in kaggle_blocks):
        raise RuntimeError("research materialization appears in a Kaggle-token step")

    forbidden = (
        "actions/upload-artifact",
        "actions/download-artifact",
        "actions/cache",
        "actions/checkout",
        "workflow_dispatch",
        "continue-on-error: true",
        '"${WORKDIR}/kaggle-venv/bin/kaggle" competitions submit',
        '"${WORKDIR}/kaggle-venv/bin/kaggle" kernels delete',
        '"${WORKDIR}/kaggle-venv/bin/kaggle" kernels cancel',
        '"${WORKDIR}/kaggle-venv/bin/kaggle" datasets create',
        '"${WORKDIR}/kaggle-venv/bin/kaggle" models',
    )
    for marker in forbidden:
        if marker in source:
            raise RuntimeError(f"forbidden launch capability present: {marker}")
    if re.search(r"\bwhile\s+true\b|\bfor\s+\(\(\s*;\s*;", source):
        raise RuntimeError("unbounded loop found in protected launch")
    if re.search(r"^\s*sleep\s+", source, flags=re.MULTILINE):
        raise RuntimeError("protected launch must not poll")


def validate(root: Path) -> dict:
    for name in (
        "KAGGLE_API_TOKEN",
        "RESEARCH_REPO_READ_TOKEN",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
    ):
        if os.environ.get(name):
            raise RuntimeError(f"static validation received unexpected credential: {name}")

    request_path = root / "requests/poisoned-chalice-mellum-transfer-v1-launch.json"
    materializer_path = root / MATERIALIZER_PATH
    loader_path = root / PRIVATE_LOADER_PATH
    launch_path = root / ".github/workflows/110-poisoned-chalice-mellum-transfer-v1-launch.yml"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    materializer = materializer_path.read_bytes()
    loader = loader_path.read_bytes()
    validate_request(request, materializer, loader)
    validate_materializer(materializer.decode("utf-8"))
    validate_private_loader(loader.decode("utf-8"))
    validate_launch(launch_path.read_text(encoding="utf-8"))

    result = {
        "status": "pass",
        "request_id": REQUEST_ID,
        "research_repository": RESEARCH_REPOSITORY,
        "research_commit": RESEARCH_COMMIT,
        "research_files": len(SOURCE_FILES),
        "materializer_blob": MATERIALIZER_BLOB,
        "private_loader_blob": PRIVATE_LOADER_BLOB,
        "private_source_token_steps": 1,
        "kaggle_token_steps": 2,
        "shared_secret_steps": 0,
        "kaggle_push_calls": 1,
        "competition_submission": False,
        "public_artifacts": 0,
        "automatic_compute_retries": 0,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    validate(args.root.resolve())


if __name__ == "__main__":
    main()
