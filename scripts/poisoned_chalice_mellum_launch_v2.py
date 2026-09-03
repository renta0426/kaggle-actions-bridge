"""Request-specific Mellum-4B Kaggle launcher with fail-closed guardrails.

Request 001 performed no Kaggle write because its quota parser could not interpret
Kaggle SDK accelerator-quota statistics. This v2 path uses the Kaggle CLI 2.2.4
``quota_view()`` contract that already succeeded in the Stage1 bridge: remaining
GPU time is ``gpu_quota.total_time_allowed - gpu_quota.time_used``.

The only authorized write is one private ``kernels push``. There is no automatic
compute retry and no Competition submission capability.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


EXPECTED_REQUEST_ID = "20260903-poisoned-chalice-mellum-transfer-v1-002"
EXPECTED_PARENT_REQUEST_ID = "20260903-poisoned-chalice-mellum-transfer-v1-001"
EXPECTED_COMPETITION = "poisoned-chalice-icse27"
EXPECTED_TARGET = "renta0426/mellum-transfer-v1"
EXPECTED_SOURCE = "renta0426/pseudo-stage2-starcoder2-7b-v1"
EXPECTED_SOURCE_VERSION = 1
EXPECTED_BUILDER_PATH = "scripts/poisoned_chalice_mellum_transfer_builder.py"
EXPECTED_BUILDER_BLOB = "0b8b48b2547454affbde6e730d4e43bde39421cf"
EXPECTED_SHIM_PATH = "scripts/poisoned_chalice_nbformat_minimal.py"
EXPECTED_SHIM_BLOB = "9e91545a7f3318ca7e033c4f49f0eeda64ce4bfd"
EXPECTED_WORKFLOW_PATH = ".github/workflows/121-poisoned-chalice-mellum-transfer-v1-launch-v2.yml"
EXPECTED_SCORE = 0.34575
EXPECTED_MODEL_ID = "JetBrains/Mellum-4b-base"
EXPECTED_MODEL_REVISION = "83cce2605fbdf6a3868627e9b0a5924e0072b94d"
EXPECTED_RESEARCH_COMMIT = "9f6be68c27dc1f3326d68bed4e2abf80db893748"
EXPECTED_UPSTREAM_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
EXPECTED_UPSTREAM_SHA256 = "6e8559279e8ebd94ec8c25561c91d38c454a4e109ea0813b07864ff6a66d4068"
EXPECTED_ROWS = 2000
MIN_GPU_HOURS = 4.0
MAX_RECENT_KERNELS = 25


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def load_request(path: Path) -> dict:
    request = json.loads(path.read_text(encoding="utf-8"))
    fixed = {
        "schema_version": 1,
        "request_id": EXPECTED_REQUEST_ID,
        "parent_request_id": EXPECTED_PARENT_REQUEST_ID,
        "parent_outcome": "preflight_failed_before_kaggle_write",
        "competition": EXPECTED_COMPETITION,
        "operation": "kernel_run",
        "target": EXPECTED_TARGET,
        "source_kernel": EXPECTED_SOURCE,
        "source_expected_version": EXPECTED_SOURCE_VERSION,
        "launcher_path": "scripts/poisoned_chalice_mellum_launch_v2.py",
        "builder_path": EXPECTED_BUILDER_PATH,
        "builder_blob_sha": EXPECTED_BUILDER_BLOB,
        "nbformat_shim_path": EXPECTED_SHIM_PATH,
        "nbformat_shim_blob_sha": EXPECTED_SHIM_BLOB,
        "canonical_research_commit": EXPECTED_RESEARCH_COMMIT,
        "stage1_prerequisite": {
            "submission_status": "complete",
            "public_score": EXPECTED_SCORE,
            "score_source": "read_only_submission_history_gate",
        },
        "target_model": {
            "id": EXPECTED_MODEL_ID,
            "revision": EXPECTED_MODEL_REVISION,
        },
        "source_data": {
            "repository": "https://github.com/Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026",
            "commit": EXPECTED_UPSTREAM_COMMIT,
            "path": "data/7b_train_test/eval_results.parquet",
            "sha256": EXPECTED_UPSTREAM_SHA256,
            "rows": EXPECTED_ROWS,
            "current_stage1_overlap_excluded": 23,
        },
        "resource": {
            "accelerator": "gpu",
            "machine_shape": "NvidiaTeslaT4",
            "expected_runtime_minutes": 120,
            "hard_timeout_minutes": 180,
            "min_remaining_quota_hours": MIN_GPU_HOURS,
            "max_active_account_runs": 0,
        },
        "api_budget": {
            "max_calls": 60,
            "max_recent_kernels_inspected": MAX_RECENT_KERNELS,
            "max_pages": 2,
        },
        "side_effects": [
            "create one private Mellum transfer Notebook version and start one T4 GPU run"
        ],
        "automatic_compute_retries": 0,
        "enable_internet": True,
        "competition_submission": False,
        "select_as_final": False,
        "runner_local_private_material_retention_days": 0,
        "clean_room": {
            "target_labels_embedded": False,
            "target_labels_used_for_training_or_normalization": False,
            "previous_model_scores_used": False,
            "hidden_stage1_validation_labels_used": False,
            "public_leaderboard_tuning_used": False,
            "competition_submission_created": False,
        },
    }
    if set(request) != set(fixed) | {"launcher_blob_sha"}:
        raise RuntimeError("Mellum v2 request fields differ from exact contract")
    for key, value in fixed.items():
        if request.get(key) != value:
            raise RuntimeError(f"Mellum v2 request contract mismatch: {key}")
    launcher_blob = str(request.get("launcher_blob_sha") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", launcher_blob):
        raise RuntimeError("Mellum v2 launcher Git blob pin is malformed")
    return request


def _literal_subprocess_commands(source: str) -> list[tuple[str, ...]]:
    tree = ast.parse(source)
    commands: list[tuple[str, ...]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr == "run"
            and node.args
            and isinstance(node.args[0], (ast.List, ast.Tuple))
        ):
            continue
        values: list[str] = []
        for element in node.args[0].elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                values.append(element.value)
            else:
                values.append("<dynamic>")
        commands.append(tuple(values))
    return commands


def _contains_pair(command: tuple[str, ...], left: str, right: str) -> bool:
    return any(
        command[index : index + 2] == (left, right)
        for index in range(max(0, len(command) - 1))
    )


def validate_local_sources(
    request: dict,
    request_path: Path,
    builder_path: Path,
    shim_path: Path,
    workflow_path: Path,
) -> None:
    if git_blob_sha(Path(__file__).read_bytes()) != request["launcher_blob_sha"]:
        raise RuntimeError("Mellum v2 launcher differs from request Git blob pin")
    for path, expected in (
        (builder_path, EXPECTED_BUILDER_BLOB),
        (shim_path, EXPECTED_SHIM_BLOB),
    ):
        data = path.read_bytes()
        if git_blob_sha(data) != expected:
            raise RuntimeError(f"Git blob mismatch: {path.name}")
        compile(data, str(path), "exec")
    if request_path.stat().st_size > 32_768:
        raise RuntimeError("Mellum v2 request exceeds byte budget")

    workflow = workflow_path.read_text(encoding="utf-8")
    required = (
        "permissions: {}",
        "group: kaggle-resource-global",
        "cancel-in-progress: false",
        "environment: kaggle-readonry",
        "runs-on: ubuntu-24.04",
        "requests/poisoned-chalice-mellum-transfer-v1-launch-v2.json",
        "scripts/poisoned_chalice_mellum_launch_v2.py",
        "KAGGLE_API_TOKEN: ${{ secrets.KAGGLE_API_TOKEN }}",
        "MELLUM_V2_STATIC PASS",
        "MELLUM_V2_EXECUTION",
    )
    missing = [value for value in required if value not in workflow]
    if missing:
        raise RuntimeError(f"Mellum v2 workflow security markers missing: {missing}")
    forbidden = (
        "pull_request_target:",
        "issue_comment:",
        "repository_dispatch:",
        "runs-on: self-hosted",
        "actions/checkout@",
        "workflow_dispatch:",
        "continue-on-error: true",
        "competitions submit",
        "datasets create",
        "datasets version",
        "kernels delete",
        "kernels cancel",
        "--public",
    )
    present = [value for value in forbidden if value in workflow]
    if present:
        raise RuntimeError(f"Mellum v2 workflow has forbidden operations: {present}")
    if workflow.count("${{ secrets.KAGGLE_API_TOKEN }}") != 1:
        raise RuntimeError("Mellum v2 Kaggle token must be scoped to exactly one step")

    own = Path(__file__).read_text(encoding="utf-8")
    commands = _literal_subprocess_commands(own)
    push_count = sum(_contains_pair(command, "kernels", "push") for command in commands)
    if push_count != 1:
        raise RuntimeError("Mellum v2 launcher must contain exactly one kernels push")
    for left, right in (
        ("competitions", "submit"),
        ("datasets", "create"),
        ("datasets", "version"),
        ("kernels", "delete"),
        ("kernels", "cancel"),
    ):
        if any(_contains_pair(command, left, right) for command in commands):
            raise RuntimeError(f"Mellum v2 launcher has forbidden write: {left} {right}")


def plain(text: str) -> str:
    normalized = text.replace(r"\%", "%")
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", normalized)).casefold()


def read_value(item, *names):
    data = item.to_dict() if hasattr(item, "to_dict") else (
        item if isinstance(item, dict) else {}
    )
    for name in names:
        candidate = getattr(item, name, None)
        if candidate is None:
            candidate = data.get(name)
        if candidate is not None:
            return candidate
    return None


def live_preflight(request: dict):  # noqa: ANN201
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    api = KaggleApi()
    api.authenticate()

    pages = api.competition_list_pages(EXPECTED_COMPETITION) or []
    content: dict[str, str] = {}
    for page in pages:
        data = page.to_dict() if hasattr(page, "to_dict") else dict(page)
        name = str(data.get("name") or "").strip().lower()
        if name:
            content[name] = str(data.get("content") or "")
    if "rules" not in content or "evaluation" not in content:
        raise RuntimeError("live Competition rules/evaluation unavailable")
    if not any("data" in name for name in content):
        raise RuntimeError("live Competition data page unavailable")
    evaluation = plain(content["evaluation"])
    if not all(term in evaluation for term in ("auc", "novelty", "1%", "false-positive")):
        raise RuntimeError("live Competition evaluation contract changed")
    rules = plain(content["rules"])
    for phrase in (
        "external data is not allowed",
        "external data are not allowed",
        "internet access is prohibited",
        "internet must be disabled",
        "pretrained models are not allowed",
        "pre-trained models are not allowed",
        "gpu use is prohibited",
    ):
        if phrase in rules:
            raise RuntimeError(f"live Competition rule conflict: {phrase}")

    try:
        submissions = api.competition_submissions(EXPECTED_COMPETITION, page_size=100) or []
    except TypeError:
        submissions = api.competition_submissions(EXPECTED_COMPETITION) or []
    score_match = False
    for submission in submissions:
        status = str(read_value(submission, "status") or "").upper()
        try:
            score = float(read_value(submission, "public_score", "publicScore"))
        except (TypeError, ValueError):
            continue
        if "COMPLETE" in status and math.isclose(
            score, EXPECTED_SCORE, rel_tol=0.0, abs_tol=5e-6
        ):
            score_match = True
            break
    if not score_match:
        raise RuntimeError("Stage1 Public score 0.34575 not found in current submission history")

    existing = api.kernels_list(user="renta0426", search="mellum-transfer-v1", page_size=20) or []
    if EXPECTED_TARGET in {str(getattr(item, "ref", "")) for item in existing}:
        raise RuntimeError("target Mellum kernel already exists; duplicate launch refused")

    source_status = str(
        getattr(api.kernels_status(EXPECTED_SOURCE), "status", "")
    ).upper()
    if "COMPLETE" not in source_status:
        raise RuntimeError(f"source kernel is not complete: {source_status}")
    owner, slug = EXPECTED_SOURCE.split("/", 1)
    with api.build_kaggle_client() as client:
        detail_request = ApiGetKernelRequest()
        detail_request.user_name = owner
        detail_request.kernel_slug = slug
        details = client.kernels.kernels_api_client.get_kernel(detail_request)
    metadata = details.metadata
    if int(metadata.current_version_number) != EXPECTED_SOURCE_VERSION:
        raise RuntimeError("source kernel version changed")
    if not bool(metadata.is_private):
        raise RuntimeError("source kernel unexpectedly public")

    quota = api.quota_view()
    gpu = getattr(quota, "gpu_quota", None)
    used = getattr(gpu, "time_used", None) if gpu is not None else None
    total = getattr(gpu, "total_time_allowed", None) if gpu is not None else None
    if used is None or total is None:
        raise RuntimeError("GPU quota information unavailable from quota_view")
    remaining_hours = max(0.0, (total - used).total_seconds() / 3600.0)
    if remaining_hours < MIN_GPU_HOURS:
        raise RuntimeError(
            f"insufficient GPU quota: remaining={remaining_hours:.2f}h "
            f"required={MIN_GPU_HOURS:.2f}h"
        )

    active: list[str] = []
    uncertain: list[str] = []
    recent = api.kernels_list(
        user="renta0426", sort_by="dateRun", page_size=MAX_RECENT_KERNELS
    ) or []
    for item in recent[:MAX_RECENT_KERNELS]:
        ref = str(getattr(item, "ref", ""))
        if not ref:
            continue
        try:
            status = str(getattr(api.kernels_status(ref), "status", "")).upper()
        except Exception:
            uncertain.append(ref)
            continue
        if any(token in status for token in ("RUNNING", "QUEUED", "PENDING")):
            active.append(ref)
    if active or uncertain:
        raise RuntimeError(
            "account-wide Kaggle admission refused: "
            f"active={len(active)} uncertain={len(uncertain)}"
        )
    print(
        "MELLUM_V2_LIVE_PREFLIGHT PASS "
        f"score={EXPECTED_SCORE:.5f} source_version={EXPECTED_SOURCE_VERSION} "
        f"gpu_quota_hours={remaining_hours:.2f} active_account_runs=0"
    )
    return api


def validate_generated_bundle(kernel_dir: Path) -> str:
    metadata_path = kernel_dir / "kernel-metadata.json"
    notebook_path = kernel_dir / "mellum-transfer-v1.ipynb"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "id": EXPECTED_TARGET,
        "code_file": notebook_path.name,
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"generated metadata mismatch: {key}")
    if not notebook_path.is_file() or notebook_path.is_symlink():
        raise RuntimeError("generated Mellum notebook missing or symbolic")
    raw = notebook_path.read_bytes()
    if len(raw) > 5_000_000:
        raise RuntimeError("generated Mellum notebook exceeds byte budget")
    notebook = json.loads(raw)
    if len(notebook.get("cells", [])) != 9:
        raise RuntimeError("generated Mellum notebook cell count changed")
    joined = "\n".join(
        "".join(cell.get("source", ""))
        if isinstance(cell.get("source", ""), list)
        else str(cell.get("source", ""))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    for marker in (
        EXPECTED_MODEL_ID,
        EXPECTED_MODEL_REVISION,
        "PREDICTION_MANIFEST = pd.DataFrame(",
        "min_kpp_zselect_10__max",
        "best_local_64__max",
        '"target_labels_embedded_in_gpu_notebook": False',
        '"target_labels_used_for_training_or_normalization": False',
        '"previous_model_scores_used": False',
        '"public_leaderboard_tuning_used": False',
    ):
        if marker not in joined:
            raise RuntimeError(f"generated scientific marker missing: {marker}")
    for forbidden in ("KAGGLE_API_TOKEN", "competitions submit"):
        if forbidden in joined:
            raise RuntimeError(f"forbidden generated capability: {forbidden}")
    digest = hashlib.sha256(raw).hexdigest()
    print(
        f"MELLUM_V2_GENERATED_BUNDLE PASS bytes={len(raw)} "
        f"sha256={digest} labels_embedded=false"
    )
    return digest


def execute(request: dict, workdir: Path, kaggle_bin: Path, builder: Path, shim: Path) -> None:
    api = live_preflight(request)
    source_dir = workdir / "source-kernel"
    kernel_dir = workdir / "mellum-kernel"
    tool_dir = workdir / "tool"
    workdir.mkdir(parents=True, exist_ok=False)
    source_dir.mkdir(parents=True, exist_ok=False)
    kernel_dir.mkdir(parents=True, exist_ok=False)
    tool_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(builder, tool_dir / "builder.py")
    shutil.copy2(shim, tool_dir / "nbformat.py")

    pull = subprocess.run(
        [str(kaggle_bin), "kernels", "pull", EXPECTED_SOURCE, "-p", str(source_dir), "-m"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if pull.returncode != 0:
        raise RuntimeError("exact private source-kernel pull failed; response suppressed")
    source_notebooks = list(source_dir.glob("*.ipynb"))
    if len(source_notebooks) != 1 or not (source_dir / "kernel-metadata.json").is_file():
        raise RuntimeError("private source-kernel materialization contract changed")

    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(tool_dir)}
    built = subprocess.run(
        [
            sys.executable,
            str(tool_dir / "builder.py"),
            "--source-notebook",
            str(source_notebooks[0]),
            "--output-dir",
            str(kernel_dir),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if built.returncode != 0:
        raise RuntimeError("Mellum label-free notebook materialization failed; response suppressed")
    digest = validate_generated_bundle(kernel_dir)

    pushed = subprocess.run(
        [str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir), "--timeout", "10800"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if pushed.returncode != 0:
        raise RuntimeError("Kaggle Mellum kernel push failed; response suppressed")

    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    with api.build_kaggle_client() as client:
        detail_request = ApiGetKernelRequest()
        detail_request.user_name = "renta0426"
        detail_request.kernel_slug = "mellum-transfer-v1"
        details = client.kernels.kernels_api_client.get_kernel(detail_request)
    metadata = details.metadata
    if int(metadata.current_version_number) != 1:
        raise RuntimeError("unexpected Mellum kernel version after push")
    if not bool(metadata.is_private) or not bool(metadata.enable_gpu) or bool(metadata.enable_tpu):
        raise RuntimeError("Mellum privacy/accelerator contract changed after push")
    if not bool(metadata.enable_internet):
        raise RuntimeError("Mellum Internet contract changed after push")
    print(
        "MELLUM_V2_EXECUTION PASS "
        f"request_id={EXPECTED_REQUEST_ID} target={EXPECTED_TARGET} version=1 "
        f"notebook_sha256={digest} accelerator=gpu machine=NvidiaTeslaT4 "
        "automatic_compute_retries=0 competition_submission=false"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--shim", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--kaggle-bin", type=Path)
    args = parser.parse_args()
    if args.static == args.execute:
        raise SystemExit("choose exactly one of --static or --execute")
    request = load_request(args.request)
    validate_local_sources(request, args.request, args.builder, args.shim, args.workflow)
    if args.static:
        print(
            "MELLUM_V2_STATIC PASS request=002 quota=quota_view rows=2000 "
            "labels_embedded=false write_calls=1 automatic_retries=0 submissions=0"
        )
        return
    if args.workdir is None or args.kaggle_bin is None:
        raise SystemExit("--execute requires --workdir and --kaggle-bin")
    execute(request, args.workdir, args.kaggle_bin, args.builder, args.shim)


if __name__ == "__main__":
    main()
