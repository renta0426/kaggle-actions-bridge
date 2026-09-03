"""Exact-once Mellum-4B transfer launcher for fresh request 003.

Request 002 reached authenticated live preflight successfully, observed adequate GPU
quota and zero active Kaggle sessions, then stopped before any Kaggle write because
the bridge builder compared a generated Min-K++ feature name against source text.
Request 003 keeps the frozen experiment unchanged, applies only that audited builder
contract correction, and separates GPU admission from unrelated CPU/TPU work.
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

REQUEST_ID = "20260903-poisoned-chalice-mellum-transfer-v1-003"
PARENT_ID = "20260903-poisoned-chalice-mellum-transfer-v1-002"
COMPETITION = "poisoned-chalice-icse27"
TARGET = "renta0426/mellum-transfer-v1"
SOURCE = "renta0426/pseudo-stage2-starcoder2-7b-v1"
SOURCE_VERSION = 1
LAUNCHER_PATH = "scripts/poisoned_chalice_mellum_launch_v3.py"
BUILDER_PATH = "scripts/poisoned_chalice_mellum_transfer_builder_v2.py"
BUILDER_BLOB = "75832c86c1dfe7db0aae01b29c03a8fd0ad6ffbe"
BASE_BUILDER_PATH = "scripts/poisoned_chalice_mellum_transfer_builder.py"
BASE_BUILDER_BLOB = "0b8b48b2547454affbde6e730d4e43bde39421cf"
SHIM_PATH = "scripts/poisoned_chalice_nbformat_minimal.py"
SHIM_BLOB = "9e91545a7f3318ca7e033c4f49f0eeda64ce4bfd"
WORKFLOW_PATH = ".github/workflows/122-poisoned-chalice-mellum-transfer-v1-launch-v3.yml"
MODEL_ID = "JetBrains/Mellum-4b-base"
MODEL_REVISION = "83cce2605fbdf6a3868627e9b0a5924e0072b94d"
RESEARCH_COMMIT = "9f6be68c27dc1f3326d68bed4e2abf80db893748"
UPSTREAM_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
UPSTREAM_SHA256 = "6e8559279e8ebd94ec8c25561c91d38c454a4e109ea0813b07864ff6a66d4068"
PUBLIC_SCORE = 0.34575
ROWS = 2000
MIN_GPU_HOURS = 4.0
MAX_RECENT = 25
ACTIVE_STATES = ("RUNNING", "QUEUED", "PENDING")


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def expected_request() -> dict:
    return {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "parent_request_id": PARENT_ID,
        "parent_outcome": "builder_failed_before_kaggle_write",
        "competition": COMPETITION,
        "operation": "kernel_run",
        "target": TARGET,
        "source_kernel": SOURCE,
        "source_expected_version": SOURCE_VERSION,
        "launcher_path": LAUNCHER_PATH,
        "builder_path": BUILDER_PATH,
        "builder_blob_sha": BUILDER_BLOB,
        "base_builder_path": BASE_BUILDER_PATH,
        "base_builder_blob_sha": BASE_BUILDER_BLOB,
        "nbformat_shim_path": SHIM_PATH,
        "nbformat_shim_blob_sha": SHIM_BLOB,
        "canonical_research_commit": RESEARCH_COMMIT,
        "stage1_prerequisite": {
            "submission_status": "complete",
            "public_score": PUBLIC_SCORE,
            "score_source": "read_only_submission_history_gate",
        },
        "target_model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "source_data": {
            "repository": "https://github.com/Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026",
            "commit": UPSTREAM_COMMIT,
            "path": "data/7b_train_test/eval_results.parquet",
            "sha256": UPSTREAM_SHA256,
            "rows": ROWS,
            "current_stage1_overlap_excluded": 23,
        },
        "resource": {
            "accelerator": "gpu",
            "machine_shape": "NvidiaTeslaT4",
            "expected_runtime_minutes": 120,
            "hard_timeout_minutes": 180,
            "min_remaining_quota_hours": MIN_GPU_HOURS,
            "max_active_same_resource_runs": 0,
            "parallel_resource_classes_allowed": ["cpu", "tpu"],
            "unknown_active_resource_policy": "fail_closed",
        },
        "api_budget": {
            "max_calls": 90,
            "max_recent_kernels_inspected": MAX_RECENT,
            "max_pages": 2,
        },
        "failure_fix": {
            "request_002_write_occurred": False,
            "builder_contract_fix": "accept literal f-string template min_kpp_zselect_{percent:02d} instead of generated min_kpp_zselect_10",
            "github_concurrency": "kaggle-resource-gpu-global",
            "live_admission": "block active GPU; allow classified CPU/TPU; fail closed on unknown active resource",
            "scientific_configuration_changed": False,
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


def load_request(path: Path) -> dict:
    request = json.loads(path.read_text(encoding="utf-8"))
    fixed = expected_request()
    if set(request) != set(fixed) | {"launcher_blob_sha"}:
        raise RuntimeError("request 003 fields differ from exact contract")
    for key, expected in fixed.items():
        if request.get(key) != expected:
            raise RuntimeError(f"request 003 contract mismatch: {key}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(request.get("launcher_blob_sha", ""))):
        raise RuntimeError("launcher Git blob pin is malformed")
    return request


def literal_commands(source: str) -> list[tuple[str, ...]]:
    commands = []
    for node in ast.walk(ast.parse(source)):
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
        values = []
        for item in node.args[0].elts:
            values.append(
                item.value
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
                else "<dynamic>"
            )
        commands.append(tuple(values))
    return commands


def has_pair(command: tuple[str, ...], left: str, right: str) -> bool:
    return any(command[index : index + 2] == (left, right) for index in range(len(command) - 1))


def validate_static(
    request: dict,
    request_path: Path,
    builder: Path,
    base_builder: Path,
    shim: Path,
    workflow: Path,
) -> None:
    if blob_sha(Path(__file__).read_bytes()) != request["launcher_blob_sha"]:
        raise RuntimeError("launcher differs from request Git blob pin")
    for path, expected in (
        (builder, BUILDER_BLOB),
        (base_builder, BASE_BUILDER_BLOB),
        (shim, SHIM_BLOB),
    ):
        data = path.read_bytes()
        if blob_sha(data) != expected:
            raise RuntimeError(f"source Git blob mismatch: {path.name}")
        compile(data, str(path), "exec")
    builder_text = builder.read_text(encoding="utf-8")
    if (
        'OLD_CHECK = \'"standard_minkpp": "min_kpp_zselect_10" in feature_cache,\'' not in builder_text
        or 'NEW_CHECK = \'"standard_minkpp": "min_kpp_zselect_{percent:02d}" in feature_cache,\'' not in builder_text
        or "BASE_BUILDER_BLOB = \"0b8b48b2547454affbde6e730d4e43bde39421cf\"" not in builder_text
    ):
        raise RuntimeError("audited builder correction markers missing")
    if request_path.stat().st_size > 32768:
        raise RuntimeError("request exceeds byte budget")

    text = workflow.read_text(encoding="utf-8")
    required = (
        "permissions: {}",
        "group: kaggle-resource-gpu-global",
        "cancel-in-progress: false",
        "environment: kaggle-readonry",
        "runs-on: ubuntu-24.04",
        "requests/poisoned-chalice-mellum-transfer-v1-launch-v3.json",
        LAUNCHER_PATH,
        BUILDER_PATH,
        BASE_BUILDER_PATH,
        "KAGGLE_API_TOKEN: ${{ secrets.KAGGLE_API_TOKEN }}",
        "--static",
        "--execute",
        "--base-builder",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"workflow security markers missing: {missing}")
    for forbidden in (
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
        "group: kaggle-resource-global\n",
    ):
        if forbidden in text:
            raise RuntimeError(f"workflow gained forbidden capability: {forbidden}")
    if text.count("${{ secrets.KAGGLE_API_TOKEN }}") != 1:
        raise RuntimeError("Kaggle token must be scoped to exactly one protected step")

    commands = literal_commands(Path(__file__).read_text(encoding="utf-8"))
    if sum(has_pair(command, "kernels", "push") for command in commands) != 1:
        raise RuntimeError("launcher must contain exactly one kernels push")
    for pair in (
        ("competitions", "submit"),
        ("datasets", "create"),
        ("datasets", "version"),
        ("kernels", "delete"),
        ("kernels", "cancel"),
    ):
        if any(has_pair(command, *pair) for command in commands):
            raise RuntimeError(f"launcher gained forbidden write: {' '.join(pair)}")


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text.replace(r"\%", "%"))).casefold()


def value(item, *names):
    data = item.to_dict() if hasattr(item, "to_dict") else (item if isinstance(item, dict) else {})
    for name in names:
        candidate = getattr(item, name, None)
        if candidate is None:
            candidate = data.get(name)
        if candidate is not None:
            return candidate
    return None


def bool_value(item, *names):
    candidate = value(item, *names)
    if isinstance(candidate, bool):
        return candidate
    if candidate is None:
        return None
    text = str(candidate).strip().casefold()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def get_kernel_details(api, ref: str):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    owner, slug = ref.split("/", 1)
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name = owner
        request.kernel_slug = slug
        return client.kernels.kernels_api_client.get_kernel(request)


def classify_kernel_resource(metadata) -> str:
    gpu = bool_value(metadata, "enable_gpu", "enableGpu")
    tpu = bool_value(metadata, "enable_tpu", "enableTpu")
    machine = str(value(metadata, "machine_shape", "machineShape") or "").strip()
    folded = machine.casefold()
    if tpu is True or "tpu" in folded:
        if gpu is True:
            raise RuntimeError("active kernel metadata enables both GPU and TPU")
        return "tpu"
    if gpu is True or any(token in folded for token in ("nvidia", "tesla", "a100", "h100", "rtx", "l4")):
        return "gpu"
    if gpu is False and tpu is False:
        return "cpu"
    raise RuntimeError("active kernel resource class is unavailable")


def resource_admission(api) -> tuple[int, int]:
    active_gpu = []
    active_other = []
    uncertain = []
    items = (api.kernels_list(user="renta0426", sort_by="dateRun", page_size=MAX_RECENT) or [])[:MAX_RECENT]
    for item in items:
        ref = str(getattr(item, "ref", ""))
        if not ref:
            continue
        try:
            state = str(getattr(api.kernels_status(ref), "status", "")).upper()
        except Exception:
            uncertain.append(ref)
            continue
        if not any(token in state for token in ACTIVE_STATES):
            continue
        try:
            resource = classify_kernel_resource(get_kernel_details(api, ref).metadata)
        except Exception:
            uncertain.append(ref)
            continue
        if resource == "gpu":
            active_gpu.append(ref)
        else:
            active_other.append((ref, resource))
    if active_gpu or uncertain:
        raise RuntimeError(
            "GPU admission refused: "
            f"active_gpu={len(active_gpu)} unknown_active={len(uncertain)} "
            f"allowed_other={len(active_other)}"
        )
    return len(active_other), len(items)


def live_preflight():  # noqa: ANN201
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    pages = api.competition_list_pages(COMPETITION) or []
    content = {}
    for page in pages:
        data = page.to_dict() if hasattr(page, "to_dict") else dict(page)
        name = str(data.get("name") or "").strip().lower()
        if name:
            content[name] = str(data.get("content") or "")
    if "rules" not in content or "evaluation" not in content or not any("data" in name for name in content):
        raise RuntimeError("live Competition policy pages unavailable")
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
        submissions = api.competition_submissions(COMPETITION, page_size=100) or []
    except TypeError:
        submissions = api.competition_submissions(COMPETITION) or []
    matched = False
    for item in submissions:
        try:
            score = float(value(item, "public_score", "publicScore"))
        except (TypeError, ValueError):
            continue
        if (
            "COMPLETE" in str(value(item, "status") or "").upper()
            and math.isclose(score, PUBLIC_SCORE, rel_tol=0.0, abs_tol=5e-6)
        ):
            matched = True
            break
    if not matched:
        raise RuntimeError("Stage1 Public score 0.34575 not found in current submission history")

    existing = {
        str(getattr(item, "ref", ""))
        for item in (api.kernels_list(user="renta0426", search="mellum-transfer-v1", page_size=20) or [])
    }
    if TARGET in existing:
        raise RuntimeError("target Mellum kernel already exists")
    if "COMPLETE" not in str(getattr(api.kernels_status(SOURCE), "status", "")).upper():
        raise RuntimeError("source kernel is not complete")
    source_meta = get_kernel_details(api, SOURCE).metadata
    if int(source_meta.current_version_number) != SOURCE_VERSION or not bool(source_meta.is_private):
        raise RuntimeError("source kernel version/privacy changed")

    quota = api.quota_view()
    gpu = getattr(quota, "gpu_quota", None)
    used = getattr(gpu, "time_used", None) if gpu is not None else None
    total = getattr(gpu, "total_time_allowed", None) if gpu is not None else None
    if used is None or total is None:
        raise RuntimeError("GPU quota unavailable from quota_view")
    remaining = max(0.0, (total - used).total_seconds() / 3600.0)
    if remaining < MIN_GPU_HOURS:
        raise RuntimeError(
            f"insufficient GPU quota: remaining={remaining:.2f}h required={MIN_GPU_HOURS:.2f}h"
        )

    allowed_other, inspected = resource_admission(api)
    print(
        f"MELLUM_V3_LIVE_PREFLIGHT PASS score={PUBLIC_SCORE:.5f} source_version=1 "
        f"gpu_quota_hours={remaining:.2f} active_gpu=0 allowed_cpu_tpu={allowed_other} "
        f"recent_inspected={inspected}"
    )
    return api


def validate_bundle(root: Path) -> str:
    metadata = json.loads((root / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook_path = root / "mellum-transfer-v1.ipynb"
    expected = {
        "id": TARGET,
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
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise RuntimeError(f"generated metadata mismatch: {key}")
    raw = notebook_path.read_bytes()
    if len(raw) > 5_000_000:
        raise RuntimeError("generated notebook exceeds byte budget")
    notebook = json.loads(raw)
    if len(notebook.get("cells", [])) != 9:
        raise RuntimeError("generated notebook cell count changed")
    records = None
    joined_parts = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else str(source)
        joined_parts.append(source)
        if "PREDICTION_MANIFEST = pd.DataFrame(" in source:
            for node in ast.parse(source).body:
                if (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "PREDICTION_MANIFEST"
                        for target in node.targets
                    )
                ):
                    records = ast.literal_eval(node.value.args[0])
    allowed = {"sample_id", "content_sha256", "language", "sample_index"}
    if not isinstance(records, list) or len(records) != ROWS or any(set(row) != allowed for row in records):
        raise RuntimeError("label or unexpected field crossed GPU prediction boundary")
    if [row["sample_index"] for row in records] != list(range(ROWS)):
        raise RuntimeError("embedded prediction order changed")
    joined = "\n".join(joined_parts)
    for marker in (
        MODEL_ID,
        MODEL_REVISION,
        "min_kpp_zselect_10__max",
        "best_local_64__max",
        '"target_labels_embedded_in_gpu_notebook": False',
        '"target_labels_used_for_training_or_normalization": False',
        '"previous_model_scores_used": False',
        '"public_leaderboard_tuning_used": False',
    ):
        if marker not in joined:
            raise RuntimeError(f"generated scientific marker missing: {marker}")
    if "KAGGLE_API_TOKEN" in joined or "competitions submit" in joined:
        raise RuntimeError("generated notebook gained forbidden capability")
    digest = hashlib.sha256(raw).hexdigest()
    print(
        f"MELLUM_V3_GENERATED_BUNDLE PASS bytes={len(raw)} sha256={digest} labels_embedded=false"
    )
    return digest


def builder_failure_code(output: str) -> str:
    markers = (
        ("BASE_BUILDER_BLOB_MISMATCH", "BASE_BLOB"),
        ("BASE_BUILDER_CHECK_SITE_CHANGED", "PATCH_SITE"),
        ("BUILDER_PATCH_CARDINALITY_FAILED", "PATCH_CARDINALITY"),
        ("private source contract changed", "SOURCE_CONTRACT"),
        ("expected exactly one", "SOURCE_DEFINITION_CARDINALITY"),
        ("TRANSFER_MANIFEST", "TRANSFER_MANIFEST_CONTRACT"),
    )
    for marker, code in markers:
        if marker in output:
            return code
    return "UNCLASSIFIED"


def execute(
    workdir: Path,
    kaggle_bin: Path,
    builder: Path,
    base_builder: Path,
    shim: Path,
) -> None:
    api = live_preflight()
    workdir.mkdir(parents=True, exist_ok=False)
    source_dir = workdir / "source"
    kernel_dir = workdir / "kernel"
    tool_dir = workdir / "tool"
    for path in (source_dir, kernel_dir, tool_dir):
        path.mkdir()
    shutil.copy2(builder, tool_dir / "builder.py")
    shutil.copy2(base_builder, tool_dir / "base_builder.py")
    shutil.copy2(shim, tool_dir / "nbformat.py")

    result = subprocess.run(
        [str(kaggle_bin), "kernels", "pull", SOURCE, "-p", str(source_dir), "-m"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    notebooks = list(source_dir.glob("*.ipynb"))
    if result.returncode != 0 or len(notebooks) != 1 or not (source_dir / "kernel-metadata.json").is_file():
        raise RuntimeError("private source-kernel pull/materialization failed")

    result = subprocess.run(
        [
            sys.executable,
            str(tool_dir / "builder.py"),
            "--source-notebook",
            str(notebooks[0]),
            "--output-dir",
            str(kernel_dir),
            "--base-builder",
            str(tool_dir / "base_builder.py"),
        ],
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(tool_dir)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        code = builder_failure_code((result.stdout or "") + "\n" + (result.stderr or ""))
        raise RuntimeError(f"label-free Mellum notebook build failed code={code}")
    digest = validate_bundle(kernel_dir)

    # Close the largest race window: another competition may have started after
    # the first preflight. CPU/TPU remains allowed; a GPU or unknown active class
    # stops this request before the sole Kaggle write.
    allowed_other, _ = resource_admission(api)
    existing = {
        str(getattr(item, "ref", ""))
        for item in (api.kernels_list(user="renta0426", search="mellum-transfer-v1", page_size=20) or [])
    }
    if TARGET in existing:
        raise RuntimeError("target appeared before push; refusing ambiguous write")
    print(f"MELLUM_V3_PREWRITE_ADMISSION PASS active_gpu=0 allowed_cpu_tpu={allowed_other}")

    result = subprocess.run(
        [str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir), "--timeout", "10800"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        # No retry here. The caller must inspect Kaggle state before any fresh request.
        raise RuntimeError("Kaggle Mellum kernel push failed or was ambiguous; automatic retry disabled")

    meta = get_kernel_details(api, TARGET).metadata
    if (
        int(meta.current_version_number) != 1
        or not bool(meta.is_private)
        or not bool(meta.enable_gpu)
        or bool(meta.enable_tpu)
        or not bool(meta.enable_internet)
    ):
        raise RuntimeError("post-push Mellum metadata contract changed")
    print(
        f"MELLUM_V3_EXECUTION PASS request_id={REQUEST_ID} target={TARGET} version=1 "
        f"notebook_sha256={digest} accelerator=gpu machine=NvidiaTeslaT4 "
        "automatic_compute_retries=0 competition_submission=false"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--base-builder", type=Path, required=True)
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
    validate_static(
        request,
        args.request,
        args.builder,
        args.base_builder,
        args.shim,
        args.workflow,
    )
    if args.static:
        print(
            "MELLUM_V3_STATIC PASS request=003 builder_fix=fstring_source_marker "
            "concurrency=gpu_only live_admission=gpu_only rows=2000 labels_embedded=false "
            "write_calls=1 automatic_retries=0 submissions=0"
        )
        return
    if args.workdir is None or args.kaggle_bin is None:
        raise SystemExit("--execute requires --workdir and --kaggle-bin")
    execute(args.workdir, args.kaggle_bin, args.builder, args.base_builder, args.shim)


if __name__ == "__main__":
    main()
