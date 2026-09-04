"""Operational retry 004 for the frozen PAT-free SmolLM2 transfer.

This wrapper preserves request 003's scientific bundle and live-rule preflight,
while adding fail-closed, hashed observability for active Kaggle compute.
It never logs private kernel refs and does not relax the one-active-GPU guard.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
from typing import Any

REQUEST_ID = "20260904-poisoned-chalice-smollm2-transfer-v1-004"
PARENT_REQUEST_ID = "20260904-poisoned-chalice-smollm2-transfer-v1-003"
PRIOR_RUN_ID = 33822965958
COMPETITION = "poisoned-chalice-icse27"
TARGET = "renta0426/smollm2-transfer-v1"
BUILDER_PATH = "scripts/poisoned_chalice_smollm2_public_builder_v3.py"
BUILDER_BLOB_SHA = "ca2d92cd3afa27ac4c3150c983207fa7c7685168"
BASE_LAUNCHER_PATH = "scripts/poisoned_chalice_smollm2_public_launch_v3.py"
BASE_LAUNCHER_BLOB_SHA = "f950041e17ae2f3820e2e15be8adbae8f1c11dfa"
LAUNCHER_PATH = "scripts/poisoned_chalice_smollm2_public_launch_v4.py"
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"
MODEL_REVISION = "effd688a12921b4cc83e3312b6feb579f70f9c71"
EXPECTED_PREDICTION_MANIFEST_SHA256 = (
    "8d74549603dbe3ac19ea54eb32aa69152a041ae198f1fad504d06f4fad22684c"
)
MAX_RECENT = 25
MIN_GPU_HOURS = 3.0


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def load_base_launcher(path: Path):
    if git_blob_sha(path.read_bytes()) != BASE_LAUNCHER_BLOB_SHA:
        raise RuntimeError("request 004 base launcher blob mismatch")
    spec = importlib.util.spec_from_file_location("smollm2_public_launch_v3_pinned", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned request 003 launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_request(
    request: dict[str, Any],
    launcher_path: Path,
    builder_path: Path,
    base_launcher_path: Path,
) -> None:
    exact = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "parent_request_id": PARENT_REQUEST_ID,
        "prior_failed_run_id": PRIOR_RUN_ID,
        "competition": COMPETITION,
        "operation": "kernel_run",
        "target": TARGET,
        "builder_path": BUILDER_PATH,
        "builder_blob_sha": BUILDER_BLOB_SHA,
        "base_launcher_path": BASE_LAUNCHER_PATH,
        "base_launcher_blob_sha": BASE_LAUNCHER_BLOB_SHA,
        "target_model_id": MODEL_ID,
        "target_model_revision": MODEL_REVISION,
        "expected_prediction_manifest_sha256": EXPECTED_PREDICTION_MANIFEST_SHA256,
        "automatic_compute_retries": 0,
        "enable_internet": True,
        "competition_submission": False,
        "select_as_final": False,
        "private_repository_content_used": False,
    }
    for key, expected in exact.items():
        if request.get(key) != expected:
            raise RuntimeError(f"SmolLM2 request 004 contract changed: {key}")

    repair = request.get("failure_repair")
    if repair != {
        "prior_failure_class": "live_gpu_admission_deferred_active_gpu_1_before_kaggle_push",
        "prior_kaggle_write_occurred": False,
        "scientific_protocol_changed": False,
        "resource_contract_changed": False,
        "repair": (
            "retain max_active_runs=1, add hashed active-kernel observability, "
            "and attempt one fresh launch after a new protected approval"
        ),
    }:
        raise RuntimeError("SmolLM2 request 004 failure-repair lineage changed")

    if request.get("resource") != {
        "accelerator": "gpu",
        "machine_shape": "NvidiaTeslaT4",
        "expected_runtime_minutes": 90,
        "hard_timeout_minutes": 180,
        "max_active_runs": 1,
        "min_remaining_quota_hours": MIN_GPU_HOURS,
    }:
        raise RuntimeError("SmolLM2 request 004 resource contract changed")
    if request.get("api_budget") != {
        "max_calls": 60,
        "max_recent_kernels_inspected": MAX_RECENT,
        "max_pages": 2,
    }:
        raise RuntimeError("SmolLM2 request 004 API budget changed")

    expected_launcher = str(request.get("launcher_blob_sha") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_launcher):
        raise RuntimeError("request 004 launcher blob pin malformed")
    if git_blob_sha(launcher_path.read_bytes()) != expected_launcher:
        raise RuntimeError("request 004 launcher blob mismatch")
    if git_blob_sha(builder_path.read_bytes()) != BUILDER_BLOB_SHA:
        raise RuntimeError("request 004 builder blob mismatch")
    if git_blob_sha(base_launcher_path.read_bytes()) != BASE_LAUNCHER_BLOB_SHA:
        raise RuntimeError("request 004 base launcher blob mismatch")

    base = load_base_launcher(base_launcher_path)
    if request.get("public_sources") != {
        "current_dataset_id": base.DATASET_ID,
        "current_dataset_revision": base.DATASET_REVISION,
        "previous_competition_repository": base.UPSTREAM_REPOSITORY,
        "previous_competition_commit": base.UPSTREAM_COMMIT,
        "previous_competition_eval_path": "data/7b_train_test/eval_results.parquet",
        "previous_competition_eval_sha256": base.UPSTREAM_SHA256,
        "current_10k_seed": 2027,
        "expected_current_overlap_excluded": 23,
        "rows_per_language_label": 200,
        "expected_rows": 2000,
    }:
        raise RuntimeError("SmolLM2 request 004 public-source contract changed")
    if request.get("stage2_v2") != {
        "nuisance_numeric": ["log1p(token_count)", "log1p(window_count)"],
        "nuisance_categorical": ["language"],
        "components": ["minkpp_length_residual", "logrank_length_residual"],
        "within_language_percentile": True,
        "weights": [0.5, 0.5],
        "weights_searched": False,
    }:
        raise RuntimeError("SmolLM2 request 004 Stage2-v2 contract changed")
    if request.get("side_effects") != [
        "read two pinned public sources to reconstruct the frozen transfer cohort",
        "create one private SmolLM2 Notebook version and start one T4 GPU run",
    ]:
        raise RuntimeError("SmolLM2 request 004 side-effect allowlist changed")
    if request.get("clean_room") != {
        "target_labels_embedded_in_gpu_notebook": False,
        "target_labels_used_for_training_or_normalization": False,
        "previous_model_scores_used": False,
        "hidden_stage1_validation_labels_used": False,
        "public_leaderboard_tuning_used": False,
        "competition_submission_created": False,
        "cohort_selection_changed_after_mellum": False,
        "stage2_v2_weights_searched": False,
    }:
        raise RuntimeError("SmolLM2 request 004 clean-room contract changed")


def literal_commands(source: str) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []
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
        values: list[str] = []
        for item in node.args[0].elts:
            values.append(
                item.value
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
                else "<dynamic>"
            )
        commands.append(tuple(values))
    return commands


def has_pair(command: tuple[str, ...], left: str, right: str) -> bool:
    return any(
        command[index : index + 2] == (left, right)
        for index in range(len(command) - 1)
    )


def validate_static(
    launcher_path: Path, builder_path: Path, base_launcher_path: Path
) -> None:
    source = launcher_path.read_text(encoding="utf-8")
    compile(source, str(launcher_path), "exec")
    compile(builder_path.read_bytes(), str(builder_path), "exec")
    compile(base_launcher_path.read_bytes(), str(base_launcher_path), "exec")
    commands = literal_commands(source)
    if sum(has_pair(command, "kernels", "push") for command in commands) != 1:
        raise RuntimeError("request 004 launcher must contain exactly one kernels push")
    for pair in (
        ("competitions", "submit"),
        ("datasets", "create"),
        ("datasets", "version"),
        ("kernels", "delete"),
        ("kernels", "cancel"),
    ):
        if any(has_pair(command, *pair) for command in commands):
            raise RuntimeError(
                f"request 004 launcher gained forbidden write: {' '.join(pair)}"
            )
    tree = ast.parse(source)
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute"
        for node in ast.walk(tree)
    ):
        raise RuntimeError("request 004 must not delegate a second write path")
    builder = builder_path.read_text(encoding="utf-8")
    for forbidden in (
        "api.github.com/repos/renta0426/The-Poisoned-Chalice-of-LLM-Evaluation",
        "raw.githubusercontent.com/renta0426/The-Poisoned-Chalice-of-LLM-Evaluation",
        "github.com/renta0426/The-Poisoned-Chalice-of-LLM-Evaluation.git",
    ):
        if forbidden in builder:
            raise RuntimeError("public builder gained private-repository fetch capability")


def _ref_digest(ref: str) -> str:
    return hashlib.sha256(ref.encode("utf-8")).hexdigest()


def active_resource_snapshot(api, base) -> tuple[dict[str, int], list[dict[str, str]]]:
    """Return request-003-equivalent counts plus non-sensitive hashed observations."""

    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    counts = {"cpu": 0, "gpu": 0, "tpu": 0, "unknown": 0}
    observations: list[dict[str, str]] = []
    recent = (
        api.kernels_list(user="renta0426", sort_by="dateRun", page_size=MAX_RECENT)
        or []
    )[:MAX_RECENT]
    active: list[tuple[str, str]] = []
    for item in recent:
        ref = str(getattr(item, "ref", ""))
        if not ref:
            continue
        try:
            state = str(getattr(api.kernels_status(ref), "status", "")).upper()
        except Exception:
            counts["unknown"] += 1
            observations.append(
                {
                    "ref_sha256": _ref_digest(ref),
                    "status": "STATUS_QUERY_FAILED",
                    "resource": "unknown",
                }
            )
            continue
        if any(token in state for token in ("RUNNING", "QUEUED", "PENDING")):
            active.append((ref, state))

    if active:
        with api.build_kaggle_client() as client:
            for ref, state in active:
                resource = "unknown"
                try:
                    owner, slug = ref.split("/", 1)
                    query = ApiGetKernelRequest()
                    query.user_name = owner
                    query.kernel_slug = slug
                    details = client.kernels.kernels_api_client.get_kernel(query)
                    resource = base.classify_kernel_resource(details.metadata)
                    counts[resource] += 1
                except Exception:
                    counts["unknown"] += 1
                observations.append(
                    {
                        "ref_sha256": _ref_digest(ref),
                        "status": state,
                        "resource": resource,
                    }
                )

    for observation in observations:
        print(
            "SMOLLM2_PUBLIC_V4_ACTIVE_RESOURCE "
            f"resource={observation['resource']} status={observation['status']} "
            f"ref_sha256={observation['ref_sha256']}"
        )
    print(
        "SMOLLM2_PUBLIC_V4_ACTIVE_COUNTS "
        f"cpu={counts['cpu']} gpu={counts['gpu']} tpu={counts['tpu']} "
        f"unknown={counts['unknown']}"
    )
    return counts, observations


def enforce_gpu_admission(api, base) -> dict[str, int]:
    counts, _ = active_resource_snapshot(api, base)
    if counts["gpu"] >= 1 or counts["unknown"] > 0:
        raise RuntimeError(
            "SmolLM2 GPU admission refused: "
            f"active_gpu={counts['gpu']} active_unknown={counts['unknown']}"
        )
    return counts


def live_preflight(base):
    """Run the frozen request-003 live preflight with enhanced admission logging."""

    original = base.enforce_gpu_admission
    base.enforce_gpu_admission = lambda api: enforce_gpu_admission(api, base)
    try:
        return base.live_preflight()
    finally:
        base.enforce_gpu_admission = original


def execute(kaggle_bin: Path, kernel_dir: Path, base) -> None:
    api = live_preflight(base)
    base.validate_bundle(kernel_dir)

    counts = enforce_gpu_admission(api, base)
    print(
        "SMOLLM2_PUBLIC_V4_PREWRITE_ADMISSION PASS "
        f"active_cpu={counts['cpu']} active_gpu={counts['gpu']} "
        f"active_tpu={counts['tpu']} active_unknown={counts['unknown']}"
    )

    result = subprocess.run(
        [
            str(kaggle_bin),
            "kernels",
            "push",
            "-p",
            str(kernel_dir),
            "--timeout",
            "10800",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        existing = {
            str(getattr(item, "ref", ""))
            for item in (
                api.kernels_list(
                    user="renta0426", search="smollm2-transfer-v1", page_size=20
                )
                or []
            )
        }
        if TARGET in existing:
            raise RuntimeError(
                "SmolLM2 push returned failure but target exists; refusing ambiguous retry"
            )
        raise RuntimeError(
            f"SmolLM2 kernels push failed before target creation: rc={result.returncode}"
        )

    existing = {
        str(getattr(item, "ref", ""))
        for item in (
            api.kernels_list(
                user="renta0426", search="smollm2-transfer-v1", page_size=20
            )
            or []
        )
    }
    if TARGET not in existing:
        raise RuntimeError(
            "SmolLM2 kernels push returned success but target is not visible"
        )
    print(
        "SMOLLM2_PUBLIC_V4_EXECUTION PASS "
        f"request={REQUEST_ID} target={TARGET} resource=gpu "
        "automatic_compute_retries=0 competition_submission=false "
        "private_repository_content_used=false"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--base-launcher", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--kaggle-bin", type=Path)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--validate-bundle", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    validate_request(
        request, args.launcher, args.builder, args.base_launcher
    )
    validate_static(args.launcher, args.builder, args.base_launcher)
    base = load_base_launcher(args.base_launcher)

    modes = sum((args.static, args.validate_bundle, args.execute))
    if modes != 1:
        raise SystemExit("choose exactly one launcher mode")
    if args.static:
        print(
            "SMOLLM2_PUBLIC_V4_STATIC PASS request=004 resource_class=gpu "
            "public_sources=2 private_repo_reads=0 write_calls=1 retries=0 "
            "submissions=0 max_active_runs=1 resource_contract_changed=false"
        )
        return

    if args.bundle_root is None:
        raise SystemExit("--bundle-root is required")
    base.validate_bundle(args.bundle_root)
    if args.validate_bundle:
        print(
            "SMOLLM2_PUBLIC_V4_BUNDLE PASS rows=2000 labels=false "
            "private_repo_reads=0 submission=false"
        )
        return

    if args.kaggle_bin is None:
        raise SystemExit("--kaggle-bin is required for execute")
    execute(args.kaggle_bin, args.bundle_root, base)


if __name__ == "__main__":
    main()
