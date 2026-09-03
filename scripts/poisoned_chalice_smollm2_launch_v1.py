"""Exact-once Kaggle launcher for frozen SmolLM2 transfer request 001."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

REQUEST_ID = "20260903-poisoned-chalice-smollm2-transfer-v1-001"
COMPETITION = "poisoned-chalice-icse27"
TARGET = "renta0426/smollm2-transfer-v1"
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"
MODEL_REVISION = "effd688a12921b4cc83e3312b6feb579f70f9c71"
RESEARCH_REPOSITORY = "renta0426/The-Poisoned-Chalice-of-LLM-Evaluation"
RESEARCH_COMMIT = "253806607ddec32364c89c39dd6f946599085868"
PUBLIC_SCORE = 0.34575
MIN_GPU_HOURS = 3.0
MAX_RECENT = 25


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def validate_request(request: dict, launcher_path: Path, materializer_path: Path) -> None:
    exact = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": COMPETITION,
        "operation": "kernel_run",
        "target": TARGET,
        "research_repository": RESEARCH_REPOSITORY,
        "research_commit": RESEARCH_COMMIT,
        "target_model_id": MODEL_ID,
        "target_model_revision": MODEL_REVISION,
        "automatic_compute_retries": 0,
        "enable_internet": True,
        "competition_submission": False,
        "select_as_final": False,
        "runner_local_private_material_retention_days": 0,
    }
    for key, value in exact.items():
        if request.get(key) != value:
            raise RuntimeError(f"SmolLM2 request contract changed: {key}")
    if request.get("resource") != {
        "accelerator": "gpu",
        "machine_shape": "NvidiaTeslaT4",
        "expected_runtime_minutes": 90,
        "hard_timeout_minutes": 180,
        "max_active_runs": 1,
        "min_remaining_quota_hours": MIN_GPU_HOURS,
    }:
        raise RuntimeError("SmolLM2 resource contract changed")
    if request.get("api_budget") != {
        "max_calls": 60,
        "max_recent_kernels_inspected": MAX_RECENT,
        "max_pages": 2,
    }:
        raise RuntimeError("SmolLM2 API budget changed")
    if request.get("side_effects") != [
        "read five files from one private research commit",
        "create one private SmolLM2 Notebook version and start one T4 GPU run",
    ]:
        raise RuntimeError("SmolLM2 side effects changed")
    if request.get("clean_room") != {
        "target_labels_embedded": False,
        "target_labels_used_for_training_or_normalization": False,
        "previous_model_scores_used": False,
        "hidden_stage1_validation_labels_used": False,
        "public_leaderboard_tuning_used": False,
        "competition_submission_created": False,
    }:
        raise RuntimeError("SmolLM2 clean-room contract changed")
    expected_files = {
        "scripts/build_smollm2_transfer_notebook.py": ("1dae05433081721d660dc2203273c66797878678", 131072),
        "src/poisoned_chalice/stage2.py": ("9c2086f3c73ec5998ac3b50d7a4e166f6b1b4443", 131072),
        "src/poisoned_chalice/stage2_v2.py": ("9121a34d0f9ea84531c405bf127f8d8846d4274d", 65536),
        "configs/smollm2_transfer_v1.json": ("f58e31ed3d64259bc57bdaa40b8cc75b43e8d5a1", 32768),
        "experiments/pseudo-stage2-transfer-v1/transfer_sample_manifest.parquet": ("dedfd34d43e53c158398ae3cc99ed508cbe37f66", 1048576),
    }
    observed = {
        str(item.get("path")): (str(item.get("git_blob_sha")), int(item.get("max_bytes")))
        for item in request.get("research_files", [])
    }
    if observed != expected_files:
        raise RuntimeError("SmolLM2 private research file contract changed")
    for key, path in (
        ("launcher_blob_sha", launcher_path),
        ("private_materializer_blob_sha", materializer_path),
    ):
        expected = str(request.get(key) or "")
        if not re.fullmatch(r"[0-9a-f]{40}", expected):
            raise RuntimeError(f"SmolLM2 bridge blob pin malformed: {key}")
        if blob_sha(path.read_bytes()) != expected:
            raise RuntimeError(f"SmolLM2 bridge source blob mismatch: {key}")


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


def validate_static(launcher_path: Path) -> None:
    commands = literal_commands(launcher_path.read_text(encoding="utf-8"))
    if sum(has_pair(command, "kernels", "push") for command in commands) != 1:
        raise RuntimeError("SmolLM2 launcher must contain exactly one kernels push")
    for pair in (
        ("competitions", "submit"),
        ("datasets", "create"),
        ("datasets", "version"),
        ("kernels", "delete"),
        ("kernels", "cancel"),
    ):
        if any(has_pair(command, *pair) for command in commands):
            raise RuntimeError(f"SmolLM2 launcher gained forbidden write: {' '.join(pair)}")


def plain(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"<[^>]+>", " ", text.replace(r"\%", "%")),
    ).casefold()


def value(item, *names):
    data = item.to_dict() if hasattr(item, "to_dict") else (item if isinstance(item, dict) else {})
    for name in names:
        candidate = getattr(item, name, None)
        if candidate is None:
            candidate = data.get(name)
        if candidate is not None:
            return candidate
    return None


def classify_kernel_resource(metadata) -> str:
    gpu = value(metadata, "enable_gpu", "enableGpu")
    tpu = value(metadata, "enable_tpu", "enableTpu")
    if tpu is True:
        return "tpu"
    if gpu is True:
        return "gpu"
    if gpu is False and tpu is False:
        return "cpu"
    raise RuntimeError("active Kaggle resource class is unknown")


def active_resource_counts(api) -> dict[str, int]:
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    counts = {"cpu": 0, "gpu": 0, "tpu": 0, "unknown": 0}
    recent = (api.kernels_list(user="renta0426", sort_by="dateRun", page_size=MAX_RECENT) or [])[:MAX_RECENT]
    active_refs = []
    for item in recent:
        ref = str(getattr(item, "ref", ""))
        if not ref:
            continue
        try:
            state = str(getattr(api.kernels_status(ref), "status", "")).upper()
        except Exception:
            counts["unknown"] += 1
            continue
        if any(token in state for token in ("RUNNING", "QUEUED", "PENDING")):
            active_refs.append(ref)
    if not active_refs:
        return counts
    with api.build_kaggle_client() as client:
        for ref in active_refs:
            try:
                owner, slug = ref.split("/", 1)
                request = ApiGetKernelRequest()
                request.user_name = owner
                request.kernel_slug = slug
                details = client.kernels.kernels_api_client.get_kernel(request)
                resource = classify_kernel_resource(details.metadata)
                counts[resource] += 1
            except Exception:
                counts["unknown"] += 1
    return counts


def enforce_gpu_admission(api) -> dict[str, int]:
    counts = active_resource_counts(api)
    if counts["gpu"] >= 1 or counts["unknown"] > 0:
        raise RuntimeError(
            "SmolLM2 GPU admission refused: "
            f"active_gpu={counts['gpu']} active_unknown={counts['unknown']}"
        )
    return counts


def live_preflight():  # noqa: ANN201
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    pages = api.competition_list_pages(COMPETITION) or []
    content: dict[str, str] = {}
    for page in pages:
        data = page.to_dict() if hasattr(page, "to_dict") else dict(page)
        name = str(data.get("name") or "").strip().lower()
        if name:
            content[name] = str(data.get("content") or "")
    if "rules" not in content or "evaluation" not in content or not any("data" in name for name in content):
        raise RuntimeError("live Competition rules/evaluation/data unavailable")
    evaluation = plain(content["evaluation"])
    if not all(term in evaluation for term in ("auc", "novelty", "1%", "false-positive")):
        raise RuntimeError("live Poisoned Chalice evaluation contract changed")
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
    score_found = False
    for item in submissions:
        try:
            score = float(value(item, "public_score", "publicScore"))
        except (TypeError, ValueError):
            continue
        if (
            "COMPLETE" in str(value(item, "status") or "").upper()
            and math.isclose(score, PUBLIC_SCORE, rel_tol=0.0, abs_tol=5e-6)
        ):
            score_found = True
            break
    if not score_found:
        raise RuntimeError("Stage1 prerequisite score 0.34575 not found")

    existing = {
        str(getattr(item, "ref", ""))
        for item in (api.kernels_list(user="renta0426", search="smollm2-transfer-v1", page_size=20) or [])
    }
    if TARGET in existing:
        raise RuntimeError("target SmolLM2 kernel already exists")

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
    counts = enforce_gpu_admission(api)
    print(
        "SMOLLM2_LIVE_PREFLIGHT PASS "
        f"score={PUBLIC_SCORE:.5f} gpu_quota_hours={remaining:.2f} "
        f"active_cpu={counts['cpu']} active_gpu={counts['gpu']} "
        f"active_tpu={counts['tpu']} active_unknown={counts['unknown']}"
    )
    return api


def validate_bundle(kernel_dir: Path) -> None:
    notebook_path = kernel_dir / "smollm2-transfer-v1.ipynb"
    metadata_path = kernel_dir / "kernel-metadata.json"
    if not notebook_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("runner-local SmolLM2 bundle incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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
            raise RuntimeError(f"runner-local SmolLM2 metadata mismatch: {key}")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if len(notebook.get("cells", [])) != 8:
        raise RuntimeError("runner-local SmolLM2 Notebook cell count changed")
    joined = "\n".join(
        "".join(cell.get("source", ""))
        if isinstance(cell.get("source", ""), list)
        else str(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for marker in (MODEL_ID, MODEL_REVISION, "membership_score_v2", "fuse_membership_scores_v2"):
        if marker not in joined:
            raise RuntimeError(f"runner-local SmolLM2 marker missing: {marker}")
    if "KAGGLE_API_TOKEN" in joined or "competitions submit" in joined:
        raise RuntimeError("runner-local SmolLM2 Notebook gained forbidden Kaggle capability")


def execute(kaggle_bin: Path, kernel_dir: Path) -> None:
    api = live_preflight()
    validate_bundle(kernel_dir)
    counts = enforce_gpu_admission(api)
    print(
        "SMOLLM2_PREWRITE_ADMISSION PASS "
        f"active_cpu={counts['cpu']} active_gpu={counts['gpu']} "
        f"active_tpu={counts['tpu']} active_unknown={counts['unknown']}"
    )
    result = subprocess.run(
        [str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir), "--timeout", "10800"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        existing = {
            str(getattr(item, "ref", ""))
            for item in (api.kernels_list(user="renta0426", search="smollm2-transfer-v1", page_size=20) or [])
        }
        if TARGET in existing:
            raise RuntimeError(
                "SmolLM2 push returned nonzero but target exists; outcome ambiguous and retry is forbidden"
            )
        raise RuntimeError("SmolLM2 Kaggle push failed before confirmed creation; retry is forbidden")

    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name = "renta0426"
        request.kernel_slug = "smollm2-transfer-v1"
        details = client.kernels.kernels_api_client.get_kernel(request)
    metadata = details.metadata
    if (
        int(metadata.current_version_number) != 1
        or not bool(metadata.is_private)
        or not bool(metadata.enable_gpu)
        or bool(metadata.enable_tpu)
        or not bool(metadata.enable_internet)
    ):
        raise RuntimeError("post-push SmolLM2 Kaggle metadata contract changed")
    print(
        f"SMOLLM2_EXECUTION PASS request_id={REQUEST_ID} target={TARGET} version=1 "
        "accelerator=gpu machine=NvidiaTeslaT4 automatic_compute_retries=0 "
        "competition_submission=false"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--kaggle-bin", type=Path)
    parser.add_argument("--kernel-dir", type=Path)
    args = parser.parse_args()
    if args.static == args.execute:
        raise SystemExit("choose exactly one of --static or --execute")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    validate_request(request, args.launcher, args.materializer)
    validate_static(args.launcher)
    if args.static:
        print(
            "SMOLLM2_STATIC PASS request=001 resource_class=gpu write_calls=1 "
            "automatic_retries=0 submissions=0"
        )
        return
    if args.kaggle_bin is None or args.kernel_dir is None:
        raise SystemExit("--execute requires --kaggle-bin and --kernel-dir")
    execute(args.kaggle_bin, args.kernel_dir)


if __name__ == "__main__":
    main()
