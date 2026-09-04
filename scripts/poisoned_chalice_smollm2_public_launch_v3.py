"""Exact-once launcher for PAT-free frozen SmolLM2 transfer request 003."""

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

REQUEST_ID = "20260904-poisoned-chalice-smollm2-transfer-v1-003"
COMPETITION = "poisoned-chalice-icse27"
TARGET = "renta0426/smollm2-transfer-v1"
BUILDER_PATH = "scripts/poisoned_chalice_smollm2_public_builder_v3.py"
BUILDER_BLOB_SHA = "ca2d92cd3afa27ac4c3150c983207fa7c7685168"
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"
MODEL_REVISION = "effd688a12921b4cc83e3312b6feb579f70f9c71"
UPSTREAM_REPOSITORY = "Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026"
UPSTREAM_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
UPSTREAM_SHA256 = "6e8559279e8ebd94ec8c25561c91d38c454a4e109ea0813b07864ff6a66d4068"
DATASET_ID = "Poisoned-Chalice/ICSE-2027-public"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
MIN_GPU_HOURS = 3.0
MAX_RECENT = 25
EXPECTED_ROWS = 2000


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def validate_request(request: dict, launcher_path: Path, builder_path: Path) -> None:
    exact = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": COMPETITION,
        "operation": "kernel_run",
        "target": TARGET,
        "builder_path": BUILDER_PATH,
        "builder_blob_sha": BUILDER_BLOB_SHA,
        "target_model_id": MODEL_ID,
        "target_model_revision": MODEL_REVISION,
        "automatic_compute_retries": 0,
        "enable_internet": True,
        "competition_submission": False,
        "select_as_final": False,
        "private_repository_content_used": False,
    }
    for key, value in exact.items():
        if request.get(key) != value:
            raise RuntimeError(f"SmolLM2 request 003 contract changed: {key}")
    if request.get("resource") != {
        "accelerator": "gpu",
        "machine_shape": "NvidiaTeslaT4",
        "expected_runtime_minutes": 90,
        "hard_timeout_minutes": 180,
        "max_active_runs": 1,
        "min_remaining_quota_hours": MIN_GPU_HOURS,
    }:
        raise RuntimeError("SmolLM2 request 003 resource contract changed")
    if request.get("api_budget") != {
        "max_calls": 60,
        "max_recent_kernels_inspected": MAX_RECENT,
        "max_pages": 2,
    }:
        raise RuntimeError("SmolLM2 request 003 API budget changed")
    if request.get("public_sources") != {
        "current_dataset_id": DATASET_ID,
        "current_dataset_revision": DATASET_REVISION,
        "previous_competition_repository": UPSTREAM_REPOSITORY,
        "previous_competition_commit": UPSTREAM_COMMIT,
        "previous_competition_eval_path": "data/7b_train_test/eval_results.parquet",
        "previous_competition_eval_sha256": UPSTREAM_SHA256,
        "current_10k_seed": 2027,
        "expected_current_overlap_excluded": 23,
        "rows_per_language_label": 200,
        "expected_rows": EXPECTED_ROWS,
    }:
        raise RuntimeError("SmolLM2 request 003 public-source contract changed")
    if request.get("side_effects") != [
        "read two pinned public sources to reconstruct the frozen transfer cohort",
        "create one private SmolLM2 Notebook version and start one T4 GPU run",
    ]:
        raise RuntimeError("SmolLM2 request 003 side-effect allowlist changed")
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
        raise RuntimeError("SmolLM2 request 003 clean-room contract changed")
    expected_launcher = str(request.get("launcher_blob_sha") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_launcher):
        raise RuntimeError("request 003 launcher blob pin malformed")
    if blob_sha(launcher_path.read_bytes()) != expected_launcher:
        raise RuntimeError("request 003 launcher blob mismatch")
    if blob_sha(builder_path.read_bytes()) != BUILDER_BLOB_SHA:
        raise RuntimeError("request 003 builder blob mismatch")


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
    return any(command[i : i + 2] == (left, right) for i in range(len(command) - 1))


def validate_static(launcher_path: Path, builder_path: Path) -> None:
    compile(launcher_path.read_bytes(), str(launcher_path), "exec")
    compile(builder_path.read_bytes(), str(builder_path), "exec")
    commands = literal_commands(launcher_path.read_text(encoding="utf-8"))
    if sum(has_pair(command, "kernels", "push") for command in commands) != 1:
        raise RuntimeError("request 003 launcher must contain exactly one kernels push")
    for pair in (
        ("competitions", "submit"),
        ("datasets", "create"),
        ("datasets", "version"),
        ("kernels", "delete"),
        ("kernels", "cancel"),
    ):
        if any(has_pair(command, *pair) for command in commands):
            raise RuntimeError(f"request 003 launcher gained forbidden write: {' '.join(pair)}")
    builder = builder_path.read_text(encoding="utf-8")
    for forbidden in (
        "api.github.com/repos/renta0426/The-Poisoned-Chalice-of-LLM-Evaluation",
        "raw.githubusercontent.com/renta0426/The-Poisoned-Chalice-of-LLM-Evaluation",
        "github.com/renta0426/The-Poisoned-Chalice-of-LLM-Evaluation.git",
    ):
        if forbidden in builder:
            raise RuntimeError("public builder gained private-repository fetch capability")


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
    active_refs: list[str] = []
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
                query = ApiGetKernelRequest()
                query.user_name = owner
                query.kernel_slug = slug
                details = client.kernels.kernels_api_client.get_kernel(query)
                counts[classify_kernel_resource(details.metadata)] += 1
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


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text.replace(r"\%", "%"))).casefold()


def live_preflight():
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
        "SMOLLM2_PUBLIC_V3_LIVE_PREFLIGHT PASS "
        f"gpu_quota_hours={remaining:.2f} active_cpu={counts['cpu']} "
        f"active_gpu={counts['gpu']} active_tpu={counts['tpu']} active_unknown={counts['unknown']}"
    )
    return api


def validate_bundle(kernel_dir: Path) -> None:
    notebook_path = kernel_dir / "smollm2-transfer-v1.ipynb"
    metadata_path = kernel_dir / "kernel-metadata.json"
    if not notebook_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("request 003 bundle incomplete")
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
            raise RuntimeError(f"request 003 metadata mismatch: {key}")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    if len(cells) != 7:
        raise RuntimeError("request 003 notebook cell count changed")
    code = "\n".join(
        "".join(cell.get("source", "")) if isinstance(cell.get("source", ""), list)
        else str(cell.get("source", ""))
        for cell in cells if cell.get("cell_type") == "code"
    )
    for marker in (
        MODEL_ID,
        MODEL_REVISION,
        UPSTREAM_COMMIT,
        UPSTREAM_SHA256,
        "standard_minkpp",
        "neg_mean_log_rank",
        "minkpp_length_residual",
        "logrank_length_residual",
        "membership_score_v2",
        "cohort_reconstructed_from_pinned_public_sources",
    ):
        if marker not in code:
            raise RuntimeError(f"request 003 scientific marker missing: {marker}")
    for forbidden in (
        "KAGGLE_API_TOKEN",
        "RESEARCH_REPO_READ_TOKEN",
        "competitions submit",
        '"label":',
        '"membership":',
        '"is_member":',
    ):
        if forbidden in code:
            raise RuntimeError(f"request 003 forbidden notebook content: {forbidden}")
    manifest_records = None
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", "")) if isinstance(cell.get("source", ""), list) else str(cell.get("source", ""))
        if "PREDICTION_MANIFEST = pd.DataFrame(" not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "PREDICTION_MANIFEST" for target in node.targets):
                continue
            call = node.value
            if not isinstance(call, ast.Call) or not call.args:
                raise RuntimeError("prediction manifest is not a literal DataFrame")
            manifest_records = ast.literal_eval(call.args[0])
    allowed = {"sample_id", "content_sha256", "language", "sample_index"}
    if not isinstance(manifest_records, list) or len(manifest_records) != EXPECTED_ROWS:
        raise RuntimeError("prediction manifest row count changed")
    if any(set(record) != allowed for record in manifest_records):
        raise RuntimeError("prediction manifest gained target labels or unexpected fields")
    if [int(record["sample_index"]) for record in manifest_records] != list(range(EXPECTED_ROWS)):
        raise RuntimeError("prediction manifest order changed")


def execute(kaggle_bin: Path, kernel_dir: Path) -> None:
    api = live_preflight()
    validate_bundle(kernel_dir)
    counts = enforce_gpu_admission(api)
    print(
        "SMOLLM2_PUBLIC_V3_PREWRITE_ADMISSION PASS "
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
            raise RuntimeError("SmolLM2 push returned failure but target exists; refusing ambiguous retry")
        raise RuntimeError(f"SmolLM2 kernels push failed before target creation: rc={result.returncode}")
    existing = {
        str(getattr(item, "ref", ""))
        for item in (api.kernels_list(user="renta0426", search="smollm2-transfer-v1", page_size=20) or [])
    }
    if TARGET not in existing:
        raise RuntimeError("SmolLM2 kernels push returned success but target is not visible")
    print(
        "SMOLLM2_PUBLIC_V3_EXECUTION PASS "
        f"request={REQUEST_ID} target={TARGET} resource=gpu automatic_compute_retries=0 "
        "competition_submission=false private_repository_content_used=false"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--kaggle-bin", type=Path)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--validate-bundle", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    validate_request(request, args.launcher, args.builder)
    validate_static(args.launcher, args.builder)
    modes = sum((args.static, args.validate_bundle, args.execute))
    if modes != 1:
        raise SystemExit("choose exactly one launcher mode")
    if args.static:
        print(
            "SMOLLM2_PUBLIC_V3_STATIC PASS request=003 resource_class=gpu "
            "public_sources=2 private_repo_reads=0 write_calls=1 retries=0 submissions=0"
        )
        return
    if args.bundle_root is None:
        raise SystemExit("--bundle-root is required")
    validate_bundle(args.bundle_root)
    if args.validate_bundle:
        print(
            "SMOLLM2_PUBLIC_V3_BUNDLE PASS rows=2000 labels=false private_repo_reads=0 submission=false"
        )
        return
    if args.kaggle_bin is None:
        raise SystemExit("--kaggle-bin is required for execute")
    execute(args.kaggle_bin, args.bundle_root)


if __name__ == "__main__":
    main()
