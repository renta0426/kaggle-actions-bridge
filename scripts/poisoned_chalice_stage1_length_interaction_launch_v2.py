"""Exact-once launcher for the repaired frozen Stage 1 length-interaction CPU Notebook."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

REQUEST_ID = "20260904-poisoned-chalice-stage1-length-interactions-run-002"
PARENT_REQUEST_ID = "20260903-poisoned-chalice-stage1-length-interactions-run-001"
COMPETITION = "poisoned-chalice-icse27"
TARGET = "renta0426/stage1-length-interactions-v2"
TARGET_OWNER = "renta0426"
TARGET_SLUG = "stage1-length-interactions-v2"
FAILED_TARGET = "renta0426/stage1-length-interactions-v1"
SOURCE_DATASET = "renta0426/stage1-raw-fim-submission-v1-output"
SOURCE_DATASET_VERSION = 1
BASE_BUILDER_PATH = "scripts/poisoned_chalice_stage1_length_interaction_builder.py"
BASE_BUILDER_BLOB_SHA = "a10c7a76be1a4d700d7b1d9846e42348c5711651"
BUILDER_PATH = "scripts/poisoned_chalice_stage1_length_interaction_builder_v2.py"
BUILDER_BLOB_SHA = "8f2cd12ecbad8160e86bbd42cf9b70649e1aa56b"
MAX_RECENT = 25
MAX_ACTIVE_CPU = 2


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def validate_request(request: dict, launcher_path: Path, builder_path: Path, base_builder_path: Path) -> None:
    expected = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "parent_request_id": PARENT_REQUEST_ID,
        "competition": COMPETITION,
        "operation": "kernel_run",
        "target": TARGET,
        "source_dataset": SOURCE_DATASET,
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "base_builder_path": BASE_BUILDER_PATH,
        "base_builder_blob_sha": BASE_BUILDER_BLOB_SHA,
        "builder_script_path": BUILDER_PATH,
        "builder_blob_sha": BUILDER_BLOB_SHA,
        "launcher_script_path": "scripts/poisoned_chalice_stage1_length_interaction_launch_v2.py",
        "launcher_blob_sha": git_blob_sha(launcher_path.read_bytes()),
        "repair": {
            "failed_target": FAILED_TARGET,
            "failed_version": 1,
            "failed_status": "ERROR",
            "failure_class": "kaggle_dataset_mount_path_compatibility",
            "scientific_candidate_changed": False,
            "source_dataset_changed": False,
            "expected_oof_auc": 0.68013844,
            "expected_tpr_at_0.01_fpr": 0.0682,
        },
        "resource": {
            "execution": "kaggle_notebook",
            "accelerator": "cpu",
            "expected_runtime_minutes": 12,
            "hard_timeout_minutes": 45,
            "max_active_runs": MAX_ACTIVE_CPU,
            "enable_internet": False,
        },
        "api_budget": {
            "max_calls": 24,
            "max_recent_kernels_inspected": MAX_RECENT,
            "max_pages": 2,
            "poll_interval_seconds": 0,
        },
        "input_budget": {
            "expected_train_shards": 40,
            "expected_validation_shards": 20,
            "expected_rows_per_shard": 250,
            "max_dataset_files": 150,
            "max_dataset_bytes": 1073741824,
        },
        "side_effects": [
            "create exactly one private CPU Kaggle Notebook at renta0426/stage1-length-interactions-v2"
        ],
        "automatic_compute_retries": 0,
        "competition_submission": False,
        "submission_inside_notebook": False,
    }
    if request != expected:
        raise RuntimeError("Stage1 length v2 request differs from exact contract")
    if git_blob_sha(builder_path.read_bytes()) != BUILDER_BLOB_SHA:
        raise RuntimeError("Stage1 length v2 builder Git blob changed")
    if git_blob_sha(base_builder_path.read_bytes()) != BASE_BUILDER_BLOB_SHA:
        raise RuntimeError("Stage1 length base builder Git blob changed")


def validate_bundle(kernel_dir: Path) -> None:
    metadata_path = kernel_dir / "kernel-metadata.json"
    notebook_path = kernel_dir / "stage1-length-interactions-v2.ipynb"
    if sorted(path.name for path in kernel_dir.iterdir()) != [
        "kernel-metadata.json", "stage1-length-interactions-v2.ipynb"
    ]:
        raise RuntimeError("Stage1 length v2 bundle allowlist changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "id": TARGET,
        "code_file": notebook_path.name,
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [SOURCE_DATASET],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Stage1 length v2 metadata mismatch: {key}")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", "")) if isinstance(cell.get("source", ""), list)
        else str(cell.get("source", ""))
        for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"
    )
    markers = (
        REQUEST_ID,
        'METHOD = "plus_length_interactions_v1"',
        "EXPECTED_MODEL_FEATURES = 523",
        "EXPECTED_OOF_AUC = 0.68013844",
        'root / "datasets" / "renta0426" / "stage1-raw-fim-submission-v1-output"',
        '"validation_labels_read": False',
        '"validation_membership_read": False',
        '"competition_submission_performed_inside_notebook": False',
    )
    for marker in markers:
        if marker not in code:
            raise RuntimeError(f"Stage1 length v2 notebook invariant missing: {marker}")
    for forbidden in ("KAGGLE_API_TOKEN", "competitions submit", "kernels push"):
        if forbidden in code:
            raise RuntimeError(f"Stage1 length v2 Notebook gained forbidden capability: {forbidden}")
    compile(code, notebook_path.name, "exec")


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


def active_resource_counts(api) -> dict[str, int]:
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    counts = {"cpu": 0, "gpu": 0, "tpu": 0, "unknown": 0}
    recent = (api.kernels_list(user=TARGET_OWNER, sort_by="dateRun", page_size=MAX_RECENT) or [])[:MAX_RECENT]
    active: list[str] = []
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
            active.append(ref)
    with api.build_kaggle_client() as client:
        for ref in active:
            try:
                owner, slug = ref.split("/", 1)
                request = ApiGetKernelRequest()
                request.user_name = owner
                request.kernel_slug = slug
                metadata = client.kernels.kernels_api_client.get_kernel(request).metadata
                if bool(value(metadata, "enable_tpu", "enableTpu")):
                    counts["tpu"] += 1
                elif bool(value(metadata, "enable_gpu", "enableGpu")):
                    counts["gpu"] += 1
                else:
                    counts["cpu"] += 1
            except Exception:
                counts["unknown"] += 1
    return counts


def validate_source_dataset(kaggle_bin: Path) -> None:
    status = subprocess.run(
        [str(kaggle_bin), "datasets", "status", SOURCE_DATASET, "--format", "json(current_version_number)"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False,
    )
    if status.returncode != 0 or not re.search(r"\b1\b", status.stdout):
        raise RuntimeError("source Dataset version 1 is unavailable")
    files = subprocess.run(
        [str(kaggle_bin), "datasets", "files", SOURCE_DATASET, "--page-size", "150", "-v"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False,
    )
    if files.returncode != 0:
        raise RuntimeError("source Dataset inventory lookup failed")
    rows = list(csv.reader(files.stdout.splitlines()))
    text = "\n".join(",".join(row).replace("\\", "/") for row in rows)
    train = set(re.findall(r"(?:[^,\s]+/)*train_10k/parts/features\.part\d{3}\.parquet", text))
    validation = set(re.findall(r"(?:[^,\s]+/)*validation_5k/parts/features\.part\d{3}\.parquet", text))
    if len(train) != 40 or len(validation) != 20:
        raise RuntimeError(
            f"source Dataset shard inventory changed: train={len(train)} validation={len(validation)}"
        )


def live_preflight(kaggle_bin: Path):
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi(); api.authenticate()
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

    failed_status = str(getattr(api.kernels_status(FAILED_TARGET), "status", "")).upper()
    if "ERROR" not in failed_status and "FAILED" not in failed_status:
        raise RuntimeError(f"expected diagnosed v1 failure, got {failed_status}")
    existing = {
        str(getattr(item, "ref", ""))
        for item in (api.kernels_list(user=TARGET_OWNER, search=TARGET_SLUG, page_size=20) or [])
    }
    if TARGET in existing:
        raise RuntimeError("repaired target already exists")

    validate_source_dataset(kaggle_bin)
    counts = active_resource_counts(api)
    if counts["unknown"] or counts["cpu"] >= MAX_ACTIVE_CPU:
        raise RuntimeError(f"Stage1 length v2 CPU admission refused: {counts}")
    print(
        "STAGE1_LENGTH_V2_LIVE_PREFLIGHT PASS "
        f"failed_v1={failed_status} source_dataset_version=1 train_shards=40 validation_shards=20 "
        f"active_cpu={counts['cpu']} active_gpu={counts['gpu']} active_tpu={counts['tpu']}"
    )
    return api


def execute(kaggle_bin: Path, kernel_dir: Path) -> None:
    api = live_preflight(kaggle_bin)
    validate_bundle(kernel_dir)
    counts = active_resource_counts(api)
    if counts["unknown"] or counts["cpu"] >= MAX_ACTIVE_CPU:
        raise RuntimeError(f"Stage1 length v2 prewrite admission refused: {counts}")
    result = subprocess.run(
        [str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir), "--timeout", "2700"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"one permitted kernels push failed: {result.stderr[:500]}")

    existing = {
        str(getattr(item, "ref", ""))
        for item in (api.kernels_list(user=TARGET_OWNER, search=TARGET_SLUG, page_size=20) or [])
    }
    if TARGET not in existing:
        raise RuntimeError("repaired target was not created after the one permitted push")
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name = TARGET_OWNER
        request.kernel_slug = TARGET_SLUG
        metadata = client.kernels.kernels_api_client.get_kernel(request).metadata
    if int(metadata.current_version_number) != 1 or not bool(metadata.is_private):
        raise RuntimeError("repaired target identity/version/privacy mismatch")
    if bool(value(metadata, "enable_gpu", "enableGpu")) or bool(value(metadata, "enable_tpu", "enableTpu")):
        raise RuntimeError("repaired target accelerator contract mismatch")
    if bool(value(metadata, "enable_internet", "enableInternet")):
        raise RuntimeError("repaired target internet contract mismatch")
    print(
        "STAGE1_LENGTH_V2_EXECUTION PASS request=002 "
        "target=renta0426/stage1-length-interactions-v2 version=1 accelerator=cpu "
        "automatic_compute_retries=0 competition_submission=false"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--base-builder", type=Path, required=True)
    parser.add_argument("--kaggle-bin", type=Path)
    parser.add_argument("--kernel-dir", type=Path)
    args = parser.parse_args()
    if args.static == args.execute:
        raise SystemExit("choose exactly one of --static or --execute")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    validate_request(request, args.launcher, args.builder, args.base_builder)
    if args.static:
        print(
            "STAGE1_LENGTH_V2_STATIC PASS request=002 resource_class=cpu "
            "scientific_candidate_changed=false automatic_retries=0 submissions=0"
        )
        return
    if args.kaggle_bin is None or args.kernel_dir is None:
        raise SystemExit("--execute requires --kaggle-bin and --kernel-dir")
    execute(args.kaggle_bin, args.kernel_dir)


if __name__ == "__main__":
    main()
