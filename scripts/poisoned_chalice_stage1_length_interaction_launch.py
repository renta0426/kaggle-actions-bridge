"""Exact-once launcher for the frozen Stage 1 length-interaction CPU Notebook."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

REQUEST_ID = "20260903-poisoned-chalice-stage1-length-interactions-run-001"
COMPETITION = "poisoned-chalice-icse27"
TARGET = "renta0426/stage1-length-interactions-v1"
TARGET_OWNER = "renta0426"
TARGET_SLUG = "stage1-length-interactions-v1"
SOURCE_DATASET = "renta0426/stage1-raw-fim-submission-v1-output"
SOURCE_DATASET_VERSION = 1
MAX_RECENT = 25
MAX_ACTIVE_CPU = 2


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def validate_request(request: dict, launcher_path: Path, builder_path: Path) -> None:
    expected_keys = {
        "schema_version", "request_id", "competition", "operation", "target",
        "source_dataset", "source_dataset_version", "builder_script_path",
        "builder_blob_sha", "launcher_script_path", "launcher_blob_sha",
        "research_evidence", "resource", "api_budget", "input_budget",
        "side_effects", "automatic_compute_retries", "competition_submission",
        "submission_inside_notebook",
    }
    if set(request) != expected_keys:
        raise RuntimeError("Stage1 length-interaction request field allowlist changed")
    exact = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": COMPETITION,
        "operation": "kernel_run",
        "target": TARGET,
        "source_dataset": SOURCE_DATASET,
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "builder_script_path": "scripts/poisoned_chalice_stage1_length_interaction_builder.py",
        "launcher_script_path": "scripts/poisoned_chalice_stage1_length_interaction_launch.py",
        "automatic_compute_retries": 0,
        "competition_submission": False,
        "submission_inside_notebook": False,
    }
    for key, expected in exact.items():
        if request.get(key) != expected:
            raise RuntimeError(f"Stage1 length-interaction request contract changed: {key}")
    if request.get("research_evidence") != {
        "repository": "renta0426/The-Poisoned-Chalice-of-LLM-Evaluation",
        "audit_commit": "faeac269c7a5f332e4991f540260a290563e35b6",
        "result_merge_commit": "915f63a97b2e90b429fffa41b1d181d64c133e55",
        "aggregate_bridge_run_id": 33759558064,
        "selected_variant": "plus_length_interactions",
        "expected_oof_auc": 0.68013844,
        "expected_tpr_at_0.01_fpr": 0.0682,
        "public_lb_used_for_selection": False,
        "mellum_labels_used": False,
    }:
        raise RuntimeError("Stage1 research-evidence contract changed")
    if request.get("resource") != {
        "execution": "kaggle_notebook",
        "accelerator": "cpu",
        "expected_runtime_minutes": 12,
        "hard_timeout_minutes": 45,
        "max_active_runs": MAX_ACTIVE_CPU,
        "enable_internet": False,
    }:
        raise RuntimeError("Stage1 CPU resource contract changed")
    if request.get("api_budget") != {
        "max_calls": 24,
        "max_recent_kernels_inspected": MAX_RECENT,
        "max_pages": 2,
        "poll_interval_seconds": 0,
    }:
        raise RuntimeError("Stage1 API budget changed")
    if request.get("input_budget") != {
        "expected_train_shards": 40,
        "expected_validation_shards": 20,
        "expected_rows_per_shard": 250,
        "max_dataset_files": 150,
        "max_dataset_bytes": 1073741824,
    }:
        raise RuntimeError("Stage1 input budget changed")
    if request.get("side_effects") != [
        "create exactly one private Kaggle Notebook version at renta0426/stage1-length-interactions-v1"
    ]:
        raise RuntimeError("Stage1 side effects changed")
    for key, path in (("builder_blob_sha", builder_path), ("launcher_blob_sha", launcher_path)):
        expected = str(request.get(key) or "")
        if not re.fullmatch(r"[0-9a-f]{40}", expected):
            raise RuntimeError(f"malformed Git blob pin: {key}")
        if blob_sha(path.read_bytes()) != expected:
            raise RuntimeError(f"Git blob mismatch: {key}")


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
                item.value if isinstance(item, ast.Constant) and isinstance(item.value, str) else "<dynamic>"
            )
        commands.append(tuple(values))
    return commands


def has_pair(command: tuple[str, ...], left: str, right: str) -> bool:
    return any(command[index:index + 2] == (left, right) for index in range(len(command) - 1))


def validate_static(launcher_path: Path) -> None:
    commands = literal_commands(launcher_path.read_text(encoding="utf-8"))
    if sum(has_pair(command, "kernels", "push") for command in commands) != 1:
        raise RuntimeError("Stage1 launcher must contain exactly one kernels push")
    for pair in (
        ("competitions", "submit"),
        ("datasets", "create"),
        ("datasets", "version"),
        ("kernels", "delete"),
        ("kernels", "cancel"),
    ):
        if any(has_pair(command, *pair) for command in commands):
            raise RuntimeError(f"Stage1 launcher gained forbidden write: {' '.join(pair)}")


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
    recent = (api.kernels_list(user=TARGET_OWNER, sort_by="dateRun", page_size=MAX_RECENT) or [])[:MAX_RECENT]
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
                request = ApiGetKernelRequest()
                request.user_name = owner
                request.kernel_slug = slug
                details = client.kernels.kernels_api_client.get_kernel(request)
                counts[classify_kernel_resource(details.metadata)] += 1
            except Exception:
                counts["unknown"] += 1
    return counts


def enforce_cpu_admission(api) -> dict[str, int]:
    counts = active_resource_counts(api)
    if counts["unknown"] > 0 or counts["cpu"] >= MAX_ACTIVE_CPU:
        raise RuntimeError(
            "Stage1 CPU admission refused: "
            f"active_cpu={counts['cpu']} active_gpu={counts['gpu']} "
            f"active_tpu={counts['tpu']} active_unknown={counts['unknown']}"
        )
    return counts


def validate_source_dataset(kaggle_bin: Path) -> None:
    status = subprocess.run(
        [str(kaggle_bin), "datasets", "status", SOURCE_DATASET, "--format", "json(current_version_number)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        raise RuntimeError("source Dataset status lookup failed")
    match = re.search(r"\b(\d+)\b", status.stdout)
    if not match or int(match.group(1)) != SOURCE_DATASET_VERSION:
        raise RuntimeError("source Dataset version changed")

    files = subprocess.run(
        [str(kaggle_bin), "datasets", "files", SOURCE_DATASET, "--page-size", "150", "-v"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if files.returncode != 0:
        raise RuntimeError("source Dataset inventory lookup failed")
    rows = list(csv.reader(files.stdout.splitlines()))
    if len(rows) < 2 or len(rows) > 151:
        raise RuntimeError(f"source Dataset inventory rows changed: {len(rows)}")
    text = "\n".join(",".join(row).replace("\\", "/") for row in rows)
    train = set(re.findall(r"(?:[^,\s]+/)*train_10k/parts/features\.part\d{3}\.parquet", text))
    validation = set(re.findall(r"(?:[^,\s]+/)*validation_5k/parts/features\.part\d{3}\.parquet", text))
    if len(train) != 40 or len(validation) != 20:
        raise RuntimeError(
            f"source Dataset shard inventory changed: train={len(train)} validation={len(validation)}"
        )


def live_preflight(kaggle_bin: Path):  # noqa: ANN201
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
        "cpu use is prohibited",
        "internet access is required",
        "submissions must use gpu",
    ):
        if phrase in rules:
            raise RuntimeError(f"live Competition rule conflict: {phrase}")

    existing = {
        str(getattr(item, "ref", ""))
        for item in (api.kernels_list(user=TARGET_OWNER, search=TARGET_SLUG, page_size=20) or [])
    }
    if TARGET in existing:
        raise RuntimeError("target Stage1 length-interaction kernel already exists")

    validate_source_dataset(kaggle_bin)
    counts = enforce_cpu_admission(api)
    print(
        "STAGE1_LENGTH_INTERACTION_LIVE_PREFLIGHT PASS "
        f"active_cpu={counts['cpu']} active_gpu={counts['gpu']} "
        f"active_tpu={counts['tpu']} active_unknown={counts['unknown']} "
        "source_dataset_version=1 remote_resource=cpu"
    )
    return api


def validate_bundle(kernel_dir: Path) -> None:
    notebook_path = kernel_dir / "stage1-length-interactions-v1.ipynb"
    metadata_path = kernel_dir / "kernel-metadata.json"
    if not notebook_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("runner-local Stage1 bundle incomplete")
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
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise RuntimeError(f"runner-local Stage1 metadata mismatch: {key}")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if len(notebook.get("cells", [])) != 2:
        raise RuntimeError("runner-local Stage1 Notebook cell count changed")
    joined = "\n".join(
        "".join(cell.get("source", "")) if isinstance(cell.get("source", ""), list)
        else str(cell.get("source", ""))
        for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    for marker in (
        'METHOD = "plus_length_interactions_v1"',
        "EXPECTED_MODEL_FEATURES = 523",
        "EXPECTED_OOF_AUC = 0.68013844",
        '"validation_labels_read": False',
        '"validation_membership_read": False',
        '"competition_submission_performed_inside_notebook": False',
    ):
        if marker not in joined:
            raise RuntimeError(f"runner-local Stage1 marker missing: {marker}")
    for forbidden in ("KAGGLE_API_TOKEN", "competitions submit", "kernels push"):
        if forbidden in joined:
            raise RuntimeError(f"runner-local Stage1 Notebook gained forbidden capability: {forbidden}")
    compile(joined, notebook_path.name, "exec")


def execute(kaggle_bin: Path, kernel_dir: Path) -> None:
    api = live_preflight(kaggle_bin)
    validate_bundle(kernel_dir)
    counts = enforce_cpu_admission(api)
    print(
        "STAGE1_LENGTH_INTERACTION_PREWRITE_ADMISSION PASS "
        f"active_cpu={counts['cpu']} active_gpu={counts['gpu']} "
        f"active_tpu={counts['tpu']} active_unknown={counts['unknown']}"
    )
    result = subprocess.run(
        [str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir), "--timeout", "2700"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        existing = {
            str(getattr(item, "ref", ""))
            for item in (api.kernels_list(user=TARGET_OWNER, search=TARGET_SLUG, page_size=20) or [])
        }
        if TARGET in existing:
            raise RuntimeError(
                "Stage1 push returned nonzero but target exists; outcome ambiguous and retry is forbidden"
            )
        raise RuntimeError("Stage1 Kaggle push failed before confirmed creation; retry is forbidden")

    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name = TARGET_OWNER
        request.kernel_slug = TARGET_SLUG
        details = client.kernels.kernels_api_client.get_kernel(request)
    metadata = details.metadata
    if (
        int(metadata.current_version_number) != 1
        or not bool(metadata.is_private)
        or bool(metadata.enable_gpu)
        or bool(metadata.enable_tpu)
        or bool(metadata.enable_internet)
    ):
        raise RuntimeError("post-push Stage1 Kaggle metadata contract changed")
    print(
        f"STAGE1_LENGTH_INTERACTION_EXECUTION PASS request_id={REQUEST_ID} target={TARGET} "
        "version=1 accelerator=cpu automatic_compute_retries=0 competition_submission=false"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--kaggle-bin", type=Path)
    parser.add_argument("--kernel-dir", type=Path)
    args = parser.parse_args()
    if args.static == args.execute:
        raise SystemExit("choose exactly one of --static or --execute")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    validate_request(request, args.launcher, args.builder)
    validate_static(args.launcher)
    if args.static:
        print(
            "STAGE1_LENGTH_INTERACTION_STATIC PASS request=001 resource_class=cpu "
            "write_calls=1 automatic_retries=0 submissions=0"
        )
        return
    if args.kaggle_bin is None or args.kernel_dir is None:
        raise SystemExit("--execute requires --kaggle-bin and --kernel-dir")
    execute(args.kaggle_bin, args.kernel_dir)


if __name__ == "__main__":
    main()
