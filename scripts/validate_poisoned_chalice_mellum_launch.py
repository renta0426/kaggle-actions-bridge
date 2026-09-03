"""Static, secret-free validation for the frozen Mellum-4B Kaggle launch."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile


EXPECTED_LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
EXPECTED_REQUEST = {
    "schema_version": 1,
    "request_id": "20260903-poisoned-chalice-mellum-transfer-v1-001",
    "competition": "poisoned-chalice-icse27",
    "operation": "kernel_run",
    "target": "renta0426/mellum-transfer-v1",
    "source_kernel": "renta0426/pseudo-stage2-starcoder2-7b-v1",
    "source_expected_version": 1,
    "builder_path": "scripts/poisoned_chalice_mellum_transfer_builder.py",
    "builder_blob_sha": "0b8b48b2547454affbde6e730d4e43bde39421cf",
    "nbformat_shim_path": "scripts/poisoned_chalice_nbformat_minimal.py",
    "nbformat_shim_blob_sha": "9e91545a7f3318ca7e033c4f49f0eeda64ce4bfd",
    "canonical_research_commit": "9f6be68c27dc1f3326d68bed4e2abf80db893748",
    "stage1_prerequisite": {
        "submission_status": "complete",
        "public_score": 0.34575,
        "score_source": "user_reported_pending_read_only_api_identity_binding",
    },
    "target_model": {
        "id": "JetBrains/Mellum-4b-base",
        "revision": "83cce2605fbdf6a3868627e9b0a5924e0072b94d",
    },
    "source_data": {
        "repository": "https://github.com/Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026",
        "commit": "413f56040e5b4805bcf15ed794dec56bc4e16b41",
        "path": "data/7b_train_test/eval_results.parquet",
        "sha256": "6e8559279e8ebd94ec8c25561c91d38c454a4e109ea0813b07864ff6a66d4068",
        "rows": 2000,
        "current_stage1_overlap_excluded": 23,
    },
    "resource": {
        "accelerator": "gpu",
        "machine_shape": "NvidiaTeslaT4",
        "expected_runtime_minutes": 120,
        "hard_timeout_minutes": 180,
        "min_remaining_quota_hours": 4.0,
        "max_active_account_runs": 0,
    },
    "api_budget": {
        "max_calls": 60,
        "max_recent_kernels_inspected": 25,
        "poll_interval_seconds": 300,
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


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def load_python(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_synthetic_source(path: Path) -> None:
    starter = '''
from dataclasses import dataclass
DEFAULT_MODEL = "bigcode/starcoder2-3b"
@dataclass
class StarterPlusConfig:
    max_length: int = 768
    max_batch_tokens: int = 1536
    vocab_chunk_tokens: int = 64
    train_samples_per_language: int = 400
def dynamic_batches(*args, **kwargs): return []
def summarize_tokens(*args, **kwargs): return {}
def seed_everything(seed): return None
def make_window_records(*args, **kwargs): return []
def aggregate_windows(*args, **kwargs): return None
def extract_window_features(*args, **kwargs): return None
'''
    feature = '''
from dataclasses import dataclass
from .starter_plus import StarterPlusConfig, dynamic_batches, summarize_tokens
@dataclass
class FeatureCacheV2Config:
    model_id: str = "bigcode/starcoder2-3b"
    model_revision: str = "test"
    max_length: int = 768
    max_batch_tokens: int = 1536
    vocab_chunk_tokens: int = 64
    rank_vocab_block_size: int = 8192
    train_samples_per_language: int = 400
    samples_per_shard: int = 100
    output_dir: str = "out"
    feature_version: str = "test"
    attention_implementation: str = "sdpa"
    retain_token_statistics: bool = False
def extract_window_features_optimized(*args, **kwargs): return None, None, {}
def compare_feature_frames(*args, **kwargs): return {"passed": True}
STANDARD_MINKPP_COLUMN = "min_kpp_zselect_10"
'''
    records = []
    for language in EXPECTED_LANGUAGES:
        for label in (0, 1):
            membership = "member" if label else "non-member"
            for index in range(200):
                digest = hashlib.sha256(
                    f"{language}:{label}:{index}".encode("utf-8")
                ).hexdigest()
                records.append(
                    {
                        "sample_id": f"previous-{language.lower()}-{digest}",
                        "content_sha256": digest,
                        "source_row_index": len(records),
                        "language": language,
                        "membership": membership,
                        "label": label,
                        "split_role": "eval",
                    }
                )
    manifest_source = "import pandas as pd\nTRANSFER_MANIFEST = pd.DataFrame(" + repr(records) + ")\n"
    notebook = {
        "cells": [
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": starter},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": feature},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": manifest_source},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


def extract_prediction_records(notebook: dict) -> list[dict]:
    matches: list[list[dict]] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else str(source)
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
                raise RuntimeError("prediction manifest is not one literal DataFrame call")
            value = ast.literal_eval(call.args[0])
            matches.append(value)
    if len(matches) != 1:
        raise RuntimeError(f"expected one embedded prediction manifest, got {len(matches)}")
    return matches[0]


def validate_workflow(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    required = (
        "permissions: {}",
        "group: kaggle-resource-global",
        "environment: kaggle-readonry",
        "runs-on: ubuntu-24.04",
        "automatic_compute_retries=0",
        "max_active_account_runs=0",
        "KAGGLE_API_TOKEN: ${{ secrets.KAGGLE_API_TOKEN }}",
        "kaggle\" kernels push",
        "--timeout 10800",
        "competition_submission=false",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(f"launch workflow missing security markers: {missing}")
    forbidden = (
        "pull_request_target:",
        "issue_comment:",
        "repository_dispatch:",
        "runs-on: self-hosted",
        "actions/checkout@",
        "competitions submit",
        "datasets create",
        "datasets version",
        "kernels delete",
        "--public",
    )
    present = [token for token in forbidden if token in text]
    if present:
        raise RuntimeError(f"launch workflow contains forbidden operations: {present}")
    if text.count("kernels push") != 1:
        raise RuntimeError("launch workflow must contain exactly one kernel push")
    if text.count("secrets.KAGGLE_API_TOKEN") != 2:
        raise RuntimeError("Kaggle token must be scoped to exactly two bounded steps")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--shim", type=Path, required=True)
    parser.add_argument("--launch-workflow", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request != EXPECTED_REQUEST:
        raise RuntimeError("Mellum launch request differs from the frozen exact contract")
    for path, expected in (
        (args.builder, request["builder_blob_sha"]),
        (args.shim, request["nbformat_shim_blob_sha"]),
    ):
        data = path.read_bytes()
        if git_blob_sha(data) != expected:
            raise RuntimeError(f"Git blob mismatch: {path}")
        compile(data, str(path), "exec")

    builder_text = args.builder.read_text(encoding="utf-8")
    builder_markers = (
        'EXPECTED_ROWS = 2_000',
        'MODEL_ID = "JetBrains/Mellum-4b-base"',
        'MODEL_REVISION = "83cce2605fbdf6a3868627e9b0a5924e0072b94d"',
        'SOURCE_KERNEL',
        'min_kpp_zselect_10__max',
        'best_local_64__max',
        'target_labels_embedded_in_gpu_notebook=False',
        'public_leaderboard_tuning_used=False',
        '"is_private": True',
        '"enable_gpu": True',
        '"enable_tpu": False',
        '"machine_shape": "NvidiaTeslaT4"',
    )
    missing = [token for token in builder_markers if token not in builder_text]
    if missing:
        raise RuntimeError(f"builder differs from frozen scientific contract: {missing}")
    validate_workflow(args.launch_workflow)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.ipynb"
        output = root / "output"
        make_synthetic_source(source)
        shim = load_python(args.shim, "poisoned_chalice_nbformat_shim")
        previous = sys.modules.get("nbformat")
        sys.modules["nbformat"] = shim
        try:
            builder = load_python(args.builder, "poisoned_chalice_mellum_builder")
            notebook_path, metadata_path = builder.build(source, output)
        finally:
            if previous is None:
                sys.modules.pop("nbformat", None)
            else:
                sys.modules["nbformat"] = previous
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if len(notebook.get("cells", [])) != 9:
            raise RuntimeError("generated Mellum notebook cell count changed")
        records = extract_prediction_records(notebook)
        if len(records) != 2000:
            raise RuntimeError("generated prediction cohort size changed")
        expected_columns = {"sample_id", "content_sha256", "language", "sample_index"}
        if any(set(row) != expected_columns for row in records):
            raise RuntimeError("label or unexpected field crossed the generated GPU boundary")
        if [row["sample_index"] for row in records] != list(range(2000)):
            raise RuntimeError("generated prediction cohort order changed")
        expected_metadata = {
            "id": "renta0426/mellum-transfer-v1",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
        }
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"generated metadata mismatch: {key}")
        if metadata.get("competition_sources") != [] or metadata.get("kernel_sources") != []:
            raise RuntimeError("generated GPU notebook has undeclared Kaggle data sources")

    print(
        "MELLUM_LAUNCH_STATIC PASS rows=2000 labels_embedded=false "
        "accelerator=gpu machine=NvidiaTeslaT4 retries=0 submissions=0"
    )


if __name__ == "__main__":
    main()
