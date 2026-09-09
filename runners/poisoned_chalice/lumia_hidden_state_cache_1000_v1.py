#!/usr/bin/env python3
"""One-shot launcher for the frozen P1-03 1k LUMIA hidden-state cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

REQUEST_ID = "20260910-poisoned-chalice-lumia-hidden-state-cache-1000-v1-001"
TARGET = "renta0426/poisoned-chalice-lumia-hidden-state-cache-1000-v1"
RESEARCH_COMMIT = "3f0370fd3366817abdec7e6241be3d299f959181"
AUTHOR_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
MODEL_ID = "bigcode/starcoder2-3b"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
NOTEBOOK_NAME = "poisoned-chalice-lumia-hidden-state-cache-1000-v1.ipynb"
OUTPUT_PREFIX = "lumia_hidden_state_cache_1000_v1"
TOP_LEVEL_OUTPUTS = [
    "cache_manifest.json",
    "cohort_manifest.jsonl",
    "sample_metadata.jsonl",
    "runtime.json",
    "run_manifest.json",
]
SHARD_OUTPUTS = [f"shards/shard_{index:04d}.npz" for index in range(20)]
PERSISTENT_OUTPUTS = [f"{OUTPUT_PREFIX}/{name}" for name in TOP_LEVEL_OUTPUTS + SHARD_OUTPUTS]
RESOURCE = {
    "accelerator": "gpu",
    "machine_shape": "NvidiaTeslaT4",
    "expected_visible_gpu_count": 2,
    "expected_runtime_minutes": 25,
    "hard_timeout_minutes": 60,
    "max_active_runs": 1,
    "min_remaining_quota_hours": 1.0,
}
CLEAN_ROOM = {
    "new_model_specific_holdout_opened": False,
    "public_leaderboard_feedback_used": False,
    "hidden_validation_labels_used": False,
    "validation_rows_used": False,
    "codeparrot_rows_or_labels_used": False,
    "row_level_persisted_membership_labels": False,
    "performance_metrics_computed": False,
    "automatic_promotion": False,
}
SCIENTIFIC_CONTRACT = {
    "task_id": "P1-03-lumia-hidden-state-cache-1000-v1",
    "purpose": "frozen_label_clean_hidden_state_cache",
    "rows": 1000,
    "rows_per_language_label": 100,
    "selection_seed": 20260909,
    "selection_labels_removed_before_model_scoring": True,
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "dataset_revision": DATASET_REVISION,
    "author_commit": AUTHOR_COMMIT,
    "max_length": 8192,
    "precision": "fp16",
    "all_transformer_layers": True,
    "expected_layers": 30,
    "expected_hidden_size": 3072,
    "pooling_outputs_checked": ["mean", "caller_weighted", "helper_weighted"],
    "same_forward_output_baselines": [
        "zsigmoid_single_sequence",
        "caller_literal_output_weighted_single_sequence",
        "helper_language_aware_output_weighted_single_sequence",
    ],
    "shard_size": 50,
    "shard_count": 20,
    "cache_dtype": "float32",
    "lossless_compression": True,
    "raw_token_layer_activations_persisted": False,
    "logits_persisted": False,
    "performance_metrics_computed": False,
    "automatic_promotion": False,
    "competition_submission": False,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-"))


def validate_request(request: dict) -> None:
    exact = {
        "request_id": REQUEST_ID,
        "competition": "poisoned-chalice-icse27",
        "operation": "kernel_run",
        "target": TARGET,
        "launcher_path": "runners/poisoned_chalice/lumia_hidden_state_cache_1000_v1.py",
        "research_repository": "renta0426/The-Poisoned-Chalice-of-LLM-Evaluation",
        "research_commit": RESEARCH_COMMIT,
        "resource": RESOURCE,
        "api_budget": {"max_calls": 60},
        "side_effects": ["create one private notebook version"],
        "persistent_outputs": PERSISTENT_OUTPUTS,
        "automatic_compute_retries": 0,
        "enable_internet": True,
        "competition_submission": False,
        "select_as_final": False,
        "clean_room": CLEAN_ROOM,
        "scientific_contract": SCIENTIFIC_CONTRACT,
    }
    for key, value in exact.items():
        if request.get(key) != value:
            raise RuntimeError(f"LUMIA cache 1k request changed: {key}")


def validate_research(root: Path) -> None:
    paths = {
        "core": root / "src/poisoned_chalice/lumia_hidden_state_pilot.py",
        "helper": root / "src/poisoned_chalice/sersem_author_faithful.py",
        "runtime": root / "src/poisoned_chalice/sersem_author_runtime.py",
        "cache": root / "src/poisoned_chalice/lumia_hidden_state_cache.py",
        "builder": root / "scripts/build_lumia_hidden_state_cache_1000_notebook_v1.py",
        "contract": root / "configs/p1_03_lumia_hidden_state_cache_1000_v1_20260910.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f"research source missing: {path}")
    for key in ("core", "helper", "runtime", "cache", "builder"):
        compile(paths[key].read_text(encoding="utf-8"), str(paths[key]), "exec")
    core = paths["core"].read_text(encoding="utf-8")
    helper = paths["helper"].read_text(encoding="utf-8")
    runtime = paths["runtime"].read_text(encoding="utf-8")
    cache = paths["cache"].read_text(encoding="utf-8")
    builder = paths["builder"].read_text(encoding="utf-8")
    contract = load_json(paths["contract"])

    for marker in (
        AUTHOR_COMMIT, MODEL_REVISION, DATASET_REVISION,
        "max_length: int = 8192", "register_forward_hook",
        'return cohort[["sample_id", "language", "content"]].copy()',
        "caller_weighted", "helper_weighted",
    ):
        if marker not in core:
            raise RuntimeError(f"scientific core marker missing: {marker}")
    for marker in ("build_author_character_weights", "token_weights_from_character_weights", "weighted_output_score"):
        if marker not in helper:
            raise RuntimeError(f"author helper marker missing: {marker}")
    for marker in ("AUTHOR_RUNTIME_PINS", '"flake8": "7.3.0"', "load_author_runtime"):
        if marker not in runtime:
            raise RuntimeError(f"author runtime marker missing: {marker}")
    for marker in (
        "rows: int = 1000", "rows_per_language_label: int = 100",
        "shard_size: int = 50", 'dtype: str = "float32"',
        'compression: str = "npz_compressed"',
        "run_hidden_state_cache", "shard_manifest", "serialization_seconds",
        '"labels_present_during_model_scoring": False',
        '"raw_token_layer_activations_persisted": False',
        '"logits_persisted": False',
    ):
        if marker not in cache:
            raise RuntimeError(f"cache source marker missing: {marker}")
    for marker in (
        TARGET, 'model = model.to("cuda:0").eval()',
        'assert layer_count == expected_layers == 30', 'assert hidden_size == 3072',
        "run_hidden_state_cache(model, tokenizer, cohort, runtime, CONFIG)",
        'len(cache["shards"]) == 20', 'for index in range(20)',
        'f"shards/shard_{index:04d}.npz"',
        "LUMIA_HIDDEN_STATE_CACHE_1000_V1 COMPLETE",
        'SITE_ROOT = Path("/tmp/lumia-cache-1000-v1-site")',
    ):
        if marker not in builder:
            raise RuntimeError(f"builder marker missing: {marker}")
    for forbidden in (
        'device_map="cuda"', "roc_auc_score", "roc_curve(", "average_precision_score",
        "competition_submit(", "submission.csv",
    ):
        if forbidden in builder or forbidden in cache:
            raise RuntimeError(f"forbidden GPU-cache code: {forbidden}")

    if contract.get("task_id") != SCIENTIFIC_CONTRACT["task_id"]:
        raise RuntimeError("cache contract task identity changed")
    if contract.get("target") != TARGET or contract.get("status") != "implementation_pending_execution":
        raise RuntimeError("cache target/status changed")
    gate = contract.get("runtime_gate_pass") or {}
    if gate.get("passed") is not True or float(gate.get("predeclared_gate_minutes", 0)) != 105.0:
        raise RuntimeError("50-row runtime gate provenance changed")
    model = contract.get("model") or {}
    if model.get("revision") != MODEL_REVISION or model.get("max_length") != 8192:
        raise RuntimeError("cache model contract changed")
    cohort = contract.get("cohort") or {}
    if cohort.get("rows") != 1000 or cohort.get("rows_per_language_label") != 100:
        raise RuntimeError("cache cohort contract changed")
    cache_contract = contract.get("cache") or {}
    if cache_contract.get("shard_size") != 50 or cache_contract.get("shard_count") != 20:
        raise RuntimeError("cache shard contract changed")
    if cache_contract.get("dtype") != "float32" or cache_contract.get("raw_token_layer_activations_persisted") is not False:
        raise RuntimeError("cache numerical/persistence contract changed")


def write_nbformat_shim(root: Path) -> None:
    (root / "nbformat.py").write_text(r'''import json
class A(dict):
    def __getattr__(self,n):
        try:return self[n]
        except KeyError as e:raise AttributeError(n) from e
    def __setattr__(self,n,v):self[n]=v
class V4:
    def new_notebook(self):return A(cells=[],metadata=A(),nbformat=4,nbformat_minor=5)
    def new_markdown_cell(self,source=""):return A(cell_type="markdown",metadata=A(),source=source)
    def new_code_cell(self,source=""):return A(cell_type="code",execution_count=None,metadata=A(),outputs=[],source=source)
v4=V4()
def write(nb,path):
    with open(path,"w",encoding="utf-8") as f:json.dump(nb,f,ensure_ascii=False,indent=1);f.write("\n")
''', encoding="utf-8")


def validate_kernel(root: Path) -> None:
    observed = sorted(path.name for path in root.iterdir() if path.is_file())
    if observed != ["kernel-metadata.json", NOTEBOOK_NAME]:
        raise RuntimeError(f"unexpected kernel files: {observed}")
    metadata = load_json(root / "kernel-metadata.json")
    expected = {
        "id": TARGET,
        "title": "Poisoned Chalice Lumia Hidden State Cache 1000 V1",
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"kernel metadata changed: {key}")
    if slugify(metadata["title"]) != TARGET.split("/", 1)[1]:
        raise RuntimeError("kernel title-derived slug mismatch")
    for key in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
        if metadata.get(key) != []:
            raise RuntimeError(f"unexpected attached source: {key}")
    notebook = load_json(root / NOTEBOOK_NAME)
    cells = notebook.get("cells") or []
    if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor") != 5 or len(cells) != 4:
        raise RuntimeError("unexpected notebook structure")
    ids = [cell.get("id") for cell in cells]
    if len(set(ids)) != 4 or not all(isinstance(value, str) and value for value in ids):
        raise RuntimeError("invalid notebook cell IDs")
    code = "\n".join(str(cell.get("source", "")) for cell in cells)
    for marker in (
        AUTHOR_COMMIT, MODEL_REVISION, DATASET_REVISION,
        "rows: int = 1000", "rows_per_language_label: int = 100",
        "max_length: int = 8192", "shard_size: int = 50",
        'model = model.to("cuda:0").eval()',
        "run_hidden_state_cache(model, tokenizer, cohort, runtime, CONFIG)",
        'len(cache["shards"]) == 20', 'for index in range(20)',
        "LUMIA_HIDDEN_STATE_CACHE_1000_V1 COMPLETE",
        "serialization_seconds", "scoring_peak_allocated_bytes",
    ):
        if marker not in code:
            raise RuntimeError(f"generated notebook marker missing: {marker}")
    for forbidden in (
        'device_map="cuda"', "roc_auc_score", "roc_curve(", "average_precision_score",
        "competition_submit(", "submission.csv",
    ):
        if forbidden in code:
            raise RuntimeError(f"forbidden generated notebook code: {forbidden}")


def materialize(request_path: Path, research_root: Path, kernel_dir: Path) -> None:
    validate_request(load_json(request_path))
    validate_research(research_root)
    with tempfile.TemporaryDirectory(prefix="lumia-cache-1000-nbformat-") as temp:
        shim = Path(temp)
        write_nbformat_shim(shim)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(shim) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        process = subprocess.run(
            [sys.executable, str(research_root / "scripts/build_lumia_hidden_state_cache_1000_notebook_v1.py")],
            cwd=str(research_root), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=120, check=False,
        )
    if process.returncode != 0:
        diagnostic = (process.stdout + process.stderr).encode("utf-8", errors="replace")
        raise RuntimeError(
            f"1k cache notebook builder failed rc={process.returncode} "
            f"diagnostic_sha256={hashlib.sha256(diagnostic).hexdigest()}"
        )
    built = research_root / "notebooks/experiments/poisoned-chalice-lumia-hidden-state-cache-1000-v1"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    shutil.copytree(built, kernel_dir)
    validate_kernel(kernel_dir)
    print("LUMIA_HIDDEN_STATE_CACHE_1000_V1_MATERIALIZE PASS rows=1000 shards=20 max_length=8192 labels_in_scoring=0 metrics=0 gpu=T4 retries=0")


def execute(request_path: Path, kernel_dir: Path, kaggle_bin: Path) -> None:
    validate_request(load_json(request_path))
    validate_kernel(kernel_dir)
    if not kaggle_bin.is_file():
        raise RuntimeError("locked Kaggle CLI missing")
    process = subprocess.run(
        [str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False,
    )
    stdout, stderr = process.stdout or b"", process.stderr or b""
    if process.returncode:
        print(
            f"LUMIA_HIDDEN_STATE_CACHE_1000_V1_AMBIGUOUS_WRITE rc={process.returncode} "
            f"stdout_bytes={len(stdout)} stderr_bytes={len(stderr)} "
            f"stdout_sha256={hashlib.sha256(stdout).hexdigest()} "
            f"stderr_sha256={hashlib.sha256(stderr).hexdigest()}"
        )
        raise RuntimeError("kaggle kernels push returned non-zero; no retry permitted")
    print(
        f"LUMIA_HIDDEN_STATE_CACHE_1000_V1_LAUNCH_ACCEPTED target={TARGET} "
        f"stdout_bytes={len(stdout)} stderr_bytes={len(stderr)} "
        f"stdout_sha256={hashlib.sha256(stdout).hexdigest()} "
        f"stderr_sha256={hashlib.sha256(stderr).hexdigest()} retries=0 submissions=0"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--kernel-dir", type=Path)
    parser.add_argument("--kaggle-bin", type=Path)
    args = parser.parse_args()
    if args.static:
        if args.research_root is None:
            parser.error("--research-root required for --static")
        validate_request(load_json(args.request))
        validate_research(args.research_root)
        print("LUMIA_HIDDEN_STATE_CACHE_1000_V1_STATIC PASS rows=1000 shards=20 metrics=0")
        return 0
    if args.materialize:
        if args.research_root is None or args.kernel_dir is None:
            parser.error("--research-root and --kernel-dir required for --materialize")
        materialize(args.request, args.research_root, args.kernel_dir)
        return 0
    if args.kernel_dir is None or args.kaggle_bin is None:
        parser.error("--kernel-dir and --kaggle-bin required for --execute")
    execute(args.request, args.kernel_dir, args.kaggle_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
