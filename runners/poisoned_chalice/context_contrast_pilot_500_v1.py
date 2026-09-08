#!/usr/bin/env python3
"""Validate, materialize, and one-shot launch the P2-01 500-row runtime/fidelity pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

REQUEST_ID = "20260908-poisoned-chalice-context-contrast-pilot-500-v1-001"
TARGET = "renta0426/poisoned-chalice-context-contrast-pilot-500-v1"
RESEARCH_COMMIT = "74d04a7f16d7ec1403ae859217e65a7adc3e40d6"
MODEL_ID = "bigcode/starcoder2-3b"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
EXPERIMENT_ID = "matched-target-context-contrast-v1-pilot-500"
NOTEBOOK_NAME = "matched-target-context-contrast-v1-pilot-500.ipynb"
OUTPUT_PREFIX = "matched-target-context-contrast-v1-pilot-500"
PERSISTENT_OUTPUTS = [
    f"{OUTPUT_PREFIX}/sample_manifest.parquet",
    f"{OUTPUT_PREFIX}/span_manifest.parquet",
    f"{OUTPUT_PREFIX}/condition_scores.parquet",
    f"{OUTPUT_PREFIX}/span_contrasts.parquet",
    f"{OUTPUT_PREFIX}/file_features.parquet",
    f"{OUTPUT_PREFIX}/fidelity.json",
    f"{OUTPUT_PREFIX}/runtime.json",
    f"{OUTPUT_PREFIX}/run_manifest.json",
    f"{OUTPUT_PREFIX}/REPORT.md",
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _slugify(title: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")
    return re.sub(r"-+", "-", value)


def validate_request(request: dict) -> None:
    exact = {
        "request_id": REQUEST_ID,
        "competition": "poisoned-chalice-icse27",
        "operation": "kernel_run",
        "target": TARGET,
        "research_repository": "renta0426/The-Poisoned-Chalice-of-LLM-Evaluation",
        "research_commit": RESEARCH_COMMIT,
        "automatic_compute_retries": 0,
        "enable_internet": True,
        "competition_submission": False,
        "select_as_final": False,
        "side_effects": ["create one private notebook version"],
        "persistent_outputs": PERSISTENT_OUTPUTS,
    }
    for key, value in exact.items():
        if request.get(key) != value:
            raise RuntimeError(f"P2-01 request changed: {key}")
    expected_resource = {
        "accelerator": "gpu",
        "machine_shape": "NvidiaTeslaT4",
        "expected_visible_gpu_count": 2,
        "expected_runtime_minutes": 60,
        "hard_timeout_minutes": 120,
        "max_active_runs": 2,
        "min_remaining_quota_hours": 1.5,
    }
    if request.get("resource") != expected_resource:
        raise RuntimeError("P2-01 resource contract changed")
    if request.get("api_budget") != {"max_calls": 60}:
        raise RuntimeError("P2-01 API budget changed")
    expected_clean = {
        "stage2_holdout_consumed": False,
        "public_leaderboard_feedback_used": False,
        "hidden_validation_labels_used": False,
        "validation_rows_used": False,
        "codeparrot_rows_or_labels_used": False,
        "performance_evaluation_forbidden": True,
    }
    if request.get("clean_room") != expected_clean:
        raise RuntimeError("P2-01 clean-room contract changed")
    expected_science = {
        "task_id": "P2-01",
        "rows": 500,
        "per_language_per_label": 50,
        "selection_labels_removed_before_model_scoring": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "context_budgets": [32, 128, 512],
        "target_tokens": 256,
        "min_target_tokens": 64,
        "spans_per_file": 3,
        "max_model_tokens_per_window": 768,
        "shared_target_identity_required": True,
        "condition_metrics": ["mean_logp", "paper_minkpp10", "local64"],
        "paper_schema": "prob_weighted_minkpp_v1",
        "paper_fraction": 0.1,
        "local_width": 64,
        "cpu_oracle_required": True,
        "dense_blocked_parity_required": True,
        "cpu_gpu_reduction_tolerance": 0.00002,
        "performance_metrics_computed": False,
    }
    if request.get("scientific_contract") != expected_science:
        raise RuntimeError("P2-01 scientific contract changed")


def validate_research(research_root: Path) -> None:
    context = research_root / "src/poisoned_chalice/context_contrast.py"
    paper = research_root / "src/poisoned_chalice/minkpp_paper.py"
    builder = research_root / "scripts/build_context_contrast_pilot_notebook.py"
    experiment_path = research_root / "experiments/matched-target-context-contrast-v1-pilot-500/experiment.json"
    for path in (context, paper, builder, experiment_path):
        if not path.is_file():
            raise RuntimeError(f"research source missing: {path}")
    context_source = context.read_text(encoding="utf-8")
    paper_source = paper.read_text(encoding="utf-8")
    builder_source = builder.read_text(encoding="utf-8")
    compile(context_source, str(context), "exec")
    compile(paper_source, str(paper), "exec")
    compile(builder_source, str(builder), "exec")
    for marker in (
        "tokenize_with_byte_offsets", "select_target_spans", "build_context_windows",
        "summarize_shared_target_scores", "contrast_deltas", "aggregate_span_contrasts",
        "context_budgets: tuple[int, ...] = (32, 128, 512)",
    ):
        if marker not in context_source:
            raise RuntimeError(f"context source marker missing: {marker}")
    for marker in (
        "probability_weighted_token_statistics_dense",
        "probability_weighted_token_statistics_blocked",
        'MINKPP_PAPER_SCHEMA_VERSION = "prob_weighted_minkpp_v1"',
    ):
        if marker not in paper_source:
            raise RuntimeError(f"paper source marker missing: {marker}")
    for marker in (
        MODEL_ID, MODEL_REVISION, DATASET_REVISION,
        'context_budgets=(32, 128, 512)', 'target_tokens=256', 'min_target_tokens=64',
        'spans_per_file=3', 'cpu_gpu_reduction_tolerance: float = 2e-5',
        '"id": "renta0426/poisoned-chalice-context-contrast-pilot-500-v1"',
        '"is_private": True', '"enable_gpu": True', '"enable_tpu": False',
        '"enable_internet": True', '"performance_metrics_computed": False',
    ):
        if marker not in builder_source:
            raise RuntimeError(f"P2-01 builder marker missing: {marker}")
    for forbidden in ("roc_auc_score", "average_precision_score", "competition_submit(", "submission.csv"):
        if forbidden in builder_source:
            raise RuntimeError(f"performance/submission code leaked into runtime pilot: {forbidden}")
    experiment = _load(experiment_path)
    if experiment.get("experiment_id") != EXPERIMENT_ID or experiment.get("task_id") != "P2-01":
        raise RuntimeError("P2-01 experiment identity changed")
    if experiment.get("status") != "implementation_complete_execution_pending":
        raise RuntimeError("P2-01 pre-run status changed")
    if experiment.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise RuntimeError("P2-01 model contract changed")
    if experiment.get("data", {}).get("rows") != 500:
        raise RuntimeError("P2-01 row contract changed")
    if experiment.get("data", {}).get("selection_labels_removed_before_model_scoring") is not True:
        raise RuntimeError("P2-01 label-separation contract changed")
    context_contract = experiment.get("context_contract") or {}
    if context_contract.get("context_budgets") != [32, 128, 512] or context_contract.get("shared_target_identity_required") is not True:
        raise RuntimeError("P2-01 shared-target context contract changed")
    if experiment.get("runtime_contract", {}).get("automatic_compute_retries") != 0:
        raise RuntimeError("P2-01 automatic compute retry guard changed")
    forbidden = set(experiment.get("forbidden_in_this_pilot") or [])
    if not {"AUC", "TPR", "pAUC", "validation rows", "CodeParrot rows or labels", "competition submission"}.issubset(forbidden):
        raise RuntimeError("P2-01 forbidden-operation guard changed")


def validate_kernel(kernel_dir: Path) -> None:
    notebook_path = kernel_dir / NOTEBOOK_NAME
    metadata_path = kernel_dir / "kernel-metadata.json"
    observed = sorted(path.name for path in kernel_dir.iterdir() if path.is_file())
    if observed != ["kernel-metadata.json", NOTEBOOK_NAME]:
        raise RuntimeError(f"unexpected kernel package files: {observed}")
    metadata = _load(metadata_path)
    expected_metadata = {
        "id": TARGET,
        "title": "Poisoned Chalice Context Contrast Pilot 500 V1",
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"kernel metadata changed: {key}")
    if _slugify(metadata["title"]) != TARGET.split("/", 1)[1]:
        raise RuntimeError("kernel title-derived slug does not match target")
    for key in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
        if metadata.get(key) != []:
            raise RuntimeError(f"unexpected attached source: {key}")
    notebook = _load(notebook_path)
    if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor") != 5 or len(notebook.get("cells") or []) != 9:
        raise RuntimeError("unexpected P2-01 notebook structure")
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    if len(set(cell_ids)) != len(cell_ids) or not all(isinstance(value, str) and value for value in cell_ids):
        raise RuntimeError("invalid P2-01 notebook cell ids")
    sources = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        sources.append(str(source))
    code = "\n".join(sources)
    for marker in (
        MODEL_REVISION, DATASET_REVISION, "context_budgets=(32, 128, 512)",
        "target_token_sha256", "shared_target_identity_failures",
        "cpu_gpu_reduction_max_abs_error", "real_dense_blocked_max_z_diff",
        '"performance_metrics_computed": False', "performance_metrics=0 submissions=0",
    ):
        if marker not in code:
            raise RuntimeError(f"generated P2-01 notebook marker missing: {marker}")
    for forbidden in ("roc_auc_score", "average_precision_score", "competition_submit(", "submission.csv"):
        if forbidden in code:
            raise RuntimeError(f"forbidden runtime-pilot code in notebook: {forbidden}")


def materialize(request_path: Path, research_root: Path, kernel_dir: Path) -> None:
    request = _load(request_path)
    validate_request(request)
    validate_research(research_root)
    builder = research_root / "scripts/build_context_contrast_pilot_notebook.py"
    completed = subprocess.run(
        [sys.executable, str(builder)],
        cwd=str(research_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stdout + completed.stderr).encode("utf-8", errors="replace")
        raise RuntimeError(
            "P2-01 notebook builder failed "
            f"rc={completed.returncode} diagnostic_sha256={hashlib.sha256(diagnostic).hexdigest()}"
        )
    built_dir = research_root / "notebooks/experiments/matched-target-context-contrast-v1-pilot-500"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    shutil.copytree(built_dir, kernel_dir)
    validate_kernel(kernel_dir)
    print("P2_CONTEXT_RUNTIME_PILOT_MATERIALIZE PASS rows=500 files=2 gpu=T4 retries=0 performance_metrics=0 submissions=0")


def execute(request_path: Path, kernel_dir: Path, kaggle_bin: Path) -> None:
    request = _load(request_path)
    validate_request(request)
    validate_kernel(kernel_dir)
    if not kaggle_bin.is_file():
        raise RuntimeError("locked Kaggle CLI missing")
    completed = subprocess.run(
        [str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        check=False,
    )
    out, err = completed.stdout or b"", completed.stderr or b""
    if completed.returncode != 0:
        print(
            "P2_CONTEXT_RUNTIME_PILOT_AMBIGUOUS_WRITE "
            f"rc={completed.returncode} stdout_bytes={len(out)} stderr_bytes={len(err)} "
            f"stdout_sha256={hashlib.sha256(out).hexdigest()} stderr_sha256={hashlib.sha256(err).hexdigest()}"
        )
        raise RuntimeError("kaggle kernels push returned non-zero; no retry permitted")
    print(
        "P2_CONTEXT_RUNTIME_PILOT_LAUNCH_ACCEPTED "
        f"target={TARGET} accelerator=gpu machine=NvidiaTeslaT4 retries=0 performance_metrics=0 submissions=0"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--request", required=True)
    parser.add_argument("--research-root")
    parser.add_argument("--kernel-dir")
    parser.add_argument("--kaggle-bin")
    args = parser.parse_args()
    request_path = Path(args.request)
    request = _load(request_path)
    validate_request(request)
    if args.static:
        if not args.research_root:
            raise RuntimeError("--research-root required")
        validate_research(Path(args.research_root))
        print("P2_CONTEXT_RUNTIME_PILOT_STATIC PASS rows=500 contexts=3 retries=0 performance_metrics=0 submissions=0")
        return 0
    if args.materialize:
        if not args.research_root or not args.kernel_dir:
            raise RuntimeError("--research-root and --kernel-dir required")
        materialize(request_path, Path(args.research_root), Path(args.kernel_dir))
        return 0
    if not args.kernel_dir or not args.kaggle_bin:
        raise RuntimeError("--kernel-dir and --kaggle-bin required")
    execute(request_path, Path(args.kernel_dir), Path(args.kaggle_bin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
