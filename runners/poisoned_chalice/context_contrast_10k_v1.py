#!/usr/bin/env python3
"""Validate, materialize, and one-shot launch P2-01 fixed-10k development evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

REQUEST_ID = "20260908-poisoned-chalice-context-contrast-10k-v1-001"
TARGET = "renta0426/poisoned-chalice-context-contrast-10k-v1"
RESEARCH_COMMIT = "7492fc68784c505cce409bd5f1e1072cc0e44511"
MODEL_ID = "bigcode/starcoder2-3b"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
EXPERIMENT_ID = "matched-target-context-contrast-v1-10k"
NOTEBOOK_NAME = "matched-target-context-contrast-v1-10k.ipynb"
OUTPUT_PREFIX = EXPERIMENT_ID
PRIMARY_FEATURES = [
    "delta_512_32_mean_logp__max_span",
    "delta_512_32_paper_minkpp10__max_span",
    "delta_512_32_local64__max_span",
]
PERSISTENT_OUTPUTS = [
    f"{OUTPUT_PREFIX}/sample_manifest.parquet",
    f"{OUTPUT_PREFIX}/file_features.parquet",
    f"{OUTPUT_PREFIX}/span_contrasts.parquet",
    f"{OUTPUT_PREFIX}/baseline_features.parquet",
    f"{OUTPUT_PREFIX}/fidelity.json",
    f"{OUTPUT_PREFIX}/runtime.json",
    f"{OUTPUT_PREFIX}/coverage_by_language_label.csv",
    f"{OUTPUT_PREFIX}/predictor_metrics.csv",
    f"{OUTPUT_PREFIX}/language_metrics.csv",
    f"{OUTPUT_PREFIX}/length_metrics.csv",
    f"{OUTPUT_PREFIX}/strict_metrics.csv",
    f"{OUTPUT_PREFIX}/detection_jaccard.csv",
    f"{OUTPUT_PREFIX}/unique_true_positives.csv",
    f"{OUTPUT_PREFIX}/bootstrap.json",
    f"{OUTPUT_PREFIX}/evaluation_summary.json",
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
            raise RuntimeError(f"P2-01 10k request changed: {key}")
    expected_resource = {
        "accelerator": "gpu",
        "machine_shape": "NvidiaTeslaT4",
        "expected_visible_gpu_count": 2,
        "expected_runtime_minutes": 90,
        "hard_timeout_minutes": 180,
        "max_active_runs": 2,
        "min_remaining_quota_hours": 2.0,
    }
    if request.get("resource") != expected_resource:
        raise RuntimeError("P2-01 10k resource contract changed")
    if request.get("api_budget") != {"max_calls": 60}:
        raise RuntimeError("P2-01 10k API budget changed")
    expected_clean = {
        "new_model_specific_holdout_opened": False,
        "public_leaderboard_feedback_used": False,
        "hidden_validation_labels_used": False,
        "validation_rows_used": False,
        "codeparrot_rows_or_labels_used": False,
        "row_level_persisted_membership_labels": False,
        "automatic_promotion": False,
    }
    if request.get("clean_room") != expected_clean:
        raise RuntimeError("P2-01 10k clean-room contract changed")
    expected_science = {
        "task_id": "P2-01",
        "rows": 10000,
        "samples_per_language": 2000,
        "per_language_per_label": 1000,
        "selection_seed": 2027,
        "selection_labels_removed_before_model_scoring": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "context_budgets": [32, 128, 512],
        "target_tokens": 256,
        "min_target_tokens": 64,
        "spans_per_file": 3,
        "primary_features": PRIMARY_FEATURES,
        "fallback_never_imputed_as_primary": True,
        "baseline_scores": ["loss_multiwindow", "paper_minkpp10", "local_64"],
        "historical_loss_local_first8_fidelity_required": True,
        "max_batch_tokens": 12288,
        "samples_per_shard": 250,
        "bootstrap_replicates": 1000,
        "automatic_promotion": False,
        "competition_submission": False,
    }
    if request.get("scientific_contract") != expected_science:
        raise RuntimeError("P2-01 10k scientific contract changed")


def validate_research(research_root: Path) -> None:
    context = research_root / "src/poisoned_chalice/context_contrast.py"
    paper = research_root / "src/poisoned_chalice/minkpp_paper.py"
    evaluation = research_root / "src/poisoned_chalice/evaluation.py"
    builder = research_root / "scripts/build_context_contrast_10k_notebook.py"
    experiment_path = research_root / "experiments/matched-target-context-contrast-v1-10k/experiment.json"
    for path in (context, paper, evaluation, builder, experiment_path):
        if not path.is_file():
            raise RuntimeError(f"research source missing: {path}")
    context_source = context.read_text(encoding="utf-8")
    paper_source = paper.read_text(encoding="utf-8")
    evaluation_source = evaluation.read_text(encoding="utf-8")
    builder_source = builder.read_text(encoding="utf-8")
    for source, path in ((context_source, context), (paper_source, paper), (evaluation_source, evaluation), (builder_source, builder)):
        compile(source, str(path), "exec")
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
        "low_fpr_metrics", "stratified_splits", "grouped_splits",
        "leave_one_language_out_splits", "length_holdout_splits",
        "overlap_tables", "stratified_bootstrap_metrics", "minhash_groups",
    ):
        if marker not in evaluation_source:
            raise RuntimeError(f"evaluation source marker missing: {marker}")
    for marker in (
        MODEL_ID, MODEL_REVISION, DATASET_REVISION,
        "samples_per_language: int = 2000", "samples_per_shard: int = 250",
        "max_batch_tokens: int = 12288", 'split="train"',
        'context_budgets=(32, 128, 512)', 'target_tokens=256', 'min_target_tokens=64',
        'spans_per_file=3', "EXPECTED_FIRST8", "historical_first8_max_abs_diff",
        "low_fpr_metrics", "minhash_group_5fold", "leave_one_language_out",
        "length_holdout", "bootstrap_replicates: int = 1000",
        '"id":"renta0426/poisoned-chalice-context-contrast-10k-v1"',
        '"is_private":True', '"enable_gpu":True', '"enable_tpu":False',
        '"enable_internet":True', '"automatic_promotion":False',
        '"competition_submission":False',
    ):
        if marker not in builder_source:
            raise RuntimeError(f"P2-01 10k builder marker missing: {marker}")
    for feature in PRIMARY_FEATURES:
        if feature not in builder_source:
            raise RuntimeError(f"primary feature missing from 10k builder: {feature}")
    for forbidden in ("competition_submit(", "submission.csv", 'split="validation"', "CodeParrot/codeparrot"):
        if forbidden in builder_source:
            raise RuntimeError(f"forbidden 10k code present: {forbidden}")
    experiment = _load(experiment_path)
    if experiment.get("experiment_id") != EXPERIMENT_ID or experiment.get("task_id") != "P2-01":
        raise RuntimeError("P2-01 10k experiment identity changed")
    if experiment.get("status") != "implementation_complete_execution_pending":
        raise RuntimeError("P2-01 10k implementation status changed")
    if experiment.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise RuntimeError("P2-01 10k model contract changed")
    data = experiment.get("data") or {}
    if data.get("rows") != 10000 or data.get("samples_per_language_per_label") != 1000:
        raise RuntimeError("P2-01 10k cohort contract changed")
    if data.get("validation_rows_used") is not False or data.get("codeparrot_rows_or_labels_used") is not False:
        raise RuntimeError("P2-01 10k source boundary changed")
    if experiment.get("primary_context_features") != PRIMARY_FEATURES:
        raise RuntimeError("P2-01 10k primary feature contract changed")
    if experiment.get("evaluation", {}).get("automatic_promotion") is not False:
        raise RuntimeError("P2-01 10k automatic promotion guard changed")
    if experiment.get("competition_submission") is not False or experiment.get("public_lb_feedback_used") is not False:
        raise RuntimeError("P2-01 10k competition boundary changed")


def validate_kernel(kernel_dir: Path) -> None:
    notebook_path = kernel_dir / NOTEBOOK_NAME
    metadata_path = kernel_dir / "kernel-metadata.json"
    observed = sorted(path.name for path in kernel_dir.iterdir() if path.is_file())
    if observed != ["kernel-metadata.json", NOTEBOOK_NAME]:
        raise RuntimeError(f"unexpected kernel package files: {observed}")
    metadata = _load(metadata_path)
    expected_metadata = {
        "id": TARGET,
        "title": "Poisoned Chalice Context Contrast 10K V1",
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
    if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor") != 5 or len(notebook.get("cells") or []) != 10:
        raise RuntimeError("unexpected P2-01 10k notebook structure")
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    if len(set(cell_ids)) != len(cell_ids) or not all(isinstance(value, str) and value for value in cell_ids):
        raise RuntimeError("invalid P2-01 10k notebook cell ids")
    sources = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        sources.append(str(source))
    code = "\n".join(sources)
    for marker in (
        MODEL_REVISION, DATASET_REVISION, "samples_per_language: int = 2000",
        "context_budgets=(32, 128, 512)", "historical_first8_max_abs_diff",
        "context_mean_delta", "context_paper_delta", "context_local_delta",
        "baseline_loss", "baseline_paper", "baseline_local",
        '"automatic_promotion":False', '"competition_submission":False',
        "P2_CONTEXT_10K DEVELOPMENT_COMPLETE",
    ):
        if marker not in code:
            raise RuntimeError(f"generated P2-01 10k notebook marker missing: {marker}")
    for feature in PRIMARY_FEATURES:
        if feature not in code:
            raise RuntimeError(f"generated primary feature missing: {feature}")
    for forbidden in ("competition_submit(", "submission.csv", 'split="validation"', "CodeParrot/codeparrot"):
        if forbidden in code:
            raise RuntimeError(f"forbidden code in generated 10k notebook: {forbidden}")


def materialize(request_path: Path, research_root: Path, kernel_dir: Path) -> None:
    request = _load(request_path)
    validate_request(request)
    validate_research(research_root)
    builder = research_root / "scripts/build_context_contrast_10k_notebook.py"
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
            "P2-01 10k notebook builder failed "
            f"rc={completed.returncode} diagnostic_sha256={hashlib.sha256(diagnostic).hexdigest()}"
        )
    built_dir = research_root / "notebooks/experiments/matched-target-context-contrast-v1-10k"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    shutil.copytree(built_dir, kernel_dir)
    validate_kernel(kernel_dir)
    print("P2_CONTEXT_10K_MATERIALIZE PASS rows=10000 files=2 gpu=T4 retries=0 auto_promote=0 submissions=0")


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
            "P2_CONTEXT_10K_AMBIGUOUS_WRITE "
            f"rc={completed.returncode} stdout_bytes={len(out)} stderr_bytes={len(err)} "
            f"stdout_sha256={hashlib.sha256(out).hexdigest()} stderr_sha256={hashlib.sha256(err).hexdigest()}"
        )
        raise RuntimeError("kaggle kernels push returned non-zero; no retry permitted")
    print(
        "P2_CONTEXT_10K_LAUNCH_ACCEPTED "
        f"target={TARGET} rows=10000 accelerator=gpu machine=NvidiaTeslaT4 retries=0 auto_promote=0 submissions=0"
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
        print("P2_CONTEXT_10K_STATIC PASS rows=10000 primary_features=3 retries=0 auto_promote=0 submissions=0")
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
