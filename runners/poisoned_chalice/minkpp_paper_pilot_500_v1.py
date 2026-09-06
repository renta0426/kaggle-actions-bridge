#!/usr/bin/env python3
"""Validate, materialize, and one-shot launch the Min-K++ 500-row pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

REQUEST_ID = "20260907-poisoned-chalice-minkpp-paper-pilot-500-001"
TARGET = "renta0426/poisoned-chalice-minkpp-paper-pilot-500-v1"
RESEARCH_COMMIT = "0895502ddb188de5af4ba780eb6f77f2185e64d1"
MODEL_ID = "bigcode/starcoder2-3b"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
EXPERIMENT_ID = "stage1-minkpp-paper-pilot-500"
NOTEBOOK_NAME = "stage1-minkpp-paper-pilot-500.ipynb"
OUTPUT_PREFIX = "stage1-minkpp-paper-pilot-500"
PERSISTENT_OUTPUTS = [
    f"{OUTPUT_PREFIX}/sample_manifest.parquet",
    f"{OUTPUT_PREFIX}/sample_scores.parquet",
    f"{OUTPUT_PREFIX}/window_scores.parquet",
    f"{OUTPUT_PREFIX}/token_statistics.parquet",
    f"{OUTPUT_PREFIX}/score_spearman.csv",
    f"{OUTPUT_PREFIX}/metrics.json",
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
            raise RuntimeError(f"Min-K++ request changed: {key}")
    resource = request.get("resource") or {}
    expected_resource = {
        "accelerator": "gpu",
        "machine_shape": "NvidiaTeslaT4",
        "expected_visible_gpu_count": 2,
        "expected_runtime_minutes": 45,
        "hard_timeout_minutes": 90,
        "max_active_runs": 2,
        "min_remaining_quota_hours": 1.0,
    }
    if resource != expected_resource:
        raise RuntimeError("Min-K++ resource contract changed")
    if request.get("api_budget") != {"max_calls": 60}:
        raise RuntimeError("Min-K++ API budget changed")
    clean = request.get("clean_room") or {}
    expected_clean = {
        "stage2_holdout_consumed": False,
        "public_leaderboard_feedback_used": False,
        "hidden_validation_labels_used": False,
        "low_fpr_selection_from_500_forbidden": True,
        "legacy_feature_semantics_modified": False,
    }
    if clean != expected_clean:
        raise RuntimeError("Min-K++ clean-room contract changed")
    science = request.get("scientific_contract") or {}
    expected_science = {
        "rows": 500,
        "member_rows": 250,
        "per_language_per_label": 50,
        "paper_schema": "prob_weighted_minkpp_v1",
        "legacy_schema": "legacy_uniform_vocab_z",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "max_model_tokens_per_window": 768,
        "same_forward": True,
        "cpu_oracle_required": True,
        "dense_blocked_parity_required": True,
        "low_fpr_selection_from_500_forbidden": True,
    }
    if science != expected_science:
        raise RuntimeError("Min-K++ scientific contract changed")


def validate_research(research_root: Path) -> None:
    scorer = research_root / "src/poisoned_chalice/minkpp_paper.py"
    builder = research_root / "scripts/build_minkpp_pilot_notebook.py"
    experiment_path = research_root / "experiments/stage1-minkpp-paper-pilot-500/experiment.json"
    for path in (scorer, builder, experiment_path):
        if not path.is_file():
            raise RuntimeError(f"research source missing: {path}")
    scorer_source = scorer.read_text(encoding="utf-8")
    builder_source = builder.read_text(encoding="utf-8")
    compile(scorer_source, str(scorer), "exec")
    compile(builder_source, str(builder), "exec")
    for marker in (
        "probability_weighted_token_statistics_dense",
        "probability_weighted_token_statistics_blocked",
        "prob_weighted_minkpp_v1",
        "legacy_uniform_vocab_z",
    ):
        if marker not in scorer_source:
            raise RuntimeError(f"paper scorer marker missing: {marker}")
    for marker in (
        MODEL_ID,
        MODEL_REVISION,
        DATASET_REVISION,
        'samples_per_language: int = 100',
        'max_length: int = 768',
        'paper_vocab_block_size: int = 8_192',
        '"id": "renta0426/poisoned-chalice-minkpp-paper-pilot-500-v1"',
        '"is_private": True',
        '"enable_gpu": True',
        '"enable_internet": True',
    ):
        if marker not in builder_source:
            raise RuntimeError(f"pilot builder marker missing: {marker}")
    experiment = _load(experiment_path)
    if experiment.get("experiment_id") != EXPERIMENT_ID or experiment.get("data", {}).get("rows") != 500:
        raise RuntimeError("experiment identity changed")
    if experiment.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise RuntimeError("model contract changed")
    scoring = experiment.get("scoring") or {}
    if scoring.get("paper_schema") != "prob_weighted_minkpp_v1" or scoring.get("legacy_schema") != "legacy_uniform_vocab_z":
        raise RuntimeError("scoring schema changed")
    if scoring.get("same_forward") is not True or scoring.get("max_model_tokens_per_window") != 768:
        raise RuntimeError("forward/window contract changed")
    if experiment.get("gates", {}).get("low_fpr_selection_from_500_forbidden") is not True:
        raise RuntimeError("500-row low-FPR guard changed")


def validate_kernel(kernel_dir: Path) -> None:
    notebook_path = kernel_dir / NOTEBOOK_NAME
    metadata_path = kernel_dir / "kernel-metadata.json"
    observed = sorted(path.name for path in kernel_dir.iterdir() if path.is_file())
    if observed != ["kernel-metadata.json", NOTEBOOK_NAME]:
        raise RuntimeError(f"unexpected kernel package files: {observed}")
    metadata = _load(metadata_path)
    expected_metadata = {
        "id": TARGET,
        "title": "Poisoned Chalice MinKPP Paper Pilot 500 V1",
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
    if notebook.get("nbformat") != 4 or len(notebook.get("cells") or []) != 8:
        raise RuntimeError("unexpected notebook structure")
    sources = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        sources.append(str(source))
    code = "\n".join(sources)
    for marker in (
        MODEL_REVISION,
        DATASET_REVISION,
        "probability_weighted_token_statistics_blocked",
        "legacy_uniform_vocab_minkpp_",
        "paper_prob_weighted_minkpp_",
        "oracle_abs_error",
        "real_dense_blocked_max_z_diff",
        "low-FPR selection",
    ):
        if marker not in code:
            raise RuntimeError(f"generated notebook marker missing: {marker}")
    for forbidden in ("competition_submit", "submission.csv", "select_as_final"):
        if forbidden in code:
            raise RuntimeError(f"forbidden competition side effect in notebook: {forbidden}")


def materialize(request_path: Path, research_root: Path, kernel_dir: Path) -> None:
    request = _load(request_path)
    validate_request(request)
    validate_research(research_root)
    builder = research_root / "scripts/build_minkpp_pilot_notebook.py"
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
        digest = hashlib.sha256((completed.stdout + completed.stderr).encode("utf-8", errors="replace")).hexdigest()
        raise RuntimeError(f"notebook builder failed rc={completed.returncode} diagnostic_sha256={digest}")
    built_dir = research_root / "notebooks/experiments/stage1-minkpp-paper-pilot-500"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    shutil.copytree(built_dir, kernel_dir)
    validate_kernel(kernel_dir)
    print("MINKPP_PAPER_PILOT_500_MATERIALIZE PASS rows=500 files=2 gpu=T4 retries=0 submissions=0")


def execute(request_path: Path, kernel_dir: Path, kaggle_bin: Path) -> None:
    request = _load(request_path)
    validate_request(request)
    validate_kernel(kernel_dir)
    if not kaggle_bin.is_file():
        raise RuntimeError("locked Kaggle CLI missing")
    command = [str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir)]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        check=False,
    )
    out, err = completed.stdout or b"", completed.stderr or b""
    if completed.returncode != 0:
        print(
            "MINKPP_PAPER_PILOT_500_AMBIGUOUS_WRITE "
            f"rc={completed.returncode} stdout_bytes={len(out)} stderr_bytes={len(err)} "
            f"stdout_sha256={hashlib.sha256(out).hexdigest()} stderr_sha256={hashlib.sha256(err).hexdigest()}"
        )
        raise RuntimeError("kaggle kernels push returned non-zero; no retry permitted")
    print(
        "MINKPP_PAPER_PILOT_500_LAUNCH_ACCEPTED "
        f"target={TARGET} accelerator=gpu machine=NvidiaTeslaT4 retries=0 submissions=0"
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
        print("MINKPP_PAPER_PILOT_500_STATIC PASS rows=500 same_forward=1 retries=0 submissions=0")
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
