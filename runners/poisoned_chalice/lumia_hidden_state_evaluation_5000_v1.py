#!/usr/bin/env python3
"""One-shot launcher for the frozen P1-03a 5k hidden-state OOF evaluation."""
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

REQUEST_ID = "20260910-poisoned-chalice-lumia-hidden-state-eval-5000-v1-001"
TARGET = "renta0426/poisoned-chalice-lumia-hidden-state-eval-5000-v1"
CACHE_KERNEL = "renta0426/poisoned-chalice-lumia-hidden-state-cache-5000-v1"
RESEARCH_COMMIT = "422771f7c28f263de031a35abc6efb4c0dbf4734"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
CACHE_MANIFEST_SHA256 = "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa"
KNOWN_CACHE_TASK_ID = "P1-03-lumia-hidden-state-cache-1000-v1"
BUILDER_BLOB = "7b77f5e42f83efc9f5c1ff1300c47efc67449656"
NOTEBOOK_NAME = "poisoned-chalice-lumia-hidden-state-eval-5000-v1.ipynb"
OUTPUT_PREFIX = "lumia_hidden_state_evaluation_5000_v1"
PERSISTENT_OUTPUTS = [
    f"{OUTPUT_PREFIX}/decision.json",
    f"{OUTPUT_PREFIX}/fold_selection.json",
    f"{OUTPUT_PREFIX}/metrics.json",
    f"{OUTPUT_PREFIX}/oof_predictions.csv",
    f"{OUTPUT_PREFIX}/run_manifest.json",
]
RESOURCE = {
    "accelerator": "gpu",
    "machine_shape": "NvidiaTeslaT4",
    "expected_visible_gpu_count": 2,
    "expected_runtime_minutes": 75,
    "hard_timeout_minutes": 120,
    "max_active_runs": 1,
    "min_remaining_quota_hours": 2.5,
}
CLEAN_ROOM = {
    "new_model_specific_holdout_opened": False,
    "public_leaderboard_feedback_used": False,
    "hidden_validation_labels_used": False,
    "validation_rows_used": False,
    "codeparrot_rows_or_labels_used": False,
    "row_level_persisted_membership_labels": True,
    "persisted_membership_labels_scope": "public_train_oof_evaluation_only",
    "performance_metrics_computed": True,
    "automatic_promotion": False,
}
SCIENTIFIC_CONTRACT = {
    "task_id": "P1-03a-lumia-hidden-state-evaluation-5000-v1",
    "purpose": "frozen_scale_stability_outer_oof_hidden_state_probe_evaluation",
    "source_cache_kernel": CACHE_KERNEL,
    "source_cache_manifest_sha256": CACHE_MANIFEST_SHA256,
    "known_cache_manifest_task_id_bug": KNOWN_CACHE_TASK_ID,
    "known_cache_manifest_task_id_bug_scientific_data_affected": False,
    "rows": 5000,
    "rows_per_language_label": 500,
    "public_train_labels_only": True,
    "outer_folds": 5,
    "outer_holdout_rows_per_fold": 1000,
    "inner_probe_train_rows": 2400,
    "inner_early_stop_rows": 800,
    "inner_layer_selection_rows": 800,
    "layers": 30,
    "probe_architecture": "Linear(input_dim,128)-ReLU-Dropout(0.5)-Linear(128,1)-Sigmoid",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "epochs_max": 100,
    "batch_size": 32,
    "validation_every_epochs": 5,
    "early_stopping_patience_checks": 5,
    "top5_ensemble_size": 5,
    "bootstrap_replicates": 1000,
    "bootstrap_seed": 20260909,
    "predeclared_same_model_signal_confirmed_all": [
        "caller top5 OOF AUC >= 0.62",
        "caller top5 minus same-input zsigmoid paired-bootstrap AUC delta lower_95 > 0",
        "caller top5 per-language AUC > 0.55 for all five languages",
    ],
    "predeclared_weighting_increment_supported_if": "caller top5 minus mean-only top5 OR helper top5 minus mean-only top5 paired-bootstrap AUC delta lower_95 > 0",
    "predeclared_low_fpr_confirmation_if": "at least one frozen hidden anchor has TPR@1% FPR >= zsigmoid + 0.01 and its paired-bootstrap TPR delta lower_95 >= 0",
    "tail_failure_does_not_negate_auc_signal": True,
    "outer_holdout_used_for_layer_selection": False,
    "visible_validation_used": False,
    "public_lb_used": False,
    "result_derived_formula_added": False,
    "automatic_promotion": False,
    "no_automatic_cross_model_promotion": True,
    "competition_submission": False,
}
EXPECTED_BLOBS = {
    "src/poisoned_chalice/evaluation.py": "cb3afd4f3e3ecafe0e0677eda4fda30e93b04ba8",
    "src/poisoned_chalice/sersem_author_faithful.py": "91f0fcdccd646902ccc939e0835090a517505347",
    "src/poisoned_chalice/lumia_hidden_state_pilot.py": "d33fa4f4b0d258083d1503e4b4b883461f41ed8f",
    "src/poisoned_chalice/lumia_hidden_state_cache.py": "61182c7fcde131ed36c0791fa16928dcdc9b6e8b",
    "src/poisoned_chalice/lumia_hidden_state_probe.py": "dffe5648eb45f6919dcd002a46f7a2c6d6fb9121",
    "src/poisoned_chalice/lumia_hidden_state_scale.py": "3f323970ab8bac0a243be9cb0fed19b3c40f74f6",
    "scripts/run_lumia_hidden_state_scale_5000_evaluation.py": "d82e37b1ca5c6962c62c409d7944e9b6e7067bc7",
    "scripts/build_lumia_hidden_state_evaluation_5000_notebook_v1.py": "7b77f5e42f83efc9f5c1ff1300c47efc67449656",
    "configs/p1_03_lumia_hidden_state_scale_5000_v1_20260910.json": "0dccee9b2e562ccab9f4b4a1e4c9f21cb5518b85",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def slugify(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-"))


def validate_request(request: dict) -> None:
    exact = {
        "request_id": REQUEST_ID,
        "competition": "poisoned-chalice-icse27",
        "operation": "kernel_run",
        "target": TARGET,
        "launcher_path": "runners/poisoned_chalice/lumia_hidden_state_evaluation_5000_v1.py",
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
            raise RuntimeError(f"P1-03a 5k evaluation request changed: {key}")


def validate_research(root: Path) -> None:
    for relative, expected in EXPECTED_BLOBS.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"research source missing: {relative}")
        if git_blob_sha(path) != expected:
            raise RuntimeError(f"research Git blob changed: {relative}")
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8"), relative, "exec")

    probe = (root / "src/poisoned_chalice/lumia_hidden_state_probe.py").read_text(encoding="utf-8")
    scale = (root / "src/poisoned_chalice/lumia_hidden_state_scale.py").read_text(encoding="utf-8")
    runner = (root / "scripts/run_lumia_hidden_state_scale_5000_evaluation.py").read_text(encoding="utf-8")
    builder_path = root / "scripts/build_lumia_hidden_state_evaluation_5000_notebook_v1.py"
    builder = builder_path.read_text(encoding="utf-8")
    config = load_json(root / "configs/p1_03_lumia_hidden_state_scale_5000_v1_20260910.json")
    if git_blob_sha(builder_path) != BUILDER_BLOB:
        raise RuntimeError("5k evaluation builder identity changed")

    for marker in (
        "outer_folds: int = 5", "hidden_dim: int = 128", "learning_rate: float = 0.001",
        "weight_decay: float = 0.0001", "epochs_max: int = 100", "batch_size: int = 32",
        "validation_every_epochs: int = 5", "early_stopping_patience_checks: int = 5",
        "ensemble_size: int = 5", "bootstrap_replicates: int = 1000",
        "copy.deepcopy(probe.state_dict())", "selection_auc", "top5_ensemble",
    ):
        if marker not in probe:
            raise RuntimeError(f"frozen probe marker missing: {marker}")
    for marker in (
        "EXPECTED_ROWS = 5000", "EXPECTED_PER_LANGUAGE_LABEL = 500",
        "EXPECTED_SPLIT_SIZES = (2400, 800, 800, 1000)", "run_outer_oof_5000",
        "paired_increment_bootstraps",
    ):
        if marker not in scale:
            raise RuntimeError(f"frozen 5k scale marker missing: {marker}")
    for marker in (
        CACHE_MANIFEST_SHA256, KNOWN_CACHE_TASK_ID, "load_sealed_cache_5000",
        "4512897483", "run_outer_oof_5000", "same_model_signal_confirmed",
        "weighting_increment_supported", "low_fpr_confirmed", "oof_predictions.csv",
        '"visible_validation_used": False', '"public_lb_used": False',
        '"result_derived_formula_added": False',
    ):
        if marker not in runner:
            raise RuntimeError(f"5k evaluation runner marker missing: {marker}")
    for marker in (
        TARGET, CACHE_KERNEL, CACHE_MANIFEST_SHA256,
        'INPUT_ROOT = Path("/kaggle/input")', 'INPUT_ROOT.rglob("cache_manifest.json")',
        "len(cache_candidates) != 1", "matching_manifests=", KNOWN_CACHE_TASK_ID,
        DATASET_REVISION, '"--device", "cuda:0"', "LUMIA_HIDDEN_STATE_EVAL_5000_V1 COMPLETE",
    ):
        if marker not in builder:
            raise RuntimeError(f"5k evaluation builder marker missing: {marker}")
    for forbidden in ("competition_submit(", "submission.csv", "kaggle competitions submit"):
        if forbidden in builder or forbidden in runner:
            raise RuntimeError(f"forbidden evaluation code: {forbidden}")

    if config.get("task_id") != "P1-03a-lumia-hidden-state-scale-5000-v1":
        raise RuntimeError("5k scale config task changed")
    if config.get("status") != "frozen_before_5000_cache_execution":
        raise RuntimeError("5k scale config freeze status changed")
    if (config.get("model") or {}).get("revision") != MODEL_REVISION:
        raise RuntimeError("5k model revision changed")
    cohort = config.get("cohort") or {}
    if cohort.get("dataset_revision") != DATASET_REVISION or cohort.get("rows") != 5000 or cohort.get("rows_per_language_label") != 500:
        raise RuntimeError("5k cohort contract changed")
    evaluation = config.get("evaluation") or {}
    inner = evaluation.get("inner_per_outer_fit") or {}
    if evaluation.get("outer_oof_folds") != 5 or evaluation.get("outer_holdout_rows_per_fold") != 1000:
        raise RuntimeError("5k outer OOF contract changed")
    if (inner.get("probe_train_rows"), inner.get("early_stop_rows"), inner.get("layer_selection_rows")) != (2400, 800, 800):
        raise RuntimeError("5k inner split contract changed")
    if evaluation.get("bootstrap_replicates") != 1000 or evaluation.get("bootstrap_seed") != 20260909:
        raise RuntimeError("5k bootstrap contract changed")


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
        "title": "Poisoned Chalice Lumia Hidden State Eval 5000 V1",
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
    if metadata.get("kernel_sources") != [CACHE_KERNEL]:
        raise RuntimeError("5k cache kernel input changed")
    for key in ("dataset_sources", "competition_sources", "model_sources"):
        if metadata.get(key) != []:
            raise RuntimeError(f"unexpected attached source: {key}")
    notebook = load_json(root / NOTEBOOK_NAME)
    cells = notebook.get("cells") or []
    if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor") != 5 or len(cells) != 4:
        raise RuntimeError("unexpected notebook structure")
    code = "\n".join(str(cell.get("source", "")) for cell in cells)
    for marker in (
        'INPUT_ROOT = Path("/kaggle/input")', 'INPUT_ROOT.rglob("cache_manifest.json")',
        CACHE_MANIFEST_SHA256, "len(cache_candidates) != 1", "matching_manifests=",
        KNOWN_CACHE_TASK_ID, DATASET_REVISION, '"--device", "cuda:0"',
        'expected_outputs = ["decision.json", "fold_selection.json", "metrics.json", "oof_predictions.csv", "run_manifest.json"]',
        "LUMIA_HIDDEN_STATE_EVAL_5000_V1 COMPLETE",
    ):
        if marker not in code:
            raise RuntimeError(f"generated 5k notebook marker missing: {marker}")
    for forbidden in ("competition_submit(", "submission.csv", "kaggle competitions submit"):
        if forbidden in code:
            raise RuntimeError(f"generated notebook contains forbidden code: {forbidden}")


def materialize(request: dict, research_root: Path, kernel_dir: Path) -> None:
    validate_request(request)
    validate_research(research_root)
    with tempfile.TemporaryDirectory(prefix="lumia-eval-5000-build-") as td:
        shim = Path(td)
        write_nbformat_shim(shim)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(shim)
        builder = research_root / "scripts/build_lumia_hidden_state_evaluation_5000_notebook_v1.py"
        completed = subprocess.run(
            [sys.executable, str(builder)], cwd=str(research_root), env=env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"5k Notebook materialization failed rc={completed.returncode}")
    generated = research_root / "notebooks/experiments/poisoned-chalice-lumia-hidden-state-eval-5000-v1"
    if not generated.is_dir():
        raise RuntimeError("generated 5k kernel directory missing")
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)
    for name in ("kernel-metadata.json", NOTEBOOK_NAME):
        shutil.copy2(generated / name, kernel_dir / name)
    validate_kernel(kernel_dir)
    print("LUMIA_HIDDEN_STATE_EVAL_5000_V1_MATERIALIZE PASS rows=5000 outer_folds=5 cache=kernel_input submissions=0")


def execute(request: dict, kernel_dir: Path, kaggle_bin: str) -> None:
    validate_request(request)
    validate_kernel(kernel_dir)
    env = os.environ.copy()
    completed = subprocess.run(
        [kaggle_bin, "kernels", "push", "-p", str(kernel_dir)],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        raise RuntimeError(
            "5k evaluation launch failed "
            f"rc={completed.returncode} stdout_sha256={hashlib.sha256(stdout.encode()).hexdigest()} "
            f"stderr_sha256={hashlib.sha256(stderr.encode()).hexdigest()}"
        )
    print(
        "LUMIA_HIDDEN_STATE_EVAL_5000_V1_LAUNCH_ACCEPTED "
        f"target={TARGET} source_cache={CACHE_KERNEL} stdout_bytes={len(stdout.encode())} stderr_bytes={len(stderr.encode())} "
        f"stdout_sha256={hashlib.sha256(stdout.encode()).hexdigest()} stderr_sha256={hashlib.sha256(stderr.encode()).hexdigest()} retries=0 submissions=0"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--kernel-dir", type=Path)
    parser.add_argument("--kaggle-bin", default="kaggle")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    request = load_json(args.request)
    if args.static:
        if args.research_root is None:
            raise RuntimeError("--research-root required for static validation")
        validate_request(request)
        validate_research(args.research_root)
        print("LUMIA_HIDDEN_STATE_EVAL_5000_V1_STATIC PASS rows=5000 outer_folds=5 layers=30 bootstrap=1000")
        return 0
    if args.materialize:
        if args.research_root is None or args.kernel_dir is None:
            raise RuntimeError("--research-root and --kernel-dir required")
        materialize(request, args.research_root, args.kernel_dir)
        return 0
    if args.kernel_dir is None:
        raise RuntimeError("--kernel-dir required")
    execute(request, args.kernel_dir, args.kaggle_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
