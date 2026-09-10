#!/usr/bin/env python3
"""One-shot launcher for the repaired P1-03 1k LUMIA hidden-state OOF evaluation.

The scientific evaluation remains the frozen v1 protocol. This v2 target repairs only
Kaggle Notebook-output input resolution and must never overwrite/reuse the failed v1
target.
"""
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

REQUEST_ID = "20260910-poisoned-chalice-lumia-hidden-state-eval-1000-v2-001"
TARGET = "renta0426/poisoned-chalice-lumia-hidden-state-eval-1000-v2"
FAILED_TARGET = "renta0426/poisoned-chalice-lumia-hidden-state-eval-1000-v1"
CACHE_KERNEL = "renta0426/poisoned-chalice-lumia-hidden-state-cache-1000-v1"
RESEARCH_COMMIT = "8162c1ee58c4766f91da029d9ea232d79de9b763"
AUTHOR_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
CACHE_MANIFEST_BLOB = "cb2ea78db7dc1552716f1b515d63b22fa363e89c"
BUILDER_BLOB = "2577426ecd058349eac3145db93697271d37124a"
NOTEBOOK_NAME = "poisoned-chalice-lumia-hidden-state-eval-1000-v2.ipynb"
OUTPUT_PREFIX = "lumia_hidden_state_evaluation_1000_v2"
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
    "expected_runtime_minutes": 60,
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
    "task_id": "P1-03-lumia-hidden-state-evaluation-1000-v1",
    "purpose": "frozen_outer_oof_hidden_state_probe_evaluation",
    "operational_wrapper": "fresh_v2_exact_manifest_input_resolution",
    "failed_target_not_reused": FAILED_TARGET,
    "source_cache_kernel": CACHE_KERNEL,
    "source_cache_manifest_git_blob": CACHE_MANIFEST_BLOB,
    "rows": 1000,
    "public_train_labels_only": True,
    "outer_folds": 5,
    "outer_holdout_rows_per_fold": 200,
    "inner_probe_train_rows": 480,
    "inner_early_stop_rows": 160,
    "inner_layer_selection_rows": 160,
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
    "scale_gate_any": [
        "caller top5 OOF AUC >= zsigmoid AUC + 0.015",
        "caller top5 TPR@1% FPR >= zsigmoid TPR@1% FPR + 0.01 absolute",
        "caller top5 adds >= 10 conservative unique true positives versus zsigmoid at 1% FPR",
        "at least one language helper-vs-caller AUC delta paired-bootstrap lower_95 > 0",
    ],
    "outer_holdout_used_for_layer_selection": False,
    "visible_validation_used": False,
    "public_lb_used": False,
    "result_derived_formula_added": False,
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
        "launcher_path": "runners/poisoned_chalice/lumia_hidden_state_evaluation_1000_v2.py",
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
            raise RuntimeError(f"LUMIA evaluation 1k v2 request changed: {key}")
    if request.get("target") == FAILED_TARGET:
        raise RuntimeError("failed v1 target must never be reused")


def validate_research(root: Path) -> None:
    paths = {
        "evaluation": root / "src/poisoned_chalice/evaluation.py",
        "helper": root / "src/poisoned_chalice/sersem_author_faithful.py",
        "pilot": root / "src/poisoned_chalice/lumia_hidden_state_pilot.py",
        "cache": root / "src/poisoned_chalice/lumia_hidden_state_cache.py",
        "probe": root / "src/poisoned_chalice/lumia_hidden_state_probe.py",
        "runner": root / "scripts/run_lumia_hidden_state_probe_evaluation.py",
        "builder": root / "scripts/build_lumia_hidden_state_evaluation_1000_notebook_v2.py",
        "eval_config": root / "configs/p1_03_lumia_hidden_state_evaluation_v1_20260910.json",
        "science_config": root / "configs/p1_03_lumia_hidden_state_pilot_v1_20260909.json",
        "cache_manifest": root / "experiments/p1-03-lumia-hidden-state-cache-1000-v1/lumia_hidden_state_cache_1000_v1/cache_manifest.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f"research source missing: {path}")
    for key in ("evaluation", "helper", "pilot", "cache", "probe", "runner", "builder"):
        compile(paths[key].read_text(encoding="utf-8"), str(paths[key]), "exec")

    probe = paths["probe"].read_text(encoding="utf-8")
    runner = paths["runner"].read_text(encoding="utf-8")
    builder = paths["builder"].read_text(encoding="utf-8")
    eval_config = load_json(paths["eval_config"])
    science_config = load_json(paths["science_config"])
    cache_manifest = load_json(paths["cache_manifest"])

    if hashlib.sha1(b"blob " + str(paths["builder"].stat().st_size).encode("ascii") + b"\0" + paths["builder"].read_bytes()).hexdigest() != BUILDER_BLOB:
        raise RuntimeError("v2 builder blob identity changed")

    for marker in (
        "outer_folds: int = 5", "hidden_dim: int = 128", "learning_rate: float = 0.001",
        "weight_decay: float = 0.0001", "epochs_max: int = 100", "batch_size: int = 32",
        "validation_every_epochs: int = 5", "early_stopping_patience_checks: int = 5",
        "ensemble_size: int = 5", "bootstrap_replicates: int = 1000",
        "copy.deepcopy(probe.state_dict())", "(480, 160, 160, 200)",
        "selection_auc", "top5_ensemble", "paired_bootstrap_delta",
    ):
        if marker not in probe:
            raise RuntimeError(f"frozen probe marker missing: {marker}")
    for marker in (
        '"P1-03-lumia-hidden-state-evaluation-1000-v1"',
        '"lumia_caller_literal_top5_ensemble"', '"zsigmoid_single_sequence"',
        '"caller_auc_plus_0_015"', '"caller_tpr_01_plus_0_01_absolute"',
        '"caller_unique_tp_at_least_10"', '"helper_language_auc_bootstrap_lower_bound_positive"',
        '"visible_validation_used": False', '"public_lb_used": False',
        '"result_derived_formula_added": False', '"oof_predictions.csv"',
    ):
        if marker not in runner:
            raise RuntimeError(f"frozen evaluation runner marker missing: {marker}")
    for marker in (
        TARGET, CACHE_KERNEL,
        'INPUT_ROOT = Path("/kaggle/input")',
        'INPUT_ROOT.rglob("cache_manifest.json")',
        "EXPECTED_CACHE_MANIFEST_SHA256", "len(cache_candidates) != 1",
        "matching_manifests=", "expected_shards", DATASET_REVISION,
        '"--device", "cuda:0"', '"outer_oof_coverage"',
        "LUMIA_HIDDEN_STATE_EVAL_1000_V2 COMPLETE",
        'sys.path.insert(0, str(SITE_ROOT))',
    ):
        if marker not in builder:
            raise RuntimeError(f"v2 evaluation builder marker missing: {marker}")
    fixed_mount = 'Path("/kaggle/input/poisoned-chalice-lumia-hidden-state-cache-1000-v1'
    if fixed_mount in builder:
        raise RuntimeError("v2 builder regressed to the failed fixed mount path")
    for forbidden in ("competition_submit(", "submission.csv", "kaggle competitions submit"):
        if forbidden in builder or forbidden in runner:
            raise RuntimeError(f"forbidden evaluation code: {forbidden}")

    if eval_config.get("task_id") != SCIENTIFIC_CONTRACT["task_id"]:
        raise RuntimeError("frozen evaluation config task changed")
    if eval_config.get("status") != "frozen_before_1000_cache_execution":
        raise RuntimeError("frozen evaluation config status changed")
    if (eval_config.get("outer_oof") or {}).get("folds") != 5:
        raise RuntimeError("outer OOF fold contract changed")
    inner = eval_config.get("inner_per_outer_fit") or {}
    if (inner.get("probe_train_rows"), inner.get("early_stop_rows"), inner.get("layer_selection_rows")) != (480, 160, 160):
        raise RuntimeError("inner evaluation split contract changed")
    if (eval_config.get("bootstrap") or {}).get("replicates") != 1000:
        raise RuntimeError("bootstrap contract changed")
    if science_config.get("author_commit") != AUTHOR_COMMIT:
        raise RuntimeError("scientific author provenance changed")
    if (science_config.get("model") or {}).get("revision") != MODEL_REVISION:
        raise RuntimeError("scientific model revision changed")
    if (science_config.get("cohort") or {}).get("dataset_revision") != DATASET_REVISION:
        raise RuntimeError("scientific dataset revision changed")
    if cache_manifest.get("rows") != 1000 or cache_manifest.get("num_layers") != 30 or cache_manifest.get("hidden_size") != 3072:
        raise RuntimeError("sealed cache dimensions changed")
    if len(cache_manifest.get("shards") or []) != 20:
        raise RuntimeError("sealed cache shard count changed")
    if cache_manifest.get("labels_present_during_model_scoring") is not False or cache_manifest.get("performance_metrics_computed") is not False:
        raise RuntimeError("sealed cache clean-room boundary changed")


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
        "title": "Poisoned Chalice Lumia Hidden State Eval 1000 V2",
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
        raise RuntimeError("cache kernel input changed")
    for key in ("dataset_sources", "competition_sources", "model_sources"):
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
        'INPUT_ROOT = Path("/kaggle/input")',
        'INPUT_ROOT.rglob("cache_manifest.json")',
        "EXPECTED_CACHE_MANIFEST_SHA256", "len(cache_candidates) != 1",
        "matching_manifests=", DATASET_REVISION,
        "load_dataset(DATASET_ID, language, split=\"train\"", '"--device", "cuda:0"',
        'expected_outputs = ["decision.json", "fold_selection.json", "metrics.json", "oof_predictions.csv", "run_manifest.json"]',
        "LUMIA_HIDDEN_STATE_EVAL_1000_V2 COMPLETE", "outer_oof_coverage",
        "caller_auc_plus_0_015", "helper_language_auc_bootstrap_lower_bound_positive",
    ):
        if marker not in code:
            raise RuntimeError(f"generated v2 evaluation notebook marker missing: {marker}")
    fixed_mount = 'Path("/kaggle/input/poisoned-chalice-lumia-hidden-state-cache-1000-v1'
    if fixed_mount in code:
        raise RuntimeError("generated v2 notebook regressed to failed fixed mount path")
    for forbidden in ("competition_submit(", "submission.csv", "kaggle competitions submit"):
        if forbidden in code:
            raise RuntimeError(f"forbidden generated evaluation notebook code: {forbidden}")


def materialize(request_path: Path, research_root: Path, kernel_dir: Path) -> None:
    validate_request(load_json(request_path))
    validate_research(research_root)
    with tempfile.TemporaryDirectory(prefix="lumia-eval-1000-v2-nbformat-") as temp:
        shim = Path(temp)
        write_nbformat_shim(shim)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(shim) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        process = subprocess.run(
            [sys.executable, str(research_root / "scripts/build_lumia_hidden_state_evaluation_1000_notebook_v2.py")],
            cwd=str(research_root), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=120, check=False,
        )
    if process.returncode != 0:
        diagnostic = (process.stdout + process.stderr).encode("utf-8", errors="replace")
        raise RuntimeError(
            f"1k v2 evaluation notebook builder failed rc={process.returncode} "
            f"diagnostic_sha256={hashlib.sha256(diagnostic).hexdigest()}"
        )
    built = research_root / "notebooks/experiments/poisoned-chalice-lumia-hidden-state-eval-1000-v2"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    shutil.copytree(built, kernel_dir)
    validate_kernel(kernel_dir)
    print("LUMIA_HIDDEN_STATE_EVAL_1000_V2_MATERIALIZE PASS rows=1000 outer_folds=5 layers=30 source_cache=kernel_input exact_manifest_resolution=1 labels=public_train_only submissions=0")


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
            f"LUMIA_HIDDEN_STATE_EVAL_1000_V2_AMBIGUOUS_WRITE rc={process.returncode} "
            f"stdout_bytes={len(stdout)} stderr_bytes={len(stderr)} "
            f"stdout_sha256={hashlib.sha256(stdout).hexdigest()} "
            f"stderr_sha256={hashlib.sha256(stderr).hexdigest()}"
        )
        raise RuntimeError("kaggle kernels push returned non-zero; no retry permitted")
    print(
        f"LUMIA_HIDDEN_STATE_EVAL_1000_V2_LAUNCH_ACCEPTED target={TARGET} source_cache={CACHE_KERNEL} "
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
        print("LUMIA_HIDDEN_STATE_EVAL_1000_V2_STATIC PASS rows=1000 outer_folds=5 layers=30 bootstrap=1000 exact_manifest_resolution=1")
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
