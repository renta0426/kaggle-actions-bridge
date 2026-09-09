#!/usr/bin/env python3
"""One-shot launcher for P1-03 LUMIA v5 explicit CUDA placement repair."""
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

REQUEST_ID = "20260909-poisoned-chalice-lumia-hs-runtime-50-v5-001"
TARGET = "renta0426/poisoned-chalice-lumia-hs-runtime-50-v5"
RESEARCH_COMMIT = "1c22d77cf7d9960edb855d41a741f0478b957c6c"
AUTHOR_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
MODEL_ID = "bigcode/starcoder2-3b"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
NOTEBOOK_NAME = "poisoned-chalice-lumia-hs-runtime-50-v5.ipynb"
OUTPUT_PREFIX = "lumia_hidden_state_runtime_pilot_50"
PERSISTENT_OUTPUTS = [f"{OUTPUT_PREFIX}/runtime.json", f"{OUTPUT_PREFIX}/run_manifest.json"]
RESOURCE = {
    "accelerator": "gpu", "machine_shape": "NvidiaTeslaT4",
    "expected_visible_gpu_count": 2, "expected_runtime_minutes": 25,
    "hard_timeout_minutes": 60, "max_active_runs": 1,
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
REQUEST_SCIENCE = {
    "task_id": "P1-03-lumia-hidden-state-runtime-pilot-50-v1",
    "purpose": "runtime_and_fidelity_only",
    "rows": 50,
    "rows_per_language_label": 5,
    "selection_seed": 20260909,
    "selection_labels_removed_before_model_scoring": True,
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "dataset_revision": DATASET_REVISION,
    "author_commit": AUTHOR_COMMIT,
    "max_length": 8192,
    "precision": "fp16",
    "all_transformer_layers": True,
    "pooling_outputs_checked": ["mean", "caller_literal_weighted_mean", "helper_language_aware_weighted_mean"],
    "same_forward_output_scores_finite_only": True,
    "raw_activations_persisted": False,
    "sample_scores_persisted": False,
    "performance_metrics_computed": False,
    "direct_1000_gate_estimated_wall_minutes_lte": 105,
    "automatic_promotion": False,
    "competition_submission": False,
}
CONFIG_SCIENCE = {
    "rows": 50,
    "rows_per_language_label": 5,
    "selection_seed": 20260909,
    "selection_labels_removed_before_model_scoring": True,
    "validation_rows_used": False,
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "dataset_revision": DATASET_REVISION,
    "author_commit": AUTHOR_COMMIT,
    "max_length": 8192,
    "precision": "fp16",
    "all_transformer_layers": True,
    "pooling_outputs_checked": ["mean", "caller_literal_weighted_mean", "helper_language_aware_weighted_mean"],
    "same_forward_output_scores_finite_only": True,
    "raw_activations_persisted": False,
    "sample_scores_persisted": False,
    "performance_metrics_computed": False,
    "direct_1000_gate_estimated_wall_minutes_lte": 105,
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
        "launcher_path": "runners/poisoned_chalice/lumia_hidden_state_runtime_50_v5.py",
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
        "scientific_contract": REQUEST_SCIENCE,
    }
    for key, value in exact.items():
        if request.get(key) != value:
            raise RuntimeError(f"LUMIA v5 request changed: {key}")


def validate_research(root: Path) -> None:
    core_path = root / "src/poisoned_chalice/lumia_hidden_state_pilot.py"
    helper_path = root / "src/poisoned_chalice/sersem_author_faithful.py"
    runtime_path = root / "src/poisoned_chalice/sersem_author_runtime.py"
    v4_path = root / "scripts/build_lumia_hidden_state_runtime_pilot_notebook_v4.py"
    v5_path = root / "scripts/build_lumia_hidden_state_runtime_pilot_notebook_v5.py"
    config_path = root / "configs/p1_03_lumia_hidden_state_runtime_pilot_50_v5_20260909.json"
    for path in (core_path, helper_path, runtime_path, v4_path, v5_path, config_path):
        if not path.is_file(): raise RuntimeError(f"research source missing: {path}")
    for path in (core_path, helper_path, runtime_path, v4_path, v5_path):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    core = core_path.read_text(encoding="utf-8")
    helper = helper_path.read_text(encoding="utf-8")
    runtime = runtime_path.read_text(encoding="utf-8")
    v4 = v4_path.read_text(encoding="utf-8")
    v5 = v5_path.read_text(encoding="utf-8")
    config = load_json(config_path)
    for marker in (
        AUTHOR_COMMIT, MODEL_REVISION, DATASET_REVISION,
        "rows: int = 50", "rows_per_language_label: int = 5", "max_length: int = 8192",
        'return cohort[["sample_id", "language", "content"]].copy()',
        "register_forward_hook", "caller_weighted", "helper_weighted",
        "estimated_1000_minutes_wall_linear",
    ):
        if marker not in core: raise RuntimeError(f"core marker missing: {marker}")
    for marker in ("build_author_character_weights", "token_weights_from_character_weights", "weighted_output_score"):
        if marker not in helper: raise RuntimeError(f"helper marker missing: {marker}")
    for marker in ("AUTHOR_RUNTIME_PINS", '"flake8": "7.3.0"', "load_author_runtime"):
        if marker not in runtime: raise RuntimeError(f"runtime marker missing: {marker}")
    for marker in (
        'TARGET = "renta0426/poisoned-chalice-lumia-hs-runtime-50-v4"',
        'SITE_ROOT = Path("/tmp/lumia-runtime-v4-site")',
        'device_map="cuda"', "fresh_child_import_gate", "target_site_packages",
    ):
        if marker not in v4: raise RuntimeError(f"v4 base marker missing: {marker}")
    for marker in (
        "build_lumia_hidden_state_runtime_pilot_notebook_v4.py",
        'model = model.to("cuda:0").eval()', 'parameter_devices == ["cuda:0"]',
        '"operational_attempt": "v5_explicit_cuda_transfer_repair"',
        "LUMIA_HIDDEN_STATE_RUNTIME_PILOT_V5 COMPLETE", "nbformat.v4.new_notebook()",
    ):
        if marker not in v5: raise RuntimeError(f"v5 repair marker missing: {marker}")
    if config.get("operational_attempt") != "v5_explicit_cuda_transfer_repair": raise RuntimeError("v5 identity changed")
    if config.get("new_target") != TARGET or config.get("failed_v4_target") != "renta0426/poisoned-chalice-lumia-hs-runtime-50-v4": raise RuntimeError("v5 target provenance changed")
    if config.get("failed_v4_current_version") != 1 or config.get("failed_v4_terminal_status") != "KERNELWORKERSTATUS.ERROR": raise RuntimeError("v5 failed version changed")
    failure = config.get("confirmed_v4_failure") or {}
    if failure.get("target_site_import_gate_passed") is not True or failure.get("public_train_rows_loaded") != 49040 or failure.get("gpu_model_forward_started") is not False: raise RuntimeError("v5 failure evidence changed")
    repair = config.get("repair") or {}
    if repair.get("target_site_runtime_unchanged_from_v4") is not True or repair.get("device_map_removed") is not True or repair.get("model_transfer_expression") != "model.to('cuda:0').eval()": raise RuntimeError("v5 repair scope changed")
    if config.get("scientific_contract_unchanged") != CONFIG_SCIENCE: raise RuntimeError("v5 frozen scientific config changed")
    if config.get("persistent_outputs") != PERSISTENT_OUTPUTS: raise RuntimeError("v5 outputs changed")


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
    observed = sorted(p.name for p in root.iterdir() if p.is_file())
    if observed != ["kernel-metadata.json", NOTEBOOK_NAME]: raise RuntimeError(f"unexpected kernel files: {observed}")
    metadata = load_json(root / "kernel-metadata.json")
    expected = {
        "id": TARGET, "title": "Poisoned Chalice Lumia Hs Runtime 50 V5",
        "code_file": NOTEBOOK_NAME, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
    }
    for key, value in expected.items():
        if metadata.get(key) != value: raise RuntimeError(f"metadata changed: {key}")
    if slugify(metadata["title"]) != TARGET.split("/",1)[1]: raise RuntimeError("slug mismatch")
    for key in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
        if metadata.get(key) != []: raise RuntimeError(f"unexpected attached source: {key}")
    notebook = load_json(root / NOTEBOOK_NAME)
    cells = notebook.get("cells") or []
    if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor") != 5 or len(cells) != 4: raise RuntimeError("notebook structure changed")
    if len({c.get("id") for c in cells}) != 4: raise RuntimeError("notebook IDs changed")
    code = "\n".join(str(c.get("source", "")) for c in cells)
    for marker in (
        AUTHOR_COMMIT, MODEL_REVISION, DATASET_REVISION, "max_length: int = 8192", "rows: int = 50",
        'SITE_ROOT = Path("/tmp/lumia-runtime-v5-site")', "fresh_child_import_gate", "target_site_packages",
        'model = model.to("cuda:0").eval()', 'parameter_devices == ["cuda:0"]',
        "run_runtime_pilot(model, tokenizer, cohort, runtime, CONFIG)",
        "LUMIA_HIDDEN_STATE_RUNTIME_PILOT_V5 COMPLETE performance_metrics=0 sample_outputs=0",
    ):
        if marker not in code: raise RuntimeError(f"generated marker missing: {marker}")
    for forbidden in ('device_map="cuda"', "roc_auc_score", "roc_curve", "average_precision_score", "competition_submit(", "submission.csv"):
        if forbidden in code: raise RuntimeError(f"forbidden generated code: {forbidden}")


def materialize(request_path: Path, research_root: Path, kernel_dir: Path) -> None:
    validate_request(load_json(request_path)); validate_research(research_root)
    with tempfile.TemporaryDirectory(prefix="lumia-v5-shim-") as temp:
        shim = Path(temp); write_nbformat_shim(shim)
        env = os.environ.copy(); env["PYTHONPATH"] = str(shim) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        p = subprocess.run([sys.executable, str(research_root / "scripts/build_lumia_hidden_state_runtime_pilot_notebook_v5.py")], cwd=str(research_root), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120, check=False)
    if p.returncode:
        diagnostic=(p.stdout+p.stderr).encode("utf-8",errors="replace")
        raise RuntimeError(f"v5 builder failed rc={p.returncode} diagnostic_sha256={hashlib.sha256(diagnostic).hexdigest()}")
    built = research_root / "notebooks/experiments/poisoned-chalice-lumia-hs-runtime-50-v5"
    if kernel_dir.exists(): shutil.rmtree(kernel_dir)
    shutil.copytree(built, kernel_dir); validate_kernel(kernel_dir)
    print("LUMIA_HS_RUNTIME_50_V5_MATERIALIZE PASS rows=50 max_length=8192 explicit_cuda0=1 labels_in_scoring=0 metrics=0 sample_outputs=0 gpu=T4 retries=0")


def execute(request_path: Path, kernel_dir: Path, kaggle_bin: Path) -> None:
    validate_request(load_json(request_path)); validate_kernel(kernel_dir)
    if not kaggle_bin.is_file(): raise RuntimeError("locked Kaggle CLI missing")
    p = subprocess.run([str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False)
    out,err=p.stdout or b"",p.stderr or b""
    if p.returncode:
        print(f"LUMIA_HS_RUNTIME_50_V5_AMBIGUOUS_WRITE rc={p.returncode} stdout_bytes={len(out)} stderr_bytes={len(err)} stdout_sha256={hashlib.sha256(out).hexdigest()} stderr_sha256={hashlib.sha256(err).hexdigest()}")
        raise RuntimeError("kaggle kernels push returned non-zero; no retry permitted")
    print(f"LUMIA_HS_RUNTIME_50_V5_LAUNCH_ACCEPTED target={TARGET} stdout_bytes={len(out)} stderr_bytes={len(err)} stdout_sha256={hashlib.sha256(out).hexdigest()} stderr_sha256={hashlib.sha256(err).hexdigest()} retries=0 submissions=0")


def main() -> int:
    parser=argparse.ArgumentParser(); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static",action="store_true"); mode.add_argument("--materialize",action="store_true"); mode.add_argument("--execute",action="store_true")
    parser.add_argument("--request",type=Path,required=True); parser.add_argument("--research-root",type=Path); parser.add_argument("--kernel-dir",type=Path); parser.add_argument("--kaggle-bin",type=Path)
    args=parser.parse_args(); validate_request(load_json(args.request))
    if args.static:
        if args.research_root is None: raise SystemExit("--research-root required")
        validate_research(args.research_root); print("LUMIA_HS_RUNTIME_50_V5_STATIC PASS explicit_cuda0=1"); return 0
    if args.materialize:
        if args.research_root is None or args.kernel_dir is None: raise SystemExit("--research-root and --kernel-dir required")
        materialize(args.request,args.research_root,args.kernel_dir); return 0
    if args.kernel_dir is None or args.kaggle_bin is None: raise SystemExit("--kernel-dir and --kaggle-bin required")
    execute(args.request,args.kernel_dir,args.kaggle_bin); return 0

if __name__ == "__main__": raise SystemExit(main())
