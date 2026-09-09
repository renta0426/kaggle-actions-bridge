#!/usr/bin/env python3
"""Validate, materialize, and one-shot launch P1-03 LUMIA 50-row runtime gate."""
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

REQUEST_ID = "20260909-poisoned-chalice-lumia-hs-runtime-50-v1-001"
TARGET = "renta0426/poisoned-chalice-lumia-hs-runtime-50-v1"
RESEARCH_COMMIT = "ce447351fabb89fb43363ac551ff7b703d6c53fa"
AUTHOR_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
MODEL_ID = "bigcode/starcoder2-3b"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
NOTEBOOK_NAME = "poisoned-chalice-lumia-hs-runtime-50-v1.ipynb"
OUTPUT_PREFIX = "lumia_hidden_state_runtime_pilot_50"
PERSISTENT_OUTPUTS = [
    f"{OUTPUT_PREFIX}/runtime.json",
    f"{OUTPUT_PREFIX}/run_manifest.json",
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
            raise RuntimeError(f"LUMIA runtime request changed: {key}")
    expected_resource = {
        "accelerator": "gpu",
        "machine_shape": "NvidiaTeslaT4",
        "expected_visible_gpu_count": 2,
        "expected_runtime_minutes": 25,
        "hard_timeout_minutes": 60,
        "max_active_runs": 1,
        "min_remaining_quota_hours": 1.0,
    }
    if request.get("resource") != expected_resource:
        raise RuntimeError("LUMIA runtime resource contract changed")
    if request.get("api_budget") != {"max_calls": 60}:
        raise RuntimeError("LUMIA runtime API budget changed")
    expected_clean = {
        "new_model_specific_holdout_opened": False,
        "public_leaderboard_feedback_used": False,
        "hidden_validation_labels_used": False,
        "validation_rows_used": False,
        "codeparrot_rows_or_labels_used": False,
        "row_level_persisted_membership_labels": False,
        "performance_metrics_computed": False,
        "automatic_promotion": False,
    }
    if request.get("clean_room") != expected_clean:
        raise RuntimeError("LUMIA runtime clean-room contract changed")
    expected_science = {
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
    if request.get("scientific_contract") != expected_science:
        raise RuntimeError("LUMIA runtime scientific contract changed")


def validate_research(research_root: Path) -> None:
    paths = {
        "core": research_root / "src/poisoned_chalice/lumia_hidden_state_pilot.py",
        "helper": research_root / "src/poisoned_chalice/sersem_author_faithful.py",
        "runtime": research_root / "src/poisoned_chalice/sersem_author_runtime.py",
        "builder": research_root / "scripts/build_lumia_hidden_state_runtime_pilot_notebook.py",
        "contract": research_root / "configs/p1_03_lumia_hidden_state_runtime_pilot_50_v1_20260909.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f"research source missing: {path}")
    for key in ("core", "helper", "runtime", "builder"):
        source = paths[key].read_text(encoding="utf-8")
        compile(source, str(paths[key]), "exec")
    core = paths["core"].read_text(encoding="utf-8")
    helper = paths["helper"].read_text(encoding="utf-8")
    runtime = paths["runtime"].read_text(encoding="utf-8")
    builder = paths["builder"].read_text(encoding="utf-8")
    contract = _load(paths["contract"])
    for marker in (
        AUTHOR_COMMIT, MODEL_REVISION, DATASET_REVISION,
        "rows: int = 50", "rows_per_language_label: int = 5", "max_length: int = 8192",
        'return cohort[["sample_id", "language", "content"]].copy()',
        "register_forward_hook", "caller_weighted", "helper_weighted",
        "performance_metrics_computed\": False", "persistent_sample_level_outputs\": False",
        "estimated_1000_minutes_wall_linear",
    ):
        if marker not in core:
            raise RuntimeError(f"LUMIA core marker missing: {marker}")
    for marker in ("build_author_character_weights", "token_weights_from_character_weights", "weighted_output_score"):
        if marker not in helper:
            raise RuntimeError(f"SERSEM helper marker missing: {marker}")
    for marker in ("AUTHOR_RUNTIME_PINS", '"flake8": "7.3.0"', "load_author_runtime", "released_split_tree_sitter_grammars"):
        if marker not in runtime:
            raise RuntimeError(f"SERSEM runtime marker missing: {marker}")
    for marker in (
        TARGET, AUTHOR_COMMIT, MODEL_REVISION, DATASET_REVISION,
        "CONFIG = LumiaRuntimePilotConfig()", "CONFIG.max_length",
        'list(cohort.columns) == ["sample_id", "language", "content"]',
        "run_runtime_pilot(model, tokenizer, cohort, runtime, CONFIG)",
        '"performance_metrics_computed": False', '"persistent_sample_level_outputs": False',
        "LUMIA_HIDDEN_STATE_RUNTIME_PILOT COMPLETE",
    ):
        if marker not in builder:
            raise RuntimeError(f"LUMIA builder marker missing: {marker}")
    for forbidden in ("roc_auc_score", "roc_curve", "average_precision_score", "competition_submit(", "submission.csv"):
        if forbidden in builder or forbidden in core:
            raise RuntimeError(f"performance/submission code forbidden in runtime gate: {forbidden}")
    if contract.get("task_id") != "P1-03-lumia-hidden-state-runtime-pilot-50-v1":
        raise RuntimeError("LUMIA runtime contract identity changed")
    if contract.get("status") != "implementation_pending_execution":
        raise RuntimeError("LUMIA runtime contract status changed")
    if contract.get("author_commit") != AUTHOR_COMMIT:
        raise RuntimeError("LUMIA author commit changed")
    if contract.get("model") != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "max_length": 8192,
        "precision": "fp16",
        "device_semantics": "transformers-4.52 author-compatible single CUDA model placement",
    }:
        raise RuntimeError("LUMIA model contract changed")
    data = contract.get("data") or {}
    if data.get("rows") != 50 or data.get("rows_per_language_label") != 5 or data.get("selection_seed") != 20260909:
        raise RuntimeError("LUMIA runtime cohort changed")
    if data.get("selection_labels_removed_before_model_scoring") is not True or data.get("validation_rows_used") is not False:
        raise RuntimeError("LUMIA runtime label/validation boundary changed")
    if contract.get("persistent_outputs") != PERSISTENT_OUTPUTS:
        raise RuntimeError("LUMIA persistent outputs changed")
    mechanism = contract.get("mechanism") or {}
    if mechanism.get("raw_activations_persisted") is not False or mechanism.get("sample_scores_persisted") is not False:
        raise RuntimeError("LUMIA sample output boundary changed")
    if (contract.get("advance_gate") or {}).get("direct_1000_row_single_t4_if_estimated_wall_minutes_lte") != 105:
        raise RuntimeError("LUMIA scale gate changed")


def _write_nbformat_shim(root: Path) -> Path:
    shim = root / "nbformat.py"
    shim.write_text(r'''import json
class AttrDict(dict):
    def __getattr__(self, name):
        try: return self[name]
        except KeyError as exc: raise AttributeError(name) from exc
    def __setattr__(self, name, value): self[name] = value
class V4:
    @staticmethod
    def new_notebook():
        return AttrDict(cells=[], metadata=AttrDict(), nbformat=4, nbformat_minor=5)
    @staticmethod
    def new_markdown_cell(source=""):
        return AttrDict(cell_type="markdown", metadata=AttrDict(), source=source)
    @staticmethod
    def new_code_cell(source=""):
        return AttrDict(cell_type="code", execution_count=None, metadata=AttrDict(), outputs=[], source=source)
v4 = V4()
def write(notebook, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(notebook, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
''', encoding="utf-8")
    return root


def validate_kernel(kernel_dir: Path) -> None:
    notebook_path = kernel_dir / NOTEBOOK_NAME
    metadata_path = kernel_dir / "kernel-metadata.json"
    observed = sorted(path.name for path in kernel_dir.iterdir() if path.is_file())
    if observed != ["kernel-metadata.json", NOTEBOOK_NAME]:
        raise RuntimeError(f"unexpected LUMIA kernel package files: {observed}")
    metadata = _load(metadata_path)
    expected = {
        "id": TARGET,
        "title": "Poisoned Chalice Lumia Hs Runtime 50 V1",
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
            raise RuntimeError(f"LUMIA kernel metadata changed: {key}")
    if _slugify(metadata["title"]) != TARGET.split("/", 1)[1]:
        raise RuntimeError("LUMIA title-derived slug mismatch")
    for key in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
        if metadata.get(key) != []:
            raise RuntimeError(f"unexpected LUMIA attached source: {key}")
    notebook = _load(notebook_path)
    cells = notebook.get("cells") or []
    if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor") != 5 or len(cells) != 10:
        raise RuntimeError("unexpected LUMIA notebook structure")
    ids = [cell.get("id") for cell in cells]
    if len(set(ids)) != 10 or not all(isinstance(value, str) and value for value in ids):
        raise RuntimeError("invalid LUMIA notebook cell IDs")
    code = "\n".join(str(cell.get("source", "")) for cell in cells)
    for marker in (
        AUTHOR_COMMIT, MODEL_REVISION, DATASET_REVISION, "max_length: int = 8192", "rows: int = 50",
        'list(cohort.columns) == ["sample_id", "language", "content"]',
        "run_runtime_pilot(model, tokenizer, cohort, runtime, CONFIG)",
        "LUMIA_HIDDEN_STATE_RUNTIME_PILOT COMPLETE performance_metrics=0 sample_outputs=0",
    ):
        if marker not in code:
            raise RuntimeError(f"generated LUMIA notebook marker missing: {marker}")
    for forbidden in ("roc_auc_score", "roc_curve", "average_precision_score", "competition_submit(", "submission.csv"):
        if forbidden in code:
            raise RuntimeError(f"forbidden LUMIA runtime code: {forbidden}")


def materialize(request_path: Path, research_root: Path, kernel_dir: Path) -> None:
    request = _load(request_path)
    validate_request(request)
    validate_research(research_root)
    builder = research_root / "scripts/build_lumia_hidden_state_runtime_pilot_notebook.py"
    with tempfile.TemporaryDirectory(prefix="lumia-nbformat-shim-") as temp:
        shim_root = _write_nbformat_shim(Path(temp))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(shim_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        completed = subprocess.run(
            [sys.executable, str(builder)], cwd=str(research_root), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120, check=False,
        )
    if completed.returncode != 0:
        diagnostic = (completed.stdout + completed.stderr).encode("utf-8", errors="replace")
        raise RuntimeError(
            "LUMIA runtime notebook builder failed "
            f"rc={completed.returncode} diagnostic_sha256={hashlib.sha256(diagnostic).hexdigest()}"
        )
    built_dir = research_root / "notebooks/experiments/poisoned-chalice-lumia-hs-runtime-50-v1"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    shutil.copytree(built_dir, kernel_dir)
    validate_kernel(kernel_dir)
    print("LUMIA_HS_RUNTIME_50_MATERIALIZE PASS rows=50 max_length=8192 labels_in_scoring=0 metrics=0 sample_outputs=0 gpu=T4 retries=0")


def execute(request_path: Path, kernel_dir: Path, kaggle_bin: Path) -> None:
    request = _load(request_path)
    validate_request(request)
    validate_kernel(kernel_dir)
    if not kaggle_bin.is_file():
        raise RuntimeError("locked Kaggle CLI missing")
    completed = subprocess.run(
        [str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False,
    )
    out, err = completed.stdout or b"", completed.stderr or b""
    if completed.returncode != 0:
        print(
            "LUMIA_HS_RUNTIME_50_AMBIGUOUS_WRITE "
            f"rc={completed.returncode} stdout_bytes={len(out)} stderr_bytes={len(err)} "
            f"stdout_sha256={hashlib.sha256(out).hexdigest()} stderr_sha256={hashlib.sha256(err).hexdigest()}"
        )
        raise RuntimeError("kaggle kernels push returned non-zero; no retry permitted")
    print(
        "LUMIA_HS_RUNTIME_50_LAUNCH_ACCEPTED "
        f"target={TARGET} stdout_bytes={len(out)} stderr_bytes={len(err)} "
        f"stdout_sha256={hashlib.sha256(out).hexdigest()} stderr_sha256={hashlib.sha256(err).hexdigest()} retries=0 submissions=0"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--kernel-dir", type=Path)
    parser.add_argument("--kaggle-bin", type=Path)
    args = parser.parse_args()
    request = _load(args.request)
    validate_request(request)
    if args.static:
        if args.research_root is None:
            raise SystemExit("--research-root required")
        validate_research(args.research_root)
        print("LUMIA_HS_RUNTIME_50_STATIC PASS")
        return 0
    if args.materialize:
        if args.research_root is None or args.kernel_dir is None:
            raise SystemExit("--research-root and --kernel-dir required")
        materialize(args.request, args.research_root, args.kernel_dir)
        return 0
    if args.kernel_dir is None or args.kaggle_bin is None:
        raise SystemExit("--kernel-dir and --kaggle-bin required")
    execute(args.request, args.kernel_dir, args.kaggle_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
