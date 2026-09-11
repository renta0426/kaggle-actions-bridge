#!/usr/bin/env python3
"""Materialize exact P1-03 confirm-003 v4 from the approved v3 payload.

The v4 repair is operational only:
1. inject the reconstructed research root into the imported development-audit
   module so the already-approved direct-Parquet loader can resolve local files;
2. persist the compact hidden-anchor result before post-audit execution.

No scientific formula, cohort, seed, CV, bootstrap, threshold, promotion gate,
or Competition submission behavior may change.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat

V3_NOTEBOOK = "poisoned-chalice-p1-03-fresh-5k-confirmation-v3.ipynb"
V4_NOTEBOOK = "poisoned-chalice-p1-03-fresh-5k-confirmation-v4.ipynb"
V3_SHA256 = "5141709b43a48787e269083d86776eb23597c56dd902fc694181d841f81eda33"
V4_SHA256 = "ca3d89ad6c3c156035ead68b31e8be931813de06598ef1205d409d64e02bfd76"

ROOT_CONTEXT_OLD = '''    spec.loader.exec_module(module)
    if tuple(module.FUSION_HIDDEN_WEIGHTS) != (0.25, 0.50, 0.75):
'''
ROOT_CONTEXT_NEW = '''    spec.loader.exec_module(module)
    module.ROOT = BASE_AUDIT_PATH.resolve().parents[1]
    print("P1_03_FRESH_CONFIRM_BASE_AUDIT_RUNTIME_CONTEXT PASS root=" + str(module.ROOT))
    if tuple(module.FUSION_HIDDEN_WEIGHTS) != (0.25, 0.50, 0.75):
'''

CHECKPOINT_OLD = '''print(
    "P1_03_FRESH_CONFIRM_HIDDEN_PHASE PASS "
    f"wall_seconds={time.perf_counter()-phase_a_started:.3f}"
)

phase_b_started = time.perf_counter()
'''
CHECKPOINT_NEW = '''print(
    "P1_03_FRESH_CONFIRM_HIDDEN_PHASE PASS "
    f"wall_seconds={time.perf_counter()-phase_a_started:.3f}"
)

# Persist the compact sealed hidden-anchor result before any post-audit step.
# This is an operational recovery checkpoint only; no raw hidden-state shards
# or target-model outputs beyond the already-computed OOF result are copied.
HIDDEN_CHECKPOINT = Path("/kaggle/working/p1_03_fresh_hidden_checkpoint_v1")
if HIDDEN_CHECKPOINT.exists():
    raise RuntimeError("fresh hidden checkpoint output already exists")
HIDDEN_CHECKPOINT.mkdir(parents=True)
checkpoint_files = (
    "metrics.json",
    "decision.json",
    "fold_selection.json",
    "run_manifest.json",
    "oof_predictions.csv",
)
checkpoint_entries = {}
for name in checkpoint_files:
    src = HIDDEN_OUTPUT / name
    if not src.is_file():
        raise RuntimeError(f"hidden checkpoint source missing: {name}")
    dst = HIDDEN_CHECKPOINT / name
    shutil.copy2(src, dst)
    checkpoint_entries[name] = {
        "bytes": int(dst.stat().st_size),
        "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
    }
checkpoint_manifest = {
    "schema_version": 1,
    "task_id": "P1-03-fresh-hidden-anchor-checkpoint-v1",
    "status": "complete",
    "source_task_id": hidden_manifest["task_id"],
    "cache_manifest_sha256": hidden_manifest["cache_manifest_sha256"],
    "rows": 5000,
    "raw_hidden_state_shards_copied": False,
    "competition_submission": False,
    "files": checkpoint_entries,
}
(HIDDEN_CHECKPOINT / "checkpoint_manifest.json").write_text(
    json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
print(
    "P1_03_FRESH_CONFIRM_HIDDEN_CHECKPOINT PASS "
    "files=6 raw_hidden_copied=0"
)

phase_b_started = time.perf_counter()
'''


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_literal_assignment(source: str, name: str) -> str:
    tree = ast.parse(source)
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(nodes) != 1:
        raise RuntimeError(f"expected one {name} assignment, found {len(nodes)}")
    value = ast.literal_eval(nodes[0].value)
    if not isinstance(value, str):
        raise RuntimeError(f"{name} not a string literal")
    return value


def replace_literal_assignment(source: str, name: str, new_value: str) -> str:
    tree = ast.parse(source)
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(nodes) != 1:
        raise RuntimeError(f"expected one {name} assignment, found {len(nodes)}")
    node = nodes[0]
    if node.lineno != node.end_lineno:
        raise RuntimeError(f"{name} assignment unexpectedly spans lines")
    lines = source.splitlines(keepends=True)
    suffix = "\n" if lines[node.lineno - 1].endswith("\n") else ""
    lines[node.lineno - 1] = f"{name} = {new_value!r}" + suffix
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-dir", required=True, type=Path)
    parser.add_argument("--kernel-dir", required=True, type=Path)
    args = parser.parse_args()
    payload_dir = args.payload_dir.resolve()
    kernel_dir = args.kernel_dir.resolve()
    v3_materializer = payload_dir / "materialize_v3.py"
    if not v3_materializer.is_file():
        raise RuntimeError("v3 materializer missing")

    completed = subprocess.run(
        [sys.executable, str(v3_materializer), "--payload-dir", str(payload_dir), "--kernel-dir", str(kernel_dir)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=360,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stdout + completed.stderr).encode("utf-8", errors="replace")
        raise RuntimeError(
            f"v3 materialization failed rc={completed.returncode} diagnostic_sha256={hashlib.sha256(diagnostic).hexdigest()}"
        )

    v3_path = kernel_dir / V3_NOTEBOOK
    if sha256_file(v3_path) != V3_SHA256:
        raise RuntimeError("approved v3 Notebook SHA-256 changed")
    nb = nbformat.read(v3_path, as_version=4)
    if len(nb.cells) != 5:
        raise RuntimeError("approved v3 Notebook cell count changed")

    cell2 = str(nb.cells[2].source)
    fresh_source = extract_literal_assignment(cell2, "FRESH_AUDIT_SOURCE_TEXT")
    if fresh_source.count(ROOT_CONTEXT_OLD) != 1:
        raise RuntimeError(f"v4 ROOT context marker changed: {fresh_source.count(ROOT_CONTEXT_OLD)}")
    repaired_fresh = fresh_source.replace(ROOT_CONTEXT_OLD, ROOT_CONTEXT_NEW, 1)
    compile(repaired_fresh, "run_p1_03_fresh_5k_confirmation_audit_v4_context.py", "exec")
    nb.cells[2].source = replace_literal_assignment(cell2, "FRESH_AUDIT_SOURCE_TEXT", repaired_fresh)

    cell3 = str(nb.cells[3].source)
    if cell3.count(CHECKPOINT_OLD) != 1:
        raise RuntimeError(f"v4 hidden-checkpoint marker changed: {cell3.count(CHECKPOINT_OLD)}")
    nb.cells[3].source = cell3.replace(CHECKPOINT_OLD, CHECKPOINT_NEW, 1)

    joined = "\n".join(str(cell.source) for cell in nb.cells)
    for marker in (
        "FROZEN_FUSION_WEIGHTS = (0.75, 0.50)",
        'PRIMARY_FUSION = "rank_fusion_hidden_0.75"',
        'SECONDARY_FUSION = "rank_fusion_hidden_0.50"',
        "FUSION_TPR_DELTA_MIN = 0.01",
        "FUSION_TPR_BOOTSTRAP_LOWER_MIN = 0.0",
        "FUSION_AUC_DELTA_MIN = -0.005",
        "C0_static_logistic",
        "C1_hashed_char_token_ngram_logistic",
        "token_5gram_simhash64_hamming_le_6",
        "scalar_features_per_pooling_variant",
        "P1_03_FRESH_CONFIRM_BASE_AUDIT_TRANSPORT_IDENTITY PASS",
        "P1_03_FRESH_CONFIRM_BASE_AUDIT_RUNTIME_CONTEXT PASS",
        "P1_03_FRESH_CONFIRM_HIDDEN_CHECKPOINT PASS",
        "p1_03_fresh_hidden_checkpoint_v1",
        "raw_hidden_state_shards_copied",
        "P1_03_FRESH_CONFIRM_DIRECT_PARQUET_GATE PASS",
        "raw_hidden_copied=0",
        "submission=0",
    ):
        if marker not in joined:
            raise RuntimeError(f"v4 contract marker missing: {marker}")
    for forbidden in ('split="validation"', "split='validation'", "competition_submit(", "submission.csv"):
        if forbidden in joined:
            raise RuntimeError(f"v4 forbidden marker present: {forbidden}")

    v4_path = kernel_dir / V4_NOTEBOOK
    nbformat.write(nb, v4_path)
    observed_v4 = sha256_file(v4_path)
    if observed_v4 != V4_SHA256:
        raise RuntimeError(f"v4 Notebook SHA mismatch: {observed_v4}")
    v3_path.unlink()

    meta_path = kernel_dir / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    before = dict(meta)
    meta["code_file"] = V4_NOTEBOOK
    if {key for key in before if before[key] != meta[key]} != {"code_file"}:
        raise RuntimeError("v4 science metadata delta changed")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    files = sorted(path.name for path in kernel_dir.iterdir() if path.is_file())
    if files != ["kernel-metadata.json", V4_NOTEBOOK]:
        raise RuntimeError(f"unexpected v4 kernel files: {files}")

    print(json.dumps({
        "status": "pass",
        "v3_notebook_sha256": V3_SHA256,
        "v4_notebook_sha256": observed_v4,
        "repair": "inject_imported_base_module_root_plus_pre_audit_hidden_checkpoint",
        "root_context_injections": 1,
        "hidden_checkpoint_files": 6,
        "science_formula_changes": 0,
        "cohort_changes": 0,
        "promotion_gate_changes": 0,
        "raw_hidden_state_shards_checkpointed": 0,
        "kaggle_write_calls": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
