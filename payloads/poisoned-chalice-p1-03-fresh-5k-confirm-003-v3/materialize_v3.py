#!/usr/bin/env python3
"""Materialize exact P1-03 confirm-003 v3 from the approved v2 payload.

The only runtime change is the fresh-audit provenance gate. The embedded v2
development audit is accepted only if reversing its exact approved direct-Parquet
loader reconstructs frozen science Git blob 3d0a45b.... No scientific formula,
cohort, execution-cell, or promotion-gate change is permitted.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import nbformat

V2_NOTEBOOK = "poisoned-chalice-p1-03-fresh-5k-confirmation-v2.ipynb"
V3_NOTEBOOK = "poisoned-chalice-p1-03-fresh-5k-confirmation-v3.ipynb"
V2_SHA256 = "b639ddb944d0b88ded392c053c183725aaaec09127bb15d1264e3d09ef83d43e"
V3_SHA256 = "5141709b43a48787e269083d86776eb23597c56dd902fc694181d841f81eda33"
EXPECTED_V2_DEV_BLOB = "c3239d6901fa82d643998b90c21641434278c256"
EXPECTED_BASE_AUDIT_GIT_BLOB = "3d0a45b20c93554e293b4db9c243cd1133f1fef9"

OLD_LOAD_BASE_AUDIT = '''def load_base_audit():
    data = BASE_AUDIT_PATH.read_bytes()
    if git_blob_sha(data) != EXPECTED_BASE_AUDIT_GIT_BLOB:
        raise RuntimeError("frozen development-audit implementation identity changed")
    spec = importlib.util.spec_from_file_location("p1_03_frozen_development_audit", BASE_AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen development-audit implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if tuple(module.FUSION_HIDDEN_WEIGHTS) != (0.25, 0.50, 0.75):
        raise RuntimeError("development fusion implementation contract changed")
    if module.MEAN_TOP5 != "mean_only_top5_ensemble":
        raise RuntimeError("hidden anchor name changed")
    return module
'''


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_v2_materializer(path: Path):
    spec = importlib.util.spec_from_file_location("p1_03_bridge_v2_materializer_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load v2 materializer contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = str(module.OLD_CONTENT_LOADER)
    new = str(module.NEW_CONTENT_LOADER)
    if not old or not new or old == new:
        raise RuntimeError("v2 transport constants invalid")
    return old, new


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


def repaired_load_base_audit(old_loader: str, new_loader: str) -> str:
    return f'''V2_TRANSPORT_OLD_CONTENT_LOADER = {old_loader!r}
V2_TRANSPORT_NEW_CONTENT_LOADER = {new_loader!r}


def load_base_audit():
    data = BASE_AUDIT_PATH.read_bytes()
    observed_blob = git_blob_sha(data)
    identity_mode = "original_exact"
    if observed_blob != EXPECTED_BASE_AUDIT_GIT_BLOB:
        source = data.decode("utf-8")
        if source.count(V2_TRANSPORT_NEW_CONTENT_LOADER) != 1:
            raise RuntimeError("development-audit transport identity changed outside approved v2 loader")
        normalized = source.replace(
            V2_TRANSPORT_NEW_CONTENT_LOADER,
            V2_TRANSPORT_OLD_CONTENT_LOADER,
            1,
        ).encode("utf-8")
        if git_blob_sha(normalized) != EXPECTED_BASE_AUDIT_GIT_BLOB:
            raise RuntimeError("development-audit transport normalization did not reconstruct frozen science blob")
        identity_mode = "v2_direct_parquet_normalized"
    print("P1_03_FRESH_CONFIRM_BASE_AUDIT_TRANSPORT_IDENTITY PASS mode=" + identity_mode)
    spec = importlib.util.spec_from_file_location("p1_03_frozen_development_audit", BASE_AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen development-audit implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if tuple(module.FUSION_HIDDEN_WEIGHTS) != (0.25, 0.50, 0.75):
        raise RuntimeError("development fusion implementation contract changed")
    if module.MEAN_TOP5 != "mean_only_top5_ensemble":
        raise RuntimeError("hidden anchor name changed")
    return module
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-dir", required=True, type=Path)
    parser.add_argument("--kernel-dir", required=True, type=Path)
    args = parser.parse_args()
    payload_dir = args.payload_dir.resolve()
    kernel_dir = args.kernel_dir.resolve()
    v2_materializer = payload_dir / "materialize_v2.py"
    if not v2_materializer.is_file():
        raise RuntimeError("v2 materializer missing")

    completed = subprocess.run(
        [sys.executable, str(v2_materializer), "--payload-dir", str(payload_dir), "--kernel-dir", str(kernel_dir)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stdout + completed.stderr).encode("utf-8", errors="replace")
        raise RuntimeError(
            f"v2 materialization failed rc={completed.returncode} diagnostic_sha256={hashlib.sha256(diagnostic).hexdigest()}"
        )

    v2_path = kernel_dir / V2_NOTEBOOK
    if sha256_file(v2_path) != V2_SHA256:
        raise RuntimeError("approved v2 Notebook SHA-256 changed")
    old_loader, new_loader = load_v2_materializer(v2_materializer)

    nb = nbformat.read(v2_path, as_version=4)
    if len(nb.cells) != 5:
        raise RuntimeError("approved v2 Notebook cell count changed")
    original_execute_cell = str(nb.cells[3].source)
    cell2 = str(nb.cells[2].source)

    dev_source = extract_literal_assignment(cell2, "DEVELOPMENT_AUDIT_SOURCE_TEXT")
    observed_dev_blob = git_blob_sha(dev_source.encode("utf-8"))
    if observed_dev_blob != EXPECTED_V2_DEV_BLOB:
        raise RuntimeError(f"v2 embedded development-audit blob changed: {observed_dev_blob}")
    if dev_source.count(new_loader) != 1:
        raise RuntimeError("v2 development audit does not contain exactly one approved direct-Parquet loader")
    normalized_dev = dev_source.replace(new_loader, old_loader, 1).encode("utf-8")
    if git_blob_sha(normalized_dev) != EXPECTED_BASE_AUDIT_GIT_BLOB:
        raise RuntimeError("v2 development audit does not normalize to frozen science blob")

    fresh_source = extract_literal_assignment(cell2, "FRESH_AUDIT_SOURCE_TEXT")
    if fresh_source.count(OLD_LOAD_BASE_AUDIT) != 1:
        raise RuntimeError(f"fresh audit patch marker count changed: {fresh_source.count(OLD_LOAD_BASE_AUDIT)}")
    repaired_fresh = fresh_source.replace(
        OLD_LOAD_BASE_AUDIT,
        repaired_load_base_audit(old_loader, new_loader),
        1,
    )
    compile(repaired_fresh, "run_p1_03_fresh_5k_confirmation_audit_v3_transport.py", "exec")
    for marker in (
        "FROZEN_FUSION_WEIGHTS = (0.75, 0.50)",
        'PRIMARY_FUSION = "rank_fusion_hidden_0.75"',
        'SECONDARY_FUSION = "rank_fusion_hidden_0.50"',
        "FUSION_TPR_DELTA_MIN = 0.01",
        "FUSION_TPR_BOOTSTRAP_LOWER_MIN = 0.0",
        "FUSION_AUC_DELTA_MIN = -0.005",
        "C0_static_logistic",
        "C1_hashed_char_token_ngram_logistic",
        "scalar_features_per_pooling_variant",
    ):
        if marker not in repaired_fresh:
            raise RuntimeError(f"scientific marker missing after repair: {marker}")

    nb.cells[2].source = replace_literal_assignment(cell2, "FRESH_AUDIT_SOURCE_TEXT", repaired_fresh)
    if str(nb.cells[3].source) != original_execute_cell:
        raise RuntimeError("v3 repair changed execution cell")

    v3_path = kernel_dir / V3_NOTEBOOK
    nbformat.write(nb, v3_path)
    observed_v3 = sha256_file(v3_path)
    if observed_v3 != V3_SHA256:
        raise RuntimeError(f"v3 Notebook SHA mismatch: {observed_v3}")
    v2_path.unlink()

    meta_path = kernel_dir / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    before = dict(meta)
    meta["code_file"] = V3_NOTEBOOK
    if {k for k in before if before[k] != meta[k]} != {"code_file"}:
        raise RuntimeError("v3 science metadata delta changed")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    files = sorted(path.name for path in kernel_dir.iterdir() if path.is_file())
    if files != ["kernel-metadata.json", V3_NOTEBOOK]:
        raise RuntimeError(f"unexpected v3 kernel files: {files}")
    print(json.dumps({
        "status": "pass",
        "v2_notebook_sha256": V2_SHA256,
        "v3_notebook_sha256": observed_v3,
        "v2_embedded_development_audit_git_blob": observed_dev_blob,
        "normalized_frozen_development_audit_git_blob": EXPECTED_BASE_AUDIT_GIT_BLOB,
        "science_formula_changes": 0,
        "execution_cell_changes": 0,
        "kaggle_write_calls": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
