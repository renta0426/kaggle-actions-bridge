#!/usr/bin/env python3
"""Materialize the science-approved P1-03 fresh-confirmation v2 repair.

Input is the exact frozen v1 kernel produced by the historical bridge materializer.
Only the two public-data transport locations changed by science PR #101 are
rewritten. The resulting Notebook must equal the science CI SHA-256 exactly.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import nbformat

V1_NOTEBOOK = "poisoned-chalice-p1-03-fresh-5k-confirmation-v1.ipynb"
V2_NOTEBOOK = "poisoned-chalice-p1-03-fresh-5k-confirmation-v2.ipynb"
V1_SHA256 = "4236585b0dbc1ea54309a91cbc72bfd226467b2209c34cb428cfdd7328c09daa"
V2_SHA256 = "b639ddb944d0b88ded392c053c183725aaaec09127bb15d1264e3d09ef83d43e"
V1_SCIENCE_ID = "renta0426/poisoned-chalice-p1-03-fresh-5k-confirmation-v1"
V1_SCIENCE_TITLE = "Poisoned Chalice P1-03 Fresh 5K Confirmation V1"

CANONICAL_TRAIN_SHA256 = {
    "Go": "bad55f40d02de12085d65a698fce51a19735e71873519230b8ffff32dac978b8",
    "Java": "2c84d0bc243ea72239df62a9a145de1f11fbfc8ca571878b7f95bef84bdb1cde",
    "Python": "77da06d94b21d4db8d7f199df8e5bac35eeda591c4120eff77c58db0a3a4f129",
    "Ruby": "807f60d3a5480ada232cf90d4c2caf1d28bcba7d6858cabfd6fbba8f712a6bb0",
    "Rust": "c0a21a8fa29bce7c9a79f8e5c687814d07f631311424868ed94cd0289a364849",
}

OLD_REMOTE_LOAD = '''from datasets import load_dataset
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
EXPECTED_ROWS = {"Go":10000, "Java":10000, "Python":10000, "Ruby":10000, "Rust":9040}
DATASET_ID = "Poisoned-Chalice/ICSE-2027-public"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
for language in LANGUAGES:
    frame = load_dataset(DATASET_ID, language, split="train", revision=DATASET_REVISION).to_pandas()
    if len(frame) != EXPECTED_ROWS[language]:
        raise RuntimeError(f"public train row count changed for {language}: {len(frame)}")
    if not {"sample_id", "membership"}.issubset(frame.columns):
        raise RuntimeError(f"public label columns missing for {language}")
    label_dir = data_root / language
    label_dir.mkdir(parents=True, exist_ok=True)
    frame[["sample_id", "membership"]].to_parquet(
        label_dir / "train-00000-of-00001.parquet", index=False
    )
    del frame
'''

NEW_DIRECT_LOAD = '''from huggingface_hub import hf_hub_download
import pandas as pd
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
EXPECTED_ROWS = {"Go":10000, "Java":10000, "Python":10000, "Ruby":10000, "Rust":9040}
DATASET_ID = "Poisoned-Chalice/ICSE-2027-public"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
CANONICAL_TRAIN_SHA256 = ''' + repr(CANONICAL_TRAIN_SHA256) + '''
HF_CACHE = Path("/tmp/p1-03-fresh-confirm-v2-hf")
HF_CACHE.mkdir(parents=True, exist_ok=True)
for language in LANGUAGES:
    filename = f"{language}/train-00000-of-00001.parquet"
    resolved = Path(hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=filename,
        revision=DATASET_REVISION,
        cache_dir=str(HF_CACHE),
    ))
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != CANONICAL_TRAIN_SHA256[language]:
        raise RuntimeError(f"canonical public train SHA mismatch for {language}: {digest}")
    frame = pd.read_parquet(resolved, columns=["sample_id", "membership"])
    if len(frame) != EXPECTED_ROWS[language]:
        raise RuntimeError(f"public train row count changed for {language}: {len(frame)}")
    if frame.sample_id.duplicated().any() or frame.membership.isna().any():
        raise RuntimeError(f"public train identity/label invalid for {language}")
    label_dir = data_root / language
    label_dir.mkdir(parents=True, exist_ok=True)
    local_path = label_dir / "train-00000-of-00001.parquet"
    shutil.copyfile(resolved, local_path)
    if hashlib.sha256(local_path.read_bytes()).hexdigest() != CANONICAL_TRAIN_SHA256[language]:
        raise RuntimeError(f"local canonical public train copy mismatch for {language}")
    del frame
print("P1_03_FRESH_CONFIRM_DIRECT_PARQUET_GATE PASS files=5 revision=" + DATASET_REVISION)
'''

OLD_CONTENT_LOADER = '''def load_public_content(hidden: pd.DataFrame) -> pd.DataFrame:
    from datasets import load_dataset
    ids = set(hidden.sample_id.astype(str)); parts = []
    for language in LANGUAGES:
        frame = load_dataset(DATASET_ID, language, split="train", revision=DATASET_REVISION).to_pandas()
        if len(frame) != EXPECTED_PUBLIC_ROWS[language]: raise RuntimeError(f"public train row count changed for {language}")
        if not {"sample_id", "membership", "content"}.issubset(frame.columns): raise RuntimeError(f"public train columns missing for {language}")
        frame = frame[frame.sample_id.astype(str).isin(ids)][["sample_id", "membership", "content"]].copy(); frame["language"] = language; parts.append(frame)
    public = pd.concat(parts, ignore_index=True)
    if len(public) != len(hidden) or public.sample_id.duplicated().any(): raise RuntimeError("public content join coverage changed")
    normalized = public.membership.astype("string").str.strip().str.lower().str.replace("_", "-", regex=False); public["label_public"] = normalized.map({"member": 1, "non-member": 0, "nonmember": 0})
    merged = hidden.merge(public[["sample_id", "language", "content", "label_public"]], on="sample_id", how="left", suffixes=("", "_public"), validate="one_to_one", sort=False)
    if merged.content.isna().any() or not (merged.language == merged.language_public).all() or not (merged.label == merged.label_public).all(): raise RuntimeError("public content identity mismatch")
    return merged.drop(columns=["language_public", "label_public"]).reset_index(drop=True)
'''

NEW_CONTENT_LOADER = '''def load_public_content(hidden: pd.DataFrame) -> pd.DataFrame:
    ids = set(hidden.sample_id.astype(str)); parts = []
    for language in LANGUAGES:
        path = ROOT / f"data/public/{language}/train-00000-of-00001.parquet"
        if not path.is_file(): raise RuntimeError(f"verified local public train missing for {language}")
        frame = pd.read_parquet(path, columns=["sample_id", "membership", "content"])
        if len(frame) != EXPECTED_PUBLIC_ROWS[language]: raise RuntimeError(f"public train row count changed for {language}")
        if not {"sample_id", "membership", "content"}.issubset(frame.columns): raise RuntimeError(f"public train columns missing for {language}")
        frame = frame[frame.sample_id.astype(str).isin(ids)][["sample_id", "membership", "content"]].copy(); frame["language"] = language; parts.append(frame)
    public = pd.concat(parts, ignore_index=True)
    if len(public) != len(hidden) or public.sample_id.duplicated().any(): raise RuntimeError("public content join coverage changed")
    normalized = public.membership.astype("string").str.strip().str.lower().str.replace("_", "-", regex=False); public["label_public"] = normalized.map({"member": 1, "non-member": 0, "nonmember": 0})
    merged = hidden.merge(public[["sample_id", "language", "content", "label_public"]], on="sample_id", how="left", suffixes=("", "_public"), validate="one_to_one", sort=False)
    if merged.content.isna().any() or not (merged.language == merged.language_public).all() or not (merged.label == merged.label_public).all(): raise RuntimeError("public content identity mismatch")
    return merged.drop(columns=["language_public", "label_public"]).reset_index(drop=True)
'''


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_literal_assignment(source: str, name: str, old: str, new: str) -> str:
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
    value = ast.literal_eval(node.value)
    if value.count(old) != 1:
        raise RuntimeError(f"{name} repair marker count changed: {value.count(old)}")
    repaired = value.replace(old, new)
    lines = source.splitlines(keepends=True)
    if node.lineno != node.end_lineno:
        raise RuntimeError(f"{name} assignment unexpectedly spans lines")
    suffix = "\n" if lines[node.lineno - 1].endswith("\n") else ""
    lines[node.lineno - 1] = f"{name} = {repaired!r}" + suffix
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-dir", required=True, type=Path)
    parser.add_argument("--kernel-dir", required=True, type=Path)
    args = parser.parse_args()
    payload_dir = args.payload_dir.resolve()
    kernel_dir = args.kernel_dir.resolve()
    v1_materializer = payload_dir / "materialize_v1.py"
    if not v1_materializer.is_file():
        raise RuntimeError("exact v1 materializer missing")

    completed = subprocess.run(
        [sys.executable, str(v1_materializer), "--payload-dir", str(payload_dir), "--kernel-dir", str(kernel_dir)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240, check=False,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stdout + completed.stderr).encode("utf-8", errors="replace")
        raise RuntimeError(
            f"v1 materialization failed rc={completed.returncode} diagnostic_sha256={hashlib.sha256(diagnostic).hexdigest()}"
        )

    v1_path = kernel_dir / V1_NOTEBOOK
    if sha256_file(v1_path) != V1_SHA256:
        raise RuntimeError("frozen v1 Notebook SHA-256 changed before repair")
    meta_path = kernel_dir / "kernel-metadata.json"
    before_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if before_meta.get("id") != V1_SCIENCE_ID or before_meta.get("title") != V1_SCIENCE_TITLE:
        raise RuntimeError("frozen v1 science metadata changed")

    nb = nbformat.read(v1_path, as_version=4)
    if len(nb.cells) != 5:
        raise RuntimeError("frozen v1 Notebook cell count changed")
    nb.cells[2].source = replace_literal_assignment(
        str(nb.cells[2].source),
        "DEVELOPMENT_AUDIT_SOURCE_TEXT",
        OLD_CONTENT_LOADER,
        NEW_CONTENT_LOADER,
    )
    cell3 = str(nb.cells[3].source)
    if cell3.count(OLD_REMOTE_LOAD) != 1:
        raise RuntimeError(f"remote loader repair marker count changed: {cell3.count(OLD_REMOTE_LOAD)}")
    nb.cells[3].source = cell3.replace(OLD_REMOTE_LOAD, NEW_DIRECT_LOAD)

    joined = "\n".join(str(cell.source) for cell in nb.cells)
    for marker in (
        "hf_hub_download",
        "P1_03_FRESH_CONFIRM_DIRECT_PARQUET_GATE PASS",
        "FROZEN_FUSION_WEIGHTS = (0.75, 0.50)",
        "rank_fusion_hidden_0.75",
        "rank_fusion_hidden_0.50",
        "C0_static_logistic",
        "C1_hashed_char_token_ngram_logistic",
        "scalar_features_per_pooling_variant",
        "run_outer_oof_5000",
        "paired_increment_bootstraps",
        "raw_hidden_copied=0",
        "submission=0",
    ):
        if marker not in joined:
            raise RuntimeError(f"v2 Notebook marker missing: {marker}")
    if 'frame = load_dataset(DATASET_ID, language, split="train"' in str(nb.cells[3].source):
        raise RuntimeError("remote split-inference loader survived v2 repair")

    v2_path = kernel_dir / V2_NOTEBOOK
    nbformat.write(nb, v2_path)
    if sha256_file(v2_path) != V2_SHA256:
        raise RuntimeError(f"science v2 Notebook SHA mismatch: {sha256_file(v2_path)}")
    v1_path.unlink()

    after_meta = dict(before_meta)
    after_meta["code_file"] = V2_NOTEBOOK
    if {key for key in before_meta if before_meta[key] != after_meta[key]} != {"code_file"}:
        raise RuntimeError("science v2 metadata delta changed")
    meta_path.write_text(json.dumps(after_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    files = sorted(path.name for path in kernel_dir.iterdir() if path.is_file())
    if files != ["kernel-metadata.json", V2_NOTEBOOK]:
        raise RuntimeError(f"unexpected v2 kernel files: {files}")
    print(json.dumps({
        "status": "pass",
        "v1_notebook_sha256": V1_SHA256,
        "v2_notebook_sha256": V2_SHA256,
        "science_formula_changes": 0,
        "kaggle_write_calls": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
