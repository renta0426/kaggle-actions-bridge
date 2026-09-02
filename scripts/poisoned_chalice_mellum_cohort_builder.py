"""Build a private CPU notebook that freezes the label-free Mellum cohort.

The public bridge stores only code and immutable public resource identifiers.
The 10k exclusion set and the selected 2,000 content hashes are materialized only
inside the private Kaggle run.  Membership labels are used solely to reproduce
the predeclared balanced evaluation cohort and are never persisted in the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap


TARGET = "renta0426/mellum-transfer-cohort-v1"
SOURCE_KERNEL = "renta0426/starter-plus-10k-v2-continuation"
UPSTREAM_REPOSITORY = "https://github.com/Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026"
UPSTREAM_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
UPSTREAM_EVAL_PATH = "data/7b_train_test/eval_results.parquet"
UPSTREAM_SHA256 = "6e8559279e8ebd94ec8c25561c91d38c454a4e109ea0813b07864ff6a66d4068"
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
EXPECTED_CURRENT_ROWS = 10_000
EXPECTED_OVERLAPS = 23
ROWS_PER_LANGUAGE_LABEL = 200
EXPECTED_OUTPUT_ROWS = 2_000


def _cell(cell_type: str, source: str, index: int) -> dict:
    payload = {
        "cell_type": cell_type,
        "id": f"pc-mellum-cohort-{index:02d}",
        "metadata": {},
        "source": textwrap.dedent(source).lstrip("\n").splitlines(keepends=True),
    }
    if cell_type == "code":
        payload |= {"execution_count": None, "outputs": []}
    return payload


SETUP_SOURCE = f'''
from pathlib import Path
import hashlib
import json
import re
import subprocess
import time

import numpy as np
import pandas as pd

TARGET = {TARGET!r}
SOURCE_KERNEL = {SOURCE_KERNEL!r}
UPSTREAM_REPOSITORY = {UPSTREAM_REPOSITORY!r}
UPSTREAM_COMMIT = {UPSTREAM_COMMIT!r}
UPSTREAM_EVAL_PATH = {UPSTREAM_EVAL_PATH!r}
UPSTREAM_SHA256 = {UPSTREAM_SHA256!r}
LANGUAGES = {LANGUAGES!r}
EXPECTED_CURRENT_ROWS = {EXPECTED_CURRENT_ROWS}
EXPECTED_OVERLAPS = {EXPECTED_OVERLAPS}
ROWS_PER_LANGUAGE_LABEL = {ROWS_PER_LANGUAGE_LABEL}
EXPECTED_OUTPUT_ROWS = {EXPECTED_OUTPUT_ROWS}
OUTPUT = Path("/kaggle/working/mellum_transfer_cohort_v1")
OUTPUT.mkdir(parents=True, exist_ok=True)
STARTED = time.perf_counter()
print({{"target": TARGET, "resource": "cpu", "rows": EXPECTED_OUTPUT_ROWS}})
'''


COHORT_SOURCE = r'''
# The completed continuation output contains the exact deterministic 10k sample
# manifest used by the current Stage 1 research.  Read only IDs/languages.
candidates = [
    path for path in Path("/kaggle/input").rglob("train_sample_manifest.parquet")
    if "starter-plus-10k-v2-continuation" in str(path)
]
if len(candidates) != 1:
    raise RuntimeError(f"expected one attached 10k manifest, got {len(candidates)}")
current_path = candidates[0]
current = pd.read_parquet(current_path, columns=["sample_id", "language"])
if len(current) != EXPECTED_CURRENT_ROWS or current.sample_id.duplicated().any():
    raise RuntimeError("current 10k manifest row/ID contract mismatch")
if set(current.language) != set(LANGUAGES):
    raise RuntimeError("current 10k manifest language contract mismatch")
current_hashes = current.sample_id.astype(str).str.rsplit("-", n=1).str[-1]
if not current_hashes.str.fullmatch(r"[0-9a-f]{64}").all():
    raise RuntimeError("current sample IDs do not end in SHA-256")
excluded_hashes = set(current_hashes)
if len(excluded_hashes) != EXPECTED_CURRENT_ROWS:
    raise RuntimeError("current 10k content hashes are not unique")

upstream_root = Path("/kaggle/working/sersem_upstream")
subprocess.run(
    [
        "git", "clone", "--filter=blob:none", "--no-checkout",
        UPSTREAM_REPOSITORY + ".git", str(upstream_root),
    ],
    check=True,
)
subprocess.run(
    ["git", "-C", str(upstream_root), "sparse-checkout", "init", "--cone"],
    check=True,
)
subprocess.run(
    ["git", "-C", str(upstream_root), "sparse-checkout", "set", "data"],
    check=True,
)
subprocess.run(
    ["git", "-C", str(upstream_root), "checkout", UPSTREAM_COMMIT],
    check=True,
)
actual_commit = subprocess.run(
    ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if actual_commit != UPSTREAM_COMMIT:
    raise RuntimeError(f"upstream commit mismatch: {actual_commit}")
source_path = upstream_root / UPSTREAM_EVAL_PATH
actual_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
if actual_sha256 != UPSTREAM_SHA256:
    raise RuntimeError(f"upstream artifact SHA-256 mismatch: {actual_sha256}")

required_columns = ["content", "language", "membership", "is_member", "split_role"]
source = pd.read_parquet(source_path, columns=required_columns)
source = source.dropna(subset=["content"]).copy()
source["content_sha256"] = source.content.map(
    lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
)
if source.content_sha256.duplicated().any():
    raise RuntimeError("upstream evaluation contains duplicate non-null content")
if set(source.language) != set(LANGUAGES):
    raise RuntimeError("upstream language contract mismatch")
if set(source.split_role.astype(str)) != {"eval"}:
    raise RuntimeError("upstream split_role contract mismatch")
source["label"] = source.is_member.astype(int)
expected_label = source.membership.map({"member": 1, "non-member": 0})
if expected_label.isna().any() or not np.array_equal(
    expected_label.to_numpy(int), source.label.to_numpy(int)
):
    raise RuntimeError("upstream membership labels are inconsistent")

overlap_mask = source.content_sha256.isin(excluded_hashes)
overlap_count = int(overlap_mask.sum())
if overlap_count != EXPECTED_OVERLAPS:
    raise RuntimeError(
        f"current-10k overlap contract mismatch: {overlap_count} != {EXPECTED_OVERLAPS}"
    )
source = source.loc[~overlap_mask].copy()

pieces = []
for language in LANGUAGES:
    for label in (0, 1):
        pool = source[(source.language == language) & (source.label == label)]
        if len(pool) < ROWS_PER_LANGUAGE_LABEL:
            raise RuntimeError(f"insufficient rows for {language}/{label}: {len(pool)}")
        pieces.append(
            pool.sort_values("content_sha256").head(ROWS_PER_LANGUAGE_LABEL).copy()
        )
selected = pd.concat(pieces, ignore_index=True)
selected["sample_id"] = (
    "previous-" + selected.language.str.lower() + "-" + selected.content_sha256
)
prediction_manifest = (
    selected[["sample_id", "content_sha256", "language"]]
    .sort_values("sample_id")
    .reset_index(drop=True)
)
prediction_manifest.insert(0, "sample_index", np.arange(len(prediction_manifest), dtype=np.int32))
if (
    len(prediction_manifest) != EXPECTED_OUTPUT_ROWS
    or prediction_manifest.sample_id.duplicated().any()
    or prediction_manifest.content_sha256.duplicated().any()
    or prediction_manifest.sample_index.tolist() != list(range(EXPECTED_OUTPUT_ROWS))
):
    raise RuntimeError("label-free cohort output coverage/order mismatch")
if prediction_manifest.groupby("language").size().to_dict() != {
    language: 2 * ROWS_PER_LANGUAGE_LABEL for language in LANGUAGES
}:
    raise RuntimeError("label-free cohort language balance mismatch")
for forbidden in ("label", "membership", "is_member", "content", "lumia_score"):
    if forbidden in prediction_manifest.columns:
        raise RuntimeError(f"forbidden output field: {forbidden}")

output_path = OUTPUT / "prediction_manifest.parquet"
temporary = output_path.with_suffix(".tmp.parquet")
prediction_manifest.to_parquet(temporary, index=False)
temporary.replace(output_path)
output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
summary = {
    "status": "complete",
    "experiment": "mellum-transfer-cohort-v1",
    "rows": len(prediction_manifest),
    "languages": prediction_manifest.language.value_counts().sort_index().to_dict(),
    "rows_per_language_label_used_for_selection": ROWS_PER_LANGUAGE_LABEL,
    "current_10k_rows": len(current),
    "current_10k_overlap_excluded": overlap_count,
    "upstream_repository": UPSTREAM_REPOSITORY,
    "upstream_commit": actual_commit,
    "upstream_eval_sha256": actual_sha256,
    "prediction_manifest_sha256": output_sha256,
    "prediction_manifest_columns": prediction_manifest.columns.tolist(),
    "target_labels_materialized_only_for_frozen_cohort_selection": True,
    "target_labels_persisted": False,
    "target_model_scores_computed": False,
    "hidden_stage1_validation_labels_used": False,
    "public_leaderboard_tuning_used": False,
    "runtime_seconds": time.perf_counter() - STARTED,
}
(OUTPUT / "run_manifest.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
# Do not emit row-level IDs/hashes to notebook output.
print(json.dumps(summary, indent=2, sort_keys=True, default=str))
'''


def build(output_dir: Path) -> tuple[Path, Path]:
    cells = [
        _cell(
            "markdown",
            """
            # Mellum transfer cohort v1

            Private CPU-only preparation of the predeclared 2,000-row alternate-model
            cohort. Target labels are used only for balanced cohort selection and are
            not persisted into the label-free output consumed by GPU scoring.
            """,
            0,
        ),
        _cell("code", SETUP_SOURCE, 1),
        _cell("code", COHORT_SOURCE, 2),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": TARGET,
        "title": "Mellum Transfer Cohort V1",
        "code_file": "mellum-transfer-cohort-v1.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["membership-inference", "transfer", "cohort", "cpu"],
        "dataset_sources": [],
        "kernel_sources": [SOURCE_KERNEL],
        "competition_sources": [],
        "model_sources": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / metadata["code_file"]
    metadata_path = output_dir / "kernel-metadata.json"
    notebook_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return notebook_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    notebook_path, metadata_path = build(args.output_dir)
    print(json.dumps({"notebook": str(notebook_path), "metadata": str(metadata_path)}))


if __name__ == "__main__":
    main()
