#!/usr/bin/env python3
"""Materialize the frozen P1-03 fresh-disjoint 5k cache from audited history.

The private science repository is not read at protected-run time.  Instead this
script consumes the exact materialized source chain already archived in the
public bridge at commit ac70f67bda548c8d0e2f29fe25f430cae9b2ff54, reproduces
the proven 1k -> 5k mechanical transform, and then applies only the fresh-cohort
transform frozen in science commit 01549b06596bb328e4c25ee0d984a039a11a306d.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

HISTORICAL_COMMIT = "ac70f67bda548c8d0e2f29fe25f430cae9b2ff54"
BASE_BUILDER_BLOB = "6ff76a25f235fb90ce9548c41ef23ba85593c787"
SCALE_BUILDER_BLOB = "6c5c5ad2674058579ec2d0a9c0c9b9b8e4c35f01"
SCALE_CONFIG_BLOB = "0dccee9b2e562ccab9f4b4a1e4c9f21cb5518b85"
SCIENCE_COMMIT = "01549b06596bb328e4c25ee0d984a039a11a306d"
SCIENCE_FRESH_BUILDER_BLOB = "4f593ac671440cf341de2c575576031f7642b1f9"
SCIENCE_FRESH_CONFIG_BLOB = "6fba79aadefd044e5f698c430a19788f03943173"
TARGET = "renta0426/poisoned-chalice-lumia-fresh-cache-5000-v1"
TITLE = "Poisoned Chalice Lumia Fresh Cache 5000 V1"
NOTEBOOK_NAME = "poisoned-chalice-lumia-fresh-cache-5000-v1.ipynb"
OUTPUT_PREFIX = "lumia_hidden_state_fresh_cache_5000_v1"
CONSUMED_MANIFEST_SHA256 = "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa"

HISTORICAL_BLOBS = {
    "scripts/build_lumia_hidden_state_cache_1000_notebook_v1.py": BASE_BUILDER_BLOB,
    "scripts/build_lumia_hidden_state_cache_5000_notebook_v1.py": SCALE_BUILDER_BLOB,
    "configs/p1_03_lumia_hidden_state_scale_5000_v1_20260910.json": SCALE_CONFIG_BLOB,
    "src/poisoned_chalice/sersem_author_faithful.py": "91f0fcdccd646902ccc939e0835090a517505347",
    "src/poisoned_chalice/sersem_author_runtime.py": "024156b0723cdc084803b400f6705bc76fbfbfc6",
    "src/poisoned_chalice/lumia_hidden_state_pilot.py": "d33fa4f4b0d258083d1503e4b4b883461f41ed8f",
    "src/poisoned_chalice/lumia_hidden_state_cache.py": "61182c7fcde131ed36c0791fa16928dcdc9b6e8b",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def replace_one(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} marker count changed: {count}")
    return text.replace(old, new)


def validate_historical(root: Path) -> None:
    for rel, expected in HISTORICAL_BLOBS.items():
        path = root / rel
        if not path.is_file():
            raise RuntimeError(f"historical source missing: {rel}")
        observed = git_blob_sha(path.read_bytes())
        if observed != expected:
            raise RuntimeError(f"historical blob mismatch: {rel} {observed} != {expected}")
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8"), rel, "exec")


def reproduce_scale_builder_source(base: str) -> str:
    # Exact semantics of historical bridge blob 6c5c5ad... .
    source = base.replace("1,000", "5,000").replace("1000", "5000")
    source = replace_one(
        source,
        "CONFIG = LumiaHiddenStateCacheConfig()",
        "CONFIG = LumiaHiddenStateCacheConfig(rows=5000, rows_per_language_label=500, shard_size=250, output_dir=\"/kaggle/working/lumia_hidden_state_cache_5000_v1\")",
        "scale CONFIG",
    )
    source = replace_one(source, '"rows_per_language_label": 100,', '"rows_per_language_label": 500,', "scale rows/cell")
    source = replace_one(source, '"shard_size": 50,', '"shard_size": 250,', "scale shard")
    source = replace_one(source, "timeout=3300", "timeout=9000", "scale timeout")
    required = (
        'TARGET = "renta0426/poisoned-chalice-lumia-hidden-state-cache-5000-v1"',
        'rows=5000, rows_per_language_label=500, shard_size=250',
        'output_dir="/kaggle/working/lumia_hidden_state_cache_5000_v1"',
        'assert cache["rows"] == 5000',
        'assert len(cache["shards"]) == 20',
        '"rows_per_language_label": 500',
        '"shard_size": 250',
        "timeout=9000",
    )
    for marker in required:
        if marker not in source:
            raise RuntimeError(f"reproduced 5k marker missing: {marker}")
    return source


def apply_fresh_transform(source: str) -> str:
    source = replace_one(
        source,
        'TARGET = "renta0426/poisoned-chalice-lumia-hidden-state-cache-5000-v1"',
        f'TARGET = "{TARGET}"',
        "fresh target",
    )
    source = replace_one(
        source,
        'TITLE = "Poisoned Chalice Lumia Hidden State Cache 5000 V1"',
        f'TITLE = "{TITLE}"',
        "fresh title",
    )
    source = source.replace("poisoned-chalice-lumia-hidden-state-cache-5000-v1", "poisoned-chalice-lumia-fresh-cache-5000-v1")
    source = source.replace("lumia_hidden_state_cache_5000_v1", OUTPUT_PREFIX)
    source = source.replace("lumia-cache-5000-v1", "lumia-fresh-cache-5000-v1")
    source = source.replace("LUMIA_CACHE_5000", "LUMIA_FRESH_CACHE_5000")
    source = source.replace("LUMIA_HIDDEN_STATE_CACHE_5000_V1", "LUMIA_HIDDEN_STATE_FRESH_CACHE_5000_V1")

    cache_line = 'CACHE_SOURCE = (ROOT / "src/poisoned_chalice/lumia_hidden_state_cache.py").read_text(encoding="utf-8")'
    source = replace_one(
        source,
        cache_line,
        cache_line + '\nCACHE_SOURCE = CACHE_SOURCE.replace("P1-03-lumia-hidden-state-cache-1000-v1", "P1-03-lumia-hidden-state-fresh-cache-5000-v1")',
        "fresh embedded cache task-id",
    )
    source = replace_one(
        source,
        '    "dataset_sources": [],\n    "kernel_sources": [],',
        '    "dataset_sources": ["renta0426/stage1-raw-fim-submission-v1-output"],\n    "kernel_sources": ["renta0426/poisoned-chalice-lumia-hidden-state-cache-5000-v1"],',
        "fresh input sources",
    )

    fresh_selector = r'''# Fresh confirmation cohort: full overlap with frozen Stage1 10k, zero overlap
# with consumed hidden development cohorts. Membership is used only here.
train_work = train.copy()
normalized_membership = (
    train_work.membership.astype("string")
    .str.strip().str.lower().str.replace("_", "-", regex=False)
)
train_work["_label"] = normalized_membership.map(
    {"member": 1, "non-member": 0, "nonmember": 0}
)
assert train_work["_label"].notna().all()

input_root = Path("/kaggle/input")
stage1_paths = sorted(
    path for path in input_root.rglob("features.part*.parquet")
    if "/train_10k/parts/" in path.as_posix()
)
assert len(stage1_paths) == 40, len(stage1_paths)
stage1_parts = []
for path in stage1_paths:
    part = pd.read_parquet(path, columns=["sample_id", "language", "label"])
    assert len(part) == 250
    stage1_parts.append(part)
stage1 = pd.concat(stage1_parts, ignore_index=True)
stage1["label"] = stage1.label.astype(int)
assert len(stage1) == 10000 and stage1.sample_id.is_unique
stage1_counts = stage1.groupby(["language", "label"]).size()
assert len(stage1_counts) == 10 and stage1_counts.eq(1000).all()

identity = stage1.merge(
    train_work[["sample_id", "language", "_label"]],
    on="sample_id", how="left", suffixes=("", "_public"), validate="one_to_one",
)
assert len(identity) == 10000
assert identity["_label"].notna().all()
assert (identity.language == identity.language_public).all()
assert (identity.label == identity._label.astype(int)).all()
stage1_ids = set(stage1.sample_id.astype(str))

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest_matches = [
    path for path in input_root.rglob("cache_manifest.json")
    if _sha256_file(path) == "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa"
]
assert len(manifest_matches) == 1, len(manifest_matches)
consumed_manifest_path = manifest_matches[0]
consumed_manifest = json.loads(consumed_manifest_path.read_text(encoding="utf-8"))
assert int(consumed_manifest["rows"]) == 5000
cohort_rel = consumed_manifest["metadata"]["cohort_manifest"]
consumed_cohort_path = consumed_manifest_path.parent / cohort_rel
assert _sha256_file(consumed_cohort_path) == consumed_manifest["metadata"]["cohort_manifest_sha256"]
consumed_rows = [json.loads(line) for line in consumed_cohort_path.read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(consumed_rows) == 5000
consumed_ids = {str(row["sample_id"]) for row in consumed_rows}
assert len(consumed_ids) == 5000

languages = list(EXPECTED_TRAIN_ROWS)
def _reconstruct_prior(rows_per_cell: int) -> set[str]:
    selected_ids = set()
    for language_index, language in enumerate(languages):
        for label in (0, 1):
            pool = train_work[(train_work.language == language) & (train_work._label == label)]
            sampled = pool.sample(
                n=rows_per_cell,
                random_state=20260909 + 100 * language_index + label,
                replace=False,
            )
            selected_ids.update(sampled.sample_id.astype(str))
    assert len(selected_ids) == rows_per_cell * 10
    return selected_ids

prior_1k_ids = _reconstruct_prior(100)
prior_50_ids = _reconstruct_prior(5)
prior_1k_subset_current = prior_1k_ids.issubset(consumed_ids)
prior_50_subset_current = prior_50_ids.issubset(consumed_ids)
exclude_ids = consumed_ids | prior_1k_ids | prior_50_ids

eligible = train_work[
    train_work.sample_id.astype(str).isin(stage1_ids)
    & ~train_work.sample_id.astype(str).isin(exclude_ids)
].copy()
eligible_counts = eligible.groupby(["language", "_label"]).size()
assert len(eligible_counts) == 10
assert eligible_counts.ge(500).all(), eligible_counts.to_dict()

pieces = []
for language_index, language in enumerate(languages):
    for label in (0, 1):
        pool = eligible[(eligible.language == language) & (eligible._label == label)]
        sampled = pool.sample(
            n=500,
            random_state=20260910 + 100 * language_index + label,
            replace=False,
        )
        pieces.append(sampled)
selected = pd.concat(pieces, ignore_index=True)
assert len(selected) == 5000 and selected.sample_id.is_unique
fresh_counts = selected.groupby(["language", "_label"]).size()
assert len(fresh_counts) == 10 and fresh_counts.eq(500).all()
fresh_ids = set(selected.sample_id.astype(str))
assert fresh_ids.issubset(stage1_ids)
assert fresh_ids.isdisjoint(consumed_ids)
assert fresh_ids.isdisjoint(prior_1k_ids)
assert fresh_ids.isdisjoint(prior_50_ids)

cohort = selected.sort_values("sample_id")[["sample_id", "language", "content"]].reset_index(drop=True)
assert list(cohort.columns) == ["sample_id", "language", "content"]
selection_audit = {
    "schema_version": 1,
    "task_id": "P1-03-fresh-disjoint-5k-confirmation-v1",
    "selection_seed": 20260910,
    "stage1_rows": 10000,
    "stage1_set_sha256": hashlib.sha256("\n".join(sorted(stage1_ids)).encode("utf-8")).hexdigest(),
    "consumed_hidden_rows": 5000,
    "consumed_hidden_manifest_sha256": "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa",
    "consumed_hidden_set_sha256": hashlib.sha256("\n".join(sorted(consumed_ids)).encode("utf-8")).hexdigest(),
    "prior_1k_reconstructed_rows": len(prior_1k_ids),
    "prior_50_reconstructed_rows": len(prior_50_ids),
    "prior_1k_subset_of_consumed_5k": prior_1k_subset_current,
    "prior_50_subset_of_consumed_5k": prior_50_subset_current,
    "eligible_rows": int(len(eligible)),
    "eligible_language_label_counts": {f"{language}|{int(label)}": int(value) for (language, label), value in eligible_counts.items()},
    "fresh_rows": 5000,
    "fresh_language_label_counts": {f"{language}|{int(label)}": int(value) for (language, label), value in fresh_counts.items()},
    "fresh_full_stage1_overlap": True,
    "fresh_current_hidden_overlap_rows": 0,
    "fresh_prior_1k_overlap_rows": 0,
    "fresh_prior_50_overlap_rows": 0,
    "labels_present_during_model_scoring": False,
}
selection_audit["fresh_cohort_set_sha256"] = hashlib.sha256("\n".join(sorted(fresh_ids)).encode("utf-8")).hexdigest()
del stage1_parts, stage1, identity, train_work, eligible, selected, pieces'''
    source = replace_one(source, "cohort = select_cache_cohort(train, CONFIG)", fresh_selector, "fresh cohort selector")
    source = replace_one(
        source,
        'cohort_set_hash = hashlib.sha256(\n    "\\n".join(sorted(cohort.sample_id.astype(str))).encode("utf-8")\n).hexdigest()',
        'cohort_set_hash = hashlib.sha256(\n    "\\n".join(sorted(cohort.sample_id.astype(str))).encode("utf-8")\n).hexdigest()\nassert cohort_set_hash == selection_audit["fresh_cohort_set_sha256"]',
        "fresh cohort hash gate",
    )
    source = replace_one(
        source,
        "cache = run_hidden_state_cache(model, tokenizer, cohort, runtime, CONFIG)",
        'cache = run_hidden_state_cache(model, tokenizer, cohort, runtime, CONFIG)\n(OUTPUT / "cohort_selection.json").write_text(json.dumps(selection_audit, indent=2, sort_keys=True) + "\\n", encoding="utf-8")',
        "fresh selection audit write",
    )
    source = replace_one(
        source,
        '    "cache_manifest.json", "cohort_manifest.jsonl", "sample_metadata.jsonl",\n    "runtime.json", "run_manifest.json", *expected_shards,',
        '    "cache_manifest.json", "cohort_manifest.jsonl", "sample_metadata.jsonl",\n    "cohort_selection.json", "runtime.json", "run_manifest.json", *expected_shards,',
        "fresh persistent outputs",
    )
    source = replace_one(
        source,
        '    "cache_manifest.json", "cohort_manifest.jsonl", "run_manifest.json",\n    "runtime.json", "sample_metadata.jsonl", "shards",',
        '    "cache_manifest.json", "cohort_manifest.jsonl", "cohort_selection.json", "run_manifest.json",\n    "runtime.json", "sample_metadata.jsonl", "shards",',
        "fresh top-level outputs",
    )
    source = source.replace("P1-03-lumia-hidden-state-cache-5000-v1", "P1-03-lumia-hidden-state-fresh-cache-5000-v1")
    return source


def write_nbformat_shim(directory: Path) -> None:
    (directory / "nbformat.py").write_text(
        '''import json\nclass A(dict):\n    def __getattr__(self,n):\n        try:return self[n]\n        except KeyError as e:raise AttributeError(n) from e\n    def __setattr__(self,n,v):self[n]=v\nclass V4:\n    def new_notebook(self):return A(cells=[],metadata=A(),nbformat=4,nbformat_minor=5)\n    def new_markdown_cell(self,source=""):return A(cell_type="markdown",metadata=A(),source=source)\n    def new_code_cell(self,source=""):return A(cell_type="code",execution_count=None,metadata=A(),outputs=[],source=source)\nv4=V4()\ndef write(nb,path):\n    with open(path,"w",encoding="utf-8") as f:json.dump(nb,f,ensure_ascii=False,indent=1);f.write("\\n")\n''',
        encoding="utf-8",
    )


def validate_kernel(kernel_dir: Path) -> None:
    meta = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    expected_meta = {
        "id": TARGET,
        "title": TITLE,
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["renta0426/stage1-raw-fim-submission-v1-output"],
        "kernel_sources": ["renta0426/poisoned-chalice-lumia-hidden-state-cache-5000-v1"],
        "competition_sources": [],
        "model_sources": [],
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            raise RuntimeError(f"fresh kernel metadata mismatch: {key}")
    notebook = json.loads((kernel_dir / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or len(notebook.get("cells") or []) != 4:
        raise RuntimeError("fresh Notebook structure changed")
    code = "\n".join(str(cell.get("source", "")) for cell in notebook["cells"])
    required = (
        "rows=5000, rows_per_language_label=500, shard_size=250",
        'output_dir="/kaggle/working/lumia_hidden_state_fresh_cache_5000_v1"',
        "stage1_paths = sorted(",
        "assert len(stage1_paths) == 40",
        CONSUMED_MANIFEST_SHA256,
        "random_state=20260910 + 100 * language_index + label",
        "fresh_ids.isdisjoint(consumed_ids)",
        "fresh_ids.isdisjoint(prior_1k_ids)",
        "fresh_ids.isdisjoint(prior_50_ids)",
        '"fresh_full_stage1_overlap": True',
        '"fresh_current_hidden_overlap_rows": 0',
        '"labels_present_during_model_scoring": False',
        "cohort_selection.json",
        'model = model.to("cuda:0").eval()',
        "run_hidden_state_cache(model, tokenizer, cohort, runtime, CONFIG)",
        'assert cache["rows"] == 5000',
        'assert len(cache["shards"]) == 20',
        "timeout=9000",
    )
    for marker in required:
        if marker not in code:
            raise RuntimeError(f"fresh Notebook marker missing: {marker}")
    forbidden = ("roc_auc_score", "roc_curve(", "average_precision_score", "competition_submit(")
    for marker in forbidden:
        if marker in code:
            raise RuntimeError(f"forbidden fresh cache capability: {marker}")


def materialize(research_root: Path, kernel_dir: Path) -> None:
    validate_historical(research_root)
    base = (research_root / "scripts/build_lumia_hidden_state_cache_1000_notebook_v1.py").read_text(encoding="utf-8")
    source = reproduce_scale_builder_source(base)
    source = apply_fresh_transform(source)
    compile(source, "fresh-derived-builder.py", "exec")
    with tempfile.TemporaryDirectory(prefix="p1fresh-nbformat-") as temp:
        shim = Path(temp)
        write_nbformat_shim(shim)
        old_path = list(sys.path)
        sys.path.insert(0, str(shim))
        try:
            namespace = {"__file__": str(research_root / "scripts/build_lumia_hidden_state_cache_5000_notebook_v1.py"), "__name__": "__main__"}
            exec(compile(source, namespace["__file__"], "exec"), namespace)
        finally:
            sys.path[:] = old_path
    built = research_root / "notebooks/experiments/poisoned-chalice-lumia-fresh-cache-5000-v1"
    if not built.is_dir():
        raise RuntimeError("fresh derived builder did not create target directory")
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    shutil.copytree(built, kernel_dir)
    validate_kernel(kernel_dir)
    notebook_sha = hashlib.sha256((kernel_dir / NOTEBOOK_NAME).read_bytes()).hexdigest()
    print(
        "P1_03_FRESH_CACHE_MATERIALIZE PASS "
        f"history={HISTORICAL_COMMIT} science={SCIENCE_COMMIT} rows=5000 "
        f"stage1_overlap=5000 consumed_overlap=0 metrics=0 notebook_sha256={notebook_sha}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--kernel-dir", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.research_root.resolve(), args.kernel_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
