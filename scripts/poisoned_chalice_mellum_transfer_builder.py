'''Build the frozen label-free Mellum-4B transfer notebook from one private source kernel.

The source kernel is the completed StarCoder2-7B transfer notebook. It contains the
already-frozen, overlap-excluded 2,000-row cohort and the previously validated
Starter++/feature-cache definitions. This builder parses the cohort as data, drops
all labels and prior-model scores, and copies only definition cells into the new
private target notebook. No private source content is written to bridge logs or Git.
'''

from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
from pathlib import Path

import nbformat


EXPECTED_LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
EXPECTED_ROWS = 2_000
MODEL_ID = "JetBrains/Mellum-4b-base"
MODEL_REVISION = "83cce2605fbdf6a3868627e9b0a5924e0072b94d"
UPSTREAM_REPOSITORY = "https://github.com/Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026"
UPSTREAM_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
UPSTREAM_EVAL_PATH = "data/7b_train_test/eval_results.parquet"
UPSTREAM_SHA256 = "6e8559279e8ebd94ec8c25561c91d38c454a4e109ea0813b07864ff6a66d4068"
CANONICAL_RESEARCH_COMMIT = "9f6be68c27dc1f3326d68bed4e2abf80db893748"


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _find_definition_cell(
    notebook: dict,
    markers: tuple[str, ...],
    label: str,
) -> str:
    matches = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = _source(cell)
        if all(marker in source for marker in markers):
            matches.append(source)
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {label} definition cell, got {len(matches)}")
    return matches[0]


def _extract_transfer_records(notebook: dict) -> list[dict]:
    candidates = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = _source(cell)
        if "TRANSFER_MANIFEST = pd.DataFrame(" not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "TRANSFER_MANIFEST"
                for target in node.targets
            ):
                continue
            call = node.value
            if (
                not isinstance(call, ast.Call)
                or not isinstance(call.func, ast.Attribute)
                or call.func.attr != "DataFrame"
                or len(call.args) != 1
            ):
                raise SystemExit("TRANSFER_MANIFEST is not one literal DataFrame call")
            value = ast.literal_eval(call.args[0])
            candidates.append(value)
    if len(candidates) != 1:
        raise SystemExit(f"expected one literal TRANSFER_MANIFEST, got {len(candidates)}")
    records = candidates[0]
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise SystemExit("TRANSFER_MANIFEST literal is not a list of dictionaries")
    return records


def _prediction_records(records: list[dict]) -> list[dict]:
    required = {"sample_id", "content_sha256", "language", "membership", "label"}
    if len(records) != EXPECTED_ROWS:
        raise SystemExit(f"unexpected transfer cohort size: {len(records)}")
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    balance: dict[tuple[str, int], int] = {}
    result = []
    for row in records:
        missing = required.difference(row)
        if missing:
            raise SystemExit(f"transfer record missing fields: {sorted(missing)}")
        sample_id = str(row["sample_id"])
        digest = str(row["content_sha256"])
        language = str(row["language"])
        membership = str(row["membership"])
        label = int(row["label"])
        if sample_id in seen_ids or digest in seen_hashes:
            raise SystemExit("transfer cohort IDs/content hashes are not unique")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SystemExit("invalid transfer content SHA-256")
        if language not in EXPECTED_LANGUAGES:
            raise SystemExit(f"unexpected transfer language: {language}")
        if (membership, label) not in {("member", 1), ("non-member", 0)}:
            raise SystemExit("transfer membership/label mismatch")
        seen_ids.add(sample_id)
        seen_hashes.add(digest)
        balance[(language, label)] = balance.get((language, label), 0) + 1
        result.append(
            {
                "sample_id": sample_id,
                "content_sha256": digest,
                "language": language,
            }
        )
    expected_balance = {
        (language, label): 200
        for language in EXPECTED_LANGUAGES
        for label in (0, 1)
    }
    if balance != expected_balance:
        raise SystemExit(f"unexpected transfer balance: {balance}")
    result.sort(key=lambda row: row["sample_id"])
    for index, row in enumerate(result):
        row["sample_index"] = index
    return result


SETUP_SOURCE = r'''
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import subprocess
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERIMENT = "mellum-transfer-v1"
MODEL_ID = "JetBrains/Mellum-4b-base"
MODEL_REVISION = "83cce2605fbdf6a3868627e9b0a5924e0072b94d"
UPSTREAM_REPOSITORY = "https://github.com/Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026"
UPSTREAM_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
UPSTREAM_EVAL_PATH = "data/7b_train_test/eval_results.parquet"
UPSTREAM_SHA256 = "6e8559279e8ebd94ec8c25561c91d38c454a4e109ea0813b07864ff6a66d4068"
CANONICAL_RESEARCH_COMMIT = "9f6be68c27dc1f3326d68bed4e2abf80db893748"
EXPECTED_ROWS = 2000
SAMPLES_PER_SHARD = 100
TOTAL_DEADLINE_SECONDS = 10800
OUTPUT = Path("/kaggle/working/mellum_transfer_v1")
PARTS = OUTPUT / "parts"
PARTS.mkdir(parents=True, exist_ok=True)

for forbidden in ("label", "membership", "is_member", "lumia_score"):
    if forbidden in PREDICTION_MANIFEST:
        raise RuntimeError(f"forbidden target field embedded: {forbidden}")
if len(PREDICTION_MANIFEST) != EXPECTED_ROWS:
    raise RuntimeError("prediction manifest row count changed")

CONFIG = FeatureCacheV2Config(
    model_id=MODEL_ID,
    model_revision=MODEL_REVISION,
    max_length=768,
    max_batch_tokens=1536,
    vocab_chunk_tokens=64,
    rank_vocab_block_size=8192,
    train_samples_per_language=400,
    samples_per_shard=SAMPLES_PER_SHARD,
    output_dir=str(OUTPUT),
    feature_version="model-invariant-v1-mellum",
    attention_implementation="sdpa",
    retain_token_statistics=False,
)
seed_everything(2027)
print({
    "experiment": EXPERIMENT,
    "rows": len(PREDICTION_MANIFEST),
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
})
'''


SOURCE_SOURCE = r'''
upstream_root = Path("/kaggle/working/sersem_upstream")
if not upstream_root.exists():
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
    raise RuntimeError(f"upstream artifact mismatch: {actual_sha256}")

# Only content and language are read. Previous labels and SERSEM scores never
# enter the target-model prediction process.
source = pd.read_parquet(source_path, columns=["content", "language"])
source = source.dropna(subset=["content"]).copy()
source["content_sha256"] = source.content.map(
    lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
)
selected_hashes = set(PREDICTION_MANIFEST.content_sha256)
source = source[source.content_sha256.isin(selected_hashes)].copy()
if len(source) != EXPECTED_ROWS or source.content_sha256.duplicated().any():
    raise RuntimeError("selected upstream content is incomplete or non-unique")

sample = PREDICTION_MANIFEST.merge(
    source[["content_sha256", "content", "language"]],
    on="content_sha256",
    how="left",
    validate="one_to_one",
    suffixes=("", "_source"),
)
if len(sample) != EXPECTED_ROWS or sample.content.isna().any():
    raise RuntimeError("Mellum transfer content coverage failure")
if not (sample.language == sample.language_source).all():
    raise RuntimeError("Mellum transfer language mismatch")
sample = (
    sample[["sample_index", "sample_id", "content", "language"]]
    .sort_values("sample_index")
    .reset_index(drop=True)
)
if sample.sample_index.tolist() != list(range(EXPECTED_ROWS)):
    raise RuntimeError("Mellum transfer order is not contiguous")
print({
    "source_commit": actual_commit,
    "source_sha256": actual_sha256,
    "rows": len(sample),
    "languages": sample.language.value_counts().to_dict(),
})
'''


MODEL_SOURCE = r'''
if not torch.cuda.is_available():
    raise RuntimeError(
        "Mellum transfer requires CUDA; TPU is not a fidelity-equivalent substitute"
    )
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    use_fast=True,
)
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Mellum tokenizer has neither pad nor EOS token")
    tokenizer.pad_token = tokenizer.eos_token

load_kwargs = {
    "revision": MODEL_REVISION,
    "dtype": dtype,
    "low_cpu_mem_usage": True,
    "attn_implementation": CONFIG.attention_implementation,
}
if torch.cuda.device_count() > 1:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="balanced",
        **load_kwargs,
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        **load_kwargs,
    ).to("cuda").eval()

resolved_model_revision = getattr(model.config, "_commit_hash", None)
if resolved_model_revision and resolved_model_revision != MODEL_REVISION:
    raise RuntimeError(f"resolved model revision mismatch: {resolved_model_revision}")
resolved_tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
if resolved_tokenizer_revision and resolved_tokenizer_revision != MODEL_REVISION:
    raise RuntimeError(
        f"resolved tokenizer revision mismatch: {resolved_tokenizer_revision}"
    )
device_map = getattr(model, "hf_device_map", {"": str(next(model.parameters()).device)})
runtime = {
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_devices": [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ],
    "dtype": str(dtype),
    "device_map": {str(key): str(value) for key, value in device_map.items()},
    "attention": model.config._attn_implementation,
}
print(json.dumps(runtime, indent=2, sort_keys=True))
'''


FIDELITY_SOURCE = r'''
pilot_frame = sample.iloc[:2].copy()
pilot_records = make_window_records(pilot_frame, tokenizer, CONFIG.max_length)
reference_config = StarterPlusConfig(
    model_id=MODEL_ID,
    max_length=CONFIG.max_length,
    max_batch_tokens=CONFIG.max_batch_tokens,
    vocab_chunk_tokens=CONFIG.vocab_chunk_tokens,
    train_samples_per_language=400,
)
started = time.perf_counter()
reference_features = extract_window_features(
    pilot_records,
    model,
    tokenizer,
    reference_config,
)
reference_seconds = time.perf_counter() - started
started = time.perf_counter()
optimized_features, token_rows, pilot_profile = extract_window_features_optimized(
    pilot_records,
    model,
    tokenizer,
    CONFIG,
    progress=False,
)
optimized_seconds = time.perf_counter() - started
if not token_rows.empty:
    raise RuntimeError("retain_token_statistics=False was not honored")
fidelity = compare_feature_frames(
    reference_features,
    optimized_features,
    atol=1e-6,
)
fidelity |= {
    "reference_seconds": reference_seconds,
    "optimized_seconds": optimized_seconds,
    "profile": pilot_profile,
}
if not fidelity["passed"]:
    raise RuntimeError("Mellum exact feature fidelity failed")
(OUTPUT / "fidelity.json").write_text(
    json.dumps(fidelity, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
print(json.dumps(fidelity, indent=2, sort_keys=True, default=str))
del reference_features, optimized_features, token_rows, pilot_records
gc.collect()
torch.cuda.empty_cache()
'''


RUN_SOURCE = r'''
run_started = time.perf_counter()
profiles = []
shard_count = math.ceil(EXPECTED_ROWS / SAMPLES_PER_SHARD)

for shard_index in range(shard_count):
    elapsed = time.perf_counter() - run_started
    if elapsed >= TOTAL_DEADLINE_SECONDS:
        raise TimeoutError("Mellum transfer deadline exceeded before shard start")
    feature_path = PARTS / f"sample_features.part{shard_index:03d}.parquet"
    profile_path = PARTS / f"profile.part{shard_index:03d}.json"
    if feature_path.exists() and profile_path.exists():
        cached = pd.read_parquet(feature_path)
        expected = min(
            SAMPLES_PER_SHARD,
            EXPECTED_ROWS - shard_index * SAMPLES_PER_SHARD,
        )
        if len(cached) != expected or not cached.sample_id.is_unique:
            raise RuntimeError(f"cached shard contract mismatch: {shard_index}")
        profiles.append(json.loads(profile_path.read_text(encoding="utf-8")))
        print({"shard": shard_index, "status": "cached", "rows": len(cached)})
        continue

    start = shard_index * SAMPLES_PER_SHARD
    stop = min(EXPECTED_ROWS, start + SAMPLES_PER_SHARD)
    shard = sample.iloc[start:stop].copy()
    records = make_window_records(shard, tokenizer, CONFIG.max_length)
    window_features, token_rows, profile = extract_window_features_optimized(
        records,
        model,
        tokenizer,
        CONFIG,
        progress=True,
    )
    if not token_rows.empty:
        raise RuntimeError("retain_token_statistics=False was not honored")
    features = aggregate_windows(window_features)
    features = features.merge(
        shard[["sample_index", "sample_id"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    ).sort_values("sample_index").reset_index(drop=True)
    if (
        len(features) != stop - start
        or not features.sample_id.is_unique
        or features.sample_index.tolist() != list(range(start, stop))
    ):
        raise RuntimeError(f"Mellum shard coverage failure: {shard_index}")

    temporary = feature_path.with_suffix(".tmp.parquet")
    features.to_parquet(temporary, index=False)
    temporary.replace(feature_path)
    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    profiles.append(profile)
    print({
        "shard": shard_index,
        "status": "complete",
        "rows": len(features),
        "profile": profile,
    })
    del records, window_features, token_rows, features
    gc.collect()
    torch.cuda.empty_cache()

parts = [
    pd.read_parquet(path)
    for path in sorted(PARTS.glob("sample_features.part*.parquet"))
]
if len(parts) != shard_count:
    raise RuntimeError(f"expected {shard_count} shards, got {len(parts)}")
raw = pd.concat(parts, ignore_index=True).sort_values("sample_index").reset_index(drop=True)
if (
    len(raw) != EXPECTED_ROWS
    or not raw.sample_id.is_unique
    or raw.sample_index.tolist() != list(range(EXPECTED_ROWS))
):
    raise RuntimeError("completed Mellum feature coverage failure")
if set(raw.sample_id) != set(PREDICTION_MANIFEST.sample_id):
    raise RuntimeError("completed Mellum sample IDs differ from frozen cohort")

output = raw[
    [
        "sample_index",
        "sample_id",
        "language",
        "token_count",
        "window_count",
        "score_loss_mean__max",
        "score_loss_mean__mean",
        "min_kpp_zselect_10__max",
        "best_local_64__max",
        "mean_log_rank__mean",
    ]
].rename(
    columns={
        "score_loss_mean__max": "loss_multiwindow",
        "score_loss_mean__mean": "loss_window_mean",
        "min_kpp_zselect_10__max": "standard_minkpp",
        "best_local_64__max": "best_local_span",
    }
)
output["neg_mean_log_rank"] = -output.pop("mean_log_rank__mean")


def label_free_percentile(frame, column, min_language_rows=20):
    result = frame[column].rank(method="average", pct=True)
    for _, indices in frame.groupby("language", sort=False).groups.items():
        indices = list(indices)
        if len(indices) >= min_language_rows:
            result.loc[indices] = frame.loc[indices, column].rank(
                method="average",
                pct=True,
            )
    return result.to_numpy(float)


output["rank_loss"] = label_free_percentile(output, "loss_multiwindow")
output["rank_minkpp"] = label_free_percentile(output, "standard_minkpp")
output["rank_local_span"] = label_free_percentile(output, "best_local_span")
output["membership_score"] = output[
    ["rank_loss", "rank_minkpp", "rank_local_span"]
].mean(axis=1)
numeric = output.select_dtypes(include=[np.number]).drop(columns=["sample_index"])
if not np.isfinite(numeric.to_numpy(float)).all():
    raise RuntimeError("non-finite Mellum feature or score")
output.to_parquet(OUTPUT / "sample_features.parquet", index=False)

profile_summary = {
    "shards": shard_count,
    "samples": EXPECTED_ROWS,
    "windows": int(sum(int(profile["windows"]) for profile in profiles)),
    "target_tokens": int(sum(int(profile["target_tokens"]) for profile in profiles)),
    "forward_seconds": float(
        sum(float(profile["forward_seconds"]) for profile in profiles)
    ),
    "reduction_seconds": float(
        sum(float(profile["reduction_seconds"]) for profile in profiles)
    ),
    "transfer_seconds": float(
        sum(float(profile["transfer_seconds"]) for profile in profiles)
    ),
    "peak_allocated_bytes": int(
        max(int(profile["peak_allocated_bytes"]) for profile in profiles)
    ),
}
manifest = {
    "status": "complete",
    "experiment": EXPERIMENT,
    "method": "model_invariant_v1",
    "canonical_research_commit": CANONICAL_RESEARCH_COMMIT,
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "rows": EXPECTED_ROWS,
    "source_repository": UPSTREAM_REPOSITORY,
    "source_commit": UPSTREAM_COMMIT,
    "source_eval_sha256": UPSTREAM_SHA256,
    "feature_contract": {
        "max_length": CONFIG.max_length,
        "max_batch_tokens": CONFIG.max_batch_tokens,
        "standard_minkpp_fraction": 0.10,
        "local_span_width": 64,
        "fusion": "equal-weight label-free within-language percentile",
    },
    "profile": profile_summary,
    "fidelity": fidelity,
    "runtime": runtime,
    "score_min": float(output.membership_score.min()),
    "score_max": float(output.membership_score.max()),
    "score_mean": float(output.membership_score.mean()),
    "target_labels_embedded_in_gpu_notebook": False,
    "target_labels_used_for_training_or_normalization": False,
    "previous_model_scores_used": False,
    "hidden_stage1_validation_labels_used": False,
    "public_leaderboard_tuning_used": False,
    "submission_created": False,
    "wall_seconds": time.perf_counter() - run_started,
}
(OUTPUT / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
'''


def build(source_notebook: Path, output_dir: Path) -> None:
    notebook = json.loads(source_notebook.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not notebook.get("cells"):
        raise SystemExit("private source notebook is invalid")

    starter = _find_definition_cell(
        notebook,
        (
            "class StarterPlusConfig",
            "def make_window_records",
            "def aggregate_windows",
            "def extract_window_features",
        ),
        "Starter++",
    )
    feature_cache = _find_definition_cell(
        notebook,
        (
            "class FeatureCacheV2Config",
            "def extract_window_features_optimized",
            "def compare_feature_frames",
        ),
        "feature-cache-v2",
    )
    prediction_records = _prediction_records(_extract_transfer_records(notebook))

    feature_cache = feature_cache.replace(
        "from .starter_plus import StarterPlusConfig, dynamic_batches, summarize_tokens\n",
        "",
    )
    setup_candidates = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "TRANSFER_MANIFEST = pd.DataFrame(" in _source(cell)
    ]
    if len(setup_candidates) != 1:
        raise SystemExit("source setup cell is ambiguous")
    source_setup = _source(setup_candidates[0])
    checks = {
        "source_model": "bigcode/starcoder2-7b" in source_setup,
        "standard_minkpp": "min_kpp_zselect_10" in feature_cache,
        "local_64": "best_local_64" in starter,
        "blocked_rank": "rank_vocab_block_size" in feature_cache,
        "rows": len(prediction_records) == EXPECTED_ROWS,
        "labels_removed": not any(
            key in row
            for row in prediction_records
            for key in ("label", "membership", "is_member", "lumia_score")
        ),
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise SystemExit(f"private source contract changed: {failed}")

    setup = (
        "PREDICTION_MANIFEST = pd.DataFrame("
        + repr(prediction_records)
        + ")\n\n"
        + textwrap.dedent(SETUP_SOURCE)
    )
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Frozen Mellum-4B transfer v1\n\n"
            "A label-free alternate-model benchmark. Only sample identifiers, "
            "content hashes, languages, and target-model outputs cross the GPU "
            "prediction boundary. No competition submission is created."
        ),
        nbformat.v4.new_code_cell(
            "%pip install -q transformers==5.0.0 accelerate pyarrow"
        ),
        nbformat.v4.new_code_cell(starter),
        nbformat.v4.new_code_cell(feature_cache),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_code_cell(textwrap.dedent(SOURCE_SOURCE)),
        nbformat.v4.new_code_cell(textwrap.dedent(MODEL_SOURCE)),
        nbformat.v4.new_code_cell(textwrap.dedent(FIDELITY_SOURCE)),
        nbformat.v4.new_code_cell(textwrap.dedent(RUN_SOURCE)),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"pc-mellum-v1-{index:02d}"

    target = nbformat.v4.new_notebook()
    target.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    target.metadata.language_info = {"name": "python", "version": "3.12"}
    target.cells = cells
    nbformat.validate(target)

    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / "mellum-transfer-v1.ipynb"
    nbformat.write(target, notebook_path)
    metadata = {
        "id": "renta0426/mellum-transfer-v1",
        "title": "Mellum Transfer V1",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu", "membership-inference", "transfer", "mellum"],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "built",
                "cells": len(cells),
                "prediction_rows": len(prediction_records),
                "source_contract": checks,
                "target": metadata["id"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-notebook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_notebook, args.output_dir)


if __name__ == "__main__":
    main()
