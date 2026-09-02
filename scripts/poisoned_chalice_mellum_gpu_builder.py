"""Build the private Mellum-4B direct-transfer GPU notebook.

The bridge pulls an already-completed private StarCoder2-7B notebook at runtime
and extracts only the tested Starter++ and feature-cache definition cells.  The
public repository therefore does not copy the private research implementation or
any row-level benchmark manifest.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import textwrap


TARGET = "renta0426/mellum-transfer-v1"
SOURCE_KERNEL = "renta0426/pseudo-stage2-starcoder2-7b-v1"
COHORT_KERNEL = "renta0426/mellum-transfer-cohort-v1"
MODEL_ID = "JetBrains/Mellum-4b-base"
MODEL_REVISION = "83cce2605fbdf6a3868627e9b0a5924e0072b94d"
UPSTREAM_REPOSITORY = "https://github.com/Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026"
UPSTREAM_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
UPSTREAM_EVAL_PATH = "data/7b_train_test/eval_results.parquet"
UPSTREAM_SHA256 = "6e8559279e8ebd94ec8c25561c91d38c454a4e109ea0813b07864ff6a66d4068"
EXPECTED_ROWS = 2_000
SAMPLES_PER_SHARD = 100


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _find_definition_cell(notebook: dict, markers: tuple[str, ...], label: str) -> str:
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


def _cell(cell_type: str, source: str, index: int) -> dict:
    payload = {
        "cell_type": cell_type,
        "id": f"pc-mellum-gpu-{index:02d}",
        "metadata": {},
        "source": textwrap.dedent(source).lstrip("\n").splitlines(keepends=True),
    }
    if cell_type == "code":
        payload |= {"execution_count": None, "outputs": []}
    return payload


SETUP_SOURCE = f'''
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

MELLUM_TARGET = {TARGET!r}
MELLUM_SOURCE_KERNEL = {SOURCE_KERNEL!r}
MELLUM_COHORT_KERNEL = {COHORT_KERNEL!r}
MELLUM_MODEL_ID = {MODEL_ID!r}
MELLUM_MODEL_REVISION = {MODEL_REVISION!r}
MELLUM_UPSTREAM_REPOSITORY = {UPSTREAM_REPOSITORY!r}
MELLUM_UPSTREAM_COMMIT = {UPSTREAM_COMMIT!r}
MELLUM_UPSTREAM_EVAL_PATH = {UPSTREAM_EVAL_PATH!r}
MELLUM_UPSTREAM_SHA256 = {UPSTREAM_SHA256!r}
MELLUM_EXPECTED_ROWS = {EXPECTED_ROWS}
MELLUM_SAMPLES_PER_SHARD = {SAMPLES_PER_SHARD}
MELLUM_TOTAL_DEADLINE_SECONDS = 10_800
MELLUM_OUTPUT = Path("/kaggle/working/mellum_transfer_v1")
MELLUM_PARTS = MELLUM_OUTPUT / "parts"
MELLUM_PARTS.mkdir(parents=True, exist_ok=True)
MELLUM_RUN_STARTED = time.perf_counter()

if not torch.cuda.is_available():
    raise RuntimeError("Mellum direct-transfer benchmark requires CUDA")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"frozen v1 expects exactly one T4 GPU, got {{torch.cuda.device_count()}}")
if "T4" not in torch.cuda.get_device_name(0).upper():
    raise RuntimeError(f"frozen v1 requires an NVIDIA T4, got {{torch.cuda.get_device_name(0)}}")

MELLUM_CONFIG = FeatureCacheV2Config(
    model_id=MELLUM_MODEL_ID,
    model_revision=MELLUM_MODEL_REVISION,
    max_length=768,
    max_batch_tokens=1_536,
    vocab_chunk_tokens=64,
    rank_vocab_block_size=8_192,
    train_samples_per_language=400,
    samples_per_shard=MELLUM_SAMPLES_PER_SHARD,
    output_dir=str(MELLUM_OUTPUT),
    feature_version="model-invariant-v1-mellum-4b",
    attention_implementation="sdpa",
    retain_token_statistics=False,
)
seed_everything(2027)
print({{"target": MELLUM_TARGET, "model": MELLUM_MODEL_ID,
       "revision": MELLUM_MODEL_REVISION, "gpu": torch.cuda.get_device_name(0)}})
'''


INPUT_SOURCE = r'''
manifest_candidates = [
    path for path in Path("/kaggle/input").rglob("prediction_manifest.parquet")
    if "mellum-transfer-cohort-v1" in str(path)
]
manifest_meta_candidates = [
    path for path in Path("/kaggle/input").rglob("run_manifest.json")
    if "mellum-transfer-cohort-v1" in str(path)
]
if len(manifest_candidates) != 1 or len(manifest_meta_candidates) != 1:
    raise RuntimeError(
        "expected exactly one attached cohort manifest and one cohort run manifest"
    )
cohort_manifest = json.loads(manifest_meta_candidates[0].read_text(encoding="utf-8"))
if (
    cohort_manifest.get("status") != "complete"
    or cohort_manifest.get("experiment") != "mellum-transfer-cohort-v1"
    or cohort_manifest.get("rows") != MELLUM_EXPECTED_ROWS
    or cohort_manifest.get("target_labels_persisted") is not False
    or cohort_manifest.get("target_model_scores_computed") is not False
):
    raise RuntimeError("attached cohort manifest contract mismatch")

prediction_manifest = pd.read_parquet(manifest_candidates[0])
expected_columns = ["sample_index", "sample_id", "content_sha256", "language"]
if prediction_manifest.columns.tolist() != expected_columns:
    raise RuntimeError(f"cohort columns differ from frozen contract: {prediction_manifest.columns.tolist()}")
if (
    len(prediction_manifest) != MELLUM_EXPECTED_ROWS
    or prediction_manifest.sample_id.duplicated().any()
    or prediction_manifest.content_sha256.duplicated().any()
    or prediction_manifest.sample_index.tolist() != list(range(MELLUM_EXPECTED_ROWS))
):
    raise RuntimeError("cohort coverage/order mismatch")
for forbidden in ("label", "membership", "is_member", "content", "lumia_score"):
    if forbidden in prediction_manifest.columns:
        raise RuntimeError(f"target field crossed GPU boundary: {forbidden}")

upstream_root = Path("/kaggle/working/sersem_upstream")
subprocess.run(
    [
        "git", "clone", "--filter=blob:none", "--no-checkout",
        MELLUM_UPSTREAM_REPOSITORY + ".git", str(upstream_root),
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
    ["git", "-C", str(upstream_root), "checkout", MELLUM_UPSTREAM_COMMIT],
    check=True,
)
actual_commit = subprocess.run(
    ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if actual_commit != MELLUM_UPSTREAM_COMMIT:
    raise RuntimeError(f"upstream commit mismatch: {actual_commit}")
source_path = upstream_root / MELLUM_UPSTREAM_EVAL_PATH
actual_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
if actual_sha256 != MELLUM_UPSTREAM_SHA256:
    raise RuntimeError(f"upstream artifact mismatch: {actual_sha256}")

# Read content/language only. Membership labels and prior-model scores are never
# materialized in this GPU scoring process.
source = pd.read_parquet(source_path, columns=["content", "language"])
source = source.dropna(subset=["content"]).copy()
source["content_sha256"] = source.content.map(
    lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
)
source = source[source.content_sha256.isin(set(prediction_manifest.content_sha256))].copy()
if len(source) != MELLUM_EXPECTED_ROWS or source.content_sha256.duplicated().any():
    raise RuntimeError("selected upstream content is incomplete or non-unique")

sample = prediction_manifest.merge(
    source[["content_sha256", "content", "language"]],
    on="content_sha256",
    how="left",
    validate="one_to_one",
    suffixes=("", "_source"),
)
if len(sample) != MELLUM_EXPECTED_ROWS or sample.content.isna().any():
    raise RuntimeError("Mellum content recovery failed")
if not (sample.language == sample.language_source).all():
    raise RuntimeError("Mellum content language mismatch")
sample = sample[["sample_index", "sample_id", "content", "language"]]
sample = sample.sort_values("sample_index").reset_index(drop=True)
if sample.sample_index.tolist() != list(range(MELLUM_EXPECTED_ROWS)):
    raise RuntimeError("Mellum input order is not contiguous")
print({"rows": len(sample), "languages": sample.language.value_counts().sort_index().to_dict(),
       "upstream_commit": actual_commit, "upstream_sha256": actual_sha256})
'''


MODEL_SOURCE = r'''
tokenizer = AutoTokenizer.from_pretrained(
    MELLUM_MODEL_ID,
    revision=MELLUM_MODEL_REVISION,
    use_fast=True,
)
if not tokenizer.is_fast:
    raise RuntimeError("frozen Mellum benchmark requires a fast tokenizer")
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Mellum tokenizer has neither pad nor EOS token")
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MELLUM_MODEL_ID,
    revision=MELLUM_MODEL_REVISION,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation=MELLUM_CONFIG.attention_implementation,
).to("cuda").eval()
resolved_model_revision = getattr(model.config, "_commit_hash", None)
resolved_tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
if resolved_model_revision and resolved_model_revision != MELLUM_MODEL_REVISION:
    raise RuntimeError(f"resolved model revision mismatch: {resolved_model_revision}")
if resolved_tokenizer_revision and resolved_tokenizer_revision != MELLUM_MODEL_REVISION:
    raise RuntimeError(f"resolved tokenizer revision mismatch: {resolved_tokenizer_revision}")
MELLUM_RUNTIME = {
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_devices": [torch.cuda.get_device_name(0)],
    "dtype": "torch.float16",
    "attention": model.config._attn_implementation,
    "input_device": str(model.get_input_embeddings().weight.device),
    "resolved_model_revision": resolved_model_revision,
    "resolved_tokenizer_revision": resolved_tokenizer_revision,
}
print(json.dumps(MELLUM_RUNTIME, indent=2, sort_keys=True))
'''


FIDELITY_SOURCE = r'''
pilot = sample.iloc[:2].copy()
pilot_records = make_window_records(pilot, tokenizer, MELLUM_CONFIG.max_length)
pilot_config = StarterPlusConfig(
    max_length=MELLUM_CONFIG.max_length,
    max_batch_tokens=MELLUM_CONFIG.max_batch_tokens,
    vocab_chunk_tokens=MELLUM_CONFIG.vocab_chunk_tokens,
    train_samples_per_language=400,
)
reference_started = time.perf_counter()
reference_features = extract_window_features(
    pilot_records, model, tokenizer, pilot_config
)
reference_seconds = time.perf_counter() - reference_started
optimized_started = time.perf_counter()
optimized_features, _, optimized_profile = extract_window_features_optimized(
    pilot_records, model, tokenizer, MELLUM_CONFIG, progress=False
)
optimized_seconds = time.perf_counter() - optimized_started
MELLUM_FIDELITY = compare_feature_frames(
    reference_features, optimized_features, atol=1e-6
)
MELLUM_FIDELITY |= {
    "reference_seconds": reference_seconds,
    "optimized_seconds": optimized_seconds,
    "profile": optimized_profile,
}
if MELLUM_FIDELITY.get("passed") is not True:
    raise RuntimeError("Mellum optimized extractor fidelity gate failed")
print(json.dumps(MELLUM_FIDELITY, indent=2, sort_keys=True, default=str))
del reference_features, optimized_features, pilot_records
cleaned = gc.collect()
torch.cuda.empty_cache()
'''


RUN_SOURCE = r'''
def _atomic_parquet(frame, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _label_free_rank(frame, column, minimum_rows=20):
    global_rank = frame[column].rank(method="average", pct=True)
    result = global_rank.copy()
    for _, index in frame.groupby("language", sort=False).groups.items():
        indices = list(index)
        if len(indices) >= minimum_rows:
            result.loc[indices] = frame.loc[indices, column].rank(
                method="average", pct=True
            )
    return result.to_numpy(float)


profiles = []
shard_count = math.ceil(MELLUM_EXPECTED_ROWS / MELLUM_SAMPLES_PER_SHARD)
for shard_index in range(shard_count):
    elapsed = time.perf_counter() - MELLUM_RUN_STARTED
    if elapsed >= MELLUM_TOTAL_DEADLINE_SECONDS:
        raise TimeoutError("Mellum global deadline exceeded before shard start")
    start = shard_index * MELLUM_SAMPLES_PER_SHARD
    stop = min(MELLUM_EXPECTED_ROWS, start + MELLUM_SAMPLES_PER_SHARD)
    shard = sample.iloc[start:stop].copy()
    records = make_window_records(shard, tokenizer, MELLUM_CONFIG.max_length)
    window_features, token_statistics, profile = extract_window_features_optimized(
        records, model, tokenizer, MELLUM_CONFIG, progress=True
    )
    if not token_statistics.empty:
        raise RuntimeError("token statistics retention must remain disabled")
    aggregated = aggregate_windows(window_features).merge(
        shard[["sample_index", "sample_id"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    compact = pd.DataFrame({
        "sample_index": aggregated.sample_index.astype(np.int32),
        "sample_id": aggregated.sample_id,
        "language": aggregated.language,
        "token_count": aggregated.token_count.astype(np.int32),
        "window_count": aggregated.window_count.astype(np.int16),
        "loss_multiwindow": aggregated["score_loss_mean__max"].astype(float),
        "loss_window_mean": aggregated["score_loss_mean__mean"].astype(float),
        "standard_minkpp": aggregated["min_kpp_zselect_10__max"].astype(float),
        "best_local_span": aggregated["best_local_64__max"].astype(float),
        "neg_mean_log_rank": -aggregated["mean_log_rank__mean"].astype(float),
    }).sort_values("sample_index").reset_index(drop=True)
    if (
        len(compact) != stop - start
        or compact.sample_id.duplicated().any()
        or compact.sample_index.tolist() != list(range(start, stop))
        or not np.isfinite(compact.select_dtypes(include=[np.number]).to_numpy(float)).all()
    ):
        raise RuntimeError(f"Mellum shard coverage/finiteness failure: {shard_index}")
    feature_path = MELLUM_PARTS / f"features.part{shard_index:03d}.parquet"
    profile_path = MELLUM_PARTS / f"profile.part{shard_index:03d}.json"
    _atomic_parquet(compact, feature_path)
    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    profiles.append(profile)
    print({"shard": shard_index, "rows": len(compact),
           "wall_seconds": profile["wall_seconds"]})
    del records, window_features, token_statistics, aggregated, compact
    gc.collect()
    torch.cuda.empty_cache()

parts = [pd.read_parquet(path) for path in sorted(MELLUM_PARTS.glob("features.part*.parquet"))]
if len(parts) != shard_count:
    raise RuntimeError(f"expected {shard_count} completed feature shards, got {len(parts)}")
features = pd.concat(parts, ignore_index=True).sort_values("sample_index").reset_index(drop=True)
if (
    len(features) != MELLUM_EXPECTED_ROWS
    or features.sample_id.duplicated().any()
    or features.sample_index.tolist() != list(range(MELLUM_EXPECTED_ROWS))
    or set(features.sample_id) != set(prediction_manifest.sample_id)
):
    raise RuntimeError("Mellum completed feature coverage/order mismatch")

features["rank_loss"] = _label_free_rank(features, "loss_multiwindow")
features["rank_minkpp"] = _label_free_rank(features, "standard_minkpp")
features["rank_local_span"] = _label_free_rank(features, "best_local_span")
features["membership_score"] = features[
    ["rank_loss", "rank_minkpp", "rank_local_span"]
].mean(axis=1)
if not np.isfinite(features.select_dtypes(include=[np.number]).to_numpy(float)).all():
    raise RuntimeError("Mellum final feature/score output contains non-finite values")

output_path = MELLUM_OUTPUT / "sample_features.parquet"
_atomic_parquet(features, output_path)
output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
profile_summary = {
    "shards": shard_count,
    "samples": MELLUM_EXPECTED_ROWS,
    "windows": int(sum(int(profile["windows"]) for profile in profiles)),
    "target_tokens": int(sum(int(profile["target_tokens"]) for profile in profiles)),
    "forward_seconds": float(sum(float(profile["forward_seconds"]) for profile in profiles)),
    "reduction_seconds": float(sum(float(profile["reduction_seconds"]) for profile in profiles)),
    "transfer_seconds": float(sum(float(profile["transfer_seconds"]) for profile in profiles)),
    "peak_allocated_bytes": int(max(int(profile["peak_allocated_bytes"]) for profile in profiles)),
}
manifest = {
    "status": "complete",
    "experiment": "mellum-transfer-v1",
    "method": "model_invariant_v1",
    "model_id": MELLUM_MODEL_ID,
    "model_revision": MELLUM_MODEL_REVISION,
    "rows": MELLUM_EXPECTED_ROWS,
    "source_definition_kernel": MELLUM_SOURCE_KERNEL,
    "cohort_kernel": MELLUM_COHORT_KERNEL,
    "cohort_prediction_manifest_sha256": cohort_manifest["prediction_manifest_sha256"],
    "source_repository": MELLUM_UPSTREAM_REPOSITORY,
    "source_commit": actual_commit,
    "source_eval_sha256": actual_sha256,
    "sample_features_sha256": output_sha256,
    "runtime": MELLUM_RUNTIME,
    "runtime_config": {
        "max_length": MELLUM_CONFIG.max_length,
        "max_batch_tokens": MELLUM_CONFIG.max_batch_tokens,
        "vocab_chunk_tokens": MELLUM_CONFIG.vocab_chunk_tokens,
        "rank_vocab_block_size": MELLUM_CONFIG.rank_vocab_block_size,
        "local_span_width": 64,
        "min_k_fraction": 0.10,
        "language_rank_min_rows": 20,
        "fusion_weights": {"loss": 1.0, "minkpp": 1.0, "local": 1.0},
        "deadline_seconds": MELLUM_TOTAL_DEADLINE_SECONDS,
    },
    "profile": profile_summary,
    "fidelity": MELLUM_FIDELITY,
    "score_min": float(features.membership_score.min()),
    "score_max": float(features.membership_score.max()),
    "score_mean": float(features.membership_score.mean()),
    "target_labels_embedded_in_gpu_notebook": False,
    "target_labels_used_for_training_or_normalization": False,
    "previous_model_scores_used": False,
    "hidden_stage1_validation_labels_used": False,
    "public_leaderboard_tuning_used": False,
    "submission_created": False,
    "runtime_seconds": time.perf_counter() - MELLUM_RUN_STARTED,
}
(MELLUM_OUTPUT / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
'''


def build(source_notebook: Path, output_dir: Path) -> tuple[Path, Path]:
    source = _load_notebook(source_notebook)
    starter = _find_definition_cell(
        source,
        (
            "class StarterPlusConfig",
            "def make_window_records",
            "def aggregate_windows",
            "def extract_window_features",
        ),
        "Starter++",
    )
    feature_cache = _find_definition_cell(
        source,
        (
            "class FeatureCacheV2Config",
            "def extract_window_features_optimized",
            "def compare_feature_frames",
        ),
        "feature-cache-v2",
    )
    feature_cache = feature_cache.replace(
        "from .starter_plus import StarterPlusConfig, dynamic_batches, summarize_tokens\n",
        "",
    )
    checks = {
        "source_model_definition": "bigcode/starcoder2-3b" in starter,
        "optimized_exact_rank": "rank_vocab_block_size" in feature_cache,
        "standard_minkpp": "min_kpp_zselect_10" in feature_cache or "add_standard_minkpp" in feature_cache,
        "retention_switch": "retain_token_statistics" in feature_cache,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"private source notebook contract changed: {failed}")

    python_sources = [
        starter,
        feature_cache,
        textwrap.dedent(SETUP_SOURCE),
        textwrap.dedent(INPUT_SOURCE),
        textwrap.dedent(MODEL_SOURCE),
        textwrap.dedent(FIDELITY_SOURCE),
        textwrap.dedent(RUN_SOURCE),
    ]
    for index, source_text in enumerate(python_sources):
        ast.parse(source_text, filename=f"mellum-cell-{index}")

    cells = [
        _cell(
            "markdown",
            """
            # Frozen Mellum-4B transfer v1

            Independent alternate-model benchmark for direct membership signals.
            The GPU notebook receives a label-free private cohort manifest and never
            materializes membership labels or previous-model scores.
            """,
            0,
        ),
        _cell(
            "code",
            "%pip install -q transformers==5.0.0 accelerate pyarrow scikit-learn\n",
            1,
        ),
    ]
    for source_text in python_sources:
        cells.append(_cell("code", source_text, len(cells)))

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
        "title": "Mellum Transfer V1",
        "code_file": "mellum-transfer-v1.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu", "membership-inference", "transfer", "mellum"],
        "dataset_sources": [],
        "kernel_sources": [COHORT_KERNEL],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
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
    print(json.dumps({"status": "built", "checks": checks, "cells": len(cells)}, sort_keys=True))
    return notebook_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-notebook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_notebook, args.output_dir)


if __name__ == "__main__":
    main()
