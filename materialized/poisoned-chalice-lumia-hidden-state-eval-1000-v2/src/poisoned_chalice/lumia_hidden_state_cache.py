"""Label-clean sharded cache for the frozen P1-03 1,000-row LUMIA experiment.

This module deliberately does not train a probe or compute a membership metric. It
serializes only the three already-frozen pooled hidden-state views and matched
same-forward scalar baselines after the 50-row runtime gate passed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable
import json
import math
import os
import time

import numpy as np
import pandas as pd

from poisoned_chalice.lumia_hidden_state_pilot import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    LumiaRuntimePilotConfig,
    extract_one_sample,
    select_runtime_pilot,
)

POOLING_VARIANTS = ("mean", "caller_weighted", "helper_weighted")
OUTPUT_BASELINES = (
    "zsigmoid_single_sequence",
    "caller_literal_output_weighted_single_sequence",
    "helper_language_aware_output_weighted_single_sequence",
)


@dataclass(frozen=True)
class LumiaHiddenStateCacheConfig:
    rows: int = 1000
    rows_per_language_label: int = 100
    seed: int = 20260909
    max_length: int = 8192
    shard_size: int = 50
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    dataset_id: str = DATASET_ID
    dataset_revision: str = DATASET_REVISION
    output_dir: str = "/kaggle/working/lumia_hidden_state_cache_1000_v1"
    dtype: str = "float32"
    compression: str = "npz_compressed"

    def extraction_config(self) -> LumiaRuntimePilotConfig:
        return LumiaRuntimePilotConfig(
            model_id=self.model_id,
            model_revision=self.model_revision,
            dataset_id=self.dataset_id,
            dataset_revision=self.dataset_revision,
            rows=self.rows,
            rows_per_language_label=self.rows_per_language_label,
            seed=self.seed,
            max_length=self.max_length,
            output_dir=self.output_dir,
            performance_metrics_computed=False,
        )


def select_cache_cohort(train: pd.DataFrame, config: LumiaHiddenStateCacheConfig) -> pd.DataFrame:
    cohort = select_runtime_pilot(train, config.extraction_config())
    if len(cohort) != config.rows:
        raise ValueError("cache cohort row count changed")
    if list(cohort.columns) != ["sample_id", "language", "content"]:
        raise ValueError("cache scoring cohort must physically exclude labels")
    return cohort


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _activation_arrays(result: dict[str, Any], expected_layers: int, expected_hidden: int) -> dict[str, np.ndarray]:
    activations = result.get("activations") or {}
    if sorted(activations) != list(range(expected_layers)):
        raise RuntimeError("cache sample does not contain every decoder layer")
    arrays: dict[str, np.ndarray] = {}
    for variant in POOLING_VARIANTS:
        value = np.stack(
            [np.asarray(activations[layer][variant], dtype=np.float32) for layer in range(expected_layers)],
            axis=0,
        )
        if value.shape != (expected_layers, expected_hidden):
            raise RuntimeError(f"cache activation shape changed for {variant}: {value.shape}")
        if not np.isfinite(value).all():
            raise RuntimeError(f"cache activation contains nonfinite values: {variant}")
        arrays[variant] = value
    return arrays


def _sample_metadata(index: int, row: Any, result: dict[str, Any], max_length: int) -> dict[str, Any]:
    output_scores = result.get("output_scores") or {}
    if sorted(output_scores) != sorted(OUTPUT_BASELINES):
        raise RuntimeError("matched output baseline set changed")
    if not all(math.isfinite(float(output_scores[name])) for name in OUTPUT_BASELINES):
        raise RuntimeError("matched output baseline contains nonfinite values")
    caller = result.get("caller_diagnostics") or {}
    helper = result.get("helper_diagnostics") or {}
    return {
        "execution_index": int(index),
        "sample_id": str(row.sample_id),
        "language": str(row.language),
        "seq_len": int(result["seq_len"]),
        "at_max_length": bool(int(result["seq_len"]) == int(max_length)),
        "forward_seconds": float(result["forward_seconds"]),
        "caller_ast_backend": str(caller.get("ast_backend")),
        "caller_linter_backend": str(caller.get("linter_backend")),
        "helper_ast_backend": str(helper.get("ast_backend")),
        "helper_linter_backend": str(helper.get("linter_backend")),
        "helper_lint_error_count": int(helper.get("lint_error_count", 0)),
        "zsigmoid_single_sequence": float(output_scores["zsigmoid_single_sequence"]),
        "caller_literal_output_weighted_single_sequence": float(
            output_scores["caller_literal_output_weighted_single_sequence"]
        ),
        "helper_language_aware_output_weighted_single_sequence": float(
            output_scores["helper_language_aware_output_weighted_single_sequence"]
        ),
    }


def run_hidden_state_cache(
    model: Any,
    tokenizer: Any,
    cohort: pd.DataFrame,
    runtime: Any,
    config: LumiaHiddenStateCacheConfig,
    *,
    extractor: Callable[..., dict[str, Any]] = extract_one_sample,
) -> dict[str, Any]:
    """Extract and seal the frozen 1k pooled-vector cache without labels.

    A whole shard is either validated and atomically committed or absent. Failed
    samples are never zero-filled. Token-by-layer activations and logits are never
    written to disk.
    """
    if list(cohort.columns) != ["sample_id", "language", "content"]:
        raise ValueError("cache scoring cohort must physically exclude labels")
    if len(cohort) != config.rows or cohort.sample_id.duplicated().any():
        raise ValueError("cache scoring cohort identity mismatch")
    if config.rows % config.shard_size:
        raise ValueError("cache rows must be divisible by shard size")
    if config.dtype != "float32" or config.compression != "npz_compressed":
        raise ValueError("frozen cache numerical/storage contract changed")

    output = Path(config.output_dir)
    shards_dir = output / "shards"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("cache output directory is not empty; refusing overwrite/resume ambiguity")
    shards_dir.mkdir(parents=True, exist_ok=True)

    extraction_config = config.extraction_config()
    expected_layers: int | None = None
    expected_hidden: int | None = None
    shard_buffers = {name: [] for name in POOLING_VARIANTS}
    metadata_rows: list[dict[str, Any]] = []
    shard_manifest: list[dict[str, Any]] = []
    forward_seconds: list[float] = []
    serialization_seconds = 0.0
    started = time.perf_counter()

    for index, row in enumerate(cohort.itertuples(index=False)):
        result = extractor(model, tokenizer, row.content, row.language, runtime, extraction_config)
        layers = int(result["num_layers"])
        hidden = int(result["hidden_size"])
        expected_layers = layers if expected_layers is None else expected_layers
        expected_hidden = hidden if expected_hidden is None else expected_hidden
        if layers != expected_layers or hidden != expected_hidden:
            raise RuntimeError("activation dimensions changed across cache samples")
        arrays = _activation_arrays(result, expected_layers, expected_hidden)
        for name in POOLING_VARIANTS:
            shard_buffers[name].append(arrays[name])
        metadata_rows.append(_sample_metadata(index, row, result, config.max_length))
        forward_seconds.append(float(result["forward_seconds"]))
        del result, arrays

        if (index + 1) % config.shard_size == 0:
            shard_index = index // config.shard_size
            start_index = index + 1 - config.shard_size
            stop_index = index + 1
            stacked = {
                name: np.stack(shard_buffers[name], axis=0).astype(np.float32, copy=False)
                for name in POOLING_VARIANTS
            }
            expected_shape = (config.shard_size, expected_layers, expected_hidden)
            for name, values in stacked.items():
                if values.shape != expected_shape or not np.isfinite(values).all():
                    raise RuntimeError(f"invalid cache shard array {name}: {values.shape}")
            shard_name = f"shard_{shard_index:04d}.npz"
            shard_path = shards_dir / shard_name
            serial_started = time.perf_counter()
            _atomic_write_npz(shard_path, **stacked)
            with np.load(shard_path, allow_pickle=False) as observed:
                if sorted(observed.files) != sorted(POOLING_VARIANTS):
                    raise RuntimeError("committed cache shard keys changed")
                for name in POOLING_VARIANTS:
                    if observed[name].shape != expected_shape or observed[name].dtype != np.float32:
                        raise RuntimeError("committed cache shard shape/dtype changed")
                    if not np.isfinite(observed[name]).all():
                        raise RuntimeError("committed cache shard contains nonfinite values")
            shard_digest = sha256_file(shard_path)
            serialization_seconds += time.perf_counter() - serial_started
            shard_manifest.append({
                "shard_index": shard_index,
                "path": f"shards/{shard_name}",
                "start_index": start_index,
                "stop_index_exclusive": stop_index,
                "rows": config.shard_size,
                "shape_per_variant": list(expected_shape),
                "dtype": "float32",
                "sha256": shard_digest,
                "bytes": shard_path.stat().st_size,
            })
            for name in POOLING_VARIANTS:
                shard_buffers[name].clear()
            del stacked

    if any(shard_buffers[name] for name in POOLING_VARIANTS):
        raise RuntimeError("partial cache shard remained after extraction")
    if len(shard_manifest) != config.rows // config.shard_size:
        raise RuntimeError("cache shard count changed")
    if len(metadata_rows) != config.rows:
        raise RuntimeError("cache metadata row count changed")

    cohort_lines = [
        _json_line({
            "execution_index": int(index),
            "sample_id": str(row.sample_id),
            "language": str(row.language),
        })
        for index, row in enumerate(cohort.itertuples(index=False))
    ]
    serial_started = time.perf_counter()
    _atomic_write_text(output / "cohort_manifest.jsonl", "\n".join(cohort_lines) + "\n")
    _atomic_write_text(
        output / "sample_metadata.jsonl",
        "\n".join(_json_line(row) for row in metadata_rows) + "\n",
    )
    serialization_seconds += time.perf_counter() - serial_started
    scoring_wall = time.perf_counter() - started

    token_counts = np.asarray([row["seq_len"] for row in metadata_rows], dtype=np.int64)
    forward = np.asarray(forward_seconds, dtype=np.float64)
    non_forward_non_serialization = scoring_wall - float(forward.sum()) - serialization_seconds
    cache_manifest = {
        "schema_version": 1,
        "task_id": "P1-03-lumia-hidden-state-cache-1000-v1",
        "config": asdict(config),
        "rows": config.rows,
        "labels_present_during_model_scoring": False,
        "performance_metrics_computed": False,
        "raw_token_layer_activations_persisted": False,
        "logits_persisted": False,
        "num_layers": int(expected_layers or 0),
        "hidden_size": int(expected_hidden or 0),
        "pooling_variants": list(POOLING_VARIANTS),
        "output_baselines": list(OUTPUT_BASELINES),
        "shards": shard_manifest,
        "metadata": {
            "cohort_manifest": "cohort_manifest.jsonl",
            "cohort_manifest_sha256": sha256_file(output / "cohort_manifest.jsonl"),
            "sample_metadata": "sample_metadata.jsonl",
            "sample_metadata_sha256": sha256_file(output / "sample_metadata.jsonl"),
        },
        "token_count": {
            "min": int(token_counts.min()),
            "median": float(np.median(token_counts)),
            "max": int(token_counts.max()),
            "sum": int(token_counts.sum()),
            "at_max_length": int((token_counts == config.max_length).sum()),
        },
        "timing": {
            "scoring_loop_wall_seconds": float(scoring_wall),
            "forward_seconds_sum": float(forward.sum()),
            "forward_seconds_median": float(np.median(forward)),
            "forward_seconds_max": float(forward.max()),
            "serialization_seconds": float(serialization_seconds),
            "non_forward_non_serialization_seconds": float(non_forward_non_serialization),
        },
    }
    serial_started = time.perf_counter()
    _atomic_write_text(output / "cache_manifest.json", json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n")
    cache_manifest["timing"]["cache_manifest_write_seconds"] = time.perf_counter() - serial_started
    return cache_manifest
