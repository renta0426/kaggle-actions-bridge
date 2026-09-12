"""Sealed-cache validation and public-label join for P1-03a 5k scale run."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from poisoned_chalice.lumia_hidden_state_cache import (
    OUTPUT_BASELINES,
    POOLING_VARIANTS,
    sha256_file,
)

EXPECTED_ROWS = 5000
EXPECTED_LAYERS = 30
EXPECTED_HIDDEN = 3072
EXPECTED_SHARDS = 20
EXPECTED_SHARD_ROWS = 250
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")


def _read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ])


def load_sealed_cache_5000(root: str | Path):
    root = Path(root)
    manifest_path = root / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("rows", -1)) != EXPECTED_ROWS:
        raise ValueError("unexpected sealed 5k cache row count")
    if int(manifest.get("num_layers", -1)) != EXPECTED_LAYERS or int(manifest.get("hidden_size", -1)) != EXPECTED_HIDDEN:
        raise ValueError("unexpected sealed 5k cache dimensions")
    if manifest.get("labels_present_during_model_scoring") is not False:
        raise ValueError("sealed 5k cache label boundary changed")
    if manifest.get("performance_metrics_computed") is not False:
        raise ValueError("sealed 5k cache unexpectedly contains performance metrics")
    if manifest.get("raw_token_layer_activations_persisted") is not False or manifest.get("logits_persisted") is not False:
        raise ValueError("sealed 5k cache persistence boundary changed")
    if tuple(manifest.get("pooling_variants") or []) != POOLING_VARIANTS:
        raise ValueError("sealed 5k cache pooling variants changed")
    if tuple(manifest.get("output_baselines") or []) != OUTPUT_BASELINES:
        raise ValueError("sealed 5k cache output baselines changed")
    shards = manifest.get("shards") or []
    if len(shards) != EXPECTED_SHARDS:
        raise ValueError("sealed 5k cache shard count changed")

    metadata_path = root / manifest["metadata"]["sample_metadata"]
    cohort_path = root / manifest["metadata"]["cohort_manifest"]
    if sha256_file(metadata_path) != manifest["metadata"]["sample_metadata_sha256"]:
        raise ValueError("5k sample metadata hash mismatch")
    if sha256_file(cohort_path) != manifest["metadata"]["cohort_manifest_sha256"]:
        raise ValueError("5k cohort manifest hash mismatch")
    metadata = _read_jsonl(metadata_path)
    cohort = _read_jsonl(cohort_path)
    if len(metadata) != EXPECTED_ROWS or len(cohort) != EXPECTED_ROWS:
        raise ValueError("5k metadata coverage changed")
    expected_index = np.arange(EXPECTED_ROWS)
    if not np.array_equal(metadata.execution_index.to_numpy(), expected_index):
        raise ValueError("5k sample metadata execution order changed")
    if not np.array_equal(cohort.execution_index.to_numpy(), expected_index):
        raise ValueError("5k cohort execution order changed")
    if metadata.sample_id.tolist() != cohort.sample_id.tolist() or metadata.language.tolist() != cohort.language.tolist():
        raise ValueError("5k metadata/cohort identity mismatch")
    if metadata.sample_id.duplicated().any():
        raise ValueError("5k cache contains duplicate sample IDs")

    buffers = {name: [] for name in POOLING_VARIANTS}
    expected_start = 0
    for shard_index, shard in enumerate(shards):
        if int(shard["shard_index"]) != shard_index:
            raise ValueError("5k shard index changed")
        if int(shard["start_index"]) != expected_start or int(shard["rows"]) != EXPECTED_SHARD_ROWS:
            raise ValueError("5k shard coverage/size changed")
        path = root / shard["path"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError("5k shard hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if tuple(sorted(archive.files)) != tuple(sorted(POOLING_VARIANTS)):
                raise ValueError("5k shard keys changed")
            for name in POOLING_VARIANTS:
                values = np.asarray(archive[name], dtype=np.float32)
                if values.shape != (EXPECTED_SHARD_ROWS, EXPECTED_LAYERS, EXPECTED_HIDDEN):
                    raise ValueError(f"5k shard shape changed for {name}: {values.shape}")
                if not np.isfinite(values).all():
                    raise ValueError("5k shard contains nonfinite values")
                buffers[name].append(values)
        expected_start = int(shard["stop_index_exclusive"])
    if expected_start != EXPECTED_ROWS:
        raise ValueError("5k shard coverage incomplete")
    arrays = {name: np.concatenate(buffers[name], axis=0) for name in POOLING_VARIANTS}
    for name, values in arrays.items():
        if values.shape != (EXPECTED_ROWS, EXPECTED_LAYERS, EXPECTED_HIDDEN) or values.dtype != np.float32:
            raise ValueError(f"sealed 5k cache array changed: {name}")
    return arrays, metadata, manifest


def join_public_labels_5000(metadata: pd.DataFrame, public_train: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "language", "membership"}
    missing = required.difference(public_train.columns)
    if missing:
        raise ValueError(f"public label table missing columns: {sorted(missing)}")
    labels = public_train[list(required)].copy()
    normalized = labels.membership.astype("string").str.strip().str.lower().str.replace("_", "-", regex=False)
    labels["label"] = normalized.map({"member": 1, "non-member": 0, "nonmember": 0})
    if labels.label.isna().any() or labels.sample_id.duplicated().any():
        raise ValueError("public train label identity invalid")
    merged = metadata.merge(
        labels[["sample_id", "language", "label"]],
        on="sample_id",
        how="left",
        suffixes=("", "_public"),
        validate="one_to_one",
    )
    if merged.label.isna().any() or not (merged.language == merged.language_public).all():
        raise ValueError("sealed 5k cache/public label join changed")
    merged = merged.drop(columns=["language_public"])
    merged["label"] = merged.label.astype(int)
    counts = merged.groupby(["language", "label"]).size().to_dict()
    expected = {(language, label): 500 for language in LANGUAGES for label in (0, 1)}
    if counts != expected:
        raise ValueError(f"unexpected frozen 5k language-label balance: {counts}")
    return merged
