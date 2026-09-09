#!/usr/bin/env python3
"""Build exact token-offset alignment for the historical Starter++ 10k cache.

The cache was produced by a notebook that explicitly installed
``transformers==5.0.0``. A credential-free differential run later established
that replaying the same pinned StarCoder2 tokenizer with the SERSEM author
runtime (transformers 4.52.0 / tokenizers 0.21.0) changes tokenization for a
material fraction of public files. This helper therefore isolates tokenization
in the historical runtime and exports only runner-local target-token offsets for
subsequent SERSEM structural weighting.

No labels are used here and no Kaggle API calls are made by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Sequence

MODEL_ID = "bigcode/starcoder2-3b"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
EXPECTED_TRANSFORMERS = "5.0.0"
EXPECTED_TOKENIZERS = "0.22.2"
EXPECTED_DATASET_ROWS = 49_040
EXPECTED_CACHE_SAMPLES = 10_000
EXPECTED_SHARDS = 40


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "<missing>"


def _runtime_contract() -> dict[str, str]:
    observed = {
        "transformers": _version("transformers"),
        "tokenizers": _version("tokenizers"),
        "huggingface-hub": _version("huggingface-hub"),
        "python": __import__("platform").python_version(),
    }
    if observed["transformers"] != EXPECTED_TRANSFORMERS:
        raise RuntimeError("historical_transformers_version_mismatch")
    if observed["tokenizers"] != EXPECTED_TOKENIZERS:
        raise RuntimeError("historical_tokenizers_version_mismatch")
    return observed


def _load_cache(cache_dir: Path):
    import pandas as pd

    pieces = []
    for shard in range(EXPECTED_SHARDS):
        path = cache_dir / f"token_statistics.part{shard:03d}.parquet"
        if not path.is_file():
            raise RuntimeError("historical_alignment_cache_shard_missing")
        frame = pd.read_parquet(path)
        # Do not use a leading-underscore column here: pandas.itertuples()
        # rewrites invalid/underscore-prefixed field names, which would make
        # source provenance ambiguous in the failure-only diagnostic path.
        frame["shard_index"] = shard
        pieces.append(frame)
    cache = pd.concat(pieces, ignore_index=True)
    required = {
        "sample_id",
        "language",
        "position",
        "window_start",
        "target_token_ids",
        "correct_z",
        "shard_index",
    }
    if not required.issubset(cache.columns):
        raise RuntimeError("historical_alignment_cache_schema_mismatch")
    key = ["sample_id", "position", "window_start"]
    if cache.duplicated(key).any():
        raise RuntimeError("historical_alignment_duplicate_window_key")
    if cache.sample_id.astype(str).nunique() != EXPECTED_CACHE_SAMPLES:
        raise RuntimeError("historical_alignment_cache_sample_count_mismatch")
    return cache


def _slice_target_offsets(
    full_ids: Sequence[int],
    full_offsets: Sequence[Sequence[int]],
    window_start: int,
    observed_target_ids: Sequence[int],
) -> tuple[bool, list[list[int]]]:
    start = int(window_start) + 1
    observed = [int(value) for value in observed_target_ids]
    stop = start + len(observed)
    expected = [int(value) for value in full_ids[start:stop]]
    if len(expected) != len(observed) or expected != observed:
        return False, []
    selected = full_offsets[start:stop]
    if len(selected) != len(observed):
        return False, []
    offsets = [[int(pair[0]), int(pair[1])] for pair in selected]
    return True, offsets


def build_alignment(
    dataset_parquet: Path,
    cache_dir: Path,
    output_parquet: Path,
    manifest_json: Path,
) -> dict[str, Any]:
    import pandas as pd
    from transformers import AutoTokenizer

    runtime = _runtime_contract()
    cache = _load_cache(cache_dir)
    scored_ids = set(cache.sample_id.astype(str))

    dataset = pd.read_parquet(dataset_parquet)
    required = {"sample_id", "language", "content", "membership"}
    if not required.issubset(dataset.columns):
        raise RuntimeError("historical_alignment_dataset_schema_mismatch")
    if (
        len(dataset) != EXPECTED_DATASET_ROWS
        or dataset.sample_id.duplicated().any()
        or dataset.membership.isna().any()
    ):
        raise RuntimeError("historical_alignment_dataset_contract_mismatch")
    source = dataset.loc[
        dataset.sample_id.astype(str).isin(scored_ids),
        ["sample_id", "language", "content"],
    ].copy()
    source["sample_id"] = source.sample_id.astype(str)
    if len(source) != EXPECTED_CACHE_SAMPLES or set(source.sample_id) != scored_ids:
        raise RuntimeError("historical_alignment_cache_ids_not_public_subset")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    by_sample = {
        sample_id: group
        for sample_id, group in cache.groupby(cache.sample_id.astype(str), sort=False)
    }

    alignment_rows: list[dict[str, Any]] = []
    mismatch_ids: list[str] = []
    mismatch_windows = 0
    mismatch_by_origin = {"primary": 0, "continuation": 0}
    aligned_windows = 0
    for row in source.itertuples(index=False):
        encoded = tokenizer(
            row.content,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        full_ids = encoded["input_ids"]
        full_offsets = encoded["offset_mapping"]
        if len(full_ids) != len(full_offsets):
            raise RuntimeError("historical_alignment_full_token_offset_mismatch")

        sample_failed = False
        token_rows = list(by_sample[row.sample_id].itertuples(index=False))
        token_rows.sort(key=lambda item: (int(item.window_start), str(item.position)))
        for token_row in token_rows:
            observed_ids = token_row.target_token_ids
            if len(observed_ids) != len(token_row.correct_z):
                raise RuntimeError("historical_alignment_cache_target_score_length_mismatch")
            matched, target_offsets = _slice_target_offsets(
                full_ids,
                full_offsets,
                int(token_row.window_start),
                observed_ids,
            )
            if not matched:
                mismatch_windows += 1
                sample_failed = True
                origin = "primary" if int(token_row.shard_index) < 24 else "continuation"
                mismatch_by_origin[origin] += 1
                continue
            aligned_windows += 1
            alignment_rows.append(
                {
                    "sample_id": row.sample_id,
                    "language": str(row.language),
                    "position": str(token_row.position),
                    "window_start": int(token_row.window_start),
                    "shard": int(token_row.shard_index),
                    "target_offsets": target_offsets,
                    "target_token_count": len(target_offsets),
                    "full_token_count": len(full_ids),
                }
            )
        if sample_failed:
            mismatch_ids.append(row.sample_id)

    mismatch_set_hash = hashlib.sha256(
        "\n".join(sorted(set(mismatch_ids))).encode("utf-8")
    ).hexdigest()
    if mismatch_windows:
        print(
            "HISTORICAL_TOKEN_ALIGNMENT_MISMATCH "
            + json.dumps(
                {
                    "mismatch_samples": len(set(mismatch_ids)),
                    "mismatch_windows": mismatch_windows,
                    "mismatch_sample_set_sha256": mismatch_set_hash,
                    "mismatch_by_origin": mismatch_by_origin,
                },
                sort_keys=True,
            )
        )
        raise RuntimeError("historical_token_alignment_failed")

    alignment = pd.DataFrame(alignment_rows)
    if len(alignment) != len(cache):
        raise RuntimeError("historical_alignment_window_count_mismatch")
    key = ["sample_id", "position", "window_start"]
    if alignment.duplicated(key).any():
        raise RuntimeError("historical_alignment_duplicate_output_key")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    alignment.to_parquet(output_parquet, index=False)

    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime": runtime,
        "samples": EXPECTED_CACHE_SAMPLES,
        "windows": int(len(alignment)),
        "aligned_windows": int(aligned_windows),
        "mismatch_samples": 0,
        "mismatch_windows": 0,
        "mismatch_sample_set_sha256": mismatch_set_hash,
        "boundary": (
            "tokenization replay only; structural weighting remains in the pinned SERSEM author runtime"
        ),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _self_test() -> None:
    ok, offsets = _slice_target_offsets(
        [10, 11, 12, 13, 14],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        1,
        [12, 13],
    )
    assert ok and offsets == [[2, 3], [3, 4]]
    bad, empty = _slice_target_offsets(
        [10, 11, 12],
        [(0, 1), (1, 2), (2, 3)],
        0,
        [99, 12],
    )
    assert not bad and empty == []
    print("HISTORICAL_TOKEN_ALIGNMENT_SELF_TEST PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-parquet", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-parquet", type=Path)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if None in (args.dataset_parquet, args.cache_dir, args.output_parquet, args.manifest_json):
        raise SystemExit("dataset/cache/output/manifest are required")
    try:
        manifest = build_alignment(
            args.dataset_parquet,
            args.cache_dir,
            args.output_parquet,
            args.manifest_json,
        )
    except Exception as exc:
        digest = hashlib.sha256(
            f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
        ).hexdigest()
        print(
            "HISTORICAL_TOKEN_ALIGNMENT_FAILURE "
            + json.dumps(
                {
                    "exception_type": type(exc).__name__,
                    "error_sha256": digest,
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        "HISTORICAL_TOKEN_ALIGNMENT_RESULT "
        + json.dumps(
            {
                "samples": manifest["samples"],
                "windows": manifest["windows"],
                "runtime": manifest["runtime"],
                "mismatch_windows": manifest["mismatch_windows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
