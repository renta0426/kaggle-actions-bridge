#!/usr/bin/env python3
"""Credential-free tokenizer runtime differential probe for Poisoned Chalice P1-03.

The historical Starter++ cache was created by a notebook that installed
``transformers==5.0.0``. The current SERSEM author runtime pins
``transformers==4.52.0`` and ``tokenizers==0.21.0``. This script compares only
public-data tokenization under those two runtimes. It never reads Kaggle output.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import struct
from pathlib import Path
from typing import Any

MODEL_ID = "bigcode/starcoder2-3b"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
PROBE_PER_LANGUAGE = 2_000


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "<missing>"


def _hash_ints(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<q", int(value)))
    return digest.hexdigest()


def _hash_offsets(values: list[tuple[int, int]] | list[list[int]]) -> str:
    digest = hashlib.sha256()
    for start, stop in values:
        digest.update(struct.pack("<qq", int(start), int(stop)))
    return digest.hexdigest()


def _probe(dataset_parquet: Path, output_json: Path, expected_transformers: str) -> dict[str, Any]:
    import pandas as pd
    from transformers import AutoTokenizer

    observed_transformers = _package_version("transformers")
    if observed_transformers != expected_transformers:
        raise RuntimeError(
            f"transformers_version_mismatch:{observed_transformers}!={expected_transformers}"
        )

    frame = pd.read_parquet(dataset_parquet)
    required = {"sample_id", "content", "language"}
    if not required.issubset(frame.columns):
        raise RuntimeError("probe_dataset_schema_mismatch")
    if frame.sample_id.duplicated().any():
        raise RuntimeError("probe_dataset_duplicate_sample_id")

    pieces = []
    for language in LANGUAGES:
        part = frame.loc[frame.language == language, ["sample_id", "content", "language"]]
        part = part.sort_values("sample_id").head(PROBE_PER_LANGUAGE)
        if len(part) != PROBE_PER_LANGUAGE:
            raise RuntimeError("probe_language_row_count_mismatch")
        pieces.append(part)
    probe = pd.concat(pieces, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    if len(probe) != PROBE_PER_LANGUAGE * len(LANGUAGES):
        raise RuntimeError("probe_total_row_count_mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    rows: list[dict[str, Any]] = []
    for row in probe.itertuples(index=False):
        encoded = tokenizer(
            row.content,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        ids = [int(value) for value in encoded["input_ids"]]
        offsets = [(int(start), int(stop)) for start, stop in encoded["offset_mapping"]]
        if len(ids) != len(offsets):
            raise RuntimeError("token_offset_length_mismatch")
        rows.append(
            {
                "sample_id": str(row.sample_id),
                "language": str(row.language),
                "token_count": len(ids),
                "token_ids_sha256": _hash_ints(ids),
                "offsets_sha256": _hash_offsets(offsets),
            }
        )

    payload = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "probe_policy": "lexicographically first 2000 public train sample_ids per language",
        "samples": len(rows),
        "runtime": {
            "transformers": observed_transformers,
            "tokenizers": _package_version("tokenizers"),
            "huggingface-hub": _package_version("huggingface-hub"),
            "python": __import__("platform").python_version(),
        },
        "rows": rows,
    }
    output_json.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _compare(left_path: Path, right_path: Path, output_json: Path) -> dict[str, Any]:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    if left.get("model_revision") != MODEL_REVISION or right.get("model_revision") != MODEL_REVISION:
        raise RuntimeError("model_revision_mismatch")
    left_rows = {row["sample_id"]: row for row in left["rows"]}
    right_rows = {row["sample_id"]: row for row in right["rows"]}
    if set(left_rows) != set(right_rows) or len(left_rows) != 10_000:
        raise RuntimeError("probe_identity_set_mismatch")

    id_mismatches: list[str] = []
    offset_mismatches: list[str] = []
    count_mismatches: list[str] = []
    by_language = {
        language: {
            "samples": 0,
            "token_id_mismatches": 0,
            "offset_mismatches": 0,
            "token_count_mismatches": 0,
        }
        for language in LANGUAGES
    }
    for sample_id in sorted(left_rows):
        a = left_rows[sample_id]
        b = right_rows[sample_id]
        if a["language"] != b["language"]:
            raise RuntimeError("probe_language_mismatch")
        language = a["language"]
        by_language[language]["samples"] += 1
        if a["token_ids_sha256"] != b["token_ids_sha256"]:
            id_mismatches.append(sample_id)
            by_language[language]["token_id_mismatches"] += 1
        if a["offsets_sha256"] != b["offsets_sha256"]:
            offset_mismatches.append(sample_id)
            by_language[language]["offset_mismatches"] += 1
        if int(a["token_count"]) != int(b["token_count"]):
            count_mismatches.append(sample_id)
            by_language[language]["token_count_mismatches"] += 1

    def set_digest(values: list[str]) -> str:
        return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()

    summary = {
        "schema_version": 1,
        "samples": len(left_rows),
        "historical_runtime": left["runtime"],
        "sersem_runtime": right["runtime"],
        "token_id_mismatch_samples": len(id_mismatches),
        "offset_mismatch_samples": len(offset_mismatches),
        "token_count_mismatch_samples": len(count_mismatches),
        "token_id_mismatch_set_sha256": set_digest(id_mismatches),
        "offset_mismatch_set_sha256": set_digest(offset_mismatches),
        "token_count_mismatch_set_sha256": set_digest(count_mismatches),
        "by_language": by_language,
    }
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _self_test() -> None:
    assert len(_hash_ints([1, 2, 3])) == 64
    assert len(_hash_offsets([(0, 1), (1, 3)])) == 64
    assert _hash_ints([1, 2]) != _hash_ints([2, 1])
    print("POISONED_CHALICE_TOKENIZER_RUNTIME_PROBE_SELF_TEST PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe")
    probe.add_argument("--dataset-parquet", type=Path, required=True)
    probe.add_argument("--output-json", type=Path, required=True)
    probe.add_argument("--expected-transformers", required=True)

    compare = sub.add_parser("compare")
    compare.add_argument("--historical-json", type=Path, required=True)
    compare.add_argument("--sersem-json", type=Path, required=True)
    compare.add_argument("--output-json", type=Path, required=True)

    sub.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "self-test":
        _self_test()
        return 0
    if args.command == "probe":
        payload = _probe(args.dataset_parquet, args.output_json, args.expected_transformers)
        print(
            "POISONED_CHALICE_TOKENIZER_PROBE_RESULT "
            + json.dumps({"samples": payload["samples"], "runtime": payload["runtime"]}, sort_keys=True)
        )
        return 0
    summary = _compare(args.historical_json, args.sersem_json, args.output_json)
    print("POISONED_CHALICE_TOKENIZER_DIFF_RESULT " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
