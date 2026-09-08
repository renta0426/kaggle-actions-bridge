#!/usr/bin/env python3
"""Aggregate the historical 10k token cache into P1-03 SERSEM metrics.

This script is read-only with respect to Kaggle. Exact current-version/status
identity is established by the calling protected workflow before and after this
process. Raw Notebook output is kept only in temporary batch directories and is
removed immediately after each batch. The only persistent result is aggregate
metrics without sample identifiers or row-level predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any


EXPECTED_SCORE_NAMES = (
    "zsigmoid_window_mean",
    "sersem_output_windowed_formula_faithful",
    "sersem_max_context_union_variant",
)
SHARD_NAME = re.compile(r"^token_statistics\.part(?P<index>\d{3})\.parquet$")
KERNEL_REF = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _safe_failure(prefix: str, completed: subprocess.CompletedProcess[str]) -> RuntimeError:
    digest = hashlib.sha256((completed.stdout + completed.stderr).encode("utf-8", errors="replace")).hexdigest()
    return RuntimeError(f"{prefix} rc={completed.returncode} diagnostic_sha256={digest}")


def _membership_to_label(series):
    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    normalized = series.astype(str).str.strip().str.lower().str.replace("_", "-", regex=False)
    mapping = {"member": 1, "non-member": 0, "nonmember": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    result = normalized.map(mapping)
    if result.isna().any():
        raise RuntimeError("public_membership_label_contract_failed")
    return result.astype(int)


def _validate_request(request: dict[str, Any]) -> None:
    if request.get("schema_version") != 1:
        raise RuntimeError("request_schema_mismatch")
    if request.get("operation") != "kernel_current_output_aggregate":
        raise RuntimeError("request_operation_mismatch")
    if request.get("side_effects") != [] or request.get("automatic_compute_retries") != 0:
        raise RuntimeError("request_side_effect_contract_changed")
    if request.get("protected_job_private_repository_access") is not False:
        raise RuntimeError("private_repository_boundary_changed")
    if request.get("kaggle_compute") is not False or request.get("competition_submission") is not False:
        raise RuntimeError("readout_compute_contract_changed")
    scientific = request.get("scientific_contract", {})
    if scientific.get("comparison") != list(EXPECTED_SCORE_NAMES):
        raise RuntimeError("scientific_comparison_changed")
    if scientific.get("predictor_sealed_before_label_join") is not True:
        raise RuntimeError("predictor_seal_contract_changed")
    if scientific.get("sample_id_used_only_as_join_key") is not True:
        raise RuntimeError("sample_id_contract_changed")
    if scientific.get("public_lb_feedback_used") is not False or scientific.get("codeparrot_reused") is not False:
        raise RuntimeError("development_feedback_contract_changed")
    if scientific.get("new_model_forward") is not False:
        raise RuntimeError("unexpected_model_forward_contract")
    sources = request.get("cache_sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise RuntimeError("cache_source_count_mismatch")
    expected = [
        ("renta0426/starter-plus-10k-v2", 1, "ERROR", 0, 24),
        ("renta0426/starter-plus-10k-v2-continuation", 1, "COMPLETE", 24, 40),
    ]
    observed = [
        (
            item.get("target"),
            item.get("expected_current_version"),
            item.get("required_terminal_status"),
            item.get("shard_start"),
            item.get("shard_stop"),
        )
        for item in sources
    ]
    if observed != expected:
        raise RuntimeError("cache_source_identity_changed")
    download = request.get("download_contract", {})
    if download.get("file_family") != "token_statistics.partNNN.parquet":
        raise RuntimeError("download_family_changed")
    if download.get("expected_total_shards") != 40 or download.get("shards_per_download") != 2:
        raise RuntimeError("download_shard_contract_changed")
    if not all(download.get(key) is False for key in ("persist_raw_cache", "persist_sample_ids", "persist_predictions")):
        raise RuntimeError("raw_persistence_contract_changed")
    if download.get("aggregate_only") is not True:
        raise RuntimeError("aggregate_only_contract_changed")


def _load_public_train(request: dict[str, Any]):
    import pandas as pd
    from datasets import load_dataset

    spec = request["public_dataset"]
    frames = []
    for language in spec["languages"]:
        dataset = load_dataset(spec["id"], language, revision=spec["revision"], split=spec["split"])
        frame = dataset.to_pandas()[["sample_id", "content", "membership"]].copy()
        frame["language"] = language
        frames.append(frame)
    train = pd.concat(frames, ignore_index=True)
    if len(train) != 49_040 or train.sample_id.astype(str).duplicated().any():
        raise RuntimeError("canonical_public_train_contract_failed")
    train["sample_id"] = train.sample_id.astype(str)
    labels = train[["sample_id", "language", "membership"]].copy()
    labels["label"] = _membership_to_label(labels.membership)
    # Predictor-facing source deliberately excludes membership/label.
    source = train[["sample_id", "language", "content"]].copy()
    return source.set_index("sample_id", drop=False), labels[["sample_id", "language", "label"]]


def _load_science(science_dir: Path):
    sys.path.insert(0, str(science_dir))
    from sersem import build_sersem_char_weights, score_cached_windows, token_weights_from_char_weights
    from evaluation import low_fpr_metrics, overlap_tables, stratified_bootstrap_metrics

    return (
        build_sersem_char_weights,
        score_cached_windows,
        token_weights_from_char_weights,
        low_fpr_metrics,
        overlap_tables,
        stratified_bootstrap_metrics,
    )


def _parser_map():
    from tree_sitter_languages import get_parser

    names = {"Go": "go", "Java": "java", "Python": "python", "Ruby": "ruby", "Rust": "rust"}
    result = {}
    for language, name in names.items():
        try:
            result[language] = get_parser(name)
        except Exception as exc:
            raise RuntimeError(f"tree_sitter_parser_unavailable language={language}") from exc
    return result


def _download_batch(
    *,
    cli: Path,
    kernel: str,
    shard_indices: list[int],
    root: Path,
    max_single_bytes: int,
    max_batch_bytes: int,
):
    import pandas as pd

    if not KERNEL_REF.fullmatch(kernel):
        raise RuntimeError("invalid_kernel_ref")
    expected_names = [f"token_statistics.part{index:03d}.parquet" for index in shard_indices]
    alternation = "|".join(f"{index:03d}" for index in shard_indices)
    pattern = rf"(^|.*/)token_statistics\.part(?:{alternation})\.parquet$"
    completed = subprocess.run(
        [
            str(cli), "kernels", "output", kernel,
            "-p", str(root), "-q", "-o", "--page-size", "20",
            "--file-pattern", pattern,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        raise _safe_failure("kaggle_output_batch_failed", completed)
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("symlink_in_kaggle_output")
    files = [path for path in paths if path.is_file()]
    unexpected = [path.name for path in files if path.name not in expected_names]
    if unexpected:
        raise RuntimeError(f"output_allowlist_violated count={len(unexpected)}")
    frames = []
    total_bytes = 0
    observed_names = []
    for name in expected_names:
        matches = [path for path in files if path.name == name]
        if len(matches) != 1:
            raise RuntimeError(f"expected_shard_missing_or_ambiguous name={name}")
        size = matches[0].stat().st_size
        if not 0 < size <= max_single_bytes:
            raise RuntimeError(f"shard_size_contract_failed name={name} bytes={size}")
        total_bytes += size
        observed_names.append(name)
        frames.append(pd.read_parquet(matches[0]))
    if total_bytes > max_batch_bytes:
        raise RuntimeError(f"batch_size_contract_failed bytes={total_bytes}")
    frame = pd.concat(frames, ignore_index=True)
    required = {"sample_id", "language", "window_start", "target_token_ids", "correct_z"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"token_cache_columns_missing count={len(missing)}")
    return frame, total_bytes, observed_names


def _score_all(request: dict[str, Any], cli: Path, science_dir: Path):
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    (
        build_sersem_char_weights,
        score_cached_windows,
        token_weights_from_char_weights,
        low_fpr_metrics,
        overlap_tables,
        stratified_bootstrap_metrics,
    ) = _load_science(science_dir)

    source, labels = _load_public_train(request)
    tokenizer_spec = request["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_spec["id"], revision=tokenizer_spec["revision"], trust_remote_code=False
    )
    parsers = _parser_map()
    download = request["download_contract"]
    max_single = int(download["max_single_shard_bytes"])
    max_batch = int(download["max_batch_bytes"])
    batch_size = int(download["shards_per_download"])

    prediction_rows: list[dict[str, Any]] = []
    diagnostic_totals = {
        "samples": 0,
        "parser_errors": 0,
        "regex_fallbacks": 0,
        "multilingual_detector_available": 0,
        "flake8_samples": 0,
        "generic_linter_samples": 0,
        "target_matches": 0,
        "token_occurrences": 0,
        "union_tokens": 0,
        "download_bytes": 0,
        "download_batches": 0,
    }
    seen_shards: list[int] = []

    for cache in request["cache_sources"]:
        shard_start = int(cache["shard_start"])
        shard_stop = int(cache["shard_stop"])
        for start in range(shard_start, shard_stop, batch_size):
            indices = list(range(start, min(start + batch_size, shard_stop)))
            with tempfile.TemporaryDirectory(prefix="sersem-cache-batch-") as tmp:
                frame, batch_bytes, _ = _download_batch(
                    cli=cli,
                    kernel=cache["target"],
                    shard_indices=indices,
                    root=Path(tmp),
                    max_single_bytes=max_single,
                    max_batch_bytes=max_batch,
                )
                diagnostic_totals["download_bytes"] += int(batch_bytes)
                diagnostic_totals["download_batches"] += 1
                seen_shards.extend(indices)
                frame["sample_id"] = frame.sample_id.astype(str)
                for sample_id, group in frame.groupby("sample_id", sort=False):
                    if sample_id not in source.index:
                        raise RuntimeError("cache_sample_absent_from_canonical_public_train")
                    row = source.loc[sample_id]
                    language = str(row.language)
                    group_languages = set(group.language.astype(str))
                    if group_languages != {language}:
                        raise RuntimeError("cache_language_mismatch")
                    encoded = tokenizer(str(row.content), add_special_tokens=False, return_offsets_mapping=True)
                    full_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
                    char_weights, diagnostic = build_sersem_char_weights(
                        str(row.content), language, parser=parsers[language]
                    )
                    token_weights = token_weights_from_char_weights(
                        str(row.content), encoded["offset_mapping"], char_weights
                    )
                    score = score_cached_windows(group, full_ids, token_weights)
                    prediction_rows.append({
                        "sample_id": sample_id,
                        "language": language,
                        **{name: float(getattr(score, name)) for name in EXPECTED_SCORE_NAMES},
                    })
                    diagnostic_totals["samples"] += 1
                    diagnostic_totals["parser_errors"] += int(diagnostic.parser_has_error)
                    diagnostic_totals["regex_fallbacks"] += int(diagnostic.used_regex_fallback)
                    diagnostic_totals["multilingual_detector_available"] += int(
                        diagnostic.multilingual_detector_available
                    )
                    diagnostic_totals["flake8_samples"] += int(diagnostic.linter_backend == "flake8")
                    diagnostic_totals["generic_linter_samples"] += int(
                        diagnostic.linter_backend == "generic_heuristic"
                    )
                    diagnostic_totals["target_matches"] += int(diagnostic.target_matches)
                    diagnostic_totals["token_occurrences"] += int(score.scored_token_occurrences)
                    diagnostic_totals["union_tokens"] += int(score.union_token_count)
            # TemporaryDirectory guarantees raw shard deletion before the next API call.

    if seen_shards != list(range(40)):
        raise RuntimeError("cache_shard_coverage_failed")
    prediction = pd.DataFrame(prediction_rows)
    scientific = request["scientific_contract"]
    expected_samples = int(scientific["expected_samples"])
    if len(prediction) != expected_samples or prediction.sample_id.duplicated().any():
        raise RuntimeError("prediction_row_contract_failed")
    if prediction.groupby("language").size().to_dict() != {
        language: int(scientific["expected_samples_per_language"])
        for language in request["public_dataset"]["languages"]
    }:
        raise RuntimeError("prediction_language_balance_failed")
    if not np.isfinite(prediction[list(EXPECTED_SCORE_NAMES)].to_numpy(float)).all():
        raise RuntimeError("prediction_nonfinite")

    # Labels are joined only after every predictor score has been materialized.
    evaluation = prediction.merge(labels, on=["sample_id", "language"], how="left", validate="one_to_one")
    if evaluation.label.isna().any():
        raise RuntimeError("label_join_failed")
    evaluation["label"] = evaluation.label.astype(int)
    balance = evaluation.groupby(["language", "label"]).size()
    if not balance.eq(int(scientific["expected_samples_per_language_label"])).all():
        raise RuntimeError("evaluation_label_balance_failed")

    metrics = {
        name: low_fpr_metrics(evaluation.label, evaluation[name])
        for name in EXPECTED_SCORE_NAMES
    }
    per_language = {
        language: {
            name: low_fpr_metrics(group.label, group[name])
            for name in EXPECTED_SCORE_NAMES
        }
        for language, group in evaluation.groupby("language", sort=True)
    }
    bootstrap = {
        name: stratified_bootstrap_metrics(
            evaluation[["language", "label"]],
            evaluation[name],
            replicates=int(scientific["bootstrap_replicates"]),
            seed=int(scientific["bootstrap_seed"]),
        )
        for name in EXPECTED_SCORE_NAMES
    }
    score_map = {name: evaluation[name].to_numpy(float) for name in EXPECTED_SCORE_NAMES}
    jaccard, unique, sets = overlap_tables(
        evaluation.sample_id,
        evaluation.label,
        score_map,
        target_fpr=float(scientific["target_fpr"]),
    )
    overlap = {
        "conservative_tp_at_1pct_fpr": {name: int(len(values)) for name, values in sets.items()},
        "unique_tp_at_1pct_fpr": {
            str(row.attack): int(row.unique_true_positives)
            for row in unique.itertuples(index=False)
        },
        "jaccard": {
            f"{row.left}__{row.right}": float(row.jaccard)
            for row in jaccard.itertuples(index=False)
        },
    }
    # The result contains aggregates only. No sample_id, prediction, source text,
    # raw cache row, threshold, or row-order artifact is persisted.
    result = {
        "schema_version": 1,
        "program_id": "P1-03",
        "science_commit": request["science_commit"],
        "request_id": request["request_id"],
        "protocol": "historical-3x768-cache-conditioned-SERSEM-output",
        "full_author_8192_reproduction": False,
        "new_model_forward": False,
        "samples": int(len(evaluation)),
        "metrics": metrics,
        "per_language": per_language,
        "bootstrap": bootstrap,
        "overlap": overlap,
        "diagnostics": diagnostic_totals,
    }
    return result


def _self_test() -> None:
    request = {
        "schema_version": 1,
        "operation": "kernel_current_output_aggregate",
        "side_effects": [],
        "automatic_compute_retries": 0,
        "protected_job_private_repository_access": False,
        "kaggle_compute": False,
        "competition_submission": False,
        "scientific_contract": {
            "comparison": list(EXPECTED_SCORE_NAMES),
            "predictor_sealed_before_label_join": True,
            "sample_id_used_only_as_join_key": True,
            "public_lb_feedback_used": False,
            "codeparrot_reused": False,
            "new_model_forward": False,
        },
        "cache_sources": [
            {"target": "renta0426/starter-plus-10k-v2", "expected_current_version": 1, "required_terminal_status": "ERROR", "shard_start": 0, "shard_stop": 24},
            {"target": "renta0426/starter-plus-10k-v2-continuation", "expected_current_version": 1, "required_terminal_status": "COMPLETE", "shard_start": 24, "shard_stop": 40},
        ],
        "download_contract": {
            "file_family": "token_statistics.partNNN.parquet",
            "expected_total_shards": 40,
            "shards_per_download": 2,
            "persist_raw_cache": False,
            "persist_sample_ids": False,
            "persist_predictions": False,
            "aggregate_only": True,
        },
    }
    _validate_request(request)
    assert SHARD_NAME.fullmatch("token_statistics.part039.parquet")
    assert not SHARD_NAME.fullmatch("sample_features.part039.parquet")
    assert not KERNEL_REF.fullmatch("owner/slug/extra")
    assert list(range(0, 24, 2))[-1] == 22
    assert list(range(24, 40, 2))[-1] == 38
    print("SERSEM_CACHE_READOUT_SELF_TEST PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--kaggle-cli", type=Path)
    parser.add_argument("--science-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not all((args.request, args.kaggle_cli, args.science_dir, args.output_json)):
        raise SystemExit("request, kaggle-cli, science-dir and output-json are required")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    _validate_request(request)
    result = _score_all(request, args.kaggle_cli, args.science_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.output_json.read_bytes()).hexdigest()
    print(
        "SERSEM_CACHE_READOUT PASS "
        f"samples={result['samples']} batches={result['diagnostics']['download_batches']} "
        f"aggregate_sha256={digest}"
    )


if __name__ == "__main__":
    main()
