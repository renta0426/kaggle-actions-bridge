#!/usr/bin/env python3
"""P1-03 SERSEM cache evaluator v4: strict Arrow nested-offset repair.

V3 proved that all 18,447 cached windows align exactly when the historical
Starter++ tokenizer runtime is replayed, then failed when PyArrow/Pandas nested
``target_offsets`` were passed directly to ``numpy.asarray(dtype=int64)``.
The failure hash is the exact SHA-256 of
``ValueError: setting an array element with a sequence.``

V4 changes only deserialization of that already-validated alignment column.  It
normalizes one ``(start, stop)`` pair at a time with strict shape/integer/bounds
checks, then delegates every weighting, scoring, metric, bootstrap, provenance,
and runtime check to the frozen v3 evaluator.  It does not retokenize under the
SERSEM runtime, perform fuzzy alignment, drop tokens/windows, or start a model
forward pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import poisoned_chalice_sersem_cache_eval_v3 as core


FAILURE_RUN = 34304780366
FAILURE_ERROR_SHA256 = "b539ea87148a46e0df439a8792530b5aa03dea0bed96330d00b560f555a4fef2"
FAILURE_MESSAGE = "setting an array element with a sequence."


def _scalar_int(value: Any) -> int:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool):
        raise RuntimeError("alignment_offset_boolean_forbidden")
    if isinstance(value, (list, tuple, dict)) or hasattr(value, "shape"):
        raise RuntimeError("alignment_offset_scalar_required")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("alignment_offset_integer_conversion_failure") from exc
    if integer < 0:
        raise RuntimeError("alignment_offset_negative")
    return integer


def normalize_target_offsets(raw: Any, *, text_length: int) -> list[tuple[int, int]]:
    """Normalize an Arrow/Pandas nested-list cell without bulk NumPy coercion."""
    values = raw.tolist() if hasattr(raw, "tolist") else raw
    if isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list):
        try:
            values = list(values)
        except TypeError as exc:
            raise RuntimeError("alignment_offset_outer_sequence_required") from exc

    pairs: list[tuple[int, int]] = []
    for raw_pair in values:
        pair = raw_pair.tolist() if hasattr(raw_pair, "tolist") else raw_pair
        if isinstance(pair, tuple):
            pair = list(pair)
        if not isinstance(pair, list) or len(pair) != 2:
            raise RuntimeError("alignment_offset_pair_shape_failure")
        start = _scalar_int(pair[0])
        stop = _scalar_int(pair[1])
        if stop < start:
            raise RuntimeError("alignment_offset_reversed")
        if stop > text_length:
            raise RuntimeError("alignment_offset_out_of_bounds")
        pairs.append((start, stop))
    return pairs


def score_sample_v4(token_rows, alignment_rows, text: str, caller_chars, helper_chars):
    np = core._np()
    alignments = {core._window_key(row): row for row in alignment_rows.itertuples(index=False)}
    token_items = list(token_rows.itertuples(index=False))
    token_items.sort(key=lambda row: (int(row.window_start), str(row.position)))
    if len(alignments) != len(token_items):
        raise RuntimeError("alignment_window_coverage_failure")

    baseline_windows: list[float] = []
    caller_windows: list[float] = []
    helper_windows: list[float] = []
    caller_union: dict[int, tuple[int, float, float]] = {}
    helper_union: dict[int, tuple[int, float, float]] = {}

    for row in token_items:
        alignment = alignments.get(core._window_key(row))
        if alignment is None:
            raise RuntimeError("alignment_window_missing")
        z = np.asarray(row.correct_z, dtype=np.float64)
        observed = np.asarray(row.target_token_ids, dtype=np.int64)
        offsets = normalize_target_offsets(alignment.target_offsets, text_length=len(text))
        if len(z) != len(observed) or len(z) != len(offsets):
            raise RuntimeError("alignment_target_length_failure")
        if int(alignment.target_token_count) != len(z):
            raise RuntimeError("alignment_manifest_target_count_failure")

        caller_weights = core.token_weights(text, offsets, caller_chars)
        helper_weights = core.token_weights(text, offsets, helper_chars)
        baseline_windows.append(float(core._sigmoid(z).mean()))
        caller_windows.append(core._weighted_score(z, caller_weights))
        helper_windows.append(core._weighted_score(z, helper_weights))

        start = int(row.window_start) + 1
        for local_index, (global_index, z_value, caller_weight, helper_weight) in enumerate(
            zip(range(start, start + len(z)), z, caller_weights, helper_weights),
            start=1,
        ):
            previous = caller_union.get(global_index)
            if previous is None or local_index > previous[0]:
                caller_union[global_index] = (local_index, float(z_value), float(caller_weight))
                helper_union[global_index] = (local_index, float(z_value), float(helper_weight))

    if not baseline_windows:
        raise RuntimeError("sample_without_cached_windows")
    return (
        float(np.mean(baseline_windows)),
        float(np.mean(caller_windows)),
        core._weighted_score(
            [value for _, value, _ in caller_union.values()],
            [weight for _, _, weight in caller_union.values()],
        ),
        float(np.mean(helper_windows)),
        core._weighted_score(
            [value for _, value, _ in helper_union.values()],
            [weight for _, _, weight in helper_union.values()],
        ),
    )


def evaluate_v4(*args, **kwargs):
    original = core._score_sample
    core._score_sample = score_sample_v4
    try:
        summary = core.evaluate(*args, **kwargs)
    finally:
        core._score_sample = original
    summary["schema_version"] = 2
    summary["protocol"] = "P1-03 SERSEM historical-cache reproduction v4"
    summary["offset_deserialization_repair"] = {
        "source_failure_run": FAILURE_RUN,
        "source_failure_error_sha256": FAILURE_ERROR_SHA256,
        "source_failure_message": FAILURE_MESSAGE,
        "repair": "strict pairwise Arrow/Pandas nested-offset normalization; no fuzzy alignment or token/window dropping",
    }
    output_json = args[4] if len(args) >= 5 else kwargs.get("output_json")
    if output_json is not None:
        Path(output_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _self_test() -> None:
    np = core._np()

    class ArrowLike:
        def tolist(self):
            return [np.array([1, 3], dtype=object), np.array([3, 3], dtype=object)]

    assert normalize_target_offsets(ArrowLike(), text_length=5) == [(1, 3), (3, 3)]
    assert normalize_target_offsets([[0, 1], (1, 5)], text_length=5) == [(0, 1), (1, 5)]
    try:
        normalize_target_offsets([[0, [1, 2]]], text_length=5)
    except RuntimeError as exc:
        assert str(exc) == "alignment_offset_scalar_required"
    else:
        raise AssertionError("nested scalar sequence must be rejected")
    assert hashlib.sha256(f"ValueError:{FAILURE_MESSAGE}".encode()).hexdigest() == FAILURE_ERROR_SHA256
    print("SERSEM_CACHE_EVAL_V4_SELF_TEST PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-parquet", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--alignment-parquet", type=Path)
    parser.add_argument("--alignment-manifest-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260909)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    required = (
        args.dataset_parquet,
        args.cache_dir,
        args.alignment_parquet,
        args.alignment_manifest_json,
        args.output_json,
    )
    if any(value is None for value in required):
        raise SystemExit("dataset/cache/alignment/output are required")
    try:
        summary = evaluate_v4(
            args.dataset_parquet,
            args.cache_dir,
            args.alignment_parquet,
            args.alignment_manifest_json,
            args.output_json,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    except Exception as exc:
        digest = hashlib.sha256(
            f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
        ).hexdigest()
        print(
            "SERSEM_CACHE_EVAL_V4_FAILURE "
            + json.dumps(
                {
                    "status": "failure",
                    "exception_type": type(exc).__name__,
                    "error_sha256": digest,
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        "SERSEM_CACHE_EVAL_V4_LOCAL_RESULT "
        + json.dumps(
            {
                "status": "success",
                "samples": summary["samples"],
                "metrics": summary["metrics"],
                "conservative_tp_at_1pct_fpr": summary["conservative_tp_at_1pct_fpr"],
                "diagnostics": summary["diagnostics"],
                "offset_deserialization_repair": summary["offset_deserialization_repair"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
