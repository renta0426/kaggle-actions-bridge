"""Narrow persistence-order repair for the Gold controlled Kaggle runtime.

Gold v1 wrote JSON objects with ``sort_keys=True`` and later treated JSON object
key order as a schema invariant.  JSON key order is not semantic, so v2 keeps
the exact v1 scientific runtime and canonicalizes only the two persisted JSONL
schemas at readback.  Missing, extra, or duplicate columns still fail closed.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd

from . import shadow_gold_kaggle as v1

SHADOW_GOLD_JSONL_REPAIR_VERSION = "shadow-gold-jsonl-column-order-v2"


def canonicalize_exact_schema(
    frame: pd.DataFrame,
    expected_columns: Sequence[str],
    *,
    artifact_name: str,
) -> pd.DataFrame:
    expected = list(expected_columns)
    actual = list(frame.columns)
    if len(actual) != len(set(actual)):
        raise v1.ShadowGoldKaggleError(f"{artifact_name} contains duplicate columns")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise v1.ShadowGoldKaggleError(
            f"{artifact_name} schema changed: missing={missing} extra={extra}"
        )
    return frame.loc[:, expected].copy()


def _schema_for_read(value: Any) -> tuple[list[str], str] | None:
    try:
        name = Path(value).name
    except (TypeError, ValueError):
        return None
    if name == "gold_features.jsonl":
        return list(v1._FEATURE_COLUMNS), "Gold child persisted feature"
    if name == "evaluation_labels.jsonl":
        return list(v1._LABEL_COLUMNS), "Gold parent label"
    return None


@contextmanager
def repaired_jsonl_readback() -> Iterator[None]:
    """Temporarily canonicalize exact Gold JSONL schemas during parent finalize."""

    original = v1.pd.read_json

    def repaired_read_json(*args: Any, **kwargs: Any) -> pd.DataFrame:
        frame = original(*args, **kwargs)
        source = args[0] if args else kwargs.get("path_or_buf")
        schema = _schema_for_read(source)
        if schema is None:
            return frame
        expected, artifact_name = schema
        return canonicalize_exact_schema(frame, expected, artifact_name=artifact_name)

    v1.pd.read_json = repaired_read_json
    try:
        yield
    finally:
        v1.pd.read_json = original


def finalize_gold_results(*args: Any, **kwargs: Any):
    """Run the unchanged v1 finalizer with order-insensitive exact-schema reads."""

    with repaired_jsonl_readback():
        return v1.finalize_gold_results(*args, **kwargs)


def run_gold_benchmark(*args: Any, **kwargs: Any):
    """Run the unchanged v1 benchmark with the readback repair scoped to parent."""

    with repaired_jsonl_readback():
        attack, metrics, predictions, manifest = v1.run_gold_benchmark(*args, **kwargs)
    manifest = dict(manifest)
    manifest["persistence_readback_repair"] = SHADOW_GOLD_JSONL_REPAIR_VERSION
    manifest["scientific_protocol_changed"] = False
    return attack, metrics, predictions, manifest


__all__ = [
    "SHADOW_GOLD_JSONL_REPAIR_VERSION",
    "canonicalize_exact_schema",
    "finalize_gold_results",
    "repaired_jsonl_readback",
    "run_gold_benchmark",
]
