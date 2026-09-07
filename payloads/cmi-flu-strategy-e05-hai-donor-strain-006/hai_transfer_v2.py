"""Compatibility adapter for the organizer-native HAI sequence reference schema.

The locked organizer file ``strain_sequences.csv`` has the exact header
``Virus,Sequence,Status_of_sequence``.  The original Phase-A helper was tested
only against an internal normalized fixture with
``virus_strain,sequence,sequence_status`` and therefore failed on the real
organizer file before any E05 model fit.

This module keeps the frozen Phase-A implementation intact and installs one
explicit schema adapter.  It accepts only the historical normalized schema or
the exact organizer-native schema; ambiguous/partial layouts fail closed.
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .contracts import DataContractError
from . import hai_transfer as _base

CANONICAL_SEQUENCE_COLUMNS = ("virus_strain", "sequence", "sequence_status")
ORGANIZER_SEQUENCE_COLUMNS = ("Virus", "Sequence", "Status_of_sequence")
ORGANIZER_TO_CANONICAL = dict(zip(ORGANIZER_SEQUENCE_COLUMNS, CANONICAL_SEQUENCE_COLUMNS, strict=True))


def normalize_sequence_reference_schema(reference: pd.DataFrame) -> pd.DataFrame:
    """Return the three canonical sequence columns under an explicit schema contract."""

    columns = set(map(str, reference.columns))
    canonical = set(CANONICAL_SEQUENCE_COLUMNS)
    organizer = set(ORGANIZER_SEQUENCE_COLUMNS)
    has_canonical = canonical.issubset(columns)
    has_organizer = organizer.issubset(columns)
    if has_canonical and has_organizer:
        raise DataContractError("strain sequence reference contains both canonical and organizer schemas")
    if has_canonical:
        return reference.loc[:, list(CANONICAL_SEQUENCE_COLUMNS)].copy()
    if has_organizer:
        return (
            reference.loc[:, list(ORGANIZER_SEQUENCE_COLUMNS)]
            .rename(columns=ORGANIZER_TO_CANONICAL)
            .copy()
        )
    raise DataContractError(
        "strain sequence reference schema unsupported: expected canonical "
        f"{list(CANONICAL_SEQUENCE_COLUMNS)} or organizer {list(ORGANIZER_SEQUENCE_COLUMNS)}"
    )


def build_sequence_lookup(reference: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    """Run the frozen lookup logic after explicit organizer-schema normalization."""

    return _ORIGINAL_BUILD_SEQUENCE_LOOKUP(normalize_sequence_reference_schema(reference))


_ORIGINAL_BUILD_SEQUENCE_LOOKUP = _base.build_sequence_lookup
_base.build_sequence_lookup = build_sequence_lookup

__all__ = [
    "CANONICAL_SEQUENCE_COLUMNS",
    "ORGANIZER_SEQUENCE_COLUMNS",
    "normalize_sequence_reference_schema",
    "build_sequence_lookup",
]
