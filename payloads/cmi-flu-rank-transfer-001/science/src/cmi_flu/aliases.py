"""Canonical naming helpers for heterogeneous assay tables."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .contracts import DataContractError

_WHITESPACE = re.compile(r"\s+")

DEFAULT_CYTOKINE_ALIASES: dict[str, str] = {
    "IP10": "CXCL10",
    "IP-10": "CXCL10",
    "CXCL10": "CXCL10",
    "IL8": "CXCL8",
    "IL-8": "CXCL8",
    "CXCL8": "CXCL8",
    "VEGF": "VEGFA",
    "VEGF-A": "VEGFA",
    "VEGFA": "VEGFA",
    "IL1RA": "IL1RN",
    "IL-1RA": "IL1RN",
    "IL1RN": "IL1RN",
    "IL18": "IL18",
    "IL-18": "IL18",
}

DEFAULT_AIM_ALIASES: dict[str, str] = {
    "CON": "Conserved",
    "CONSERVED": "Conserved",
    "CONSERVED POOL": "Conserved",
}

DEFAULT_FLOW_ALIASES: dict[str, str] = {
    "CLASSICAL MONOCYTES": "Classical_monocytes",
    "CLASSICAL_MONOCYTE": "Classical_monocytes",
    "CLASSICAL_MONOCYTES": "Classical_monocytes",
    "ANTIBODY-SECRETING CELLS (ASC)": "Antibody-secreting_cells_(ASC)",
    "ANTIBODY SECRETING CELLS (ASC)": "Antibody-secreting_cells_(ASC)",
    "ANTIBODY-SECRETING_CELLS_(ASC)": "Antibody-secreting_cells_(ASC)",
    "ASC": "Antibody-secreting_cells_(ASC)",
}


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return _WHITESPACE.sub(" ", str(value).strip())


def _alias_key(value: object) -> str:
    return _text(value).upper()


@dataclass(frozen=True)
class AliasRegistry:
    """Immutable alias registry with explicit normalization boundaries."""

    mapping: Mapping[str, str]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> "AliasRegistry":
        normalized = {_alias_key(key): _text(value) for key, value in mapping.items()}
        if "" in normalized:
            raise DataContractError("alias mappings cannot contain an empty source key")
        return cls(normalized)

    def canonicalize(self, value: object) -> str:
        raw = _text(value)
        if not raw:
            return raw
        return self.mapping.get(_alias_key(raw), raw)

    def canonicalize_series(self, series: pd.Series) -> pd.Series:
        return series.map(self.canonicalize).astype("string")

    def merged(self, extra: Mapping[str, str]) -> "AliasRegistry":
        combined = dict(self.mapping)
        combined.update({_alias_key(key): _text(value) for key, value in extra.items()})
        return AliasRegistry(combined)


CYTOKINE_ALIASES = AliasRegistry.from_mapping(DEFAULT_CYTOKINE_ALIASES)
AIM_ALIASES = AliasRegistry.from_mapping(DEFAULT_AIM_ALIASES)
FLOW_ALIASES = AliasRegistry.from_mapping(DEFAULT_FLOW_ALIASES)


def canonicalize_cytokine(value: object) -> str:
    return CYTOKINE_ALIASES.canonicalize(value)


def canonicalize_aim_stimulation(value: object) -> str:
    return AIM_ALIASES.canonicalize(value)


def canonicalize_flow_population(value: object) -> str:
    raw = _text(value)
    if not raw:
        return raw
    normalized = re.sub(r"[\s/]+", "_", raw)
    normalized = re.sub(r"_+", "_", normalized)
    return FLOW_ALIASES.canonicalize(normalized)


def canonicalize_timepoint(value: object) -> str:
    """Normalize timepoint labels while preserving non-numeric organizer labels."""

    raw = _text(value)
    if not raw:
        return raw
    compact = raw.casefold().replace("_", "-").replace(" ", "")
    if compact in {"pre-vacc", "prevacc", "pre-vax", "prevax", "baseline"}:
        return "Pre-vacc"
    if compact.startswith("day"):
        compact = compact[3:]
    if compact.startswith("d") and compact[1:].lstrip("-").replace(".", "", 1).isdigit():
        compact = compact[1:]
    try:
        numeric = float(compact)
    except ValueError:
        return raw
    if numeric.is_integer():
        return str(int(numeric))
    return format(numeric, "g")


def canonicalize_strain(value: object) -> str:
    """Build a matching key while retaining cell-culture substrate information."""

    raw = _text(value)
    if not raw:
        return raw
    raw = re.sub(r"(?i)_cell$", "_MDCK", raw)
    raw = re.sub(r"(?i)\s+cell$", "_MDCK", raw)
    return raw


def load_alias_csv(
    path: str | Path,
    *,
    source_column: str | None = None,
    target_column: str | None = None,
) -> AliasRegistry:
    """Load organizer mapping CSVs while preserving rows with a trailing empty field.

    The delivered ``hai_map.csv`` declares two header fields but contains a third,
    empty field on every data row. Generic CSV parsers may reject or silently skip
    those rows. This reader removes only fields beyond the header that are proven
    empty and rejects non-empty overflow rather than discarding data.
    """

    csv_path = Path(path)
    try:
        stream = csv_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise DataContractError(f"cannot open alias file: {csv_path}") from error

    with stream:
        reader = csv.reader(stream)
        try:
            raw_header = next(reader)
        except StopIteration as error:
            raise DataContractError(f"alias file is empty: {csv_path}") from error

        header = [_text(value) for value in raw_header]
        nonempty_header = [value for value in header if value]
        if len(nonempty_header) < 2:
            raise DataContractError(
                f"alias file must have at least two named columns: {csv_path}"
            )
        source = source_column or nonempty_header[0]
        target = target_column or nonempty_header[1]
        try:
            source_index = header.index(source)
            target_index = header.index(target)
        except ValueError as error:
            raise DataContractError(
                f"alias columns not found in {csv_path}: source={source!r}, target={target!r}"
            ) from error

        minimum_width = max(source_index, target_index) + 1
        pairs: dict[str, str] = {}
        for row_number, raw_row in enumerate(reader, start=2):
            row = list(raw_row)
            if len(row) > len(header):
                overflow = row[len(header) :]
                if any(_text(value) for value in overflow):
                    raise DataContractError(
                        f"alias file {csv_path} has non-empty overflow fields on row {row_number}"
                    )
                row = row[: len(header)]
            if len(row) < minimum_width:
                row.extend([""] * (minimum_width - len(row)))

            source_text = _text(row[source_index])
            target_text = _text(row[target_index])
            if not source_text and not target_text:
                continue
            if not source_text or not target_text:
                raise DataContractError(
                    f"alias file {csv_path} has a one-sided mapping on row {row_number}"
                )
            source_key = _alias_key(source_text)
            existing = pairs.get(source_key)
            if existing is not None and existing != target_text:
                raise DataContractError(
                    f"alias file {csv_path} maps {source_text!r} to conflicting targets"
                )
            pairs[source_key] = target_text

    if not pairs:
        raise DataContractError(f"alias file contains no complete mappings: {csv_path}")
    return AliasRegistry(pairs)


def canonicalize_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    registry: AliasRegistry,
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            raise DataContractError(f"cannot canonicalize missing column: {column}")
        result[column] = registry.canonicalize_series(result[column])
    return result
