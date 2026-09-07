"""Compatibility correction for Strategy E02 repeat-baseline pairing.

The science conditions are unchanged from :mod:`cmi_flu.strategy_e02`.  This
module only replaces the pandas-pivot-dependent pairing of CXCL10 -14/day0
rows with an explicit one-to-one merge after the same canonicalization and
aggregation.  It is intentionally small so the correction is independently
hashable/auditable in the Kaggle relay.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import strategy_e02 as _base
from .aliases import canonicalize_cytokine, canonicalize_timepoint
from .contracts import require_columns
from .strategy_e02 import *  # noqa: F401,F403


def _repeat_feature_table(cytokine: pd.DataFrame) -> pd.DataFrame:
    """Build paired -14/day0 CXCL10 summaries without pandas pivot semantics."""

    require_columns(
        cytokine,
        ["participant_id", "study_accession", "timepoint", "analyte", "value"],
        table_name="E02 repeat cytokine",
    )
    work = cytokine[
        ["participant_id", "study_accession", "timepoint", "analyte", "value"]
    ].copy()
    work["analyte"] = work["analyte"].map(canonicalize_cytokine)
    work["timepoint"] = work["timepoint"].map(canonicalize_timepoint)
    work = work.loc[
        work["analyte"].eq("CXCL10") & work["timepoint"].isin(REPEAT_TIMEPOINTS)
    ].copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work.loc[work["value"] < 0.0, "value"] = np.nan
    work = work.dropna(subset=["value"])
    if work.empty:
        return pd.DataFrame(
            columns=["participant_id", "study_group", *REPEAT_COLUMNS]
        )

    grouped = (
        work.groupby(
            ["participant_id", "study_accession", "timepoint"],
            dropna=False,
            observed=True,
        )["value"]
        .mean()
        .reset_index()
    )
    left = grouped.loc[
        grouped["timepoint"].eq(REPEAT_TIMEPOINTS[0]),
        ["participant_id", "study_accession", "value"],
    ].rename(columns={"value": "value_left"})
    right = grouped.loc[
        grouped["timepoint"].eq(REPEAT_TIMEPOINTS[1]),
        ["participant_id", "study_accession", "value"],
    ].rename(columns={"value": "value_right"})
    paired = left.merge(
        right,
        on=["participant_id", "study_accession"],
        how="inner",
        validate="one_to_one",
    )
    if paired.empty:
        return pd.DataFrame(
            columns=["participant_id", "study_group", *REPEAT_COLUMNS]
        )

    left_log = np.log1p(paired["value_left"].to_numpy(dtype=float))
    right_log = np.log1p(paired["value_right"].to_numpy(dtype=float))
    paired[REPEAT_COLUMNS[0]] = right_log - left_log
    paired[REPEAT_COLUMNS[1]] = np.var(
        np.column_stack([left_log, right_log]), axis=1, ddof=0
    )
    return paired[
        ["participant_id", "study_accession", *REPEAT_COLUMNS]
    ].rename(columns={"study_accession": "study_group"})


def run_strategy_e02(config: Any, inputs: Any) -> Mapping[str, Any]:
    """Run frozen E02 with only the repeat-pairing compatibility correction."""

    original = _base._repeat_feature_table
    _base._repeat_feature_table = _repeat_feature_table
    try:
        return _base.run_strategy_e02(config, inputs)
    finally:
        _base._repeat_feature_table = original
