"""Compatibility correction for strategy E01 evaluation identity keys.

E01 conditions are uniquely identified by the study-local subject identity, not
by a globally unique participant_id assumption.  The original E01 metric helper
was stricter than the repository/data contract and could reject otherwise valid
frames when participant labels repeat across studies.  This module changes only
that validation layer and reuses the frozen E01 numerical implementation.
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .contracts import DataContractError, require_columns
from . import strategy_e01 as _base


_ORIGINAL_METRIC_SUMMARY = _base._metric_summary


def _metric_summary(frame: pd.DataFrame, *, prediction_scale: str) -> Mapping[str, Any]:
    """Evaluate one-row-per-study-subject conditions without global-ID assumptions."""

    require_columns(
        frame,
        ["participant_id", "subject_group", "study_group", "target", "prediction"],
        table_name="E01 condition frame",
    )
    if frame[["study_group", "subject_group"]].duplicated().any():
        raise DataContractError(
            "E01 condition frame must be one row per study_group/subject_group"
        )

    # The frozen v1 helper only uses participant_id for this now-overstrict
    # uniqueness guard.  Replace it with a deterministic internal row identity
    # before delegating; no participant identifier is emitted in E01 outputs.
    working = frame.copy()
    working["participant_id"] = [f"__e01_row_{i}" for i in range(len(working))]
    return _ORIGINAL_METRIC_SUMMARY(working, prediction_scale=prediction_scale)


# All E01 functions resolve _metric_summary from the base module globals at call
# time, so installing this correction updates compact/HAI controls consistently.
_base._metric_summary = _metric_summary

RANDOM_SEED = _base.RANDOM_SEED
SUBSET_N = _base.SUBSET_N
SUBSET_REPETITIONS = _base.SUBSET_REPETITIONS
_paired_comparison = _base._paired_comparison
_sensitivity_28 = _base._sensitivity_28
_shuffle_hai_train = _base._shuffle_hai_train
run_strategy_e01 = _base.run_strategy_e01

__all__ = [
    "RANDOM_SEED",
    "SUBSET_N",
    "SUBSET_REPETITIONS",
    "_metric_summary",
    "_paired_comparison",
    "_sensitivity_28",
    "_shuffle_hai_train",
    "run_strategy_e01",
]
