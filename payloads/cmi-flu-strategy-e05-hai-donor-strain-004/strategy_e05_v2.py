"""Compatibility correction for strategy E05 donor-rank identity.

The original E05 implementation grouped donor-state ranks by ``subject_group``.
That is too strict for the historical CMI-Flu data because one biological
subject can map to multiple participant/vaccination units.  E05 predicts those
participant units, while ``subject_group`` is the leakage/weighting group.

This correction changes only the donor-state rank identity:

* donor Z features are constant and ranked per ``study_group/participant_id``;
* biological ``subject_group`` remains unchanged for purging and hierarchical
  sample weighting;
* the frozen E05 model family, alpha, interactions, controls, panels and split
  definitions are otherwise reused verbatim.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import DataContractError, require_columns
from .metrics import percentile_rank
from . import strategy_e05 as _base


def _donor_level_rank(frame: pd.DataFrame, value_column: str) -> np.ndarray:
    """Rank one participant/vaccination-unit X-only value within study.

    ``subject_group`` is intentionally not the identity here.  A biological
    subject may contribute multiple participant units in historical studies;
    those units can have different baseline immune states and are separate
    prediction targets, while later split/weight code still purges/equalizes the
    shared biological subject.
    """

    require_columns(
        frame,
        ["study_group", "participant_id", value_column],
        table_name=f"E05 donor-rank {value_column}",
    )
    working = frame[["study_group", "participant_id", value_column]].copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")

    records: list[pd.DataFrame] = []
    for (study, participant), group in working.groupby(
        ["study_group", "participant_id"],
        dropna=False,
        observed=True,
        sort=False,
    ):
        finite = group[value_column].dropna().to_numpy(dtype=float)
        if finite.size and not np.allclose(finite, finite[0], rtol=0.0, atol=1e-12):
            raise DataContractError(
                f"E05 donor feature {value_column} varies by strain for "
                f"study={study}/participant={participant}"
            )
        value = float(finite[0]) if finite.size else np.nan
        records.append(
            pd.DataFrame(
                {
                    "study_group": [str(study)],
                    "participant_id": [str(participant)],
                    value_column: [value],
                }
            )
        )

    donors = pd.concat(records, ignore_index=True)
    donors["__rank"] = np.nan
    for _, positions in donors.groupby(
        "study_group", dropna=False, observed=True, sort=False
    ).indices.items():
        pos = np.asarray(positions, dtype=int)
        values = donors.iloc[pos][value_column].to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            continue
        donors.loc[donors.index[pos[finite_mask]], "__rank"] = percentile_rank(
            values[finite_mask]
        )

    mapping = {
        (str(row["study_group"]), str(row["participant_id"])): row["__rank"]
        for row in donors.to_dict(orient="records")
    }
    return np.asarray(
        [
            mapping[(str(study), str(participant))]
            for study, participant in zip(
                frame["study_group"], frame["participant_id"], strict=True
            )
        ],
        dtype=float,
    )


# E05 resolves this helper from its module globals at call time.  Install the
# correction without changing the frozen v1 numerical implementation elsewhere.
_base._subject_level_rank = _donor_level_rank


def run_strategy_e05(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    result = dict(_base.run_strategy_e05(*args, **kwargs))
    feature_contract = dict(result.get("feature_contract") or {})
    feature_contract.update(
        {
            "donor_rank_unit": "participant_id_within_study",
            "subject_group_role": "leakage_purge_and_hierarchical_weighting_only",
        }
    )
    result["feature_contract"] = feature_contract
    return result


# Export the constants/helpers used by bridge tests and provenance checks.
EXPERIMENT = _base.EXPERIMENT
RIDGE_ALPHA = _base.RIDGE_ALPHA
MAX_INTERACTIONS = _base.MAX_INTERACTIONS
INTERACTION_COLUMNS = _base.INTERACTION_COLUMNS
MAIN_NUMERIC_COLUMNS = _base.MAIN_NUMERIC_COLUMNS
MAIN_CATEGORICAL_COLUMNS = _base.MAIN_CATEGORICAL_COLUMNS
SIMULTANEOUS_HOLDOUTS = _base.SIMULTANEOUS_HOLDOUTS
_e05_design = _base._e05_design
_hierarchical_sample_weights = _base._hierarchical_sample_weights
_simultaneous_subject_strain_splits = _base._simultaneous_subject_strain_splits

__all__ = [
    "EXPERIMENT",
    "RIDGE_ALPHA",
    "MAX_INTERACTIONS",
    "INTERACTION_COLUMNS",
    "MAIN_NUMERIC_COLUMNS",
    "MAIN_CATEGORICAL_COLUMNS",
    "SIMULTANEOUS_HOLDOUTS",
    "_donor_level_rank",
    "_e05_design",
    "_hierarchical_sample_weights",
    "_simultaneous_subject_strain_splits",
    "run_strategy_e05",
]
