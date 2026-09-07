"""E05 compatibility entrypoint with both real-data contract corrections.

v2 corrected donor-state rank identity for repeated biological subjects.
v3 additionally installs the exact organizer ``strain_sequences.csv`` schema
adapter before delegating to v2.  Scientific model family, alpha, interactions,
controls, split definitions, weighting and routing criteria are unchanged.
"""
from __future__ import annotations

from typing import Any, Mapping

from . import hai_transfer_v2 as _hai_transfer_v2  # noqa: F401 - installs schema adapter
from . import strategy_e05_v2 as _base


def run_strategy_e05(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    result = dict(_base.run_strategy_e05(*args, **kwargs))
    feature_contract = dict(result.get("feature_contract") or {})
    feature_contract.update(
        {
            "sequence_reference_schema": "organizer_native_or_canonical_explicit_adapter",
            "organizer_sequence_reference_columns": [
                "Virus",
                "Sequence",
                "Status_of_sequence",
            ],
        }
    )
    result["feature_contract"] = feature_contract
    return result


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
    "_e05_design",
    "_hierarchical_sample_weights",
    "_simultaneous_subject_strain_splits",
    "run_strategy_e05",
]
