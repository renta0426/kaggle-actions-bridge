"""Strategy-v3 V3-05 repair: canonicalize frozen study aliases in context only.

The first real V3-05 Kaggle run reached the Task1.3 scale-head science path but
failed before any fit because participant context retained the raw organizer
label ``2024_UGA`` while the measurement audit uses the already-frozen V3-01
canonical label ``2024UGA``.  V3-01 explicitly established this alias and its
hash; this module applies that same outcome-independent alias only to
``study_group`` returned by ``build_participant_context`` while the frozen V3-05
implementation runs.

No teacher, target, baseline value, feature, split, candidate formula, fit,
prediction, unit, parent/gate metadata, Public score, or submission behavior is
changed by this repair.
"""
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from . import strategy_v3_v05 as _base
from .strategy_v3_batch1 import canonical_study, study_alias_hash

EXPERIMENT = _base.EXPERIMENT
SCHEMA_VERSION = _base.SCHEMA_VERSION
NEW_CONDITIONS = _base.NEW_CONDITIONS
MAX_FITS = _base.MAX_FITS

REPAIR_VERSION = "strategy_v3_v05_v2_context_study_alias_repair"
OBSERVED_FAILURE_STAGE = "run_task13_scale_heads"
OBSERVED_FAILURE_TYPE = "DataContractError"
OBSERVED_FAILURE_MESSAGE = "V3-05 source cohort is not the frozen single 2024UGA domain"
OBSERVED_FAILURE_CODE = "88315152273aaa70245f"
EXPECTED_ALIAS_HASH = "6626de8ebc839dbdac44faa19e8c3280348325d2a214fe940984a9d209fe2985"
_ALLOWED_CHANGED_PAIRS = {("2024_UGA", "2024UGA")}
_ORIGINAL_BUILD_PARTICIPANT_CONTEXT = _base.build_participant_context


def observed_failure_code() -> str:
    raw = f"{OBSERVED_FAILURE_STAGE}:{OBSERVED_FAILURE_TYPE}:{OBSERVED_FAILURE_MESSAGE}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _canonical_context_builder(
    participants: pd.DataFrame,
    investigations: pd.DataFrame,
    *,
    include_geolocation: bool = False,
) -> pd.DataFrame:
    """Apply only the frozen V3-01 study alias to participant context identity."""
    context = _ORIGINAL_BUILD_PARTICIPANT_CONTEXT(
        participants,
        investigations,
        include_geolocation=include_geolocation,
    ).copy()
    if "study_group" not in context:
        raise _base.DataContractError("V3-05 v2 participant context lacks study_group")
    if study_alias_hash() != EXPECTED_ALIAS_HASH:
        raise _base.DataContractError("V3-05 v2 frozen study alias hash changed")

    raw = context["study_group"].copy()
    canonical = raw.copy()
    present = raw.notna()
    canonical.loc[present] = raw.loc[present].map(canonical_study)
    changed = present & raw.astype("string").ne(canonical.astype("string"))
    if changed.any():
        pairs = set(
            zip(
                raw.loc[changed].astype(str).tolist(),
                canonical.loc[changed].astype(str).tolist(),
            )
        )
        if not pairs.issubset(_ALLOWED_CHANGED_PAIRS):
            raise _base.DataContractError(
                "V3-05 v2 study alias change escaped the frozen 2024_UGA mapping"
            )
    context["study_group"] = canonical
    return context


def run_v3_05_task13(config: Any, inputs: Any):
    """Run frozen V3-05 with the V3-01 study-identity alias applied fail-closed."""
    if observed_failure_code() != OBSERVED_FAILURE_CODE:
        raise _base.DataContractError("V3-05 v2 observed failure fingerprint changed")
    previous = _base.build_participant_context
    _base.build_participant_context = _canonical_context_builder
    try:
        aggregate, oof_bank, challenge_bank = _base.run_v3_05_task13(config, inputs)
    finally:
        _base.build_participant_context = previous

    aggregate = dict(aggregate)
    aggregate["runtime_repair"] = {
        "repair_version": REPAIR_VERSION,
        "parent_failure_stage": OBSERVED_FAILURE_STAGE,
        "parent_failure_type": OBSERVED_FAILURE_TYPE,
        "parent_failure_code": OBSERVED_FAILURE_CODE,
        "parent_failure_message": OBSERVED_FAILURE_MESSAGE,
        "study_alias_sha256": EXPECTED_ALIAS_HASH,
        "allowed_changed_pairs": [list(pair) for pair in sorted(_ALLOWED_CHANGED_PAIRS)],
        "scope": "participant_context.study_group_identity_only",
        "teacher_changed": False,
        "target_changed": False,
        "measurement_values_changed": False,
        "features_changed": False,
        "splits_changed": False,
        "candidate_formulas_changed": False,
        "fit_budget_changed": False,
        "unit_conversion_performed": False,
        "public_leaderboard_used": False,
        "competition_submission_attempted": False,
    }
    return aggregate, oof_bank, challenge_bank


def write_v3_05_outputs(aggregate, oof_bank, challenge_bank, output_dir):
    return _base.write_v3_05_outputs(aggregate, oof_bank, challenge_bank, output_dir)
