"""Strategy-v3 batch-1 V2 audit repair for data-limited Task1.2.

This module leaves the frozen V3-00/V3-03 contracts unchanged.  It repairs one
V3-01 implementation defect observed on the real Competition input: the generic
``TaskDataset.validate`` path raises when Task1.2 has an empty strict baseline
partition, although V3-01 is an audit whose contract is to record zero/data-
limited support rather than abort the whole seven-task ledger.

No model is fit here.  The reference expectations are never changed to force a
PASS, and unknown/absent measurement metadata remain unresolved/incompatible.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd

from .contracts import DataContractError
from .features.flow import build_flow_baseline_features
from .features.metadata import build_participant_context
from .targets import build_task_12_target
from . import strategy_v3_batch1 as _base

SCHEMA_VERSION = 1
EXPERIMENT = "strategy_v3_batch1_v2_task12_data_limited_audit_repair"
PARENT_EXPERIMENT = _base.EXPERIMENT
OBSERVED_RUNTIME_FAILURE_CODE = "ebb7c658334065658611"
OBSERVED_RUNTIME_FAILURE_STAGE = "run_no_fit_audit"
OBSERVED_RUNTIME_FAILURE_TYPE = "DataContractError"
OBSERVED_RUNTIME_FAILURE_MESSAGE = "Task1.2 dataset has an empty partition"
EXPECTED_TASK12_TARGET_BY_STUDY = {"SDY296": 36, "SDY301": 40, "SDY416": 5}

_RECOVERY_OBSERVATION: dict[str, Any] = {}


def _flow_feature_ids(frame: pd.DataFrame, *, mode: str) -> tuple[pd.DataFrame, int]:
    features = build_flow_baseline_features(frame, mode=mode)
    ids = features[["participant_id"]].drop_duplicates().copy()
    return ids, int(len(features))


def _recover_task12_audit_dataset(
    public_flow: pd.DataFrame,
    challenge_flow: pd.DataFrame,
    participants: pd.DataFrame,
    investigations: pd.DataFrame,
    *,
    mode: str,
) -> SimpleNamespace:
    """Reconstruct only the row support needed by the no-fit audit.

    This deliberately does not create a model-ready feature matrix.  The generic
    dataset builder's empty-partition rejection is appropriate for fitting but is
    not appropriate for a support audit.  We retain the same canonical strict
    baseline feature selection and target builder, then count their participant
    join exactly, including zero.
    """
    public_ids, public_baseline_rows = _flow_feature_ids(public_flow, mode=mode)
    challenge_ids, challenge_baseline_rows = _flow_feature_ids(challenge_flow, mode=mode)
    context = build_participant_context(participants, investigations)[
        ["participant_id", "subject_group", "study_group"]
    ].copy()
    target = build_task_12_target(public_flow)[["participant_id", "target"]].copy()

    train = (
        public_ids.merge(context, on="participant_id", how="left", validate="one_to_one")
        .merge(target, on="participant_id", how="inner", validate="one_to_one")
    )
    challenge = challenge_ids.merge(
        context, on="participant_id", how="left", validate="one_to_one"
    )
    return SimpleNamespace(
        task="Task1.2",
        train=train,
        challenge=challenge,
        target_column="target",
        metadata={
            "audit_only": True,
            "flow_mode": mode,
            "public_strict_baseline_rows": public_baseline_rows,
            "challenge_strict_baseline_rows": challenge_baseline_rows,
            "strict_baseline_join_rows": int(len(train)),
            "challenge_baseline_rows": int(len(challenge)),
        },
    )


def _audit_safe_build_task_12_dataset(*args: Any, **kwargs: Any) -> Any:
    global _RECOVERY_OBSERVATION
    try:
        return _ORIGINAL_BUILD_TASK_12_DATASET(*args, **kwargs)
    except DataContractError as error:
        if str(error) != OBSERVED_RUNTIME_FAILURE_MESSAGE:
            raise
        dataset = _recover_task12_audit_dataset(*args, **kwargs)
        _RECOVERY_OBSERVATION = {
            "activated": True,
            "original_error": str(error),
            **dict(dataset.metadata),
        }
        return dataset


def _audit_safe_flow_signature_rows(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Keep an empty measurement slice typed instead of producing a KeyError.

    Challenge data are baseline-only, so a requested D1 target measurement can
    legitimately have zero rows.  That absence is unresolved support, not proof
    of compatibility and not a reason to crash the no-fit ledger.
    """
    result = _ORIGINAL_FLOW_SIGNATURE_ROWS(*args, **kwargs)
    columns = [
        "participant_id",
        "study_accession",
        "signature",
        "metadata_complete",
        "finite_nonnegative",
    ]
    if result.empty:
        return result.reindex(columns=columns)
    return result


_ORIGINAL_BUILD_TASK_12_DATASET = _base.build_task_12_dataset
_ORIGINAL_FLOW_SIGNATURE_ROWS = _base._flow_signature_rows


def run_v3_01_audit(data_dir: str | Any) -> dict[str, Any]:
    """Run the unchanged V3-01 audit with data-limited Task1.2 semantics."""
    global _RECOVERY_OBSERVATION
    _RECOVERY_OBSERVATION = {"activated": False}
    previous_builder = _base.build_task_12_dataset
    previous_signature = _base._flow_signature_rows
    _base.build_task_12_dataset = _audit_safe_build_task_12_dataset
    _base._flow_signature_rows = _audit_safe_flow_signature_rows
    try:
        result = _base.run_v3_01_audit(data_dir)
    finally:
        _base.build_task_12_dataset = previous_builder
        _base._flow_signature_rows = previous_signature

    result = dict(result)
    source_alignment = dict(result.get("source_alignment_audit") or {})
    ledger = (result.get("teacher_ledger") or {}).get("Task1.2") or {}
    observed_by_study = dict(ledger.get("target_by_study") or {})
    baseline_join_rows = int(ledger.get("baseline_join_rows", 0))
    source_alignment["Task1.2_strict_baseline_join"] = {
        "repair_version": EXPERIMENT,
        "observed_parent_failure_code": OBSERVED_RUNTIME_FAILURE_CODE,
        "generic_model_dataset_empty_partition_is_not_audit_failure": True,
        "target_participant_years": int(ledger.get("target_bearing_participant_years", 0)),
        "target_by_study": observed_by_study,
        "target_counts_match_reference": observed_by_study == EXPECTED_TASK12_TARGET_BY_STUDY,
        "strict_baseline_join_rows": baseline_join_rows,
        "strict_support_status": "data_limited" if baseline_join_rows == 0 else "observed",
        "raw_scale_compatible": ledger.get("raw_scale_compatible"),
        "recovery_observation": dict(_RECOVERY_OBSERVATION),
        "zero_support_recorded_not_replaced": True,
        "reference_expectations_mutated": False,
        "model_fit_performed": False,
    }
    source_alignment["Task1.2_measurement_absence_policy"] = {
        "empty_challenge_D1_measurement_slice": "unresolved_not_compatible",
        "unknown_metadata": "incompatible_not_matched",
    }
    result["source_alignment_audit"] = source_alignment
    if int(result.get("model_fit_count", -1)) != 0:
        raise DataContractError("V3 batch1 V2 repair unexpectedly changed model-fit count")
    if result.get("competition_submission_attempted") is not False:
        raise DataContractError("V3 batch1 V2 repair unexpectedly changed submission boundary")
    return result


generate_singleton_diagnostics = _base.generate_singleton_diagnostics
write_aggregate_jsons = _base.write_aggregate_jsons
validate_e12c_source_csv = _base.validate_e12c_source_csv
singleton_score_interval = _base.singleton_score_interval
summarize_singleton_public_scores = _base.summarize_singleton_public_scores

__all__ = [
    "EXPERIMENT",
    "PARENT_EXPERIMENT",
    "OBSERVED_RUNTIME_FAILURE_CODE",
    "OBSERVED_RUNTIME_FAILURE_MESSAGE",
    "run_v3_01_audit",
    "generate_singleton_diagnostics",
    "write_aggregate_jsons",
]
