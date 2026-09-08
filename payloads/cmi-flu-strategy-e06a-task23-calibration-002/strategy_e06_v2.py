"""E06a compatibility for historical D365 studies without task-panel overlap.

The first real E06a run exposed a data-coverage edge case that the synthetic
fixture did not represent: ``build_hai_model_dataset(day=365)`` legitimately
contains studies whose measured D365 HAI strains have zero overlap with the
fixed 12-strain Challenge panel.  Such a study cannot define a Task2.3
panel-GM proxy and therefore must not be treated as an evaluable outer fold.

This adapter changes only evaluation eligibility.  Eligibility is determined
from study/strain availability (X/metadata) before any outcome is inspected:

* an outer validation study must contain at least one fixed-panel strain;
* the corresponding outer-training partition must contain at least two studies
  with fixed-panel strain overlap so the calibrator remains cross-study;
* zero-overlap studies remain available to the raw HAI model's training rows,
  but are not scored as if they supplied a Task2.3 proxy target.

The frozen raw predictors, nested subject-purged study split logic, calibrator
families, equal-study calibration weights, promotion thresholds, and Challenge
rank-preservation requirements are unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .aliases import canonicalize_strain
from .contracts import DataContractError, require_columns
from .cv import NamedSplit, purged_leave_one_study_out
from .evaluation import evaluate_hai_spec
from .models import ModelSpec, fit_final_model
from . import strategy_e06 as _base


_ORIGINAL_FIT_CALIBRATOR = _base._fit_calibrator


def _panel_overlap_mask(
    frame: pd.DataFrame,
    *,
    panel_strains: Sequence[str],
) -> np.ndarray:
    require_columns(
        frame,
        ["study_group", "virus_strain"],
        table_name="E06a panel-overlap eligibility",
    )
    panel = set(_base._canonical_panel(panel_strains))
    strain = frame["virus_strain"].map(canonicalize_strain)
    return strain.isin(panel).to_numpy(dtype=bool)


def _panel_overlap_study_count(
    frame: pd.DataFrame,
    *,
    panel_strains: Sequence[str],
) -> int:
    mask = _panel_overlap_mask(frame, panel_strains=panel_strains)
    if not bool(mask.any()):
        return 0
    return int(frame.loc[mask, "study_group"].astype(str).nunique())


def _fit_calibrator(
    frame: pd.DataFrame,
    *,
    kind: str,
) -> Mapping[str, Any]:
    """Keep E06a calibration cross-study after panel-overlap filtering."""
    require_columns(frame, ["study_group", "target", "prediction"])
    studies = int(frame["study_group"].astype(str).nunique())
    if studies < 2:
        raise DataContractError(
            "E06a calibration requires at least two task-panel-overlap studies"
        )
    return _ORIGINAL_FIT_CALIBRATOR(frame, kind=kind)


# Challenge calibration and the nested evaluator resolve this helper from the
# base module globals at call time.
_base._fit_calibrator = _fit_calibrator


def _eligible_outer_splits(
    dataset,
    *,
    panel_strains: Sequence[str],
) -> tuple[list[tuple[NamedSplit, list[NamedSplit]]], list[Mapping[str, Any]], int]:
    """Select evaluable outer folds from strain availability only."""
    train = dataset.train.reset_index(drop=True)
    candidates = purged_leave_one_study_out(
        train["study_group"].astype(str),
        train["subject_group"].astype(str),
    )
    eligible: list[tuple[NamedSplit, list[NamedSplit]]] = []
    skipped: list[Mapping[str, Any]] = []

    for outer in candidates:
        outer_train = train.iloc[outer.train_indices].reset_index(drop=True)
        outer_validation = train.iloc[outer.validation_indices].reset_index(drop=True)
        if outer_validation.empty:
            skipped.append(
                {
                    "held_out_group": str(outer.held_out_group),
                    "reason": "empty_outer_validation",
                    "validation_rows": 0,
                    "validation_panel_overlap_rows": 0,
                    "outer_training_panel_overlap_studies": 0,
                }
            )
            continue

        validation_overlap = _panel_overlap_mask(
            outer_validation,
            panel_strains=panel_strains,
        )
        overlap_rows = int(validation_overlap.sum())
        training_overlap_studies = _panel_overlap_study_count(
            outer_train,
            panel_strains=panel_strains,
        )
        if overlap_rows == 0:
            skipped.append(
                {
                    "held_out_group": str(outer.held_out_group),
                    "reason": "zero_fixed_panel_overlap",
                    "validation_rows": int(len(outer_validation)),
                    "validation_panel_overlap_rows": 0,
                    "outer_training_panel_overlap_studies": training_overlap_studies,
                }
            )
            continue
        if training_overlap_studies < 2:
            skipped.append(
                {
                    "held_out_group": str(outer.held_out_group),
                    "reason": "fewer_than_two_calibration_panel_studies",
                    "validation_rows": int(len(outer_validation)),
                    "validation_panel_overlap_rows": overlap_rows,
                    "outer_training_panel_overlap_studies": training_overlap_studies,
                }
            )
            continue

        inner_splits = purged_leave_one_study_out(
            outer_train["study_group"].astype(str),
            outer_train["subject_group"].astype(str),
        )
        if len(inner_splits) < 2:
            skipped.append(
                {
                    "held_out_group": str(outer.held_out_group),
                    "reason": "fewer_than_two_inner_study_folds",
                    "validation_rows": int(len(outer_validation)),
                    "validation_panel_overlap_rows": overlap_rows,
                    "outer_training_panel_overlap_studies": training_overlap_studies,
                }
            )
            continue
        eligible.append((outer, inner_splits))

    return eligible, skipped, len(candidates)


def _nested_calibration(
    dataset,
    *,
    spec: ModelSpec,
    panel_strains: Sequence[str],
) -> Mapping[str, Any]:
    """Nested E06a evaluation over only outcome-independent evaluable folds."""
    train = dataset.train.reset_index(drop=True)
    eligible, skipped, candidate_count = _eligible_outer_splits(
        dataset,
        panel_strains=panel_strains,
    )
    if len(eligible) < 3:
        raise DataContractError(
            "E06a requires at least three task-panel-overlap outer study folds"
        )

    raw_parts: list[pd.DataFrame] = []
    calibrated_parts: dict[str, list[pd.DataFrame]] = {
        kind: [] for kind in _base.CALIBRATION_KINDS
    }
    fold_params: dict[str, list[Mapping[str, Any]]] = {
        kind: [] for kind in _base.CALIBRATION_KINDS
    }

    for outer, inner_splits in eligible:
        outer_train = train.iloc[outer.train_indices].reset_index(drop=True)
        outer_validation = train.iloc[outer.validation_indices].reset_index(drop=True)
        inner_dataset = replace(dataset, train=outer_train)
        inner_evaluation = evaluate_hai_spec(
            inner_dataset,
            spec=spec,
            splits=inner_splits,
            panel_strains=panel_strains,
        )
        calibration_train = _base._panel_frame_from_oof(
            inner_evaluation,
            panel_strains=panel_strains,
        )

        # Explicitly verify the scientific cross-study calibration contract after
        # OOF panel filtering.  The helper itself repeats this check before fit.
        if int(calibration_train["study_group"].astype(str).nunique()) < 2:
            raise DataContractError(
                "E06a inner OOF calibration has fewer than two panel-overlap studies"
            )

        _, target_prediction = fit_final_model(
            outer_train,
            outer_validation,
            target_column=dataset.target_column,
            spec=spec,
            excluded_columns=dataset.excluded_columns,
        )
        post_prediction = dataset.target_prediction_to_post_hai(
            target_prediction,
            frame=outer_validation,
        )
        raw_panel = _base._panel_frame(
            outer_validation,
            post_prediction,
            panel_strains=panel_strains,
            split=str(outer.name),
        )
        raw_parts.append(raw_panel)

        for kind in _base.CALIBRATION_KINDS:
            params = _fit_calibrator(calibration_train, kind=kind)
            calibrated = raw_panel.copy()
            calibrated["prediction"] = _base._apply_calibrator(
                calibrated["prediction"].to_numpy(dtype=float),
                params,
            )
            calibrated_parts[kind].append(calibrated)
            fold_params[kind].append(
                {
                    "outer_fold": str(outer.name),
                    "held_out_group": str(outer.held_out_group),
                    "inner_split_count": int(len(inner_splits)),
                    **dict(params),
                }
            )

    raw = pd.concat(raw_parts, ignore_index=True)
    raw_metrics = _base._metric_bundle(raw)
    candidates: dict[str, Any] = {}
    for kind in _base.CALIBRATION_KINDS:
        calibrated = pd.concat(calibrated_parts[kind], ignore_index=True)
        calibrated_metrics = _base._metric_bundle(calibrated)
        rank = _base._rank_preservation(raw, calibrated)
        rmse = _base._rmse_diagnostics(raw_metrics, calibrated_metrics)
        candidates[kind] = _base._json_safe(
            {
                "metrics": calibrated_metrics,
                "rank_preservation": rank,
                "rmse_diagnostics": rmse,
                "outer_fold_calibrators": fold_params[kind],
            }
        )

    return _base._json_safe(
        {
            "outer_candidate_split_count": int(candidate_count),
            "outer_split_count": int(len(eligible)),
            "skipped_outer_split_count": int(len(skipped)),
            "skipped_outer_folds": skipped,
            "outer_fold_eligibility": (
                "validation study has >=1 fixed-panel strain and outer-training "
                "partition has >=2 fixed-panel-overlap studies; availability only"
            ),
            "raw_metrics": raw_metrics,
            "calibrated": candidates,
        }
    )


# Install the corrected evaluator for all base E06a condition calls.
_base._nested_calibration = _nested_calibration


def run_strategy_e06a(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    result = dict(_base.run_strategy_e06a(*args, **kwargs))
    contract = dict(result.get("calibration_contract") or {})
    contract.update(
        {
            "outer_fold_panel_overlap_filter": (
                "fixed_panel_strain_availability_only_before_outcome_access"
            ),
            "zero_panel_overlap_studies_scored_as_task23_proxy": False,
            "minimum_calibrator_panel_overlap_studies": 2,
        }
    )
    result["calibration_contract"] = contract
    return _base._json_safe(result)


EXPERIMENT = _base.EXPERIMENT
TASK = _base.TASK
TARGET_DAY = _base.TARGET_DAY
MODEL_NAME = _base.MODEL_NAME
CALIBRATION_KINDS = _base.CALIBRATION_KINDS
MIN_SLOPE = _base.MIN_SLOPE
MIN_EQUAL_STUDY_RMSE_REDUCTION = _base.MIN_EQUAL_STUDY_RMSE_REDUCTION
MIN_MEDIAN_STUDY_RMSE_REDUCTION = _base.MIN_MEDIAN_STUDY_RMSE_REDUCTION
MAX_WORST_STUDY_RMSE_RATIO = _base.MAX_WORST_STUDY_RMSE_RATIO
RANK_TOLERANCE = _base.RANK_TOLERANCE

__all__ = [
    "EXPERIMENT",
    "TASK",
    "TARGET_DAY",
    "MODEL_NAME",
    "CALIBRATION_KINDS",
    "MIN_SLOPE",
    "MIN_EQUAL_STUDY_RMSE_REDUCTION",
    "MIN_MEDIAN_STUDY_RMSE_REDUCTION",
    "MAX_WORST_STUDY_RMSE_RATIO",
    "RANK_TOLERANCE",
    "_eligible_outer_splits",
    "_fit_calibrator",
    "_nested_calibration",
    "run_strategy_e06a",
]
