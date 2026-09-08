"""Strategy v2 E06b: D28/D365 partially pooled two-head HAI model.

E06a showed that a single global post-hoc calibration map does not transfer
across historical studies.  E06b therefore moves the sharing *inside* the
predictive model instead of calibrating its output.

The primary estimand remains Task2.3 / D365.  Day-28 outcomes are auxiliary
training labels only: no observed Day-28 response is ever used as a feature for
a D365 validation or Challenge donor.

The shared model is deliberately low capacity.  After a train-only copy of the
frozen B2.1 preprocessing, a Ridge(alpha=100) is fit on

    [Z, day_code * Z, day_code]

where ``day_code=-0.5`` for D28 and ``+0.5`` for D365.  The first block is a
shared baseline/strain representation and the second block is a regularized
D28-vs-D365 coefficient deviation.  This is equivalent to two linear heads
with partial pooling rather than two unrelated models.

Evaluation uses the same outcome-independent D365 fixed-panel fold eligibility
as repaired E06a.  For every held D365 study, its study and biological subjects
are purged from *both* the D365 and auxiliary D28 training rows before the
preprocessor or Ridge is fit.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .aliases import canonicalize_strain
from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns, require_finite
from .cv import NamedSplit, purged_leave_one_study_out
from .datasets import HAIModelDataset, build_hai_model_dataset
from .evaluation import aggregate_hai_task_predictions
from .hai_transfer import add_hai_ontology_features
from .hai_transfer_v2 import normalize_sequence_reference_schema
from .metrics import percentile_rank, safe_spearman
from .models import ModelSpec, build_estimator, fit_final_model, prepare_model_frame
from .strategy_e06 import (
    _canonical_panel,
    _find_model,
    _historical_panel_coverage,
    _json_safe,
    _metric_bundle,
    _panel_frame,
    _rmse_diagnostics,
)


EXPERIMENT = "strategy_v2_e06b_task23_d28_d365_two_head"
TASK = "Task2.3"
PRIMARY_DAY = 365
AUXILIARY_DAY = 28
MODEL_NAME = "ridge_exact_a100"
BASE_CONDITIONS: tuple[str, ...] = (
    "b21_reference",
    "phase_a_target_domain",
)
DAY_CODE = {AUXILIARY_DAY: -0.5, PRIMARY_DAY: 0.5}
MIN_EQUAL_STUDY_RMSE_REDUCTION = 0.02
MIN_MEDIAN_STUDY_RMSE_REDUCTION = 0.0
MAX_WORST_STUDY_RMSE_RATIO = 1.05
MIN_EQUAL_STUDY_SPEARMAN_DELTA = -0.01
LARGE_STUDY_N = 10
MAX_LARGE_STUDY_SPEARMAN_DECLINE = -0.10


def _feature_columns(dataset: HAIModelDataset, spec: ModelSpec) -> list[str]:
    columns = [
        column
        for column in dataset.feature_columns(dataset.train)
        if column not in set(spec.drop_columns)
    ]
    if not columns:
        raise DataContractError("E06b has no model features after exclusions")
    return columns


def _require_compatible_feature_space(
    day28: HAIModelDataset,
    day365: HAIModelDataset,
    spec: ModelSpec,
) -> list[str]:
    first = _feature_columns(day28, spec)
    second = _feature_columns(day365, spec)
    if first != second:
        raise DataContractError(
            "E06b D28/D365 feature spaces differ after frozen model exclusions"
        )
    challenge = [
        column
        for column in day365.feature_columns(day365.challenge)
        if column not in set(spec.drop_columns)
    ]
    if challenge != second:
        raise DataContractError("E06b D365 train/challenge feature spaces differ")
    return second


def _panel_overlap_rows(
    frame: pd.DataFrame,
    *,
    panel_strains: Sequence[str],
) -> int:
    require_columns(frame, ["virus_strain"], table_name="E06b panel eligibility")
    panel = set(_canonical_panel(panel_strains))
    strain = frame["virus_strain"].map(canonicalize_strain)
    return int(strain.isin(panel).sum())


def _eligible_outer_splits(
    dataset: HAIModelDataset,
    *,
    panel_strains: Sequence[str],
) -> tuple[list[NamedSplit], list[Mapping[str, Any]], int]:
    train = dataset.train.reset_index(drop=True)
    candidates = purged_leave_one_study_out(
        train["study_group"].astype(str),
        train["subject_group"].astype(str),
    )
    eligible: list[NamedSplit] = []
    skipped: list[Mapping[str, Any]] = []
    for split in candidates:
        validation = train.iloc[split.validation_indices]
        overlap = _panel_overlap_rows(validation, panel_strains=panel_strains)
        if overlap == 0:
            skipped.append(
                {
                    "held_out_group": str(split.held_out_group),
                    "reason": "zero_fixed_panel_overlap",
                    "validation_rows": int(len(validation)),
                    "validation_panel_overlap_rows": 0,
                }
            )
            continue
        eligible.append(split)
    if len(eligible) < 3:
        raise DataContractError(
            "E06b requires at least three D365 task-panel-overlap study folds"
        )
    return eligible, skipped, int(len(candidates))


def _purge_auxiliary_rows(
    day28: HAIModelDataset,
    *,
    held_study: str,
    held_subjects: set[str],
) -> pd.DataFrame:
    train = day28.train
    study = train["study_group"].astype(str)
    subject = train["subject_group"].astype(str)
    keep = study.ne(str(held_study)) & ~subject.isin(held_subjects)
    result = train.loc[keep].reset_index(drop=True)
    if result.empty:
        raise DataContractError("E06b auxiliary D28 training became empty after purge")
    if result["study_group"].astype(str).eq(str(held_study)).any():
        raise DataContractError("E06b auxiliary purge retained held study")
    if result["subject_group"].astype(str).isin(held_subjects).any():
        raise DataContractError("E06b auxiliary purge retained held subject")
    return result


def _fit_two_head_ridge(
    day28_train: pd.DataFrame,
    day365_train: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    *,
    day28: HAIModelDataset,
    day365: HAIModelDataset,
    spec: ModelSpec,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    if spec.family != "ridge" or spec.name != MODEL_NAME:
        raise DataContractError("E06b two-head control requires ridge_exact_a100")
    alpha = float(spec.params.get("alpha", np.nan))
    if not np.isfinite(alpha) or alpha != 100.0:
        raise DataContractError("E06b two-head control requires Ridge alpha=100")

    features = _require_compatible_feature_space(day28, day365, spec)
    d28_x = day28_train[features].copy()
    d365_x = day365_train[features].copy()
    pred_x = prediction_frame[features].copy()
    stacked = pd.concat([d28_x, d365_x], ignore_index=True)

    model_stack, numeric, categorical = prepare_model_frame(
        stacked,
        excluded_columns=(),
    )
    model_pred, pred_numeric, pred_categorical = prepare_model_frame(
        pred_x,
        excluded_columns=(),
    )
    if numeric != pred_numeric or categorical != pred_categorical:
        raise DataContractError("E06b train/prediction feature typing differs")

    pipeline = build_estimator(
        spec,
        numeric_columns=numeric,
        categorical_columns=categorical,
    )
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = np.asarray(preprocessor.fit_transform(model_stack), dtype=float)
    transformed_pred = np.asarray(preprocessor.transform(model_pred), dtype=float)
    require_finite(transformed, name="E06b transformed training features")
    require_finite(transformed_pred, name="E06b transformed prediction features")

    d28_target = pd.to_numeric(
        day28_train[day28.target_column], errors="coerce"
    ).to_numpy(dtype=float)
    d365_target = pd.to_numeric(
        day365_train[day365.target_column], errors="coerce"
    ).to_numpy(dtype=float)
    target = np.concatenate([d28_target, d365_target])
    require_finite(target, name="E06b two-head target")

    day_code = np.concatenate(
        [
            np.full(len(d28_train), DAY_CODE[AUXILIARY_DAY], dtype=float),
            np.full(len(day365_train), DAY_CODE[PRIMARY_DAY], dtype=float),
        ]
    )
    design = np.column_stack(
        [
            transformed,
            transformed * day_code[:, None],
            day_code,
        ]
    )
    prediction_code = np.full(
        len(prediction_frame), DAY_CODE[PRIMARY_DAY], dtype=float
    )
    prediction_design = np.column_stack(
        [
            transformed_pred,
            transformed_pred * prediction_code[:, None],
            prediction_code,
        ]
    )
    require_finite(design, name="E06b two-head design")
    require_finite(prediction_design, name="E06b two-head prediction design")

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(design, target)
    prediction = np.asarray(model.predict(prediction_design), dtype=float).reshape(-1)
    require_finite(prediction, name="E06b two-head prediction")

    return prediction, {
        "alpha": alpha,
        "day_code_d28": DAY_CODE[AUXILIARY_DAY],
        "day_code_d365": DAY_CODE[PRIMARY_DAY],
        "d28_training_rows": int(len(day28_train)),
        "d365_training_rows": int(len(day365_train)),
        "d28_training_studies": int(day28_train["study_group"].astype(str).nunique()),
        "d365_training_studies": int(day365_train["study_group"].astype(str).nunique()),
        "raw_feature_count": int(len(features)),
        "transformed_feature_count": int(transformed.shape[1]),
        "two_head_design_feature_count": int(design.shape[1]),
        "observed_d28_used_as_d365_feature": False,
    }


def _spearman_diagnostics(
    independent_metrics: Mapping[str, Any],
    shared_metrics: Mapping[str, Any],
) -> Mapping[str, Any]:
    left = {
        str(row["study_group"]): row
        for row in independent_metrics["study_metrics"]
    }
    right = {
        str(row["study_group"]): row
        for row in shared_metrics["study_metrics"]
    }
    if set(left) != set(right):
        raise DataContractError("E06b study sets differ between paired conditions")
    rows: list[Mapping[str, Any]] = []
    large_declines: list[float] = []
    for study in sorted(left):
        raw = float(left[study]["spearman"])
        shared = float(right[study]["spearman"])
        n = int(left[study]["n"])
        delta = shared - raw
        if n >= LARGE_STUDY_N:
            large_declines.append(delta)
        rows.append(
            {
                "study": study,
                "n": n,
                "independent_spearman": raw,
                "shared_two_head_spearman": shared,
                "delta": delta,
            }
        )
    independent_mean = float(
        independent_metrics["equal_study_summary"]["spearman_mean"]
    )
    shared_mean = float(shared_metrics["equal_study_summary"]["spearman_mean"])
    return _json_safe(
        {
            "equal_study_mean_delta": shared_mean - independent_mean,
            "worst_large_study_delta": (
                float(min(large_declines)) if large_declines else None
            ),
            "by_study": rows,
        }
    )


def _challenge_rank_agreement(
    independent_task: pd.DataFrame,
    shared_task: pd.DataFrame,
) -> Mapping[str, Any]:
    require_columns(independent_task, ["participant_id", "prediction"])
    require_columns(shared_task, ["participant_id", "prediction"])
    aligned = independent_task[["participant_id", "prediction"]].rename(
        columns={"prediction": "independent_prediction"}
    ).merge(
        shared_task[["participant_id", "prediction"]].rename(
            columns={"prediction": "shared_prediction"}
        ),
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )
    if len(aligned) != len(independent_task) or len(aligned) != len(shared_task):
        raise DataContractError("E06b Challenge task predictions do not align")
    independent = aligned["independent_prediction"].to_numpy(dtype=float)
    shared = aligned["shared_prediction"].to_numpy(dtype=float)
    require_finite(independent, name="E06b Challenge independent prediction")
    require_finite(shared, name="E06b Challenge shared prediction")
    left_rank = percentile_rank(independent)
    right_rank = percentile_rank(shared)
    delta = np.abs(left_rank - right_rank)
    return _json_safe(
        {
            "rank_spearman": safe_spearman(independent, shared).to_dict(),
            "mean_absolute_percentile_difference": float(delta.mean()),
            "max_absolute_percentile_difference": float(delta.max()),
            "donor_count": int(len(aligned)),
        }
    )


def _promotion(
    independent_metrics: Mapping[str, Any],
    shared_metrics: Mapping[str, Any],
    spearman: Mapping[str, Any],
) -> Mapping[str, Any]:
    rmse = _rmse_diagnostics(independent_metrics, shared_metrics)
    worst_large = spearman.get("worst_large_study_delta")
    checks = {
        "equal_study_mean_rmse_reduction_at_least_2pct": bool(
            float(rmse["equal_study_mean_relative_reduction"])
            >= MIN_EQUAL_STUDY_RMSE_REDUCTION
        ),
        "median_study_rmse_improves": bool(
            float(rmse["median_study_relative_reduction"])
            > MIN_MEDIAN_STUDY_RMSE_REDUCTION
        ),
        "no_study_rmse_worse_than_5pct": bool(
            float(rmse["worst_study_rmse_ratio"]) <= MAX_WORST_STUDY_RMSE_RATIO
        ),
        "equal_study_spearman_not_materially_worse": bool(
            float(spearman["equal_study_mean_delta"])
            >= MIN_EQUAL_STUDY_SPEARMAN_DELTA
        ),
        "no_large_study_spearman_decline_below_minus_0.10": bool(
            worst_large is None
            or float(worst_large) >= MAX_LARGE_STUDY_SPEARMAN_DECLINE
        ),
    }
    return _json_safe(
        {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "rmse_diagnostics": rmse,
            "spearman_diagnostics": spearman,
        }
    )


def _evaluate_condition(
    name: str,
    day28: HAIModelDataset,
    day365: HAIModelDataset,
    *,
    spec: ModelSpec,
    panel_strains: Sequence[str],
    expected_challenge_donors: int = 40,
) -> Mapping[str, Any]:
    train365 = day365.train.reset_index(drop=True)
    eligible, skipped, candidate_count = _eligible_outer_splits(
        day365,
        panel_strains=panel_strains,
    )
    independent_parts: list[pd.DataFrame] = []
    shared_parts: list[pd.DataFrame] = []
    fold_training: list[Mapping[str, Any]] = []

    for split in eligible:
        outer_train365 = train365.iloc[split.train_indices].reset_index(drop=True)
        outer_validation = train365.iloc[split.validation_indices].reset_index(drop=True)
        held_subjects = set(outer_validation["subject_group"].astype(str))
        outer_train28 = _purge_auxiliary_rows(
            day28,
            held_study=str(split.held_out_group),
            held_subjects=held_subjects,
        )

        _, independent_target = fit_final_model(
            outer_train365,
            outer_validation,
            target_column=day365.target_column,
            spec=spec,
            excluded_columns=day365.excluded_columns,
        )
        shared_target, training_contract = _fit_two_head_ridge(
            outer_train28,
            outer_train365,
            outer_validation,
            day28=day28,
            day365=day365,
            spec=spec,
        )
        independent_post = day365.target_prediction_to_post_hai(
            independent_target,
            frame=outer_validation,
        )
        shared_post = day365.target_prediction_to_post_hai(
            shared_target,
            frame=outer_validation,
        )
        independent_parts.append(
            _panel_frame(
                outer_validation,
                independent_post,
                panel_strains=panel_strains,
                split=str(split.name),
            )
        )
        shared_parts.append(
            _panel_frame(
                outer_validation,
                shared_post,
                panel_strains=panel_strains,
                split=str(split.name),
            )
        )
        fold_training.append(
            {
                "outer_fold": str(split.name),
                "held_out_group": str(split.held_out_group),
                **dict(training_contract),
            }
        )

    independent_panel = pd.concat(independent_parts, ignore_index=True)
    shared_panel = pd.concat(shared_parts, ignore_index=True)
    independent_metrics = _metric_bundle(independent_panel)
    shared_metrics = _metric_bundle(shared_panel)
    spearman = _spearman_diagnostics(independent_metrics, shared_metrics)
    promotion = _promotion(independent_metrics, shared_metrics, spearman)

    # Challenge prediction uses all historical labels. D28 remains auxiliary
    # training only; no post-vaccination Challenge value is available or used.
    _, independent_target = fit_final_model(
        day365.train,
        day365.challenge,
        target_column=day365.target_column,
        spec=spec,
        excluded_columns=day365.excluded_columns,
    )
    shared_target, challenge_training = _fit_two_head_ridge(
        day28.train,
        day365.train,
        day365.challenge,
        day28=day28,
        day365=day365,
        spec=spec,
    )
    independent_post = day365.target_prediction_to_post_hai(
        independent_target,
        frame=day365.challenge,
    )
    shared_post = day365.target_prediction_to_post_hai(
        shared_target,
        frame=day365.challenge,
    )
    independent_rows = day365.challenge[["participant_id", "virus_strain"]].copy()
    independent_rows["prediction"] = independent_post
    shared_rows = day365.challenge[["participant_id", "virus_strain"]].copy()
    shared_rows["prediction"] = shared_post
    independent_task = aggregate_hai_task_predictions(
        independent_rows,
        panel_strains=panel_strains,
        task=TASK,
    )
    shared_task = aggregate_hai_task_predictions(
        shared_rows,
        panel_strains=panel_strains,
        task=TASK,
    )
    if len(independent_task) != expected_challenge_donors:
        raise DataContractError(
            f"E06b expected {expected_challenge_donors} Challenge donors; "
            f"found {len(independent_task)}"
        )

    return _json_safe(
        {
            "condition": name,
            "model": spec.to_dict(),
            "historical_study_out": {
                "outer_candidate_split_count": candidate_count,
                "outer_split_count": int(len(eligible)),
                "skipped_outer_split_count": int(len(skipped)),
                "skipped_outer_folds": skipped,
                "independent_d365": independent_metrics,
                "shared_two_head": shared_metrics,
                "fold_training_contract": fold_training,
            },
            "paired_diagnostics": {
                "rmse": promotion["rmse_diagnostics"],
                "spearman": promotion["spearman_diagnostics"],
            },
            "promotion": {
                "passed": promotion["passed"],
                "checks": promotion["checks"],
            },
            "challenge_shared_vs_independent_rank": _challenge_rank_agreement(
                independent_task,
                shared_task,
            ),
            "challenge_training_contract": challenge_training,
        }
    )


def run_strategy_e06b(
    config: BaselineConfig,
    inputs: Any,
    *,
    sequence_reference: pd.DataFrame,
    vaccine_reference: pd.DataFrame,
) -> Mapping[str, Any]:
    """Run the predeclared E06b D28/D365 partial-pooling comparison."""
    if str(config.section("hai").get("target_representation", "residual")) != "residual":
        raise DataContractError(
            "E06b frozen controls require hai.target_representation=residual"
        )
    if str(config.section("selection", required=False).get("policy", "legacy")) != "robust_v1":
        raise DataContractError("E06b requires selection.policy=robust_v1")

    panel = _canonical_panel(inputs.challenge_strains)
    sequence_reference = normalize_sequence_reference_schema(sequence_reference)
    tables = inputs.tables
    spec = _find_model(config)

    def build_day(day: int) -> HAIModelDataset:
        return build_hai_model_dataset(
            tables["public_serology"],
            tables["challenge_serology"],
            tables["participants"],
            tables["investigations"],
            day=day,
            target_representation="residual",
            challenge_panel_strains=panel,
        )

    base28 = build_day(AUXILIARY_DAY)
    base365 = build_day(PRIMARY_DAY)
    target28 = add_hai_ontology_features(
        base28,
        vaccine_reference=vaccine_reference,
        sequence_reference=sequence_reference,
        vaccine_2025=inputs.vaccine_strains,
        condition="ontology_sequence_target_domain",
    )
    target365 = add_hai_ontology_features(
        base365,
        vaccine_reference=vaccine_reference,
        sequence_reference=sequence_reference,
        vaccine_2025=inputs.vaccine_strains,
        condition="ontology_sequence_target_domain",
    )

    conditions = {
        "b21_reference": _evaluate_condition(
            "b21_reference",
            base28,
            base365,
            spec=spec,
            panel_strains=panel,
        ),
        "phase_a_target_domain": _evaluate_condition(
            "phase_a_target_domain",
            target28,
            target365,
            spec=spec,
            panel_strains=panel,
        ),
    }

    passed: list[tuple[float, str]] = []
    for name, payload in conditions.items():
        if payload["promotion"]["passed"]:
            value = float(
                payload["historical_study_out"]["shared_two_head"][
                    "equal_study_summary"
                ]["rmse_mean"]
            )
            passed.append((value, name))
    selected = min(passed)[1] if passed else None

    coverage = _historical_panel_coverage(base365, panel_strains=panel)
    return _json_safe(
        {
            "experiment": EXPERIMENT,
            "task": TASK,
            "primary_day": PRIMARY_DAY,
            "auxiliary_day": AUXILIARY_DAY,
            "conditions": conditions,
            "selected_promoted_condition": selected,
            "promotion_thresholds": {
                "minimum_equal_study_mean_rmse_reduction": MIN_EQUAL_STUDY_RMSE_REDUCTION,
                "minimum_median_study_rmse_reduction_strictly_greater_than": MIN_MEDIAN_STUDY_RMSE_REDUCTION,
                "maximum_worst_study_rmse_ratio": MAX_WORST_STUDY_RMSE_RATIO,
                "minimum_equal_study_spearman_delta": MIN_EQUAL_STUDY_SPEARMAN_DELTA,
                "large_study_n": LARGE_STUDY_N,
                "maximum_large_study_spearman_decline": MAX_LARGE_STUDY_SPEARMAN_DECLINE,
            },
            "historical_panel_proxy_coverage": coverage,
            "model_contract": {
                "family": "ridge",
                "alpha": 100.0,
                "shared_design": "[Z, day_code*Z, day_code]",
                "day_code_d28": DAY_CODE[AUXILIARY_DAY],
                "day_code_d365": DAY_CODE[PRIMARY_DAY],
                "preprocessing_fit": "outer_training_rows_only_across_D28_and_D365",
                "held_study_and_subjects_purged_from_both_days": True,
                "observed_d28_used_as_d365_feature": False,
                "d28_role": "auxiliary_training_label_only",
                "d365_role": "primary_Task2.3_target",
            },
            "historical_target_contract": (
                "D365 fixed Challenge-panel intersection; incomplete public panel "
                "proxy, not complete 12-strain official-target CV"
            ),
            "negative_result_scope": (
                "fixed low-capacity Ridge partial pooling between D28 and D365; "
                "not a rejection of all durability or longitudinal models"
            ),
            "competition_submission_attempted": False,
            "leaderboard_used_for_selection": False,
            "output_policy": (
                "aggregate_only_public_study_names_no_participant_ids_or_row_predictions"
            ),
        }
    )


__all__ = [
    "EXPERIMENT",
    "TASK",
    "PRIMARY_DAY",
    "AUXILIARY_DAY",
    "MODEL_NAME",
    "BASE_CONDITIONS",
    "run_strategy_e06b",
    "_fit_two_head_ridge",
    "_eligible_outer_splits",
]
