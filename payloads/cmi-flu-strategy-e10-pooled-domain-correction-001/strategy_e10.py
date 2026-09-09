"""Strategy-v2 E10: pooled X-only domain-conditioned residual correction.

E10 preserves one pooled predictor.  It never fits isolated source-study experts.
For every outer held-study pseudo-challenge, target-domain X may determine source
study weights, but held outcomes are unavailable until after predictions and
weights are frozen.  The supervised correction target itself is built from
inner study-purged base predictions.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .anchor_residual import _anchor_direction, _oriented_anchor, _target_rank
from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns, require_finite
from .cv import purged_leave_one_study_out
from .datasets import TaskDataset
from .evaluation import default_splits_for_task
from .metrics import percentile_rank, safe_spearman
from .models import ModelSpec, build_estimator, fit_final_model, prepare_model_frame
from .runner import InputBundle, build_b02_datasets
from .study_similarity import similarity_weights, study_distribution_distance

EXPERIMENT = "strategy_v2_e10_pooled_domain_conditioned_residual"
COMPARISON_CONTRACT = "paired_subject_purged_v2"
TASKS = ("Task1.1", "Task1.2")
RIDGE_ALPHA = 10.0
CORRECTION_SHRINKAGE = 0.25
MAX_ABSOLUTE_RANK_CORRECTION = 0.05
SOURCE_WEIGHT_MIN = 0.5
SOURCE_WEIGHT_MAX = 2.0
DEVIATION_SCALE = 0.25
MIN_SOURCE_SUBJECTS = 8
PROMOTION_MEAN_DELTA = 0.02
PROMOTION_MIN_STUDY_DELTA = -0.10
WEIGHT_TOLERANCE = 1e-10

TASK_CONTRACT: Mapping[str, Mapping[str, Any]] = {
    "Task1.1": {
        "model_set": "task_11",
        "base_model": "pls_2",
        "base_kind": "b21",
        "assay_prefix": "cytokine_",
        "anchor_column": None,
        "anchor_lambda": None,
    },
    "Task1.2": {
        "model_set": "task_12",
        "base_model": "et_d5_l5_sqrt",
        "base_kind": "anchor_residual",
        "assay_prefix": "flow_",
        "anchor_column": "flow_rank__Classical_monocytes",
        "anchor_lambda": 0.5,
    },
}
CORRECTION_TARGET = "__e10_crossfit_rank_residual"


def _find_spec(config: BaselineConfig, *, set_name: str, model_name: str) -> ModelSpec:
    matches = [spec for spec in config.model_specs(set_name) if spec.name == model_name]
    if len(matches) != 1:
        raise DataContractError(
            f"E10 frozen model {model_name!r} in {set_name!r} resolved to {len(matches)} specs"
        )
    return matches[0]


def _identity_spec(spec: ModelSpec) -> ModelSpec:
    return replace(spec, target_transform="identity", clip_min=None)


def _base_score(
    task: str,
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    dataset: TaskDataset,
    spec: ModelSpec,
) -> np.ndarray:
    """Fit the task's frozen competition base on `training` and score `prediction`."""
    contract = TASK_CONTRACT[task]
    if contract["base_kind"] == "b21":
        _, score = fit_final_model(
            training,
            prediction,
            target_column=dataset.target_column,
            spec=spec,
            excluded_columns=dataset.excluded_columns,
        )
        require_finite(score, name=f"E10.{task}.b21_base")
        return np.asarray(score, dtype=float)

    if contract["base_kind"] != "anchor_residual":
        raise DataContractError(f"E10 unknown base kind:{contract['base_kind']}")
    anchor_column = str(contract["anchor_column"])
    weight = float(contract["anchor_lambda"])
    require_columns(
        training,
        [dataset.target_column, anchor_column, "study_group"],
        table_name=f"E10 {task} anchor-residual training",
    )
    require_columns(prediction, [anchor_column], table_name=f"E10 {task} anchor prediction")
    direction = _anchor_direction(
        task,
        training,
        anchor_column=anchor_column,
        target_column=dataset.target_column,
    )
    train_anchor = _oriented_anchor(training, anchor_column=anchor_column, direction=direction)
    pred_anchor = _oriented_anchor(prediction, anchor_column=anchor_column, direction=direction)
    work = training.copy()
    work[CORRECTION_TARGET] = (
        _target_rank(work, target_column=dataset.target_column) - train_anchor
    )
    excluded = tuple(dict.fromkeys([*dataset.excluded_columns, dataset.target_column]))
    _, correction = fit_final_model(
        work,
        prediction,
        target_column=CORRECTION_TARGET,
        spec=_identity_spec(spec),
        excluded_columns=excluded,
    )
    score = pred_anchor + weight * np.asarray(correction, dtype=float)
    require_finite(score, name=f"E10.{task}.anchor_residual_base")
    return score


def _crossfit_rank_residual(
    task: str,
    training: pd.DataFrame,
    *,
    dataset: TaskDataset,
    spec: ModelSpec,
) -> np.ndarray:
    """Build leakage-safe residual targets using inner leave-one-study-out base scores."""
    require_columns(
        training,
        ["study_group", "subject_group", dataset.target_column],
        table_name=f"E10 {task} outer training",
    )
    if training["study_group"].nunique(dropna=False) < 2:
        raise DataContractError("E10 residual cross-fit requires at least two training studies")
    splits = purged_leave_one_study_out(
        training["study_group"], training["subject_group"]
    )
    residual = np.full(len(training), np.nan, dtype=float)
    for split in splits:
        inner_train = training.iloc[split.train_indices].copy()
        inner_held = training.iloc[split.validation_indices].copy()
        base = _base_score(
            task,
            inner_train,
            inner_held,
            dataset=dataset,
            spec=spec,
        )
        base_rank = percentile_rank(base)
        target_rank = _target_rank(inner_held, target_column=dataset.target_column)
        residual[split.validation_indices] = target_rank - base_rank
    require_finite(residual, name=f"E10.{task}.crossfit_residual")
    return residual


def _bounded_weight_scale(
    relative: Mapping[str, float],
    counts: Mapping[str, int],
) -> dict[str, float]:
    """Find a global multiplier so row-weighted mean is one while preserving bounds."""
    names = sorted(relative)
    if set(names) != set(counts):
        raise DataContractError("E10 source-weight/count domains differ")
    if any(int(counts[name]) <= 0 for name in names):
        raise DataContractError("E10 source-study counts must be positive")

    raw = np.asarray([float(relative[name]) for name in names], dtype=float)
    n = np.asarray([int(counts[name]) for name in names], dtype=float)
    if not np.isfinite(raw).all() or (raw <= 0).any():
        raise DataContractError("E10 relative weights must be finite and positive")

    def mean_for(scale: float) -> float:
        values = np.clip(scale * raw, SOURCE_WEIGHT_MIN, SOURCE_WEIGHT_MAX)
        return float(np.sum(values * n) / np.sum(n))

    lo, hi = 1e-8, 1e8
    if mean_for(lo) > 1.0 + WEIGHT_TOLERANCE or mean_for(hi) < 1.0 - WEIGHT_TOLERANCE:
        raise DataContractError("E10 cannot calibrate bounded weights to mean one")
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if mean_for(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    scale = math.sqrt(lo * hi)
    values = np.clip(scale * raw, SOURCE_WEIGHT_MIN, SOURCE_WEIGHT_MAX)
    result = {name: float(value) for name, value in zip(names, values, strict=True)}
    row_mean = sum(result[name] * int(counts[name]) for name in names) / sum(
        int(counts[name]) for name in names
    )
    if abs(row_mean - 1.0) > 1e-8:
        raise DataContractError(f"E10 bounded source weights do not mean-normalize:{row_mean}")
    if min(result.values()) < SOURCE_WEIGHT_MIN - 1e-12 or max(result.values()) > SOURCE_WEIGHT_MAX + 1e-12:
        raise DataContractError("E10 bounded source weights escaped clip")
    return result


def xonly_source_weights(
    training: pd.DataFrame,
    target_x: pd.DataFrame,
    *,
    assay_prefix: str,
) -> dict[str, Any]:
    """Compute source weights using only source/target X and study membership."""
    require_columns(training, ["study_group"], table_name="E10 source X")
    source_names = sorted(training["study_group"].astype(str).unique())
    if len(source_names) < 2:
        raise DataContractError("E10 X-only weighting requires at least two source studies")
    distances: dict[str, float] = {}
    components: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for study in source_names:
        source = training.loc[training["study_group"].astype(str).eq(study)].copy()
        counts[study] = int(len(source))
        if counts[study] < MIN_SOURCE_SUBJECTS:
            raise DataContractError(
                f"E10 source study {study} has {counts[study]} < {MIN_SOURCE_SUBJECTS} rows"
            )
        diagnostic = study_distribution_distance(
            source,
            target_x,
            assay_prefix=assay_prefix,
        )
        distances[study] = float(diagnostic["distance"])
        components[study] = diagnostic["components"]

    probabilities = similarity_weights(distances)
    uniform = 1.0 / len(source_names)
    relative = {study: probabilities[study] / uniform for study in source_names}
    bounded = _bounded_weight_scale(relative, counts)
    row_weight = (
        training["study_group"].astype(str).map(bounded).to_numpy(dtype=float)
    )
    require_finite(row_weight, name="E10.X_only_row_weight")
    ess = float(np.square(row_weight.sum()) / np.square(row_weight).sum())
    return {
        "source_weights": bounded,
        "source_probabilities": probabilities,
        "source_distances": distances,
        "distance_components": components,
        "source_counts": counts,
        "row_weight_mean": float(row_weight.mean()),
        "row_weight_min": float(row_weight.min()),
        "row_weight_max": float(row_weight.max()),
        "row_weight_ess": ess,
        "row_weight_ess_fraction": float(ess / len(row_weight)),
        "effective_source_count": float(
            1.0 / sum(float(value) ** 2 for value in probabilities.values())
        ),
        "held_outcomes_used": False,
        "_row_weight": row_weight,
    }


def _residual_feature_frames(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    dataset: TaskDataset,
    residual: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    work = training.copy()
    work[CORRECTION_TARGET] = np.asarray(residual, dtype=float)
    excluded = tuple(
        dict.fromkeys([*dataset.excluded_columns, dataset.target_column, CORRECTION_TARGET])
    )
    train_features, numeric, categorical = prepare_model_frame(
        work,
        excluded_columns=excluded,
    )
    pred_features, _, _ = prepare_model_frame(
        prediction,
        excluded_columns=excluded,
    )
    missing = [column for column in train_features.columns if column not in pred_features]
    if missing:
        raise DataContractError(f"E10 prediction frame missing correction features:{missing}")
    pred_features = pred_features[train_features.columns]
    return train_features, pred_features, numeric, categorical


def _weighted_shared_predict(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    dataset: TaskDataset,
    residual: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_x, pred_x, numeric, categorical = _residual_feature_frames(
        training, prediction, dataset=dataset, residual=residual
    )
    spec = ModelSpec(name="e10_ridge_a10", family="ridge", params={"alpha": RIDGE_ALPHA})
    estimator = build_estimator(
        spec, numeric_columns=numeric, categorical_columns=categorical
    )
    estimator.fit(
        train_x,
        np.asarray(residual, dtype=float),
        regressor__sample_weight=np.asarray(sample_weight, dtype=float),
    )
    prediction_values = np.asarray(estimator.predict(pred_x), dtype=float).reshape(-1)
    require_finite(prediction_values, name="E10.weighted_shared_prediction")
    return prediction_values, {
        "alpha": RIDGE_ALPHA,
        "pooled_model_count": 1,
        "isolated_source_models_fit": 0,
        "numeric_features": len(numeric),
        "categorical_features": len(categorical),
    }


def _shrunk_deviation_predict(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    dataset: TaskDataset,
    residual: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit shared coefficients + source-study deviations; predict unseen domain with shared part."""
    train_x, pred_x, numeric, categorical = _residual_feature_frames(
        training, prediction, dataset=dataset, residual=residual
    )
    spec = ModelSpec(name="e10_preprocessor", family="ridge", params={"alpha": RIDGE_ALPHA})
    pipeline = build_estimator(
        spec, numeric_columns=numeric, categorical_columns=categorical
    )
    preprocessor = pipeline.named_steps["preprocessor"]
    shared = np.asarray(preprocessor.fit_transform(train_x), dtype=float)
    pred_shared = np.asarray(preprocessor.transform(pred_x), dtype=float)
    require_finite(shared, name="E10.hierarchical_shared_X")
    require_finite(pred_shared, name="E10.hierarchical_prediction_X")

    studies = training["study_group"].astype(str).to_numpy()
    source_names = sorted(set(studies))
    if len(source_names) < 2:
        raise DataContractError("E10 hierarchical correction requires >=2 source studies")
    deviation_blocks = []
    intercept_blocks = []
    for study in source_names:
        indicator = (studies == study).astype(float).reshape(-1, 1)
        intercept_blocks.append(DEVIATION_SCALE * indicator)
        deviation_blocks.append(DEVIATION_SCALE * indicator * shared)
    augmented = np.column_stack([shared, *intercept_blocks, *deviation_blocks])
    pred_augmented = np.column_stack(
        [
            pred_shared,
            np.zeros((len(pred_shared), len(source_names)), dtype=float),
            np.zeros((len(pred_shared), len(source_names) * shared.shape[1]), dtype=float),
        ]
    )
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(augmented, np.asarray(residual, dtype=float))
    values = np.asarray(model.predict(pred_augmented), dtype=float).reshape(-1)
    require_finite(values, name="E10.shrunk_deviation_prediction")
    return values, {
        "alpha": RIDGE_ALPHA,
        "deviation_scale": DEVIATION_SCALE,
        "effective_deviation_penalty_multiplier": float(1.0 / (DEVIATION_SCALE**2)),
        "source_studies": len(source_names),
        "shared_transformed_features": int(shared.shape[1]),
        "augmented_features": int(augmented.shape[1]),
        "pooled_model_count": 1,
        "isolated_source_models_fit": 0,
        "unseen_target_deviation_columns": "all_zero",
    }


def _quantile_remap(base_score: Sequence[float], corrected_rank: Sequence[float]) -> np.ndarray:
    base = np.asarray(base_score, dtype=float)
    rank = np.asarray(corrected_rank, dtype=float)
    require_finite(base, name="E10.quantile_base")
    require_finite(rank, name="E10.corrected_rank")
    if base.ndim != 1 or rank.shape != base.shape or len(base) < 2:
        raise DataContractError("E10 quantile remap shape mismatch")
    rank = np.clip(rank, 0.0, 1.0)
    return np.asarray(np.quantile(np.sort(base), rank, method="linear"), dtype=float)


def _apply_correction(base_score: np.ndarray, correction: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    base_rank = percentile_rank(base_score)
    raw_delta = CORRECTION_SHRINKAGE * np.asarray(correction, dtype=float)
    delta = np.clip(
        raw_delta,
        -MAX_ABSOLUTE_RANK_CORRECTION,
        MAX_ABSOLUTE_RANK_CORRECTION,
    )
    corrected_rank = np.clip(base_rank + delta, 0.0, 1.0)
    candidate = _quantile_remap(base_score, corrected_rank)
    return candidate, {
        "shrinkage": CORRECTION_SHRINKAGE,
        "absolute_rank_correction_cap": MAX_ABSOLUTE_RANK_CORRECTION,
        "mean_absolute_rank_correction": float(np.mean(np.abs(delta))),
        "max_absolute_rank_correction": float(np.max(np.abs(delta))),
    }


def _spearman_value(y: Sequence[float], score: Sequence[float]) -> float | None:
    metric = safe_spearman(y, score)
    return float(metric.value) if metric.status == "ok" else None


def _fold_evaluation(
    task: str,
    training: pd.DataFrame,
    held: pd.DataFrame,
    *,
    dataset: TaskDataset,
    spec: ModelSpec,
) -> dict[str, Any]:
    base = _base_score(task, training, held, dataset=dataset, spec=spec)
    residual = _crossfit_rank_residual(task, training, dataset=dataset, spec=spec)
    weight_audit = xonly_source_weights(
        training,
        held,
        assay_prefix=str(TASK_CONTRACT[task]["assay_prefix"]),
    )
    row_weight = np.asarray(weight_audit.pop("_row_weight"), dtype=float)

    weighted_correction, weighted_fit = _weighted_shared_predict(
        training,
        held,
        dataset=dataset,
        residual=residual,
        sample_weight=row_weight,
    )
    hierarchical_correction, hierarchical_fit = _shrunk_deviation_predict(
        training,
        held,
        dataset=dataset,
        residual=residual,
    )
    weighted_score, weighted_move = _apply_correction(base, weighted_correction)
    hierarchical_score, hierarchical_move = _apply_correction(base, hierarchical_correction)
    target = pd.to_numeric(held[dataset.target_column], errors="coerce").to_numpy(dtype=float)
    require_finite(target, name=f"E10.{task}.held_target")
    base_metric = _spearman_value(target, base)
    weighted_metric = _spearman_value(target, weighted_score)
    hierarchical_metric = _spearman_value(target, hierarchical_score)
    return {
        "held_study": str(held["study_group"].iloc[0]),
        "n": int(len(held)),
        "training_rows": int(len(training)),
        "training_studies": int(training["study_group"].nunique(dropna=False)),
        "base_spearman": base_metric,
        "xonly_weighted_shared_spearman": weighted_metric,
        "shrunk_study_deviation_spearman": hierarchical_metric,
        "xonly_weighted_shared_delta": (
            None if base_metric is None or weighted_metric is None else weighted_metric - base_metric
        ),
        "shrunk_study_deviation_delta": (
            None if base_metric is None or hierarchical_metric is None else hierarchical_metric - base_metric
        ),
        "xonly_weight_audit": weight_audit,
        "xonly_weighted_fit": weighted_fit,
        "shrunk_study_deviation_fit": hierarchical_fit,
        "xonly_weighted_movement": weighted_move,
        "shrunk_study_deviation_movement": hierarchical_move,
    }


def _promotion(folds: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    deltas = [
        float(fold[f"{key}_delta"])
        for fold in folds
        if fold.get(f"{key}_delta") is not None and np.isfinite(float(fold[f"{key}_delta"]))
    ]
    if not deltas:
        return {
            "passed": False,
            "reason": "no_usable_held_studies",
            "mean_delta": None,
            "minimum_delta": None,
            "wins": 0,
            "required_wins": 0,
        }
    wins = int(sum(value > 0 for value in deltas))
    required = len(deltas) // 2 + 1
    mean_delta = float(np.mean(deltas))
    minimum_delta = float(np.min(deltas))
    passed = bool(
        mean_delta >= PROMOTION_MEAN_DELTA
        and minimum_delta >= PROMOTION_MIN_STUDY_DELTA
        and wins >= required
    )
    return {
        "passed": passed,
        "usable_held_studies": len(deltas),
        "mean_delta": mean_delta,
        "median_delta": float(np.median(deltas)),
        "minimum_delta": minimum_delta,
        "maximum_delta": float(np.max(deltas)),
        "wins": wins,
        "required_wins": required,
        "mean_delta_threshold": PROMOTION_MEAN_DELTA,
        "minimum_study_delta_threshold": PROMOTION_MIN_STUDY_DELTA,
    }


def _challenge_agreement(base: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    base_rank = percentile_rank(base)
    candidate_rank = percentile_rank(candidate)
    return {
        "rank_spearman": safe_spearman(base_rank, candidate_rank).to_dict(),
        "changed_rank_count": int(np.sum(np.abs(base_rank - candidate_rank) > 1e-12)),
        "mean_absolute_percentile_shift": float(np.mean(np.abs(base_rank - candidate_rank))),
        "max_absolute_percentile_shift": float(np.max(np.abs(base_rank - candidate_rank))),
    }


def evaluate_task_e10(
    dataset: TaskDataset,
    *,
    config: BaselineConfig,
) -> dict[str, Any]:
    task = dataset.task
    contract = TASK_CONTRACT[task]
    spec = _find_spec(
        config,
        set_name=str(contract["model_set"]),
        model_name=str(contract["base_model"]),
    )
    outer_splits = default_splits_for_task(dataset, random_state=config.random_state)
    folds = []
    for split in outer_splits:
        training = dataset.train.iloc[split.train_indices].copy()
        held = dataset.train.iloc[split.validation_indices].copy()
        if held["study_group"].nunique(dropna=False) != 1:
            raise DataContractError("E10 outer fold must hold exactly one study")
        folds.append(
            _fold_evaluation(
                task,
                training,
                held,
                dataset=dataset,
                spec=spec,
            )
        )

    weighted_promotion = _promotion(folds, "xonly_weighted_shared")
    hierarchical_promotion = _promotion(folds, "shrunk_study_deviation")

    full_residual = _crossfit_rank_residual(
        task, dataset.train, dataset=dataset, spec=spec
    )
    base_challenge = _base_score(
        task,
        dataset.train,
        dataset.challenge,
        dataset=dataset,
        spec=spec,
    )
    challenge_weight_audit = xonly_source_weights(
        dataset.train,
        dataset.challenge,
        assay_prefix=str(contract["assay_prefix"]),
    )
    challenge_row_weight = np.asarray(
        challenge_weight_audit.pop("_row_weight"), dtype=float
    )
    weighted_correction, weighted_fit = _weighted_shared_predict(
        dataset.train,
        dataset.challenge,
        dataset=dataset,
        residual=full_residual,
        sample_weight=challenge_row_weight,
    )
    hierarchical_correction, hierarchical_fit = _shrunk_deviation_predict(
        dataset.train,
        dataset.challenge,
        dataset=dataset,
        residual=full_residual,
    )
    weighted_challenge, weighted_move = _apply_correction(
        base_challenge, weighted_correction
    )
    hierarchical_challenge, hierarchical_move = _apply_correction(
        base_challenge, hierarchical_correction
    )

    candidates = {
        "xonly_weighted_shared": {
            "promotion": weighted_promotion,
            "challenge_agreement_vs_base": _challenge_agreement(
                base_challenge, weighted_challenge
            ),
            "challenge_movement": weighted_move,
            "fit": weighted_fit,
        },
        "shrunk_study_deviation": {
            "promotion": hierarchical_promotion,
            "challenge_agreement_vs_base": _challenge_agreement(
                base_challenge, hierarchical_challenge
            ),
            "challenge_movement": hierarchical_move,
            "fit": hierarchical_fit,
        },
    }
    passed = [name for name, payload in candidates.items() if payload["promotion"]["passed"]]
    selected = (
        max(
            passed,
            key=lambda name: float(candidates[name]["promotion"]["mean_delta"]),
        )
        if passed
        else None
    )
    return {
        "task": task,
        "base_contract": {
            "kind": contract["base_kind"],
            "model": contract["base_model"],
            "anchor_column": contract["anchor_column"],
            "anchor_lambda": contract["anchor_lambda"],
            "current_competition_incumbent": True,
        },
        "folds": folds,
        "candidates": candidates,
        "selected_local_candidate": selected,
        "competition_candidate": False,
        "public_probe_authorized": False,
        "challenge_source_weight_audit": challenge_weight_audit,
        "challenge_rows": int(len(dataset.challenge)),
    }


def run_e10(config: BaselineConfig, inputs: InputBundle) -> dict[str, Any]:
    if config.baseline != "b021_taskwise_robust":
        raise DataContractError("E10 requires the B2.1 robust config")
    if str(config.section("selection").get("policy", "")) != "robust_v1":
        raise DataContractError("E10 requires robust_v1 selection policy")
    if str(config.section("flow").get("task_12_mode", "")) != "broad":
        raise DataContractError("E10 Task1.2 requires broad-flow B2.1 feature space")

    datasets = build_b02_datasets(config, inputs)
    tasks = {}
    for task in TASKS:
        dataset = datasets.get(task)
        if not isinstance(dataset, TaskDataset):
            raise DataContractError(f"E10 missing compact dataset:{task}")
        dataset.validate()
        tasks[task] = evaluate_task_e10(dataset, config=config)

    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "comparison_contract": COMPARISON_CONTRACT,
        "tasks": tasks,
        "frozen_conditions": {
            "ridge_alpha": RIDGE_ALPHA,
            "correction_shrinkage": CORRECTION_SHRINKAGE,
            "absolute_rank_correction_cap": MAX_ABSOLUTE_RANK_CORRECTION,
            "source_weight_clip": [SOURCE_WEIGHT_MIN, SOURCE_WEIGHT_MAX],
            "deviation_scale": DEVIATION_SCALE,
            "minimum_source_subjects": MIN_SOURCE_SUBJECTS,
            "promotion_mean_delta": PROMOTION_MEAN_DELTA,
            "promotion_minimum_study_delta": PROMOTION_MIN_STUDY_DELTA,
            "task_contract": {task: dict(payload) for task, payload in TASK_CONTRACT.items()},
        },
        "held_target_outcomes_used_for_domain_weights": False,
        "isolated_source_study_models_allowed": False,
        "leaderboard_used_for_selection": False,
        "competition_submission_attempted": False,
        "incumbent_changed": False,
        "public_probe_authorized": False,
        "automatic_compute_retries": 0,
        "interpretation_limits": [
            "E10 tests this fixed X-only distance and this fixed low-capacity correction only.",
            "A negative result does not reject domain adaptation generally.",
            "Task1.2 is compared against the promoted fixed anchor-residual incumbent, not merely raw B2.1.",
            "Task1.1 is compared against frozen B2.1 pls_2.",
            "Quantile remapping preserves the base score marginal distribution while allowing donor re-ordering.",
        ],
    }


__all__ = [
    "EXPERIMENT",
    "TASK_CONTRACT",
    "xonly_source_weights",
    "evaluate_task_e10",
    "run_e10",
]
