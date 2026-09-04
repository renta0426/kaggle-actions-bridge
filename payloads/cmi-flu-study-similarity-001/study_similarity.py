"""Phase A pseudo-challenge study-similarity weighting diagnostics.

The experiment deliberately uses only baseline X from the held-out pseudo-challenge
study to choose source-study weights. Outcomes from the held study are used only after
predictions and weights are frozen, for evaluation.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns
from .cv import purged_leave_one_study_out
from .datasets import TaskDataset, build_task_11_dataset, build_task_12_dataset
from .metrics import percentile_rank, root_mean_squared_error, safe_spearman
from .models import ModelSpec, fit_final_model

TASK_CONTRACTS: Mapping[str, Mapping[str, str]] = {
    "Task1.1": {
        "model_set": "task_11",
        "model_name": "pls_2",
        "assay_prefix": "cytokine_",
    },
    "Task1.2": {
        "model_set": "task_12",
        "model_name": "enet_a0.001_l0.5",
        "assay_prefix": "flow_",
    },
}

SIMILARITY_SHRINK_TO_UNIFORM = 0.5
LOCATION_CLIP = 5.0
WORST_STUDY_TOLERANCE = 0.10
_EPS = 1e-12


def _finite_mean(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else None


def _robust_scale(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale <= _EPS:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad
    return scale if np.isfinite(scale) and scale > _EPS else None


def _numeric_distribution_components(
    source: pd.DataFrame,
    target: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[float | None, float | None]:
    location: list[float] = []
    missingness: list[float] = []
    for column in columns:
        left = pd.to_numeric(source[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(target[column], errors="coerce").to_numpy(dtype=float)
        left_finite = left[np.isfinite(left)]
        right_finite = right[np.isfinite(right)]
        missingness.append(
            abs(float(np.mean(~np.isfinite(left))) - float(np.mean(~np.isfinite(right))))
        )
        if left_finite.size == 0 or right_finite.size == 0:
            continue
        scale = _robust_scale(np.concatenate([left_finite, right_finite]))
        if scale is None:
            continue
        delta = abs(float(np.median(left_finite)) - float(np.median(right_finite))) / scale
        location.append(min(delta, LOCATION_CLIP))
    return _finite_mean(location), _finite_mean(missingness)


def _categorical_tv_distance(
    source: pd.DataFrame,
    target: pd.DataFrame,
    columns: Sequence[str],
) -> float | None:
    distances: list[float] = []
    for column in columns:
        left = source[column].astype("string").fillna("__MISSING__").astype(str)
        right = target[column].astype("string").fillna("__MISSING__").astype(str)
        categories = sorted(set(left).union(set(right)))
        if not categories:
            continue
        left_counts = left.value_counts(normalize=True, dropna=False)
        right_counts = right.value_counts(normalize=True, dropna=False)
        tv = 0.5 * sum(
            abs(float(left_counts.get(category, 0.0)) - float(right_counts.get(category, 0.0)))
            for category in categories
        )
        distances.append(float(tv))
    return _finite_mean(distances)


def study_distribution_distance(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    assay_prefix: str,
) -> dict[str, Any]:
    """Compute an X-only, low-dimensional distance between two study domains.

    Four components receive equal weight when available: assay location, assay
    missingness, age location/missingness, and categorical demographics. No target or
    study identifier column is read by this function.
    """

    if source.empty or target.empty:
        raise DataContractError("study distance requires non-empty source and target frames")
    assay_columns = sorted(
        column
        for column in source.columns
        if column.startswith(assay_prefix) and column in target.columns
    )
    if not assay_columns:
        raise DataContractError(f"no shared assay columns for prefix {assay_prefix!r}")
    require_columns(source, ["age", "biological_sex", "race"], table_name="source study X")
    require_columns(target, ["age", "biological_sex", "race"], table_name="target study X")

    assay_location, assay_missingness = _numeric_distribution_components(
        source, target, assay_columns
    )
    age_location, age_missingness = _numeric_distribution_components(
        source, target, ["age"]
    )
    demographic_categorical = _categorical_tv_distance(
        source, target, ["biological_sex", "race"]
    )
    age_component_values = [
        value for value in (age_location, age_missingness) if value is not None
    ]
    age_component = _finite_mean(age_component_values)
    components = {
        "assay_location": assay_location,
        "assay_missingness": assay_missingness,
        "age": age_component,
        "demographic_categorical": demographic_categorical,
    }
    finite_components = [
        float(value) for value in components.values() if value is not None and np.isfinite(value)
    ]
    if not finite_components:
        raise DataContractError("study distance produced no finite components")
    distance = float(np.mean(finite_components))
    return {
        "distance": distance,
        "components": components,
        "assay_feature_count": len(assay_columns),
        "source_rows": int(len(source)),
        "target_rows": int(len(target)),
    }


def similarity_weights(distances: Mapping[str, float]) -> dict[str, float]:
    """Convert distances to a fixed 50%-shrunk softmax similarity weighting."""

    if len(distances) < 2:
        raise DataContractError("similarity weighting requires at least two source studies")
    names = sorted(distances)
    values = np.asarray([float(distances[name]) for name in names], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise DataContractError("study distances must be finite and non-negative")
    positive = values[values > _EPS]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, _EPS)
    logits = -(values - float(np.min(values))) / scale
    raw = np.exp(np.clip(logits, -50.0, 0.0))
    raw /= float(raw.sum())
    uniform = 1.0 / len(names)
    weights = SIMILARITY_SHRINK_TO_UNIFORM * uniform + (
        1.0 - SIMILARITY_SHRINK_TO_UNIFORM
    ) * raw
    weights /= float(weights.sum())
    return {name: float(weight) for name, weight in zip(names, weights, strict=True)}


def _find_locked_spec(config: BaselineConfig, *, set_name: str, model_name: str) -> ModelSpec:
    matches = [spec for spec in config.model_specs(set_name) if spec.name == model_name]
    if len(matches) != 1:
        raise DataContractError(
            f"locked model {model_name!r} in set {set_name!r} resolved to {len(matches)} specs"
        )
    return matches[0]


def _rank_prediction(
    train_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    *,
    target_column: str,
    spec: ModelSpec,
    excluded_columns: Sequence[str],
) -> np.ndarray:
    """Fit the frozen model API and return only within-target-domain prediction ranks."""

    _, prediction = fit_final_model(
        train_frame,
        prediction_frame,
        target_column=target_column,
        spec=spec,
        excluded_columns=excluded_columns,
    )
    return percentile_rank(prediction)


def _rank_metrics(target: Sequence[float], score: Sequence[float]) -> dict[str, Any]:
    target_array = np.asarray(target, dtype=float)
    score_array = np.asarray(score, dtype=float)
    target_rank = percentile_rank(target_array)
    return {
        "spearman": asdict(safe_spearman(target_array, score_array)),
        "rank_rmse": asdict(root_mean_squared_error(target_rank, score_array)),
        "prediction_unique": int(np.unique(score_array).size),
    }


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
    }


def _promotion(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usable = [
        fold
        for fold in folds
        if all(
            np.isfinite(float(fold[key]["spearman"]["value"]))
            for key in ("reference", "uniform", "similarity_weighted")
        )
    ]
    if not usable:
        return {"passed": False, "reason": "no_usable_held_studies", "wins": 0, "required_wins": 0}
    reference = np.asarray(
        [float(fold["reference"]["spearman"]["value"]) for fold in usable], dtype=float
    )
    uniform = np.asarray(
        [float(fold["uniform"]["spearman"]["value"]) for fold in usable], dtype=float
    )
    weighted = np.asarray(
        [float(fold["similarity_weighted"]["spearman"]["value"]) for fold in usable], dtype=float
    )
    wins = int(np.sum(weighted > np.maximum(reference, uniform) + _EPS))
    required_wins = len(usable) // 2 + 1
    reference_mean = float(reference.mean())
    uniform_mean = float(uniform.mean())
    weighted_mean = float(weighted.mean())
    reference_min = float(reference.min())
    uniform_min = float(uniform.min())
    weighted_min = float(weighted.min())
    passed = bool(
        weighted_mean > reference_mean + _EPS
        and weighted_mean > uniform_mean + _EPS
        and weighted_min >= reference_min - WORST_STUDY_TOLERANCE
        and weighted_min >= uniform_min - WORST_STUDY_TOLERANCE
        and wins >= required_wins
    )
    return {
        "passed": passed,
        "usable_held_studies": len(usable),
        "wins": wins,
        "required_wins": required_wins,
        "reference_mean": reference_mean,
        "uniform_mean": uniform_mean,
        "similarity_weighted_mean": weighted_mean,
        "reference_min": reference_min,
        "uniform_min": uniform_min,
        "similarity_weighted_min": weighted_min,
        "worst_study_tolerance": WORST_STUDY_TOLERANCE,
    }


def _source_ensemble(
    training: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    dataset: TaskDataset,
    spec: ModelSpec,
    assay_prefix: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_studies = sorted(training["study_group"].astype(str).unique())
    predictions: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for study in source_studies:
        source = training.loc[training["study_group"].astype(str).eq(study)].copy()
        if len(source) < 2:
            failures[study] = "insufficient_source_rows"
            continue
        try:
            predictions[study] = _rank_prediction(
                source,
                target_frame,
                target_column=dataset.target_column,
                spec=spec,
                excluded_columns=dataset.excluded_columns,
            )
            diagnostics[study] = study_distribution_distance(
                source,
                target_frame,
                assay_prefix=assay_prefix,
            )
        except (DataContractError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
            failures[study] = f"{type(error).__name__}: {error}"
    if len(predictions) < 2:
        raise DataContractError(
            f"fewer than two usable source-study experts; failures={failures}"
        )
    names = sorted(predictions)
    matrix = np.column_stack([predictions[name] for name in names])
    uniform = matrix.mean(axis=1)
    distances = {name: float(diagnostics[name]["distance"]) for name in names}
    weights = similarity_weights(distances)
    weighted = matrix @ np.asarray([weights[name] for name in names], dtype=float)
    for name in names:
        diagnostics[name]["weight"] = weights[name]
    return uniform, weighted, {
        "sources": diagnostics,
        "source_failures": failures,
        "source_count": len(names),
        "effective_source_count": float(1.0 / sum(weight * weight for weight in weights.values())),
    }


def evaluate_task_similarity(
    dataset: TaskDataset,
    *,
    spec: ModelSpec,
    assay_prefix: str,
) -> dict[str, Any]:
    require_columns(
        dataset.train,
        [dataset.target_column, "study_group", "subject_group"],
        table_name=f"{dataset.task} study-similarity training frame",
    )
    splits = purged_leave_one_study_out(
        dataset.train["study_group"], dataset.train["subject_group"]
    )
    folds: list[dict[str, Any]] = []
    for split in splits:
        training = dataset.train.iloc[split.train_indices].copy()
        held = dataset.train.iloc[split.validation_indices].copy()
        reference_rank = _rank_prediction(
            training,
            held,
            target_column=dataset.target_column,
            spec=spec,
            excluded_columns=dataset.excluded_columns,
        )
        uniform, weighted, source_diagnostics = _source_ensemble(
            training,
            held,
            dataset=dataset,
            spec=spec,
            assay_prefix=assay_prefix,
        )
        target = pd.to_numeric(held[dataset.target_column], errors="coerce").to_numpy(dtype=float)
        folds.append(
            {
                "held_study": str(split.held_out_group),
                "validation_rows": int(len(held)),
                "training_rows": int(len(training)),
                "purged_subject_count": len(split.purged_subjects),
                "reference": _rank_metrics(target, reference_rank),
                "uniform": _rank_metrics(target, uniform),
                "similarity_weighted": _rank_metrics(target, weighted),
                "source_diagnostics": source_diagnostics,
            }
        )

    summaries: dict[str, Any] = {}
    for condition in ("reference", "uniform", "similarity_weighted"):
        summaries[condition] = _summary(
            [float(fold[condition]["spearman"]["value"]) for fold in folds]
        )
    return {
        "task": dataset.task,
        "selected_model": spec.name,
        "assay_prefix": assay_prefix,
        "held_study_count": len(folds),
        "folds": folds,
        "summary": summaries,
        "promotion": _promotion(folds),
    }


def challenge_similarity(
    dataset: TaskDataset,
    *,
    spec: ModelSpec,
    assay_prefix: str,
) -> dict[str, Any]:
    reference_rank = _rank_prediction(
        dataset.train,
        dataset.challenge,
        target_column=dataset.target_column,
        spec=spec,
        excluded_columns=dataset.excluded_columns,
    )
    uniform, weighted, source_diagnostics = _source_ensemble(
        dataset.train,
        dataset.challenge,
        dataset=dataset,
        spec=spec,
        assay_prefix=assay_prefix,
    )

    def agreement(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
        metric = safe_spearman(left, right)
        difference = np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
        return {
            "rank_spearman": asdict(metric),
            "n": int(len(left)),
            "mean_absolute_percentile_difference": float(np.mean(difference)),
            "max_absolute_percentile_difference": float(np.max(difference)),
        }

    return {
        "rows": int(len(dataset.challenge)),
        "uniform_vs_reference": agreement(uniform, reference_rank),
        "similarity_weighted_vs_reference": agreement(weighted, reference_rank),
        "similarity_weighted_vs_uniform": agreement(weighted, uniform),
        "source_diagnostics": source_diagnostics,
    }


def _build_dataset(config: BaselineConfig, inputs: Any, task: str) -> TaskDataset:
    tables = inputs.tables
    if task == "Task1.1":
        return build_task_11_dataset(
            tables["public_cytokine"],
            tables["challenge_cytokine"],
            tables["participants"],
            tables["investigations"],
        )
    if task == "Task1.2":
        flow = config.section("flow")
        return build_task_12_dataset(
            tables["public_flow"],
            tables["challenge_flow"],
            tables["participants"],
            tables["investigations"],
            mode=str(flow.get("task_12_mode", "broad")),
        )
    raise DataContractError(f"unsupported study-similarity task: {task}")


def run_study_similarity_experiment(config: BaselineConfig, inputs: Any) -> dict[str, Any]:
    """Run the locked Phase A step-6 pseudo-challenge experiment for Task1.1/1.2."""

    tasks: dict[str, Any] = {}
    challenge: dict[str, Any] = {}
    for task, contract in TASK_CONTRACTS.items():
        dataset = _build_dataset(config, inputs, task)
        spec = _find_locked_spec(
            config,
            set_name=contract["model_set"],
            model_name=contract["model_name"],
        )
        tasks[task] = evaluate_task_similarity(
            dataset,
            spec=spec,
            assay_prefix=contract["assay_prefix"],
        )
        challenge[task] = challenge_similarity(
            dataset,
            spec=spec,
            assay_prefix=contract["assay_prefix"],
        )

    return {
        "experiment": "phase_a_study_similarity_weighting",
        "design": {
            "tasks": list(TASK_CONTRACTS),
            "weighting": "source-study expert rank ensemble; 50% shrink to uniform",
            "distance_components": [
                "assay_location",
                "assay_missingness",
                "age",
                "demographic_categorical",
            ],
            "held_target_outcomes_used_for_weights": False,
            "model_selection_retuned": False,
            "worst_study_tolerance": WORST_STUDY_TOLERANCE,
        },
        "tasks": tasks,
        "challenge": challenge,
        "promotion": {task: result["promotion"] for task, result in tasks.items()},
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
    }
