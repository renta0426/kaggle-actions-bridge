"""Phase A step 2: within-study rank and transductive feature normalization.

The experiment is intentionally limited to Task1.1 and Task1.2.  It keeps the
frozen B2.1 split/model-family contract as a reference, then asks whether the
competition estimand is better learned after removing historical study scale.
Held-out study baseline-X distributions may be used for feature ranking because
the real 2025 baseline cohort is fully visible; held-out outcomes are never used
to fit a feature transform or model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns, require_finite
from .cv import NamedSplit
from .datasets import TaskDataset
from .evaluation import default_splits_for_task, run_compact_task
from .metrics import (
    percentile_rank,
    root_mean_squared_error,
    safe_spearman,
    within_group_rank_spearman,
)
from .models import ModelSpec, fit_final_model, summarize_metric_frame
from .runner import InputBundle, build_b02_datasets


RANK_TARGET_COLUMN = "__within_study_target_rank"
SUPPORTED_TASKS = ("Task1.1", "Task1.2")
VARIANTS = ("target_rank_raw", "target_rank_rank_only", "target_rank_raw_plus_rank")


@dataclass
class RankTransferRun:
    task: str
    variant: str
    selected_spec: ModelSpec
    oof_predictions: pd.DataFrame
    challenge_predictions: pd.DataFrame
    metrics: Mapping[str, Any]
    candidate_summaries: Sequence[Mapping[str, Any]]
    candidate_failures: Mapping[str, str]


def _rank_finite(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    if finite.any():
        finite_index = values.index[finite]
        result.loc[finite_index] = percentile_rank(numeric.loc[finite_index].to_numpy(dtype=float))
    return result


def within_group_rank_series(
    frame: pd.DataFrame,
    *,
    group_column: str,
    value_column: str,
) -> pd.Series:
    """Rank finite values inside each observed group while preserving missingness."""

    require_columns(frame, [group_column, value_column], table_name="rank-transform frame")
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby(group_column, dropna=False, observed=True):
        result.loc[group.index] = _rank_finite(group[value_column])
    return result


def add_rank_target(
    frame: pd.DataFrame,
    *,
    target_column: str,
    group_column: str = "study_group",
) -> pd.DataFrame:
    """Add a study-local target rank.  All supervised targets must be finite."""

    working = frame.copy()
    target = pd.to_numeric(working[target_column], errors="coerce").to_numpy(dtype=float)
    require_finite(target, name=target_column)
    working[RANK_TARGET_COLUMN] = within_group_rank_series(
        working,
        group_column=group_column,
        value_column=target_column,
    )
    require_finite(working[RANK_TARGET_COLUMN].to_numpy(dtype=float), name=RANK_TARGET_COLUMN)
    return working


def transform_assay_features(
    frame: pd.DataFrame,
    *,
    assay_prefix: str,
    mode: str,
    group_column: str = "study_group",
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Apply a label-free study-local rank transform to assay features.

    `rank_only` replaces each assay feature by its within-study percentile rank.
    `raw_plus_rank` retains the raw feature and appends a `__study_rank` feature.
    `raw` is returned unchanged and is useful for common evaluation plumbing.
    """

    if mode not in {"raw", "rank_only", "raw_plus_rank"}:
        raise DataContractError(f"unknown rank feature mode: {mode}")
    require_columns(frame, [group_column], table_name="feature-rank frame")
    assay_columns = tuple(column for column in frame.columns if column.startswith(assay_prefix))
    if not assay_columns:
        raise DataContractError(f"no assay features found for prefix {assay_prefix!r}")
    working = frame.copy()
    if mode == "raw":
        return working, assay_columns

    created: list[str] = []
    for column in assay_columns:
        ranked = within_group_rank_series(
            working,
            group_column=group_column,
            value_column=column,
        )
        if mode == "rank_only":
            working[column] = ranked
        else:
            rank_column = f"{column}__study_rank"
            if rank_column in working:
                raise DataContractError(f"rank feature already exists: {rank_column}")
            working[rank_column] = ranked
            created.append(rank_column)
    return working, tuple(created if mode == "raw_plus_rank" else assay_columns)


def _rank_spec(spec: ModelSpec) -> ModelSpec:
    """Keep model family/hyperparameters but fit directly to a [0,1] rank target."""

    return replace(spec, target_transform="identity", clip_min=None)


def _fold_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        oof,
        ["split", "target", "target_rank", "prediction"],
        table_name="rank-transfer OOF",
    )
    rows: list[dict[str, Any]] = []
    for split, group in oof.groupby("split", dropna=False, observed=True):
        spearman = safe_spearman(group["target"], group["prediction"])
        rank_rmse = root_mean_squared_error(group["target_rank"], group["prediction"])
        rows.append(
            {
                "split": split,
                "n": int(len(group)),
                "spearman": spearman.value,
                "spearman_status": spearman.status,
                "rmse": rank_rmse.value,
                "prediction_unique": int(group["prediction"].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def _pooled_rank_rmse(oof: pd.DataFrame) -> Mapping[str, Any]:
    return root_mean_squared_error(oof["target_rank"], oof["prediction"]).to_dict()


def _evaluate_spec(
    dataset: TaskDataset,
    *,
    spec: ModelSpec,
    splits: Sequence[NamedSplit],
    assay_prefix: str,
    feature_mode: str,
) -> Mapping[str, Any]:
    rank_spec = _rank_spec(spec)
    parts: list[pd.DataFrame] = []
    excluded = tuple(dict.fromkeys([*dataset.excluded_columns, dataset.target_column]))

    for split in splits:
        train = dataset.train.iloc[split.train_indices].copy()
        validation = dataset.train.iloc[split.validation_indices].copy()

        train, _ = transform_assay_features(
            train,
            assay_prefix=assay_prefix,
            mode=feature_mode,
        )
        validation, _ = transform_assay_features(
            validation,
            assay_prefix=assay_prefix,
            mode=feature_mode,
        )
        train = add_rank_target(train, target_column=dataset.target_column)
        validation_rank = add_rank_target(validation, target_column=dataset.target_column)

        _, prediction = fit_final_model(
            train,
            validation,
            target_column=RANK_TARGET_COLUMN,
            spec=rank_spec,
            excluded_columns=excluded,
        )
        require_finite(prediction, name=f"{dataset.task}.{feature_mode}.{spec.name} prediction")
        parts.append(
            pd.DataFrame(
                {
                    "row_index": split.validation_indices,
                    "split": split.name,
                    "held_out_group": split.held_out_group,
                    "study_group": validation["study_group"].astype(str).to_numpy(),
                    "target": pd.to_numeric(validation[dataset.target_column], errors="coerce").to_numpy(dtype=float),
                    "target_rank": validation_rank[RANK_TARGET_COLUMN].to_numpy(dtype=float),
                    "prediction": prediction,
                }
            )
        )

    oof = pd.concat(parts, ignore_index=True)
    if oof.empty:
        raise DataContractError("rank-transfer OOF predictions are empty")
    if oof["row_index"].duplicated().any():
        raise DataContractError(
            "rank-transfer currently requires one held-study prediction per original row"
        )
    if set(oof["row_index"].astype(int)) != set(range(len(dataset.train))):
        raise DataContractError("rank-transfer OOF does not cover every training row exactly once")

    folds = _fold_metrics(oof)
    within = within_group_rank_spearman(
        oof,
        group_column="study_group",
        target_column="target",
        prediction_column="prediction",
    ).to_dict()
    pooled = safe_spearman(oof["target"], oof["prediction"]).to_dict()
    rank_rmse = _pooled_rank_rmse(oof)
    return {
        "spec": rank_spec,
        "oof": oof,
        "fold_metrics": folds,
        "metrics": {
            "within_group_rank_spearman": within,
            "pooled_spearman": pooled,
            "rank_rmse": rank_rmse,
            "fold_summary": summarize_metric_frame(folds),
            "rows": int(len(oof)),
            "groups": int(oof["study_group"].nunique(dropna=False)),
        },
    }


def _finite_or(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if np.isfinite(number) else fallback


def _selection_key(item: Mapping[str, Any]) -> tuple[float, float, float, float, float, str]:
    metrics = item["metrics"]
    summary = metrics["fold_summary"]
    return (
        _finite_or(summary.get("spearman_mean"), -np.inf),
        _finite_or(summary.get("spearman_median"), -np.inf),
        _finite_or(summary.get("spearman_min"), -np.inf),
        _finite_or(metrics["within_group_rank_spearman"].get("value"), -np.inf),
        -_finite_or(metrics["rank_rmse"].get("value"), np.inf),
        str(item["spec"].name),
    )


def run_rank_variant(
    dataset: TaskDataset,
    *,
    specs: Sequence[ModelSpec],
    splits: Sequence[NamedSplit],
    assay_prefix: str,
    variant: str,
) -> RankTransferRun:
    if variant not in VARIANTS:
        raise DataContractError(f"unknown rank-transfer variant: {variant}")
    feature_mode = {
        "target_rank_raw": "raw",
        "target_rank_rank_only": "rank_only",
        "target_rank_raw_plus_rank": "raw_plus_rank",
    }[variant]

    evaluations: list[Mapping[str, Any]] = []
    failures: dict[str, str] = {}
    for spec in specs:
        try:
            evaluations.append(
                _evaluate_spec(
                    dataset,
                    spec=spec,
                    splits=splits,
                    assay_prefix=assay_prefix,
                    feature_mode=feature_mode,
                )
            )
        except (DataContractError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
            failures[spec.name] = f"{type(error).__name__}: {error}"
    if not evaluations:
        raise DataContractError(f"all {dataset.task} {variant} candidates failed: {failures}")
    ranked = sorted(evaluations, key=_selection_key, reverse=True)
    selected = ranked[0]
    selected_spec = selected["spec"]

    final_train, _ = transform_assay_features(
        dataset.train,
        assay_prefix=assay_prefix,
        mode=feature_mode,
    )
    final_challenge, _ = transform_assay_features(
        dataset.challenge,
        assay_prefix=assay_prefix,
        mode=feature_mode,
    )
    final_train = add_rank_target(final_train, target_column=dataset.target_column)
    excluded = tuple(dict.fromkeys([*dataset.excluded_columns, dataset.target_column]))
    _, challenge_prediction = fit_final_model(
        final_train,
        final_challenge,
        target_column=RANK_TARGET_COLUMN,
        spec=selected_spec,
        excluded_columns=excluded,
    )
    challenge = dataset.challenge[["participant_id"]].copy()
    challenge["prediction"] = challenge_prediction

    candidate_summaries = [
        {
            "model": item["spec"].to_dict(),
            **item["metrics"],
        }
        for item in evaluations
    ]
    return RankTransferRun(
        task=dataset.task,
        variant=variant,
        selected_spec=selected_spec,
        oof_predictions=selected["oof"],
        challenge_predictions=challenge,
        metrics={
            **selected["metrics"],
            "selection_policy": "robust_v1_rank_target",
            "feature_mode": feature_mode,
            "assay_prefix": assay_prefix,
            "split_count": len(splits),
            "training_rows": len(dataset.train),
            "training_subjects": int(dataset.train["subject_group"].nunique()),
            "training_studies": int(dataset.train["study_group"].nunique()),
        },
        candidate_summaries=candidate_summaries,
        candidate_failures=failures,
    )


def _reference_metrics(dataset: TaskDataset, result: Any) -> Mapping[str, Any]:
    source = dataset.train[["study_group"]].reset_index().rename(columns={"index": "row_index"})
    oof = result.oof_predictions.merge(source, on="row_index", how="left", validate="one_to_one")
    within = within_group_rank_spearman(
        oof,
        group_column="study_group",
        target_column="target",
        prediction_column="prediction",
    ).to_dict()
    return {
        "selected_model": result.selected_spec.name,
        "within_group_rank_spearman": within,
        "pooled_spearman": result.metrics["pooled"]["spearman"],
        "fold_summary": result.metrics["fold_summary"],
        "rows": int(len(oof)),
        "groups": int(oof["study_group"].nunique(dropna=False)),
    }


def _challenge_agreement(left: pd.DataFrame, right: pd.DataFrame) -> Mapping[str, Any]:
    merged = left.merge(right, on="participant_id", how="inner", suffixes=("_left", "_right"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise DataContractError("challenge agreement frames do not align one-to-one")
    left_rank = percentile_rank(merged["prediction_left"].to_numpy(dtype=float))
    right_rank = percentile_rank(merged["prediction_right"].to_numpy(dtype=float))
    return {
        "rows": int(len(merged)),
        "rank_spearman": safe_spearman(left_rank, right_rank).to_dict(),
        "mean_absolute_percentile_rank_difference": float(np.mean(np.abs(left_rank - right_rank))),
        "max_absolute_percentile_rank_difference": float(np.max(np.abs(left_rank - right_rank))),
    }


def run_rank_transfer_experiment(
    config: BaselineConfig,
    inputs: InputBundle,
) -> Mapping[str, Any]:
    """Run the pre-registered Task1.1/Task1.2 rank-transfer comparison."""

    if config.baseline != "b021_taskwise_robust":
        raise DataContractError("rank-transfer experiment requires the frozen B2.1 config")
    if str(config.section("selection").get("policy", "")) != "robust_v1":
        raise DataContractError("rank-transfer experiment requires selection.policy=robust_v1")

    datasets = build_b02_datasets(config, inputs)
    task_contract = {
        "Task1.1": ("task_11", "cytokine_"),
        "Task1.2": ("task_12", "flow_"),
    }
    payload: dict[str, Any] = {}
    for task, (model_set, assay_prefix) in task_contract.items():
        dataset = datasets[task]
        if not isinstance(dataset, TaskDataset):
            raise DataContractError(f"{task} is not a compact TaskDataset")
        splits = default_splits_for_task(dataset, random_state=config.random_state)
        specs = config.model_specs(model_set)
        reference = run_compact_task(
            dataset,
            specs=specs,
            splits=splits,
            random_state=config.random_state,
            selection_policy="robust_v1",
        )
        variants = {
            variant: run_rank_variant(
                dataset,
                specs=specs,
                splits=splits,
                assay_prefix=assay_prefix,
                variant=variant,
            )
            for variant in VARIANTS
        }
        payload[task] = {
            "raw_b21_reference": _reference_metrics(dataset, reference),
            "rank_variants": {
                variant: {
                    "selected_model": run.selected_spec.name,
                    **run.metrics,
                    "candidate_failures": dict(run.candidate_failures),
                }
                for variant, run in variants.items()
            },
            "challenge_agreement_vs_raw_b21": {
                variant: _challenge_agreement(reference.challenge_predictions, run.challenge_predictions)
                for variant, run in variants.items()
            },
        }

    return {
        "schema_version": 1,
        "experiment": "phase_a_rank_transfer_task11_task12",
        "selection_policy": "robust_v1",
        "target_estimand": "within_study_percentile_rank",
        "transductive_contract": (
            "Held-out study and 2025 assay-feature ranks use baseline X from that study only; held-out outcomes are never used to fit feature transforms or models."
        ),
        "tasks": payload,
        "notes": [
            "The rank-target variants reuse B2.1 model families/hyperparameters but set target_transform=identity and remove raw-scale clipping.",
            "Feature ranking is limited to task assay features (cytokine_ for Task1.1; flow_ for Task1.2); demographics remain on their original representation.",
            "This experiment does not create or submit a Kaggle submission.",
        ],
    }
