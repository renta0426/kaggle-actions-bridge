"""Phase A step 3: same-readout anchor plus shrunken residual correction.

The experiment is limited to Task1.1 and Task1.2.  It keeps the frozen B2.1
subject-purged held-study splits and compact model families, but changes the
supervised problem from direct response prediction to a correction around the
B1 same-readout anchor on a within-study rank scale.

No participant-level output is returned by the public experiment API.
"""

from __future__ import annotations

from dataclasses import replace
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
from .rank_transfer import within_group_rank_series
from .runner import InputBundle, build_b02_datasets


RESIDUAL_TARGET_COLUMN = "__anchor_rank_residual"
SHRINKAGE_WEIGHTS = (0.25, 0.5, 1.0)
SUPPORTED_TASKS = ("Task1.1", "Task1.2")
TASK_CONTRACT: Mapping[str, tuple[str, str]] = {
    "Task1.1": ("task_11", "cytokine_rank__CXCL10"),
    "Task1.2": ("task_12", "flow_rank__Classical_monocytes"),
}


def _finite_or(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if np.isfinite(number) else fallback


def _task11_direction(
    frame: pd.DataFrame,
    *,
    anchor_column: str,
    target_column: str,
) -> int:
    """Infer Task1.1 anchor direction using training studies only."""

    values: list[float] = []
    for _, group in frame.groupby("study_group", dropna=False, observed=True):
        metric = safe_spearman(group[anchor_column], group[target_column])
        if metric.status == "ok" and metric.value != 0:
            values.append(metric.value)
    if values:
        median = float(np.median(values))
        if median != 0:
            return 1 if median > 0 else -1
        sign_sum = int(np.sign(values).sum())
        if sign_sum:
            return 1 if sign_sum > 0 else -1
    pooled = safe_spearman(frame[anchor_column], frame[target_column])
    if pooled.status == "ok" and pooled.value != 0:
        return 1 if pooled.value > 0 else -1
    return -1


def _anchor_direction(
    task: str,
    frame: pd.DataFrame,
    *,
    anchor_column: str,
    target_column: str,
) -> int:
    if task == "Task1.1":
        return _task11_direction(
            frame,
            anchor_column=anchor_column,
            target_column=target_column,
        )
    if task == "Task1.2":
        return 1
    raise DataContractError(f"unsupported anchor-residual task: {task}")


def _oriented_anchor(frame: pd.DataFrame, *, anchor_column: str, direction: int) -> np.ndarray:
    if direction not in {-1, 1}:
        raise DataContractError(f"anchor direction must be +/-1, found {direction}")
    values = pd.to_numeric(frame[anchor_column], errors="coerce").to_numpy(dtype=float)
    require_finite(values, name=anchor_column)
    # The B2.1 anchor columns are already study-local percentile ranks.  Reverse
    # the unit interval instead of ranking again so ties remain deterministic.
    return values if direction > 0 else 1.0 - values


def _target_rank(frame: pd.DataFrame, *, target_column: str) -> np.ndarray:
    ranked = within_group_rank_series(
        frame,
        group_column="study_group",
        value_column=target_column,
    ).to_numpy(dtype=float)
    require_finite(ranked, name=f"{target_column}.within_study_rank")
    return ranked


def _rank_predictions_within_study(frame: pd.DataFrame, *, score_column: str) -> np.ndarray:
    ranked = within_group_rank_series(
        frame,
        group_column="study_group",
        value_column=score_column,
    ).to_numpy(dtype=float)
    require_finite(ranked, name=f"{score_column}.within_study_rank")
    return ranked


def _score_frame(frame: pd.DataFrame, *, score_column: str) -> Mapping[str, Any]:
    require_columns(
        frame,
        ["study_group", "target", "target_rank", score_column],
        table_name="anchor-residual score frame",
    )
    working = frame.copy()
    working["prediction_rank"] = _rank_predictions_within_study(
        working, score_column=score_column
    )
    folds: list[dict[str, Any]] = []
    for study, group in working.groupby("study_group", dropna=False, observed=True):
        spearman = safe_spearman(group["target"], group[score_column])
        rank_rmse = root_mean_squared_error(group["target_rank"], group["prediction_rank"])
        folds.append(
            {
                "study_group": str(study),
                "n": int(len(group)),
                "spearman": spearman.value,
                "spearman_status": spearman.status,
                "rmse": rank_rmse.value,
                "prediction_unique": int(group[score_column].nunique(dropna=True)),
            }
        )
    fold_frame = pd.DataFrame(folds)
    return {
        "within_group_rank_spearman": within_group_rank_spearman(
            working,
            group_column="study_group",
            target_column="target",
            prediction_column=score_column,
        ).to_dict(),
        "pooled_spearman": safe_spearman(working["target"], working[score_column]).to_dict(),
        "rank_rmse": root_mean_squared_error(
            working["target_rank"], working["prediction_rank"]
        ).to_dict(),
        "folds": folds,
        "fold_summary": summarize_metric_frame(fold_frame),
        "rows": int(len(working)),
        "groups": int(working["study_group"].nunique(dropna=False)),
    }


def _anchor_oof(
    dataset: TaskDataset,
    *,
    splits: Sequence[NamedSplit],
    anchor_column: str,
) -> pd.DataFrame:
    require_columns(
        dataset.train,
        ["study_group", dataset.target_column, anchor_column],
        table_name=f"{dataset.task} anchor-residual training frame",
    )
    parts: list[pd.DataFrame] = []
    for split in splits:
        train = dataset.train.iloc[split.train_indices]
        validation = dataset.train.iloc[split.validation_indices]
        direction = _anchor_direction(
            dataset.task,
            train,
            anchor_column=anchor_column,
            target_column=dataset.target_column,
        )
        parts.append(
            pd.DataFrame(
                {
                    "row_index": split.validation_indices,
                    "study_group": validation["study_group"].astype(str).to_numpy(),
                    "target": pd.to_numeric(
                        validation[dataset.target_column], errors="coerce"
                    ).to_numpy(dtype=float),
                    "target_rank": _target_rank(
                        validation, target_column=dataset.target_column
                    ),
                    "anchor_score": _oriented_anchor(
                        validation,
                        anchor_column=anchor_column,
                        direction=direction,
                    ),
                }
            )
        )
    oof = pd.concat(parts, ignore_index=True)
    if oof["row_index"].duplicated().any():
        raise DataContractError("anchor OOF predicts an original row more than once")
    if set(oof["row_index"].astype(int)) != set(range(len(dataset.train))):
        raise DataContractError("anchor OOF does not cover every training row exactly once")
    return oof


def _identity_spec(spec: ModelSpec) -> ModelSpec:
    return replace(spec, target_transform="identity", clip_min=None)


def _correction_oof(
    dataset: TaskDataset,
    *,
    spec: ModelSpec,
    splits: Sequence[NamedSplit],
    anchor_column: str,
) -> pd.DataFrame:
    residual_spec = _identity_spec(spec)
    excluded = tuple(dict.fromkeys([*dataset.excluded_columns, dataset.target_column]))
    parts: list[pd.DataFrame] = []
    for split in splits:
        train = dataset.train.iloc[split.train_indices].copy()
        validation = dataset.train.iloc[split.validation_indices].copy()
        direction = _anchor_direction(
            dataset.task,
            train,
            anchor_column=anchor_column,
            target_column=dataset.target_column,
        )
        train_anchor = _oriented_anchor(
            train,
            anchor_column=anchor_column,
            direction=direction,
        )
        validation_anchor = _oriented_anchor(
            validation,
            anchor_column=anchor_column,
            direction=direction,
        )
        train_target_rank = _target_rank(train, target_column=dataset.target_column)
        validation_target_rank = _target_rank(
            validation, target_column=dataset.target_column
        )
        train[RESIDUAL_TARGET_COLUMN] = train_target_rank - train_anchor
        _, correction = fit_final_model(
            train,
            validation,
            target_column=RESIDUAL_TARGET_COLUMN,
            spec=residual_spec,
            excluded_columns=excluded,
        )
        require_finite(
            correction,
            name=f"{dataset.task}.{spec.name}.anchor_residual_correction",
        )
        parts.append(
            pd.DataFrame(
                {
                    "row_index": split.validation_indices,
                    "study_group": validation["study_group"].astype(str).to_numpy(),
                    "target": pd.to_numeric(
                        validation[dataset.target_column], errors="coerce"
                    ).to_numpy(dtype=float),
                    "target_rank": validation_target_rank,
                    "anchor_score": validation_anchor,
                    "correction": correction,
                }
            )
        )
    oof = pd.concat(parts, ignore_index=True)
    if oof["row_index"].duplicated().any():
        raise DataContractError("correction OOF predicts an original row more than once")
    if set(oof["row_index"].astype(int)) != set(range(len(dataset.train))):
        raise DataContractError("correction OOF does not cover every training row exactly once")
    return oof


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, float, float, float, str]:
    metrics = candidate["metrics"]
    summary = metrics["fold_summary"]
    return (
        _finite_or(summary.get("spearman_mean"), -np.inf),
        _finite_or(summary.get("spearman_median"), -np.inf),
        _finite_or(summary.get("spearman_min"), -np.inf),
        _finite_or(metrics["within_group_rank_spearman"].get("value"), -np.inf),
        -_finite_or(metrics["rank_rmse"].get("value"), np.inf),
        -float(candidate["lambda"]),
        str(candidate["model"]),
    )


def _challenge_agreement(left: pd.DataFrame, right: pd.DataFrame) -> Mapping[str, Any]:
    merged = left.merge(
        right,
        on="participant_id",
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise DataContractError("challenge agreement frames do not align one-to-one")
    left_rank = percentile_rank(merged["prediction_left"].to_numpy(dtype=float))
    right_rank = percentile_rank(merged["prediction_right"].to_numpy(dtype=float))
    return {
        "rows": int(len(merged)),
        "rank_spearman": safe_spearman(left_rank, right_rank).to_dict(),
        "mean_absolute_percentile_rank_difference": float(
            np.mean(np.abs(left_rank - right_rank))
        ),
        "max_absolute_percentile_rank_difference": float(
            np.max(np.abs(left_rank - right_rank))
        ),
    }


def run_anchor_residual_task(
    dataset: TaskDataset,
    *,
    specs: Sequence[ModelSpec],
    splits: Sequence[NamedSplit],
    anchor_column: str,
    random_state: int,
) -> Mapping[str, Any]:
    """Run one task's predeclared anchor + residual candidate family."""

    anchor_oof = _anchor_oof(dataset, splits=splits, anchor_column=anchor_column)
    anchor_metrics = _score_frame(anchor_oof, score_column="anchor_score")
    candidates: list[dict[str, Any]] = [
        {
            "model": "anchor_only",
            "lambda": 0.0,
            "metrics": anchor_metrics,
            "oof": anchor_oof,
            "spec": None,
        }
    ]
    failures: dict[str, str] = {}
    for spec in specs:
        try:
            correction_oof = _correction_oof(
                dataset,
                spec=spec,
                splits=splits,
                anchor_column=anchor_column,
            )
            for weight in SHRINKAGE_WEIGHTS:
                scored = correction_oof.copy()
                scored["combined_score"] = (
                    scored["anchor_score"] + float(weight) * scored["correction"]
                )
                candidates.append(
                    {
                        "model": spec.name,
                        "lambda": float(weight),
                        "metrics": _score_frame(scored, score_column="combined_score"),
                        "oof": scored,
                        "spec": _identity_spec(spec),
                    }
                )
        except (DataContractError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
            failures[spec.name] = f"{type(error).__name__}: {error}"

    ranked = sorted(candidates, key=_candidate_key, reverse=True)
    selected = ranked[0]

    all_direction = _anchor_direction(
        dataset.task,
        dataset.train,
        anchor_column=anchor_column,
        target_column=dataset.target_column,
    )
    challenge_anchor = _oriented_anchor(
        dataset.challenge,
        anchor_column=anchor_column,
        direction=all_direction,
    )
    b1_challenge = dataset.challenge[["participant_id"]].copy()
    b1_challenge["prediction"] = challenge_anchor

    selected_lambda = float(selected["lambda"])
    if selected_lambda == 0.0:
        selected_challenge = b1_challenge.copy()
    else:
        selected_spec = selected["spec"]
        if not isinstance(selected_spec, ModelSpec):
            raise DataContractError("selected residual candidate is missing a model spec")
        final_train = dataset.train.copy()
        train_anchor = _oriented_anchor(
            final_train,
            anchor_column=anchor_column,
            direction=all_direction,
        )
        final_train[RESIDUAL_TARGET_COLUMN] = (
            _target_rank(final_train, target_column=dataset.target_column) - train_anchor
        )
        excluded = tuple(dict.fromkeys([*dataset.excluded_columns, dataset.target_column]))
        _, correction = fit_final_model(
            final_train,
            dataset.challenge,
            target_column=RESIDUAL_TARGET_COLUMN,
            spec=selected_spec,
            excluded_columns=excluded,
        )
        selected_challenge = dataset.challenge[["participant_id"]].copy()
        selected_challenge["prediction"] = challenge_anchor + selected_lambda * correction

    b21 = run_compact_task(
        dataset,
        specs=specs,
        splits=splits,
        random_state=random_state,
        selection_policy="robust_v1",
    )
    b21_source = dataset.train[["study_group"]].reset_index().rename(
        columns={"index": "row_index"}
    )
    b21_oof = b21.oof_predictions.merge(
        b21_source,
        on="row_index",
        how="left",
        validate="one_to_one",
    ).copy()
    b21_oof["target_rank"] = _target_rank(
        dataset.train.loc[b21_oof["row_index"].to_numpy(dtype=int)].reset_index(drop=True),
        target_column=dataset.target_column,
    )
    # `_target_rank` above is safe for LOSO tasks because row_index is one-to-one,
    # but restore study-local ranking from the aligned OOF frame to avoid relying
    # on row order if a future dataset builder changes it.
    b21_oof["target_rank"] = within_group_rank_series(
        b21_oof,
        group_column="study_group",
        value_column="target",
    ).to_numpy(dtype=float)
    b21_metrics = _score_frame(
        b21_oof.rename(columns={"prediction": "b21_score"}),
        score_column="b21_score",
    )

    candidate_summaries = [
        {
            "model": str(item["model"]),
            "lambda": float(item["lambda"]),
            "within_group_rank_spearman": item["metrics"]["within_group_rank_spearman"],
            "pooled_spearman": item["metrics"]["pooled_spearman"],
            "rank_rmse": item["metrics"]["rank_rmse"],
            "fold_summary": item["metrics"]["fold_summary"],
        }
        for item in candidates
    ]

    return {
        "anchor_column": anchor_column,
        "b1_reference": {
            **anchor_metrics,
            "challenge_rows": int(len(b1_challenge)),
        },
        "raw_b21_reference": {
            **b21_metrics,
            "selected_model": b21.selected_spec.name,
        },
        "selected": {
            "model": str(selected["model"]),
            "lambda": selected_lambda,
            **selected["metrics"],
            "challenge_agreement_vs_b1": _challenge_agreement(
                selected_challenge, b1_challenge
            ),
            "challenge_agreement_vs_raw_b21": _challenge_agreement(
                selected_challenge, b21.challenge_predictions
            ),
        },
        "candidate_summaries": candidate_summaries,
        "candidate_failures": failures,
        "selection_policy": "robust_anchor_residual_v1",
        "shrinkage_weights": [0.0, *SHRINKAGE_WEIGHTS],
    }


def run_anchor_residual_experiment(
    config: BaselineConfig,
    inputs: InputBundle,
) -> Mapping[str, Any]:
    """Run the pre-registered Task1.1/Task1.2 anchor-residual experiment."""

    if config.baseline != "b021_taskwise_robust":
        raise DataContractError("anchor-residual experiment requires B2.1 config")
    if str(config.section("selection").get("policy", "")) != "robust_v1":
        raise DataContractError("anchor-residual experiment requires selection.policy=robust_v1")
    flow = config.section("flow")
    if str(flow.get("task_12_mode")) != "broad":
        raise DataContractError("anchor-residual Task1.2 requires B2.1 broad-flow features")

    datasets = build_b02_datasets(config, inputs)
    payload: dict[str, Any] = {}
    for task, (model_set, anchor_column) in TASK_CONTRACT.items():
        dataset = datasets[task]
        if not isinstance(dataset, TaskDataset):
            raise DataContractError(f"{task} is not a compact TaskDataset")
        if anchor_column not in dataset.train or anchor_column not in dataset.challenge:
            raise DataContractError(f"{task} same-readout anchor missing: {anchor_column}")
        splits = default_splits_for_task(dataset, random_state=config.random_state)
        payload[task] = run_anchor_residual_task(
            dataset,
            specs=config.model_specs(model_set),
            splits=splits,
            anchor_column=anchor_column,
            random_state=config.random_state,
        )

    return {
        "schema_version": 1,
        "experiment": "phase_a_anchor_residual_task11_task12",
        "selection_policy": "robust_anchor_residual_v1",
        "target_estimand": "within_study_percentile_rank_residual_from_b1_anchor",
        "shrinkage_weights": [0.0, *SHRINKAGE_WEIGHTS],
        "tasks": payload,
        "notes": [
            "Task1.1 anchor direction is inferred separately inside each held-study fold using training studies only.",
            "Task1.2 reuses the B2.1 broad-flow study-rank feature space; no heterogeneous raw flow units are combined.",
            "Residual models reuse the frozen B2.1 compact model families and hyperparameters with identity target transform.",
            "Rank RMSE is computed after converting prediction scores to within-study percentile ranks, avoiding unbounded score-scale artifacts.",
            "This experiment creates no Kaggle Competition submission.",
        ],
    }
