"""Strategy-v2 E11a: finite Task1.1 pairwise-ranking comparison.

E11a is deliberately narrow. It compares one pre-registered nonlinear pairwise
ranker against the frozen B2.1 Task1.1 PLS-2 incumbent on exactly the same
subject-purged outer folds. Training pairs are created only within source
studies so that cross-study target scale is never used as a pair label.

No held-study outcome is accepted by the scoring API. A held/challenge score is
the study-equal mean probability that the item outranks source-study reference
rows. The first gate is CPU feasibility and leakage safety; real-data execution
remains a separate protected bridge action.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns, require_finite
from .datasets import TaskDataset
from .evaluation import default_splits_for_task
from .metrics import percentile_rank, safe_spearman
from .models import ModelSpec, build_estimator, fit_final_model, prepare_model_frame
from .runner import InputBundle, build_b02_datasets

EXPERIMENT = "strategy_v2_e11a_task11_pairwise_ranker"
COMPARISON_CONTRACT = "paired_subject_purged_v2"
TASK = "Task1.1"
BASE_MODEL_SET = "task_11"
BASE_MODEL = "pls_2"
EXPECTED_TRAIN_ROWS = 127
EXPECTED_TRAIN_STUDIES = 4
EXPECTED_CHALLENGE_ROWS = 40
MAX_ROWS_PER_STUDY = 80
MAX_TRANSFORMED_FEATURES = 512
MIN_DIRECTED_PAIRS = 100
PAIR_TIE_TOLERANCE = 1e-12
PAIRWISE_PARAMS: Mapping[str, Any] = {
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 10,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 20260910,
}
PROMOTION_MEAN_DELTA = 0.02
PROMOTION_MIN_STUDY_DELTA = -0.10


@dataclass
class PairwiseRankModel:
    preprocessor: Any
    classifier: HistGradientBoostingClassifier
    feature_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    references_by_study: Mapping[str, np.ndarray]
    fit_audit: Mapping[str, Any]


def _find_base_spec(config: BaselineConfig) -> ModelSpec:
    matches = [spec for spec in config.model_specs(BASE_MODEL_SET) if spec.name == BASE_MODEL]
    if len(matches) != 1:
        raise DataContractError(
            f"E11a frozen base {BASE_MODEL!r} resolved to {len(matches)} specs"
        )
    return matches[0]


def _feature_frames(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    dataset: TaskDataset,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    excluded = tuple(
        dict.fromkeys([*dataset.excluded_columns, dataset.target_column, "study_group"])
    )
    train_x, numeric, categorical = prepare_model_frame(
        training,
        excluded_columns=excluded,
    )
    pred_x, _, _ = prepare_model_frame(
        prediction,
        excluded_columns=excluded,
    )
    missing = [column for column in train_x.columns if column not in pred_x]
    extra = [column for column in pred_x.columns if column not in train_x]
    if missing or extra:
        raise DataContractError(
            f"E11a train/prediction feature mismatch:missing={missing},extra={extra}"
        )
    pred_x = pred_x[train_x.columns]
    if "study_group" in train_x or "study_accession" in train_x:
        raise DataContractError("E11a study identity escaped feature exclusions")
    return train_x, pred_x, numeric, categorical


def build_within_study_pairs(
    transformed: np.ndarray,
    target: Sequence[float],
    studies: Sequence[object],
    subjects: Sequence[object],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create symmetric directed pairs using source-study outcomes only."""
    x = np.asarray(transformed, dtype=float)
    y = np.asarray(target, dtype=float)
    study = pd.Series(studies, dtype="string")
    subject = pd.Series(subjects, dtype="string")
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(study) or len(study) != len(subject):
        raise DataContractError("E11a pair input shapes differ")
    require_finite(x, name="E11a.transformed_features")
    require_finite(y, name="E11a.target")
    if study.isna().any() or subject.isna().any():
        raise DataContractError("E11a pair metadata contains missing study/subject")
    study_values = study.astype(str).to_numpy()
    subject_values = subject.astype(str).to_numpy()

    pair_x: list[np.ndarray] = []
    pair_y: list[int] = []
    unordered_by_study: dict[str, int] = {}
    ties_by_study: dict[str, int] = {}
    skipped_same_subject_by_study: dict[str, int] = {}

    for name in sorted(set(study_values)):
        indices = np.flatnonzero(study_values == name)
        if len(indices) > MAX_ROWS_PER_STUDY:
            raise DataContractError(
                f"E11a source study {name} has {len(indices)} > {MAX_ROWS_PER_STUDY} rows"
            )
        unordered = 0
        ties = 0
        same_subject = 0
        for left, right in combinations(indices.tolist(), 2):
            if subject_values[left] == subject_values[right]:
                same_subject += 1
                continue
            delta = float(y[left] - y[right])
            if abs(delta) <= PAIR_TIE_TOLERANCE:
                ties += 1
                continue
            difference = x[left] - x[right]
            label = int(delta > 0.0)
            pair_x.append(difference)
            pair_y.append(label)
            pair_x.append(-difference)
            pair_y.append(1 - label)
            unordered += 1
        unordered_by_study[name] = unordered
        ties_by_study[name] = ties
        skipped_same_subject_by_study[name] = same_subject

    if not pair_x:
        raise DataContractError("E11a produced no usable within-study pairs")
    design = np.asarray(pair_x, dtype=float)
    labels = np.asarray(pair_y, dtype=int)
    require_finite(design, name="E11a.pair_design")
    if design.shape[0] < MIN_DIRECTED_PAIRS:
        raise DataContractError(
            f"E11a has {design.shape[0]} < {MIN_DIRECTED_PAIRS} directed pairs"
        )
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives != negatives:
        raise DataContractError("E11a symmetric directed pairs are class-imbalanced")
    return design, labels, {
        "pairs_within_study_only": True,
        "directed_pairs": int(len(labels)),
        "unordered_pairs": int(len(labels) // 2),
        "positive_pairs": positives,
        "negative_pairs": negatives,
        "unordered_pairs_by_study": unordered_by_study,
        "ties_skipped_by_study": ties_by_study,
        "same_subject_pairs_skipped_by_study": skipped_same_subject_by_study,
        "cross_study_pair_labels_used": False,
    }


def fit_pairwise_ranker(
    training: pd.DataFrame,
    *,
    dataset: TaskDataset,
) -> PairwiseRankModel:
    """Fit one nonlinear pairwise ranker using training rows only."""
    require_columns(
        training,
        [dataset.target_column, "study_group", "subject_group"],
        table_name="E11a pairwise training",
    )
    train_x, _, numeric, categorical = _feature_frames(
        training,
        training.iloc[:1].copy(),
        dataset=dataset,
    )
    preprocessor = build_estimator(
        ModelSpec(name="e11a_preprocessor", family="ridge", params={"alpha": 1.0}),
        numeric_columns=numeric,
        categorical_columns=categorical,
    ).named_steps["preprocessor"]
    transformed = np.asarray(preprocessor.fit_transform(train_x), dtype=float)
    require_finite(transformed, name="E11a.preprocessed_training")
    if transformed.shape[1] > MAX_TRANSFORMED_FEATURES:
        raise DataContractError(
            f"E11a transformed features {transformed.shape[1]} > {MAX_TRANSFORMED_FEATURES}"
        )

    target = pd.to_numeric(
        training[dataset.target_column], errors="coerce"
    ).to_numpy(dtype=float)
    pair_x, pair_y, pair_audit = build_within_study_pairs(
        transformed,
        target,
        training["study_group"],
        training["subject_group"],
    )
    classifier = HistGradientBoostingClassifier(**dict(PAIRWISE_PARAMS))
    started = time.perf_counter()
    classifier.fit(pair_x, pair_y)
    fit_seconds = float(time.perf_counter() - started)

    studies = training["study_group"].astype(str).to_numpy()
    references = {
        name: transformed[studies == name].copy()
        for name in sorted(set(studies))
    }
    if len(references) < 2:
        raise DataContractError("E11a requires at least two source studies")
    return PairwiseRankModel(
        preprocessor=preprocessor,
        classifier=classifier,
        feature_columns=tuple(train_x.columns),
        numeric_columns=tuple(numeric),
        categorical_columns=tuple(categorical),
        references_by_study=references,
        fit_audit={
            **pair_audit,
            "training_rows": int(len(training)),
            "training_subjects": int(training["subject_group"].nunique()),
            "training_studies": int(training["study_group"].nunique()),
            "transformed_features": int(transformed.shape[1]),
            "numeric_features": int(len(numeric)),
            "categorical_features": int(len(categorical)),
            "reference_aggregation": "study_equal",
            "held_outcomes_used_for_fit": False,
            "fit_seconds": fit_seconds,
            "classifier": "HistGradientBoostingClassifier",
            "classifier_params": dict(PAIRWISE_PARAMS),
        },
    )


def score_pairwise_ranker(
    model: PairwiseRankModel,
    prediction: pd.DataFrame,
    *,
    dataset: TaskDataset,
) -> np.ndarray:
    """Score prediction rows without accepting outcomes."""
    pred_x, _, _ = prepare_model_frame(
        prediction,
        excluded_columns=tuple(
            dict.fromkeys([*dataset.excluded_columns, dataset.target_column, "study_group"])
        ),
    )
    missing = [column for column in model.feature_columns if column not in pred_x]
    extra = [column for column in pred_x.columns if column not in model.feature_columns]
    if missing or extra:
        raise DataContractError(
            f"E11a scoring feature mismatch:missing={missing},extra={extra}"
        )
    pred_x = pred_x[list(model.feature_columns)]
    transformed = np.asarray(model.preprocessor.transform(pred_x), dtype=float)
    require_finite(transformed, name="E11a.preprocessed_prediction")

    scores = np.empty(len(transformed), dtype=float)
    for row_index, row in enumerate(transformed):
        study_means: list[float] = []
        for name in sorted(model.references_by_study):
            references = np.asarray(model.references_by_study[name], dtype=float)
            differences = row.reshape(1, -1) - references
            probabilities = np.asarray(
                model.classifier.predict_proba(differences)[:, 1],
                dtype=float,
            )
            require_finite(probabilities, name=f"E11a.win_probability.{name}")
            study_means.append(float(np.mean(probabilities)))
        scores[row_index] = float(np.mean(study_means))
    require_finite(scores, name="E11a.rank_scores")
    return scores


def _base_score(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    dataset: TaskDataset,
    spec: ModelSpec,
) -> np.ndarray:
    _, values = fit_final_model(
        training,
        prediction,
        target_column=dataset.target_column,
        spec=spec,
        excluded_columns=dataset.excluded_columns,
    )
    values = np.asarray(values, dtype=float)
    require_finite(values, name="E11a.base_score")
    return values


def _spearman_value(target: Sequence[float], score: Sequence[float]) -> float | None:
    metric = safe_spearman(target, score)
    return float(metric.value) if metric.status == "ok" else None


def _promotion(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [
        float(fold["pairwise_delta"])
        for fold in folds
        if fold.get("pairwise_delta") is not None
        and np.isfinite(float(fold["pairwise_delta"]))
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
    return {
        "passed": bool(
            mean_delta >= PROMOTION_MEAN_DELTA
            and minimum_delta >= PROMOTION_MIN_STUDY_DELTA
            and wins >= required
        ),
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
        "mean_absolute_percentile_shift": float(
            np.mean(np.abs(base_rank - candidate_rank))
        ),
        "max_absolute_percentile_shift": float(
            np.max(np.abs(base_rank - candidate_rank))
        ),
    }


def audit_e11a_feasibility(dataset: TaskDataset) -> dict[str, Any]:
    """Outcome-safe structural gate run before any real E11a comparison."""
    dataset.validate()
    if dataset.task != TASK:
        raise DataContractError(f"E11a only supports {TASK}")
    if len(dataset.train) != EXPECTED_TRAIN_ROWS:
        raise DataContractError(
            f"E11a expected {EXPECTED_TRAIN_ROWS} train rows, found {len(dataset.train)}"
        )
    if dataset.train["study_group"].nunique() != EXPECTED_TRAIN_STUDIES:
        raise DataContractError("E11a Task1.1 study count changed")
    if len(dataset.challenge) != EXPECTED_CHALLENGE_ROWS:
        raise DataContractError("E11a Challenge row count changed")
    counts = (
        dataset.train["study_group"].astype(str).value_counts().sort_index().to_dict()
    )
    if max(int(value) for value in counts.values()) > MAX_ROWS_PER_STUDY:
        raise DataContractError("E11a exhaustive within-study pairing exceeds row guard")
    splits = default_splits_for_task(dataset)
    split_audit = []
    for split in splits:
        train = dataset.train.iloc[split.train_indices]
        held = dataset.train.iloc[split.validation_indices]
        train_subjects = set(train["subject_group"].astype(str))
        held_subjects = set(held["subject_group"].astype(str))
        if train_subjects.intersection(held_subjects):
            raise DataContractError(f"E11a subject leakage:{split.name}")
        if held["study_group"].nunique() != 1:
            raise DataContractError("E11a outer split must hold exactly one study")
        split_audit.append(
            {
                "split": split.name,
                "held_study": str(held["study_group"].iloc[0]),
                "train_rows": int(len(train)),
                "validation_rows": int(len(held)),
                "train_studies": int(train["study_group"].nunique()),
                "subject_overlap": 0,
            }
        )
    return {
        "task": TASK,
        "train_rows": int(len(dataset.train)),
        "challenge_rows": int(len(dataset.challenge)),
        "study_counts": {str(k): int(v) for k, v in counts.items()},
        "outer_splits": split_audit,
        "pair_policy": "within_source_study_only_symmetric",
        "reference_aggregation": "study_equal",
        "held_outcomes_accepted_by_scoring_api": False,
        "public_probe_authorized": False,
        "competition_submission_attempted": False,
        "ready_for_one_cpu_pairwise_comparison": True,
    }


def evaluate_task_e11a(
    dataset: TaskDataset,
    *,
    config: BaselineConfig,
) -> dict[str, Any]:
    """Run the one pre-registered real-data comparison after feasibility closure."""
    feasibility = audit_e11a_feasibility(dataset)
    base_spec = _find_base_spec(config)
    folds: list[dict[str, Any]] = []
    for split in default_splits_for_task(dataset, random_state=config.random_state):
        training = dataset.train.iloc[split.train_indices].copy()
        held = dataset.train.iloc[split.validation_indices].copy()
        base = _base_score(training, held, dataset=dataset, spec=base_spec)
        model = fit_pairwise_ranker(training, dataset=dataset)
        candidate = score_pairwise_ranker(model, held, dataset=dataset)
        target = pd.to_numeric(
            held[dataset.target_column], errors="coerce"
        ).to_numpy(dtype=float)
        require_finite(target, name="E11a.held_target")
        base_metric = _spearman_value(target, base)
        candidate_metric = _spearman_value(target, candidate)
        folds.append(
            {
                "held_study": str(held["study_group"].iloc[0]),
                "n": int(len(held)),
                "base_spearman": base_metric,
                "pairwise_spearman": candidate_metric,
                "pairwise_delta": (
                    None
                    if base_metric is None or candidate_metric is None
                    else candidate_metric - base_metric
                ),
                "fit_audit": dict(model.fit_audit),
                "held_outcomes_used_for_fit": False,
                "held_outcomes_used_for_scoring": False,
            }
        )

    promotion = _promotion(folds)
    base_challenge = _base_score(
        dataset.train,
        dataset.challenge,
        dataset=dataset,
        spec=base_spec,
    )
    final_model = fit_pairwise_ranker(dataset.train, dataset=dataset)
    pairwise_challenge = score_pairwise_ranker(
        final_model,
        dataset.challenge,
        dataset=dataset,
    )
    return {
        "task": TASK,
        "base_contract": {
            "kind": "b21",
            "model": BASE_MODEL,
            "current_competition_incumbent": True,
        },
        "candidate": {
            "name": "pairwise_hgb",
            "promotion": promotion,
            "challenge_agreement_vs_base": _challenge_agreement(
                base_challenge, pairwise_challenge
            ),
            "fit": dict(final_model.fit_audit),
        },
        "folds": folds,
        "selected_local_candidate": "pairwise_hgb" if promotion["passed"] else None,
        "competition_candidate": False,
        "public_probe_authorized": False,
        "challenge_rows": int(len(dataset.challenge)),
        "feasibility": feasibility,
    }


def run_e11a(config: BaselineConfig, inputs: InputBundle) -> dict[str, Any]:
    if config.baseline != "b021_taskwise_robust":
        raise DataContractError("E11a requires the B2.1 robust config")
    if str(config.section("selection").get("policy", "")) != "robust_v1":
        raise DataContractError("E11a requires robust_v1 selection policy")
    datasets = build_b02_datasets(config, inputs)
    dataset = datasets.get(TASK)
    if not isinstance(dataset, TaskDataset):
        raise DataContractError("E11a missing Task1.1 compact dataset")
    result = evaluate_task_e11a(dataset, config=config)
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "comparison_contract": COMPARISON_CONTRACT,
        "task": result,
        "frozen_conditions": {
            "task": TASK,
            "base_model": BASE_MODEL,
            "pair_policy": "within_source_study_only_symmetric",
            "pair_tie_tolerance": PAIR_TIE_TOLERANCE,
            "max_rows_per_study": MAX_ROWS_PER_STUDY,
            "max_transformed_features": MAX_TRANSFORMED_FEATURES,
            "minimum_directed_pairs": MIN_DIRECTED_PAIRS,
            "reference_aggregation": "study_equal",
            "classifier": "HistGradientBoostingClassifier",
            "classifier_params": dict(PAIRWISE_PARAMS),
            "promotion_mean_delta": PROMOTION_MEAN_DELTA,
            "promotion_minimum_study_delta": PROMOTION_MIN_STUDY_DELTA,
            "promotion_requires_strict_majority_wins": True,
        },
        "cross_study_pair_labels_used": False,
        "held_outcomes_used_for_fit": False,
        "held_outcomes_used_for_scoring": False,
        "leaderboard_used_for_selection": False,
        "competition_submission_attempted": False,
        "incumbent_changed": False,
        "public_probe_authorized": False,
        "automatic_compute_retries": 0,
        "interpretation_limits": [
            "E11a tests one fixed Task1.1 pairwise HGB objective only.",
            "A negative result does not reject pairwise ranking, nonlinear models, or E11 generally.",
            "Pairs compare outcomes only within source studies to avoid cross-study target-scale supervision.",
            "Held and Challenge scores are study-equal averages against source-study reference rows.",
            "No hyperparameter sweep is authorized by this module.",
        ],
    }


__all__ = [
    "EXPERIMENT",
    "PAIRWISE_PARAMS",
    "build_within_study_pairs",
    "fit_pairwise_ranker",
    "score_pairwise_ranker",
    "audit_e11a_feasibility",
    "evaluate_task_e11a",
    "run_e11a",
]
