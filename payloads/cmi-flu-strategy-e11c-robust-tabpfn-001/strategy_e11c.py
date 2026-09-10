"""Strategy-v2 E11c: bounded robust TabPFN fusion for Task1.1.

E11c is the final finite high-capacity Task1.1 branch before E12.  It keeps
the frozen B2.1 PLS-2 model as a 75% rank anchor, reuses the exact E11b
TabPFN identity as a 25% complementary expert, and tests one source-study
jackknife mechanism.  The module deliberately exposes no row-level output
payloads and never authorizes a Public probe or Competition submission.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .configuration import BaselineConfig
from .contracts import DataContractError, require_finite
from .datasets import TaskDataset
from .evaluation import default_splits_for_task
from .metrics import percentile_rank, root_mean_squared_error, safe_spearman
from .runner import InputBundle, build_b02_datasets
from .strategy_e11b import (
    BASE_MODEL,
    BASE_MODEL_SET,
    COMPARISON_CONTRACT,
    EXPECTED_CHALLENGE_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_TRAIN_STUDIES,
    MAX_RAW_FEATURES,
    PROMOTION_MEAN_DELTA,
    PROMOTION_MIN_STUDY_DELTA,
    TABPFN_CHECKPOINT_BYTES,
    TABPFN_CHECKPOINT_FILENAME,
    TABPFN_CHECKPOINT_SHA256,
    TABPFN_KAGGLE_MODEL_SOURCE,
    TABPFN_LICENSE,
    TABPFN_PACKAGE_VERSION,
    TABPFN_PARAMS,
    TABPFN_SOURCE_COMMIT,
    TABPFN_WHEEL_FILENAME,
    TABPFN_WHEEL_SHA256,
    TARGET_TRANSFORM,
    TASK,
    EstimatorFactory,
    _base_score,
    _find_base_spec,
    _spearman_value,
    _tabpfn_frames,
    audit_e11b_feasibility,
    fit_tabpfn_regressor,
    score_tabpfn_regressor,
)

EXPERIMENT = "strategy_v2_e11c_task11_robust_tabpfn"
BASE_WEIGHT = 0.75
TABPFN_WEIGHT = 0.25
ELIGIBLE_CONDITIONS = ("C2", "C4")
SOURCE_IQR_EPSILON = 1e-12
SENSITIVITY_28_N = 28
SENSITIVITY_28_REPETITIONS = 200
SENSITIVITY_28_SEED = 20260907
SENSITIVITY_28_STUDIES = frozenset({"SDY180", "SDY56"})


def deterministic_percentile_rank(values: Sequence[float]) -> np.ndarray:
    """Return the repository-standard average-tie percentile rank."""
    ranked = percentile_rank(values)
    require_finite(ranked, name="E11c.percentile_rank")
    return ranked


def fixed_rank_fusion(
    base_prediction: Sequence[float],
    nonlinear_rank: Sequence[float],
) -> np.ndarray:
    """Fuse the frozen B2.1 rank and a precomputed nonlinear percentile rank.

    The function has no outcome argument by construction.
    """
    base = np.asarray(base_prediction, dtype=float).reshape(-1)
    nonlinear = np.asarray(nonlinear_rank, dtype=float).reshape(-1)
    if base.shape != nonlinear.shape:
        raise DataContractError(
            f"E11c fusion shapes differ:{base.shape}!={nonlinear.shape}"
        )
    require_finite(base, name="E11c.base_prediction")
    require_finite(nonlinear, name="E11c.nonlinear_rank")
    if ((nonlinear < 0.0) | (nonlinear > 1.0)).any():
        raise DataContractError("E11c nonlinear rank lies outside [0,1]")
    fused = (
        BASE_WEIGHT * deterministic_percentile_rank(base)
        + TABPFN_WEIGHT * nonlinear
    )
    require_finite(fused, name="E11c.fused_rank_score")
    return fused


def base_marginal_rank_remap(
    base_prediction: Sequence[float],
    fused_rank_score: Sequence[float],
) -> np.ndarray:
    """Preserve the exact B2.1 raw multiset while imposing fused ordering.

    Stable index order is the deterministic tie-break for equal fused scores.
    This mapping is used only for raw-scale diagnostics/future candidate values;
    primary E11c Spearman scoring uses the fused rank score itself.
    """
    base = np.asarray(base_prediction, dtype=float).reshape(-1)
    fused = np.asarray(fused_rank_score, dtype=float).reshape(-1)
    if base.shape != fused.shape:
        raise DataContractError(
            f"E11c rank-remap shapes differ:{base.shape}!={fused.shape}"
        )
    require_finite(base, name="E11c.rank_remap_base")
    require_finite(fused, name="E11c.rank_remap_fused")
    order = np.argsort(fused, kind="mergesort")
    sorted_base = np.sort(base, kind="mergesort")
    remapped = np.empty_like(base, dtype=float)
    remapped[order] = sorted_base
    if not np.array_equal(
        np.sort(remapped, kind="mergesort"),
        sorted_base,
    ):
        raise DataContractError("E11c base-marginal remap changed the base multiset")
    return remapped


def subset28_sensitivity(
    target: Sequence[float],
    base_prediction: Sequence[float],
    candidate_score: Sequence[float],
    *,
    held_study: str,
) -> dict[str, Any]:
    """Reuse fixed outer predictions for the predeclared 28-subject diagnostic.

    No model is refit and this result never participates in promotion.
    """
    target_array = np.asarray(target, dtype=float).reshape(-1)
    base = np.asarray(base_prediction, dtype=float).reshape(-1)
    candidate = np.asarray(candidate_score, dtype=float).reshape(-1)
    if not (target_array.shape == base.shape == candidate.shape):
        raise DataContractError("E11c sensitivity arrays differ in shape")
    require_finite(target_array, name="E11c.sensitivity.target")
    require_finite(base, name="E11c.sensitivity.base")
    require_finite(candidate, name="E11c.sensitivity.candidate")

    if held_study not in SENSITIVITY_28_STUDIES:
        return {
            "status": "not_run_nonpredeclared_study",
            "held_study": held_study,
            "n": int(target_array.size),
            "subset_n": SENSITIVITY_28_N,
            "repetitions": 0,
            "used_for_promotion": False,
            "model_refits": 0,
        }
    if target_array.size < SENSITIVITY_28_N:
        return {
            "status": "not_run_n_lt_28",
            "held_study": held_study,
            "n": int(target_array.size),
            "subset_n": SENSITIVITY_28_N,
            "repetitions": 0,
            "used_for_promotion": False,
            "model_refits": 0,
        }

    rng = np.random.default_rng(SENSITIVITY_28_SEED)
    indices = np.arange(target_array.size)
    deltas: list[float] = []
    rank_rmse_deltas: list[float] = []
    positive = 0
    invalid = 0
    for _ in range(SENSITIVITY_28_REPETITIONS):
        chosen = rng.choice(indices, size=SENSITIVITY_28_N, replace=False)
        target_subset = target_array[chosen]
        base_subset = base[chosen]
        candidate_subset = candidate[chosen]
        base_metric = safe_spearman(target_subset, base_subset)
        candidate_metric = safe_spearman(target_subset, candidate_subset)
        if base_metric.status != "ok" or candidate_metric.status != "ok":
            invalid += 1
            continue
        delta = float(candidate_metric.value - base_metric.value)
        deltas.append(delta)
        positive += int(delta > 0.0)

        target_rank = deterministic_percentile_rank(target_subset)
        base_rank = deterministic_percentile_rank(base_subset)
        candidate_rank = deterministic_percentile_rank(candidate_subset)
        base_rank_rmse = root_mean_squared_error(target_rank, base_rank)
        candidate_rank_rmse = root_mean_squared_error(target_rank, candidate_rank)
        rank_rmse_deltas.append(
            float(candidate_rank_rmse.value - base_rank_rmse.value)
        )

    values = np.asarray(deltas, dtype=float)
    rmse_values = np.asarray(rank_rmse_deltas, dtype=float)
    return {
        "status": "ok" if values.size else "all_invalid",
        "held_study": held_study,
        "n": int(target_array.size),
        "subset_n": SENSITIVITY_28_N,
        "repetitions": SENSITIVITY_28_REPETITIONS,
        "valid_repetitions": int(values.size),
        "invalid_repetitions": int(invalid),
        "seed": SENSITIVITY_28_SEED,
        "without_replacement": True,
        "spearman_delta_mean": float(np.mean(values)) if values.size else None,
        "spearman_delta_median": float(np.median(values)) if values.size else None,
        "spearman_delta_q10": float(np.quantile(values, 0.10)) if values.size else None,
        "spearman_delta_q90": float(np.quantile(values, 0.90)) if values.size else None,
        "candidate_better_fraction": (
            float(positive / values.size) if values.size else None
        ),
        "rank_rmse_delta_mean": (
            float(np.mean(rmse_values)) if rmse_values.size else None
        ),
        "used_for_promotion": False,
        "model_refits": 0,
    }


def _rank_disagreement(
    left: Sequence[float],
    right: Sequence[float],
) -> dict[str, Any]:
    left_rank = deterministic_percentile_rank(left)
    right_rank = deterministic_percentile_rank(right)
    metric = safe_spearman(left_rank, right_rank)
    delta = np.abs(left_rank - right_rank)
    return {
        "rank_spearman": float(metric.value) if metric.status == "ok" else None,
        "rank_spearman_status": metric.status,
        "changed_rank_count": int(np.sum(delta > 1e-12)),
        "mean_absolute_percentile_shift": float(np.mean(delta)),
        "max_absolute_percentile_shift": float(np.max(delta)),
    }


def prediction_disagreement_audit(
    base_prediction: Sequence[float],
    full_tabpfn_prediction: Sequence[float],
    jackknife_rank: Sequence[float],
    source_drop_ranks: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Outcome-independent aggregate disagreement audit."""
    base_rank = deterministic_percentile_rank(base_prediction)
    full_rank = deterministic_percentile_rank(full_tabpfn_prediction)
    jack_rank = np.asarray(jackknife_rank, dtype=float).reshape(-1)
    require_finite(jack_rank, name="E11c.jackknife_rank")
    if not (
        base_rank.shape == full_rank.shape == jack_rank.shape
    ):
        raise DataContractError("E11c disagreement rank shapes differ")
    drops = [np.asarray(values, dtype=float).reshape(-1) for values in source_drop_ranks]
    if len(drops) != EXPECTED_TRAIN_STUDIES - 1:
        raise DataContractError(
            "E11c outer disagreement requires exactly three source-drop ranks"
        )
    for values in drops:
        if values.shape != base_rank.shape:
            raise DataContractError("E11c source-drop rank shape differs")
        require_finite(values, name="E11c.source_drop_rank")

    pairwise: list[float] = []
    pairwise_status: list[str] = []
    for left_idx, right_idx in combinations(range(len(drops)), 2):
        metric = safe_spearman(drops[left_idx], drops[right_idx])
        pairwise_status.append(metric.status)
        if metric.status == "ok":
            pairwise.append(float(metric.value))
    return {
        "base_vs_full_tabpfn": _rank_disagreement(base_rank, full_rank),
        "base_vs_jackknife_tabpfn": _rank_disagreement(base_rank, jack_rank),
        "full_vs_jackknife_tabpfn": _rank_disagreement(full_rank, jack_rank),
        "source_dropout_pair_count": 3,
        "source_dropout_pairwise_rank_spearman_min": (
            float(np.min(pairwise)) if pairwise else None
        ),
        "source_dropout_pairwise_rank_spearman_median": (
            float(np.median(pairwise)) if pairwise else None
        ),
        "source_dropout_pairwise_rank_spearman_max": (
            float(np.max(pairwise)) if pairwise else None
        ),
        "source_dropout_pairwise_statuses": pairwise_status,
        "held_outcomes_used": False,
    }


def _normalize_category(series: pd.Series) -> pd.Series:
    values = series.astype("object")
    missing = pd.isna(values)
    normalized = values.astype(str)
    normalized.loc[missing] = "__E11C_MISSING__"
    return normalized


def feature_shift_audit(
    source_features: pd.DataFrame,
    held_features: pd.DataFrame,
    *,
    dataset: TaskDataset,
) -> dict[str, Any]:
    """Aggregate X-only source-versus-held transfer diagnostics.

    Both frames must be target-free. Reference medians/IQRs and category
    distributions are fitted exclusively on source_features.
    """
    if dataset.target_column in source_features.columns:
        raise DataContractError("E11c source audit frame must not contain outcome")
    if dataset.target_column in held_features.columns:
        raise DataContractError("E11c held audit frame must not contain outcome")

    source_x, held_x, _, numeric, categorical = _tabpfn_frames(
        source_features,
        held_features,
        dataset=dataset,
    )

    numeric_shifts: list[float] = []
    numeric_missing_deltas: list[float] = []
    zero_iqr = 0
    unavailable_reference = 0
    for column in numeric:
        source = pd.to_numeric(source_x[column], errors="coerce").to_numpy(dtype=float)
        held = pd.to_numeric(held_x[column], errors="coerce").to_numpy(dtype=float)
        numeric_missing_deltas.append(
            abs(float(np.mean(~np.isfinite(source))) - float(np.mean(~np.isfinite(held))))
        )
        source_finite = source[np.isfinite(source)]
        held_finite = held[np.isfinite(held)]
        if source_finite.size == 0 or held_finite.size == 0:
            unavailable_reference += 1
            continue
        source_median = float(np.median(source_finite))
        q25, q75 = np.percentile(source_finite, [25.0, 75.0])
        iqr = float(q75 - q25)
        if not np.isfinite(iqr) or abs(iqr) <= SOURCE_IQR_EPSILON:
            zero_iqr += 1
            denominator = SOURCE_IQR_EPSILON
        else:
            denominator = abs(iqr)
        held_median = float(np.median(held_finite))
        numeric_shifts.append(abs(held_median - source_median) / denominator)

    tv_distances: list[float] = []
    unseen_value_count = 0
    unseen_unique_levels = 0
    categorical_value_count = 0
    for column in categorical:
        source = _normalize_category(source_x[column])
        held = _normalize_category(held_x[column])
        source_levels = set(source.tolist())
        held_levels = set(held.tolist())
        unseen_levels = held_levels.difference(source_levels)
        unseen_unique_levels += len(unseen_levels)
        unseen_value_count += int(held.isin(unseen_levels).sum())
        categorical_value_count += int(len(held))
        source_dist = source.value_counts(normalize=True, dropna=False)
        held_dist = held.value_counts(normalize=True, dropna=False)
        levels = sorted(set(source_dist.index).union(held_dist.index))
        source_probs = source_dist.reindex(levels, fill_value=0.0).to_numpy(dtype=float)
        held_probs = held_dist.reindex(levels, fill_value=0.0).to_numpy(dtype=float)
        tv_distances.append(float(0.5 * np.abs(source_probs - held_probs).sum()))

    shifts = np.asarray(numeric_shifts, dtype=float)
    missing_deltas = np.asarray(numeric_missing_deltas, dtype=float)
    tv = np.asarray(tv_distances, dtype=float)
    return {
        "numeric": {
            "feature_count": int(len(numeric)),
            "usable_shift_feature_count": int(len(numeric_shifts)),
            "source_reference_unavailable_count": int(unavailable_reference),
            "mean_absolute_held_median_shift_in_source_median_iqr_units": (
                float(np.mean(shifts)) if shifts.size else None
            ),
            "p90_absolute_held_median_shift_in_source_median_iqr_units": (
                float(np.percentile(shifts, 90.0)) if shifts.size else None
            ),
            "mean_absolute_missingness_rate_delta": (
                float(np.mean(missing_deltas)) if missing_deltas.size else 0.0
            ),
            "zero_iqr_feature_count": int(zero_iqr),
            "iqr_epsilon": SOURCE_IQR_EPSILON,
        },
        "categorical": {
            "feature_count": int(len(categorical)),
            "held_unseen_level_count": int(unseen_value_count),
            "held_unseen_unique_level_count": int(unseen_unique_levels),
            "held_unseen_level_fraction": (
                float(unseen_value_count / categorical_value_count)
                if categorical_value_count
                else 0.0
            ),
            "mean_category_total_variation_distance": (
                float(np.mean(tv)) if tv.size else 0.0
            ),
            "max_category_total_variation_distance": (
                float(np.max(tv)) if tv.size else 0.0
            ),
        },
        "held_outcomes_used": False,
        "used_for_candidate_routing": False,
    }


def source_study_jackknife_ranks(
    source_training: pd.DataFrame,
    prediction_features: pd.DataFrame,
    *,
    dataset: TaskDataset,
    checkpoint_path: Path | None,
    estimator_factory: EstimatorFactory | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
    """Fit all three outer source-study-drop TabPFNs and median their ranks."""
    studies = sorted(source_training["study_group"].astype(str).unique().tolist())
    if len(studies) != EXPECTED_TRAIN_STUDIES - 1:
        raise DataContractError(
            f"E11c outer jackknife requires 3 source studies, found {len(studies)}"
        )
    rank_vectors: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for dropped in studies:
        training = source_training.loc[
            source_training["study_group"].astype(str).ne(dropped)
        ].copy()
        fit_studies = sorted(training["study_group"].astype(str).unique().tolist())
        if len(fit_studies) != EXPECTED_TRAIN_STUDIES - 2:
            raise DataContractError(
                "E11c source-drop fit must contain exactly two source studies"
            )
        if dropped in fit_studies:
            raise DataContractError("E11c dropped source study escaped purge")
        model = fit_tabpfn_regressor(
            training,
            dataset=dataset,
            checkpoint_path=checkpoint_path,
            estimator_factory=estimator_factory,
        )
        prediction = score_tabpfn_regressor(
            model,
            prediction_features,
            dataset=dataset,
        )
        rank_vectors.append(deterministic_percentile_rank(prediction))
        audits.append(
            {
                "dropped_source_study": dropped,
                "training_studies": fit_studies,
                "training_study_count": int(len(fit_studies)),
                "training_rows": int(len(training)),
                "fit_seconds": float(model.fit_audit["fit_seconds"]),
                "backend": str(model.fit_audit["backend"]),
                "held_outcomes_used_for_fit": False,
            }
        )
    jackknife = np.median(np.vstack(rank_vectors), axis=0)
    require_finite(jackknife, name="E11c.outer_jackknife_rank")
    return jackknife, rank_vectors, audits


def _final_source_study_jackknife_ranks(
    training: pd.DataFrame,
    prediction_features: pd.DataFrame,
    *,
    dataset: TaskDataset,
    checkpoint_path: Path | None,
    estimator_factory: EstimatorFactory | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build the C4 final four-model leave-one-historical-study ensemble."""
    studies = sorted(training["study_group"].astype(str).unique().tolist())
    if len(studies) != EXPECTED_TRAIN_STUDIES:
        raise DataContractError(
            f"E11c final jackknife requires 4 historical studies, found {len(studies)}"
        )
    ranks: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for dropped in studies:
        fit_frame = training.loc[training["study_group"].astype(str).ne(dropped)].copy()
        fit_studies = sorted(fit_frame["study_group"].astype(str).unique().tolist())
        if len(fit_studies) != EXPECTED_TRAIN_STUDIES - 1 or dropped in fit_studies:
            raise DataContractError("E11c final source-study drop contract failed")
        model = fit_tabpfn_regressor(
            fit_frame,
            dataset=dataset,
            checkpoint_path=checkpoint_path,
            estimator_factory=estimator_factory,
        )
        prediction = score_tabpfn_regressor(model, prediction_features, dataset=dataset)
        ranks.append(deterministic_percentile_rank(prediction))
        audits.append(
            {
                "dropped_source_study": dropped,
                "training_study_count": int(len(fit_studies)),
                "training_rows": int(len(fit_frame)),
                "fit_seconds": float(model.fit_audit["fit_seconds"]),
                "backend": str(model.fit_audit["backend"]),
            }
        )
    jackknife = np.median(np.vstack(ranks), axis=0)
    require_finite(jackknife, name="E11c.final_jackknife_rank")
    return jackknife, audits


def _candidate_fold_metrics(
    target: np.ndarray,
    base_prediction: np.ndarray,
    candidate_score: np.ndarray,
    *,
    held_study: str,
) -> dict[str, Any]:
    base_spearman = _spearman_value(target, base_prediction)
    candidate_spearman = _spearman_value(target, candidate_score)
    delta = (
        float(candidate_spearman - base_spearman)
        if base_spearman is not None and candidate_spearman is not None
        else None
    )
    remapped = base_marginal_rank_remap(base_prediction, candidate_score)
    base_rmse = root_mean_squared_error(target, base_prediction)
    remapped_rmse = root_mean_squared_error(target, remapped)
    disagreement = _rank_disagreement(base_prediction, candidate_score)
    return {
        "spearman": candidate_spearman,
        "delta_vs_c0": delta,
        "base_marginal_rank_remap_rmse": (
            float(remapped_rmse.value) if remapped_rmse.status == "ok" else None
        ),
        "c0_raw_rmse": float(base_rmse.value) if base_rmse.status == "ok" else None,
        "changed_rank_count_vs_c0": disagreement["changed_rank_count"],
        "mean_absolute_percentile_shift_vs_c0": disagreement[
            "mean_absolute_percentile_shift"
        ],
        "max_absolute_percentile_shift_vs_c0": disagreement[
            "max_absolute_percentile_shift"
        ],
        "raw_multiset_preserved": bool(
            np.array_equal(
                np.sort(remapped, kind="mergesort"),
                np.sort(base_prediction, kind="mergesort"),
            )
        ),
        "sensitivity_28": subset28_sensitivity(
            target,
            base_prediction,
            candidate_score,
            held_study=held_study,
        ),
    }


def promotion_from_folds(
    folds: Sequence[Mapping[str, Any]],
    *,
    condition: str,
) -> dict[str, Any]:
    """Apply the fixed E11b gate independently to C2 or C4."""
    if condition not in ELIGIBLE_CONDITIONS:
        raise DataContractError(
            f"E11c promotion condition must be one of {ELIGIBLE_CONDITIONS}"
        )
    deltas: list[float] = []
    weights: list[int] = []
    for fold in folds:
        conditions = fold.get("conditions", {})
        candidate = conditions.get(condition, {})
        delta = candidate.get("delta_vs_c0")
        if delta is None or not np.isfinite(float(delta)):
            continue
        deltas.append(float(delta))
        weights.append(int(fold["n"]))
    if not deltas:
        return {
            "condition": condition,
            "passed": False,
            "reason": "no_usable_held_studies",
            "usable_held_studies": 0,
            "mean_delta": None,
            "median_delta": None,
            "minimum_delta": None,
            "maximum_delta": None,
            "wins": 0,
            "required_wins": 0,
            "row_weighted_delta_diagnostic": None,
            "mean_delta_threshold": PROMOTION_MEAN_DELTA,
            "minimum_study_delta_threshold": PROMOTION_MIN_STUDY_DELTA,
        }
    values = np.asarray(deltas, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    wins = int(np.sum(values > 0.0))
    required = int(len(values) // 2 + 1)
    mean_delta = float(np.mean(values))
    minimum_delta = float(np.min(values))
    return {
        "condition": condition,
        "passed": bool(
            mean_delta >= PROMOTION_MEAN_DELTA
            and minimum_delta >= PROMOTION_MIN_STUDY_DELTA
            and wins >= required
        ),
        "usable_held_studies": int(len(values)),
        "mean_delta": mean_delta,
        "median_delta": float(np.median(values)),
        "minimum_delta": minimum_delta,
        "maximum_delta": float(np.max(values)),
        "wins": wins,
        "required_wins": required,
        "row_weighted_delta_diagnostic": float(
            np.average(values, weights=weights_array)
        ),
        "mean_delta_threshold": PROMOTION_MEAN_DELTA,
        "minimum_study_delta_threshold": PROMOTION_MIN_STUDY_DELTA,
        "study_equal_primary": True,
        "row_weighted_diagnostic_only": True,
    }


def audit_e11c_feasibility(dataset: TaskDataset) -> dict[str, Any]:
    """Reuse E11b structural checks and add the exact E11c jackknife contract."""
    parent = audit_e11b_feasibility(dataset)
    outer: list[dict[str, Any]] = []
    for split in default_splits_for_task(dataset):
        source = dataset.train.iloc[split.train_indices].copy()
        held = dataset.train.iloc[split.validation_indices].copy()
        source_studies = sorted(source["study_group"].astype(str).unique().tolist())
        held_studies = sorted(held["study_group"].astype(str).unique().tolist())
        if len(source_studies) != 3 or len(held_studies) != 1:
            raise DataContractError("E11c outer study cardinality changed")
        if held_studies[0] in source_studies:
            raise DataContractError("E11c held study appears in outer source")
        drop_training_study_counts: list[int] = []
        for dropped in source_studies:
            remaining = source.loc[source["study_group"].astype(str).ne(dropped)]
            remaining_studies = sorted(
                remaining["study_group"].astype(str).unique().tolist()
            )
            if len(remaining_studies) != 2 or dropped in remaining_studies:
                raise DataContractError("E11c planned source-drop membership is invalid")
            if held_studies[0] in remaining_studies:
                raise DataContractError("E11c held study escaped into source-drop fit")
            drop_training_study_counts.append(len(remaining_studies))
        outer.append(
            {
                "held_study": held_studies[0],
                "source_study_count": 3,
                "planned_source_drop_fits": 3,
                "source_drop_training_study_counts": drop_training_study_counts,
                "subject_overlap": int(
                    len(
                        set(source["subject_group"].astype(str)).intersection(
                            set(held["subject_group"].astype(str))
                        )
                    )
                ),
            }
        )
    return {
        "ready_for_e11c_cpu_execution": True,
        "parent_e11b_feasibility": parent,
        "outer_splits": outer,
        "expected_outer_full_tabpfn_fits": 4,
        "expected_outer_source_drop_tabpfn_fits": 12,
        "expected_max_real_tabpfn_fits_if_both_candidates_constructed": 21,
        "fixed_base_weight": BASE_WEIGHT,
        "fixed_tabpfn_weight": TABPFN_WEIGHT,
        "held_outcomes_used_for_feature_audit": False,
        "held_outcomes_used_for_prediction_disagreement_audit": False,
        "dynamic_gate_fitted": False,
    }


def evaluate_task_e11c(
    dataset: TaskDataset,
    *,
    config: BaselineConfig,
    checkpoint_path: Path | None,
    estimator_factory: EstimatorFactory | None = None,
) -> dict[str, Any]:
    """Evaluate C0-C4, then construct Challenge candidates only after local gates."""
    feasibility = audit_e11c_feasibility(dataset)
    base_spec = _find_base_spec(config)
    folds: list[dict[str, Any]] = []

    for split in default_splits_for_task(dataset):
        source = dataset.train.iloc[split.train_indices].copy()
        held = dataset.train.iloc[split.validation_indices].copy()
        held_features = held.drop(columns=[dataset.target_column]).copy()
        source_features = source.drop(columns=[dataset.target_column]).copy()
        target = pd.to_numeric(
            held[dataset.target_column], errors="coerce"
        ).to_numpy(dtype=float)
        require_finite(target, name="E11c.held_target")
        held_study = str(split.held_out_group)
        source_studies = sorted(source["study_group"].astype(str).unique().tolist())
        if held_study in source_studies:
            raise DataContractError("E11c held study escaped outer purge")

        base = _base_score(source, held_features, dataset=dataset, spec=base_spec)
        full_model = fit_tabpfn_regressor(
            source,
            dataset=dataset,
            checkpoint_path=checkpoint_path,
            estimator_factory=estimator_factory,
        )
        full = score_tabpfn_regressor(full_model, held_features, dataset=dataset)
        full_rank = deterministic_percentile_rank(full)
        c2 = fixed_rank_fusion(base, full_rank)
        jackknife_rank, source_drop_ranks, source_drop_audits = (
            source_study_jackknife_ranks(
                source,
                held_features,
                dataset=dataset,
                checkpoint_path=checkpoint_path,
                estimator_factory=estimator_factory,
            )
        )
        c4 = fixed_rank_fusion(base, jackknife_rank)

        base_spearman = _spearman_value(target, base)
        c1_spearman = _spearman_value(target, full)
        c3_spearman = _spearman_value(target, jackknife_rank)
        folds.append(
            {
                "held_study": held_study,
                "n": int(len(held)),
                "source_study_count": int(len(source_studies)),
                "conditions": {
                    "C0": {
                        "name": "b21_pls2_reference",
                        "spearman": base_spearman,
                        "raw_rmse": float(
                            root_mean_squared_error(target, base).value
                        ),
                    },
                    "C1": {
                        "name": "tabpfn_full_reproduction_control",
                        "spearman": c1_spearman,
                        "delta_vs_c0": (
                            float(c1_spearman - base_spearman)
                            if c1_spearman is not None and base_spearman is not None
                            else None
                        ),
                        "eligible_for_e12": False,
                        "fit_seconds": float(full_model.fit_audit["fit_seconds"]),
                        "backend": str(full_model.fit_audit["backend"]),
                    },
                    "C2": {
                        "name": "pls75_tabpfn25_full_rank_fusion",
                        "eligible_for_e12": True,
                        **_candidate_fold_metrics(
                            target, base, c2, held_study=held_study
                        ),
                    },
                    "C3": {
                        "name": "tabpfn_source_study_jackknife_rank_median",
                        "spearman": c3_spearman,
                        "delta_vs_c0": (
                            float(c3_spearman - base_spearman)
                            if c3_spearman is not None and base_spearman is not None
                            else None
                        ),
                        "eligible_for_e12": False,
                        "source_drop_fit_count": int(len(source_drop_audits)),
                    },
                    "C4": {
                        "name": "pls75_tabpfn25_jackknife_rank_fusion",
                        "eligible_for_e12": True,
                        **_candidate_fold_metrics(
                            target, base, c4, held_study=held_study
                        ),
                    },
                },
                "source_drop_fits": source_drop_audits,
                "feature_shift_audit": feature_shift_audit(
                    source_features,
                    held_features,
                    dataset=dataset,
                ),
                "prediction_disagreement_audit": prediction_disagreement_audit(
                    base,
                    full,
                    jackknife_rank,
                    source_drop_ranks,
                ),
                "held_outcomes_used_for_fit": False,
                "held_outcomes_used_for_fusion": False,
                "held_outcomes_used_for_transfer_audit": False,
                "held_outcomes_used_for_evaluation_only": True,
            }
        )

    promotions = {
        condition: promotion_from_folds(folds, condition=condition)
        for condition in ELIGIBLE_CONDITIONS
    }
    passing = [
        condition
        for condition in ELIGIBLE_CONDITIONS
        if promotions[condition]["passed"]
    ]

    challenge: dict[str, Any] = {
        "rows": int(len(dataset.challenge)),
        "constructed_conditions": [],
        "candidate_construction_performed": False,
        "public_probe_authorized": False,
        "competition_submission_authorized": False,
        "challenge_labels_used": False,
        "final_tabpfn_fit_count": 0,
    }
    if passing:
        base_challenge = _base_score(
            dataset.train,
            dataset.challenge,
            dataset=dataset,
            spec=base_spec,
        )
        challenge["candidate_construction_performed"] = True
        candidate_diagnostics: dict[str, Any] = {}
        if "C2" in passing:
            final_full_model = fit_tabpfn_regressor(
                dataset.train,
                dataset=dataset,
                checkpoint_path=checkpoint_path,
                estimator_factory=estimator_factory,
            )
            full_challenge = score_tabpfn_regressor(
                final_full_model,
                dataset.challenge,
                dataset=dataset,
            )
            c2_score = fixed_rank_fusion(
                base_challenge,
                deterministic_percentile_rank(full_challenge),
            )
            c2_raw = base_marginal_rank_remap(base_challenge, c2_score)
            candidate_diagnostics["C2"] = {
                "agreement_vs_c0": _rank_disagreement(base_challenge, c2_score),
                "base_marginal_multiset_preserved": bool(
                    np.array_equal(
                        np.sort(c2_raw, kind="mergesort"),
                        np.sort(base_challenge, kind="mergesort"),
                    )
                ),
                "full_tabpfn_fit_seconds": float(
                    final_full_model.fit_audit["fit_seconds"]
                ),
            }
            challenge["constructed_conditions"].append("C2")
            challenge["final_tabpfn_fit_count"] += 1
        if "C4" in passing:
            final_jackknife_rank, final_drop_audits = (
                _final_source_study_jackknife_ranks(
                    dataset.train,
                    dataset.challenge,
                    dataset=dataset,
                    checkpoint_path=checkpoint_path,
                    estimator_factory=estimator_factory,
                )
            )
            c4_score = fixed_rank_fusion(base_challenge, final_jackknife_rank)
            c4_raw = base_marginal_rank_remap(base_challenge, c4_score)
            candidate_diagnostics["C4"] = {
                "agreement_vs_c0": _rank_disagreement(base_challenge, c4_score),
                "base_marginal_multiset_preserved": bool(
                    np.array_equal(
                        np.sort(c4_raw, kind="mergesort"),
                        np.sort(base_challenge, kind="mergesort"),
                    )
                ),
                "source_drop_fit_count": int(len(final_drop_audits)),
                "source_drop_fit_seconds": [
                    float(item["fit_seconds"]) for item in final_drop_audits
                ],
            }
            challenge["constructed_conditions"].append("C4")
            challenge["final_tabpfn_fit_count"] += len(final_drop_audits)
        challenge["candidate_diagnostics"] = candidate_diagnostics

    outer_full_fits = len(folds)
    outer_drop_fits = int(sum(len(fold["source_drop_fits"]) for fold in folds))
    total_fits = (
        outer_full_fits
        + outer_drop_fits
        + int(challenge["final_tabpfn_fit_count"])
    )
    if total_fits > 21:
        raise DataContractError("E11c exceeded the frozen 21-fit resource bound")
    next_step = (
        "close_Task1.1_E11_and_proceed_E12"
        if not passing
        else "carry_only_gate_passing_candidates_to_E12"
    )
    return {
        "task": TASK,
        "conditions": {
            "C0": {"eligible_for_e12": False, "kind": "b21_pls2_reference"},
            "C1": {"eligible_for_e12": False, "kind": "full_tabpfn_control"},
            "C2": {
                "eligible_for_e12": True,
                "base_weight": BASE_WEIGHT,
                "tabpfn_weight": TABPFN_WEIGHT,
            },
            "C3": {
                "eligible_for_e12": False,
                "kind": "source_study_jackknife_diagnostic",
            },
            "C4": {
                "eligible_for_e12": True,
                "base_weight": BASE_WEIGHT,
                "tabpfn_weight": TABPFN_WEIGHT,
            },
        },
        "folds": folds,
        "promotion": promotions,
        "e12_eligible_conditions": passing,
        "e11c_winner_selected": False,
        "challenge": challenge,
        "fit_accounting": {
            "outer_full_tabpfn_fits": int(outer_full_fits),
            "outer_source_drop_tabpfn_fits": int(outer_drop_fits),
            "final_tabpfn_fits": int(challenge["final_tabpfn_fit_count"]),
            "total_tabpfn_fits": int(total_fits),
            "maximum_allowed_if_both_candidates_constructed": 21,
        },
        "stopping_rule_decision": next_step,
        "feasibility": feasibility,
        "dynamic_domain_gate_fitted": False,
        "public_probe_authorized": False,
        "competition_submission_authorized": False,
        "incumbent_changed": False,
    }


def run_e11c(
    config: BaselineConfig,
    inputs: InputBundle,
    *,
    checkpoint_path: Path | None,
    estimator_factory: EstimatorFactory | None = None,
) -> dict[str, Any]:
    if config.baseline != "b021_taskwise_robust":
        raise DataContractError("E11c requires the B2.1 robust config")
    if str(config.section("selection").get("policy", "")) != "robust_v1":
        raise DataContractError("E11c requires robust_v1 selection policy")
    datasets = build_b02_datasets(config, inputs)
    dataset = datasets.get(TASK)
    if not isinstance(dataset, TaskDataset):
        raise DataContractError("E11c missing Task1.1 compact dataset")
    task = evaluate_task_e11c(
        dataset,
        config=config,
        checkpoint_path=checkpoint_path,
        estimator_factory=estimator_factory,
    )
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "parent_experiment": "E11b",
        "comparison_contract": COMPARISON_CONTRACT,
        "frozen_conditions": {
            "task": TASK,
            "base_model": BASE_MODEL,
            "base_model_set": BASE_MODEL_SET,
            "target_transform": TARGET_TRANSFORM,
            "expected_train_rows": EXPECTED_TRAIN_ROWS,
            "expected_train_studies": EXPECTED_TRAIN_STUDIES,
            "expected_challenge_rows": EXPECTED_CHALLENGE_ROWS,
            "max_raw_features": MAX_RAW_FEATURES,
            "base_weight": BASE_WEIGHT,
            "tabpfn_weight": TABPFN_WEIGHT,
            "tabpfn_package_version": TABPFN_PACKAGE_VERSION,
            "tabpfn_source_commit": TABPFN_SOURCE_COMMIT,
            "tabpfn_wheel_filename": TABPFN_WHEEL_FILENAME,
            "tabpfn_wheel_sha256": TABPFN_WHEEL_SHA256,
            "kaggle_model_source": TABPFN_KAGGLE_MODEL_SOURCE,
            "checkpoint_filename": TABPFN_CHECKPOINT_FILENAME,
            "checkpoint_bytes": TABPFN_CHECKPOINT_BYTES,
            "checkpoint_sha256": TABPFN_CHECKPOINT_SHA256,
            "tabpfn_params": dict(TABPFN_PARAMS),
            "promotion_mean_delta": PROMOTION_MEAN_DELTA,
            "promotion_minimum_study_delta": PROMOTION_MIN_STUDY_DELTA,
            "promotion_requires_strict_majority_wins": True,
            "sensitivity_28": {
                "studies": sorted(SENSITIVITY_28_STUDIES),
                "subset_n": SENSITIVITY_28_N,
                "repetitions": SENSITIVITY_28_REPETITIONS,
                "seed": SENSITIVITY_28_SEED,
                "used_for_promotion": False,
                "model_refits": 0,
            },
        },
        "external_model_license": dict(TABPFN_LICENSE),
        "task": task,
        "aggregate_only_serialization": True,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
        "held_outcomes_used_for_fit": False,
        "held_outcomes_used_for_fusion": False,
        "held_outcomes_used_for_transfer_audit": False,
        "automatic_compute_retries": 0,
        "leaderboard_used_for_selection": False,
        "public_probe_authorized": False,
        "competition_submission_attempted": False,
        "incumbent_changed": False,
    }


__all__ = [
    "BASE_WEIGHT",
    "ELIGIBLE_CONDITIONS",
    "EXPERIMENT",
    "SENSITIVITY_28_N",
    "SENSITIVITY_28_REPETITIONS",
    "SENSITIVITY_28_SEED",
    "SENSITIVITY_28_STUDIES",
    "SOURCE_IQR_EPSILON",
    "TABPFN_WEIGHT",
    "audit_e11c_feasibility",
    "base_marginal_rank_remap",
    "deterministic_percentile_rank",
    "evaluate_task_e11c",
    "feature_shift_audit",
    "fixed_rank_fusion",
    "prediction_disagreement_audit",
    "promotion_from_folds",
    "run_e11c",
    "source_study_jackknife_ranks",
    "subset28_sensitivity",
]
