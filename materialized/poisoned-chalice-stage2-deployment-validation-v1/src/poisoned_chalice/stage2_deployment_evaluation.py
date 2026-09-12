"""Evaluation-only helpers for STAGE2-DEPLOYMENT-VALIDATION-V1.

These functions are intentionally separate from scoring.  They consume labels only
after a label-free prediction artifact has been sealed and hashed.  S_proxy is a
local competition-shaped proxy and is never presented as the organizer score.
"""
from __future__ import annotations

from typing import Sequence
import math

import numpy as np
import pandas as pd

from poisoned_chalice.evaluation import low_fpr_metrics

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260911
REFERENCE_SUITES = {
    "family_balanced_v1": ("content", "portable_geometry", "legacy_output"),
    "without_content_sensitivity": ("portable_geometry", "legacy_output"),
    "without_geometry_sensitivity": ("content", "legacy_output"),
}


def conservative_mask(y_true: Sequence[int], score: Sequence[float], target_fpr: float = 0.01):
    y = np.asarray(y_true, dtype=int)
    values = np.asarray(score, dtype=float)
    if y.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("invalid conservative-threshold inputs")
    negatives = np.sort(values[y == 0])[::-1]
    if len(negatives) == 0:
        raise ValueError("conservative threshold requires negatives")
    allowed = max(1, int(math.floor(target_fpr * len(negatives))))
    threshold = float(negatives[allowed - 1])
    detected = values > threshold
    fp = int(np.sum((y == 0) & detected))
    tp = int(np.sum((y == 1) & detected))
    return detected, threshold, fp, tp


def metric_block(frame: pd.DataFrame, score: Sequence[float]) -> dict:
    values = np.asarray(score, dtype=float)
    result = low_fpr_metrics(frame.label, values)
    _, threshold, fp, tp = conservative_mask(frame.label, values, 0.01)
    result.update({
        "conservative_threshold_1pct": threshold,
        "conservative_fp_1pct": fp,
        "conservative_tp_1pct": tp,
    })
    result["per_language"] = {
        language: low_fpr_metrics(group.label, values[group.index.to_numpy()])
        for language, group in frame.groupby("language", sort=True)
    }
    if "token_count" in frame:
        length_bin = pd.qcut(frame.token_count, 5, labels=False, duplicates="drop")
        result["per_length_quintile"] = {}
        for group_id, index in length_bin.groupby(length_bin).groups.items():
            subset = frame.loc[index]
            if len(subset) >= 20 and subset.label.nunique() == 2:
                result["per_length_quintile"][str(int(group_id))] = low_fpr_metrics(
                    subset.label, values[np.asarray(index, dtype=int)]
                )
    if "content_group_non_singleton" in frame:
        result["per_content_group_type"] = {}
        for key, subset in frame.groupby("content_group_non_singleton"):
            if len(subset) >= 20 and subset.label.nunique() == 2:
                result["per_content_group_type"]["non_singleton" if bool(key) else "singleton"] = low_fpr_metrics(
                    subset.label, values[subset.index.to_numpy()]
                )
    return result


def reference_family_masks(frame: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    y = frame.label.to_numpy(int)
    families = {
        "content": ("C1_source_content",),
        "portable_geometry": ("GR_mean96",),
        "legacy_output": (
            "zsigmoid_single_sequence",
            "caller_literal_output_weighted_single_sequence",
            "helper_language_aware_output_weighted_single_sequence",
        ),
    }
    output = {}
    for family, names in families.items():
        masks = [conservative_mask(y, scores[name], 0.01)[0].astype(float) for name in names]
        output[family] = np.mean(np.stack(masks, axis=0), axis=0)
    return output


def proxy_score(
    frame: pd.DataFrame,
    candidate: Sequence[float],
    scores: dict[str, np.ndarray],
    reference_families: Sequence[str],
) -> dict:
    y = frame.label.to_numpy(int)
    values = np.asarray(candidate, dtype=float)
    auc = low_fpr_metrics(y, values)["auc"]
    detected, _, _, tp = conservative_mask(y, values, 0.01)
    refs = reference_family_masks(frame, scores)
    overlap = np.mean(np.stack([refs[name] for name in reference_families], axis=0), axis=0)
    member = y == 1
    novelty = float(np.sum(member * detected * (1.0 - overlap)) / int(np.sum(member)))
    return {
        "auc": auc,
        "novelty_proxy": novelty,
        "S_proxy": 0.5 * (auc + novelty),
        "conservative_tp": tp,
        "reference_families": list(reference_families),
    }


def paired_bootstrap(
    frame: pd.DataFrame,
    candidate_name: str,
    baseline_name: str,
    scores: dict[str, np.ndarray],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    strata = [group.index.to_numpy() for _, group in frame.groupby(["language", "label"], sort=True)]
    observed: list[dict[str, float]] = []
    full_reference = REFERENCE_SUITES["family_balanced_v1"]
    for _ in range(replicates):
        sampled = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in strata])
        work = frame.iloc[sampled].reset_index(drop=True)
        local = {name: np.asarray(value, dtype=float)[sampled] for name, value in scores.items()}
        candidate_proxy = proxy_score(work, local[candidate_name], local, full_reference)
        baseline_proxy = proxy_score(work, local[baseline_name], local, full_reference)
        candidate_metrics = low_fpr_metrics(work.label, local[candidate_name])
        baseline_metrics = low_fpr_metrics(work.label, local[baseline_name])
        observed.append({
            "delta_S_proxy": candidate_proxy["S_proxy"] - baseline_proxy["S_proxy"],
            "delta_auc": candidate_metrics["auc"] - baseline_metrics["auc"],
            "delta_tpr_001": candidate_metrics["tpr_at_0.001_fpr"] - baseline_metrics["tpr_at_0.001_fpr"],
            "delta_tpr_005": candidate_metrics["tpr_at_0.005_fpr"] - baseline_metrics["tpr_at_0.005_fpr"],
            "delta_tpr_01": candidate_metrics["tpr_at_0.01_fpr"] - baseline_metrics["tpr_at_0.01_fpr"],
            "delta_tpr_02": candidate_metrics["tpr_at_0.02_fpr"] - baseline_metrics["tpr_at_0.02_fpr"],
        })
    result = {}
    for key in observed[0]:
        values = np.asarray([row[key] for row in observed], dtype=float)
        result[key] = {
            "mean": float(values.mean()),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
    result["replicates"] = replicates
    result["resampling"] = "language x label stratified paired rows; candidate/baseline/reference together; every 1% threshold recomputed"
    result["row_bootstrap_scope"] = "sampling uncertainty on this one model/cohort only; not a model-transfer generalization CI"
    return result


def evaluate_prediction_frame(
    frame: pd.DataFrame,
    *,
    score_names: Sequence[str],
    primary_candidate: str,
    primary_baseline: str,
    extra_bootstrap_pairs: Sequence[tuple[str, str]] = (),
) -> dict:
    if frame.label.isna().any() or sorted(frame.label.unique().tolist()) != [0, 1]:
        raise ValueError("evaluation frame labels invalid")
    scores = {name: frame[name].to_numpy(float) for name in score_names}
    for name, values in scores.items():
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite score: {name}")
    metrics = {name: metric_block(frame, scores[name]) for name in score_names}
    proxy = {
        suite: {
            name: proxy_score(frame, scores[name], scores, families)
            for name in ("HR_mean_top5", "TR_stage2_v1", "TR_HR_rank_50_50")
            if name in scores
        }
        for suite, families in REFERENCE_SUITES.items()
    }
    baseline_mask, _, _, _ = conservative_mask(frame.label, scores[primary_baseline], 0.01)
    candidate_mask, _, _, _ = conservative_mask(frame.label, scores[primary_candidate], 0.01)
    member = frame.label.to_numpy(int) == 1
    bootstrap_pairs = [(primary_candidate, primary_baseline), *extra_bootstrap_pairs]
    bootstraps = {
        f"{left}_minus_{right}": paired_bootstrap(frame, left, right, scores, seed=BOOTSTRAP_SEED + index)
        for index, (left, right) in enumerate(bootstrap_pairs)
    }
    return {
        "metrics": metrics,
        "proxy_definition": "local-only S_proxy=(AUC+novelty_proxy)/2; equal reference-family weight; not the official score",
        "reference_suite_sensitivity": proxy,
        "paired_bootstrap": bootstraps,
        "detection_delta_at_1pct": {
            "new_tp_vs_primary_baseline": int(np.sum(member & candidate_mask & ~baseline_mask)),
            "lost_tp_vs_primary_baseline": int(np.sum(member & baseline_mask & ~candidate_mask)),
        },
        "low_fpr_interpolation_note": "ROC TPR values and conservative actual FP/TP counts are both reported and are not treated as identical",
    }
