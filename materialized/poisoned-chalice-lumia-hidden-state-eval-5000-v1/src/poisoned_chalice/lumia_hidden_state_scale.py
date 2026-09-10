"""Frozen 5,000-row scale/stability evaluator for P1-03 LUMIA hidden states.

This module preserves the 1k probe architecture, optimizer, layer-selection rule,
and outer-OOF semantics.  Only the cohort size and therefore exact split counts
change: 5,000 total rows -> 4,000 outer-fit / 1,000 holdout, with the outer-fit
split 2,400 / 800 / 800 for probe training, early stopping, and layer selection.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from poisoned_chalice.lumia_hidden_state_probe import (
    HIDDEN_SCORE_NAMES,
    LumiaProbeConfig,
    paired_bootstrap_delta,
    predict_probe,
    train_probe,
    variant_layer_matrix,
)

LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
EXPECTED_ROWS = 5000
EXPECTED_PER_LANGUAGE_LABEL = 500
EXPECTED_SPLIT_SIZES = (2400, 800, 800, 1000)


def outer_and_inner_splits_5000(frame: pd.DataFrame, config: LumiaProbeConfig):
    from sklearn.model_selection import StratifiedKFold, train_test_split

    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(frame)}")
    counts = frame.groupby(["language", "label"]).size().to_dict()
    expected = {(language, label): EXPECTED_PER_LANGUAGE_LABEL for language in LANGUAGES for label in (0, 1)}
    if counts != expected:
        raise ValueError(f"unexpected 5k language-label balance: {counts}")

    indices = np.arange(len(frame))
    strata = frame.language.astype(str) + "_" + frame.label.astype(str)
    splitter = StratifiedKFold(n_splits=config.outer_folds, shuffle=True, random_state=config.seed)
    result = []
    for fold, (outer_fit, outer_hold) in enumerate(splitter.split(indices, strata)):
        fit_strata = strata.iloc[outer_fit].to_numpy()
        trainval, select = train_test_split(
            outer_fit,
            test_size=0.20,
            random_state=config.seed + 1000 + fold,
            stratify=fit_strata,
        )
        trainval_strata = strata.iloc[trainval].to_numpy()
        train, early = train_test_split(
            trainval,
            test_size=0.25,
            random_state=config.seed + 2000 + fold,
            stratify=trainval_strata,
        )
        if (len(train), len(early), len(select), len(outer_hold)) != EXPECTED_SPLIT_SIZES:
            raise RuntimeError("frozen 5k outer/inner split sizes changed")
        expected_per_cell = (240, 80, 80, 100)
        for subset, per_cell in zip((train, early, select, outer_hold), expected_per_cell):
            observed = frame.iloc[subset].groupby(["language", "label"]).size()
            if len(observed) != 10 or not observed.eq(per_cell).all():
                raise RuntimeError("5k language-label stratification changed within a split")
        sets = [set(train), set(early), set(select), set(outer_hold)]
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                if sets[left] & sets[right]:
                    raise RuntimeError("5k outer/inner split overlap detected")
        if set().union(*sets) != set(indices):
            raise RuntimeError("5k outer/inner split coverage changed")
        result.append((train, early, select, outer_hold))
    return result


def run_outer_oof_5000(
    arrays: dict[str, np.ndarray],
    frame: pd.DataFrame,
    config: LumiaProbeConfig,
    *,
    device: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Run the frozen 1k probe protocol at the predeclared 5k split sizes."""
    from sklearn.metrics import roc_auc_score

    if len(frame) != EXPECTED_ROWS or sorted(frame.label.unique()) != [0, 1]:
        raise ValueError("frozen 5k LUMIA evaluation frame changed")
    layers = arrays["mean"].shape[1]
    if layers != 30:
        raise ValueError(f"expected 30 decoder layers, got {layers}")
    splits = outer_and_inner_splits_5000(frame, config)
    predictions = {name: np.full(len(frame), np.nan, dtype=np.float64) for name in HIDDEN_SCORE_NAMES}
    fold_records: list[dict[str, Any]] = []
    y = frame.label.to_numpy(int)
    variants = ("mean_only", "lumia_caller_literal", "lumia_helper_language_aware")

    for fold, (train_index, early_index, select_index, hold_index) in enumerate(splits):
        for variant_index, variant in enumerate(variants):
            selection: list[tuple[int, float]] = []
            hold_predictions: dict[int, np.ndarray] = {}
            layer_records = []
            for layer in range(layers):
                X = variant_layer_matrix(arrays, variant, layer)
                seed = config.seed + fold * 10000 + variant_index * 1000 + layer
                probe, training_record = train_probe(
                    X[train_index], y[train_index], X[early_index], y[early_index],
                    config=config, seed=seed, device=device,
                )
                select_prediction = predict_probe(
                    probe, X[select_index], batch_size=config.batch_size, device=device
                )
                selection_auc = float(roc_auc_score(y[select_index], select_prediction))
                hold_prediction = predict_probe(
                    probe, X[hold_index], batch_size=config.batch_size, device=device
                )
                selection.append((layer, selection_auc))
                hold_predictions[layer] = hold_prediction
                layer_records.append({"layer": layer, "selection_auc": selection_auc, **training_record})
                del probe
            ordered = sorted(selection, key=lambda item: (-item[1], item[0]))
            best_layer = ordered[0][0]
            top_layers = [layer for layer, _ in ordered[:config.ensemble_size]]
            predictions[f"{variant}_best_layer"][hold_index] = hold_predictions[best_layer]
            predictions[f"{variant}_top5_ensemble"][hold_index] = np.mean(
                np.stack([hold_predictions[layer] for layer in top_layers], axis=0), axis=0
            )
            fold_records.append({
                "fold": fold,
                "variant": variant,
                "best_layer": best_layer,
                "top5_layers": top_layers,
                "layer_records": layer_records,
            })
    for name, values in predictions.items():
        if values.shape != (EXPECTED_ROWS,) or not np.isfinite(values).all():
            raise RuntimeError(f"5k OOF prediction coverage failed: {name}")
    return predictions, fold_records


def paired_increment_bootstraps(
    frame: pd.DataFrame,
    hidden_scores: dict[str, np.ndarray],
    config: LumiaProbeConfig,
) -> dict[str, dict[str, dict[str, float]]]:
    comparisons = {
        "caller_top5_minus_mean_top5": (
            "lumia_caller_literal_top5_ensemble", "mean_only_top5_ensemble"
        ),
        "helper_top5_minus_mean_top5": (
            "lumia_helper_language_aware_top5_ensemble", "mean_only_top5_ensemble"
        ),
        "helper_top5_minus_caller_top5": (
            "lumia_helper_language_aware_top5_ensemble", "lumia_caller_literal_top5_ensemble"
        ),
    }
    result = {}
    for index, (name, (left, right)) in enumerate(comparisons.items()):
        result[name] = paired_bootstrap_delta(
            frame,
            hidden_scores[left],
            hidden_scores[right],
            replicates=config.bootstrap_replicates,
            seed=config.seed + 80000 + index,
        )
    return result
