"""E05 compatibility for frozen runtimes predating panel fold diagnostics.

The original E05 source was authored against an ``evaluation.py`` contract where
``evaluate_hai_panel_proxy`` returns per-split fold metrics/summary and
``HAICandidateEvaluation`` exposes the same fields.  The frozen B2/B2.1 Kaggle
package used as the incumbent runtime predates those additions.  Model fitting
still succeeds there, but E05 control readout used to fail when it accessed the
missing attributes.

This adapter recomputes the exact current panel-fold diagnostics from the
already-produced OOF rows.  It does not change model family, fit inputs,
regularization, interactions, split definitions, weighting, panels, target
construction, donor-rank identity, or organizer sequence handling.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .aliases import canonicalize_strain
from .contracts import DataContractError, require_columns
from .metrics import evaluate_predictions, grouped_metrics, within_group_rank_spearman
from .targets import geometric_mean
from . import strategy_e05_v3 as _base

# strategy_e05_v3 -> strategy_e05_v2 -> strategy_e05
_root = _base._base._base


def _summarize_metric_frame(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Mirror the current equal-fold summary without depending on newer models.py."""

    if frame.empty:
        return {
            "count": 0,
            "spearman_mean": None,
            "spearman_median": None,
            "spearman_min": None,
            "spearman_max": None,
            "spearman_std": None,
            "rmse_mean": None,
            "rmse_median": None,
            "rmse_max": None,
            "rmse_std": None,
        }
    spearman = pd.to_numeric(frame["spearman"], errors="coerce").to_numpy(dtype=float)
    rmse = pd.to_numeric(frame["rmse"], errors="coerce").to_numpy(dtype=float)
    spearman = spearman[np.isfinite(spearman)]
    rmse = rmse[np.isfinite(rmse)]
    return {
        "count": int(spearman.size),
        "spearman_mean": float(np.mean(spearman)) if spearman.size else None,
        "spearman_median": float(np.median(spearman)) if spearman.size else None,
        "spearman_min": float(np.min(spearman)) if spearman.size else None,
        "spearman_max": float(np.max(spearman)) if spearman.size else None,
        "spearman_std": float(np.std(spearman)) if spearman.size else None,
        "rmse_mean": float(np.mean(rmse)) if rmse.size else None,
        "rmse_median": float(np.median(rmse)) if rmse.size else None,
        "rmse_max": float(np.max(rmse)) if rmse.size else None,
        "rmse_std": float(np.std(rmse)) if rmse.size else None,
    }


def _panel_proxy_with_folds(
    oof_predictions: pd.DataFrame,
    *,
    panel_strains: Sequence[str],
) -> Mapping[str, Any]:
    """Current panel-proxy contract reconstructed from OOF rows only."""

    require_columns(
        oof_predictions,
        ["split", "participant_id", "virus_strain", "post_hai", "post_prediction"],
        table_name="E05 panel proxy compatibility OOF",
    )
    panel = {canonicalize_strain(strain) for strain in panel_strains}
    if not panel:
        raise DataContractError("E05 panel proxy compatibility requires a non-empty panel")
    working = oof_predictions.copy()
    working["virus_strain"] = working["virus_strain"].map(canonicalize_strain)
    working = working.loc[working["virus_strain"].isin(panel)].copy()
    if working.empty:
        empty_folds = pd.DataFrame()
        return {
            "metrics": evaluate_predictions([], []),
            "fold_metrics": [],
            "fold_summary": _summarize_metric_frame(empty_folds),
            "participants": 0,
            "available_strains": 0,
            "requested_strains": len(panel),
            "panel_size_min": 0,
            "panel_size_max": 0,
        }
    grouped = (
        working.groupby(["split", "participant_id"], dropna=False, observed=True)
        .agg(
            panel_target=("post_hai", geometric_mean),
            panel_prediction=("post_prediction", geometric_mean),
            panel_size=("virus_strain", "nunique"),
        )
        .reset_index()
    )
    folds = grouped_metrics(
        grouped,
        group_columns=["split"],
        target_column="panel_target",
        prediction_column="panel_prediction",
    )
    return {
        "metrics": evaluate_predictions(grouped["panel_target"], grouped["panel_prediction"]),
        "fold_metrics": folds.to_dict(orient="records"),
        "fold_summary": _summarize_metric_frame(folds),
        "participants": int(grouped["participant_id"].nunique()),
        "available_strains": int(working["virus_strain"].nunique()),
        "requested_strains": len(panel),
        "panel_size_min": int(grouped["panel_size"].min()),
        "panel_size_max": int(grouped["panel_size"].max()),
    }


def _condition_metrics_compat(
    oof: pd.DataFrame,
    *,
    panel_strains: Sequence[str],
) -> Mapping[str, Any]:
    require_columns(
        oof,
        [
            "split",
            "participant_id",
            "subject_group",
            "study_group",
            "virus_strain",
            "post_hai",
            "post_prediction",
        ],
        table_name="E05 condition OOF",
    )
    panel = set(_root._canonical_panel(panel_strains))
    working = oof.copy()
    working["virus_strain"] = working["virus_strain"].map(canonicalize_strain)
    selection = working.loc[working["virus_strain"].isin(panel)].copy()
    if selection.empty:
        raise DataContractError("E05 condition has no task-panel OOF rows")
    proxy = _panel_proxy_with_folds(working, panel_strains=tuple(panel))
    within = within_group_rank_spearman(
        selection,
        group_column="virus_strain",
        target_column="post_hai",
        prediction_column="post_prediction",
    ).to_dict()
    return _root._json_safe(
        {
            "post_hai": evaluate_predictions(
                selection["post_hai"], selection["post_prediction"]
            ),
            "within_strain_spearman": within,
            "panel_proxy": proxy["metrics"],
            "panel_proxy_fold_summary": proxy["fold_summary"],
            "panel_proxy_fold_metrics": proxy["fold_metrics"],
            "panel_proxy_coverage": {
                key: value
                for key, value in proxy.items()
                if key not in {"metrics", "fold_metrics", "fold_summary"}
            },
        }
    )


def _control_run_compat(
    dataset,
    *,
    spec,
    splits,
    panel_strains,
    name: str,
    fit_challenge: bool,
):
    evaluated = _root.evaluate_hai_spec(
        dataset,
        spec=spec,
        splits=splits,
        panel_strains=panel_strains,
    )
    proxy = _panel_proxy_with_folds(
        evaluated.enriched_oof,
        panel_strains=panel_strains,
    )
    if fit_challenge:
        _, target_prediction = _root.fit_final_model(
            dataset.train,
            dataset.challenge,
            target_column=dataset.target_column,
            spec=spec,
            excluded_columns=dataset.excluded_columns,
        )
        post_prediction = dataset.target_prediction_to_post_hai(
            target_prediction,
            frame=dataset.challenge,
        )
        challenge = dataset.challenge[["participant_id", "virus_strain"]].copy()
        challenge["prediction"] = np.asarray(post_prediction, dtype=float)
    else:
        challenge = pd.DataFrame(columns=["participant_id", "virus_strain", "prediction"])
    return _root._ConditionRun(
        name=name,
        oof=evaluated.enriched_oof.copy(),
        challenge_strain_predictions=challenge,
        metrics=_root._json_safe(
            {
                "post_hai": evaluated.post_metrics,
                "within_strain_spearman": evaluated.within_strain_spearman,
                "panel_proxy": proxy["metrics"],
                "panel_proxy_fold_summary": proxy["fold_summary"],
                "panel_proxy_fold_metrics": proxy["fold_metrics"],
                "panel_proxy_coverage": {
                    key: value
                    for key, value in proxy.items()
                    if key not in {"metrics", "fold_metrics", "fold_summary"}
                },
            }
        ),
    )


# E05 resolves both helpers from module globals at call time.  Install a local
# compatibility layer rather than replacing the frozen incumbent package.
_root._condition_metrics = _condition_metrics_compat
_root._control_run = _control_run_compat


def run_strategy_e05(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    result = dict(_base.run_strategy_e05(*args, **kwargs))
    feature_contract = dict(result.get("feature_contract") or {})
    feature_contract.update(
        {
            "panel_proxy_fold_metrics_contract": "recomputed_from_oof_split_participant_panel",
            "legacy_evaluation_without_fold_fields_supported": True,
        }
    )
    result["feature_contract"] = feature_contract
    return result


EXPERIMENT = _base.EXPERIMENT
RIDGE_ALPHA = _base.RIDGE_ALPHA
MAX_INTERACTIONS = _base.MAX_INTERACTIONS
INTERACTION_COLUMNS = _base.INTERACTION_COLUMNS
MAIN_NUMERIC_COLUMNS = _base.MAIN_NUMERIC_COLUMNS
MAIN_CATEGORICAL_COLUMNS = _base.MAIN_CATEGORICAL_COLUMNS
SIMULTANEOUS_HOLDOUTS = _base.SIMULTANEOUS_HOLDOUTS
_e05_design = _base._e05_design
_hierarchical_sample_weights = _base._hierarchical_sample_weights
_simultaneous_subject_strain_splits = _base._simultaneous_subject_strain_splits

__all__ = [
    "EXPERIMENT",
    "RIDGE_ALPHA",
    "MAX_INTERACTIONS",
    "INTERACTION_COLUMNS",
    "MAIN_NUMERIC_COLUMNS",
    "MAIN_CATEGORICAL_COLUMNS",
    "SIMULTANEOUS_HOLDOUTS",
    "_panel_proxy_with_folds",
    "_condition_metrics_compat",
    "_control_run_compat",
    "_e05_design",
    "_hierarchical_sample_weights",
    "_simultaneous_subject_strain_splits",
    "run_strategy_e05",
]
