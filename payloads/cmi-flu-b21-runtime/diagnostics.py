"""Aggregate-safe diagnostics for B1 anchors and B2 negative controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .aliases import canonicalize_flow_population, canonicalize_strain
from .contracts import DataContractError
from .cv import purged_leave_one_study_out
from .evaluation import run_compact_task, run_hai_compact_for_panels
from .features.flow import build_flow_baseline_features
from .metrics import evaluate_predictions, grouped_metrics, percentile_rank, safe_spearman, signed_percentile_rank
from .negative_controls import (
    DEMOGRAPHIC_FEATURES,
    HAI_CONTEXT_FEATURES,
    direct_hai_control_dataset,
    restricted_ridge_spec,
)
from .runner import InputBundle, build_b02_datasets
from .targets import build_hai_long_target, build_task_11_target, build_task_12_target, build_task_13_target, geometric_mean


def _direction_from_task11_target(target: pd.DataFrame) -> int:
    """Infer the B1 Task1.1 sign from a training partition only."""

    study_values: list[float] = []
    if "study_accession" in target:
        for _, group in target.groupby("study_accession", dropna=False, observed=True):
            metric = safe_spearman(group["pre_vacc"], group["target"])
            if metric.status == "ok" and metric.value != 0:
                study_values.append(metric.value)
    if study_values:
        median = float(np.median(study_values))
        if median != 0:
            return 1 if median > 0 else -1
        sign_sum = int(np.sign(study_values).sum())
        if sign_sum != 0:
            return 1 if sign_sum > 0 else -1
    pooled = safe_spearman(target["pre_vacc"], target["target"])
    if pooled.status == "ok" and pooled.value != 0:
        return 1 if pooled.value > 0 else -1
    return -1


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            clean[str(key)] = value
        records.append(clean)
    return records


def task11_b1_oof_diagnostic(public_cytokine: pd.DataFrame) -> Mapping[str, Any]:
    """Evaluate the B1 Task1.1 rank anchor with sign learned outside each held-out study."""

    target = build_task_11_target(public_cytokine).reset_index(drop=True)
    if "study_accession" not in target or "subject" not in target:
        raise DataContractError("Task1.1 B1 diagnostic requires study_accession and subject")
    splits = purged_leave_one_study_out(target["study_accession"], target["subject"])
    parts: list[pd.DataFrame] = []
    directions: dict[str, int] = {}
    for split in splits:
        train = target.iloc[split.train_indices]
        validation = target.iloc[split.validation_indices].copy()
        direction = _direction_from_task11_target(train)
        directions[split.name] = direction
        validation["prediction"] = signed_percentile_rank(validation["pre_vacc"], direction=direction)
        validation["split"] = split.name
        parts.append(validation[["split", "target", "prediction"]])
    oof = pd.concat(parts, ignore_index=True)
    return {
        "status": "ok",
        "method": "subject-purged leave-one-study-out; sign learned on training studies; held-out-study percentile rank",
        "pooled": evaluate_predictions(oof["target"], oof["prediction"]),
        "folds": _records(grouped_metrics(oof, group_columns=["split"], target_column="target", prediction_column="prediction")),
        "split_count": len(splits),
        "rows": int(len(oof)),
        "target_rows": int(len(target)),
        "studies": int(target["study_accession"].nunique()),
        "directions": directions,
    }


def _flow_anchor_diagnostic(
    public_flow: pd.DataFrame,
    *,
    task: str,
    population: str,
    target: pd.DataFrame,
    mode: str,
) -> Mapping[str, Any]:
    canonical = canonicalize_flow_population(population)
    column = f"flow_rank__{canonical}"
    try:
        features = build_flow_baseline_features(
            public_flow,
            populations=[canonical],
            mode=mode,
            include_study_rank=True,
        )
    except DataContractError as error:
        return {
            "status": "not_evaluable",
            "mode": mode,
            "reason": str(error),
            "target_rows": int(len(target)),
            "matched_rows": 0,
            "coverage": 0.0,
        }
    if column not in features:
        return {
            "status": "not_evaluable",
            "mode": mode,
            "reason": f"rank feature {column!r} is unavailable",
            "target_rows": int(len(target)),
            "matched_rows": 0,
            "coverage": 0.0,
        }
    merge_keys = [key for key in ("participant_id", "subject", "study_accession") if key in target and key in features]
    merged = target.merge(features[[*merge_keys, column]], on=merge_keys, how="inner", validate="one_to_one")
    if merged.empty:
        return {
            "status": "not_evaluable",
            "mode": mode,
            "reason": "target cohort has zero overlap with baseline anchor cohort",
            "target_rows": int(len(target)),
            "matched_rows": 0,
            "coverage": 0.0,
        }
    merged = merged.rename(columns={column: "prediction"})
    fold_columns = ["study_accession"] if "study_accession" in merged else []
    folds = (
        _records(grouped_metrics(merged, group_columns=fold_columns, target_column="target", prediction_column="prediction"))
        if fold_columns
        else []
    )
    return {
        "status": "ok",
        "mode": mode,
        "method": "label-free study/population rank anchor" if mode == "strict" else "label-free study/population/unit/material-stratified rank proxy",
        "pooled": evaluate_predictions(merged["target"], merged["prediction"]),
        "folds": folds,
        "target_rows": int(len(target)),
        "matched_rows": int(len(merged)),
        "coverage": float(len(merged) / len(target)) if len(target) else 0.0,
        "studies": int(merged["study_accession"].nunique()) if "study_accession" in merged else None,
        "task": task,
    }


def flow_b1_anchor_diagnostics(public_flow: pd.DataFrame) -> Mapping[str, Any]:
    """Evaluate exact strict B1 flow anchors and the broad transfer proxy separately."""

    task12 = build_task_12_target(public_flow)
    task13 = build_task_13_target(public_flow, include_sdy272_asc_proxy=False)
    return {
        "Task1.2": {
            "exact_b1_strict": _flow_anchor_diagnostic(
                public_flow,
                task="Task1.2",
                population="Classical_monocytes",
                target=task12,
                mode="strict",
            ),
            "broad_transfer_proxy": _flow_anchor_diagnostic(
                public_flow,
                task="Task1.2",
                population="Classical_monocytes",
                target=task12,
                mode="broad",
            ),
        },
        "Task1.3": {
            "exact_b1_strict": _flow_anchor_diagnostic(
                public_flow,
                task="Task1.3",
                population="Antibody-secreting_cells_(ASC)",
                target=task13,
                mode="strict",
            ),
            "broad_transfer_proxy": _flow_anchor_diagnostic(
                public_flow,
                task="Task1.3",
                population="Antibody-secreting_cells_(ASC)",
                target=task13,
                mode="broad",
            ),
        },
    }


def _hai_b1_proxy(
    public_serology: pd.DataFrame,
    *,
    task: str,
    day: int,
    panel_strains: Sequence[str],
) -> Mapping[str, Any]:
    paired = build_hai_long_target(public_serology, day=day).copy()
    panel = {canonicalize_strain(strain) for strain in panel_strains}
    paired["virus_strain"] = paired["virus_strain"].map(canonicalize_strain)
    working = paired.loc[paired["virus_strain"].isin(panel)].copy()
    if working.empty:
        return {
            "status": "not_evaluable",
            "reason": "no public paired HAI rows overlap requested panel",
            "requested_strains": len(panel),
        }
    groups = [key for key in ("participant_id", "subject", "study_accession") if key in working]
    grouped = (
        working.groupby(groups, dropna=False, observed=True)
        .agg(
            anchor=("pre_hai", geometric_mean),
            target=("post_hai", geometric_mean),
            panel_size=("virus_strain", "nunique"),
        )
        .reset_index()
    )
    if "study_accession" in grouped:
        grouped["prediction"] = grouped.groupby("study_accession", dropna=False, observed=True)["anchor"].transform(
            lambda values: percentile_rank(values.to_numpy(dtype=float))
        )
        folds = _records(grouped_metrics(grouped, group_columns=["study_accession"], target_column="target", prediction_column="prediction"))
    else:
        grouped["prediction"] = percentile_rank(grouped["anchor"])
        folds = []
    return {
        "status": "proxy_only",
        "method": "same observed public panel subset for pre/post geometric means; study-local baseline rank",
        "warning": "public participants do not carry the complete Challenge panel; this is not the true task target",
        "pooled": evaluate_predictions(grouped["target"], grouped["prediction"]),
        "folds": folds,
        "participants": int(grouped["participant_id"].nunique()),
        "studies": int(grouped["study_accession"].nunique()) if "study_accession" in grouped else None,
        "requested_strains": len(panel),
        "available_strains": int(working["virus_strain"].nunique()),
        "panel_size_min": int(grouped["panel_size"].min()),
        "panel_size_max": int(grouped["panel_size"].max()),
        "task": task,
    }


def b1_anchor_diagnostics(inputs: InputBundle) -> Mapping[str, Any]:
    tables = inputs.tables
    flow = flow_b1_anchor_diagnostics(tables["public_flow"])
    return {
        "Task1.1": task11_b1_oof_diagnostic(tables["public_cytokine"]),
        "Task1.2": flow["Task1.2"],
        "Task1.3": flow["Task1.3"],
        "Task1.4": {
            "status": "not_evaluable",
            "reason": "no public AIM labels are available",
        },
        "Task2.1": _hai_b1_proxy(
            tables["public_serology"],
            task="Task2.1",
            day=28,
            panel_strains=inputs.vaccine_strains,
        ),
        "Task2.2": _hai_b1_proxy(
            tables["public_serology"],
            task="Task2.2",
            day=28,
            panel_strains=inputs.challenge_strains,
        ),
        "Task2.3": _hai_b1_proxy(
            tables["public_serology"],
            task="Task2.3",
            day=365,
            panel_strains=inputs.challenge_strains,
        ),
    }


def _compact_task_control(dataset: Any, *, allowed: Sequence[str], name: str, transform: str, random_state: int) -> Mapping[str, Any]:
    spec = restricted_ridge_spec(
        dataset,
        name=name,
        allowed_features=allowed,
        target_transform=transform,
    )
    result = run_compact_task(dataset, specs=[spec], random_state=random_state)
    return {
        "model": result.selected_spec.to_dict(),
        "metrics": result.metrics,
    }


def negative_control_diagnostics(config: Any, inputs: InputBundle) -> Mapping[str, Any]:
    """Run low-information controls under the exact B2 grouped validation contract."""

    datasets = build_b02_datasets(config, inputs)
    results: dict[str, Any] = {}
    for task, transform in (("Task1.1", "log"), ("Task1.2", "log1p"), ("Task1.3", "log1p")):
        dataset = datasets[task]
        results[task] = {
            "age_only": _compact_task_control(
                dataset,
                allowed=("age", "age_missing"),
                name="age_only_ridge",
                transform=transform,
                random_state=config.random_state,
            ),
            "demographics_only": _compact_task_control(
                dataset,
                allowed=DEMOGRAPHIC_FEATURES,
                name="demographics_only_ridge",
                transform=transform,
                random_state=config.random_state,
            ),
        }

    d28 = direct_hai_control_dataset(datasets["HAI_D28"])
    d365 = direct_hai_control_dataset(datasets["HAI_D365"])
    hai_spec_28 = restricted_ridge_spec(
        d28,
        name="demographics_plus_strain_context_ridge",
        allowed_features=HAI_CONTEXT_FEATURES,
        target_transform="identity",
        clip_min=None,
    )
    hai_spec_365 = restricted_ridge_spec(
        d365,
        name="demographics_plus_strain_context_ridge",
        allowed_features=HAI_CONTEXT_FEATURES,
        target_transform="identity",
        clip_min=None,
    )
    d28_results = run_hai_compact_for_panels(
        d28,
        specs=[hai_spec_28],
        selection_panels={"Task2.1": inputs.vaccine_strains, "Task2.2": inputs.challenge_strains},
    )
    d365_results = run_hai_compact_for_panels(
        d365,
        specs=[hai_spec_365],
        selection_panels={"Task2.3": inputs.challenge_strains},
    )
    for task, result in {**d28_results, **d365_results}.items():
        results[task] = {
            "demographics_plus_strain_context": {
                "model": result.selected_spec.to_dict(),
                "metrics": result.metrics,
            }
        }
    results["Task1.4"] = {
        "status": "not_evaluable",
        "reason": "no public AIM labels are available",
    }
    return results
