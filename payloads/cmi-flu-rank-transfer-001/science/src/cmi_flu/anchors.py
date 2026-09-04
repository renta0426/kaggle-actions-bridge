"""Rank-only same-readout anchors for the first Part 1 submission."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .aliases import canonicalize_cytokine, canonicalize_flow_population
from .contracts import DataContractError, require_columns, require_unique
from .metrics import safe_spearman, signed_percentile_rank
from .targets import (
    build_hai_anchor,
    build_task_11_target,
    build_task_14_anchor,
    select_pre_vacc,
)

Canonicalizer = Callable[[object], str]


def _single_readout_anchor(
    frame: pd.DataFrame,
    *,
    feature_column: str,
    feature_value: str,
    canonicalizer: Canonicalizer,
    strict_flow: bool = False,
) -> pd.DataFrame:
    require_columns(frame, ["participant_id", "timepoint", feature_column, "value"])
    working = frame.copy()
    working[feature_column] = working[feature_column].map(canonicalizer)
    wanted = canonicalizer(feature_value)
    working = working.loc[working[feature_column] == wanted].copy()
    if strict_flow:
        require_columns(working, ["unit", "material"])
        unit = working["unit"].fillna("").astype(str).str.strip().str.casefold()
        material = working["material"].fillna("").astype(str).str.strip().str.casefold()
        working = working.loc[
            unit.isin({"percentage", "% of parent", "percent", "%"})
            & material.str.startswith("pbmc")
        ]
    ids = [
        column
        for column in ("participant_id", "subject", "study_accession")
        if column in working
    ]
    baseline = select_pre_vacc(
        working,
        group_columns=[*ids, feature_column],
        output_column="anchor",
    )
    if baseline.empty:
        raise DataContractError(f"no anchor rows found for {feature_value!r}")
    return baseline[[*ids, "anchor"]]


def infer_task_11_anchor_direction(public_cytokine: pd.DataFrame) -> int:
    """Infer sign from labels, preferring consistency across studies."""

    target = build_task_11_target(public_cytokine)
    study_values: list[float] = []
    if "study_accession" in target.columns:
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


def build_b1_anchor_predictions(
    *,
    public_cytokine: pd.DataFrame,
    challenge_cytokine: pd.DataFrame,
    challenge_flow: pd.DataFrame,
    challenge_aim: pd.DataFrame,
    challenge_serology: pd.DataFrame,
    vaccine_strains: Sequence[str],
    challenge_strains: Sequence[str],
    task_11_direction: int | str = "auto",
) -> tuple[dict[str, pd.DataFrame], Mapping[str, Any]]:
    """Build rank-only predictions for all seven Part 1 tasks."""

    if task_11_direction == "auto":
        direction_11 = infer_task_11_anchor_direction(public_cytokine)
    else:
        direction_11 = int(task_11_direction)
        if direction_11 not in {-1, 1}:
            raise DataContractError("Task1.1 anchor direction must be auto, -1, or 1")

    anchors: dict[str, pd.DataFrame] = {
        "Task1.1": _single_readout_anchor(
            challenge_cytokine,
            feature_column="analyte",
            feature_value="CXCL10",
            canonicalizer=canonicalize_cytokine,
        ),
        "Task1.2": _single_readout_anchor(
            challenge_flow,
            feature_column="name",
            feature_value="Classical_monocytes",
            canonicalizer=canonicalize_flow_population,
            strict_flow=True,
        ),
        "Task1.3": _single_readout_anchor(
            challenge_flow,
            feature_column="name",
            feature_value="Antibody-secreting_cells_(ASC)",
            canonicalizer=canonicalize_flow_population,
            strict_flow=True,
        ),
        "Task1.4": build_task_14_anchor(challenge_aim),
        "Task2.1": build_hai_anchor(
            challenge_serology,
            panel_strains=vaccine_strains,
            output_column="anchor",
        ),
        "Task2.2": build_hai_anchor(
            challenge_serology,
            panel_strains=challenge_strains,
            output_column="anchor",
        ),
        "Task2.3": build_hai_anchor(
            challenge_serology,
            panel_strains=challenge_strains,
            output_column="anchor",
        ),
    }
    directions = {
        "Task1.1": direction_11,
        "Task1.2": 1,
        "Task1.3": 1,
        "Task1.4": 1,
        "Task2.1": 1,
        "Task2.2": 1,
        "Task2.3": 1,
    }
    predictions: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, Any] = {"directions": directions, "tasks": {}}
    for task, anchor in anchors.items():
        require_unique(anchor, ["participant_id"], table_name=f"{task} anchor")
        prediction = anchor[["participant_id"]].copy()
        prediction["prediction"] = signed_percentile_rank(
            anchor["anchor"], direction=directions[task]
        )
        predictions[task] = prediction
        diagnostics["tasks"][task] = {
            "rows": len(prediction),
            "anchor_unique": int(anchor["anchor"].nunique(dropna=False)),
            "prediction_unique": int(prediction["prediction"].nunique(dropna=False)),
        }
    return predictions, diagnostics
