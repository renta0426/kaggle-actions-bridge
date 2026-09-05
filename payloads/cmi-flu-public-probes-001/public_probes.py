"""Controlled Public-LB probe generation for promoted Task1.2/Task1.3 experts.

This module does not contact Kaggle. It takes an already-frozen B2.1 submission as
its backbone, computes exactly two previously selected challenge experts, and
returns three submission variants whose only changed columns are predeclared.

The three probes are a single experiment family. Their composition is fixed before
any new Public score is observed; Public outcomes must not be used to create further
micro-variants.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np
import pandas as pd

from .aliases import canonicalize_flow_population
from .configuration import BaselineConfig
from .contracts import DataContractError, TASK_COLUMNS, require_columns, require_finite, validate_submission
from .datasets import TaskDataset, build_task_13_dataset
from .metrics import percentile_rank, safe_spearman
from .models import ModelSpec, fit_final_model
from .runner import InputBundle, build_b02_datasets

TASK12_ANCHOR_COLUMN = "flow_rank__Classical_monocytes"
TASK12_MODEL_NAME = "et_d5_l5_sqrt"
TASK12_LAMBDA = 0.5
TASK13_MODEL_NAME = "enet_a0.1_l0.5"
RESIDUAL_TARGET_COLUMN = "__public_probe_anchor_residual"
PROBE_NAMES = ("task13_only", "task12_only", "task12_task13")
PROXY_STUDY = "SDY272"
ASC_NAME = "Antibody-secreting_cells_(ASC)"


def _identity_spec(spec: ModelSpec) -> ModelSpec:
    return replace(spec, target_transform="identity", clip_min=None)


def _find_spec(config: BaselineConfig, model_set: str, name: str) -> ModelSpec:
    matches = [spec for spec in config.model_specs(model_set) if spec.name == name]
    if len(matches) != 1:
        raise DataContractError(f"locked model {model_set}/{name} resolved to {len(matches)} specs")
    return matches[0]


def _within_study_target_rank(frame: pd.DataFrame, *, target_column: str) -> np.ndarray:
    require_columns(frame, ["study_group", target_column], table_name="Public probe target-rank frame")
    ranked = frame.groupby("study_group", dropna=False, observed=True)[target_column].transform(
        lambda values: percentile_rank(pd.to_numeric(values, errors="raise").to_numpy(dtype=float))
    )
    result = pd.to_numeric(ranked, errors="raise").to_numpy(dtype=float)
    require_finite(result, name=f"{target_column}.within_study_rank")
    return result


def _sdy272_asc_proxy_mask(flow: pd.DataFrame) -> pd.Series:
    """Exact locked proxy predicate from the promoted Task1.3 experiment."""

    require_columns(
        flow,
        ["study_accession", "name", "population_definition"],
        table_name="SDY272 ASC proxy source",
    )
    canonical_name = flow["name"].map(canonicalize_flow_population)
    definition = flow["population_definition"].fillna("").astype(str).str.upper()
    return (
        flow["study_accession"].astype(str).str.upper().eq(PROXY_STUDY)
        & canonical_name.astype(str).str.casefold().eq("b_cells")
        & definition.str.contains("CD19+", regex=False)
        & definition.str.contains(r"(?:MS4A1|CD20)-", regex=True)
        & definition.str.contains("CD27+", regex=False)
        & definition.str.contains("CD38++", regex=False)
    )


def _harmonize_sdy272_asc_proxy(flow: pd.DataFrame) -> tuple[pd.DataFrame, Mapping[str, int]]:
    mask = _sdy272_asc_proxy_mask(flow)
    if not bool(mask.any()):
        raise DataContractError("no SDY272 rows satisfy the locked ASC proxy predicate")
    result = flow.copy()
    result.loc[mask, "name"] = ASC_NAME
    return result, {
        "matched_rows": int(mask.sum()),
        "matched_participants": int(result.loc[mask, "participant_id"].nunique()),
    }


def _to_within_study_rank_target(dataset: TaskDataset) -> TaskDataset:
    dataset.validate()
    train = dataset.train.copy()
    target_column = dataset.target_column
    train["target_rank"] = train.groupby(
        "study_group", dropna=False, observed=True
    )[target_column].transform(lambda values: percentile_rank(values.to_numpy(dtype=float)))
    excluded = tuple(dict.fromkeys([*dataset.excluded_columns, target_column]))
    ranked = TaskDataset(
        task=dataset.task,
        train=train,
        challenge=dataset.challenge.copy(),
        target_column="target_rank",
        excluded_columns=excluded,
        metadata={**dict(dataset.metadata), "target_representation": "within_study_rank"},
    )
    ranked.validate()
    return ranked


def task12_anchor_residual_prediction(
    config: BaselineConfig,
    inputs: InputBundle,
) -> pd.DataFrame:
    """Reproduce the locked Task1.2 anchor-residual champion on all challenge rows."""

    dataset = build_b02_datasets(config, inputs)["Task1.2"]
    require_columns(
        dataset.train,
        ["participant_id", "study_group", dataset.target_column, TASK12_ANCHOR_COLUMN],
        table_name="Task1.2 Public-probe training frame",
    )
    require_columns(
        dataset.challenge,
        ["participant_id", TASK12_ANCHOR_COLUMN],
        table_name="Task1.2 Public-probe challenge frame",
    )
    spec = _identity_spec(_find_spec(config, "task_12", TASK12_MODEL_NAME))
    train = dataset.train.copy()
    train_anchor = pd.to_numeric(train[TASK12_ANCHOR_COLUMN], errors="raise").to_numpy(dtype=float)
    challenge_anchor = pd.to_numeric(
        dataset.challenge[TASK12_ANCHOR_COLUMN], errors="raise"
    ).to_numpy(dtype=float)
    require_finite(train_anchor, name="Task1.2 train anchor")
    require_finite(challenge_anchor, name="Task1.2 challenge anchor")
    train[RESIDUAL_TARGET_COLUMN] = (
        _within_study_target_rank(train, target_column=dataset.target_column) - train_anchor
    )
    excluded = tuple(dict.fromkeys([*dataset.excluded_columns, dataset.target_column]))
    _, correction = fit_final_model(
        train,
        dataset.challenge,
        target_column=RESIDUAL_TARGET_COLUMN,
        spec=spec,
        excluded_columns=excluded,
    )
    correction = np.asarray(correction, dtype=float)
    require_finite(correction, name="Task1.2 challenge residual correction")
    score = challenge_anchor + TASK12_LAMBDA * correction
    require_finite(score, name="Task1.2 challenge combined score")
    result = dataset.challenge[["participant_id"]].copy()
    result["prediction"] = score
    return result


def task13_gate_rank_prediction(
    config: BaselineConfig,
    inputs: InputBundle,
) -> pd.DataFrame:
    """Reproduce the promoted SDY272 gate-harmonized rank-target Task1.3 expert."""

    tables = inputs.tables
    harmonized_flow, diagnostics = _harmonize_sdy272_asc_proxy(tables["public_flow"])
    if diagnostics.get("matched_participants") != 45:
        raise DataContractError(
            f"locked SDY272 gate expected 45 participants, found {diagnostics.get('matched_participants')}"
        )
    harmonized = build_task_13_dataset(
        harmonized_flow,
        tables["challenge_flow"],
        tables["participants"],
        tables["investigations"],
        mode="broad",
        include_sdy272_asc_proxy=False,
    )
    if int(harmonized.train["study_group"].nunique()) != 2 or len(harmonized.train) != 68:
        raise DataContractError("locked Task1.3 harmonized cohort must be 68 rows across two studies")
    ranked = _to_within_study_rank_target(harmonized)
    spec = _identity_spec(_find_spec(config, "task_13", TASK13_MODEL_NAME))
    _, prediction = fit_final_model(
        ranked.train,
        ranked.challenge,
        target_column=ranked.target_column,
        spec=spec,
        excluded_columns=ranked.excluded_columns,
    )
    score = np.asarray(prediction, dtype=float)
    require_finite(score, name="Task1.3 gate-rank challenge score")
    result = ranked.challenge[["participant_id"]].copy()
    result["prediction"] = score
    return result


def _aligned_prediction(base: pd.DataFrame, prediction: pd.DataFrame, *, task: str) -> np.ndarray:
    require_columns(base, ["participant_id"], table_name="B2.1 probe backbone")
    require_columns(prediction, ["participant_id", "prediction"], table_name=f"{task} probe prediction")
    if prediction["participant_id"].duplicated().any():
        raise DataContractError(f"{task} probe prediction has duplicate participant IDs")
    indexed = prediction.set_index("participant_id")["prediction"]
    aligned = base["participant_id"].map(indexed)
    if aligned.isna().any():
        raise DataContractError(f"{task} probe prediction does not cover all backbone participants")
    values = pd.to_numeric(aligned, errors="raise").to_numpy(dtype=float)
    require_finite(values, name=f"{task} aligned probe prediction")
    return values


def assemble_controlled_probes(
    base_submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    *,
    task12_prediction: pd.DataFrame,
    task13_prediction: pd.DataFrame,
) -> tuple[Mapping[str, pd.DataFrame], Mapping[str, object]]:
    """Create exactly three predeclared variants from an immutable B2.1 backbone."""

    validate_submission(base_submission, sample_submission, require_nonconstant_public_tasks=True)
    if tuple(base_submission.columns) != tuple(sample_submission.columns):
        raise DataContractError("B2.1 backbone submission columns differ from sample template")

    task12 = _aligned_prediction(base_submission, task12_prediction, task="Task1.2")
    task13 = _aligned_prediction(base_submission, task13_prediction, task="Task1.3")

    variants: dict[str, pd.DataFrame] = {}
    changed = {
        "task13_only": ("Task1.3",),
        "task12_only": ("Task1.2",),
        "task12_task13": ("Task1.2", "Task1.3"),
    }
    for name in PROBE_NAMES:
        frame = base_submission.copy(deep=True)
        if "Task1.2" in changed[name]:
            frame["Task1.2"] = task12
        if "Task1.3" in changed[name]:
            frame["Task1.3"] = task13
        validate_submission(frame, sample_submission, require_nonconstant_public_tasks=True)
        for column in TASK_COLUMNS:
            if column in changed[name]:
                continue
            left = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            right = pd.to_numeric(base_submission[column], errors="raise").to_numpy(dtype=float)
            if not np.array_equal(left, right, equal_nan=True):
                raise DataContractError(f"probe {name} unexpectedly changed backbone column {column}")
        variants[name] = frame

    diagnostics = {
        "probe_names": list(PROBE_NAMES),
        "changed_columns": {key: list(value) for key, value in changed.items()},
        "task12_model": TASK12_MODEL_NAME,
        "task12_lambda": TASK12_LAMBDA,
        "task13_model": TASK13_MODEL_NAME,
        "task12_vs_b21_rank_spearman": safe_spearman(
            task12, pd.to_numeric(base_submission["Task1.2"], errors="raise").to_numpy(dtype=float)
        ).to_dict(),
        "task13_vs_b21_rank_spearman": safe_spearman(
            task13, pd.to_numeric(base_submission["Task1.3"], errors="raise").to_numpy(dtype=float)
        ).to_dict(),
        "public_scores_used_to_define_family": False,
    }
    return variants, diagnostics


def build_controlled_public_probes(
    config: BaselineConfig,
    inputs: InputBundle,
    base_submission: pd.DataFrame,
) -> tuple[Mapping[str, pd.DataFrame], Mapping[str, object]]:
    if config.baseline != "b021_taskwise_robust":
        raise DataContractError("controlled Public probes require the B2.1 robust configuration")
    if str(config.section("selection").get("policy", "")) != "robust_v1":
        raise DataContractError("controlled Public probes require selection.policy=robust_v1")
    task12 = task12_anchor_residual_prediction(config, inputs)
    task13 = task13_gate_rank_prediction(config, inputs)
    return assemble_controlled_probes(
        base_submission,
        inputs.tables["sample_submission"],
        task12_prediction=task12,
        task13_prediction=task13,
    )
