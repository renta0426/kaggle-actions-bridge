"""Phase A Task1.3 SDY272 gate-harmonization diagnostics.

This module intentionally returns aggregate-only diagnostics. Participant identifiers and
row-level OOF/challenge predictions are used only transiently for alignment and are never
returned by the public experiment API.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .aliases import canonicalize_flow_population
from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns
from .cv import NamedSplit, purged_leave_one_study_out
from .datasets import TaskDataset, build_task_13_dataset
from .evaluation import TaskRunResult, run_compact_task
from .metrics import percentile_rank, safe_spearman, within_group_rank_spearman
from .models import CandidateEvaluation, ModelSpec, evaluate_candidates

PROXY_STUDY = "SDY272"
GATE_CONFIDENCE_WEIGHTS: tuple[float, ...] = (0.50, 0.75, 1.00)
ASC_NAME = "Antibody-secreting_cells_(ASC)"


def sdy272_asc_proxy_mask(flow: pd.DataFrame) -> pd.Series:
    """Return the exact narrow proxy predicate already used by Task1.3 target code."""

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


def harmonize_sdy272_asc_proxy(flow: pd.DataFrame) -> tuple[pd.DataFrame, Mapping[str, int]]:
    """Relabel only gate-qualified SDY272 B-cell rows to canonical ASC."""

    mask = sdy272_asc_proxy_mask(flow)
    if not bool(mask.any()):
        raise DataContractError("no SDY272 rows satisfy the locked ASC proxy predicate")
    result = flow.copy()
    result.loc[mask, "name"] = ASC_NAME
    diagnostics = {
        "matched_rows": int(mask.sum()),
        "matched_participants": int(result.loc[mask, "participant_id"].nunique()),
    }
    return result, diagnostics


def _task13_specs(config: BaselineConfig, *, rank_target: bool) -> list[ModelSpec]:
    specs = config.model_specs("task_13")
    if not specs:
        raise DataContractError("Task1.3 model set is empty")
    if not rank_target:
        return specs
    return [
        replace(spec, target_transform="identity", clip_min=None)
        for spec in specs
    ]


def to_within_study_rank_target(dataset: TaskDataset) -> TaskDataset:
    """Replace the supervised target with a within-study percentile rank."""

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


def _default_loso(dataset: TaskDataset) -> list[NamedSplit]:
    if int(dataset.train["study_group"].nunique()) < 2:
        raise DataContractError("SDY272 harmonization LOSO requires at least two studies")
    return purged_leave_one_study_out(
        dataset.train["study_group"].astype(str),
        dataset.train["subject_group"].astype(str),
    )


def _finite(value: Any, fallback: float = -np.inf) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if np.isfinite(number) else fallback


def _prediction_rank_rmse(dataset: TaskDataset, result: TaskRunResult) -> float:
    oof = result.oof_predictions[["row_index", "target", "prediction"]].copy()
    study = (
        dataset.train[["study_group"]]
        .reset_index()
        .rename(columns={"index": "row_index"})
    )
    oof = oof.merge(study, on="row_index", how="left", validate="many_to_one")
    squared: list[np.ndarray] = []
    for _, group in oof.groupby("study_group", dropna=False, observed=True):
        target_rank = percentile_rank(group["target"].to_numpy(dtype=float))
        prediction_rank = percentile_rank(group["prediction"].to_numpy(dtype=float))
        squared.append(np.square(target_rank - prediction_rank))
    if not squared:
        raise DataContractError("cannot compute Task1.3 rank RMSE without OOF groups")
    return float(np.sqrt(np.mean(np.concatenate(squared))))


def _within_study_spearman(dataset: TaskDataset, result: TaskRunResult) -> Mapping[str, Any]:
    oof = result.oof_predictions[["row_index", "target", "prediction"]].copy()
    study = (
        dataset.train[["study_group"]]
        .reset_index()
        .rename(columns={"index": "row_index"})
    )
    oof = oof.merge(study, on="row_index", how="left", validate="many_to_one")
    metric = within_group_rank_spearman(
        oof,
        group_column="study_group",
        target_column="target",
        prediction_column="prediction",
    )
    return metric.to_dict()


def _fold_records(result: TaskRunResult) -> list[Mapping[str, Any]]:
    columns = [
        column
        for column in (
            "split",
            "n",
            "spearman",
            "spearman_status",
            "rmse",
            "prediction_unique",
        )
        if column in result.fold_metrics.columns
    ]
    return result.fold_metrics[columns].to_dict(orient="records")


def _result_summary(dataset: TaskDataset, result: TaskRunResult) -> Mapping[str, Any]:
    return {
        "selected_model": result.selected_spec.to_dict(),
        "training_rows": int(len(dataset.train)),
        "training_subjects": int(dataset.train["subject_group"].nunique()),
        "training_studies": int(dataset.train["study_group"].nunique()),
        "pooled": result.metrics["pooled"],
        "fold_summary": result.metrics["fold_summary"],
        "within_study_rank_spearman": _within_study_spearman(dataset, result),
        "rank_rmse": _prediction_rank_rmse(dataset, result),
        "fold_metrics": _fold_records(result),
        "candidate_summaries": result.candidate_summaries,
        "candidate_failures": result.candidate_failures,
    }


def _challenge_rank_agreement(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> Mapping[str, Any]:
    require_columns(reference, ["participant_id", "prediction"], table_name="reference predictions")
    require_columns(candidate, ["participant_id", "prediction"], table_name="candidate predictions")
    aligned = reference.merge(
        candidate,
        on="participant_id",
        how="inner",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    if len(aligned) != len(reference) or len(aligned) != len(candidate):
        raise DataContractError("challenge prediction sets do not align one-to-one")
    ref = aligned["prediction_reference"].to_numpy(dtype=float)
    cand = aligned["prediction_candidate"].to_numpy(dtype=float)
    ref_rank = percentile_rank(ref)
    cand_rank = percentile_rank(cand)
    difference = np.abs(ref_rank - cand_rank)
    return {
        "n": int(len(aligned)),
        "rank_spearman": safe_spearman(ref, cand).to_dict(),
        "mean_absolute_percentile_difference": float(np.mean(difference)),
        "max_absolute_percentile_difference": float(np.max(difference)),
    }


def _study_from_split(split: object) -> str:
    text = str(split)
    prefix = "study="
    return text[len(prefix) :] if text.startswith(prefix) else text


def _weighted_candidate_key(
    evaluation: CandidateEvaluation,
    *,
    proxy_study: str,
    proxy_weight: float,
) -> tuple[float, float, float, float, float, str]:
    folds = evaluation.fold_metrics.copy()
    if folds.empty or "split" not in folds or "spearman" not in folds:
        return (-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, evaluation.spec.name)
    folds["study"] = folds["split"].map(_study_from_split)
    spearman = pd.to_numeric(folds["spearman"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(spearman)
    if not finite.any():
        return (-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, evaluation.spec.name)
    weights = np.where(
        folds["study"].astype(str).str.upper().to_numpy() == proxy_study.upper(),
        float(proxy_weight),
        1.0,
    )
    weighted_mean = float(np.average(spearman[finite], weights=weights[finite]))
    ordinary = spearman[finite]
    pooled_spearman = _finite(evaluation.pooled_metrics["spearman"]["value"])
    pooled_rmse = _finite(evaluation.pooled_metrics["rmse"]["value"], np.inf)
    return (
        weighted_mean,
        float(np.mean(ordinary)),
        float(np.min(ordinary)),
        pooled_spearman,
        -pooled_rmse,
        evaluation.spec.name,
    )


def gate_confidence_selection_stability(
    dataset: TaskDataset,
    *,
    specs: Sequence[ModelSpec],
    splits: Sequence[NamedSplit],
    proxy_study: str = PROXY_STUDY,
    confidence_weights: Sequence[float] = GATE_CONFIDENCE_WEIGHTS,
) -> Mapping[str, Any]:
    """Audit candidate selection when the proxy validation fold is downweighted."""

    search = evaluate_candidates(
        dataset.train,
        target_column=dataset.target_column,
        splits=splits,
        specs=specs,
        excluded_columns=dataset.excluded_columns,
        aggregate_repeats=False,
    )
    if not search.evaluations:
        raise DataContractError("gate-confidence sensitivity has no successful candidates")

    results: list[Mapping[str, Any]] = []
    for weight in confidence_weights:
        if not 0.0 < float(weight) <= 1.0:
            raise DataContractError(f"invalid gate confidence weight: {weight}")
        selected = max(
            search.evaluations,
            key=lambda evaluation: _weighted_candidate_key(
                evaluation,
                proxy_study=proxy_study,
                proxy_weight=float(weight),
            ),
        )
        key = _weighted_candidate_key(
            selected,
            proxy_study=proxy_study,
            proxy_weight=float(weight),
        )
        fold_spearman = {
            _study_from_split(row["split"]): (
                None if not np.isfinite(float(row["spearman"])) else float(row["spearman"])
            )
            for row in selected.fold_metrics[["split", "spearman"]].to_dict(orient="records")
        }
        results.append(
            {
                "proxy_validation_weight": float(weight),
                "selected_model": selected.spec.name,
                "weighted_spearman_mean": key[0],
                "ordinary_spearman_mean": key[1],
                "ordinary_spearman_min": key[2],
                "pooled_spearman": key[3],
                "rmse": -key[4],
                "fold_spearman": fold_spearman,
            }
        )
    return {
        "proxy_study": proxy_study,
        "weights": [float(value) for value in confidence_weights],
        "results": results,
        "candidate_failures": search.failures,
        "selected_model_stable": len({item["selected_model"] for item in results}) == 1,
    }


def _promotion_gate(rank_result: TaskRunResult) -> Mapping[str, Any]:
    fold_spearman: dict[str, float | None] = {}
    for row in rank_result.fold_metrics[["split", "spearman"]].to_dict(orient="records"):
        value = float(row["spearman"])
        fold_spearman[_study_from_split(row["split"])] = value if np.isfinite(value) else None
    values = list(fold_spearman.values())
    passed = bool(
        len(values) >= 2
        and all(value is not None and float(value) > 0.0 for value in values)
    )
    return {
        "rule": "positive Spearman in every usable held-study direction",
        "passed": passed,
        "fold_spearman": fold_spearman,
    }


def run_task13_sdy272_harmonization_experiment(
    config: BaselineConfig,
    inputs: Any,
) -> Mapping[str, Any]:
    """Run the locked aggregate-only Task1.3 SDY272 experiment."""

    selection_policy = str(
        config.section("selection", required=False).get("policy", "legacy")
    )
    if selection_policy != "robust_v1":
        raise DataContractError("Task1.3 SDY272 experiment requires selection.policy=robust_v1")
    flow_config = config.section("flow")
    if str(flow_config.get("task_13_mode", "strict")) != "broad":
        raise DataContractError("Task1.3 SDY272 experiment requires B2.1 broad flow mode")

    tables = inputs.tables
    public_flow = tables["public_flow"]
    challenge_flow = tables["challenge_flow"]
    participants = tables["participants"]
    investigations = tables["investigations"]

    strict = build_task_13_dataset(
        public_flow,
        challenge_flow,
        participants,
        investigations,
        mode="broad",
        include_sdy272_asc_proxy=False,
    )
    proxy_target_only = build_task_13_dataset(
        public_flow,
        challenge_flow,
        participants,
        investigations,
        mode="broad",
        include_sdy272_asc_proxy=True,
    )
    harmonized_flow, gate_diagnostics = harmonize_sdy272_asc_proxy(public_flow)
    harmonized = build_task_13_dataset(
        harmonized_flow,
        challenge_flow,
        participants,
        investigations,
        mode="broad",
        include_sdy272_asc_proxy=False,
    )
    if len(proxy_target_only.train) != len(harmonized.train):
        raise DataContractError(
            "target-only and fully harmonized Task1.3 cohorts differ in training row count"
        )
    if PROXY_STUDY not in set(harmonized.train["study_group"].astype(str)):
        raise DataContractError("harmonized Task1.3 dataset does not contain SDY272")

    raw_specs = _task13_specs(config, rank_target=False)
    rank_specs = _task13_specs(config, rank_target=True)

    strict_result = run_compact_task(
        strict,
        specs=raw_specs,
        random_state=config.random_state,
        selection_policy="robust_v1",
    )
    target_only_result = run_compact_task(
        proxy_target_only,
        specs=raw_specs,
        random_state=config.random_state,
        selection_policy="robust_v1",
    )
    harmonized_raw_result = run_compact_task(
        harmonized,
        specs=raw_specs,
        random_state=config.random_state,
        selection_policy="robust_v1",
    )

    rank_dataset = to_within_study_rank_target(harmonized)
    rank_splits = _default_loso(rank_dataset)
    harmonized_rank_result = run_compact_task(
        rank_dataset,
        specs=rank_specs,
        splits=rank_splits,
        random_state=config.random_state,
        selection_policy="robust_v1",
    )
    confidence = gate_confidence_selection_stability(
        rank_dataset,
        specs=rank_specs,
        splits=rank_splits,
    )
    one_weight = next(
        item for item in confidence["results"] if item["proxy_validation_weight"] == 1.0
    )
    confidence = {
        **confidence,
        "weight_1_matches_robust_v1": (
            one_weight["selected_model"] == harmonized_rank_result.selected_spec.name
        ),
    }

    strict_predictions = strict_result.challenge_predictions
    agreements = {
        "proxy_target_only_vs_strict": _challenge_rank_agreement(
            strict_predictions, target_only_result.challenge_predictions
        ),
        "gate_harmonized_raw_vs_strict": _challenge_rank_agreement(
            strict_predictions, harmonized_raw_result.challenge_predictions
        ),
        "gate_harmonized_rank_vs_strict": _challenge_rank_agreement(
            strict_predictions, harmonized_rank_result.challenge_predictions
        ),
        "gate_harmonized_rank_vs_raw": _challenge_rank_agreement(
            harmonized_raw_result.challenge_predictions,
            harmonized_rank_result.challenge_predictions,
        ),
    }

    return {
        "experiment": "phase_a_task13_sdy272_harmonization",
        "selection_policy": "robust_v1",
        "flow_mode": "broad",
        "proxy_study": PROXY_STUDY,
        "gate_diagnostics": gate_diagnostics,
        "conditions": {
            "strict": _result_summary(strict, strict_result),
            "proxy_target_only": _result_summary(proxy_target_only, target_only_result),
            "gate_harmonized_raw": _result_summary(harmonized, harmonized_raw_result),
            "gate_harmonized_rank": _result_summary(rank_dataset, harmonized_rank_result),
        },
        "gate_confidence_selection": confidence,
        "challenge_rank_agreements": agreements,
        "promotion_gate": _promotion_gate(harmonized_rank_result),
        "leaderboard_used_for_selection": False,
        "competition_submission_attempted": False,
        "output_policy": "aggregate_only_no_participant_ids_or_row_level_predictions",
    }
