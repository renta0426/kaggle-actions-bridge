"""Phase B Task1.1 prior-immunity late-fusion diagnostic.

This experiment asks whether low-dimensional pre-vaccination HAI summaries add
cross-study ranking signal to the existing Task1.1 cytokine systems.  Serology
is treated as an optional expert: participants without baseline HAI always fall
back to the fixed base system, so no complete-case filtering is introduced.

Held-study outcomes never enter serology feature construction or feature ranks.
No participant-level predictions are returned by the public experiment API.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns, require_finite
from .evaluation import default_splits_for_task
from .features.serology import build_hai_baseline_long, build_hai_panel_summaries
from .metrics import percentile_rank, safe_spearman
from .models import ModelSpec, fit_final_model
from .runner import InputBundle, build_b02_datasets


TASK = "Task1.1"
ANCHOR_COLUMN = "cytokine_rank__CXCL10"
B21_MODEL_NAME = "pls_2"
ANCHOR_RESIDUAL_MODEL_NAME = "pls_1"
ANCHOR_RESIDUAL_LAMBDA = 0.25
SERology_MODEL = ModelSpec(
    name="prior_immunity_ridge_a10",
    family="ridge",
    params={"alpha": 10.0},
    target_transform="identity",
    clip_min=None,
)
FUSION_WEIGHTS = (0.25, 0.5)
WORST_STUDY_TOLERANCE = 0.10
MIN_HELD_SEROLOGY_ROWS = 5
_EPS = 1e-12

# Deliberately exclude panel count and vaccine-membership fields: those encode
# study assay design / legacy metadata more directly than participant biology.
SEROLOGY_SUMMARY_COLUMNS = (
    "hai_log2_mean",
    "hai_log2_median",
    "hai_log2_min",
    "hai_log2_max",
    "hai_log2_std",
    "hai_breadth_ge_10",
    "hai_breadth_ge_20",
    "hai_breadth_ge_40",
    "hai_breadth_ge_80",
)
SEROLOGY_RANK_COLUMNS = tuple(f"prior_rank__{column}" for column in SEROLOGY_SUMMARY_COLUMNS)
SEROLOGY_TARGET_COLUMN = "__prior_immunity_target_rank"
ANCHOR_RESIDUAL_TARGET_COLUMN = "__task11_anchor_residual"


def _finite_mean(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else None


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
    }


def _find_spec(config: BaselineConfig, name: str) -> ModelSpec:
    matches = [spec for spec in config.model_specs("task_11") if spec.name == name]
    if len(matches) != 1:
        raise DataContractError(f"Task1.1 locked model {name!r} resolved to {len(matches)} specs")
    return matches[0]


def _identity_spec(spec: ModelSpec) -> ModelSpec:
    return replace(spec, target_transform="identity", clip_min=None)


def _target_rank(frame: pd.DataFrame, *, target_column: str) -> np.ndarray:
    result = np.empty(len(frame), dtype=float)
    result.fill(np.nan)
    groups = frame.groupby("study_group", dropna=False, observed=True).indices
    for _, positions in groups.items():
        positions = np.asarray(positions, dtype=int)
        target = pd.to_numeric(frame.iloc[positions][target_column], errors="coerce").to_numpy(
            dtype=float
        )
        require_finite(target, name=f"{target_column}.study_target")
        result[positions] = percentile_rank(target)
    require_finite(result, name=f"{target_column}.within_study_rank")
    return result


def _task11_direction(frame: pd.DataFrame, *, target_column: str) -> int:
    values: list[float] = []
    for _, group in frame.groupby("study_group", dropna=False, observed=True):
        metric = safe_spearman(group[ANCHOR_COLUMN], group[target_column])
        if metric.status == "ok" and metric.value != 0:
            values.append(float(metric.value))
    if values:
        median = float(np.median(values))
        if median != 0:
            return 1 if median > 0 else -1
        sign_sum = int(np.sign(values).sum())
        if sign_sum:
            return 1 if sign_sum > 0 else -1
    pooled = safe_spearman(frame[ANCHOR_COLUMN], frame[target_column])
    if pooled.status == "ok" and pooled.value != 0:
        return 1 if pooled.value > 0 else -1
    return -1


def _oriented_anchor(frame: pd.DataFrame, *, direction: int) -> np.ndarray:
    require_columns(frame, [ANCHOR_COLUMN], table_name="Task1.1 anchor frame")
    values = pd.to_numeric(frame[ANCHOR_COLUMN], errors="coerce").to_numpy(dtype=float)
    require_finite(values, name=ANCHOR_COLUMN)
    if direction == 1:
        return values
    if direction == -1:
        return 1.0 - values
    raise DataContractError(f"Task1.1 anchor direction must be +/-1, found {direction}")


def _within_study_serology_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        summary,
        ["participant_id", "study_accession", *SEROLOGY_SUMMARY_COLUMNS],
        table_name="prior-immunity summary",
    )
    output = summary[["participant_id", "study_accession", "hai_panel_count"]].copy()
    for source, target in zip(SEROLOGY_SUMMARY_COLUMNS, SEROLOGY_RANK_COLUMNS, strict=True):
        ranked = np.empty(len(summary), dtype=float)
        ranked.fill(np.nan)
        for _, positions in summary.groupby("study_accession", dropna=False, observed=True).indices.items():
            positions = np.asarray(positions, dtype=int)
            values = pd.to_numeric(summary.iloc[positions][source], errors="coerce").to_numpy(
                dtype=float
            )
            require_finite(values, name=f"prior-immunity.{source}")
            ranked[positions] = percentile_rank(values)
        require_finite(ranked, name=target)
        output[target] = ranked
    if output["participant_id"].duplicated().any():
        raise DataContractError("prior-immunity summaries are not unique by participant")
    return output


def build_prior_immunity_features(serology: pd.DataFrame) -> pd.DataFrame:
    """Build participant-level, study-ranked baseline HAI summaries."""

    baseline = build_hai_baseline_long(serology)
    summary = build_hai_panel_summaries(
        baseline,
        group_columns=("participant_id", "study_accession"),
    )
    return _within_study_serology_ranks(summary)


def _attach_prior_immunity(base: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    require_columns(base, ["participant_id", "study_group"], table_name="Task1.1 base frame")
    merged = base.merge(
        prior,
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    has = merged["study_accession"].notna()
    mismatch = has & merged["study_accession"].astype(str).ne(merged["study_group"].astype(str))
    if mismatch.any():
        raise DataContractError("prior-immunity study accession disagrees with Task1.1 study_group")
    merged["__has_prior_immunity"] = has
    merged = merged.drop(columns=["study_accession"])
    return merged


def _spearman(target: Sequence[float], score: Sequence[float]) -> dict[str, Any]:
    return safe_spearman(target, score).to_dict()


def _agreement(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    require_finite(left_array, name="challenge agreement left")
    require_finite(right_array, name="challenge agreement right")
    if left_array.shape != right_array.shape:
        raise DataContractError("challenge agreement arrays differ in shape")
    left_rank = percentile_rank(left_array)
    right_rank = percentile_rank(right_array)
    return {
        "n": int(len(left_array)),
        "rank_spearman": safe_spearman(left_rank, right_rank).to_dict(),
        "mean_absolute_percentile_difference": float(np.mean(np.abs(left_rank - right_rank))),
        "max_absolute_percentile_difference": float(np.max(np.abs(left_rank - right_rank))),
    }


def _fit_b21(
    train: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    target_column: str,
    spec: ModelSpec,
    excluded_columns: Sequence[str],
) -> np.ndarray:
    _, values = fit_final_model(
        train,
        prediction,
        target_column=target_column,
        spec=spec,
        excluded_columns=excluded_columns,
    )
    require_finite(values, name="Task1.1 frozen B2.1 prediction")
    return values


def _fit_anchor_residual(
    train: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    target_column: str,
    spec: ModelSpec,
    excluded_columns: Sequence[str],
    direction: int,
) -> np.ndarray:
    working = train.copy()
    train_anchor = _oriented_anchor(working, direction=direction)
    working[ANCHOR_RESIDUAL_TARGET_COLUMN] = _target_rank(
        working, target_column=target_column
    ) - train_anchor
    _, correction = fit_final_model(
        working,
        prediction,
        target_column=ANCHOR_RESIDUAL_TARGET_COLUMN,
        spec=_identity_spec(spec),
        excluded_columns=tuple(
            dict.fromkeys([*excluded_columns, target_column, ANCHOR_RESIDUAL_TARGET_COLUMN])
        ),
    )
    require_finite(correction, name="Task1.1 fixed anchor-residual correction")
    return _oriented_anchor(prediction, direction=direction) + ANCHOR_RESIDUAL_LAMBDA * correction


def _fit_serology_expert(
    train: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    target_rank: np.ndarray,
) -> tuple[np.ndarray, int]:
    train_has = train["__has_prior_immunity"].to_numpy(dtype=bool)
    prediction_has = prediction["__has_prior_immunity"].to_numpy(dtype=bool)
    train_serology = train.loc[train_has, list(SEROLOGY_RANK_COLUMNS)].copy()
    train_serology[SEROLOGY_TARGET_COLUMN] = np.asarray(target_rank, dtype=float)[train_has]
    prediction_values = np.full(len(prediction), np.nan, dtype=float)
    if int(train_has.sum()) < 8:
        raise DataContractError("Task1.1 prior-immunity expert has fewer than eight source rows")
    if prediction_has.any():
        prediction_serology = prediction.loc[prediction_has, list(SEROLOGY_RANK_COLUMNS)].copy()
        _, values = fit_final_model(
            train_serology,
            prediction_serology,
            target_column=SEROLOGY_TARGET_COLUMN,
            spec=SERology_MODEL,
            excluded_columns=(SEROLOGY_TARGET_COLUMN,),
        )
        values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        require_finite(values, name="Task1.1 prior-immunity expert prediction")
        prediction_values[prediction_has] = values
    return prediction_values, int(train_has.sum())


def _blend(base_rank: np.ndarray, expert: np.ndarray, weight: float) -> np.ndarray:
    base = np.asarray(base_rank, dtype=float).copy()
    expert_array = np.asarray(expert, dtype=float)
    require_finite(base, name="Task1.1 base rank")
    mask = np.isfinite(expert_array)
    base[mask] = (1.0 - weight) * base[mask] + weight * expert_array[mask]
    require_finite(base, name="Task1.1 late-fusion score")
    return base


def _promotion(
    fold_rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    base: str,
) -> dict[str, Any]:
    usable = []
    for fold in fold_rows:
        candidate = (fold["conditions"].get(condition) or {}).get("value")
        base_value = (fold["conditions"].get(base) or {}).get("value")
        incumbent = (fold["conditions"].get("anchor_residual") or {}).get("value")
        if all(value is not None and np.isfinite(float(value)) for value in (candidate, base_value, incumbent)):
            usable.append((float(candidate), float(base_value), float(incumbent)))
    if not usable:
        return {"passed": False, "reason": "no_usable_held_studies", "wins": 0, "required_wins": 0}
    candidate = np.asarray([row[0] for row in usable], dtype=float)
    base_values = np.asarray([row[1] for row in usable], dtype=float)
    incumbent = np.asarray([row[2] for row in usable], dtype=float)
    wins = int(np.sum(candidate > np.maximum(base_values, incumbent) + _EPS))
    required_wins = len(usable) // 2 + 1
    passed = bool(
        candidate.mean() > max(base_values.mean(), incumbent.mean()) + _EPS
        and candidate.min() >= base_values.min() - WORST_STUDY_TOLERANCE
        and candidate.min() >= incumbent.min() - WORST_STUDY_TOLERANCE
        and wins >= required_wins
    )
    return {
        "passed": passed,
        "usable_held_studies": len(usable),
        "wins": wins,
        "required_wins": required_wins,
        "candidate_mean": float(candidate.mean()),
        "base_mean": float(base_values.mean()),
        "incumbent_anchor_residual_mean": float(incumbent.mean()),
        "candidate_min": float(candidate.min()),
        "base_min": float(base_values.min()),
        "incumbent_anchor_residual_min": float(incumbent.min()),
        "worst_study_tolerance": WORST_STUDY_TOLERANCE,
    }


def run_task11_prior_immunity_experiment(
    config: BaselineConfig,
    inputs: InputBundle,
) -> Mapping[str, Any]:
    """Run the pre-registered Task1.1 prior-immunity late-fusion experiment."""

    if config.baseline != "b021_taskwise_robust":
        raise DataContractError("Task1.1 prior-immunity experiment requires B2.1 config")
    if str(config.section("selection").get("policy", "")) != "robust_v1":
        raise DataContractError("Task1.1 prior-immunity experiment requires robust_v1 selection")

    datasets = build_b02_datasets(config, inputs)
    dataset = datasets[TASK]
    b21_spec = _find_spec(config, B21_MODEL_NAME)
    anchor_residual_spec = _find_spec(config, ANCHOR_RESIDUAL_MODEL_NAME)
    require_columns(
        dataset.train,
        ["participant_id", "study_group", dataset.target_column, ANCHOR_COLUMN],
        table_name="Task1.1 training dataset",
    )
    require_columns(
        dataset.challenge,
        ["participant_id", "study_group", ANCHOR_COLUMN],
        table_name="Task1.1 challenge dataset",
    )

    public_prior = build_prior_immunity_features(inputs.tables["public_serology"])
    challenge_prior = build_prior_immunity_features(inputs.tables["challenge_serology"])
    train_augmented = _attach_prior_immunity(dataset.train, public_prior)
    challenge_augmented = _attach_prior_immunity(dataset.challenge, challenge_prior)

    splits = default_splits_for_task(dataset, random_state=config.random_state)
    folds: list[dict[str, Any]] = []
    condition_names = ["b1", "b21", "anchor_residual"]
    for base in ("b1", "b21", "anchor_residual"):
        for weight in FUSION_WEIGHTS:
            condition_names.append(f"{base}_plus_prior_w{weight:g}")

    for split in splits:
        train_base = dataset.train.iloc[split.train_indices].copy()
        held_base = dataset.train.iloc[split.validation_indices].copy()
        train = train_augmented.iloc[split.train_indices].copy()
        held = train_augmented.iloc[split.validation_indices].copy()
        direction = _task11_direction(train_base, target_column=dataset.target_column)
        target = pd.to_numeric(held_base[dataset.target_column], errors="coerce").to_numpy(dtype=float)
        require_finite(target, name="Task1.1 held target")

        b1_score = _oriented_anchor(held_base, direction=direction)
        b21_score = _fit_b21(
            train_base,
            held_base,
            target_column=dataset.target_column,
            spec=b21_spec,
            excluded_columns=dataset.excluded_columns,
        )
        anchor_residual_score = _fit_anchor_residual(
            train_base,
            held_base,
            target_column=dataset.target_column,
            spec=anchor_residual_spec,
            excluded_columns=dataset.excluded_columns,
            direction=direction,
        )
        base_ranks = {
            "b1": percentile_rank(b1_score),
            "b21": percentile_rank(b21_score),
            "anchor_residual": percentile_rank(anchor_residual_score),
        }
        expert, source_serology_rows = _fit_serology_expert(
            train,
            held,
            target_rank=_target_rank(train_base, target_column=dataset.target_column),
        )
        held_mask = np.isfinite(expert)
        held_serology_rows = int(held_mask.sum())
        if held_serology_rows < MIN_HELD_SEROLOGY_ROWS:
            raise DataContractError(
                f"held study {split.held_out_group} has only {held_serology_rows} prior-immunity rows"
            )

        conditions: dict[str, Any] = {
            name: _spearman(target, values) for name, values in base_ranks.items()
        }
        overlap_conditions: dict[str, Any] = {
            name: _spearman(target[held_mask], values[held_mask])
            for name, values in base_ranks.items()
        }
        for base, base_rank in base_ranks.items():
            for weight in FUSION_WEIGHTS:
                name = f"{base}_plus_prior_w{weight:g}"
                score = _blend(base_rank, expert, weight)
                conditions[name] = _spearman(target, score)
                overlap_conditions[name] = _spearman(target[held_mask], score[held_mask])
        conditions["prior_only_overlap"] = _spearman(target[held_mask], expert[held_mask])

        folds.append(
            {
                "held_study": str(split.held_out_group),
                "validation_rows": int(len(held)),
                "held_serology_rows": held_serology_rows,
                "held_serology_fraction": float(held_serology_rows / len(held)),
                "source_serology_rows": source_serology_rows,
                "conditions": conditions,
                "overlap_conditions": overlap_conditions,
            }
        )

    summaries = {
        name: _summary(
            [
                float(fold["conditions"][name]["value"])
                if fold["conditions"][name]["value"] is not None
                else float("nan")
                for fold in folds
            ]
        )
        for name in condition_names
    }

    promotions: dict[str, Any] = {}
    for base in ("b1", "b21", "anchor_residual"):
        for weight in FUSION_WEIGHTS:
            name = f"{base}_plus_prior_w{weight:g}"
            promotions[name] = _promotion(folds, condition=name, base=base)
    passing = [name for name, item in promotions.items() if item.get("passed") is True]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda name: (
                float(summaries[name]["mean"]),
                float(summaries[name]["median"]),
                float(summaries[name]["min"]),
                -float(name.rsplit("w", 1)[1]),
                name,
            ),
        )

    direction = _task11_direction(dataset.train, target_column=dataset.target_column)
    challenge_b1 = _oriented_anchor(dataset.challenge, direction=direction)
    challenge_b21 = _fit_b21(
        dataset.train,
        dataset.challenge,
        target_column=dataset.target_column,
        spec=b21_spec,
        excluded_columns=dataset.excluded_columns,
    )
    challenge_anchor_residual = _fit_anchor_residual(
        dataset.train,
        dataset.challenge,
        target_column=dataset.target_column,
        spec=anchor_residual_spec,
        excluded_columns=dataset.excluded_columns,
        direction=direction,
    )
    challenge_base_ranks = {
        "b1": percentile_rank(challenge_b1),
        "b21": percentile_rank(challenge_b21),
        "anchor_residual": percentile_rank(challenge_anchor_residual),
    }
    challenge_expert, final_source_serology_rows = _fit_serology_expert(
        train_augmented,
        challenge_augmented,
        target_rank=_target_rank(dataset.train, target_column=dataset.target_column),
    )
    challenge_mask = np.isfinite(challenge_expert)
    challenge_diagnostics: dict[str, Any] = {
        "rows": int(len(dataset.challenge)),
        "serology_rows": int(challenge_mask.sum()),
        "serology_fraction": float(challenge_mask.mean()),
        "source_serology_rows": final_source_serology_rows,
        "conditions": {},
    }
    for base, base_rank in challenge_base_ranks.items():
        for weight in FUSION_WEIGHTS:
            name = f"{base}_plus_prior_w{weight:g}"
            fused = _blend(base_rank, challenge_expert, weight)
            challenge_diagnostics["conditions"][name] = {
                "agreement_vs_base": _agreement(fused, base_rank),
                "agreement_vs_anchor_residual": _agreement(
                    fused, challenge_base_ranks["anchor_residual"]
                ),
            }

    coverage_by_study = {
        fold["held_study"]: {
            "target_rows": fold["validation_rows"],
            "serology_rows": fold["held_serology_rows"],
            "serology_fraction": fold["held_serology_fraction"],
        }
        for fold in folds
    }
    return {
        "experiment": "phase_b_task11_prior_immunity_late_fusion",
        "task": TASK,
        "serology_summary_columns": list(SEROLOGY_SUMMARY_COLUMNS),
        "serology_feature_representation": "within-study X-only percentile ranks",
        "serology_model": SERology_MODEL.to_dict(),
        "base_contract": {
            "b21_model": B21_MODEL_NAME,
            "anchor_residual_model": ANCHOR_RESIDUAL_MODEL_NAME,
            "anchor_residual_lambda": ANCHOR_RESIDUAL_LAMBDA,
        },
        "fusion_weights": list(FUSION_WEIGHTS),
        "missing_serology_policy": "fallback_to_base_without_complete_case_filtering",
        "held_target_outcomes_used_for_serology_features": False,
        "leaderboard_used_for_selection": False,
        "competition_submission_attempted": False,
        "folds": folds,
        "summary": summaries,
        "promotion": promotions,
        "selected_promoted_condition": selected,
        "historical_coverage_by_study": coverage_by_study,
        "challenge": challenge_diagnostics,
        "output_policy": "aggregate_only_no_participant_ids_or_row_level_predictions",
    }
