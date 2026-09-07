"""Strategy v2 E06a: strictly monotone Task2.3 D365 panel-GM calibration.

This first E06 slice is intentionally narrow.  It does not change donor ordering.
It asks whether two already-frozen D365 HAI predictors can be calibrated on the
official task scale more reliably:

1. frozen B2.1 ``ridge_exact_a100``;
2. frozen Phase-A target-domain sequence features with the same Ridge model.

Two predeclared two-parameter mappings are evaluated:

* positive affine: y = a + b*x, a >= 0, b > 0;
* log2 affine/power: log2(y) = a + b*log2(x), b > 0.

Calibration is evaluated with nested subject-purged leave-one-study-out.  For
each outer held study, the calibrator is trained only on inner OOF panel-level
predictions from the remaining studies.  No held-study outcomes enter either
the base fit or the calibration fit.  Public D365 data do not contain a
complete 12-strain Challenge panel, so all historical scores are explicitly
fixed-panel proxies rather than complete official-target CV.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .aliases import canonicalize_strain
from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns, require_finite
from .cv import purged_leave_one_study_out
from .datasets import HAIModelDataset, build_hai_model_dataset
from .evaluation import evaluate_hai_spec, aggregate_hai_task_predictions
from .hai_transfer import add_hai_ontology_features
from .hai_transfer_v2 import normalize_sequence_reference_schema
from .metrics import evaluate_predictions, grouped_metrics, percentile_rank, safe_spearman
from .models import ModelSpec, fit_final_model
from .targets import geometric_mean


EXPERIMENT = "strategy_v2_e06a_task23_rank_preserving_calibration"
TASK = "Task2.3"
TARGET_DAY = 365
MODEL_NAME = "ridge_exact_a100"
BASE_CONDITIONS: tuple[str, ...] = (
    "b21_reference",
    "phase_a_target_domain",
)
CALIBRATION_KINDS: tuple[str, ...] = (
    "positive_affine",
    "log2_affine",
)
MIN_SLOPE = 1e-6
MIN_EQUAL_STUDY_RMSE_REDUCTION = 0.02
MIN_MEDIAN_STUDY_RMSE_REDUCTION = 0.0
MAX_WORST_STUDY_RMSE_RATIO = 1.05
RANK_TOLERANCE = 1e-12


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _find_model(config: BaselineConfig) -> ModelSpec:
    matches = [spec for spec in config.model_specs("hai") if spec.name == MODEL_NAME]
    if len(matches) != 1:
        raise DataContractError(
            f"E06a expected exactly one hai/{MODEL_NAME} model; found {len(matches)}"
        )
    return matches[0]


def _canonical_panel(values: Sequence[str]) -> tuple[str, ...]:
    panel = tuple(dict.fromkeys(canonicalize_strain(value) for value in values))
    if not panel or any(not value for value in panel):
        raise DataContractError("E06a task panel is empty after canonicalization")
    return panel


def _panel_frame(
    frame: pd.DataFrame,
    prediction: Sequence[float],
    *,
    panel_strains: Sequence[str],
    split: Sequence[str] | str,
) -> pd.DataFrame:
    """Aggregate matched target/prediction strains to one task proxy per donor."""
    require_columns(
        frame,
        [
            "participant_id",
            "study_group",
            "subject_group",
            "virus_strain",
            "post_hai",
        ],
        table_name="E06a panel source",
    )
    values = np.asarray(prediction, dtype=float).reshape(-1)
    if len(values) != len(frame):
        raise DataContractError("E06a prediction length differs from panel source")
    require_finite(values, name="E06a panel prediction")
    if (values <= 0).any():
        raise DataContractError("E06a panel predictions must be positive")

    work = frame[
        [
            "participant_id",
            "study_group",
            "subject_group",
            "virus_strain",
            "post_hai",
        ]
    ].copy()
    work["virus_strain"] = work["virus_strain"].map(canonicalize_strain)
    work["prediction"] = values
    if isinstance(split, str):
        work["split"] = split
    else:
        labels = np.asarray(list(split), dtype=object)
        if len(labels) != len(work):
            raise DataContractError("E06a split labels differ in length from panel source")
        work["split"] = labels.astype(str)

    panel = set(_canonical_panel(panel_strains))
    work = work.loc[work["virus_strain"].isin(panel)].copy()
    if work.empty:
        raise DataContractError("E06a panel source has zero overlap with task panel")
    work["post_hai"] = pd.to_numeric(work["post_hai"], errors="coerce")
    if not np.isfinite(work["post_hai"].to_numpy(dtype=float)).all():
        raise DataContractError("E06a panel target contains non-finite values")
    if (work["post_hai"].to_numpy(dtype=float) <= 0).any():
        raise DataContractError("E06a panel target must be positive")

    # Collapse any repeated assay rows within a strain before the task-level GM,
    # so a duplicated strain cannot receive accidental extra weight.
    per_strain = (
        work.groupby(
            [
                "split",
                "study_group",
                "participant_id",
                "subject_group",
                "virus_strain",
            ],
            dropna=False,
            observed=True,
        )
        .agg(
            target=("post_hai", geometric_mean),
            prediction=("prediction", geometric_mean),
        )
        .reset_index()
    )
    panel_frame = (
        per_strain.groupby(
            ["split", "study_group", "participant_id", "subject_group"],
            dropna=False,
            observed=True,
        )
        .agg(
            target=("target", geometric_mean),
            prediction=("prediction", geometric_mean),
            panel_size=("virus_strain", "nunique"),
        )
        .reset_index()
    )
    return panel_frame


def _panel_frame_from_oof(
    evaluation: Any,
    *,
    panel_strains: Sequence[str],
) -> pd.DataFrame:
    oof = evaluation.enriched_oof
    require_columns(
        oof,
        [
            "split",
            "participant_id",
            "study_group",
            "subject_group",
            "virus_strain",
            "post_hai",
            "post_prediction",
        ],
        table_name="E06a enriched OOF",
    )
    return _panel_frame(
        oof,
        oof["post_prediction"].to_numpy(dtype=float),
        panel_strains=panel_strains,
        split=oof["split"].astype(str).tolist(),
    )


def _equal_study_weights(frame: pd.DataFrame) -> np.ndarray:
    require_columns(frame, ["study_group"], table_name="E06a calibration frame")
    studies = frame["study_group"].astype(str)
    counts = studies.value_counts()
    if counts.empty:
        raise DataContractError("E06a calibration requires at least one study")
    weights = np.asarray([1.0 / float(counts.loc[study]) for study in studies], dtype=float)
    require_finite(weights, name="E06a equal-study calibration weights")
    weights *= len(weights) / float(weights.sum())
    return weights


def _fit_calibrator(frame: pd.DataFrame, *, kind: str) -> Mapping[str, Any]:
    if kind not in CALIBRATION_KINDS:
        raise DataContractError(f"E06a unknown calibration kind: {kind}")
    require_columns(frame, ["study_group", "target", "prediction"])
    x = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype=float)
    require_finite(x, name="E06a calibration prediction")
    require_finite(y, name="E06a calibration target")
    if (x <= 0).any() or (y <= 0).any():
        raise DataContractError("E06a calibration values must be positive")
    if len(x) < 4:
        raise DataContractError("E06a calibration requires at least four panel donors")
    weights = _equal_study_weights(frame)
    root_w = np.sqrt(weights)

    if kind == "positive_affine":
        design = np.column_stack([np.ones(len(x), dtype=float), x])
        result = lsq_linear(
            design * root_w[:, None],
            y * root_w,
            bounds=([0.0, MIN_SLOPE], [np.inf, np.inf]),
            method="trf",
        )
    else:
        log_x = np.log2(x)
        log_y = np.log2(y)
        design = np.column_stack([np.ones(len(x), dtype=float), log_x])
        result = lsq_linear(
            design * root_w[:, None],
            log_y * root_w,
            bounds=([-np.inf, MIN_SLOPE], [np.inf, np.inf]),
            method="trf",
        )
    if not bool(result.success):
        raise DataContractError(f"E06a calibration fit failed: {kind}")
    intercept, slope = map(float, result.x)
    if not np.isfinite(intercept) or not np.isfinite(slope) or slope < MIN_SLOPE:
        raise DataContractError(f"E06a calibration parameters invalid: {kind}")
    return {
        "kind": kind,
        "intercept": intercept,
        "slope": slope,
        "training_panel_donors": int(len(frame)),
        "training_studies": int(frame["study_group"].astype(str).nunique()),
    }


def _apply_calibrator(values: Sequence[float], params: Mapping[str, Any]) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    require_finite(x, name="E06a calibration input")
    if (x <= 0).any():
        raise DataContractError("E06a calibration input must be positive")
    slope = float(params["slope"])
    intercept = float(params["intercept"])
    if slope < MIN_SLOPE or not np.isfinite(slope) or not np.isfinite(intercept):
        raise DataContractError("E06a calibration parameters are not strictly monotone")
    if params["kind"] == "positive_affine":
        calibrated = intercept + slope * x
    elif params["kind"] == "log2_affine":
        calibrated = np.exp2(intercept + slope * np.log2(x))
    else:
        raise DataContractError(f"E06a unknown calibration kind: {params['kind']}")
    require_finite(calibrated, name="E06a calibrated prediction")
    if (calibrated <= 0).any():
        raise DataContractError("E06a calibrated prediction must remain positive")
    return calibrated


def _summarize_fold_metrics(frame: pd.DataFrame) -> Mapping[str, Any]:
    if frame.empty:
        return {
            "count": 0,
            "spearman_mean": None,
            "spearman_median": None,
            "spearman_min": None,
            "rmse_mean": None,
            "rmse_median": None,
            "rmse_max": None,
        }
    spearman = pd.to_numeric(frame["spearman"], errors="coerce").to_numpy(dtype=float)
    rmse = pd.to_numeric(frame["rmse"], errors="coerce").to_numpy(dtype=float)
    spearman = spearman[np.isfinite(spearman)]
    rmse = rmse[np.isfinite(rmse)]
    return {
        "count": int(len(frame)),
        "spearman_mean": float(np.mean(spearman)) if spearman.size else None,
        "spearman_median": float(np.median(spearman)) if spearman.size else None,
        "spearman_min": float(np.min(spearman)) if spearman.size else None,
        "rmse_mean": float(np.mean(rmse)) if rmse.size else None,
        "rmse_median": float(np.median(rmse)) if rmse.size else None,
        "rmse_max": float(np.max(rmse)) if rmse.size else None,
    }


def _metric_bundle(frame: pd.DataFrame) -> Mapping[str, Any]:
    require_columns(frame, ["study_group", "target", "prediction"])
    folds = grouped_metrics(
        frame,
        group_columns=["study_group"],
        target_column="target",
        prediction_column="prediction",
    )
    return _json_safe(
        {
            "pooled": evaluate_predictions(frame["target"], frame["prediction"]),
            "equal_study_summary": _summarize_fold_metrics(folds),
            "study_metrics": folds.to_dict(orient="records"),
        }
    )


def _rank_preservation(raw: pd.DataFrame, calibrated: pd.DataFrame) -> Mapping[str, Any]:
    keys = ["split", "study_group", "participant_id", "subject_group"]
    left = raw[keys + ["prediction"]].rename(columns={"prediction": "raw_prediction"})
    right = calibrated[keys + ["prediction"]].rename(
        columns={"prediction": "calibrated_prediction"}
    )
    aligned = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(aligned) != len(raw) or len(aligned) != len(calibrated):
        raise DataContractError("E06a rank-preservation frames do not align")

    per_study: list[Mapping[str, Any]] = []
    maximum = 0.0
    for study, group in aligned.groupby("study_group", observed=True):
        raw_rank = percentile_rank(group["raw_prediction"].to_numpy(dtype=float))
        calibrated_rank = percentile_rank(
            group["calibrated_prediction"].to_numpy(dtype=float)
        )
        delta = np.abs(raw_rank - calibrated_rank)
        maximum = max(maximum, float(delta.max()) if len(delta) else 0.0)
        agreement = safe_spearman(
            group["raw_prediction"], group["calibrated_prediction"]
        ).to_dict()
        per_study.append(
            {
                "study": str(study),
                "n": int(len(group)),
                "rank_spearman_raw_vs_calibrated": agreement,
                "max_absolute_percentile_difference": (
                    float(delta.max()) if len(delta) else 0.0
                ),
            }
        )
    return _json_safe(
        {
            "max_absolute_percentile_difference": maximum,
            "preserved_within_tolerance": bool(maximum <= RANK_TOLERANCE),
            "by_study": per_study,
        }
    )


def _rmse_diagnostics(
    raw_metrics: Mapping[str, Any],
    calibrated_metrics: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw_rows = {
        str(row["study_group"]): row for row in raw_metrics["study_metrics"]
    }
    calibrated_rows = {
        str(row["study_group"]): row for row in calibrated_metrics["study_metrics"]
    }
    if set(raw_rows) != set(calibrated_rows):
        raise DataContractError("E06a RMSE study sets do not align")
    reductions: list[float] = []
    ratios: list[float] = []
    by_study: list[Mapping[str, Any]] = []
    for study in sorted(raw_rows):
        raw_rmse = float(raw_rows[study]["rmse"])
        calibrated_rmse = float(calibrated_rows[study]["rmse"])
        if not np.isfinite(raw_rmse) or raw_rmse <= 0 or not np.isfinite(calibrated_rmse):
            raise DataContractError("E06a study RMSE is invalid")
        reduction = 1.0 - calibrated_rmse / raw_rmse
        ratio = calibrated_rmse / raw_rmse
        reductions.append(reduction)
        ratios.append(ratio)
        by_study.append(
            {
                "study": study,
                "raw_rmse": raw_rmse,
                "calibrated_rmse": calibrated_rmse,
                "relative_reduction": reduction,
            }
        )
    raw_mean = float(raw_metrics["equal_study_summary"]["rmse_mean"])
    calibrated_mean = float(calibrated_metrics["equal_study_summary"]["rmse_mean"])
    return _json_safe(
        {
            "equal_study_mean_relative_reduction": 1.0 - calibrated_mean / raw_mean,
            "median_study_relative_reduction": float(np.median(reductions)),
            "worst_study_rmse_ratio": float(np.max(ratios)),
            "by_study": by_study,
        }
    )


def _nested_calibration(
    dataset: HAIModelDataset,
    *,
    spec: ModelSpec,
    panel_strains: Sequence[str],
) -> Mapping[str, Any]:
    """Nested outer-study evaluation of both predeclared calibration mappings."""
    train = dataset.train.reset_index(drop=True)
    outer_splits = purged_leave_one_study_out(
        train["study_group"].astype(str),
        train["subject_group"].astype(str),
    )
    if len(outer_splits) < 3:
        raise DataContractError("E06a requires at least three usable outer study folds")

    raw_parts: list[pd.DataFrame] = []
    calibrated_parts: dict[str, list[pd.DataFrame]] = {
        kind: [] for kind in CALIBRATION_KINDS
    }
    fold_params: dict[str, list[Mapping[str, Any]]] = {
        kind: [] for kind in CALIBRATION_KINDS
    }

    for outer in outer_splits:
        outer_train = train.iloc[outer.train_indices].reset_index(drop=True)
        outer_validation = train.iloc[outer.validation_indices].reset_index(drop=True)
        if outer_validation.empty:
            raise DataContractError("E06a outer validation fold is empty")

        inner_splits = purged_leave_one_study_out(
            outer_train["study_group"].astype(str),
            outer_train["subject_group"].astype(str),
        )
        if len(inner_splits) < 2:
            raise DataContractError(
                f"E06a outer fold {outer.name} has fewer than two inner study folds"
            )
        inner_dataset = replace(dataset, train=outer_train)
        inner_evaluation = evaluate_hai_spec(
            inner_dataset,
            spec=spec,
            splits=inner_splits,
            panel_strains=panel_strains,
        )
        calibration_train = _panel_frame_from_oof(
            inner_evaluation,
            panel_strains=panel_strains,
        )

        _, target_prediction = fit_final_model(
            outer_train,
            outer_validation,
            target_column=dataset.target_column,
            spec=spec,
            excluded_columns=dataset.excluded_columns,
        )
        post_prediction = dataset.target_prediction_to_post_hai(
            target_prediction,
            frame=outer_validation,
        )
        raw_panel = _panel_frame(
            outer_validation,
            post_prediction,
            panel_strains=panel_strains,
            split=str(outer.name),
        )
        raw_parts.append(raw_panel)

        for kind in CALIBRATION_KINDS:
            params = _fit_calibrator(calibration_train, kind=kind)
            calibrated = raw_panel.copy()
            calibrated["prediction"] = _apply_calibrator(
                calibrated["prediction"].to_numpy(dtype=float),
                params,
            )
            calibrated_parts[kind].append(calibrated)
            fold_params[kind].append(
                {
                    "outer_fold": str(outer.name),
                    "held_out_group": str(outer.held_out_group),
                    "inner_split_count": int(len(inner_splits)),
                    **dict(params),
                }
            )

    raw = pd.concat(raw_parts, ignore_index=True)
    raw_metrics = _metric_bundle(raw)
    candidates: dict[str, Any] = {}
    for kind in CALIBRATION_KINDS:
        calibrated = pd.concat(calibrated_parts[kind], ignore_index=True)
        calibrated_metrics = _metric_bundle(calibrated)
        rank = _rank_preservation(raw, calibrated)
        rmse = _rmse_diagnostics(raw_metrics, calibrated_metrics)
        candidates[kind] = _json_safe(
            {
                "metrics": calibrated_metrics,
                "rank_preservation": rank,
                "rmse_diagnostics": rmse,
                "outer_fold_calibrators": fold_params[kind],
            }
        )
    return _json_safe(
        {
            "outer_split_count": int(len(outer_splits)),
            "raw_metrics": raw_metrics,
            "calibrated": candidates,
        }
    )


def _challenge_calibration(
    dataset: HAIModelDataset,
    *,
    spec: ModelSpec,
    panel_strains: Sequence[str],
    expected_donors: int = 40,
) -> Mapping[str, Any]:
    full_splits = purged_leave_one_study_out(
        dataset.train["study_group"].astype(str),
        dataset.train["subject_group"].astype(str),
    )
    evaluation = evaluate_hai_spec(
        dataset,
        spec=spec,
        splits=full_splits,
        panel_strains=panel_strains,
    )
    calibration_train = _panel_frame_from_oof(
        evaluation,
        panel_strains=panel_strains,
    )
    _, target_prediction = fit_final_model(
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
    strain_prediction = dataset.challenge[["participant_id", "virus_strain"]].copy()
    strain_prediction["prediction"] = post_prediction
    raw_task = aggregate_hai_task_predictions(
        strain_prediction,
        panel_strains=panel_strains,
        task=TASK,
    )
    if len(raw_task) != expected_donors:
        raise DataContractError(
            f"E06a challenge expected {expected_donors} donors; found {len(raw_task)}"
        )

    result: dict[str, Any] = {}
    raw_values = raw_task["prediction"].to_numpy(dtype=float)
    for kind in CALIBRATION_KINDS:
        params = _fit_calibrator(calibration_train, kind=kind)
        calibrated = _apply_calibrator(raw_values, params)
        raw_rank = percentile_rank(raw_values)
        calibrated_rank = percentile_rank(calibrated)
        delta = np.abs(raw_rank - calibrated_rank)
        result[kind] = _json_safe(
            {
                "calibrator": params,
                "rank_spearman_raw_vs_calibrated": safe_spearman(
                    raw_values, calibrated
                ).to_dict(),
                "mean_absolute_percentile_difference": float(delta.mean()),
                "max_absolute_percentile_difference": float(delta.max()),
                "rank_preserved_within_tolerance": bool(
                    float(delta.max()) <= RANK_TOLERANCE
                ),
            }
        )
    return _json_safe(result)


def _promotion(
    nested: Mapping[str, Any],
    challenge: Mapping[str, Any],
) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for kind in CALIBRATION_KINDS:
        candidate = nested["calibrated"][kind]
        rmse = candidate["rmse_diagnostics"]
        rank = candidate["rank_preservation"]
        challenge_rank = challenge[kind]
        slopes = [
            float(item["slope"]) for item in candidate["outer_fold_calibrators"]
        ]
        checks = {
            "strictly_positive_outer_slopes": bool(
                slopes and min(slopes) >= MIN_SLOPE
            ),
            "held_study_rank_preserved": bool(
                rank["preserved_within_tolerance"]
            ),
            "challenge_rank_preserved": bool(
                challenge_rank["rank_preserved_within_tolerance"]
            ),
            "equal_study_mean_rmse_reduction_at_least_2pct": bool(
                float(rmse["equal_study_mean_relative_reduction"])
                >= MIN_EQUAL_STUDY_RMSE_REDUCTION
            ),
            "median_study_rmse_improves": bool(
                float(rmse["median_study_relative_reduction"])
                > MIN_MEDIAN_STUDY_RMSE_REDUCTION
            ),
            "no_study_rmse_worse_than_5pct": bool(
                float(rmse["worst_study_rmse_ratio"])
                <= MAX_WORST_STUDY_RMSE_RATIO
            ),
        }
        result[kind] = {
            "passed": bool(all(checks.values())),
            "checks": checks,
        }
    return result


def _historical_panel_coverage(
    dataset: HAIModelDataset,
    *,
    panel_strains: Sequence[str],
) -> Mapping[str, Any]:
    panel = set(_canonical_panel(panel_strains))
    work = dataset.train[["study_group", "participant_id", "virus_strain"]].copy()
    work["virus_strain"] = work["virus_strain"].map(canonicalize_strain)
    work = work.loc[work["virus_strain"].isin(panel)].drop_duplicates()
    donor_sizes = (
        work.groupby(["study_group", "participant_id"], observed=True)["virus_strain"]
        .nunique()
    )
    return {
        "requested_strains": int(len(panel)),
        "historically_observed_requested_strains": int(
            work["virus_strain"].nunique()
        ),
        "panel_proxy_donors": int(len(donor_sizes)),
        "panel_size_min": int(donor_sizes.min()) if len(donor_sizes) else 0,
        "panel_size_max": int(donor_sizes.max()) if len(donor_sizes) else 0,
        "complete_requested_panel_donors": int((donor_sizes == len(panel)).sum()),
    }


def _run_condition(
    name: str,
    dataset: HAIModelDataset,
    *,
    spec: ModelSpec,
    panel_strains: Sequence[str],
) -> Mapping[str, Any]:
    nested = _nested_calibration(
        dataset,
        spec=spec,
        panel_strains=panel_strains,
    )
    challenge = _challenge_calibration(
        dataset,
        spec=spec,
        panel_strains=panel_strains,
    )
    promotion = _promotion(nested, challenge)
    return _json_safe(
        {
            "condition": name,
            "model": spec.to_dict(),
            "historical_nested_study_out": nested,
            "challenge_rank_preservation": challenge,
            "promotion": promotion,
        }
    )


def run_strategy_e06a(
    config: BaselineConfig,
    inputs: Any,
    *,
    sequence_reference: pd.DataFrame,
    vaccine_reference: pd.DataFrame,
) -> Mapping[str, Any]:
    """Run the predeclared E06a D365 calibration comparison."""
    if str(config.section("hai").get("target_representation", "residual")) != "residual":
        raise DataContractError(
            "E06a frozen controls require hai.target_representation=residual"
        )
    if str(config.section("selection", required=False).get("policy", "legacy")) != "robust_v1":
        raise DataContractError("E06a requires selection.policy=robust_v1")

    panel = _canonical_panel(inputs.challenge_strains)
    sequence_reference = normalize_sequence_reference_schema(sequence_reference)
    tables = inputs.tables
    base = build_hai_model_dataset(
        tables["public_serology"],
        tables["challenge_serology"],
        tables["participants"],
        tables["investigations"],
        day=TARGET_DAY,
        target_representation="residual",
        challenge_panel_strains=panel,
    )
    target_domain = add_hai_ontology_features(
        base,
        vaccine_reference=vaccine_reference,
        sequence_reference=sequence_reference,
        vaccine_2025=inputs.vaccine_strains,
        condition="ontology_sequence_target_domain",
    )
    spec = _find_model(config)
    conditions = {
        "b21_reference": _run_condition(
            "b21_reference",
            base,
            spec=spec,
            panel_strains=panel,
        ),
        "phase_a_target_domain": _run_condition(
            "phase_a_target_domain",
            target_domain,
            spec=spec,
            panel_strains=panel,
        ),
    }

    passed: list[tuple[float, str]] = []
    for condition, payload in conditions.items():
        for kind, decision in payload["promotion"].items():
            if decision["passed"]:
                rmse_mean = float(
                    payload["historical_nested_study_out"]["calibrated"][kind][
                        "metrics"
                    ]["equal_study_summary"]["rmse_mean"]
                )
                passed.append((rmse_mean, f"{condition}__{kind}"))
    selected = min(passed)[1] if passed else None

    coverage = _historical_panel_coverage(base, panel_strains=panel)
    return _json_safe(
        {
            "experiment": EXPERIMENT,
            "task": TASK,
            "target_day": TARGET_DAY,
            "conditions": conditions,
            "selected_promoted_condition": selected,
            "promotion_thresholds": {
                "minimum_equal_study_mean_rmse_reduction": MIN_EQUAL_STUDY_RMSE_REDUCTION,
                "minimum_median_study_rmse_reduction_strictly_greater_than": MIN_MEDIAN_STUDY_RMSE_REDUCTION,
                "maximum_worst_study_rmse_ratio": MAX_WORST_STUDY_RMSE_RATIO,
                "rank_tolerance": RANK_TOLERANCE,
            },
            "historical_panel_proxy_coverage": coverage,
            "historical_target_contract": (
                "fixed Challenge-panel intersection at D365; incomplete public "
                "panel proxy, not complete 12-strain official-target CV"
            ),
            "calibration_contract": {
                "calibration_kinds": list(CALIBRATION_KINDS),
                "fit_unit": "panel_geometric_mean_donor",
                "calibration_weighting": "equal_study_total_weight",
                "outer_evaluation": "subject_purged_leave_one_study_out",
                "calibrator_training": (
                    "inner_subject_purged_study_out_oof_from_outer_training_only"
                ),
                "strict_monotonicity_required": True,
                "isotonic_used": False,
                "held_study_outcomes_used_for_calibrator": False,
                "observed_d28_used_as_d365_feature": False,
            },
            "competition_submission_attempted": False,
            "leaderboard_used_for_selection": False,
            "output_policy": (
                "aggregate_only_public_study_names_no_participant_ids_or_row_predictions"
            ),
        }
    )


__all__ = [
    "EXPERIMENT",
    "TASK",
    "TARGET_DAY",
    "MODEL_NAME",
    "BASE_CONDITIONS",
    "CALIBRATION_KINDS",
    "MIN_SLOPE",
    "MIN_EQUAL_STUDY_RMSE_REDUCTION",
    "MIN_MEDIAN_STUDY_RMSE_REDUCTION",
    "MAX_WORST_STUDY_RMSE_RATIO",
    "RANK_TOLERANCE",
    "_fit_calibrator",
    "_apply_calibrator",
    "_nested_calibration",
    "_challenge_calibration",
    "run_strategy_e06a",
]
