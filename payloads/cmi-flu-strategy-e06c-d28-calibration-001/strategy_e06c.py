"""Strategy-v2 E06c: Task2.1/Task2.2 D28 rank-preserving calibration.

E06a established that one global calibration map was not transferable for D365.
E06c asks the corresponding D28 question on the two scored HAI endpoints while
keeping donor order fixed.  It reuses the frozen E06a calibration primitives and
the E06a-v2 outcome-independent panel-overlap eligibility repair.

Only two already-frozen raw predictors are compared:

* B2.1 ridge_exact_a100;
* Phase-A target-domain sequence/ontology representation with the same Ridge.

The E05 donor/strain responder models are intentionally not mixed into this
slice because they alter donor rank and therefore answer a different question.
No Public-LB information enters this experiment.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .configuration import BaselineConfig
from .contracts import DataContractError
from .cv import purged_leave_one_study_out
from .datasets import HAIModelDataset, build_hai_model_dataset
from .evaluation import evaluate_hai_spec, aggregate_hai_task_predictions
from .hai_transfer import add_hai_ontology_features
from .hai_transfer_v2 import normalize_sequence_reference_schema
from .metrics import percentile_rank, safe_spearman
from .models import ModelSpec, fit_final_model
from . import strategy_e06 as _base
from . import strategy_e06_v2 as _compat

EXPERIMENT = "strategy_v2_e06c_d28_rank_preserving_calibration"
TARGET_DAY = 28
TASKS = ("Task2.1", "Task2.2")
CONDITIONS = ("b21_reference", "phase_a_target_domain")
MODEL_NAME = _base.MODEL_NAME
CALIBRATION_KINDS = _base.CALIBRATION_KINDS
MIN_EQUAL_STUDY_RMSE_REDUCTION = _base.MIN_EQUAL_STUDY_RMSE_REDUCTION
MIN_MEDIAN_STUDY_RMSE_REDUCTION = _base.MIN_MEDIAN_STUDY_RMSE_REDUCTION
MAX_WORST_STUDY_RMSE_RATIO = _base.MAX_WORST_STUDY_RMSE_RATIO
RANK_TOLERANCE = _base.RANK_TOLERANCE


def _find_model(config: BaselineConfig) -> ModelSpec:
    matches = [spec for spec in config.model_specs("hai") if spec.name == MODEL_NAME]
    if len(matches) != 1:
        raise DataContractError(
            f"E06c expected exactly one hai/{MODEL_NAME} model; found {len(matches)}"
        )
    return matches[0]


def _challenge_calibration(
    dataset: HAIModelDataset,
    *,
    spec: ModelSpec,
    panel_strains: Sequence[str],
    task: str,
    expected_donors: int,
) -> Mapping[str, Any]:
    if task not in TASKS:
        raise DataContractError(f"E06c unsupported task: {task}")
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
    calibration_train = _base._panel_frame_from_oof(
        evaluation,
        panel_strains=panel_strains,
    )
    if int(calibration_train["study_group"].astype(str).nunique()) < 2:
        raise DataContractError("E06c Challenge calibrator needs >=2 panel-overlap studies")

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
        task=task,
    )
    if len(raw_task) != expected_donors:
        raise DataContractError(
            f"E06c {task} expected {expected_donors} Challenge donors; found {len(raw_task)}"
        )

    raw_values = raw_task["prediction"].to_numpy(dtype=float)
    out: dict[str, Any] = {}
    for kind in CALIBRATION_KINDS:
        params = _compat._fit_calibrator(calibration_train, kind=kind)
        calibrated = _base._apply_calibrator(raw_values, params)
        raw_rank = percentile_rank(raw_values)
        calibrated_rank = percentile_rank(calibrated)
        delta = np.abs(raw_rank - calibrated_rank)
        out[kind] = _base._json_safe(
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
    return _base._json_safe(out)


def _run_condition(
    dataset: HAIModelDataset,
    *,
    name: str,
    spec: ModelSpec,
    panel_strains: Sequence[str],
    task: str,
    expected_donors: int,
) -> Mapping[str, Any]:
    nested = _compat._nested_calibration(
        dataset,
        spec=spec,
        panel_strains=panel_strains,
    )
    challenge = _challenge_calibration(
        dataset,
        spec=spec,
        panel_strains=panel_strains,
        task=task,
        expected_donors=expected_donors,
    )
    promotion = _base._promotion(nested, challenge)
    return _base._json_safe(
        {
            "condition": name,
            "model": spec.to_dict(),
            "historical_nested_study_out": nested,
            "challenge_rank_preservation": challenge,
            "promotion": promotion,
        }
    )


def run_e06c_on_datasets(
    *,
    b21_dataset: HAIModelDataset,
    target_domain_dataset: HAIModelDataset,
    spec: ModelSpec,
    task_panels: Mapping[str, Sequence[str]],
    expected_donors: int,
) -> Mapping[str, Any]:
    if set(task_panels) != set(TASKS):
        raise DataContractError("E06c requires exactly Task2.1 and Task2.2 panels")
    if b21_dataset.day != TARGET_DAY or target_domain_dataset.day != TARGET_DAY:
        raise DataContractError("E06c datasets must be D28")
    if b21_dataset.target_representation != "residual" or target_domain_dataset.target_representation != "residual":
        raise DataContractError("E06c requires residual HAI target representation")

    datasets = {
        "b21_reference": b21_dataset,
        "phase_a_target_domain": target_domain_dataset,
    }
    tasks: dict[str, Any] = {}
    for task in TASKS:
        panel = _base._canonical_panel(task_panels[task])
        conditions = {
            name: _run_condition(
                dataset,
                name=name,
                spec=spec,
                panel_strains=panel,
                task=task,
                expected_donors=expected_donors,
            )
            for name, dataset in datasets.items()
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
        tasks[task] = _base._json_safe(
            {
                "panel_size": len(panel),
                "historical_panel_proxy_coverage": _base._historical_panel_coverage(
                    b21_dataset, panel_strains=panel
                ),
                "conditions": conditions,
                "selected_promoted_condition": min(passed)[1] if passed else None,
            }
        )

    return _base._json_safe(
        {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "target_day": TARGET_DAY,
            "tasks": tasks,
            "promotion_thresholds": {
                "minimum_equal_study_mean_rmse_reduction": MIN_EQUAL_STUDY_RMSE_REDUCTION,
                "minimum_median_study_rmse_reduction_strictly_greater_than": MIN_MEDIAN_STUDY_RMSE_REDUCTION,
                "maximum_worst_study_rmse_ratio": MAX_WORST_STUDY_RMSE_RATIO,
                "rank_tolerance": RANK_TOLERANCE,
            },
            "calibration_contract": {
                "calibration_kinds": list(CALIBRATION_KINDS),
                "fit_unit": "task_panel_geometric_mean_donor",
                "calibration_weighting": "equal_study_total_weight",
                "outer_evaluation": "subject_purged_leave_one_study_out",
                "calibrator_training": "inner_subject_purged_study_out_oof_from_outer_training_only",
                "outer_fold_panel_overlap_filter": "fixed_panel_strain_availability_only_before_outcome_access",
                "minimum_calibrator_panel_overlap_studies": 2,
                "strict_monotonicity_required": True,
                "held_study_outcomes_used_for_calibrator": False,
                "challenge_outcomes_available": False,
                "public_leaderboard_used_for_selection": False,
                "e05_rank_changing_models_included": False,
            },
            "historical_target_contract": {
                "Task2.1": "fixed 2025 vaccine-panel intersection at D28; incomplete historical proxy when a requested strain is absent",
                "Task2.2": "fixed 12-strain Challenge-panel intersection at D28; incomplete historical proxy when requested strains are absent",
            },
            "incumbent_changed": False,
            "public_probe_authorized": False,
            "competition_submission_attempted": False,
            "leaderboard_used_for_selection": False,
            "automatic_compute_retries": 0,
        }
    )


def run_strategy_e06c(
    config: BaselineConfig,
    inputs: Any,
    *,
    sequence_reference: pd.DataFrame,
    vaccine_reference: pd.DataFrame,
) -> Mapping[str, Any]:
    if str(config.section("hai").get("target_representation", "residual")) != "residual":
        raise DataContractError("E06c frozen controls require hai.target_representation=residual")
    if str(config.section("selection", required=False).get("policy", "legacy")) != "robust_v1":
        raise DataContractError("E06c requires selection.policy=robust_v1")

    sequence_reference = normalize_sequence_reference_schema(sequence_reference)
    tables = inputs.tables
    base = build_hai_model_dataset(
        tables["public_serology"],
        tables["challenge_serology"],
        tables["participants"],
        tables["investigations"],
        day=TARGET_DAY,
        target_representation="residual",
        challenge_panel_strains=inputs.challenge_strains,
    )
    target_domain = add_hai_ontology_features(
        base,
        vaccine_reference=vaccine_reference,
        sequence_reference=sequence_reference,
        vaccine_2025=inputs.vaccine_strains,
        condition="ontology_sequence_target_domain",
    )
    result = dict(
        run_e06c_on_datasets(
            b21_dataset=base,
            target_domain_dataset=target_domain,
            spec=_find_model(config),
            task_panels={
                "Task2.1": inputs.vaccine_strains,
                "Task2.2": inputs.challenge_strains,
            },
            expected_donors=40,
        )
    )
    result["raw_conditions"] = list(CONDITIONS)
    result["model_name"] = MODEL_NAME
    return _base._json_safe(result)


__all__ = [
    "EXPERIMENT",
    "TARGET_DAY",
    "TASKS",
    "CONDITIONS",
    "MODEL_NAME",
    "CALIBRATION_KINDS",
    "run_e06c_on_datasets",
    "run_strategy_e06c",
]
