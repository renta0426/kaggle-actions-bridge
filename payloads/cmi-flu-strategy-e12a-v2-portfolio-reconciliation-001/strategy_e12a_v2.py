"""Strategy-v2 E12a-v2: reconcile and reproduce the true final portfolio.

E12a-v1 correctly reproduced the accepted E01 harness but incorrectly treated
the E01 Task1.3 B2.1 control as the current competition incumbent. Later frozen
records (E04/E04b/E08 and subsequent checkpoints) retain the strict ASC anchor
for Task1.3. This module fixes only that manifest/portfolio identity.

No candidate is selected from E12a-v2 outcomes. The final portfolio is fixed
before execution, and all exported content is aggregate-only.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .configuration import BaselineConfig
from .contracts import DataContractError
from .runner import InputBundle
from .strategy_e01 import run_strategy_e01

EXPERIMENT = "strategy_v2_e12a_v2_portfolio_reconciliation"
COMPARISON_CONTRACT = "paired_subject_purged_v2"
REPRODUCTION_ABSOLUTE_TOLERANCE = 2e-6

TASK13_FINAL_INCUMBENT = "strict_asc_anchor"
TASK13_ANCHOR_COLUMN = "flow_rank__Antibody-secreting_cells_(ASC)"
TASK13_EXPECTED_TRAIN_ROWS = 23
TASK13_EXPECTED_CHALLENGE_ROWS = 40
TASK13_EXPECTED_STUDY_EQUAL_SPEARMAN = 0.38158872734833993
TASK13_E01_CONTROL_INCUMBENT = "b21_pls_1"
TASK13_E01_CONTROL_SPEARMAN = 0.10630410834652516

EXPECTED_FINAL_PORTFOLIO = {
    "Task1.1": "b21_pls_2",
    "Task1.2": "task12_anchor_residual_et_d5_l5_sqrt_lambda0.5",
    "Task1.3": TASK13_FINAL_INCUMBENT,
    "Task1.4": "raw_pre_vacc_conserved_anchor",
    "Task2.1": "b21_et_subtype_d3_l5",
    "Task2.2": "b21_et_subtype_d5_l10",
    "Task2.3": "b21_ridge_exact_a100",
}

EXPECTED_E01_FINAL_TASKS = {
    "Task1.1": ("b21_pls_2", 0.09970718035376519),
    "Task1.2": ("task12_anchor_residual_et_d5_l5_sqrt_lambda0.5", 0.5271033295423541),
    "Task2.1": ("b21_et_subtype_d3_l5", 0.623449034547404),
    "Task2.2": ("b21_et_subtype_d5_l10", 0.576505972340535),
    "Task2.3": ("b21_ridge_exact_a100", 0.6957305642219944),
}

PORTFOLIO_PROVENANCE = {
    "Task1.1": "B2.1 retained through E11c closeout",
    "Task1.2": "locally selected anchor-residual expert supported by controlled Public probe",
    "Task1.3": "strict ASC anchor retained by E04/E04b and subsequent checkpoints",
    "Task1.4": "E07 raw Pre-vacc Conserved AIM anchor",
    "Task2.1": "B2.1 HAI reference retained after E05/E06c/E09b",
    "Task2.2": "B2.1 HAI reference retained after E05/E06c/E09b",
    "Task2.3": "B2.1 HAI reference retained after E06a/E06b",
}

_BANNED_AGGREGATE_KEYS = {
    "participant_id",
    "subject_group",
    "row_index",
    "oof_predictions",
    "challenge_predictions",
}


def _finite_float(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"E12a-v2 nonnumeric value:{label}") from exc
    if not np.isfinite(number):
        raise DataContractError(f"E12a-v2 nonfinite value:{label}")
    return number


def _assert_aggregate_only(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _BANNED_AGGREGATE_KEYS:
                raise DataContractError(f"E12a-v2 aggregate serialization leaked row-level key:{key}")
            _assert_aggregate_only(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_aggregate_only(item)


def _spearman(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if target.ndim != 1 or prediction.shape != target.shape or len(target) < 3:
        raise DataContractError("E12a-v2 Task1.3 invalid paired arrays")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise DataContractError("E12a-v2 Task1.3 nonfinite paired arrays")
    if len(np.unique(target)) < 2 or len(np.unique(prediction)) < 2:
        raise DataContractError("E12a-v2 Task1.3 constant target or prediction")
    value = float(spearmanr(target, prediction).statistic)
    if not np.isfinite(value):
        raise DataContractError("E12a-v2 Task1.3 undefined Spearman")
    return value


def summarize_strict_task13_dataset(dataset: Any) -> dict[str, Any]:
    """Summarize the fixed strict ASC anchor without exporting identifiers."""
    train = dataset.train
    challenge = dataset.challenge
    if len(train) != TASK13_EXPECTED_TRAIN_ROWS:
        raise DataContractError("E12a-v2 Task1.3 strict train row count changed")
    if len(challenge) != TASK13_EXPECTED_CHALLENGE_ROWS:
        raise DataContractError("E12a-v2 Task1.3 challenge row count changed")
    for frame, label in ((train, "train"), (challenge, "challenge")):
        if TASK13_ANCHOR_COLUMN not in frame.columns:
            raise DataContractError(f"E12a-v2 Task1.3 anchor column missing:{label}")
    if dataset.target_column not in train.columns:
        raise DataContractError("E12a-v2 Task1.3 target column missing")

    target = pd.to_numeric(train[dataset.target_column], errors="coerce").to_numpy(float)
    anchor = pd.to_numeric(train[TASK13_ANCHOR_COLUMN], errors="coerce").to_numpy(float)
    challenge_anchor = pd.to_numeric(
        challenge[TASK13_ANCHOR_COLUMN], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(challenge_anchor).all():
        raise DataContractError("E12a-v2 Task1.3 challenge anchor is not complete")

    if "study_group" not in train.columns:
        raise DataContractError("E12a-v2 Task1.3 study key missing")
    study = train["study_group"].astype(str).to_numpy()
    unique_studies = sorted(set(study))
    if len(unique_studies) != 1:
        raise DataContractError("E12a-v2 Task1.3 strict cohort must remain one study")

    study_scores = []
    for value in unique_studies:
        idx = np.flatnonzero(study == value)
        study_scores.append(_spearman(target[idx], anchor[idx]))
    score = float(np.mean(study_scores))
    deviation = score - TASK13_EXPECTED_STUDY_EQUAL_SPEARMAN

    return {
        "task": "Task1.3",
        "incumbent": TASK13_FINAL_INCUMBENT,
        "predictor_feature": TASK13_ANCHOR_COLUMN,
        "source_experiments": [
            "strategy_v2_e04_task13_anchor_preserving_rescue",
            "strategy_v2_e04b_task13_material_bridge",
        ],
        "train_rows": int(len(train)),
        "historical_study_count": int(len(unique_studies)),
        "study_equal_spearman": score,
        "expected_study_equal_spearman": TASK13_EXPECTED_STUDY_EQUAL_SPEARMAN,
        "deviation": deviation,
        "absolute_deviation": abs(deviation),
        "reproduced": bool(abs(deviation) <= REPRODUCTION_ABSOLUTE_TOLERANCE),
        "challenge_rows": int(len(challenge)),
        "challenge_anchor_complete": True,
        "challenge_anchor_unique_values": int(len(np.unique(challenge_anchor))),
        "challenge_anchor_min": float(np.min(challenge_anchor)),
        "challenge_anchor_max": float(np.max(challenge_anchor)),
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }


def run_strict_task13_anchor_audit(config: BaselineConfig, inputs: InputBundle) -> dict[str, Any]:
    """Build the accepted strict Task1.3 dataset and reproduce the anchor."""
    del config  # The strict anchor itself has no fitted hyperparameters.
    from .datasets import build_task_13_dataset

    tables = inputs.tables
    required = ("public_flow", "challenge_flow", "participants", "investigations")
    if any(name not in tables for name in required):
        raise DataContractError("E12a-v2 Task1.3 required table missing")
    dataset = build_task_13_dataset(
        tables["public_flow"],
        tables["challenge_flow"],
        tables["participants"],
        tables["investigations"],
        mode="broad",
        include_sdy272_asc_proxy=False,
    )
    return summarize_strict_task13_dataset(dataset)


def _validate_e01_boundary(e01: Mapping[str, Any]) -> None:
    if e01.get("experiment") != "strategy_v2_e01_paired_evaluation":
        raise DataContractError("E12a-v2 requires the accepted E01 aggregate result")
    if e01.get("comparison_contract") != COMPARISON_CONTRACT:
        raise DataContractError("E12a-v2 E01 comparison contract mismatch")
    for key in (
        "contains_participant_identifiers",
        "contains_row_level_predictions",
        "leaderboard_used_for_selection",
        "competition_submission_attempted",
    ):
        if e01.get(key) is not False:
            raise DataContractError(f"E12a-v2 E01 boundary changed:{key}")


def evaluate_e12a_v2(
    e01: Mapping[str, Any],
    task13_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile frozen provenance and evaluate reproduction only."""
    _validate_e01_boundary(e01)
    tasks = e01.get("tasks") or {}
    frozen = e01.get("frozen_incumbent") or {}
    if "Task1.3" not in tasks or "Task1.3" not in frozen:
        raise DataContractError("E12a-v2 E01 Task1.3 control missing")

    task13_e01_name = str((tasks["Task1.3"].get("incumbent") or {}).get("name", ""))
    if task13_e01_name != TASK13_E01_CONTROL_INCUMBENT:
        raise DataContractError("E12a-v2 E01 Task1.3 control identity changed")
    task13_e01_score = _finite_float(
        ((tasks["Task1.3"].get("incumbent") or {}).get("metrics") or {}).get(
            "study_equal_weight_spearman_mean_strict"
        ),
        label="Task1.3.e01_control",
    )
    if abs(task13_e01_score - TASK13_E01_CONTROL_SPEARMAN) > REPRODUCTION_ABSOLUTE_TOLERANCE:
        raise DataContractError("E12a-v2 E01 Task1.3 control reproduction changed")

    rows: list[dict[str, Any]] = []
    all_reproduced = True
    for task, (expected_name, expected_score) in EXPECTED_E01_FINAL_TASKS.items():
        if task not in tasks or task not in frozen:
            raise DataContractError(f"E12a-v2 E01 final task missing:{task}")
        observed_name = str((tasks[task].get("incumbent") or {}).get("name", ""))
        if observed_name != expected_name or str(frozen.get(task, "")) != expected_name:
            raise DataContractError(f"E12a-v2 incumbent identity mismatch:{task}")
        metrics = (tasks[task].get("incumbent") or {}).get("metrics") or {}
        observed = _finite_float(
            metrics.get("study_equal_weight_spearman_mean_strict"),
            label=f"{task}.study_equal_spearman",
        )
        deviation = observed - expected_score
        reproduced = abs(deviation) <= REPRODUCTION_ABSOLUTE_TOLERANCE
        all_reproduced = all_reproduced and reproduced
        rows.append(
            {
                "task": task,
                "incumbent": expected_name,
                "source": "accepted_e01_reproduction_harness",
                "expected_study_equal_spearman": expected_score,
                "observed_study_equal_spearman": observed,
                "deviation": deviation,
                "absolute_deviation": abs(deviation),
                "reproduced": bool(reproduced),
                "undefined_fold_count": int(metrics.get("undefined_fold_count", -1)),
                "constant_fold_count": int(metrics.get("constant_fold_count", -1)),
            }
        )

    if task13_anchor.get("task") != "Task1.3":
        raise DataContractError("E12a-v2 Task1.3 anchor audit identity mismatch")
    if task13_anchor.get("incumbent") != TASK13_FINAL_INCUMBENT:
        raise DataContractError("E12a-v2 Task1.3 final incumbent identity mismatch")
    if task13_anchor.get("predictor_feature") != TASK13_ANCHOR_COLUMN:
        raise DataContractError("E12a-v2 Task1.3 anchor feature mismatch")
    if task13_anchor.get("contains_participant_identifiers") is not False:
        raise DataContractError("E12a-v2 Task1.3 identifier boundary changed")
    if task13_anchor.get("contains_row_level_predictions") is not False:
        raise DataContractError("E12a-v2 Task1.3 row-level boundary changed")
    if int(task13_anchor.get("train_rows", -1)) != TASK13_EXPECTED_TRAIN_ROWS:
        raise DataContractError("E12a-v2 Task1.3 strict train rows changed")
    if int(task13_anchor.get("challenge_rows", -1)) != TASK13_EXPECTED_CHALLENGE_ROWS:
        raise DataContractError("E12a-v2 Task1.3 challenge rows changed")
    if task13_anchor.get("challenge_anchor_complete") is not True:
        raise DataContractError("E12a-v2 Task1.3 challenge anchor incomplete")

    task13_observed = _finite_float(
        task13_anchor.get("study_equal_spearman"),
        label="Task1.3.strict_anchor",
    )
    task13_deviation = task13_observed - TASK13_EXPECTED_STUDY_EQUAL_SPEARMAN
    task13_reproduced = abs(task13_deviation) <= REPRODUCTION_ABSOLUTE_TOLERANCE
    if bool(task13_anchor.get("reproduced")) is not task13_reproduced:
        raise DataContractError("E12a-v2 Task1.3 reproduction arithmetic mismatch")
    all_reproduced = all_reproduced and task13_reproduced
    rows.append(
        {
            "task": "Task1.3",
            "incumbent": TASK13_FINAL_INCUMBENT,
            "source": "E04/E04b retained strict ASC anchor",
            "predictor_feature": TASK13_ANCHOR_COLUMN,
            "expected_study_equal_spearman": TASK13_EXPECTED_STUDY_EQUAL_SPEARMAN,
            "observed_study_equal_spearman": task13_observed,
            "deviation": task13_deviation,
            "absolute_deviation": abs(task13_deviation),
            "reproduced": bool(task13_reproduced),
            "challenge_rows": int(task13_anchor["challenge_rows"]),
            "challenge_anchor_complete": True,
            "challenge_anchor_unique_values": int(
                task13_anchor.get("challenge_anchor_unique_values", 0)
            ),
        }
    )

    rows.sort(key=lambda row: row["task"])
    final_portfolio = dict(EXPECTED_FINAL_PORTFOLIO)
    if final_portfolio["Task1.3"] == TASK13_E01_CONTROL_INCUMBENT:
        raise DataContractError("E12a-v2 stale Task1.3 E01 control entered final portfolio")

    task14 = {
        "task": "Task1.4",
        "incumbent": EXPECTED_FINAL_PORTFOLIO["Task1.4"],
        "source_experiment": "strategy_v2_e07_task14_formal_closure",
        "supervised_cv_available": False,
        "outcomes_accessed": False,
        "challenge_subjects_expected": 40,
        "incumbent_changed": False,
    }
    next_step = (
        "proceed_to_E12b_challenge_prediction_freeze"
        if all_reproduced
        else "investigate_reconciliation_contract_no_model_selection"
    )
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "comparison_contract": COMPARISON_CONTRACT,
        "reproduction_absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
        "source_e12a_v1_experiment": "strategy_v2_e12a_final_system_reproduction",
        "source_e12a_v1_disposition":
            "reproduction_harness_passed_final_portfolio_invalid_stale_task13_identity",
        "source_e01_experiment": "strategy_v2_e01_paired_evaluation",
        "portfolio_manifest_corrected": True,
        "corrected_task": "Task1.3",
        "stale_task13_e01_control": {
            "incumbent": TASK13_E01_CONTROL_INCUMBENT,
            "study_equal_spearman": task13_e01_score,
            "final_portfolio_member": False,
        },
        "task13_strict_anchor_gain_vs_e01_control":
            TASK13_EXPECTED_STUDY_EQUAL_SPEARMAN - task13_e01_score,
        "supervised_reproduction": rows,
        "all_supervised_tasks_reproduced": bool(all_reproduced),
        "task14_contract": task14,
        "e11_task11_candidates_carried": [],
        "frozen_portfolio": final_portfolio,
        "portfolio_provenance": dict(PORTFOLIO_PROVENANCE),
        "portfolio_task_count": 7,
        "new_model_selection_performed": False,
        "competition_incumbent_changed": False,
        "public_probe_authorized": False,
        "competition_submission_authorized": False,
        "leaderboard_used_for_selection": False,
        "competition_submission_attempted": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
        "next_step": next_step,
    }
    _assert_aggregate_only(result)
    return result


def run_e12a_v2(config: BaselineConfig, inputs: InputBundle) -> dict[str, Any]:
    """Run accepted E01 reproduction plus the current strict Task1.3 anchor."""
    e01 = run_strategy_e01(config, inputs)
    task13 = run_strict_task13_anchor_audit(config, inputs)
    return evaluate_e12a_v2(e01, task13)
