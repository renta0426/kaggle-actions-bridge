"""Strategy-v2 E12a: reproduce the frozen final-system portfolio before packaging.

E12a is intentionally not a model-search experiment. It reruns the accepted E01
paired subject-purged harness for the six supervised tasks, compares only the
frozen incumbent top-line metrics and identities against predeclared references,
and carries Task1.4 as the E07-closed unsupervised raw Pre-vacc Conserved AIM
anchor. No Challenge prediction rows, participant identifiers, or submissions
leave this module.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .configuration import BaselineConfig
from .contracts import DataContractError
from .runner import InputBundle
from .strategy_e01 import run_strategy_e01

EXPERIMENT = "strategy_v2_e12a_final_system_reproduction"
COMPARISON_CONTRACT = "paired_subject_purged_v2"
REPRODUCTION_ABSOLUTE_TOLERANCE = 2e-6

EXPECTED_INCUMBENTS = {
    "Task1.1": "b21_pls_2",
    "Task1.2": "task12_anchor_residual_et_d5_l5_sqrt_lambda0.5",
    "Task1.3": "b21_pls_1",
    "Task1.4": "raw_pre_vacc_conserved_anchor",
    "Task2.1": "b21_et_subtype_d3_l5",
    "Task2.2": "b21_et_subtype_d5_l10",
    "Task2.3": "b21_ridge_exact_a100",
}

EXPECTED_SUPERVISED_SPEARMAN = {
    "Task1.1": 0.099707180,
    "Task1.2": 0.527103,
    "Task1.3": 0.106304108,
    "Task2.1": 0.623449,
    "Task2.2": 0.576506,
    "Task2.3": 0.695731,
}

E11_TASK11_CANDIDATES_CARRIED: tuple[str, ...] = ()
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
        raise DataContractError(f"E12a nonnumeric reproduction value:{label}") from exc
    if not np.isfinite(number):
        raise DataContractError(f"E12a nonfinite reproduction value:{label}")
    return number


def _assert_aggregate_only(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _BANNED_AGGREGATE_KEYS:
                raise DataContractError(f"E12a aggregate serialization leaked row-level key:{key}")
            _assert_aggregate_only(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_aggregate_only(item)


def evaluate_e12a_from_e01_result(e01: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one aggregate E01 result against the frozen E12a references."""
    if e01.get("experiment") != "strategy_v2_e01_paired_evaluation":
        raise DataContractError("E12a requires the accepted E01 aggregate result")
    if e01.get("comparison_contract") != COMPARISON_CONTRACT:
        raise DataContractError("E12a comparison contract mismatch")
    if e01.get("contains_participant_identifiers") is not False:
        raise DataContractError("E12a E01 aggregate identifier boundary changed")
    if e01.get("contains_row_level_predictions") is not False:
        raise DataContractError("E12a E01 aggregate row-level boundary changed")
    if e01.get("leaderboard_used_for_selection") is not False:
        raise DataContractError("E12a E01 leaderboard boundary changed")
    if e01.get("competition_submission_attempted") is not False:
        raise DataContractError("E12a E01 submission boundary changed")

    frozen = dict(e01.get("frozen_incumbent") or {})
    supervised = tuple(EXPECTED_SUPERVISED_SPEARMAN)
    if set(frozen) != set(supervised):
        raise DataContractError("E12a E01 frozen incumbent task set mismatch")
    tasks = e01.get("tasks") or {}
    if set(tasks) != set(supervised):
        raise DataContractError("E12a E01 task aggregate set mismatch")

    rows: list[dict[str, Any]] = []
    all_reproduced = True
    for task in supervised:
        expected_name = EXPECTED_INCUMBENTS[task]
        observed_name = str((tasks[task].get("incumbent") or {}).get("name", ""))
        frozen_name = str(frozen.get(task, ""))
        if observed_name != expected_name or frozen_name != expected_name:
            raise DataContractError(f"E12a incumbent identity mismatch:{task}")
        metrics = (tasks[task].get("incumbent") or {}).get("metrics") or {}
        observed = _finite_float(
            metrics.get("study_equal_weight_spearman_mean_strict"),
            label=f"{task}.study_equal_spearman",
        )
        expected = float(EXPECTED_SUPERVISED_SPEARMAN[task])
        deviation = observed - expected
        reproduced = abs(deviation) <= REPRODUCTION_ABSOLUTE_TOLERANCE
        all_reproduced = all_reproduced and reproduced
        rows.append(
            {
                "task": task,
                "incumbent": expected_name,
                "expected_study_equal_spearman": expected,
                "observed_study_equal_spearman": observed,
                "deviation": deviation,
                "absolute_deviation": abs(deviation),
                "reproduced": bool(reproduced),
                "undefined_fold_count": int(metrics.get("undefined_fold_count", -1)),
                "constant_fold_count": int(metrics.get("constant_fold_count", -1)),
            }
        )

    task14 = {
        "task": "Task1.4",
        "incumbent": EXPECTED_INCUMBENTS["Task1.4"],
        "source_experiment": "strategy_v2_e07_task14_formal_closure",
        "supervised_cv_available": False,
        "outcomes_accessed": False,
        "challenge_subjects_expected": 40,
        "incumbent_changed": False,
    }
    portfolio = {task: EXPECTED_INCUMBENTS[task] for task in EXPECTED_INCUMBENTS}
    next_step = (
        "proceed_to_E12b_challenge_prediction_freeze"
        if all_reproduced
        else "investigate_reproduction_contract_no_model_selection"
    )
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "comparison_contract": COMPARISON_CONTRACT,
        "source_e01_experiment": "strategy_v2_e01_paired_evaluation",
        "source_e07_experiment": "strategy_v2_e07_task14_formal_closure",
        "reproduction_absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
        "supervised_reproduction": rows,
        "all_supervised_tasks_reproduced": bool(all_reproduced),
        "task14_contract": task14,
        "e11_task11_candidates_carried": list(E11_TASK11_CANDIDATES_CARRIED),
        "frozen_portfolio": portfolio,
        "portfolio_task_count": 7,
        "new_model_selection_performed": False,
        "incumbent_changed": False,
        "public_probe_authorized": False,
        "competition_submission_authorized": False,
        "leaderboard_used_for_selection": False,
        "competition_submission_attempted": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
        "next_step": next_step,
    }


def run_e12a(config: BaselineConfig, inputs: InputBundle) -> dict[str, Any]:
    """Rerun E01 and apply the frozen E12a reproduction-only decision rule."""
    result = evaluate_e12a_from_e01_result(run_strategy_e01(config, inputs))
    _assert_aggregate_only(result)
    return result
