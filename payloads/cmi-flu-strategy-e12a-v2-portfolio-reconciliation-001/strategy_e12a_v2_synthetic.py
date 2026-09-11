"""Competition-data-free regression checks for Strategy-v2 E12a-v2."""
from __future__ import annotations

from copy import deepcopy

from .contracts import DataContractError
from .strategy_e12a_v2 import (
    EXPECTED_E01_FINAL_TASKS,
    EXPECTED_FINAL_PORTFOLIO,
    REPRODUCTION_ABSOLUTE_TOLERANCE,
    TASK13_ANCHOR_COLUMN,
    TASK13_E01_CONTROL_INCUMBENT,
    TASK13_E01_CONTROL_SPEARMAN,
    TASK13_EXPECTED_STUDY_EQUAL_SPEARMAN,
    TASK13_FINAL_INCUMBENT,
    evaluate_e12a_v2,
)


def synthetic_e01_result() -> dict:
    tasks = {}
    frozen = {}
    for task, (name, score) in EXPECTED_E01_FINAL_TASKS.items():
        tasks[task] = {
            "incumbent": {
                "name": name,
                "metrics": {
                    "study_equal_weight_spearman_mean_strict": score,
                    "undefined_fold_count": 0,
                    "constant_fold_count": 0,
                },
            }
        }
        frozen[task] = name
    tasks["Task1.3"] = {
        "incumbent": {
            "name": TASK13_E01_CONTROL_INCUMBENT,
            "metrics": {
                "study_equal_weight_spearman_mean_strict": TASK13_E01_CONTROL_SPEARMAN,
                "undefined_fold_count": 0,
                "constant_fold_count": 0,
            },
        }
    }
    frozen["Task1.3"] = TASK13_E01_CONTROL_INCUMBENT
    return {
        "schema_version": 1,
        "experiment": "strategy_v2_e01_paired_evaluation",
        "comparison_contract": "paired_subject_purged_v2",
        "frozen_incumbent": frozen,
        "tasks": tasks,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
        "leaderboard_used_for_selection": False,
        "competition_submission_attempted": False,
    }


def synthetic_task13_anchor_audit() -> dict:
    return {
        "task": "Task1.3",
        "incumbent": TASK13_FINAL_INCUMBENT,
        "predictor_feature": TASK13_ANCHOR_COLUMN,
        "train_rows": 23,
        "historical_study_count": 1,
        "study_equal_spearman": TASK13_EXPECTED_STUDY_EQUAL_SPEARMAN,
        "expected_study_equal_spearman": TASK13_EXPECTED_STUDY_EQUAL_SPEARMAN,
        "deviation": 0.0,
        "absolute_deviation": 0.0,
        "reproduced": True,
        "challenge_rows": 40,
        "challenge_anchor_complete": True,
        "challenge_anchor_unique_values": 17,
        "challenge_anchor_min": 0.0,
        "challenge_anchor_max": 1.0,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }


def run_synthetic_e12a_v2() -> dict:
    e01 = synthetic_e01_result()
    anchor = synthetic_task13_anchor_audit()
    result = evaluate_e12a_v2(e01, anchor)
    if result["all_supervised_tasks_reproduced"] is not True:
        raise AssertionError("E12a-v2 exact synthetic reproduction did not pass")
    if result["frozen_portfolio"] != EXPECTED_FINAL_PORTFOLIO:
        raise AssertionError("E12a-v2 final portfolio mismatch")
    if result["frozen_portfolio"]["Task1.3"] != TASK13_FINAL_INCUMBENT:
        raise AssertionError("E12a-v2 failed to retain strict Task1.3 anchor")
    if result["stale_task13_e01_control"]["final_portfolio_member"] is not False:
        raise AssertionError("E12a-v2 stale E01 Task1.3 entered final portfolio")

    drifted_anchor = deepcopy(anchor)
    drifted_anchor["study_equal_spearman"] += 2 * REPRODUCTION_ABSOLUTE_TOLERANCE
    drifted_anchor["reproduced"] = False
    drifted = evaluate_e12a_v2(e01, drifted_anchor)
    if drifted["all_supervised_tasks_reproduced"] is not False:
        raise AssertionError("E12a-v2 failed to reject strict-anchor drift")
    if drifted["next_step"] != "investigate_reconciliation_contract_no_model_selection":
        raise AssertionError("E12a-v2 drift next-step mismatch")

    bad_identity = deepcopy(e01)
    bad_identity["tasks"]["Task2.1"]["incumbent"]["name"] = "unexpected_model"
    try:
        evaluate_e12a_v2(bad_identity, anchor)
    except DataContractError:
        pass
    else:
        raise AssertionError("E12a-v2 accepted final-task identity drift")

    return {
        "schema_version": 1,
        "experiment": "synthetic_e12a_v2_portfolio_reconciliation",
        "exact_reproduction_passed": True,
        "stale_task13_control_rejected_from_final_portfolio": True,
        "strict_task13_drift_rejected": True,
        "identity_drift_rejected": True,
        "portfolio_task_count": 7,
        "new_model_selection_performed": False,
        "public_probe_authorized": False,
        "competition_submission_authorized": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
