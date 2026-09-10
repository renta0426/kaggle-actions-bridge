"""Competition-data-free synthetic regression contract for Strategy-v2 E11c."""
from __future__ import annotations

import time

import numpy as np

from .strategy_e11b_synthetic import (
    CPU_FEASIBILITY_SECONDS,
    STUDY_COUNTS,
    make_config,
    make_dataset,
    synthetic_estimator_factory,
)
from .strategy_e11c import (
    BASE_WEIGHT,
    TABPFN_WEIGHT,
    audit_e11c_feasibility,
    evaluate_task_e11c,
)

E11C_CPU_FEASIBILITY_SECONDS = max(60.0, CPU_FEASIBILITY_SECONDS * 2.0)


def run_synthetic_e11c() -> dict:
    dataset = make_dataset()
    feasibility = audit_e11c_feasibility(dataset)
    started = time.perf_counter()
    task = evaluate_task_e11c(
        dataset,
        config=make_config(),
        checkpoint_path=None,
        estimator_factory=synthetic_estimator_factory,
    )
    elapsed = float(time.perf_counter() - started)

    folds = task["folds"]
    source_drop_counts = [len(fold["source_drop_fits"]) for fold in folds]
    held_names = [fold["held_study"] for fold in folds]
    all_drop_membership_safe = True
    all_fusions_fixed = True
    all_audits_outcome_free = True
    all_scores_finite = True
    for fold in folds:
        held = fold["held_study"]
        for audit in fold["source_drop_fits"]:
            if held in audit["training_studies"]:
                all_drop_membership_safe = False
            if audit["dropped_source_study"] in audit["training_studies"]:
                all_drop_membership_safe = False

        c2 = fold["conditions"]["C2"]
        c4 = fold["conditions"]["C4"]
        all_scores_finite = all_scores_finite and all(
            np.isfinite(float(item["spearman"]))
            for item in (c2, c4)
            if item["spearman"] is not None
        )
        all_fusions_fixed = all_fusions_fixed and (
            task["conditions"]["C2"]["base_weight"] == BASE_WEIGHT
            and task["conditions"]["C2"]["tabpfn_weight"] == TABPFN_WEIGHT
            and task["conditions"]["C4"]["base_weight"] == BASE_WEIGHT
            and task["conditions"]["C4"]["tabpfn_weight"] == TABPFN_WEIGHT
        )
        all_audits_outcome_free = all_audits_outcome_free and (
            fold["feature_shift_audit"]["held_outcomes_used"] is False
            and fold["prediction_disagreement_audit"]["held_outcomes_used"] is False
            and fold["held_outcomes_used_for_fusion"] is False
            and fold["held_outcomes_used_for_transfer_audit"] is False
        )

    result = {
        "schema_version": 1,
        "experiment": "synthetic_e11c_robust_tabpfn_contract",
        "study_counts": dict(STUDY_COUNTS),
        "outer_split_count": int(len(folds)),
        "held_studies": held_names,
        "source_drop_fit_counts": source_drop_counts,
        "outer_full_tabpfn_fits": int(
            task["fit_accounting"]["outer_full_tabpfn_fits"]
        ),
        "outer_source_drop_tabpfn_fits": int(
            task["fit_accounting"]["outer_source_drop_tabpfn_fits"]
        ),
        "total_tabpfn_fits": int(task["fit_accounting"]["total_tabpfn_fits"]),
        "maximum_tabpfn_fits": int(
            task["fit_accounting"]["maximum_allowed_if_both_candidates_constructed"]
        ),
        "all_drop_membership_safe": bool(all_drop_membership_safe),
        "all_fusions_fixed": bool(all_fusions_fixed),
        "all_audits_outcome_free": bool(all_audits_outcome_free),
        "all_candidate_scores_finite": bool(all_scores_finite),
        "promotion_conditions": sorted(task["promotion"]),
        "e11c_winner_selected": bool(task["e11c_winner_selected"]),
        "public_probe_authorized": bool(task["public_probe_authorized"]),
        "competition_submission_authorized": bool(
            task["competition_submission_authorized"]
        ),
        "dynamic_domain_gate_fitted": bool(task["dynamic_domain_gate_fitted"]),
        "fit_and_score_seconds": elapsed,
        "cpu_feasibility_seconds": E11C_CPU_FEASIBILITY_SECONDS,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }

    if set(result["held_studies"]) != set(STUDY_COUNTS):
        raise RuntimeError("E11c synthetic outer held-study coverage changed")
    if result["outer_split_count"] != 4:
        raise RuntimeError("E11c synthetic must have exactly four outer folds")
    if result["source_drop_fit_counts"] != [3, 3, 3, 3]:
        raise RuntimeError("E11c synthetic must fit exactly three source drops per fold")
    if result["outer_full_tabpfn_fits"] != 4:
        raise RuntimeError("E11c synthetic full-control fit count changed")
    if result["outer_source_drop_tabpfn_fits"] != 12:
        raise RuntimeError("E11c synthetic source-drop fit count changed")
    if not result["all_drop_membership_safe"]:
        raise RuntimeError("E11c synthetic source/held purge contract failed")
    if not result["all_fusions_fixed"]:
        raise RuntimeError("E11c synthetic fusion weights changed")
    if not result["all_audits_outcome_free"]:
        raise RuntimeError("E11c synthetic audit consumed held outcomes")
    if not result["all_candidate_scores_finite"]:
        raise RuntimeError("E11c synthetic candidate score is non-finite")
    if result["promotion_conditions"] != ["C2", "C4"]:
        raise RuntimeError("E11c synthetic promotion eligibility changed")
    if result["e11c_winner_selected"]:
        raise RuntimeError("E11c synthetic performed forbidden winner selection")
    if result["public_probe_authorized"] or result["competition_submission_authorized"]:
        raise RuntimeError("E11c synthetic authorized Competition-facing selection")
    if result["dynamic_domain_gate_fitted"]:
        raise RuntimeError("E11c synthetic fitted a forbidden dynamic gate")
    if result["total_tabpfn_fits"] > result["maximum_tabpfn_fits"]:
        raise RuntimeError("E11c synthetic exceeded the frozen fit budget")
    if result["fit_and_score_seconds"] > E11C_CPU_FEASIBILITY_SECONDS:
        raise RuntimeError("E11c synthetic CPU feasibility budget exceeded")
    return result


__all__ = ["run_synthetic_e11c"]
