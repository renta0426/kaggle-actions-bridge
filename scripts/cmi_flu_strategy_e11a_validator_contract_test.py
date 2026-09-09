#!/usr/bin/env python3
"""Exercise generated E11a full aggregate validator without Competition Data."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 10,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 20260910,
}


def fit_audit(studies: int, rows: int) -> dict:
    return {
        "pairs_within_study_only": True,
        "directed_pairs": 200,
        "unordered_pairs": 100,
        "positive_pairs": 100,
        "negative_pairs": 100,
        "unordered_pairs_by_study": {f"S{i}": 50 for i in range(studies)},
        "ties_skipped_by_study": {f"S{i}": 0 for i in range(studies)},
        "same_subject_pairs_skipped_by_study": {f"S{i}": 0 for i in range(studies)},
        "cross_study_pair_labels_used": False,
        "training_rows": rows,
        "training_subjects": rows,
        "training_studies": studies,
        "transformed_features": 10,
        "numeric_features": 8,
        "categorical_features": 2,
        "reference_aggregation": "study_equal",
        "held_outcomes_used_for_fit": False,
        "fit_seconds": 0.1,
        "classifier": "HistGradientBoostingClassifier",
        "classifier_params": dict(PARAMS),
    }


def fixture() -> dict:
    held = ["SDY180", "SDY515", "SDY519", "SDY56"]
    folds = [
        {
            "held_study": study,
            "n": 10,
            "base_spearman": 0.20,
            "pairwise_spearman": 0.23,
            "pairwise_delta": 0.03,
            "fit_audit": fit_audit(3, 80),
            "held_outcomes_used_for_fit": False,
            "held_outcomes_used_for_scoring": False,
        }
        for study in held
    ]
    feasibility = {
        "task": "Task1.1",
        "train_rows": 127,
        "challenge_rows": 40,
        "study_counts": {"SDY180": 34, "SDY515": 16, "SDY519": 17, "SDY56": 60},
        "outer_splits": [
            {
                "split": f"study={study}",
                "held_study": study,
                "train_rows": 80,
                "validation_rows": 10,
                "train_studies": 3,
                "subject_overlap": 0,
            }
            for study in held
        ],
        "pair_policy": "within_source_study_only_symmetric",
        "reference_aggregation": "study_equal",
        "held_outcomes_accepted_by_scoring_api": False,
        "public_probe_authorized": False,
        "competition_submission_attempted": False,
        "ready_for_one_cpu_pairwise_comparison": True,
    }
    return {
        "schema_version": 1,
        "experiment": "strategy_v2_e11a_task11_pairwise_ranker",
        "comparison_contract": "paired_subject_purged_v2",
        "task": {
            "task": "Task1.1",
            "base_contract": {"kind": "b21", "model": "pls_2", "current_competition_incumbent": True},
            "candidate": {
                "name": "pairwise_hgb",
                "promotion": {
                    "passed": True,
                    "usable_held_studies": 4,
                    "mean_delta": 0.03,
                    "median_delta": 0.03,
                    "minimum_delta": 0.03,
                    "maximum_delta": 0.03,
                    "wins": 4,
                    "required_wins": 3,
                    "mean_delta_threshold": 0.02,
                    "minimum_study_delta_threshold": -0.10,
                },
                "challenge_agreement_vs_base": {
                    "rank_spearman": {"status": "ok", "value": 0.95, "n": 40},
                    "changed_rank_count": 4,
                    "mean_absolute_percentile_shift": 0.01,
                    "max_absolute_percentile_shift": 0.03,
                },
                "fit": fit_audit(4, 127),
            },
            "folds": folds,
            "selected_local_candidate": "pairwise_hgb",
            "competition_candidate": False,
            "public_probe_authorized": False,
            "challenge_rows": 40,
            "feasibility": feasibility,
        },
        "frozen_conditions": {
            "task": "Task1.1",
            "base_model": "pls_2",
            "pair_policy": "within_source_study_only_symmetric",
            "pair_tie_tolerance": 1e-12,
            "max_rows_per_study": 80,
            "max_transformed_features": 512,
            "minimum_directed_pairs": 100,
            "reference_aggregation": "study_equal",
            "classifier": "HistGradientBoostingClassifier",
            "classifier_params": dict(PARAMS),
            "promotion_mean_delta": 0.02,
            "promotion_minimum_study_delta": -0.10,
            "promotion_requires_strict_majority_wins": True,
        },
        "cross_study_pair_labels_used": False,
        "held_outcomes_used_for_fit": False,
        "held_outcomes_used_for_scoring": False,
        "leaderboard_used_for_selection": False,
        "competition_submission_attempted": False,
        "incumbent_changed": False,
        "public_probe_authorized": False,
        "automatic_compute_retries": 0,
        "interpretation_limits": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    path = parser.parse_args().runtime.resolve()
    spec = importlib.util.spec_from_file_location("generated_e11a_validator_contract", path)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load generated E11a runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_result(fixture(), synthetic=False)
    print("CMI_FLU_E11A_FULL_VALIDATOR_CONTRACT_PASS competition_data=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
