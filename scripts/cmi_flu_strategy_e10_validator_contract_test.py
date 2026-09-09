#!/usr/bin/env python3
"""Exercise the generated E10 full aggregate validator without competition data."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def audit() -> dict:
    return {
        "source_weights": {"A": 1.0, "B": 1.0},
        "source_counts": {"A": 5, "B": 8},
        "row_weight_mean": 1.0,
        "row_weight_ess_fraction": 1.0,
        "held_outcomes_used": False,
    }


def fold(study: str) -> dict:
    return {
        "held_study": study,
        "n": 8,
        "training_rows": 20,
        "training_studies": 2,
        "base_spearman": 0.20,
        "xonly_weighted_shared_spearman": 0.23,
        "shrunk_study_deviation_spearman": 0.23,
        "xonly_weighted_shared_delta": 0.03,
        "shrunk_study_deviation_delta": 0.03,
        "xonly_weight_audit": audit(),
        "xonly_weighted_fit": {"pooled_model_count": 1, "isolated_source_models_fit": 0},
        "shrunk_study_deviation_fit": {
            "pooled_model_count": 1,
            "isolated_source_models_fit": 0,
            "unseen_target_deviation_columns": "all_zero",
        },
        "xonly_weighted_movement": {"max_absolute_rank_correction": 0.01},
        "shrunk_study_deviation_movement": {"max_absolute_rank_correction": 0.01},
    }


def candidate(fold_count: int) -> dict:
    return {
        "promotion": {
            "passed": True,
            "mean_delta": 0.03,
            "minimum_delta": 0.03,
            "wins": fold_count,
            "required_wins": fold_count // 2 + 1,
        },
        "challenge_agreement_vs_base": {
            "rank_spearman": {"status": "ok", "value": 0.95},
            "changed_rank_count": 4,
        },
        "challenge_movement": {"max_absolute_rank_correction": 0.01},
    }


def task_payload(task: str, fold_count: int) -> dict:
    if task == "Task1.1":
        base = {
            "kind": "b21",
            "model": "pls_2",
            "anchor_column": None,
            "anchor_lambda": None,
            "current_competition_incumbent": True,
        }
    else:
        base = {
            "kind": "anchor_residual",
            "model": "et_d5_l5_sqrt",
            "anchor_column": "flow_rank__Classical_monocytes",
            "anchor_lambda": 0.5,
            "current_competition_incumbent": True,
        }
    folds = [fold(f"S{i}") for i in range(fold_count)]
    return {
        "task": task,
        "base_contract": base,
        "folds": folds,
        "candidates": {
            "xonly_weighted_shared": candidate(fold_count),
            "shrunk_study_deviation": candidate(fold_count),
        },
        "selected_local_candidate": "xonly_weighted_shared",
        "competition_candidate": False,
        "public_probe_authorized": False,
        "challenge_source_weight_audit": audit(),
        "challenge_rows": 40,
    }


def fixture() -> dict:
    return {
        "schema_version": 1,
        "experiment": "strategy_v2_e10_pooled_domain_conditioned_residual",
        "comparison_contract": "paired_subject_purged_v2",
        "tasks": {
            "Task1.1": task_payload("Task1.1", 4),
            "Task1.2": task_payload("Task1.2", 3),
        },
        "frozen_conditions": {
            "ridge_alpha": 10.0,
            "correction_shrinkage": 0.25,
            "absolute_rank_correction_cap": 0.05,
            "source_weight_clip": [0.5, 2.0],
            "deviation_scale": 0.25,
            "minimum_source_subjects": 5,
            "promotion_mean_delta": 0.02,
            "promotion_minimum_study_delta": -0.10,
            "task_contract": {
                "Task1.1": {
                    "model_set": "task_11",
                    "base_model": "pls_2",
                    "base_kind": "b21",
                    "assay_prefix": "cytokine_",
                    "anchor_column": None,
                    "anchor_lambda": None,
                },
                "Task1.2": {
                    "model_set": "task_12",
                    "base_model": "et_d5_l5_sqrt",
                    "base_kind": "anchor_residual",
                    "assay_prefix": "flow_",
                    "anchor_column": "flow_rank__Classical_monocytes",
                    "anchor_lambda": 0.5,
                },
            },
        },
        "held_target_outcomes_used_for_domain_weights": False,
        "isolated_source_study_models_allowed": False,
        "leaderboard_used_for_selection": False,
        "competition_submission_attempted": False,
        "incumbent_changed": False,
        "public_probe_authorized": False,
        "automatic_compute_retries": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    path = parser.parse_args().runtime.resolve()
    spec = importlib.util.spec_from_file_location("generated_e10_validator_contract_test", path)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load generated E10 runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_result(fixture(), synthetic=False)
    print("CMI_FLU_E10_FULL_VALIDATOR_CONTRACT_PASS competition_data=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
