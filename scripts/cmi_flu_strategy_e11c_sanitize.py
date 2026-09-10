#!/usr/bin/env python3
"""Validate and print only aggregate-safe E11c Task1.1 outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260910-cmi-flu-strategy-e11c-robust-tabpfn-001"
TARGET = "renta0426/cmi-flu-e11c-robust-tabpfn-20260910-001"
SCIENCE_COMMIT = "0815f4517345e127dbcbcae0f380a54f3a3d15bd"
E11C_BLOB = "acf876996b5849f7ba794999d45ecb862d1be507"
E11C_SYNTH_BLOB = "9b94633192b5254a7d1debe8db10606b48104076"
PACKAGE_VERSION = "8.5.0"
SOURCE_COMMIT = "9ed44abd5882140b88c9f2816c5791987ce059b9"
WHEEL_FILENAME = "tabpfn-8.5.0-py3-none-any.whl"
WHEEL_SHA256 = "4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0"
MODEL_SOURCE = "prior-labsai/tabpfn-3/pytorch/default/1"
CHECKPOINT_FILENAME = "tabpfn-v3-regressor-v3_default.ckpt"
CHECKPOINT_BYTES = 233_289_807
CHECKPOINT_SHA256 = "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301"
PARAMS = {
    "n_estimators": 8,
    "device": "cpu",
    "fit_mode": "fit_preprocessors",
    "memory_saving_mode": "auto",
    "random_state": 20260910,
    "n_preprocessing_jobs": 1,
    "show_progress_bar": False,
    "ignore_pretraining_limits": False,
}
BANNED_TEXT = (
    '"participant_id"',
    '"subject_group"',
    '"row_index"',
    '"oof_predictions"',
    '"challenge_predictions"',
    "KGAT_",
    "KAGGLE_API_TOKEN",
    "Authorization:",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SystemExit(f"E11c nonfinite numeric value:{label}")
    return number


def _validate_bridge(root: Path, bridge: dict, *, synthetic: bool) -> None:
    expected = {
        "request_id": REQUEST_ID,
        "target_kernel": TARGET,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e11c_blob_sha": E11C_BLOB,
        "strategy_e11c_synthetic_blob_sha": E11C_SYNTH_BLOB,
        "tabpfn_package_version": PACKAGE_VERSION,
        "tabpfn_wheel_sha256": WHEEL_SHA256,
        "tabpfn_model_source": MODEL_SOURCE,
        "tabpfn_checkpoint_sha256": CHECKPOINT_SHA256,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
    for key, value in expected.items():
        if bridge.get(key) != value:
            raise SystemExit(f"E11c bridge contract mismatch:{key}")
    if bool(bridge.get("synthetic", False)) is not synthetic:
        raise SystemExit("E11c bridge synthetic flag mismatch")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json"):
        raise SystemExit("E11c metrics hash mismatch")
    if bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E11c summary hash mismatch")


def _validate_synthetic(metrics: dict) -> dict:
    if int(metrics.get("schema_version", -1)) != 1 or metrics.get("experiment") != "synthetic_e11c_robust_tabpfn_contract":
        raise SystemExit("E11c synthetic identity mismatch")
    if metrics.get("study_counts") != {"SDY180": 34, "SDY515": 16, "SDY519": 17, "SDY56": 60}:
        raise SystemExit("E11c synthetic study counts mismatch")
    if int(metrics.get("outer_split_count", -1)) != 4 or metrics.get("source_drop_fit_counts") != [3, 3, 3, 3]:
        raise SystemExit("E11c synthetic outer/drop contract mismatch")
    if int(metrics.get("outer_full_tabpfn_fits", -1)) != 4 or int(metrics.get("outer_source_drop_tabpfn_fits", -1)) != 12:
        raise SystemExit("E11c synthetic fit-count mismatch")
    if int(metrics.get("maximum_tabpfn_fits", -1)) != 21 or int(metrics.get("total_tabpfn_fits", 99)) > 21:
        raise SystemExit("E11c synthetic fit budget mismatch")
    for key in ("all_drop_membership_safe", "all_fusions_fixed", "all_audits_outcome_free", "all_candidate_scores_finite"):
        if metrics.get(key) is not True:
            raise SystemExit(f"E11c synthetic contract mismatch:{key}")
    for key in ("e11c_winner_selected", "public_probe_authorized", "competition_submission_authorized", "dynamic_domain_gate_fitted", "contains_participant_identifiers", "contains_row_level_predictions"):
        if metrics.get(key) is not False:
            raise SystemExit(f"E11c synthetic boundary mismatch:{key}")
    if metrics.get("promotion_conditions") != ["C2", "C4"]:
        raise SystemExit("E11c synthetic promotion set mismatch")
    return {
        "outer_split_count": 4,
        "outer_full_tabpfn_fits": 4,
        "outer_source_drop_tabpfn_fits": 12,
        "total_tabpfn_fits": int(metrics["total_tabpfn_fits"]),
        "maximum_tabpfn_fits": 21,
    }


def _validate_real(metrics: dict) -> dict:
    if int(metrics.get("schema_version", -1)) != 1 or metrics.get("experiment") != "strategy_v2_e11c_task11_robust_tabpfn":
        raise SystemExit("E11c experiment identity mismatch")
    if metrics.get("parent_experiment") != "E11b" or metrics.get("comparison_contract") != "paired_subject_purged_v2":
        raise SystemExit("E11c comparison/parent contract mismatch")
    for key in (
        "contains_participant_identifiers",
        "contains_row_level_predictions",
        "held_outcomes_used_for_fit",
        "held_outcomes_used_for_fusion",
        "held_outcomes_used_for_transfer_audit",
        "leaderboard_used_for_selection",
        "public_probe_authorized",
        "competition_submission_attempted",
        "incumbent_changed",
    ):
        if metrics.get(key) is not False:
            raise SystemExit(f"E11c execution/leakage boundary mismatch:{key}")
    if metrics.get("aggregate_only_serialization") is not True or int(metrics.get("automatic_compute_retries", -1)) != 0:
        raise SystemExit("E11c aggregate/retry boundary mismatch")

    frozen = metrics.get("frozen_conditions") or {}
    expected_frozen = {
        "task": "Task1.1",
        "base_model": "pls_2",
        "base_model_set": "task_11",
        "target_transform": "log",
        "expected_train_rows": 127,
        "expected_train_studies": 4,
        "expected_challenge_rows": 40,
        "max_raw_features": 200,
        "base_weight": 0.75,
        "tabpfn_weight": 0.25,
        "tabpfn_package_version": PACKAGE_VERSION,
        "tabpfn_source_commit": SOURCE_COMMIT,
        "tabpfn_wheel_filename": WHEEL_FILENAME,
        "tabpfn_wheel_sha256": WHEEL_SHA256,
        "kaggle_model_source": MODEL_SOURCE,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "tabpfn_params": PARAMS,
        "promotion_mean_delta": 0.02,
        "promotion_minimum_study_delta": -0.1,
        "promotion_requires_strict_majority_wins": True,
        "sensitivity_28": {
            "studies": ["SDY180", "SDY56"],
            "subset_n": 28,
            "repetitions": 200,
            "seed": 20260907,
            "used_for_promotion": False,
            "model_refits": 0,
        },
    }
    if frozen != expected_frozen:
        raise SystemExit("E11c frozen-condition mismatch")
    license_info = metrics.get("external_model_license") or {}
    if (
        license_info.get("model_license") != "tabpfn-3-license-v1.0"
        or license_info.get("weight_redistribution_by_project") is not False
        or license_info.get("official_kaggle_model_input") is not True
    ):
        raise SystemExit("E11c external-model license mismatch")

    task = metrics.get("task") or {}
    if task.get("task") != "Task1.1" or task.get("e11c_winner_selected") is not False:
        raise SystemExit("E11c task/winner contract mismatch")
    for key in ("dynamic_domain_gate_fitted", "public_probe_authorized", "competition_submission_authorized", "incumbent_changed"):
        if task.get(key) is not False:
            raise SystemExit(f"E11c task boundary mismatch:{key}")
    condition_contract = task.get("conditions") or {}
    if set(condition_contract) != {"C0", "C1", "C2", "C3", "C4"}:
        raise SystemExit("E11c condition set mismatch")
    if condition_contract.get("C2") != {"eligible_for_e12": True, "base_weight": 0.75, "tabpfn_weight": 0.25}:
        raise SystemExit("E11c C2 fixed-weight mismatch")
    if condition_contract.get("C4") != {"eligible_for_e12": True, "base_weight": 0.75, "tabpfn_weight": 0.25}:
        raise SystemExit("E11c C4 fixed-weight mismatch")
    if any(condition_contract[name].get("eligible_for_e12") is not False for name in ("C0", "C1", "C3")):
        raise SystemExit("E11c non-candidate eligibility mismatch")

    folds = task.get("folds") or []
    if len(folds) != 4 or {str(fold.get("held_study")) for fold in folds} != {"SDY180", "SDY515", "SDY519", "SDY56"}:
        raise SystemExit("E11c held-study contract mismatch")
    candidate_deltas = {"C2": [], "C4": []}
    aggregate_folds = []
    for fold in folds:
        held = str(fold.get("held_study"))
        n = int(fold.get("n", 0))
        if n < 3 or int(fold.get("source_study_count", -1)) != 3:
            raise SystemExit("E11c fold support/source count mismatch")
        conditions = fold.get("conditions") or {}
        if set(conditions) != {"C0", "C1", "C2", "C3", "C4"}:
            raise SystemExit("E11c fold condition set mismatch")
        c0 = finite(conditions["C0"].get("spearman"), f"{held}:C0")
        fold_result = {"held_study": held, "n": n, "conditions": {"C0": {"spearman": c0}}}
        for name in ("C1", "C2", "C3", "C4"):
            score = finite(conditions[name].get("spearman"), f"{held}:{name}:spearman")
            delta = finite(conditions[name].get("delta_vs_c0"), f"{held}:{name}:delta")
            if abs((score - c0) - delta) > 1e-10:
                raise SystemExit("E11c fold delta arithmetic mismatch")
            fold_result["conditions"][name] = {"spearman": score, "delta_vs_c0": delta}
        for name in ("C2", "C4"):
            candidate = conditions[name]
            if candidate.get("eligible_for_e12") is not True or candidate.get("raw_multiset_preserved") is not True:
                raise SystemExit("E11c fusion candidate mapping mismatch")
            rmse = finite(candidate.get("base_marginal_rank_remap_rmse"), f"{held}:{name}:rmse")
            c0_rmse = finite(candidate.get("c0_raw_rmse"), f"{held}:{name}:c0_rmse")
            candidate_deltas[name].append(float(candidate["delta_vs_c0"]))
            sensitivity = candidate.get("sensitivity_28") or {}
            if sensitivity.get("used_for_promotion") is not False or int(sensitivity.get("model_refits", -1)) != 0:
                raise SystemExit("E11c sensitivity selection/refit mismatch")
            if held in {"SDY180", "SDY56"}:
                if (
                    sensitivity.get("status") != "ok"
                    or int(sensitivity.get("subset_n", -1)) != 28
                    or int(sensitivity.get("repetitions", -1)) != 200
                    or int(sensitivity.get("seed", -1)) != 20260907
                ):
                    raise SystemExit("E11c predeclared sensitivity mismatch")
                sensitivity_safe = {
                    "status": "ok",
                    "spearman_delta_mean": finite(sensitivity.get("spearman_delta_mean"), "sensitivity_mean"),
                    "spearman_delta_median": finite(sensitivity.get("spearman_delta_median"), "sensitivity_median"),
                    "spearman_delta_q10": finite(sensitivity.get("spearman_delta_q10"), "sensitivity_q10"),
                    "spearman_delta_q90": finite(sensitivity.get("spearman_delta_q90"), "sensitivity_q90"),
                    "candidate_better_fraction": finite(sensitivity.get("candidate_better_fraction"), "sensitivity_fraction"),
                    "rank_rmse_delta_mean": finite(sensitivity.get("rank_rmse_delta_mean"), "sensitivity_rank_rmse"),
                }
            else:
                if sensitivity.get("status") != "not_run_nonpredeclared_study" or int(sensitivity.get("repetitions", -1)) != 0:
                    raise SystemExit("E11c nonpredeclared sensitivity mismatch")
                sensitivity_safe = {"status": "not_run_nonpredeclared_study"}
            fold_result["conditions"][name].update(
                {
                    "base_marginal_rank_remap_rmse": rmse,
                    "c0_raw_rmse": c0_rmse,
                    "changed_rank_count_vs_c0": int(candidate.get("changed_rank_count_vs_c0", -1)),
                    "mean_absolute_percentile_shift_vs_c0": finite(candidate.get("mean_absolute_percentile_shift_vs_c0"), "shift_mean"),
                    "max_absolute_percentile_shift_vs_c0": finite(candidate.get("max_absolute_percentile_shift_vs_c0"), "shift_max"),
                    "sensitivity_28": sensitivity_safe,
                }
            )
        drops = fold.get("source_drop_fits") or []
        if len(drops) != 3 or int(conditions["C3"].get("source_drop_fit_count", -1)) != 3:
            raise SystemExit("E11c source-drop fit-count mismatch")
        dropped = set()
        safe_drop_fits = []
        for audit in drops:
            dropped_name = str(audit.get("dropped_source_study"))
            training_studies = [str(value) for value in (audit.get("training_studies") or [])]
            if (
                audit.get("backend") != "tabpfn"
                or int(audit.get("training_study_count", -1)) != 2
                or len(training_studies) != 2
                or dropped_name in training_studies
                or held in training_studies
                or audit.get("held_outcomes_used_for_fit") is not False
            ):
                raise SystemExit("E11c source-drop membership/leakage mismatch")
            dropped.add(dropped_name)
            safe_drop_fits.append(
                {
                    "dropped_source_study": dropped_name,
                    "training_study_count": 2,
                    "training_rows": int(audit.get("training_rows", -1)),
                    "fit_seconds": finite(audit.get("fit_seconds"), "drop_fit_seconds"),
                }
            )
        if len(dropped) != 3:
            raise SystemExit("E11c source-drop uniqueness mismatch")

        feature = fold.get("feature_shift_audit") or {}
        if feature.get("held_outcomes_used") is not False or feature.get("used_for_candidate_routing") is not False:
            raise SystemExit("E11c feature-shift audit boundary mismatch")
        numeric = feature.get("numeric") or {}
        categorical = feature.get("categorical") or {}
        safe_feature = {
            "numeric": {
                "feature_count": int(numeric.get("feature_count", -1)),
                "usable_shift_feature_count": int(numeric.get("usable_shift_feature_count", -1)),
                "mean_absolute_held_median_shift_in_source_median_iqr_units": finite(numeric.get("mean_absolute_held_median_shift_in_source_median_iqr_units"), "numeric_shift_mean"),
                "p90_absolute_held_median_shift_in_source_median_iqr_units": finite(numeric.get("p90_absolute_held_median_shift_in_source_median_iqr_units"), "numeric_shift_p90"),
                "mean_absolute_missingness_rate_delta": finite(numeric.get("mean_absolute_missingness_rate_delta"), "numeric_missing_delta"),
                "zero_iqr_feature_count": int(numeric.get("zero_iqr_feature_count", -1)),
            },
            "categorical": {
                "feature_count": int(categorical.get("feature_count", -1)),
                "held_unseen_level_count": int(categorical.get("held_unseen_level_count", -1)),
                "held_unseen_unique_level_count": int(categorical.get("held_unseen_unique_level_count", -1)),
                "held_unseen_level_fraction": finite(categorical.get("held_unseen_level_fraction"), "unseen_fraction"),
                "mean_category_total_variation_distance": finite(categorical.get("mean_category_total_variation_distance"), "category_tv_mean"),
                "max_category_total_variation_distance": finite(categorical.get("max_category_total_variation_distance"), "category_tv_max"),
            },
        }
        disagreement = fold.get("prediction_disagreement_audit") or {}
        if disagreement.get("held_outcomes_used") is not False or int(disagreement.get("source_dropout_pair_count", -1)) != 3:
            raise SystemExit("E11c prediction-disagreement boundary mismatch")
        safe_disagreement = {
            "base_vs_full_tabpfn": disagreement.get("base_vs_full_tabpfn"),
            "base_vs_jackknife_tabpfn": disagreement.get("base_vs_jackknife_tabpfn"),
            "full_vs_jackknife_tabpfn": disagreement.get("full_vs_jackknife_tabpfn"),
            "source_dropout_pairwise_rank_spearman_min": disagreement.get("source_dropout_pairwise_rank_spearman_min"),
            "source_dropout_pairwise_rank_spearman_median": disagreement.get("source_dropout_pairwise_rank_spearman_median"),
            "source_dropout_pairwise_rank_spearman_max": disagreement.get("source_dropout_pairwise_rank_spearman_max"),
        }
        if (
            fold.get("held_outcomes_used_for_fit") is not False
            or fold.get("held_outcomes_used_for_fusion") is not False
            or fold.get("held_outcomes_used_for_transfer_audit") is not False
            or fold.get("held_outcomes_used_for_evaluation_only") is not True
        ):
            raise SystemExit("E11c outer-fold leakage boundary mismatch")
        fold_result["source_drop_fits"] = safe_drop_fits
        fold_result["feature_shift_audit"] = safe_feature
        fold_result["prediction_disagreement_audit"] = safe_disagreement
        aggregate_folds.append(fold_result)

    promotions = task.get("promotion") or {}
    safe_promotions = {}
    passing = []
    for name in ("C2", "C4"):
        values = candidate_deltas[name]
        wins = sum(value > 0 for value in values)
        required = 3
        mean_delta = sum(values) / len(values)
        minimum_delta = min(values)
        passed = bool(mean_delta >= 0.02 and minimum_delta >= -0.1 and wins >= required)
        observed = promotions.get(name) or {}
        if (
            observed.get("condition") != name
            or observed.get("passed") is not passed
            or int(observed.get("usable_held_studies", -1)) != 4
            or int(observed.get("wins", -1)) != wins
            or int(observed.get("required_wins", -1)) != required
            or observed.get("study_equal_primary") is not True
            or observed.get("row_weighted_diagnostic_only") is not True
        ):
            raise SystemExit("E11c promotion boolean/weighting mismatch")
        if abs(finite(observed.get("mean_delta"), "promotion_mean") - mean_delta) > 1e-10:
            raise SystemExit("E11c promotion mean arithmetic mismatch")
        if abs(finite(observed.get("minimum_delta"), "promotion_min") - minimum_delta) > 1e-10:
            raise SystemExit("E11c promotion min arithmetic mismatch")
        safe_promotions[name] = {
            "passed": passed,
            "mean_delta": mean_delta,
            "median_delta": finite(observed.get("median_delta"), "promotion_median"),
            "minimum_delta": minimum_delta,
            "maximum_delta": finite(observed.get("maximum_delta"), "promotion_max"),
            "wins": wins,
            "required_wins": required,
            "row_weighted_delta_diagnostic": finite(observed.get("row_weighted_delta_diagnostic"), "promotion_row_weighted"),
        }
        if passed:
            passing.append(name)
    if task.get("e12_eligible_conditions") != passing:
        raise SystemExit("E11c E12 eligible set mismatch")

    challenge = task.get("challenge") or {}
    final_expected = (1 if "C2" in passing else 0) + (4 if "C4" in passing else 0)
    if (
        int(challenge.get("rows", -1)) != 40
        or challenge.get("constructed_conditions") != passing
        or int(challenge.get("final_tabpfn_fit_count", -1)) != final_expected
        or challenge.get("candidate_construction_performed") is not bool(passing)
        or challenge.get("public_probe_authorized") is not False
        or challenge.get("competition_submission_authorized") is not False
        or challenge.get("challenge_labels_used") is not False
    ):
        raise SystemExit("E11c Challenge construction/boundary mismatch")
    safe_challenge = {
        "rows": 40,
        "constructed_conditions": passing,
        "candidate_construction_performed": bool(passing),
        "final_tabpfn_fit_count": final_expected,
        "candidate_diagnostics": challenge.get("candidate_diagnostics", {}),
    }

    accounting = task.get("fit_accounting") or {}
    if (
        int(accounting.get("outer_full_tabpfn_fits", -1)) != 4
        or int(accounting.get("outer_source_drop_tabpfn_fits", -1)) != 12
        or int(accounting.get("final_tabpfn_fits", -1)) != final_expected
        or int(accounting.get("total_tabpfn_fits", -1)) != 16 + final_expected
        or int(accounting.get("maximum_allowed_if_both_candidates_constructed", -1)) != 21
    ):
        raise SystemExit("E11c fit-accounting mismatch")
    stopping = "carry_only_gate_passing_candidates_to_E12" if passing else "close_Task1.1_E11_and_proceed_E12"
    if task.get("stopping_rule_decision") != stopping:
        raise SystemExit("E11c stopping-rule mismatch")
    feasibility = task.get("feasibility") or {}
    if (
        feasibility.get("ready_for_e11c_cpu_execution") is not True
        or int(feasibility.get("expected_outer_full_tabpfn_fits", -1)) != 4
        or int(feasibility.get("expected_outer_source_drop_tabpfn_fits", -1)) != 12
        or int(feasibility.get("expected_max_real_tabpfn_fits_if_both_candidates_constructed", -1)) != 21
        or feasibility.get("fixed_base_weight") != 0.75
        or feasibility.get("fixed_tabpfn_weight") != 0.25
        or feasibility.get("held_outcomes_used_for_feature_audit") is not False
        or feasibility.get("held_outcomes_used_for_prediction_disagreement_audit") is not False
        or feasibility.get("dynamic_gate_fitted") is not False
    ):
        raise SystemExit("E11c feasibility contract mismatch")

    return {
        "folds": aggregate_folds,
        "promotion": safe_promotions,
        "e12_eligible_conditions": passing,
        "challenge": safe_challenge,
        "fit_accounting": {
            "outer_full_tabpfn_fits": 4,
            "outer_source_drop_tabpfn_fits": 12,
            "final_tabpfn_fits": final_expected,
            "total_tabpfn_fits": 16 + final_expected,
            "maximum_allowed": 21,
        },
        "stopping_rule_decision": stopping,
        "external_model": {
            "package_version": PACKAGE_VERSION,
            "model_source": MODEL_SOURCE,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    root = args.input_dir.resolve()
    files = sorted(path.name for path in root.iterdir() if path.is_file())
    if files != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit(f"E11c safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED_TEXT):
            raise SystemExit(f"E11c aggregate privacy/secret contract failed:{name}")
    bridge = json.loads(texts["bridge-result.json"])
    metrics = json.loads(texts["metrics.json"])
    synthetic = bool(bridge.get("synthetic", False))
    if synthetic and not args.allow_synthetic:
        raise SystemExit("E11c real sanitizer refused synthetic output")
    if not synthetic and args.allow_synthetic:
        raise SystemExit("E11c synthetic validation expected synthetic output")
    _validate_bridge(root, bridge, synthetic=synthetic)
    aggregate = _validate_synthetic(metrics) if synthetic else _validate_real(metrics)
    marker = "CMI_FLU_E11C_SYNTHETIC_RESULT" if synthetic else "CMI_FLU_E11C_RESULT"
    print(
        f"{marker} PASS aggregate_only=true submission=false public_probe=false "
        f"result={json.dumps(aggregate, sort_keys=True, separators=(',', ':'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
