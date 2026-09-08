#!/usr/bin/env python3
"""Build E06a 003: retain non-rank-preserving candidates as valid non-promotions."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import cmi_flu_strategy_e06a_prepare_v2 as prior

REQUEST_ID = "20260908-cmi-flu-strategy-e06a-task23-calibration-003"
TARGET_KERNEL = "renta0426/cmi-flu-e06a-task23-calibration-20260908-003"
SCIENCE_COMMIT = "c968d00eb209e55e53b88643d99382ad6915c87e"
E06_BLOB = "dd5b5ed12e7667e527c97a99e86b8821e68e6442"
E06_V2_BLOB = "270a9616c4ab4726cac72e220f86c071161765dc"
OLD_REQUEST_ID = "20260908-cmi-flu-strategy-e06a-task23-calibration-002"
OLD_TARGET_KERNEL = "renta0426/cmi-flu-e06a-task23-calibration-20260908-002"
OLD_SCIENCE_COMMIT = "6974d1f3c6f2e9b52bc8f1076e46051c71ea62c3"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_prior_contract() -> None:
    expected = {
        "REQUEST_ID": OLD_REQUEST_ID,
        "TARGET_KERNEL": OLD_TARGET_KERNEL,
        "SCIENCE_COMMIT": OLD_SCIENCE_COMMIT,
        "E06_BLOB": E06_BLOB,
        "E06_V2_BLOB": E06_V2_BLOB,
    }
    for key, value in expected.items():
        if getattr(prior, key, None) != value:
            raise SystemExit(f"E06a v2 builder contract changed:{key}")


def build_v2_runtime(root: Path, reference_dir: Path):
    prior.verify_prior_contract()
    e06_v2 = prior.load_e06_v2_source(root)
    v3, v1_runtime = prior.build_v1_runtime(root, reference_dir)
    return v3, prior.patch_runtime(v3, v1_runtime, e06_v2)


def patch_runtime(v3, runtime: str) -> str:
    for old, new in (
        (OLD_REQUEST_ID, REQUEST_ID),
        (OLD_TARGET_KERNEL, TARGET_KERNEL),
        (OLD_SCIENCE_COMMIT, SCIENCE_COMMIT),
    ):
        if runtime.count(old) < 1:
            raise SystemExit(f"E06a v3 identity anchor missing:{old}")
        runtime = runtime.replace(old, new)

    validate_replacement = '''def validate_result(result: dict) -> None:
    if result.get("experiment") != "strategy_v2_e06a_task23_rank_preserving_calibration":
        raise BridgeContractError("e06a_experiment_identity_mismatch")
    if result.get("task") != "Task2.3" or int(result.get("target_day", -1)) != 365:
        raise BridgeContractError("e06a_task_identity_mismatch")
    conditions = result.get("conditions") or {}
    if set(conditions.keys()) != {"b21_reference", "phase_a_target_domain"}:
        raise BridgeContractError("e06a_condition_set_mismatch")
    thresholds = result.get("promotion_thresholds") or {}
    if float(thresholds.get("minimum_equal_study_mean_rmse_reduction", -1)) != 0.02:
        raise BridgeContractError("e06a_rmse_threshold_mismatch")
    if float(thresholds.get("maximum_worst_study_rmse_ratio", -1)) != 1.05:
        raise BridgeContractError("e06a_worst_study_threshold_mismatch")
    contract = result.get("calibration_contract") or {}
    expected_contract = {
        "calibration_kinds": ["positive_affine", "log2_affine"],
        "calibration_weighting": "equal_study_total_weight",
        "outer_evaluation": "subject_purged_leave_one_study_out",
        "calibrator_training": "inner_subject_purged_study_out_oof_from_outer_training_only",
        "strict_monotonicity_required": True,
        "isotonic_used": False,
        "held_study_outcomes_used_for_calibrator": False,
        "observed_d28_used_as_d365_feature": False,
        "outer_fold_panel_overlap_filter": "fixed_panel_strain_availability_only_before_outcome_access",
        "zero_panel_overlap_studies_scored_as_task23_proxy": False,
        "minimum_calibrator_panel_overlap_studies": 2,
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise BridgeContractError(f"e06a_v3_contract_mismatch:{key}")
    if result.get("leaderboard_used_for_selection") is not False or result.get("competition_submission_attempted") is not False:
        raise BridgeContractError("e06a_selection_submission_contract_mismatch")

    allowed_skip_reasons = {"empty_outer_validation", "zero_fixed_panel_overlap", "fewer_than_two_calibration_panel_studies", "fewer_than_two_inner_study_folds"}
    expected_promotion_checks = {
        "strictly_positive_outer_slopes",
        "held_study_rank_preserved",
        "challenge_rank_preserved",
        "equal_study_mean_rmse_reduction_at_least_2pct",
        "median_study_rmse_improves",
        "no_study_rmse_worse_than_5pct",
    }
    for condition, payload in conditions.items():
        model = payload.get("model") or {}
        if model.get("name") != "ridge_exact_a100" or model.get("family") != "ridge":
            raise BridgeContractError(f"e06a_model_contract_mismatch:{condition}")
        nested = payload.get("historical_nested_study_out") or {}
        candidate_count = int(nested.get("outer_candidate_split_count", -1))
        scored_count = int(nested.get("outer_split_count", -1))
        skipped_count = int(nested.get("skipped_outer_split_count", -1))
        skipped = nested.get("skipped_outer_folds") or []
        if scored_count < 3 or candidate_count < scored_count or candidate_count != scored_count + skipped_count or skipped_count != len(skipped):
            raise BridgeContractError(f"e06a_v3_outer_split_accounting:{condition}")
        if any((item or {}).get("reason") not in allowed_skip_reasons for item in skipped):
            raise BridgeContractError(f"e06a_v3_skip_reason:{condition}")
        for item in skipped:
            if item.get("reason") == "zero_fixed_panel_overlap" and int(item.get("validation_panel_overlap_rows", -1)) != 0:
                raise BridgeContractError(f"e06a_v3_zero_overlap_evidence:{condition}")
        calibrated = nested.get("calibrated") or {}
        promotion = payload.get("promotion") or {}
        challenge = payload.get("challenge_rank_preservation") or {}
        if set(calibrated.keys()) != {"positive_affine", "log2_affine"}:
            raise BridgeContractError(f"e06a_calibrated_set:{condition}")
        if set(promotion.keys()) != {"positive_affine", "log2_affine"}:
            raise BridgeContractError(f"e06a_promotion_set:{condition}")
        for kind in ("positive_affine", "log2_affine"):
            candidate = calibrated.get(kind) or {}
            rank = candidate.get("rank_preservation") or {}
            held_rank = rank.get("preserved_within_tolerance")
            challenge_rank = (challenge.get(kind) or {}).get("rank_preserved_within_tolerance")
            if not isinstance(held_rank, bool) or not isinstance(challenge_rank, bool):
                raise BridgeContractError(f"e06a_v3_rank_boolean_missing:{condition}:{kind}")
            calibrators = candidate.get("outer_fold_calibrators") or []
            if len(calibrators) != scored_count or any(int(item.get("training_studies", 0)) < 2 for item in calibrators):
                raise BridgeContractError(f"e06a_v3_cross_study_calibrator:{condition}:{kind}")
            decision = promotion.get(kind) or {}
            passed = decision.get("passed")
            checks = decision.get("checks") or {}
            if not isinstance(passed, bool) or set(checks.keys()) != expected_promotion_checks or any(not isinstance(value, bool) for value in checks.values()):
                raise BridgeContractError(f"e06a_v3_promotion_contract:{condition}:{kind}")
            if checks.get("held_study_rank_preserved") is not held_rank or checks.get("challenge_rank_preserved") is not challenge_rank:
                raise BridgeContractError(f"e06a_v3_rank_promotion_consistency:{condition}:{kind}")
            if passed is not bool(all(checks.values())):
                raise BridgeContractError(f"e06a_v3_promotion_boolean_consistency:{condition}:{kind}")
            if (not held_rank or not challenge_rank) and passed:
                raise BridgeContractError(f"e06a_v3_nonpreserving_candidate_promoted:{condition}:{kind}")

    selected = result.get("selected_promoted_condition")
    allowed = {None, "b21_reference__positive_affine", "b21_reference__log2_affine", "phase_a_target_domain__positive_affine", "phase_a_target_domain__log2_affine"}
    if selected not in allowed:
        raise BridgeContractError("e06a_selected_condition_invalid")
    if selected is not None:
        condition, kind = selected.split("__", 1)
        if ((conditions.get(condition) or {}).get("promotion") or {}).get(kind, {}).get("passed") is not True:
            raise BridgeContractError("e06a_v3_selected_candidate_not_promoted")

    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"')
    if any(token in serialized for token in banned):
        raise BridgeContractError("e06a_aggregate_privacy_contract")'''
    runtime = v3.replace_top_level_function(runtime, "validate_result", validate_replacement)

    summary_replacement = '''def render_summary(result: dict) -> str:
    lines = ["# CMI-Flu strategy E06a Task2.3 D365 calibration v3", "", "Nested aggregate-only calibration; a non-rank-preserving candidate is a valid non-promotion rather than a runtime failure. No Competition submission was attempted.", "", f"- science commit: `{SCIENCE_COMMIT}`", f"- E06 blob: `{E06_BLOB}`", f"- E06 v2 blob: `{E06_V2_BLOB}`", f"- selected: `{result.get('selected_promoted_condition')}`", ""]
    for condition, payload in (result.get("conditions") or {}).items():
        nested = payload.get("historical_nested_study_out") or {}; raw = (nested.get("raw_metrics") or {}).get("equal_study_summary") or {}; challenge = payload.get("challenge_rank_preservation") or {}; promotion = payload.get("promotion") or {}
        lines.append(f"## {condition}")
        lines.append(f"- outer candidate/scored/skipped: `{nested.get('outer_candidate_split_count')}/{nested.get('outer_split_count')}/{nested.get('skipped_outer_split_count')}`")
        lines.append(f"- raw equal-study RMSE mean: `{raw.get('rmse_mean')}`")
        for kind, candidate in ((nested.get("calibrated") or {}).items()):
            diag = candidate.get("rmse_diagnostics") or {}; held = (candidate.get("rank_preservation") or {}).get("preserved_within_tolerance"); challenge_rank = (challenge.get(kind) or {}).get("rank_preserved_within_tolerance"); gate = (promotion.get(kind) or {}).get("passed")
            lines.append(f"- {kind}: mean_rmse_reduction=`{diag.get('equal_study_mean_relative_reduction')}`, median_study_reduction=`{diag.get('median_study_relative_reduction')}`, worst_ratio=`{diag.get('worst_study_rmse_ratio')}`, held_rank_preserved=`{held}`, challenge_rank_preserved=`{challenge_rank}`, promotion=`{gate}`")
        lines.append("")
    lines.append("Historical D365 values are incomplete fixed-panel proxies; zero-overlap studies are not scored as if a Task2.3 proxy existed.")
    return "\\n".join(lines)'''
    runtime = v3.replace_top_level_function(runtime, "render_summary", summary_replacement)

    names = v3.top_level_functions(runtime)
    for name in ("json_safe", "load_e06_module", "validate_result", "render_summary", "locked_reference_bytes", "locate_locked_reference"):
        if names.count(name) != 1:
            raise SystemExit(f"E06a v3 top-level binding contract failed:{name}:{names.count(name)}")
    for prior_identity in (OLD_REQUEST_ID, OLD_TARGET_KERNEL, OLD_SCIENCE_COMMIT):
        if prior_identity in runtime:
            raise SystemExit(f"prior E06a 002 identity remained in E06a 003 runtime:{prior_identity}")
    if "competition_submit" in runtime or "kaggle competitions submit" in runtime:
        raise SystemExit("E06a v3 runtime contains submission path")
    compile(runtime, "generated_e06a_v3.py", "exec")
    return runtime


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    verify_prior_contract()
    v3, v2_runtime = build_v2_runtime(root, reference_dir)
    runtime = patch_runtime(v3, v2_runtime)
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E06A_V3_BUILD PASS "
        f"science_commit={SCIENCE_COMMIT} request_id={REQUEST_ID} target={TARGET_KERNEL} "
        f"runtime_sha256={sha256(runtime.encode())} e06_blob={E06_BLOB} e06_v2_blob={E06_V2_BLOB} "
        "candidate_rank_failure_is_nonpromotion=true submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
