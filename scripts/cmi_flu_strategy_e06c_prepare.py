#!/usr/bin/env python3
"""Build the exact E06c Task2.1/Task2.2 D28 calibration Kaggle runtime."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import cmi_flu_strategy_e06a_prepare_v3 as prior

REQUEST_ID = "20260909-cmi-flu-strategy-e06c-d28-calibration-001"
TARGET_KERNEL = "renta0426/cmi-flu-e06c-d28-calibration-20260909-001"
SCIENCE_COMMIT = "044d28e3c7e9ab23fa531f57479b03e2437bb1cb"
E06_BLOB = "dd5b5ed12e7667e527c97a99e86b8821e68e6442"
E06_V2_BLOB = "270a9616c4ab4726cac72e220f86c071161765dc"
E06C_BLOB = "90c6e3e2791cc2089d8666f4c8672a47953f5e12"
E06C_PATH = "payloads/cmi-flu-strategy-e06c-d28-calibration-001/strategy_e06c.py"
OLD_REQUEST_ID = "20260908-cmi-flu-strategy-e06a-task23-calibration-003"
OLD_TARGET_KERNEL = "renta0426/cmi-flu-e06a-task23-calibration-20260908-003"
OLD_SCIENCE_COMMIT = "c968d00eb209e55e53b88643d99382ad6915c87e"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


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
            raise SystemExit(f"E06a v3 builder contract changed:{key}")


def load_e06c_source(root: Path) -> str:
    data = (root / E06C_PATH).read_bytes()
    if git_blob_sha(data) != E06C_BLOB:
        raise SystemExit("E06c exact science blob mismatch")
    source = data.decode("utf-8")
    compile(source, "cmi_flu/strategy_e06c.py", "exec")
    required = (
        'EXPERIMENT = "strategy_v2_e06c_d28_rank_preserving_calibration"',
        'TARGET_DAY = 28',
        'TASKS = ("Task2.1", "Task2.2")',
        'CONDITIONS = ("b21_reference", "phase_a_target_domain")',
        '"task_panel_geometric_mean_donor"',
        '"fixed_panel_strain_availability_only_before_outcome_access"',
        '"public_leaderboard_used_for_selection": False',
        '"e05_rank_changing_models_included": False',
        '"competition_submission_attempted": False',
    )
    if any(token not in source for token in required):
        raise SystemExit("E06c frozen contract token missing")
    if "kaggle competitions submit" in source or "competition_submit" in source:
        raise SystemExit("E06c source contains submission path")
    return source


def build_prior_runtime(root: Path, reference_dir: Path):
    prior.verify_prior_contract()
    v3, v2_runtime = prior.build_v2_runtime(root, reference_dir)
    return v3, prior.patch_runtime(v3, v2_runtime)


def patch_runtime(v3, runtime: str, e06c_source: str) -> str:
    for old, new in (
        (OLD_REQUEST_ID, REQUEST_ID),
        (OLD_TARGET_KERNEL, TARGET_KERNEL),
        (OLD_SCIENCE_COMMIT, SCIENCE_COMMIT),
    ):
        if runtime.count(old) < 1:
            raise SystemExit(f"E06c identity anchor missing:{old}")
        runtime = runtime.replace(old, new)

    marker = f'E06_V2_BLOB = "{E06_V2_BLOB}"\n'
    if runtime.count(marker) != 1:
        raise SystemExit("E06c source injection anchor changed")
    runtime = runtime.replace(
        marker,
        marker + f'E06C_BLOB = "{E06C_BLOB}"\n' + f'E06C_SOURCE = {e06c_source!r}\n',
        1,
    )

    load_replacement = '''def load_e06_module() -> object:
    hai = types.ModuleType("cmi_flu.hai_transfer")
    hai.__file__ = "<cmi_flu.hai_transfer>"; hai.__package__ = "cmi_flu"; sys.modules["cmi_flu.hai_transfer"] = hai
    exec(compile(HAI_TRANSFER_SOURCE, "cmi_flu/hai_transfer.py", "exec"), hai.__dict__, hai.__dict__)
    hai_v2 = types.ModuleType("cmi_flu.hai_transfer_v2")
    hai_v2.__file__ = "<cmi_flu.hai_transfer_v2>"; hai_v2.__package__ = "cmi_flu"; sys.modules["cmi_flu.hai_transfer_v2"] = hai_v2
    exec(compile(HAI_TRANSFER_V2_SOURCE, "cmi_flu/hai_transfer_v2.py", "exec"), hai_v2.__dict__, hai_v2.__dict__)
    e06 = types.ModuleType("cmi_flu.strategy_e06")
    e06.__file__ = "<cmi_flu.strategy_e06>"; e06.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e06"] = e06
    exec(compile(E06_SOURCE, "cmi_flu/strategy_e06.py", "exec"), e06.__dict__, e06.__dict__)
    e06v2 = types.ModuleType("cmi_flu.strategy_e06_v2")
    e06v2.__file__ = "<cmi_flu.strategy_e06_v2>"; e06v2.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e06_v2"] = e06v2
    exec(compile(E06_V2_SOURCE, "cmi_flu/strategy_e06_v2.py", "exec"), e06v2.__dict__, e06v2.__dict__)
    e06c = types.ModuleType("cmi_flu.strategy_e06c")
    e06c.__file__ = "<cmi_flu.strategy_e06c>"; e06c.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e06c"] = e06c
    exec(compile(E06C_SOURCE, "cmi_flu/strategy_e06c.py", "exec"), e06c.__dict__, e06c.__dict__)
    run = getattr(e06c, "run_strategy_e06c", None)
    if not callable(run):
        raise BridgeContractError("e06c_entry_missing")
    if getattr(hai, "build_sequence_lookup", None) is not getattr(hai_v2, "build_sequence_lookup", None):
        raise BridgeContractError("e06c_sequence_schema_patch_not_installed")
    if getattr(e06, "_nested_calibration", None) is not getattr(e06v2, "_nested_calibration", None):
        raise BridgeContractError("e06c_panel_overlap_nested_patch_not_installed")
    if getattr(e06, "_fit_calibrator", None) is not getattr(e06v2, "_fit_calibrator", None):
        raise BridgeContractError("e06c_cross_study_calibrator_patch_not_installed")
    return run, hai, e06c'''
    runtime = v3.replace_top_level_function(runtime, "load_e06_module", load_replacement)

    self_test_replacement = '''def self_test() -> int:
    package = package_bytes()
    for source, expected, label in ((E06_SOURCE,E06_BLOB,"e06"),(E06_V2_SOURCE,E06_V2_BLOB,"e06_v2"),(E06C_SOURCE,E06C_BLOB,"e06c"),(HAI_TRANSFER_SOURCE,HAI_TRANSFER_BLOB,"hai_transfer"),(HAI_TRANSFER_V2_SOURCE,HAI_TRANSFER_V2_BLOB,"hai_transfer_v2")):
        if git_blob_sha(source.encode("utf-8")) != expected:
            raise BridgeContractError(f"{label}_blob_mismatch")
    compile(E06C_SOURCE, "cmi_flu/strategy_e06c.py", "exec")
    if "/kaggle/working/.e05-locked-references" in globals().get("__file__", ""):
        raise BridgeContractError("e06c_locked_reference_output_path")
    print(f"CMI_FLU_E06C_RUNTIME_SELF_TEST PASS request_id={REQUEST_ID} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} e06c_blob={E06C_BLOB}")
    return 0'''
    runtime = v3.replace_top_level_function(runtime, "self_test", self_test_replacement)

    validate_replacement = '''def validate_result(result: dict) -> None:
    if result.get("experiment") != "strategy_v2_e06c_d28_rank_preserving_calibration" or int(result.get("target_day", -1)) != 28:
        raise BridgeContractError("e06c_experiment_day_identity_mismatch")
    if result.get("raw_conditions") != ["b21_reference", "phase_a_target_domain"] or result.get("model_name") != "ridge_exact_a100":
        raise BridgeContractError("e06c_raw_condition_model_identity_mismatch")
    tasks = result.get("tasks") or {}
    if set(tasks) != {"Task2.1", "Task2.2"}:
        raise BridgeContractError("e06c_task_set_mismatch")
    thresholds = result.get("promotion_thresholds") or {}
    expected_thresholds = {
        "minimum_equal_study_mean_rmse_reduction": 0.02,
        "minimum_median_study_rmse_reduction_strictly_greater_than": 0.0,
        "maximum_worst_study_rmse_ratio": 1.05,
    }
    for key, value in expected_thresholds.items():
        if float(thresholds.get(key, -999)) != float(value):
            raise BridgeContractError(f"e06c_threshold_mismatch:{key}")
    rank_tolerance = float(thresholds.get("rank_tolerance", float("nan")))
    if not math.isfinite(rank_tolerance) or rank_tolerance > 1e-12:
        raise BridgeContractError("e06c_rank_tolerance_mismatch")
    contract = result.get("calibration_contract") or {}
    expected_contract = {
        "calibration_kinds": ["positive_affine", "log2_affine"],
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
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise BridgeContractError(f"e06c_calibration_contract_mismatch:{key}")
    if result.get("incumbent_changed") is not False or result.get("public_probe_authorized") is not False:
        raise BridgeContractError("e06c_incumbent_probe_contract_mismatch")
    if result.get("leaderboard_used_for_selection") is not False or result.get("competition_submission_attempted") is not False or int(result.get("automatic_compute_retries", -1)) != 0:
        raise BridgeContractError("e06c_selection_submission_retry_contract_mismatch")

    allowed_skip_reasons = {"empty_outer_validation", "zero_fixed_panel_overlap", "fewer_than_two_calibration_panel_studies", "fewer_than_two_inner_study_folds"}
    promotion_checks = {"strictly_positive_outer_slopes", "held_study_rank_preserved", "challenge_rank_preserved", "equal_study_mean_rmse_reduction_at_least_2pct", "median_study_rmse_improves", "no_study_rmse_worse_than_5pct"}
    expected_panels = {"Task2.1": 3, "Task2.2": 12}
    for task, task_payload in tasks.items():
        if int(task_payload.get("panel_size", -1)) != expected_panels[task]:
            raise BridgeContractError(f"e06c_panel_size_mismatch:{task}")
        conditions = task_payload.get("conditions") or {}
        if set(conditions) != {"b21_reference", "phase_a_target_domain"}:
            raise BridgeContractError(f"e06c_condition_set_mismatch:{task}")
        for condition, payload in conditions.items():
            model = payload.get("model") or {}
            if model.get("name") != "ridge_exact_a100" or model.get("family") != "ridge" or float((model.get("params") or {}).get("alpha", -1)) != 100.0:
                raise BridgeContractError(f"e06c_frozen_model_mismatch:{task}:{condition}")
            nested = payload.get("historical_nested_study_out") or {}
            candidate_count = int(nested.get("outer_candidate_split_count", -1)); scored_count = int(nested.get("outer_split_count", -1)); skipped_count = int(nested.get("skipped_outer_split_count", -1)); skipped = nested.get("skipped_outer_folds") or []
            if scored_count < 3 or candidate_count != scored_count + skipped_count or skipped_count != len(skipped):
                raise BridgeContractError(f"e06c_outer_split_accounting:{task}:{condition}")
            if any((item or {}).get("reason") not in allowed_skip_reasons for item in skipped):
                raise BridgeContractError(f"e06c_skip_reason:{task}:{condition}")
            raw = (nested.get("raw_metrics") or {}).get("equal_study_summary") or {}
            if not math.isfinite(float(raw.get("rmse_mean", float("nan")))):
                raise BridgeContractError(f"e06c_raw_rmse_missing:{task}:{condition}")
            calibrated = nested.get("calibrated") or {}; promotion = payload.get("promotion") or {}; challenge = payload.get("challenge_rank_preservation") or {}
            if set(calibrated) != {"positive_affine", "log2_affine"} or set(promotion) != {"positive_affine", "log2_affine"}:
                raise BridgeContractError(f"e06c_candidate_set_mismatch:{task}:{condition}")
            for kind in ("positive_affine", "log2_affine"):
                item = calibrated.get(kind) or {}; diag = item.get("rmse_diagnostics") or {}
                for key in ("equal_study_mean_relative_reduction", "median_study_relative_reduction", "worst_study_rmse_ratio"):
                    if not math.isfinite(float(diag.get(key, float("nan")))):
                        raise BridgeContractError(f"e06c_nonfinite_diagnostic:{task}:{condition}:{kind}:{key}")
                held_rank = (item.get("rank_preservation") or {}).get("preserved_within_tolerance"); challenge_rank = (challenge.get(kind) or {}).get("rank_preserved_within_tolerance")
                if not isinstance(held_rank, bool) or not isinstance(challenge_rank, bool):
                    raise BridgeContractError(f"e06c_rank_boolean_missing:{task}:{condition}:{kind}")
                calibrators = item.get("outer_fold_calibrators") or []
                if len(calibrators) != scored_count or any(int(entry.get("training_studies", 0)) < 2 for entry in calibrators):
                    raise BridgeContractError(f"e06c_calibrator_study_contract:{task}:{condition}:{kind}")
                decision = promotion.get(kind) or {}; checks = decision.get("checks") or {}; passed = decision.get("passed")
                if not isinstance(passed, bool) or set(checks) != promotion_checks or any(not isinstance(value, bool) for value in checks.values()):
                    raise BridgeContractError(f"e06c_promotion_contract:{task}:{condition}:{kind}")
                if checks.get("held_study_rank_preserved") is not held_rank or checks.get("challenge_rank_preserved") is not challenge_rank or passed is not bool(all(checks.values())):
                    raise BridgeContractError(f"e06c_promotion_consistency:{task}:{condition}:{kind}")
                if (not held_rank or not challenge_rank) and passed:
                    raise BridgeContractError(f"e06c_nonpreserving_candidate_promoted:{task}:{condition}:{kind}")
        selected = task_payload.get("selected_promoted_condition")
        allowed = {None, "b21_reference__positive_affine", "b21_reference__log2_affine", "phase_a_target_domain__positive_affine", "phase_a_target_domain__log2_affine"}
        if selected not in allowed:
            raise BridgeContractError(f"e06c_selected_condition_invalid:{task}")
        if selected is not None:
            condition, kind = selected.split("__", 1)
            if ((conditions.get(condition) or {}).get("promotion") or {}).get(kind, {}).get("passed") is not True:
                raise BridgeContractError(f"e06c_selected_candidate_not_promoted:{task}")

    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"','"predictions"')
    if any(token in serialized for token in banned):
        raise BridgeContractError("e06c_aggregate_privacy_contract")'''
    runtime = v3.replace_top_level_function(runtime, "validate_result", validate_replacement)

    summary_replacement = '''def render_summary(result: dict) -> str:
    lines = ["# CMI-Flu strategy E06c D28 Task2.1/Task2.2 calibration", "", "Nested aggregate-only rank-preserving calibration. This experiment cannot improve Public Spearman by construction; it evaluates final-scale/RMSE calibration only. No Competition submission was attempted.", "", f"- science commit: `{SCIENCE_COMMIT}`", f"- E06c blob: `{E06C_BLOB}`", ""]
    for task, task_payload in (result.get("tasks") or {}).items():
        lines.append(f"## {task}")
        lines.append(f"- panel size: `{task_payload.get('panel_size')}`")
        lines.append(f"- selected RMSE-calibration candidate: `{task_payload.get('selected_promoted_condition')}`")
        for condition, payload in (task_payload.get("conditions") or {}).items():
            nested = payload.get("historical_nested_study_out") or {}; raw = (nested.get("raw_metrics") or {}).get("equal_study_summary") or {}; promotion = payload.get("promotion") or {}; challenge = payload.get("challenge_rank_preservation") or {}
            lines.append(f"### {condition}")
            lines.append(f"- outer candidate/scored/skipped: `{nested.get('outer_candidate_split_count')}/{nested.get('outer_split_count')}/{nested.get('skipped_outer_split_count')}`")
            lines.append(f"- raw equal-study RMSE mean: `{raw.get('rmse_mean')}`")
            for kind, candidate in (nested.get("calibrated") or {}).items():
                diag = candidate.get("rmse_diagnostics") or {}; held = (candidate.get("rank_preservation") or {}).get("preserved_within_tolerance"); challenge_rank = (challenge.get(kind) or {}).get("rank_preserved_within_tolerance"); gate = (promotion.get(kind) or {}).get("passed")
                lines.append(f"- {kind}: mean_rmse_reduction=`{diag.get('equal_study_mean_relative_reduction')}`, median_study_reduction=`{diag.get('median_study_relative_reduction')}`, worst_ratio=`{diag.get('worst_study_rmse_ratio')}`, held_rank_preserved=`{held}`, challenge_rank_preserved=`{challenge_rank}`, promotion=`{gate}`")
        lines.append("")
    lines.append("Historical D28 targets are fixed-panel intersection proxies where requested strains are absent. Incumbent predictions remain unchanged by this audit.")
    return "\\n".join(lines)'''
    runtime = v3.replace_top_level_function(runtime, "render_summary", summary_replacement)

    replace_block = prior.prior.prior.replace_block
    runtime = replace_block(
        runtime,
        '        stage = "load_e06a"\n',
        '        stage = "write_outputs"\n',
        '''        stage = "load_e06c"
        run_e06c, hai_module, e06c_module = load_e06_module()
        stage = "load_locked_references"
        import pandas as pd
        sequence_reference = pd.read_csv(locate_locked_reference("strain_sequences.csv"))
        vaccine_reference = hai_module.load_vaccine_strain_reference(locate_locked_reference("vaccine_strains_per_season.txt"))
        stage = "run_e06c"
        result = json_safe(dict(run_e06c(config, inputs, sequence_reference=sequence_reference, vaccine_reference=vaccine_reference)))
        stage = "validate_e06c"
        validate_result(result)''',
    )

    provenance = '            "strategy_e06_v2_blob_sha": E06_V2_BLOB,\n'
    if runtime.count(provenance) != 1:
        raise SystemExit("E06c bridge provenance anchor changed")
    runtime = runtime.replace(provenance, provenance + '            "strategy_e06c_blob_sha": E06C_BLOB,\n', 1)
    runtime = runtime.replace("CMI-Flu strategy E06a D365 calibration. Aggregate outputs only; no submission.", "CMI-Flu strategy E06c D28 calibration. Aggregate outputs only; no submission.")
    runtime = runtime.replace("CMI_FLU_E06A_FAILED", "CMI_FLU_E06C_FAILED")
    runtime = runtime.replace("CMI_FLU_E06A_COMPLETE", "CMI_FLU_E06C_COMPLETE")
    old_completion = "conditions={len((result.get('conditions') or {}))}"
    new_completion = "tasks={len((result.get('tasks') or {}))}"
    if runtime.count(old_completion) != 1:
        raise SystemExit(f"E06c completion marker anchor count={runtime.count(old_completion)}")
    runtime = runtime.replace(old_completion, new_completion, 1)

    names = v3.top_level_functions(runtime)
    for name in ("json_safe", "load_e06_module", "validate_result", "render_summary", "locked_reference_bytes", "locate_locked_reference"):
        if names.count(name) != 1:
            raise SystemExit(f"E06c top-level binding contract failed:{name}:{names.count(name)}")
    for old in (OLD_REQUEST_ID, OLD_TARGET_KERNEL, OLD_SCIENCE_COMMIT):
        if old in runtime:
            raise SystemExit(f"prior E06a identity remained in E06c runtime:{old}")
    if "/kaggle/working/.e05-locked-references" in runtime:
        raise SystemExit("E06c inherited saved-output reference staging")
    if "competition_submit" in runtime or "kaggle competitions submit" in runtime:
        raise SystemExit("E06c runtime contains submission path")
    compile(runtime, "generated_e06c.py", "exec")
    return runtime


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    verify_prior_contract()
    e06c_source = load_e06c_source(root)
    v3, prior_runtime = build_prior_runtime(root, reference_dir)
    runtime = patch_runtime(v3, prior_runtime, e06c_source)
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E06C_BUILD PASS "
        f"science_commit={SCIENCE_COMMIT} request_id={REQUEST_ID} target={TARGET_KERNEL} "
        f"runtime_sha256={sha256(runtime.encode())} e06c_blob={E06C_BLOB} "
        "tasks=Task2.1,Task2.2 day=28 staging=/tmp submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
