#!/usr/bin/env python3
"""Build repaired E06a 002 with outcome-independent panel-overlap eligibility."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import cmi_flu_strategy_e06a_prepare as prior

REQUEST_ID = "20260908-cmi-flu-strategy-e06a-task23-calibration-002"
TARGET_KERNEL = "renta0426/cmi-flu-e06a-task23-calibration-20260908-002"
SCIENCE_COMMIT = "6974d1f3c6f2e9b52bc8f1076e46051c71ea62c3"
E06_BLOB = "dd5b5ed12e7667e527c97a99e86b8821e68e6442"
E06_V2_BLOB = "270a9616c4ab4726cac72e220f86c071161765dc"
E06_V2_PATH = "payloads/cmi-flu-strategy-e06a-task23-calibration-002/strategy_e06_v2.py"
OLD_REQUEST_ID = "20260908-cmi-flu-strategy-e06a-task23-calibration-001"
OLD_TARGET_KERNEL = "renta0426/cmi-flu-e06a-task23-calibration-20260908-001"
OLD_SCIENCE_COMMIT = "9fd643e01b4aae4a8f7815006960022947eaa4e3"
REF_SHA256 = dict(prior.REF_SHA256)


def parse_args():
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
        "REF_SHA256": REF_SHA256,
    }
    for key, value in expected.items():
        if getattr(prior, key, None) != value:
            raise SystemExit(f"E06a v1 builder contract changed:{key}")


def load_e06_v2_source(root: Path) -> str:
    data = (root / E06_V2_PATH).read_bytes()
    if git_blob_sha(data) != E06_V2_BLOB:
        raise SystemExit("E06a v2 exact science blob mismatch")
    source = data.decode("utf-8")
    compile(source, "cmi_flu/strategy_e06_v2.py", "exec")
    required = (
        '"zero_fixed_panel_overlap"',
        '"fewer_than_two_calibration_panel_studies"',
        '"fixed_panel_strain_availability_only_before_outcome_access"',
        '"zero_panel_overlap_studies_scored_as_task23_proxy": False',
        '"minimum_calibrator_panel_overlap_studies": 2',
        "_base._nested_calibration = _nested_calibration",
        "_base._fit_calibrator = _fit_calibrator",
    )
    if any(token not in source for token in required):
        raise SystemExit("E06a v2 panel-overlap contract token missing")
    if "kaggle competitions submit" in source or "competition_submit" in source:
        raise SystemExit("E06a v2 source contains submission path")
    return source


def build_v1_runtime(root: Path, reference_dir: Path):
    e05_v6 = prior.load_e05_v6(root)
    e06 = prior.load_e06_source(root)
    v3, e05_runtime = prior.build_e05_runtime(e05_v6, root, reference_dir)
    return v3, prior.patch_runtime(v3, e05_runtime, e06)


def patch_runtime(v3, runtime: str, e06_v2: str) -> str:
    for old, new in (
        (OLD_REQUEST_ID, REQUEST_ID),
        (OLD_TARGET_KERNEL, TARGET_KERNEL),
        (OLD_SCIENCE_COMMIT, SCIENCE_COMMIT),
    ):
        if runtime.count(old) < 1:
            raise SystemExit(f"E06a v2 identity anchor missing:{old}")
        runtime = runtime.replace(old, new)

    marker = f'E06_BLOB = "{E06_BLOB}"\n'
    if runtime.count(marker) != 1:
        raise SystemExit("E06a v2 source injection anchor changed")
    runtime = runtime.replace(
        marker,
        marker
        + f'E06_V2_BLOB = "{E06_V2_BLOB}"\n'
        + f'E06_V2_SOURCE = {e06_v2!r}\n',
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
    run = getattr(e06v2, "run_strategy_e06a", None)
    if not callable(run):
        raise BridgeContractError("e06a_v2_entry_missing")
    if getattr(hai, "build_sequence_lookup", None) is not getattr(hai_v2, "build_sequence_lookup", None):
        raise BridgeContractError("e06a_sequence_schema_patch_not_installed")
    if getattr(e06, "_nested_calibration", None) is not getattr(e06v2, "_nested_calibration", None):
        raise BridgeContractError("e06a_panel_overlap_nested_patch_not_installed")
    if getattr(e06, "_fit_calibrator", None) is not getattr(e06v2, "_fit_calibrator", None):
        raise BridgeContractError("e06a_cross_study_calibrator_patch_not_installed")
    return run, hai, e06v2'''
    runtime = v3.replace_top_level_function(runtime, "load_e06_module", load_replacement)

    self_test_replacement = '''def self_test() -> int:
    package = package_bytes()
    if git_blob_sha(E06_SOURCE.encode("utf-8")) != E06_BLOB:
        raise BridgeContractError("e06a_blob_mismatch")
    if git_blob_sha(E06_V2_SOURCE.encode("utf-8")) != E06_V2_BLOB:
        raise BridgeContractError("e06a_v2_blob_mismatch")
    compile(E06_SOURCE, "cmi_flu/strategy_e06.py", "exec")
    compile(E06_V2_SOURCE, "cmi_flu/strategy_e06_v2.py", "exec")
    compile(HAI_TRANSFER_SOURCE, "cmi_flu/hai_transfer.py", "exec")
    compile(HAI_TRANSFER_V2_SOURCE, "cmi_flu/hai_transfer_v2.py", "exec")
    print(f"CMI_FLU_E06A_V2_RUNTIME_SELF_TEST PASS request_id={REQUEST_ID} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} e06_blob={E06_BLOB} e06_v2_blob={E06_V2_BLOB}")
    return 0'''
    runtime = v3.replace_top_level_function(runtime, "self_test", self_test_replacement)

    validate_replacement = '''def validate_result(result: dict) -> None:
    if result.get("experiment") != "strategy_v2_e06a_task23_rank_preserving_calibration":
        raise BridgeContractError("e06a_experiment_identity_mismatch")
    if result.get("task") != "Task2.3" or int(result.get("target_day", -1)) != 365:
        raise BridgeContractError("e06a_task_identity_mismatch")
    if set((result.get("conditions") or {}).keys()) != {"b21_reference", "phase_a_target_domain"}:
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
            raise BridgeContractError(f"e06a_v2_contract_mismatch:{key}")
    if result.get("leaderboard_used_for_selection") is not False or result.get("competition_submission_attempted") is not False:
        raise BridgeContractError("e06a_selection_submission_contract_mismatch")
    allowed_skip_reasons = {"empty_outer_validation", "zero_fixed_panel_overlap", "fewer_than_two_calibration_panel_studies", "fewer_than_two_inner_study_folds"}
    for condition, payload in result["conditions"].items():
        model = payload.get("model") or {}
        if model.get("name") != "ridge_exact_a100" or model.get("family") != "ridge":
            raise BridgeContractError(f"e06a_model_contract_mismatch:{condition}")
        nested = payload.get("historical_nested_study_out") or {}
        candidate_count = int(nested.get("outer_candidate_split_count", -1))
        scored_count = int(nested.get("outer_split_count", -1))
        skipped_count = int(nested.get("skipped_outer_split_count", -1))
        skipped = nested.get("skipped_outer_folds") or []
        if scored_count < 3 or candidate_count < scored_count or candidate_count != scored_count + skipped_count or skipped_count != len(skipped):
            raise BridgeContractError(f"e06a_v2_outer_split_accounting:{condition}")
        if any((item or {}).get("reason") not in allowed_skip_reasons for item in skipped):
            raise BridgeContractError(f"e06a_v2_skip_reason:{condition}")
        for item in skipped:
            if item.get("reason") == "zero_fixed_panel_overlap" and int(item.get("validation_panel_overlap_rows", -1)) != 0:
                raise BridgeContractError(f"e06a_v2_zero_overlap_evidence:{condition}")
        if set((nested.get("calibrated") or {}).keys()) != {"positive_affine", "log2_affine"}:
            raise BridgeContractError(f"e06a_calibrated_set:{condition}")
        if set((payload.get("promotion") or {}).keys()) != {"positive_affine", "log2_affine"}:
            raise BridgeContractError(f"e06a_promotion_set:{condition}")
        challenge = payload.get("challenge_rank_preservation") or {}
        for kind in ("positive_affine", "log2_affine"):
            if (challenge.get(kind) or {}).get("rank_preserved_within_tolerance") is not True:
                raise BridgeContractError(f"e06a_challenge_rank_changed:{condition}:{kind}")
            candidate = (nested.get("calibrated") or {}).get(kind) or {}
            rank = candidate.get("rank_preservation") or {}
            if rank.get("preserved_within_tolerance") is not True:
                raise BridgeContractError(f"e06a_held_study_rank_changed:{condition}:{kind}")
            calibrators = candidate.get("outer_fold_calibrators") or []
            if len(calibrators) != scored_count or any(int(item.get("training_studies", 0)) < 2 for item in calibrators):
                raise BridgeContractError(f"e06a_v2_cross_study_calibrator:{condition}:{kind}")
    selected = result.get("selected_promoted_condition")
    allowed = {None, "b21_reference__positive_affine", "b21_reference__log2_affine", "phase_a_target_domain__positive_affine", "phase_a_target_domain__log2_affine"}
    if selected not in allowed:
        raise BridgeContractError("e06a_selected_condition_invalid")
    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"')
    if any(token in serialized for token in banned):
        raise BridgeContractError("e06a_aggregate_privacy_contract")'''
    runtime = v3.replace_top_level_function(runtime, "validate_result", validate_replacement)

    summary_replacement = '''def render_summary(result: dict) -> str:
    lines = ["# CMI-Flu strategy E06a Task2.3 D365 calibration v2", "", "Nested aggregate-only rank-preserving calibration with availability-only panel-fold eligibility; no Competition submission was attempted.", "", f"- science commit: `{SCIENCE_COMMIT}`", f"- E06 blob: `{E06_BLOB}`", f"- E06 v2 blob: `{E06_V2_BLOB}`", f"- selected: `{result.get('selected_promoted_condition')}`", ""]
    for condition, payload in (result.get("conditions") or {}).items():
        nested = payload.get("historical_nested_study_out") or {}; raw = (nested.get("raw_metrics") or {}).get("equal_study_summary") or {}
        lines.append(f"## {condition}")
        lines.append(f"- outer candidate/scored/skipped: `{nested.get('outer_candidate_split_count')}/{nested.get('outer_split_count')}/{nested.get('skipped_outer_split_count')}`")
        lines.append(f"- raw equal-study RMSE mean: `{raw.get('rmse_mean')}`")
        for kind, candidate in ((nested.get("calibrated") or {}).items()):
            diag = candidate.get("rmse_diagnostics") or {}; gate = (payload.get("promotion") or {}).get(kind) or {}
            lines.append(f"- {kind}: mean_rmse_reduction=`{diag.get('equal_study_mean_relative_reduction')}`, median_study_reduction=`{diag.get('median_study_relative_reduction')}`, worst_ratio=`{diag.get('worst_study_rmse_ratio')}`, promotion=`{gate.get('passed')}`")
        lines.append("")
    lines.append("Historical D365 values are incomplete fixed-panel proxies; zero-overlap studies are not scored as if a Task2.3 proxy existed.")
    return "\\n".join(lines)'''
    runtime = v3.replace_top_level_function(runtime, "render_summary", summary_replacement)

    provenance = '            "strategy_e06_blob_sha": E06_BLOB,\n'
    if runtime.count(provenance) != 1:
        raise SystemExit("E06a v2 bridge provenance anchor changed")
    runtime = runtime.replace(
        provenance,
        provenance + '            "strategy_e06_v2_blob_sha": E06_V2_BLOB,\n',
        1,
    )

    names = v3.top_level_functions(runtime)
    for name in ("json_safe", "load_e06_module", "validate_result", "render_summary", "locked_reference_bytes", "locate_locked_reference"):
        if names.count(name) != 1:
            raise SystemExit(f"E06a v2 top-level binding contract failed:{name}:{names.count(name)}")
    for old in (OLD_REQUEST_ID, OLD_TARGET_KERNEL, OLD_SCIENCE_COMMIT):
        if old in runtime:
            raise SystemExit(f"prior E06a identity remained in v2 runtime:{old}")
    if "competition_submit" in runtime or "kaggle competitions submit" in runtime:
        raise SystemExit("E06a v2 runtime contains submission path")
    compile(runtime, "generated_e06a_v2.py", "exec")
    return runtime


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    verify_prior_contract()
    e06_v2 = load_e06_v2_source(root)
    v3, prior_runtime = build_v1_runtime(root, reference_dir)
    runtime = patch_runtime(v3, prior_runtime, e06_v2)
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E06A_V2_BUILD PASS "
        f"science_commit={SCIENCE_COMMIT} request_id={REQUEST_ID} target={TARGET_KERNEL} "
        f"runtime_sha256={sha256(runtime.encode())} e06_blob={E06_BLOB} e06_v2_blob={E06_V2_BLOB} "
        "panel_overlap_filter=true nested_calibration=true submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
