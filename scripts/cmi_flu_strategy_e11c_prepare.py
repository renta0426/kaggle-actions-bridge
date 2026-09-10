#!/usr/bin/env python3
"""Build the exact E11c runtime on the proven E11b-005 transport ancestry."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REQUEST_ID = "20260910-cmi-flu-strategy-e11c-robust-tabpfn-001"
TARGET_KERNEL = "renta0426/cmi-flu-e11c-robust-tabpfn-20260910-001"
SCIENCE_COMMIT = "0815f4517345e127dbcbcae0f380a54f3a3d15bd"
E11C_BLOB = "acf876996b5849f7ba794999d45ecb862d1be507"
E11C_SYNTH_BLOB = "9b94633192b5254a7d1debe8db10606b48104076"
E11B_BLOB = "8a291ae4952786bf16ee667caa5557599310e3fd"
E11B_SYNTH_BLOB = "f25d04d75db584db78d26a232ab84dc5ab1bbf7d"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
E11C_CONFIG_BLOB = "3b475f777307238bfcad61c124ab99175207bac2"

OLD_REQUEST = "20260910-cmi-flu-strategy-e11b-tabpfn3-005"
OLD_TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-005"
OLD_SCIENCE = "95c692bf2e27f6d07ea31a45fd788293666ea94a"
PRIOR = Path(__file__).with_name("cmi_flu_strategy_e11b_prepare_v6.py")
PRIOR_BLOB = "6fa25ec0a751f22027bc118bfb0fd584bcb29500"
REQUEST_PATH = "requests/cmi-flu-strategy-e11c-robust-tabpfn-001.json"
MAX_RUNTIME_BYTES = 900_000

SCIENCE_PATHS = {
    "src/cmi_flu/strategy_e11c.py": E11C_BLOB,
    "src/cmi_flu/strategy_e11c_synthetic.py": E11C_SYNTH_BLOB,
    "src/cmi_flu/strategy_e11b.py": E11B_BLOB,
    "src/cmi_flu/strategy_e11b_synthetic.py": E11B_SYNTH_BLOB,
    "configs/baseline_b021_robust.yaml": CONFIG_BLOB,
    "configs/strategy_e11c_task11_robust_tabpfn.json": E11C_CONFIG_BLOB,
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--science-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--tabpfn-wheel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def require_science(root: Path, path: str, expected_blob: str, *, python: bool) -> str:
    data = (root / path).read_bytes()
    found = git_blob_sha(data)
    if found != expected_blob:
        raise SystemExit(f"E11c science relay mismatch:{path}:{found}")
    text = data.decode("utf-8")
    if python:
        compile(text, path, "exec")
        lowered = text.casefold()
        if "kaggle competitions submit" in lowered or "competition_submit" in lowered:
            raise SystemExit(f"E11c science relay contains submission path:{path}")
    return text


def validate_request(root: Path) -> None:
    request = json.loads((root / REQUEST_PATH).read_text())
    exact = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "parent_request_id": OLD_REQUEST,
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_repository": "renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": "exact_commit_archive_plus_git_blob_verification_and_runtime_embedding",
        "expected_kernel_version": 1,
        "enable_internet": True,
        "automatic_compute_retries": 0,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "publish_only_sanitized_aggregate": True,
    }
    for key, value in exact.items():
        if request.get(key) != value:
            raise SystemExit(f"E11c request mismatch:{key}")
    if request.get("science_blobs") != SCIENCE_PATHS:
        raise SystemExit("E11c science blob map mismatch")
    resource = request.get("resource") or {}
    if resource != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 30,
        "hard_timeout_minutes": 60,
        "max_active_runs": 1,
        "max_real_tabpfn_fits": 21,
    }:
        raise SystemExit("E11c resource contract mismatch")
    if request.get("api_budget") != {
        "max_calls": 80,
        "poll_interval_seconds": 120,
        "max_polls": 35,
        "max_pages": 2,
    }:
        raise SystemExit("E11c API budget mismatch")
    preflight = request.get("model_access_preflight") or {}
    if preflight != {
        "required_before_kernel_write": True,
        "official_kaggle_model_source": "prior-labsai/tabpfn-3/pytorch/default/1",
        "failure_classification": "competition_input_model_access_staging",
        "write_on_failure": False,
        "cli_output_format": "--format json",
    }:
        raise SystemExit("E11c model-access preflight mismatch")
    repair = request.get("source_attachment_validator_repair") or {}
    if repair.get("consumed_e11b_target") != OLD_TARGET or repair.get("consumed_e11b_version") != 1:
        raise SystemExit("E11c consumed E11b provenance mismatch")
    if repair.get("rerun_e11b_005") is not False:
        raise SystemExit("E11c must not rerun E11b-005")
    if repair.get("successful_audit_002_actions_run") != 34460007561 or repair.get("successful_audit_002_actions_job") != 102815418161:
        raise SystemExit("E11c metadata audit provenance mismatch")
    if repair.get("observed_model_source") != "prior-labsai/tabpfn-3/PyTorch/default/1":
        raise SystemExit("E11c observed model source mismatch")
    if repair.get("requested_model_source") != "prior-labsai/tabpfn-3/pytorch/default/1":
        raise SystemExit("E11c requested model source mismatch")
    if repair.get("observed_competition_source") != "cmi-flu-first-prediction-challenge":
        raise SystemExit("E11c observed competition source mismatch")
    if repair.get("repair_scope") != "casefold_framework_segment_only_owner_model_variant_version_and_competition_strict":
        raise SystemExit("E11c source-validator repair scope mismatch")
    for key in ("audit_write", "audit_compute", "audit_submission", "audit_output_download"):
        if repair.get(key) is not False:
            raise SystemExit(f"E11c audit side effect mismatch:{key}")
    external = request.get("external_model") or {}
    expected_external = {
        "package": "tabpfn",
        "package_version": "8.5.0",
        "source_commit": "9ed44abd5882140b88c9f2816c5791987ce059b9",
        "wheel_filename": "tabpfn-8.5.0-py3-none-any.whl",
        "wheel_bytes": 771167,
        "wheel_sha256": "4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0",
        "kaggle_model_source": "prior-labsai/tabpfn-3/pytorch/default/1",
        "checkpoint_filename": "tabpfn-v3-regressor-v3_default.ckpt",
        "checkpoint_bytes": 233289807,
        "checkpoint_sha256": "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301",
        "model_license": "tabpfn-3-license-v1.0",
        "project_redistributes_weights": False,
    }
    if external != expected_external:
        raise SystemExit("E11c external-model identity mismatch")
    experiment = request.get("experiment_contract") or {}
    required = {
        "experiment": "strategy_v2_e11c_task11_robust_tabpfn",
        "task": "Task1.1",
        "comparison_contract": "paired_subject_purged_v2",
        "base_model_set": "task_11",
        "base_model": "pls_2",
        "target_transform": "log",
        "expected_train_rows": 127,
        "expected_train_studies": 4,
        "expected_challenge_rows": 40,
        "max_raw_features": 200,
        "base_weight": 0.75,
        "tabpfn_weight": 0.25,
        "eligible_conditions": ["C2", "C4"],
        "c3_eligible_for_e12": False,
        "source_drop_selection_by_held_score": False,
        "dynamic_domain_gate_fitted": False,
        "promotion_mean_delta": 0.02,
        "promotion_minimum_study_delta": -0.1,
        "promotion_requires_strict_majority_wins": True,
        "promotion_expected_required_wins": 3,
        "study_equal_primary": True,
        "row_weighted_diagnostic_only": True,
        "outer_full_tabpfn_fits": 4,
        "outer_source_drop_tabpfn_fits": 12,
        "max_real_tabpfn_fits_if_final_constructed": 21,
        "held_outcomes_used_for_fit": False,
        "held_outcomes_used_for_fusion": False,
        "held_outcomes_used_for_transfer_audit": False,
        "public_probe_authorized": False,
        "competition_submission_authorized": False,
        "automatic_incumbent_change_authorized": False,
        "if_both_pass": "carry_both_to_E12_no_E11c_winner_selection",
        "if_both_fail": "close_Task1.1_E11_and_proceed_E12_no_more_TabPFN_HPO",
    }
    for key, value in required.items():
        if experiment.get(key) != value:
            raise SystemExit(f"E11c experiment contract mismatch:{key}")
    if experiment.get("conditions") != {
        "C0": "b21_pls2_reference",
        "C1": "tabpfn_full_reproduction_control",
        "C2": "pls75_tabpfn25_full_rank_fusion",
        "C3": "tabpfn_source_study_jackknife_rank_median_all_three_source_drops",
        "C4": "pls75_tabpfn25_jackknife_rank_fusion",
    }:
        raise SystemExit("E11c condition map mismatch")
    if experiment.get("tabpfn_params") != {
        "n_estimators": 8,
        "device": "cpu",
        "fit_mode": "fit_preprocessors",
        "memory_saving_mode": "auto",
        "random_state": 20260910,
        "n_preprocessing_jobs": 1,
        "show_progress_bar": False,
        "ignore_pretraining_limits": False,
    }:
        raise SystemExit("E11c TabPFN parameter mismatch")
    if experiment.get("sensitivity_28") != {
        "studies": ["SDY180", "SDY56"],
        "subset_n": 28,
        "repetitions": 200,
        "seed": 20260907,
        "used_for_promotion": False,
        "model_refits": 0,
    }:
        raise SystemExit("E11c sensitivity contract mismatch")
    if request.get("allowed_output_paths") != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E11c output allowlist mismatch")


def replace_function(text: str, name: str, replacement: str) -> str:
    tree = ast.parse(text)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(nodes) != 1:
        raise SystemExit(f"E11c runtime function binding changed:{name}:{len(nodes)}")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    return "".join(lines[: node.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno :])


def e11c_validator_source() -> str:
    return r'''def validate_result(result: dict, *, synthetic: bool = False) -> None:
    import math as _math
    def finite(value, label):
        number=float(value)
        if not _math.isfinite(number): raise BridgeContractError(f"e11c_nonfinite:{label}")
        return number
    def scan(value):
        if isinstance(value,dict):
            for key,item in value.items():
                lowered=str(key).casefold()
                if lowered in {"participant_id","subject_group","row_index","oof_predictions","challenge_predictions"}: raise BridgeContractError("e11c_row_level_key")
                scan(item)
        elif isinstance(value,(list,tuple)):
            for item in value: scan(item)
    scan(result)
    if synthetic:
        if int(result.get("schema_version",-1))!=1 or result.get("experiment")!="synthetic_e11c_robust_tabpfn_contract": raise BridgeContractError("e11c_synthetic_identity")
        if result.get("study_counts")!={"SDY180":34,"SDY515":16,"SDY519":17,"SDY56":60}: raise BridgeContractError("e11c_synthetic_counts")
        if int(result.get("outer_split_count",-1))!=4 or result.get("source_drop_fit_counts")!=[3,3,3,3]: raise BridgeContractError("e11c_synthetic_outer")
        if int(result.get("outer_full_tabpfn_fits",-1))!=4 or int(result.get("outer_source_drop_tabpfn_fits",-1))!=12: raise BridgeContractError("e11c_synthetic_fit_counts")
        if int(result.get("total_tabpfn_fits",99))>21 or int(result.get("maximum_tabpfn_fits",-1))!=21: raise BridgeContractError("e11c_synthetic_fit_budget")
        for key in ("all_drop_membership_safe","all_fusions_fixed","all_audits_outcome_free","all_candidate_scores_finite"):
            if result.get(key) is not True: raise BridgeContractError(f"e11c_synthetic_contract:{key}")
        if result.get("promotion_conditions")!=["C2","C4"] or result.get("e11c_winner_selected") is not False: raise BridgeContractError("e11c_synthetic_selection")
        if result.get("public_probe_authorized") is not False or result.get("competition_submission_authorized") is not False or result.get("dynamic_domain_gate_fitted") is not False: raise BridgeContractError("e11c_synthetic_boundary")
        if result.get("contains_participant_identifiers") is not False or result.get("contains_row_level_predictions") is not False: raise BridgeContractError("e11c_synthetic_privacy")
        return
    if int(result.get("schema_version",-1))!=1 or result.get("experiment")!="strategy_v2_e11c_task11_robust_tabpfn": raise BridgeContractError("e11c_identity")
    if result.get("parent_experiment")!="E11b" or result.get("comparison_contract")!="paired_subject_purged_v2": raise BridgeContractError("e11c_parent_contract")
    for key in ("contains_participant_identifiers","contains_row_level_predictions","held_outcomes_used_for_fit","held_outcomes_used_for_fusion","held_outcomes_used_for_transfer_audit","leaderboard_used_for_selection","public_probe_authorized","competition_submission_attempted","incumbent_changed"):
        if result.get(key) is not False: raise BridgeContractError(f"e11c_boundary:{key}")
    if result.get("aggregate_only_serialization") is not True or int(result.get("automatic_compute_retries",-1))!=0: raise BridgeContractError("e11c_aggregate_retry")
    params={"n_estimators":8,"device":"cpu","fit_mode":"fit_preprocessors","memory_saving_mode":"auto","random_state":20260910,"n_preprocessing_jobs":1,"show_progress_bar":False,"ignore_pretraining_limits":False}
    frozen=result.get("frozen_conditions") or {}
    expected={"task":"Task1.1","base_model":"pls_2","base_model_set":"task_11","target_transform":"log","expected_train_rows":127,"expected_train_studies":4,"expected_challenge_rows":40,"max_raw_features":200,"base_weight":0.75,"tabpfn_weight":0.25,"tabpfn_package_version":"8.5.0","tabpfn_source_commit":"9ed44abd5882140b88c9f2816c5791987ce059b9","tabpfn_wheel_filename":"tabpfn-8.5.0-py3-none-any.whl","tabpfn_wheel_sha256":"4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0","kaggle_model_source":"prior-labsai/tabpfn-3/pytorch/default/1","checkpoint_filename":"tabpfn-v3-regressor-v3_default.ckpt","checkpoint_bytes":233289807,"checkpoint_sha256":"311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301","tabpfn_params":params,"promotion_mean_delta":0.02,"promotion_minimum_study_delta":-0.1,"promotion_requires_strict_majority_wins":True,"sensitivity_28":{"studies":["SDY180","SDY56"],"subset_n":28,"repetitions":200,"seed":20260907,"used_for_promotion":False,"model_refits":0}}
    if frozen!=expected: raise BridgeContractError("e11c_frozen_contract")
    license_info=result.get("external_model_license") or {}
    if license_info.get("model_license")!="tabpfn-3-license-v1.0" or license_info.get("weight_redistribution_by_project") is not False or license_info.get("official_kaggle_model_input") is not True: raise BridgeContractError("e11c_license")
    task=result.get("task") or {}
    if task.get("task")!="Task1.1" or task.get("e11c_winner_selected") is not False: raise BridgeContractError("e11c_task_identity")
    if task.get("dynamic_domain_gate_fitted") is not False or task.get("public_probe_authorized") is not False or task.get("competition_submission_authorized") is not False or task.get("incumbent_changed") is not False: raise BridgeContractError("e11c_task_boundary")
    conditions=task.get("conditions") or {}
    if set(conditions)!={"C0","C1","C2","C3","C4"}: raise BridgeContractError("e11c_conditions")
    if conditions["C2"]!={"eligible_for_e12":True,"base_weight":0.75,"tabpfn_weight":0.25} or conditions["C4"]!={"eligible_for_e12":True,"base_weight":0.75,"tabpfn_weight":0.25}: raise BridgeContractError("e11c_fusion_weights")
    if conditions["C0"].get("eligible_for_e12") is not False or conditions["C1"].get("eligible_for_e12") is not False or conditions["C3"].get("eligible_for_e12") is not False: raise BridgeContractError("e11c_eligibility")
    folds=task.get("folds") or []
    if len(folds)!=4 or {str(f.get("held_study")) for f in folds}!={"SDY180","SDY515","SDY519","SDY56"}: raise BridgeContractError("e11c_folds")
    deltas={"C2":[],"C4":[]}
    for fold in folds:
        held=str(fold.get("held_study")); n=int(fold.get("n",0)); cond=fold.get("conditions") or {}
        if set(cond)!={"C0","C1","C2","C3","C4"} or n<3: raise BridgeContractError("e11c_fold_conditions")
        c0=finite(cond["C0"].get("spearman"),f"{held}:C0")
        for name in ("C1","C2","C3","C4"):
            score=finite(cond[name].get("spearman"),f"{held}:{name}")
            delta=finite(cond[name].get("delta_vs_c0"),f"{held}:{name}:delta")
            if abs((score-c0)-delta)>1e-10: raise BridgeContractError("e11c_delta_arithmetic")
        for name in ("C2","C4"):
            item=cond[name]
            if item.get("eligible_for_e12") is not True or item.get("raw_multiset_preserved") is not True: raise BridgeContractError("e11c_candidate_contract")
            finite(item.get("base_marginal_rank_remap_rmse"),f"{held}:{name}:rmse")
            finite(item.get("c0_raw_rmse"),f"{held}:{name}:c0_rmse")
            deltas[name].append(float(item["delta_vs_c0"]))
            sensitivity=item.get("sensitivity_28") or {}
            if sensitivity.get("used_for_promotion") is not False or int(sensitivity.get("model_refits",-1))!=0: raise BridgeContractError("e11c_sensitivity_boundary")
            if held in {"SDY180","SDY56"}:
                if sensitivity.get("status")!="ok" or int(sensitivity.get("subset_n",-1))!=28 or int(sensitivity.get("repetitions",-1))!=200 or int(sensitivity.get("seed",-1))!=20260907: raise BridgeContractError("e11c_sensitivity_predeclared")
            elif sensitivity.get("status")!="not_run_nonpredeclared_study" or int(sensitivity.get("repetitions",-1))!=0:
                raise BridgeContractError("e11c_sensitivity_nonpredeclared")
        drops=fold.get("source_drop_fits") or []
        if len(drops)!=3 or int(cond["C3"].get("source_drop_fit_count",-1))!=3: raise BridgeContractError("e11c_source_drop_count")
        dropped=set()
        for audit in drops:
            name=str(audit.get("dropped_source_study")); training=[str(v) for v in (audit.get("training_studies") or [])]
            if audit.get("backend")!="tabpfn" or int(audit.get("training_study_count",-1))!=2 or len(training)!=2 or name in training or held in training: raise BridgeContractError("e11c_source_drop_membership")
            if audit.get("held_outcomes_used_for_fit") is not False: raise BridgeContractError("e11c_source_drop_leakage")
            finite(audit.get("fit_seconds"),"source_drop_fit_seconds"); dropped.add(name)
        if len(dropped)!=3: raise BridgeContractError("e11c_source_drop_uniqueness")
        feature=fold.get("feature_shift_audit") or {}; disagreement=fold.get("prediction_disagreement_audit") or {}
        if feature.get("held_outcomes_used") is not False or feature.get("used_for_candidate_routing") is not False: raise BridgeContractError("e11c_feature_audit_boundary")
        if disagreement.get("held_outcomes_used") is not False or int(disagreement.get("source_dropout_pair_count",-1))!=3: raise BridgeContractError("e11c_disagreement_boundary")
        if fold.get("held_outcomes_used_for_fit") is not False or fold.get("held_outcomes_used_for_fusion") is not False or fold.get("held_outcomes_used_for_transfer_audit") is not False or fold.get("held_outcomes_used_for_evaluation_only") is not True: raise BridgeContractError("e11c_fold_boundary")
    passing=[]
    promotions=task.get("promotion") or {}
    for name in ("C2","C4"):
        values=deltas[name]; wins=sum(value>0 for value in values); mean=sum(values)/len(values); minimum=min(values); passed=bool(mean>=0.02 and minimum>=-0.1 and wins>=3)
        observed=promotions.get(name) or {}
        if observed.get("condition")!=name or observed.get("passed") is not passed or int(observed.get("usable_held_studies",-1))!=4 or int(observed.get("wins",-1))!=wins or int(observed.get("required_wins",-1))!=3: raise BridgeContractError("e11c_promotion_boolean")
        if abs(finite(observed.get("mean_delta"),"promotion_mean")-mean)>1e-10 or abs(finite(observed.get("minimum_delta"),"promotion_min")-minimum)>1e-10: raise BridgeContractError("e11c_promotion_arithmetic")
        if observed.get("study_equal_primary") is not True or observed.get("row_weighted_diagnostic_only") is not True: raise BridgeContractError("e11c_promotion_weighting")
        if passed: passing.append(name)
    if task.get("e12_eligible_conditions")!=passing: raise BridgeContractError("e11c_passing_set")
    challenge=task.get("challenge") or {}; final_expected=(1 if "C2" in passing else 0)+(4 if "C4" in passing else 0)
    if int(challenge.get("rows",-1))!=40 or challenge.get("constructed_conditions")!=passing or int(challenge.get("final_tabpfn_fit_count",-1))!=final_expected: raise BridgeContractError("e11c_challenge_construction")
    if challenge.get("candidate_construction_performed") is not bool(passing) or challenge.get("public_probe_authorized") is not False or challenge.get("competition_submission_authorized") is not False or challenge.get("challenge_labels_used") is not False: raise BridgeContractError("e11c_challenge_boundary")
    accounting=task.get("fit_accounting") or {}
    if int(accounting.get("outer_full_tabpfn_fits",-1))!=4 or int(accounting.get("outer_source_drop_tabpfn_fits",-1))!=12 or int(accounting.get("final_tabpfn_fits",-1))!=final_expected or int(accounting.get("total_tabpfn_fits",-1))!=16+final_expected or int(accounting.get("maximum_allowed_if_both_candidates_constructed",-1))!=21: raise BridgeContractError("e11c_fit_accounting")
    expected_step="carry_only_gate_passing_candidates_to_E12" if passing else "close_Task1.1_E11_and_proceed_E12"
    if task.get("stopping_rule_decision")!=expected_step: raise BridgeContractError("e11c_stopping_rule")
'''


def main() -> int:
    args = parse_args()
    bridge_root = args.repository_root.resolve()
    science_root = args.science_root.resolve()
    validate_request(bridge_root)
    if git_blob_sha(PRIOR.read_bytes()) != PRIOR_BLOB:
        raise SystemExit("E11c prepare ancestry changed")

    e11c = require_science(science_root, "src/cmi_flu/strategy_e11c.py", E11C_BLOB, python=True)
    e11c_synthetic = require_science(science_root, "src/cmi_flu/strategy_e11c_synthetic.py", E11C_SYNTH_BLOB, python=True)
    require_science(science_root, "src/cmi_flu/strategy_e11b.py", E11B_BLOB, python=True)
    require_science(science_root, "src/cmi_flu/strategy_e11b_synthetic.py", E11B_SYNTH_BLOB, python=True)
    require_science(science_root, "configs/baseline_b021_robust.yaml", CONFIG_BLOB, python=False)
    require_science(science_root, "configs/strategy_e11c_task11_robust_tabpfn.json", E11C_CONFIG_BLOB, python=False)

    with tempfile.TemporaryDirectory(prefix="cmi-flu-e11c-base-") as tmp:
        prior_runtime = Path(tmp) / "runtime.py"
        subprocess.run(
            [sys.executable, str(PRIOR), "--repository-root", str(bridge_root),
             "--reference-dir", str(args.reference_dir.resolve()),
             "--tabpfn-wheel", str(args.tabpfn_wheel.resolve()),
             "--output", str(prior_runtime)],
            check=True,
        )
        runtime = prior_runtime.read_text(encoding="utf-8")

    for old, new in ((OLD_REQUEST, REQUEST_ID), (OLD_TARGET, TARGET_KERNEL), (OLD_SCIENCE, SCIENCE_COMMIT)):
        if old not in runtime:
            raise SystemExit(f"E11c runtime identity anchor missing:{old}")
        runtime = runtime.replace(old, new)
    if OLD_REQUEST in runtime or OLD_TARGET in runtime or OLD_SCIENCE in runtime:
        raise SystemExit("E11c old runtime identity remained")

    marker = "def load_e11b_modules() -> tuple[object, object, object]:\n"
    if runtime.count(marker) != 1:
        raise SystemExit("E11c module injection anchor changed")
    injected = f'''E11C_BLOB = "{E11C_BLOB}"
E11C_SYNTH_BLOB = "{E11C_SYNTH_BLOB}"
E11C_SOURCE = {e11c!r}
E11C_SYNTH_SOURCE = {e11c_synthetic!r}

'''
    runtime = runtime.replace(marker, injected + marker, 1)
    loader = r'''def load_e11b_modules() -> tuple[object, object, object]:
    load_e05_module()
    def install(name, source):
        module=types.ModuleType(name); module.__file__=f"<{name}>"; module.__package__="cmi_flu"; sys.modules[name]=module
        exec(compile(source,name.replace(".","/")+".py","exec"),module.__dict__,module.__dict__)
        return module
    install("cmi_flu.strategy_e11b",E11B_SOURCE)
    install("cmi_flu.strategy_e11b_synthetic",E11B_SYNTH_SOURCE)
    e11c=install("cmi_flu.strategy_e11c",E11C_SOURCE)
    synthetic=install("cmi_flu.strategy_e11c_synthetic",E11C_SYNTH_SOURCE)
    run=getattr(e11c,"run_e11c",None); run_synthetic=getattr(synthetic,"run_synthetic_e11c",None)
    if not callable(run) or not callable(run_synthetic): raise BridgeContractError("e11c_entry_missing")
    return run,run_synthetic,e11c
'''
    runtime = replace_function(runtime, "load_e11b_modules", loader)
    runtime = replace_function(runtime, "validate_result", e11c_validator_source())

    self_test = r'''def self_test() -> int:
    package=package_bytes()
    for source,expected,label in ((E11B_SOURCE,E11B_BLOB,"e11b"),(E11B_SYNTH_SOURCE,E11B_SYNTH_BLOB,"e11b_synthetic"),(E11C_SOURCE,E11C_BLOB,"e11c"),(E11C_SYNTH_SOURCE,E11C_SYNTH_BLOB,"e11c_synthetic")):
        if git_blob_sha(source.encode("utf-8"))!=expected: raise BridgeContractError(f"{label}_blob_mismatch")
        compile(source,f"cmi_flu/{label}.py","exec")
    if TABPFN_WHEEL_B64!="": raise BridgeContractError("e11c_embedded_wheel_present")
    if TABPFN_WHEEL_FILENAME!="tabpfn-8.5.0-py3-none-any.whl" or int(TABPFN_WHEEL_BYTES)!=771167: raise BridgeContractError("e11c_wheel_shape")
    if TABPFN_WHEEL_SHA256!="4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0": raise BridgeContractError("e11c_wheel_sha")
    if TABPFN_MODEL_SOURCE!="prior-labsai/tabpfn-3/pytorch/default/1": raise BridgeContractError("e11c_model_source")
    if "BASE_WEIGHT = 0.75" not in E11C_SOURCE or "TABPFN_WEIGHT = 0.25" not in E11C_SOURCE: raise BridgeContractError("e11c_weight_anchor")
    if "SENSITIVITY_28_REPETITIONS = 200" not in E11C_SOURCE or "maximum_allowed_if_both_candidates_constructed" not in E11C_SOURCE: raise BridgeContractError("e11c_resource_anchor")
    print(f"CMI_FLU_E11C_RUNTIME_SELF_TEST PASS request_id={REQUEST_ID} science_commit={SCIENCE_COMMIT} e11c_blob={E11C_BLOB} e11c_synth_blob={E11C_SYNTH_BLOB} package_bytes={len(package)} wheel_transport=pypi_exact_hash_verified")
    return 0
'''
    runtime = replace_function(runtime, "self_test", self_test)

    real_anchor = '            "strategy_e11b_synthetic_blob_sha": E11B_SYNTH_BLOB,\n'
    if runtime.count(real_anchor) != 1:
        raise SystemExit(f"E11c real provenance anchor changed:{runtime.count(real_anchor)}")
    runtime = runtime.replace(
        real_anchor,
        real_anchor
        + '            "strategy_e11c_blob_sha": E11C_BLOB,\n'
        + '            "strategy_e11c_synthetic_blob_sha": E11C_SYNTH_BLOB,\n',
        1,
    )
    compact_anchor = '"strategy_e11b_synthetic_blob_sha":E11B_SYNTH_BLOB,'
    if runtime.count(compact_anchor) != 1:
        raise SystemExit(f"E11c synthetic provenance anchor changed:{runtime.count(compact_anchor)}")
    runtime = runtime.replace(
        compact_anchor,
        compact_anchor + '"strategy_e11c_blob_sha":E11C_BLOB,"strategy_e11c_synthetic_blob_sha":E11C_SYNTH_BLOB,',
        1,
    )
    runtime = runtime.replace("CMI_FLU_E11B_COMPLETE", "CMI_FLU_E11C_COMPLETE")
    runtime = runtime.replace("# CMI-Flu E11b", "# CMI-Flu E11c")
    runtime = runtime.replace('stage = "load_e11b"', 'stage = "load_e11c"')
    runtime = runtime.replace('stage = "run_e11b"', 'stage = "run_e11c"')
    runtime = runtime.replace('stage = "validate_e11b"', 'stage = "validate_e11c"')

    raw = runtime.encode("utf-8")
    if len(raw) >= MAX_RUNTIME_BYTES:
        raise SystemExit(f"E11c runtime exceeds source budget:{len(raw)}")
    if 'TABPFN_WHEEL_B64 = ""' not in runtime or '"pip","download"' not in runtime:
        raise SystemExit("E11c exact online wheel transport anchors missing")
    if runtime.count("E11C_SOURCE = ") != 1 or runtime.count("E11C_SYNTH_SOURCE = ") != 1:
        raise SystemExit("E11c science source binding changed")
    compile(runtime, "generated_e11c_runtime.py", "exec")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_E11C_PREPARE PASS science_change=e11c_only transport=e11b005_frozen "
        f"request_id={REQUEST_ID} target={TARGET_KERNEL} science_commit={SCIENCE_COMMIT} "
        f"e11c_blob={E11C_BLOB} runtime_bytes={len(raw)} runtime_sha256={hashlib.sha256(raw).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
