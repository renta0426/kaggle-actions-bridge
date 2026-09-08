#!/usr/bin/env python3
"""Build the exact E06b Task2.3 D28/D365 two-head Kaggle runtime."""
from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import cmi_flu_strategy_e06a_prepare_v3 as prior
from cmi_flu_e06a_completion_marker import patch_runtime as patch_completion_marker

REQUEST_ID = "20260908-cmi-flu-strategy-e06b-task23-two-head-001"
TARGET_KERNEL = "renta0426/cmi-flu-e06b-task23-two-head-20260908-001"
SCIENCE_COMMIT = "d172adc2778f3e4fea7692d2a2129db402a994c4"
E06_BLOB = "dd5b5ed12e7667e527c97a99e86b8821e68e6442"
E06_V2_BLOB = "270a9616c4ab4726cac72e220f86c071161765dc"
E06B_BLOB = "5b27abb1bb515e9388e63e7dd4c350aebea90574"
E06B_V2_BLOB = "fe05fd9497f5a80b56b96fbf5f2971a9d43f4e10"
E06B_PATH = "payloads/cmi-flu-strategy-e06b-task23-two-head-001/strategy_e06b.py"
E06B_V2_PATH = "payloads/cmi-flu-strategy-e06b-task23-two-head-001/strategy_e06b_v2.py"
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


def load_exact_source(root: Path, path: str, expected: str, label: str) -> str:
    data = (root / path).read_bytes()
    if git_blob_sha(data) != expected:
        raise SystemExit(f"{label} exact blob mismatch")
    text = data.decode("utf-8")
    compile(text, f"cmi_flu/{Path(path).name}", "exec")
    return text


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


def build_prior_runtime(root: Path, reference_dir: Path):
    prior.verify_prior_contract()
    v3, v2_runtime = prior.build_v2_runtime(root, reference_dir)
    return v3, prior.patch_runtime(v3, v2_runtime)


def _has_result_tasks_subscript(text: str) -> bool:
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != "result":
            continue
        value = node.slice
        if isinstance(value, ast.Constant) and value.value == "tasks":
            return True
    return False


def patch_runtime(v3, runtime: str, e06b_source: str, e06b_v2_source: str) -> str:
    for old, new in (
        (OLD_REQUEST_ID, REQUEST_ID),
        (OLD_TARGET_KERNEL, TARGET_KERNEL),
        (OLD_SCIENCE_COMMIT, SCIENCE_COMMIT),
    ):
        if runtime.count(old) < 1:
            raise SystemExit(f"E06b identity anchor missing:{old}")
        runtime = runtime.replace(old, new)

    marker = f'E06_V2_BLOB = "{E06_V2_BLOB}"\n'
    if runtime.count(marker) != 1:
        raise SystemExit("E06b source injection anchor changed")
    runtime = runtime.replace(
        marker,
        marker
        + f'E06B_BLOB = "{E06B_BLOB}"\n'
        + f'E06B_V2_BLOB = "{E06B_V2_BLOB}"\n'
        + f'E06B_SOURCE = {e06b_source!r}\n'
        + f'E06B_V2_SOURCE = {e06b_v2_source!r}\n',
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
    e06b = types.ModuleType("cmi_flu.strategy_e06b")
    e06b.__file__ = "<cmi_flu.strategy_e06b>"; e06b.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e06b"] = e06b
    exec(compile(E06B_SOURCE, "cmi_flu/strategy_e06b.py", "exec"), e06b.__dict__, e06b.__dict__)
    e06bv2 = types.ModuleType("cmi_flu.strategy_e06b_v2")
    e06bv2.__file__ = "<cmi_flu.strategy_e06b_v2>"; e06bv2.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e06b_v2"] = e06bv2
    exec(compile(E06B_V2_SOURCE, "cmi_flu/strategy_e06b_v2.py", "exec"), e06bv2.__dict__, e06bv2.__dict__)
    run = getattr(e06bv2, "run_strategy_e06b", None)
    if not callable(run):
        raise BridgeContractError("e06b_v2_entry_missing")
    if getattr(hai, "build_sequence_lookup", None) is not getattr(hai_v2, "build_sequence_lookup", None):
        raise BridgeContractError("e06b_sequence_schema_patch_not_installed")
    if getattr(e06b, "_fit_two_head_ridge", None) is not getattr(e06bv2, "_fit_two_head_ridge", None):
        raise BridgeContractError("e06b_two_head_fit_patch_not_installed")
    return run, hai, e06bv2'''
    runtime = v3.replace_top_level_function(runtime, "load_e06_module", load_replacement)

    self_test_replacement = '''def self_test() -> int:
    package = package_bytes()
    for source, expected, label in ((E06_SOURCE,E06_BLOB,"e06"),(E06_V2_SOURCE,E06_V2_BLOB,"e06_v2"),(E06B_SOURCE,E06B_BLOB,"e06b"),(E06B_V2_SOURCE,E06B_V2_BLOB,"e06b_v2"),(HAI_TRANSFER_SOURCE,HAI_TRANSFER_BLOB,"hai_transfer"),(HAI_TRANSFER_V2_SOURCE,HAI_TRANSFER_V2_BLOB,"hai_transfer_v2")):
        if git_blob_sha(source.encode("utf-8")) != expected:
            raise BridgeContractError(f"{label}_blob_mismatch")
    compile(E06B_SOURCE, "cmi_flu/strategy_e06b.py", "exec")
    compile(E06B_V2_SOURCE, "cmi_flu/strategy_e06b_v2.py", "exec")
    print(f"CMI_FLU_E06B_RUNTIME_SELF_TEST PASS request_id={REQUEST_ID} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} e06b_blob={E06B_BLOB} e06b_v2_blob={E06B_V2_BLOB}")
    return 0'''
    runtime = v3.replace_top_level_function(runtime, "self_test", self_test_replacement)

    validate_replacement = '''def validate_result(result: dict) -> None:
    if result.get("experiment") != "strategy_v2_e06b_task23_d28_d365_two_head":
        raise BridgeContractError("e06b_experiment_identity_mismatch")
    if result.get("task") != "Task2.3" or int(result.get("primary_day", -1)) != 365 or int(result.get("auxiliary_day", -1)) != 28:
        raise BridgeContractError("e06b_task_day_identity_mismatch")
    conditions = result.get("conditions") or {}
    if set(conditions) != {"b21_reference", "phase_a_target_domain"}:
        raise BridgeContractError("e06b_condition_set_mismatch")
    thresholds = result.get("promotion_thresholds") or {}
    expected_thresholds = {
        "minimum_equal_study_mean_rmse_reduction": 0.02,
        "minimum_median_study_rmse_reduction_strictly_greater_than": 0.0,
        "maximum_worst_study_rmse_ratio": 1.05,
        "minimum_equal_study_spearman_delta": -0.01,
        "large_study_n": 10,
        "maximum_large_study_spearman_decline": -0.10,
    }
    for key, value in expected_thresholds.items():
        if float(thresholds.get(key, -999)) != float(value):
            raise BridgeContractError(f"e06b_threshold_mismatch:{key}")
    contract = result.get("model_contract") or {}
    expected_contract = {
        "family": "ridge",
        "alpha": 100.0,
        "shared_design": "[Z, day_code*Z, day_code]",
        "day_code_d28": -0.5,
        "day_code_d365": 0.5,
        "preprocessing_fit": "outer_training_rows_only_across_D28_and_D365",
        "held_study_and_subjects_purged_from_both_days": True,
        "observed_d28_used_as_d365_feature": False,
        "d28_role": "auxiliary_training_label_only",
        "d365_role": "primary_Task2.3_target",
        "two_head_fit_adapter": "strategy_e06b_v2_ci_typo_fix",
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise BridgeContractError(f"e06b_model_contract_mismatch:{key}")
    if result.get("leaderboard_used_for_selection") is not False or result.get("competition_submission_attempted") is not False:
        raise BridgeContractError("e06b_selection_submission_contract_mismatch")
    expected_checks = {
        "equal_study_mean_rmse_reduction_at_least_2pct",
        "median_study_rmse_improves",
        "no_study_rmse_worse_than_5pct",
        "equal_study_spearman_not_materially_worse",
        "no_large_study_spearman_decline_below_minus_0.10",
    }
    allowed_skip_reasons = {"zero_fixed_panel_overlap"}
    for condition, payload in conditions.items():
        model = payload.get("model") or {}
        if model.get("name") != "ridge_exact_a100" or model.get("family") != "ridge" or float((model.get("params") or {}).get("alpha", -1)) != 100.0:
            raise BridgeContractError(f"e06b_frozen_model_mismatch:{condition}")
        historical = payload.get("historical_study_out") or {}
        candidate_count = int(historical.get("outer_candidate_split_count", -1))
        scored_count = int(historical.get("outer_split_count", -1))
        skipped = historical.get("skipped_outer_folds") or []
        skipped_count = int(historical.get("skipped_outer_split_count", -1))
        if scored_count < 3 or candidate_count != scored_count + skipped_count or skipped_count != len(skipped):
            raise BridgeContractError(f"e06b_outer_split_accounting:{condition}")
        if any((item or {}).get("reason") not in allowed_skip_reasons for item in skipped):
            raise BridgeContractError(f"e06b_skip_reason:{condition}")
        for key in ("independent_d365", "shared_two_head"):
            summary = ((historical.get(key) or {}).get("equal_study_summary") or {})
            for metric in ("rmse_mean", "spearman_mean"):
                value = float(summary.get(metric, float("nan")))
                if not math.isfinite(value):
                    raise BridgeContractError(f"e06b_nonfinite_summary:{condition}:{key}:{metric}")
        folds = historical.get("fold_training_contract") or []
        if len(folds) != scored_count:
            raise BridgeContractError(f"e06b_fold_contract_count:{condition}")
        for fold in folds:
            if float(fold.get("alpha", -1)) != 100.0 or fold.get("observed_d28_used_as_d365_feature") is not False or fold.get("compatibility_patch") != "ci_caught_day28_variable_typo_only":
                raise BridgeContractError(f"e06b_fold_training_contract:{condition}")
        paired = payload.get("paired_diagnostics") or {}
        rmse = paired.get("rmse") or {}; spearman = paired.get("spearman") or {}
        for key in ("equal_study_mean_relative_reduction", "median_study_relative_reduction", "worst_study_rmse_ratio"):
            if not math.isfinite(float(rmse.get(key, float("nan")))):
                raise BridgeContractError(f"e06b_rmse_diagnostic:{condition}:{key}")
        if not math.isfinite(float(spearman.get("equal_study_mean_delta", float("nan")))):
            raise BridgeContractError(f"e06b_spearman_diagnostic:{condition}")
        decision = payload.get("promotion") or {}; checks = decision.get("checks") or {}; passed = decision.get("passed")
        if set(checks) != expected_checks or any(not isinstance(v, bool) for v in checks.values()) or not isinstance(passed, bool) or passed is not bool(all(checks.values())):
            raise BridgeContractError(f"e06b_promotion_contract:{condition}")
        challenge = payload.get("challenge_shared_vs_independent_rank") or {}
        if int(challenge.get("donor_count", 0)) != 40:
            raise BridgeContractError(f"e06b_challenge_donor_count:{condition}")
        rho = ((challenge.get("rank_spearman") or {}).get("spearman"))
        if rho is None or not math.isfinite(float(rho)):
            raise BridgeContractError(f"e06b_challenge_rank_metric:{condition}")
        challenge_training = payload.get("challenge_training_contract") or {}
        if challenge_training.get("observed_d28_used_as_d365_feature") is not False or challenge_training.get("compatibility_patch") != "ci_caught_day28_variable_typo_only":
            raise BridgeContractError(f"e06b_challenge_training_contract:{condition}")
    selected = result.get("selected_promoted_condition")
    if selected not in {None, "b21_reference", "phase_a_target_domain"}:
        raise BridgeContractError("e06b_selected_condition_invalid")
    if selected is not None and ((conditions.get(selected) or {}).get("promotion") or {}).get("passed") is not True:
        raise BridgeContractError("e06b_selected_condition_not_promoted")
    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"')
    if any(token in serialized for token in banned):
        raise BridgeContractError("e06b_aggregate_privacy_contract")'''
    runtime = v3.replace_top_level_function(runtime, "validate_result", validate_replacement)

    summary_replacement = '''def render_summary(result: dict) -> str:
    lines = ["# CMI-Flu strategy E06b Task2.3 D28/D365 two-head", "", "Low-capacity partial pooling between D28 and D365; D28 outcomes are auxiliary training labels only. No Competition submission was attempted.", "", f"- science commit: `{SCIENCE_COMMIT}`", f"- E06b blob: `{E06B_BLOB}`", f"- E06b v2 blob: `{E06B_V2_BLOB}`", f"- selected: `{result.get('selected_promoted_condition')}`", ""]
    for condition, payload in (result.get("conditions") or {}).items():
        historical = payload.get("historical_study_out") or {}; independent = (historical.get("independent_d365") or {}).get("equal_study_summary") or {}; shared = (historical.get("shared_two_head") or {}).get("equal_study_summary") or {}; paired = payload.get("paired_diagnostics") or {}; rmse = paired.get("rmse") or {}; spearman = paired.get("spearman") or {}; challenge = payload.get("challenge_shared_vs_independent_rank") or {}
        lines.extend([f"## {condition}", f"- outer candidate/scored/skipped: `{historical.get('outer_candidate_split_count')}/{historical.get('outer_split_count')}/{historical.get('skipped_outer_split_count')}`", f"- independent D365 RMSE/Spearman: `{independent.get('rmse_mean')}` / `{independent.get('spearman_mean')}`", f"- shared two-head RMSE/Spearman: `{shared.get('rmse_mean')}` / `{shared.get('spearman_mean')}`", f"- paired mean RMSE reduction: `{rmse.get('equal_study_mean_relative_reduction')}`", f"- paired mean Spearman delta: `{spearman.get('equal_study_mean_delta')}`", f"- Challenge shared-vs-independent rank Spearman: `{(challenge.get('rank_spearman') or {}).get('spearman')}`", f"- promotion: `{(payload.get('promotion') or {}).get('passed')}`", ""])
    lines.append("Historical D365 values are incomplete fixed-panel proxies, not complete 12-strain official-target CV.")
    return "\\n".join(lines)'''
    runtime = v3.replace_top_level_function(runtime, "render_summary", summary_replacement)

    replacements = {
        '        stage = "load_e06a"\n': '        stage = "load_e06b"\n',
        '        run_e06a, hai_module, e06_module = load_e06_module()\n': '        run_e06b, hai_module, e06_module = load_e06_module()\n',
        '        stage = "run_e06a"\n': '        stage = "run_e06b"\n',
        '        result = json_safe(dict(run_e06a(config, inputs, sequence_reference=sequence_reference, vaccine_reference=vaccine_reference)))\n': '        result = json_safe(dict(run_e06b(config, inputs, sequence_reference=sequence_reference, vaccine_reference=vaccine_reference)))\n',
        '        stage = "validate_e06a"\n': '        stage = "validate_e06b"\n',
    }
    for old, new in replacements.items():
        if runtime.count(old) != 1:
            raise SystemExit(f"E06b execute anchor changed:{old!r}:{runtime.count(old)}")
        runtime = runtime.replace(old, new, 1)

    provenance = '            "strategy_e06_v2_blob_sha": E06_V2_BLOB,\n'
    if runtime.count(provenance) != 1:
        raise SystemExit("E06b bridge provenance anchor changed")
    runtime = runtime.replace(
        provenance,
        provenance
        + '            "strategy_e06b_blob_sha": E06B_BLOB,\n'
        + '            "strategy_e06b_v2_blob_sha": E06B_V2_BLOB,\n',
        1,
    )
    runtime = runtime.replace(
        "CMI-Flu strategy E06a D365 calibration. Aggregate outputs only; no submission.",
        "CMI-Flu strategy E06b D28/D365 two-head. Aggregate outputs only; no submission.",
        1,
    )
    runtime = runtime.replace("CMI_FLU_E06A_FAILED", "CMI_FLU_E06B_FAILED")
    runtime = runtime.replace("CMI_FLU_E06A_COMPLETE", "CMI_FLU_E06B_COMPLETE")

    # E06a 003 exposed an inherited E05-only completion marker after outputs were
    # already written. Apply the reusable schema patch and then assert via AST.
    runtime = patch_completion_marker(runtime)
    if _has_result_tasks_subscript(runtime):
        raise SystemExit("E06b generated runtime retained result['tasks'] subscript")
    completion = "conditions={len((result.get('conditions') or {}))}"
    if runtime.count(completion) != 1:
        raise SystemExit("E06b completion marker contract changed")

    names = v3.top_level_functions(runtime)
    for name in ("json_safe", "load_e06_module", "validate_result", "render_summary", "execute", "locked_reference_bytes", "locate_locked_reference"):
        if names.count(name) != 1:
            raise SystemExit(f"E06b top-level binding contract failed:{name}:{names.count(name)}")
    for prior_identity in (OLD_REQUEST_ID, OLD_TARGET_KERNEL, OLD_SCIENCE_COMMIT):
        if prior_identity in runtime:
            raise SystemExit(f"prior E06a identity remained in E06b runtime:{prior_identity}")
    if "competition_submit" in runtime or "kaggle competitions submit" in runtime:
        raise SystemExit("E06b runtime contains submission path")
    compile(runtime, "generated_e06b.py", "exec")
    return runtime


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    verify_prior_contract()
    e06b = load_exact_source(root, E06B_PATH, E06B_BLOB, "E06b")
    e06b_v2 = load_exact_source(root, E06B_V2_PATH, E06B_V2_BLOB, "E06b v2")
    for token in (
        'EXPERIMENT = "strategy_v2_e06b_task23_d28_d365_two_head"',
        '"observed_d28_used_as_d365_feature": False',
        '"shared_design": "[Z, day_code*Z, day_code]"',
    ):
        if token not in e06b:
            raise SystemExit(f"E06b frozen token missing:{token}")
    if '_base._fit_two_head_ridge = _fit_two_head_ridge' not in e06b_v2:
        raise SystemExit("E06b v2 fit patch token missing")
    v3, prior_runtime = build_prior_runtime(root, reference_dir)
    runtime = patch_runtime(v3, prior_runtime, e06b, e06b_v2)
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E06B_BUILD PASS "
        f"science_commit={SCIENCE_COMMIT} request_id={REQUEST_ID} target={TARGET_KERNEL} "
        f"runtime_sha256={sha256(runtime.encode())} e06b_blob={E06B_BLOB} e06b_v2_blob={E06B_V2_BLOB} "
        "completion_conditions=true d28_feature=false submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
