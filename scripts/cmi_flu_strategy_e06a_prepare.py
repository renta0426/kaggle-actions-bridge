#!/usr/bin/env python3
"""Build the exact aggregate-only E06a Task2.3 calibration Kaggle runtime."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

REQUEST_ID = "20260908-cmi-flu-strategy-e06a-task23-calibration-001"
TARGET_KERNEL = "renta0426/cmi-flu-e06a-task23-calibration-20260908-001"
SCIENCE_COMMIT = "9fd643e01b4aae4a8f7815006960022947eaa4e3"
E06_BLOB = "dd5b5ed12e7667e527c97a99e86b8821e68e6442"
E06_PATH = "payloads/cmi-flu-strategy-e06a-task23-calibration-001/strategy_e06.py"
E05_V6_PREPARE = "scripts/cmi_flu_strategy_e05_prepare_v6.py"
OLD_REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-007"
OLD_TARGET_KERNEL = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-007"
OLD_SCIENCE_COMMIT = "92fac34e52485f6f73ea1b7aedb983d8e8e0ee27"
REF_SHA256 = {
    "strain_sequences.csv": "63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887",
    "vaccine_strains_per_season.txt": "8f6c7116f37f29df0bb21d6049d82fa28b4e42b2d10ed9394a1ae6f926bd9f35",
}


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


def load_e05_v6(root: Path):
    path = root / E05_V6_PREPARE
    spec = importlib.util.spec_from_file_location("cmi_flu_e05_prepare_v6_for_e06a", path)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load E05 v6 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        "REQUEST_ID": OLD_REQUEST_ID,
        "TARGET_KERNEL": OLD_TARGET_KERNEL,
        "SCIENCE_COMMIT": OLD_SCIENCE_COMMIT,
        "REF_SHA256": REF_SHA256,
    }
    for key, value in expected.items():
        if getattr(module, key, None) != value:
            raise SystemExit(f"E05 v6 builder contract changed:{key}")
    return module


def load_e06_source(root: Path) -> str:
    data = (root / E06_PATH).read_bytes()
    if git_blob_sha(data) != E06_BLOB:
        raise SystemExit("E06a exact science blob mismatch")
    source = data.decode("utf-8")
    compile(source, "cmi_flu/strategy_e06.py", "exec")
    required = (
        'EXPERIMENT = "strategy_v2_e06a_task23_rank_preserving_calibration"',
        'TASK = "Task2.3"',
        "TARGET_DAY = 365",
        'MODEL_NAME = "ridge_exact_a100"',
        '"positive_affine"',
        '"log2_affine"',
        "MIN_EQUAL_STUDY_RMSE_REDUCTION = 0.02",
        "MAX_WORST_STUDY_RMSE_RATIO = 1.05",
        '"inner_subject_purged_study_out_oof_from_outer_training_only"',
        '"competition_submission_attempted": False',
    )
    if any(token not in source for token in required):
        raise SystemExit("E06a frozen contract token missing")
    if "kaggle competitions submit" in source or "competition_submit" in source:
        raise SystemExit("E06a source contains submission path")
    return source


def build_e05_runtime(v6, root: Path, reference_dir: Path) -> tuple[object, str]:
    v5 = v6.load_v5(root)
    e05_v4 = v6.load_exact_source(root)
    v3, base_runtime = v6.build_v5_runtime(v5, root, reference_dir)
    return v3, v6.patch_runtime(v3, base_runtime, e05_v4)


def replace_block(text: str, start: str, end: str, replacement: str) -> str:
    a = text.find(start)
    b = text.find(end, a + len(start))
    if a < 0 or b < 0 or b <= a:
        raise SystemExit(f"E06a runtime block anchors changed:{start!r}->{end!r}")
    return text[:a] + replacement.rstrip() + "\n" + text[b:]


def patch_runtime(v3, runtime: str, e06: str) -> str:
    for old, new in (
        (OLD_REQUEST_ID, REQUEST_ID),
        (OLD_TARGET_KERNEL, TARGET_KERNEL),
        (OLD_SCIENCE_COMMIT, SCIENCE_COMMIT),
    ):
        if runtime.count(old) < 1:
            raise SystemExit(f"E06a identity anchor missing:{old}")
        runtime = runtime.replace(old, new)

    marker = f'E05_V4_BLOB = "{v3.REF_SHA256.get("__never__", "ab93726fe0810299892091b7e0e8f63dcacec4f3")}"\n'
    # The exact E05 v4 blob is stable and retained only as runtime ancestry.
    marker = 'E05_V4_BLOB = "ab93726fe0810299892091b7e0e8f63dcacec4f3"\n'
    if runtime.count(marker) != 1:
        raise SystemExit("E06a source injection anchor changed")
    runtime = runtime.replace(
        marker,
        marker + f'E06_BLOB = "{E06_BLOB}"\n' + f'E06_SOURCE = {e06!r}\n',
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
    run = getattr(e06, "run_strategy_e06a", None)
    if not callable(run):
        raise BridgeContractError("e06a_entry_missing")
    if getattr(hai, "build_sequence_lookup", None) is not getattr(hai_v2, "build_sequence_lookup", None):
        raise BridgeContractError("e06a_sequence_schema_patch_not_installed")
    return run, hai, e06'''
    runtime = v3.replace_top_level_function(runtime, "load_e05_module", load_replacement)

    self_test_replacement = '''def self_test() -> int:
    package = package_bytes()
    if git_blob_sha(E06_SOURCE.encode("utf-8")) != E06_BLOB:
        raise BridgeContractError("e06a_blob_mismatch")
    compile(E06_SOURCE, "cmi_flu/strategy_e06.py", "exec")
    compile(HAI_TRANSFER_SOURCE, "cmi_flu/hai_transfer.py", "exec")
    compile(HAI_TRANSFER_V2_SOURCE, "cmi_flu/hai_transfer_v2.py", "exec")
    print(f"CMI_FLU_E06A_RUNTIME_SELF_TEST PASS request_id={REQUEST_ID} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} e06_blob={E06_BLOB}")
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
    if contract.get("calibration_kinds") != ["positive_affine", "log2_affine"]:
        raise BridgeContractError("e06a_calibration_kind_mismatch")
    if contract.get("calibration_weighting") != "equal_study_total_weight":
        raise BridgeContractError("e06a_calibration_weight_mismatch")
    if contract.get("outer_evaluation") != "subject_purged_leave_one_study_out":
        raise BridgeContractError("e06a_outer_split_mismatch")
    if contract.get("calibrator_training") != "inner_subject_purged_study_out_oof_from_outer_training_only":
        raise BridgeContractError("e06a_nested_calibration_mismatch")
    if contract.get("strict_monotonicity_required") is not True or contract.get("isotonic_used") is not False:
        raise BridgeContractError("e06a_monotonicity_contract_mismatch")
    if contract.get("held_study_outcomes_used_for_calibrator") is not False or contract.get("observed_d28_used_as_d365_feature") is not False:
        raise BridgeContractError("e06a_leakage_contract_mismatch")
    if result.get("leaderboard_used_for_selection") is not False or result.get("competition_submission_attempted") is not False:
        raise BridgeContractError("e06a_selection_submission_contract_mismatch")
    for condition, payload in result["conditions"].items():
        model = payload.get("model") or {}
        if model.get("name") != "ridge_exact_a100" or model.get("family") != "ridge":
            raise BridgeContractError(f"e06a_model_contract_mismatch:{condition}")
        nested = payload.get("historical_nested_study_out") or {}
        if int(nested.get("outer_split_count", 0)) < 3:
            raise BridgeContractError(f"e06a_outer_split_count:{condition}")
        if set((nested.get("calibrated") or {}).keys()) != {"positive_affine", "log2_affine"}:
            raise BridgeContractError(f"e06a_calibrated_set:{condition}")
        if set((payload.get("promotion") or {}).keys()) != {"positive_affine", "log2_affine"}:
            raise BridgeContractError(f"e06a_promotion_set:{condition}")
        challenge = payload.get("challenge_rank_preservation") or {}
        for kind in ("positive_affine", "log2_affine"):
            if (challenge.get(kind) or {}).get("rank_preserved_within_tolerance") is not True:
                raise BridgeContractError(f"e06a_challenge_rank_changed:{condition}:{kind}")
            rank = ((nested.get("calibrated") or {}).get(kind) or {}).get("rank_preservation") or {}
            if rank.get("preserved_within_tolerance") is not True:
                raise BridgeContractError(f"e06a_held_study_rank_changed:{condition}:{kind}")
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
    lines = ["# CMI-Flu strategy E06a Task2.3 D365 calibration", "", "Nested aggregate-only rank-preserving calibration; no Competition submission was attempted.", "", f"- science commit: `{SCIENCE_COMMIT}`", f"- E06 blob: `{E06_BLOB}`", f"- selected: `{result.get('selected_promoted_condition')}`", ""]
    for condition, payload in (result.get("conditions") or {}).items():
        raw = ((payload.get("historical_nested_study_out") or {}).get("raw_metrics") or {}).get("equal_study_summary") or {}
        lines.append(f"## {condition}")
        lines.append(f"- raw equal-study RMSE mean: `{raw.get('rmse_mean')}`")
        for kind, candidate in (((payload.get("historical_nested_study_out") or {}).get("calibrated") or {}).items()):
            diag = candidate.get("rmse_diagnostics") or {}; gate = (payload.get("promotion") or {}).get(kind) or {}
            lines.append(f"- {kind}: mean_rmse_reduction=`{diag.get('equal_study_mean_relative_reduction')}`, median_study_reduction=`{diag.get('median_study_relative_reduction')}`, worst_ratio=`{diag.get('worst_study_rmse_ratio')}`, promotion=`{gate.get('passed')}`")
        lines.append("")
    lines.append("Historical D365 values are incomplete fixed-panel proxies, not complete 12-strain official-target CV.")
    return "\\n".join(lines)'''
    runtime = v3.replace_top_level_function(runtime, "render_summary", summary_replacement)

    runtime = replace_block(
        runtime,
        '        stage = "load_e05"\n',
        '        stage = "write_outputs"\n',
        '''        stage = "load_e06a"
        run_e06a, hai_module, e06_module = load_e06_module()
        stage = "load_locked_references"
        import pandas as pd
        sequence_reference = pd.read_csv(locate_locked_reference("strain_sequences.csv"))
        vaccine_reference = hai_module.load_vaccine_strain_reference(locate_locked_reference("vaccine_strains_per_season.txt"))
        stage = "run_e06a"
        result = json_safe(dict(run_e06a(config, inputs, sequence_reference=sequence_reference, vaccine_reference=vaccine_reference)))
        stage = "validate_e06a"
        validate_result(result)''',
    )

    provenance = '            "strategy_e05_v4_blob_sha": E05_V4_BLOB,\n'
    if runtime.count(provenance) != 1:
        raise SystemExit("E06a bridge provenance anchor changed")
    runtime = runtime.replace(
        provenance,
        provenance + '            "strategy_e06_blob_sha": E06_BLOB,\n',
        1,
    )
    runtime = runtime.replace(
        "CMI-Flu strategy E05 HAI donor x strain. Aggregate outputs only; no submission.",
        "CMI-Flu strategy E06a D365 calibration. Aggregate outputs only; no submission.",
        1,
    )
    runtime = runtime.replace("CMI_FLU_E05_FAILED", "CMI_FLU_E06A_FAILED")
    runtime = runtime.replace("CMI_FLU_E05_COMPLETE", "CMI_FLU_E06A_COMPLETE")

    names = v3.top_level_functions(runtime)
    for name in ("json_safe", "load_e06_module", "validate_result", "render_summary", "locked_reference_bytes", "locate_locked_reference"):
        if names.count(name) != 1:
            raise SystemExit(f"E06a top-level binding contract failed:{name}:{names.count(name)}")
    if "load_e05_module" in names:
        raise SystemExit("E06a retained obsolete E05 loader")
    for prior in (OLD_REQUEST_ID, OLD_TARGET_KERNEL, OLD_SCIENCE_COMMIT):
        if prior in runtime:
            raise SystemExit(f"prior E05 identity remained in E06a runtime:{prior}")
    if "competition_submit" in runtime or "kaggle competitions submit" in runtime:
        raise SystemExit("E06a runtime contains submission path")
    compile(runtime, "generated_e06a.py", "exec")
    return runtime


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    e05_v6 = load_e05_v6(root)
    e06 = load_e06_source(root)
    v3, e05_runtime = build_e05_runtime(e05_v6, root, reference_dir)
    runtime = patch_runtime(v3, e05_runtime, e06)
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E06A_BUILD PASS "
        f"science_commit={SCIENCE_COMMIT} request_id={REQUEST_ID} target={TARGET_KERNEL} "
        f"runtime_sha256={sha256(runtime.encode())} e06_blob={E06_BLOB} "
        "nested_calibration=true submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
