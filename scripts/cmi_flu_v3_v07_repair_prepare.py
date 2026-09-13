#!/usr/bin/env python3
"""Build V3-07 H1/H2 repaired -002 runtimes from the frozen -001 runtime."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

import cmi_flu_v3_v07_prepare as base

SCIENCE_COMMIT = "bfaef18b9f703be4d88b8a170794ace2c47a0e55"
REPAIR_BLOB = "f547b894235542981a4ad48d523fe53393c52e34"
REPAIR_PATH = "payloads/cmi-flu-v3-v07-repair-002/strategy_v3_v07_v2.py"
REPAIR_VERSION = "strategy_v3_v07_v2_execution_contract_20260913"
CONTRACTS = {
    "task22_panel_mean": {
        "request_id": "20260913-cmi-flu-strategy-v3-v07-h1-panel-mean-002",
        "target": "renta0426/cmi-flu-v3-07-h1-panel-mean-20260913-002",
        "title": "CMI Flu V3 07 H1 Panel Mean 20260913 002",
        "prefix": "v3_v07_h1",
        "fit_ceiling": 384,
    },
    "task23_retention": {
        "request_id": "20260913-cmi-flu-strategy-v3-v07-h2-retention-002",
        "target": "renta0426/cmi-flu-v3-07-h2-retention-20260913-002",
        "title": "CMI Flu V3 07 H2 Retention 20260913 002",
        "prefix": "v3_v07_h2",
        "fit_ceiling": 256,
    },
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--condition", choices=sorted(CONTRACTS), required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load_repair(root: Path) -> str:
    data = (root / REPAIR_PATH).read_bytes()
    if git_blob_sha(data) != REPAIR_BLOB:
        raise SystemExit("V3-07 repair exact science blob mismatch")
    source = data.decode("utf-8")
    compile(source, "cmi_flu/strategy_v3_v07_v2.py", "exec")
    required = (
        f'REPAIR_VERSION = "{REPAIR_VERSION}"',
        "H1_ENGINEERING_FIT_CEILING = 384",
        "H2_EXPECTED_TEACHER_ROWS = 18593",
        "H2_EXPECTED_BIOLOGICAL_PAIRS = 11913",
        '"fit_limit_policy": "exact_outcome_independent_planned_count"',
        '"fit_limit_policy": "exact_outcome_independent_planned_count_within_frozen_256_bound"',
    )
    if any(token not in source for token in required):
        raise SystemExit("V3-07 repair source contract token missing")
    low = source.casefold()
    forbidden = ("competition_" + "submit(", "kaggle competitions " + "submit", "competitions " + "submit")
    if any(token in low for token in forbidden):
        raise SystemExit("V3-07 repair source contains Competition submission path")
    return source


def patch_runtime(v3, runtime: str, repair_source: str, condition: str) -> str:
    c = CONTRACTS[condition]
    old = base.CONTRACTS[condition]
    replacements = (
        (old["request_id"], c["request_id"]),
        (old["target"], c["target"]),
        (old["title"], c["title"]),
        (base.SCIENCE_COMMIT, SCIENCE_COMMIT),
    )
    for before, after in replacements:
        if before not in runtime:
            raise SystemExit(f"V3-07 repair identity anchor missing:{before}")
        runtime = runtime.replace(before, after)

    anchor = f"V307_OUTPUT_PREFIX = {c['prefix']!r}\n"
    if runtime.count(anchor) != 1:
        raise SystemExit("V3-07 repair injection anchor changed")
    runtime = runtime.replace(
        anchor,
        anchor
        + f'V307_REPAIR_BLOB = "{REPAIR_BLOB}"\n'
        + f'V307_REPAIR_SOURCE = {repair_source!r}\n'
        + f'V307_REPAIR_VERSION = "{REPAIR_VERSION}"\n',
        1,
    )

    wrapper = r'''_load_v307_v1_module = load_v307_module
def load_v307_module() -> tuple[object, object]:
    import sys, types
    v1, hai = _load_v307_v1_module()
    repaired = types.ModuleType("cmi_flu.strategy_v3_v07_v2")
    repaired.__file__ = "<cmi_flu.strategy_v3_v07_v2>"; repaired.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_v3_v07_v2"] = repaired
    exec(compile(V307_REPAIR_SOURCE, "cmi_flu/strategy_v3_v07_v2.py", "exec"), repaired.__dict__, repaired.__dict__)
    if getattr(repaired, "REPAIR_VERSION", None) != V307_REPAIR_VERSION:
        raise BridgeContractError("v307_repair_version_changed")
    if int(getattr(repaired, "H1_ENGINEERING_FIT_CEILING", -1)) != 384:
        raise BridgeContractError("v307_repair_h1_ceiling_changed")
    if int(getattr(repaired, "H2_EXPECTED_TEACHER_ROWS", -1)) != 18593 or int(getattr(repaired, "H2_EXPECTED_BIOLOGICAL_PAIRS", -1)) != 11913:
        raise BridgeContractError("v307_repair_h2_support_changed")
    if not callable(getattr(repaired, "run_v3_07", None)) or not callable(getattr(repaired, "write_v3_07_outputs", None)):
        raise BridgeContractError("v307_repair_entry_missing")
    return repaired, hai
'''
    insert = "def self_test() -> int:\n"
    if runtime.count(insert) != 1:
        raise SystemExit("V3-07 repair loader insertion anchor changed")
    runtime = runtime.replace(insert, wrapper.rstrip() + "\n\n" + insert, 1)

    self_test = r'''def self_test() -> int:
    package = package_bytes()
    if git_blob_sha(V307_SOURCE.encode("utf-8")) != V307_BLOB:
        raise BridgeContractError("v307_blob_mismatch")
    if git_blob_sha(V307_REPAIR_SOURCE.encode("utf-8")) != V307_REPAIR_BLOB:
        raise BridgeContractError("v307_repair_blob_mismatch")
    compile(V307_SOURCE, "cmi_flu/strategy_v3_v07.py", "exec")
    compile(V307_REPAIR_SOURCE, "cmi_flu/strategy_v3_v07_v2.py", "exec")
    if V307_CONDITION not in ("task22_panel_mean", "task23_retention"):
        raise BridgeContractError("v307_condition_invalid")
    low = (V307_SOURCE + "\n" + V307_REPAIR_SOURCE).casefold()
    forbidden = ("competition_" + "submit(", "kaggle competitions " + "submit", "competitions " + "submit")
    if any(token in low for token in forbidden):
        raise BridgeContractError("v307_submission_path_present")
    print(f"CMI_FLU_V307_REPAIR_SELF_TEST PASS request_id={V307_REQUEST_ID} condition={V307_CONDITION} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} v1_blob={V307_BLOB} repair_blob={V307_REPAIR_BLOB} submission=false")
    return 0'''
    runtime = v3.replace_top_level_function(runtime, "self_test", self_test)

    validate_result = r'''def validate_result(result: dict) -> None:
    if result.get("experiment") != "strategy_v3_v07_hai_panel_mean_and_retention":
        raise BridgeContractError("v307_experiment_identity")
    if result.get("new_candidate_conditions") != [V307_CONDITION] or result.get("execution_repair_version") != V307_REPAIR_VERSION:
        raise BridgeContractError("v307_repair_condition_or_version")
    for key in ("competition_submission_attempted", "final_submission_selection_attempted", "public_leaderboard_used", "saved_outer_oof_reused_as_inner_teacher", "automatic_parameter_sweep"):
        if result.get(key) is not False:
            raise BridgeContractError("v307_forbidden_boundary:" + key)
    payload = result.get("result") or {}
    repair = payload.get("execution_repair") or {}
    fit = int(result.get("fit_count", -1)); payload_fit = int(payload.get("fit_count", -2)); planned = int(repair.get("planned_fit_count", -3))
    if payload.get("condition") != V307_CONDITION or fit != payload_fit or fit != planned:
        raise BridgeContractError("v307_repair_fit_plan_mismatch")
    if repair.get("version") != V307_REPAIR_VERSION:
        raise BridgeContractError("v307_repair_payload_version")
    if V307_CONDITION == "task22_panel_mean":
        if not (256 < fit <= 384) or repair.get("fit_limit_policy") != "exact_outcome_independent_planned_count":
            raise BridgeContractError("v307_h1_repaired_fit_budget")
    else:
        if not (0 < fit <= 256) or repair.get("fit_limit_policy") != "exact_outcome_independent_planned_count_within_frozen_256_bound":
            raise BridgeContractError("v307_h2_repaired_fit_budget")
    if payload.get("competition_submission_attempted") is not False or payload.get("public_leaderboard_used") is not False:
        raise BridgeContractError("v307_result_selection_boundary")
    contract = payload.get("contract") or {}; proxy = payload.get("target_proxy") or {}
    metrics = payload.get("metrics") or {}; proxy_metrics = payload.get("target_proxy_metrics") or {}
    if V307_CONDITION == "task22_panel_mean":
        if proxy != {"panel_size":9,"subjects":238,"studies":["2023_UGA","2024UGA"]}:
            raise BridgeContractError("v307_h1_proxy_support")
        expected = {"alpha":10.0,"features":["log2_pre_panel_gm","pre_panel_log2_sd"],"fixed_offset":"nested_inner_OOF_E05_main_effects","outer_subject_purge":True,"inner_oof_only":True,"positivity":True,"panel_strata_not_pooled_for_decision":True}
        if any(contract.get(k) != v for k,v in expected.items()): raise BridgeContractError("v307_h1_model_contract")
        if int(repair.get("learning_rows", -1)) != 62285 or int(repair.get("learning_studies", -1)) != 48:
            raise BridgeContractError("v307_h1_learning_support")
        controls = {"task22_panel_mean", "e05_main_effects", "et_subtype_d5_l10", "pre_hai"}
    else:
        teacher = payload.get("paired_teacher") or {}
        expected_teacher = {"rows":18593,"unique_subjects":567,"target":"log2(Y365/Y28)","participant_year_strain_rows":18593,"participant_years":1067,"studies":11,"biological_subject_strain_pairs":11913,"biological_subjects":567,"row_unit":"participant_year_strain","biological_pair_unit":"subject_strain_deduplicated_across_participant_years"}
        if any(teacher.get(k) != v for k,v in expected_teacher.items()): raise BridgeContractError("v307_h2_teacher_support")
        if proxy != {"panel_size":8,"subjects":439,"studies":3}: raise BridgeContractError("v307_h2_proxy_support")
        expected = {"alpha":100.0,"features":["baseline HAI","age","predicted D28"],"retention_target":"log2(Y365/Y28)","inner_oof_d28_only":True,"held_observed_d28_used":False,"challenge_observed_d28_used":False,"outer_subject_purge":True,"teacher_row_unit":"participant_year_strain","biological_support_unit":"subject_strain_deduplicated_across_participant_years"}
        if any(contract.get(k) != v for k,v in expected.items()): raise BridgeContractError("v307_h2_model_contract")
        controls = {"task23_retention", "independent_d365", "e06b_two_head"}
    for block in (metrics, proxy_metrics):
        if set((block.get("pooled") or {})) != controls or set((block.get("equal_study") or {})) != controls:
            raise BridgeContractError("v307_metric_control_set")
        if not isinstance(block.get("by_study"), list) or not isinstance(block.get("panel_strata"), list):
            raise BridgeContractError("v307_metric_strata_missing")
    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"','"subject_group"','"oof_bank"','"challenge_bank"','"row_index"')
    if any(token in serialized for token in banned): raise BridgeContractError("v307_aggregate_privacy_contract")'''
    runtime = v3.replace_top_level_function(runtime, "validate_result", validate_result)

    old_guard = 'if int(aggregate.get("fit_count", 999999)) > 256: raise BridgeContractError("v307_fit_budget_exceeded")'
    new_guard = 'if V307_CONDITION == "task22_panel_mean" and int(aggregate.get("fit_count", 999999)) > 384: raise BridgeContractError("v307_h1_fit_ceiling_exceeded")\n            if V307_CONDITION == "task23_retention" and int(aggregate.get("fit_count", 999999)) > 256: raise BridgeContractError("v307_h2_fit_ceiling_exceeded")'
    if runtime.count(old_guard) != 1:
        raise SystemExit("V3-07 repair execute fit guard anchor changed")
    runtime = runtime.replace(old_guard, new_guard, 1)
    runtime = runtime.replace('"science_blob": V307_BLOB, "condition": V307_CONDITION,', '"science_blob": V307_BLOB, "science_repair_blob": V307_REPAIR_BLOB, "condition": V307_CONDITION,', 1)

    for old_identity in (old["request_id"], old["target"], old["title"], base.SCIENCE_COMMIT):
        if old_identity in runtime:
            raise SystemExit(f"V3-07 repair retained consumed identity:{old_identity}")
    if c["target"] not in runtime or c["title"] not in runtime or c["request_id"] not in runtime:
        raise SystemExit("V3-07 repair new identity missing")
    if "cmi-flu-v3-v07-" in c["target"]:
        raise SystemExit("V3-07 repair target uses noncanonical v3-v07 slug")
    compile(runtime, f"generated_v307_repair_{condition}.py", "exec")
    return runtime


def build_runtime(root: Path, reference_dir: Path, condition: str) -> str:
    repair_source = load_repair(root)
    v307 = base.load_v307(root)
    v3, parent_runtime = base.load_parent(root, reference_dir)
    runtime = base.patch_runtime(v3, parent_runtime, v307, condition)
    return patch_runtime(v3, runtime, repair_source, condition)


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve(); refs = args.reference_dir.expanduser().resolve()
    runtime = build_runtime(root, refs, args.condition)
    output = args.output.expanduser().resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    c = CONTRACTS[args.condition]
    print(f"CMI_FLU_V307_REPAIR_BUILD PASS condition={args.condition} request_id={c['request_id']} target={c['target']} science_commit={SCIENCE_COMMIT} v1_blob={base.V307_BLOB} repair_blob={REPAIR_BLOB} bytes={len(runtime.encode())} runtime_sha256={sha256(runtime.encode())} submission=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
