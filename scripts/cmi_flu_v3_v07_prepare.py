#!/usr/bin/env python3
"""Build deterministic V3-07 H1/H2 runtimes from the proven E06b ancestry."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

import cmi_flu_strategy_e06b_prepare as parent

SCIENCE_COMMIT = "9bfe05664e4d96f09631a3210f7951bf9dfefe71"
V307_BLOB = "4cf804e0c10889d0e037457ac096711b77200ea9"
V307_PATH = "payloads/cmi-flu-v3-v07-source-001/strategy_v3_v07.py"
CONTRACTS = {
    "task22_panel_mean": {
        "request_id": "20260912-cmi-flu-strategy-v3-v07-h1-panel-mean-001",
        "target": "renta0426/cmi-flu-v3-v07-h1-panel-mean-20260912-001",
        "title": "CMI Flu V3-07 H1 Panel Mean 20260912 001",
        "prefix": "v3_v07_h1",
        "fit_limit": 256,
    },
    "task23_retention": {
        "request_id": "20260912-cmi-flu-strategy-v3-v07-h2-retention-001",
        "target": "renta0426/cmi-flu-v3-v07-h2-retention-20260912-001",
        "title": "CMI Flu V3-07 H2 Retention 20260912 001",
        "prefix": "v3_v07_h2",
        "fit_limit": 256,
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


def submission_markers() -> tuple[str, ...]:
    """Return forbidden submission markers without embedding them contiguously.

    The generated runtime contains its own no-submit guard. Constructing the
    literals here prevents the final static scan from self-matching that guard
    while preserving the exact forbidden strings at runtime.
    """
    return (
        "competition_" + "submit(",
        "kaggle competitions " + "submit",
        "competitions " + "submit",
    )


def load_v307(root: Path) -> str:
    data = (root / V307_PATH).read_bytes()
    if git_blob_sha(data) != V307_BLOB:
        raise SystemExit("V3-07 exact science blob mismatch")
    source = data.decode("utf-8")
    compile(source, "cmi_flu/strategy_v3_v07.py", "exec")
    required = (
        'EXPERIMENT = "strategy_v3_v07_hai_panel_mean_and_retention"',
        'CONDITIONS = ("task22_panel_mean", "task23_retention")',
        "H1_ALPHA = 10.0",
        "H2_ALPHA = 100.0",
        "H1_EXPECTED_PROXY_SUBJECTS = 238",
        "H2_EXPECTED_PROXY_SUBJECTS = 439",
        "H2_EXPECTED_PAIR_ROWS = 11913",
        '"saved_outer_oof_reused_as_inner_teacher":False',
        '"competition_submission_attempted":False',
    )
    if any(token not in source for token in required):
        raise SystemExit("V3-07 frozen source contract token missing")
    low = source.casefold()
    if any(token in low for token in submission_markers()):
        raise SystemExit("V3-07 science source contains Competition submission path")
    return source


def load_parent(root: Path, reference_dir: Path) -> tuple[object, str]:
    parent.verify_prior_contract()
    e06b = parent.load_exact_source(root, parent.E06B_PATH, parent.E06B_BLOB, "E06b")
    e06b_v2 = parent.load_exact_source(root, parent.E06B_V2_PATH, parent.E06B_V2_BLOB, "E06b v2")
    v3, prior_runtime = parent.build_prior_runtime(root, reference_dir)
    return v3, parent.patch_runtime(v3, prior_runtime, e06b, e06b_v2)


def _insert_before_execute(runtime: str, loader: str) -> str:
    anchor = "def execute(input_dir: Path, output_dir: Path) -> int:\n"
    if runtime.count(anchor) != 1:
        raise SystemExit(f"V3-07 execute insertion anchor count={runtime.count(anchor)}")
    return runtime.replace(anchor, loader.rstrip() + "\n\n" + anchor, 1)


def patch_runtime(v3, runtime: str, v307: str, condition: str) -> str:
    contract = CONTRACTS[condition]
    for old, new in (
        (parent.REQUEST_ID, contract["request_id"]),
        (parent.TARGET_KERNEL, contract["target"]),
        (parent.SCIENCE_COMMIT, SCIENCE_COMMIT),
    ):
        if runtime.count(old) < 1:
            raise SystemExit(f"V3-07 parent identity anchor missing:{old}")
        runtime = runtime.replace(old, new)

    marker = f'E06B_V2_BLOB = "{parent.E06B_V2_BLOB}"\n'
    if runtime.count(marker) != 1:
        raise SystemExit("V3-07 source injection anchor changed")
    runtime = runtime.replace(
        marker,
        marker
        + f'V307_BLOB = "{V307_BLOB}"\n'
        + f'V307_SOURCE = {v307!r}\n'
        + f'V307_CONDITION = {condition!r}\n'
        + f'V307_REQUEST_ID = {contract["request_id"]!r}\n'
        + f'V307_TARGET_KERNEL = {contract["target"]!r}\n'
        + f'V307_TITLE = {contract["title"]!r}\n'
        + f'V307_OUTPUT_PREFIX = {contract["prefix"]!r}\n',
        1,
    )

    loader = r'''def load_v307_module() -> tuple[object, object]:
    import sys, types
    _run_e06b, hai, e06bv2 = load_e06_module()
    e01 = types.ModuleType("cmi_flu.strategy_e01")
    e01.__file__ = "<cmi_flu.strategy_e01>"; e01.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e01"] = e01
    exec(compile(E01_SOURCE, "cmi_flu/strategy_e01.py", "exec"), e01.__dict__, e01.__dict__)
    e01v2 = types.ModuleType("cmi_flu.strategy_e01_v2")
    e01v2.__file__ = "<cmi_flu.strategy_e01_v2>"; e01v2.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e01_v2"] = e01v2
    exec(compile(E01_V2_SOURCE, "cmi_flu/strategy_e01_v2.py", "exec"), e01v2.__dict__, e01v2.__dict__)
    e05 = types.ModuleType("cmi_flu.strategy_e05")
    e05.__file__ = "<cmi_flu.strategy_e05>"; e05.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e05"] = e05
    exec(compile(E05_SOURCE, "cmi_flu/strategy_e05.py", "exec"), e05.__dict__, e05.__dict__)
    e05v2 = types.ModuleType("cmi_flu.strategy_e05_v2")
    e05v2.__file__ = "<cmi_flu.strategy_e05_v2>"; e05v2.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e05_v2"] = e05v2
    exec(compile(E05_V2_SOURCE, "cmi_flu/strategy_e05_v2.py", "exec"), e05v2.__dict__, e05v2.__dict__)
    e05v3 = types.ModuleType("cmi_flu.strategy_e05_v3")
    e05v3.__file__ = "<cmi_flu.strategy_e05_v3>"; e05v3.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e05_v3"] = e05v3
    exec(compile(E05_V3_SOURCE, "cmi_flu/strategy_e05_v3.py", "exec"), e05v3.__dict__, e05v3.__dict__)
    e05v4 = types.ModuleType("cmi_flu.strategy_e05_v4")
    e05v4.__file__ = "<cmi_flu.strategy_e05_v4>"; e05v4.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e05_v4"] = e05v4
    exec(compile(E05_V4_SOURCE, "cmi_flu/strategy_e05_v4.py", "exec"), e05v4.__dict__, e05v4.__dict__)
    if getattr(e05, "_subject_level_rank", None) is not getattr(e05v2, "_donor_level_rank", None):
        raise BridgeContractError("v307_e05_donor_rank_patch_not_installed")
    if getattr(e05, "_control_run", None) is not getattr(e05v4, "_control_run_compat", None):
        raise BridgeContractError("v307_e05_frozen_control_patch_not_installed")
    v307 = types.ModuleType("cmi_flu.strategy_v3_v07")
    v307.__file__ = "<cmi_flu.strategy_v3_v07>"; v307.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_v3_v07"] = v307
    exec(compile(V307_SOURCE, "cmi_flu/strategy_v3_v07.py", "exec"), v307.__dict__, v307.__dict__)
    if v307.CONDITIONS != ("task22_panel_mean", "task23_retention"):
        raise BridgeContractError("v307_condition_family_changed")
    if float(v307.H1_ALPHA) != 10.0 or float(v307.H2_ALPHA) != 100.0:
        raise BridgeContractError("v307_alpha_contract_changed")
    if int(v307.MAX_FITS_H1) != 256 or int(v307.MAX_FITS_H2) != 256:
        raise BridgeContractError("v307_fit_limit_changed")
    if not callable(getattr(v307, "run_v3_07", None)) or not callable(getattr(v307, "write_v3_07_outputs", None)):
        raise BridgeContractError("v307_entry_missing")
    return v307, hai
'''
    runtime = _insert_before_execute(runtime, loader)

    self_test = r'''def self_test() -> int:
    package = package_bytes()
    if git_blob_sha(V307_SOURCE.encode("utf-8")) != V307_BLOB:
        raise BridgeContractError("v307_blob_mismatch")
    compile(V307_SOURCE, "cmi_flu/strategy_v3_v07.py", "exec")
    if V307_CONDITION not in ("task22_panel_mean", "task23_retention"):
        raise BridgeContractError("v307_condition_invalid")
    low = V307_SOURCE.casefold()
    forbidden = (
        "competition_" + "submit(",
        "kaggle competitions " + "submit",
        "competitions " + "submit",
    )
    if any(token in low for token in forbidden):
        raise BridgeContractError("v307_submission_path_present")
    print(f"CMI_FLU_V307_SELF_TEST PASS request_id={V307_REQUEST_ID} condition={V307_CONDITION} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} science_blob={V307_BLOB} submission=false")
    return 0'''
    runtime = v3.replace_top_level_function(runtime, "self_test", self_test)

    validate_result = r'''def validate_result(result: dict) -> None:
    if result.get("experiment") != "strategy_v3_v07_hai_panel_mean_and_retention":
        raise BridgeContractError("v307_experiment_identity")
    if result.get("new_candidate_conditions") != [V307_CONDITION]:
        raise BridgeContractError("v307_condition_set")
    if int(result.get("fit_count", 999999)) > 256:
        raise BridgeContractError("v307_fit_budget")
    for key in ("competition_submission_attempted", "final_submission_selection_attempted", "public_leaderboard_used", "saved_outer_oof_reused_as_inner_teacher", "automatic_parameter_sweep"):
        if result.get(key) is not False:
            raise BridgeContractError("v307_forbidden_boundary:" + key)
    payload = result.get("result") or {}
    if payload.get("condition") != V307_CONDITION or int(payload.get("fit_count", 999999)) > 256:
        raise BridgeContractError("v307_result_condition_or_fit")
    if payload.get("competition_submission_attempted") is not False or payload.get("public_leaderboard_used") is not False:
        raise BridgeContractError("v307_result_selection_boundary")
    contract = payload.get("contract") or {}
    proxy = payload.get("target_proxy") or {}
    metrics = payload.get("metrics") or {}
    proxy_metrics = payload.get("target_proxy_metrics") or {}
    if V307_CONDITION == "task22_panel_mean":
        if int(proxy.get("panel_size", -1)) != 9 or int(proxy.get("subjects", -1)) != 238 or proxy.get("studies") != ["2023_UGA", "2024UGA"]:
            raise BridgeContractError("v307_h1_proxy_support")
        if float(contract.get("alpha", -1)) != 10.0 or contract.get("features") != ["log2_pre_panel_gm", "pre_panel_log2_sd"]:
            raise BridgeContractError("v307_h1_model_contract")
        expected = {"fixed_offset":"nested_inner_OOF_E05_main_effects", "outer_subject_purge":True, "inner_oof_only":True, "positivity":True, "panel_strata_not_pooled_for_decision":True}
        if any(contract.get(k) != v for k,v in expected.items()):
            raise BridgeContractError("v307_h1_leakage_or_scale_contract")
        controls = {"task22_panel_mean", "e05_main_effects", "et_subtype_d5_l10", "pre_hai"}
    else:
        teacher = payload.get("paired_teacher") or {}
        if int(teacher.get("rows", -1)) != 11913 or int(teacher.get("unique_subjects", -1)) != 567 or teacher.get("target") != "log2(Y365/Y28)":
            raise BridgeContractError("v307_h2_teacher_support")
        if int(proxy.get("panel_size", -1)) != 8 or int(proxy.get("subjects", -1)) != 439 or int(proxy.get("studies", -1)) != 3:
            raise BridgeContractError("v307_h2_proxy_support")
        expected = {"alpha":100.0, "features":["baseline HAI","age","predicted D28"], "retention_target":"log2(Y365/Y28)", "inner_oof_d28_only":True, "held_observed_d28_used":False, "challenge_observed_d28_used":False, "outer_subject_purge":True}
        if any(contract.get(k) != v for k,v in expected.items()):
            raise BridgeContractError("v307_h2_leakage_contract")
        controls = {"task23_retention", "independent_d365", "e06b_two_head"}
    for block in (metrics, proxy_metrics):
        if set((block.get("pooled") or {})) != controls or set((block.get("equal_study") or {})) != controls:
            raise BridgeContractError("v307_metric_control_set")
        if not isinstance(block.get("by_study"), list) or not isinstance(block.get("panel_strata"), list):
            raise BridgeContractError("v307_metric_strata_missing")
    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"','"subject_group"','"oof_bank"','"challenge_bank"','"row_index"')
    if any(token in serialized for token in banned):
        raise BridgeContractError("v307_aggregate_privacy_contract")'''
    runtime = v3.replace_top_level_function(runtime, "validate_result", validate_result)

    render_summary = r'''def render_summary(result: dict) -> str:
    payload = result.get("result") or {}
    lines = ["# CMI-Flu Strategy v3 V3-07", "", f"- condition: `{V307_CONDITION}`", f"- science commit: `{SCIENCE_COMMIT}`", f"- science blob: `{V307_BLOB}`", f"- fits: `{result.get('fit_count')}`", "- Competition submission: `false`", "- Public-LB selection: `false`", ""]
    metrics = payload.get("target_proxy_metrics") or {}
    for name, row in (metrics.get("pooled") or {}).items():
        lines.append(f"- target-proxy {name}: RMSE=`{row.get('rmse')}` Spearman=`{row.get('spearman')}` n=`{row.get('n')}`")
    return "\\n".join(lines) + "\\n"'''
    runtime = v3.replace_top_level_function(runtime, "render_summary", render_summary)

    execute = r'''def execute(input_dir: Path, output_dir: Path) -> int:
    import hashlib, importlib.metadata, json, shutil, sys, tempfile
    from dataclasses import replace as dc_replace
    published = []
    previous_path = list(sys.path)
    success = False
    stage = "initialize"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.is_symlink() or any(output_dir.iterdir()):
            raise BridgeContractError("v307_final_output_must_be_empty")
        with tempfile.TemporaryDirectory(prefix="cmi-v307-", dir="/tmp") as tmp:
            tmp_root = Path(tmp)
            runtime_root = tmp_root / "runtime"; runtime_root.mkdir()
            staged = tmp_root / "staged"
            package_path = runtime_root / "cmi_flu_bundle.zip"; package_path.write_bytes(package_bytes()); sys.path.insert(0, str(package_path))
            stage = "prepare_input_tree"
            data_parent = runtime_root / "data"; data_parent.mkdir(); (data_parent / "raw").symlink_to(input_dir, target_is_directory=True)
            vaccine_n, challenge_n = derive_reference_files(input_dir, runtime_root / "external")
            if (vaccine_n, challenge_n) != (3, 12): raise BridgeContractError("v307_panel_count_mismatch")
            config_dir = runtime_root / "configs"; config_dir.mkdir()
            canonical = config_dir / "baseline_b021_robust.yaml"; canonical.write_text(CONFIG_TEXT, encoding="utf-8")
            if git_blob_sha(canonical.read_bytes()) != CONFIG_BLOB: raise BridgeContractError("v307_config_blob_mismatch")
            compat = config_dir / "baseline_b02_transport_compat.yaml"
            if CONFIG_TEXT.count("baseline: b021_taskwise_robust") != 1: raise BridgeContractError("v307_config_anchor")
            compat.write_text(CONFIG_TEXT.replace("baseline: b021_taskwise_robust", "baseline: b02_taskwise_compact", 1), encoding="utf-8")
            stage = "install_b21_adapter"
            adapter_ns = {}; exec(compile(B21_ADAPTER_SOURCE, "<b21_runtime_adapter>", "exec"), adapter_ns, adapter_ns)
            install = adapter_ns.get("install")
            if not callable(install): raise BridgeContractError("v307_b21_adapter_missing")
            install()
            stage = "load_inputs"
            from cmi_flu.configuration import load_baseline_config
            from cmi_flu.runner import load_inputs
            config = load_baseline_config(compat, repository_root=runtime_root)
            raw_cfg = dict(config.raw); raw_cfg["baseline"] = "b021_taskwise_robust"
            config = dc_replace(config, source_path=canonical, raw=raw_cfg, baseline="b021_taskwise_robust")
            if not config.verify_md5 or str(config.section("selection").get("policy", "")) != "robust_v1": raise BridgeContractError("v307_config_contract")
            inputs = load_inputs(config)
            if inputs.checksum_report is None: raise BridgeContractError("v307_md5_missing")
            stage = "load_exact_v307"
            v307, hai_module = load_v307_module()
            kwargs = {}
            if V307_CONDITION == "task22_panel_mean":
                import pandas as pd
                kwargs["sequence_reference"] = pd.read_csv(locate_locked_reference("strain_sequences.csv"))
                kwargs["vaccine_reference"] = hai_module.load_vaccine_strain_reference(locate_locked_reference("vaccine_strains_per_season.txt"))
            stage = "run_v307"
            aggregate, oof_bank, challenge_bank = v307.run_v3_07(V307_CONDITION, config, inputs, **kwargs)
            aggregate = json_safe(dict(aggregate))
            aggregate["runtime"] = {
                "request_id": V307_REQUEST_ID, "target_kernel": V307_TARGET_KERNEL, "science_commit": SCIENCE_COMMIT,
                "science_blob": V307_BLOB, "condition": V307_CONDITION,
                "library_versions": {name: importlib.metadata.version(name) for name in ("numpy","pandas","scipy","scikit-learn")},
                "runtime_terminal_marker": "CMI_FLU_V307_RUNTIME_PASS",
            }
            stage = "validate_aggregate"
            validate_result(aggregate)
            if int(aggregate.get("fit_count", 999999)) > 256: raise BridgeContractError("v307_fit_budget_exceeded")
            stage = "write_private_outputs"
            manifest = v307.write_v3_07_outputs(V307_CONDITION, aggregate, oof_bank, challenge_bank, staged)
            expected = sorted([f"{V307_OUTPUT_PREFIX}_oof_bank.csv", f"{V307_OUTPUT_PREFIX}_challenge_bank.csv", f"{V307_OUTPUT_PREFIX}_summary.json", f"{V307_OUTPUT_PREFIX}_manifest.json"])
            if sorted(p.name for p in staged.iterdir()) != expected: raise BridgeContractError("v307_output_allowlist")
            if any(p.is_symlink() or not p.is_file() for p in staged.iterdir()): raise BridgeContractError("v307_nonregular_output")
            if manifest.get("condition") != V307_CONDITION or manifest.get("row_level_banks_private") is not True: raise BridgeContractError("v307_manifest_contract")
            for item in manifest.get("files", []):
                p = staged / item["filename"]
                if p.stat().st_size != int(item["bytes"]) or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]: raise BridgeContractError("v307_output_hash")
            stage = "publish_validated_outputs"
            for name in expected:
                destination = output_dir / name
                with destination.open("xb") as stream:
                    published.append(destination); stream.write((staged / name).read_bytes())
            success = True
        print(f"CMI_FLU_V307_RUNTIME_PASS request_id={V307_REQUEST_ID} condition={V307_CONDITION} fit_count={aggregate.get('fit_count')} submission=false public_used=false private_banks=true")
        return 0
    except Exception as exc:
        code = hashlib.sha256(f"{stage}:{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")).hexdigest()[:20]
        print(f"CMI_FLU_V307_RUNTIME_FAIL stage={stage} type={type(exc).__name__} code={code}", file=sys.stderr)
        return 2
    finally:
        sys.path[:] = previous_path
        if not success:
            for path in reversed(published): path.unlink(missing_ok=True)'''
    runtime = v3.replace_top_level_function(runtime, "execute", execute)

    old_description = "CMI-Flu strategy E06b D28/D365 two-head. Aggregate outputs only; no submission."
    runtime = runtime.replace(old_description, "CMI-Flu Strategy v3 V3-07. Private row banks plus aggregate summary; no submission.", 1)
    runtime = runtime.replace("CMI_FLU_E06B_FAILED", "CMI_FLU_V307_FAILED").replace("CMI_FLU_E06B_COMPLETE", "CMI_FLU_V307_COMPLETE")
    names = v3.top_level_functions(runtime)
    for name in ("load_v307_module", "validate_result", "render_summary", "execute", "locked_reference_bytes", "locate_locked_reference"):
        if names.count(name) != 1:
            raise SystemExit(f"V3-07 top-level binding failed:{name}:{names.count(name)}")
    for old in (parent.REQUEST_ID, parent.TARGET_KERNEL, parent.SCIENCE_COMMIT):
        if old in runtime:
            raise SystemExit(f"V3-07 retained E06b identity:{old}")
    low = runtime.casefold()
    if any(token in low for token in submission_markers()):
        raise SystemExit("V3-07 generated runtime contains Competition submit path")
    if "/kaggle/working/.e05-locked-references" in runtime or 'base / ".e05-locked-references"' in runtime:
        raise SystemExit("V3-07 inherited unsafe locked-reference output staging")
    compile(runtime, f"generated_v307_{condition}.py", "exec")
    return runtime


def build_runtime(root: Path, reference_dir: Path, condition: str) -> str:
    v307 = load_v307(root)
    v3, parent_runtime = load_parent(root, reference_dir)
    return patch_runtime(v3, parent_runtime, v307, condition)


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    refs = args.reference_dir.expanduser().resolve()
    runtime = build_runtime(root, refs, args.condition)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    c = CONTRACTS[args.condition]
    print(f"CMI_FLU_V307_BUILD PASS condition={args.condition} request_id={c['request_id']} target={c['target']} science_commit={SCIENCE_COMMIT} science_blob={V307_BLOB} bytes={len(runtime.encode())} runtime_sha256={sha256(runtime.encode())} submission=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
