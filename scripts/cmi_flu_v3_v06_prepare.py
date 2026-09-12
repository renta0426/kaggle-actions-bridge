#!/usr/bin/env python3
"""Build deterministic V3-06 actual D28 ET calibration runtime."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import tempfile

import cmi_flu_v3_v02_prepare as parent
from cmi_flu_v3_batch1_prepare import replace_function

REQUEST_ID = "20260912-cmi-flu-strategy-v3-v06-actual-et-calibration-001"
TARGET = "renta0426/cmi-flu-v3-v06-actual-et-calibration-20260912-001"
PAYLOAD = "payloads/cmi-flu-v3-v06-actual-et-calibration-001/strategy_v3_v06.py"
SCIENCE_COMMIT = "dc35aa83fe32c725394e0bc354345a4535e9df45"
SCIENCE_BLOB = "5f75baa73750abed53b476ef9cf525e5dbf37a50"


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def load_source(root: Path) -> str:
    raw = (root / PAYLOAD).read_bytes()
    if git_blob(raw) != SCIENCE_BLOB:
        raise ValueError("V3-06 exact science relay blob mismatch")
    source = raw.decode("utf-8")
    compile(source, "cmi_flu/strategy_v3_v06.py", "exec")
    required = (
        'MODEL_BY_TASK = {', '"Task2.1": "et_subtype_d3_l5"',
        '"Task2.2": "et_subtype_d5_l10"', 'CALIBRATION_KIND = "log2_affine"',
        'INNER_STUDY_FOLDS = 3', 'MAX_FITS = 256', 'def run_v3_06(',
        'affine_or_power_grid_reopened": False',
    )
    if any(token not in source for token in required):
        raise ValueError("V3-06 frozen source contract token missing")
    low = source.casefold()
    if "competition_submit(" in low or "kaggle competitions submit" in low:
        raise ValueError("V3-06 science source contains submission path")
    return source


def build_runtime(root: Path, output: Path) -> str:
    source = load_source(root)
    with tempfile.TemporaryDirectory(prefix="v306-parent-") as td:
        raw = parent.build_runtime(root, Path(td) / "parent.py")
    raw = raw.replace(parent.REQUEST_ID, REQUEST_ID).replace(parent.TARGET, TARGET)
    marker = f'V302_SCIENCE_BLOB = {parent.SCIENCE_BLOB!r}\n'
    if raw.count(marker) != 1:
        raise ValueError("V3-06 parent insertion anchor changed")
    raw = raw.replace(
        marker,
        marker
        + f'V306_REQUEST_ID = {REQUEST_ID!r}\n'
        + f'V306_TARGET_KERNEL = {TARGET!r}\n'
        + f'V306_SCIENCE_COMMIT = {SCIENCE_COMMIT!r}\n'
        + f'V306_SCIENCE_BLOB = {SCIENCE_BLOB!r}\n'
        + f'V306_SOURCE = {source!r}\n'
        + 'V306_RUNTIME_SCRATCH_ROOT = Path("/tmp/cmi-flu-v3-v06-runtime")\n',
        1,
    )
    loader = r'''def load_v3_v06_module() -> object:
    import sys, types
    load_e12c_v2_module()
    load_v3_dependency_closure()
    module = types.ModuleType("cmi_flu.strategy_v3_v06")
    module.__file__ = "<cmi_flu.strategy_v3_v06>"
    module.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_v3_v06"] = module
    exec(compile(V306_SOURCE, "cmi_flu/strategy_v3_v06.py", "exec"), module.__dict__, module.__dict__)
    for name in ("run_v3_06", "write_v3_06_outputs"):
        if not callable(getattr(module, name, None)):
            raise BridgeContractError("v306_entry_missing:" + name)
    if module.MODEL_BY_TASK != {"Task2.1":"et_subtype_d3_l5", "Task2.2":"et_subtype_d5_l10"}:
        raise BridgeContractError("v306_model_set_changed")
    if module.CALIBRATION_KIND != "log2_affine" or module.INNER_STUDY_FOLDS != 3 or module.MAX_FITS != 256:
        raise BridgeContractError("v306_calibration_contract_changed")
    return module
'''
    anchor = "def execute(input_dir: Path, output_dir: Path) -> int:\n"
    if raw.count(anchor) != 1:
        raise ValueError("V3-06 execute anchor changed")
    raw = raw.replace(anchor, loader + "\n" + anchor, 1)
    execution = r'''def execute(input_dir: Path, output_dir: Path) -> int:
    import contextlib, hashlib, importlib.metadata, io, json, shutil, sys, tempfile
    from dataclasses import replace as dc_replace
    global V306_RUNTIME_SCRATCH_ROOT
    declared = ["v3_v06_oof_bank.csv", "v3_v06_challenge_bank.csv", "v3_v06_summary.json", "v3_v06_bank_manifest.json"]
    published = []
    previous_path = list(sys.path)
    success = False
    stage = "initialize"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.is_symlink() or any(output_dir.iterdir()):
            raise BridgeContractError("v306_final_output_must_be_empty")
        with tempfile.TemporaryDirectory(prefix="cmi-v306-", dir="/tmp") as tmp:
            tmp_root = Path(tmp)
            V306_RUNTIME_SCRATCH_ROOT = tmp_root / "runtime"; V306_RUNTIME_SCRATCH_ROOT.mkdir()
            staged = tmp_root / "staged"
            package = V306_RUNTIME_SCRATCH_ROOT / "cmi_flu_bundle.zip"; package.write_bytes(package_bytes()); sys.path.insert(0, str(package))
            stage = "load_exact_v306_module"
            v306 = load_v3_v06_module()
            stage = "prepare_input_tree"
            data_parent = V306_RUNTIME_SCRATCH_ROOT / "data"; data_parent.mkdir(); (data_parent / "raw").symlink_to(input_dir, target_is_directory=True)
            vaccine_n, challenge_n = derive_reference_files(input_dir, V306_RUNTIME_SCRATCH_ROOT / "external")
            if (vaccine_n, challenge_n) != (3, 12): raise BridgeContractError("v306_panel_count_mismatch")
            config_dir = V306_RUNTIME_SCRATCH_ROOT / "configs"; config_dir.mkdir()
            canonical = config_dir / "baseline_b021_robust.yaml"; canonical.write_text(CONFIG_TEXT, encoding="utf-8")
            if git_blob_sha(canonical.read_bytes()) != CONFIG_BLOB: raise BridgeContractError("v306_config_blob_mismatch")
            compat = config_dir / "baseline_b02_transport_compat.yaml"
            compat.write_text(CONFIG_TEXT.replace("baseline: b021_taskwise_robust", "baseline: b02_taskwise_compact", 1), encoding="utf-8")
            from cmi_flu.configuration import load_baseline_config
            from cmi_flu.runner import load_inputs
            config = load_baseline_config(compat, repository_root=V306_RUNTIME_SCRATCH_ROOT)
            raw_cfg = dict(config.raw); raw_cfg["baseline"] = "b021_taskwise_robust"
            config = dc_replace(config, source_path=canonical, raw=raw_cfg, baseline="b021_taskwise_robust")
            inputs = load_inputs(config)
            if inputs.checksum_report is None: raise BridgeContractError("v306_md5_verification_missing")
            stage = "run_actual_et_calibration"
            aggregate, oof_bank, challenge_bank = v306.run_v3_06(config, inputs)
            if int(aggregate.get("fit_count", 999999)) > 256: raise BridgeContractError("v306_fit_budget_exceeded")
            if aggregate.get("new_candidate_conditions") != ["task21_et_log_affine", "task22_et_log_affine"]: raise BridgeContractError("v306_condition_set_changed")
            contract = aggregate.get("calibration_contract") or {}
            if contract.get("kind") != "log2_affine" or contract.get("inner_study_folds") != 3 or contract.get("affine_or_power_grid_reopened") is not False: raise BridgeContractError("v306_calibration_contract_changed")
            if aggregate.get("public_leaderboard_used") is not False or aggregate.get("competition_submission_attempted") is not False: raise BridgeContractError("v306_public_or_submit_boundary_changed")
            stage = "write_private_bank"
            manifest = v306.write_v3_06_outputs(aggregate, oof_bank, challenge_bank, staged)
            if sorted(p.name for p in staged.iterdir()) != sorted(declared): raise BridgeContractError("v306_output_allowlist_mismatch")
            for item in manifest.get("files", []):
                p = staged / item["filename"]
                if p.stat().st_size != item["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]: raise BridgeContractError("v306_private_bank_hash_mismatch")
            summary_path = staged / "v3_v06_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["runtime"] = {"request_id":V306_REQUEST_ID,"science_commit":V306_SCIENCE_COMMIT,"science_blob":V306_SCIENCE_BLOB,"library_versions":{n:importlib.metadata.version(n) for n in ("numpy","pandas","scipy","scikit-learn")},"runtime_terminal_marker":"CMI_FLU_V3_V06_RUNTIME_PASS"}
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n")
            stage = "publish_validated_outputs"
            for name in declared:
                destination = output_dir / name
                with destination.open("xb") as stream: published.append(destination); stream.write((staged / name).read_bytes())
            success = True
        print(f"CMI_FLU_V3_V06_RUNTIME_PASS fit_count={aggregate.get('fit_count')} oof_rows={len(oof_bank)} challenge_rows={len(challenge_bank)} conditions=2 competition_submit=false public_used=false private_bank=true")
        return 0
    except Exception as exc:
        code = hashlib.sha256(f"{stage}:{type(exc).__name__}:{exc}".encode()).hexdigest()[:20]
        print(f"CMI_FLU_V3_V06_RUNTIME_FAIL stage={stage} type={type(exc).__name__} code={code}", file=sys.stderr)
        return 1
    finally:
        sys.path[:] = previous_path
        if not success:
            for path in reversed(published): path.unlink(missing_ok=True)
'''
    raw = replace_function(raw, "execute", execution)
    self_test = r'''def self_test() -> int:
    if git_blob_sha(V306_SOURCE.encode()) != V306_SCIENCE_BLOB: raise BridgeContractError("v306_blob_mismatch")
    module = load_v3_v06_module()
    if module.EXPERIMENT != "strategy_v3_v06_actual_d28_et_log2_affine_calibration": raise BridgeContractError("v306_experiment_identity")
    print(f"CMI_FLU_V3_V06_SELF_TEST PASS request_id={V306_REQUEST_ID} science_commit={V306_SCIENCE_COMMIT} science_blob={V306_SCIENCE_BLOB} models=2 conditions=2 inner_folds=3 fit_limit=256 submission=false")
    return 0
'''
    raw = replace_function(raw, "self_test", self_test)
    raw = raw.replace('description="CMI-Flu Strategy-v3 V3-02 fixed validation/private bank"', 'description="CMI-Flu Strategy-v3 V3-06 actual ET calibration"')
    compile(raw, "v306-runtime.py", "exec")
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(raw, encoding="utf-8")
    return raw


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--repository-root",required=True,type=Path); p.add_argument("--output",required=True,type=Path); a=p.parse_args()
    raw=build_runtime(a.repository_root.resolve(),a.output.resolve()).encode()
    print(f"V306_PREPARE PASS bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()} science_commit={SCIENCE_COMMIT} science_blob={SCIENCE_BLOB} models=2 conditions=2 submission=false")
    return 0

if __name__ == "__main__": raise SystemExit(main())
