#!/usr/bin/env python3
"""Build the Strategy-v3 V3-02 fixed validation/private-bank Kaggle runtime."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import tempfile

import cmi_flu_v3_batch1_full_runtime_prepare as parent
from cmi_flu_v3_batch1_prepare import replace_function

REQUEST_ID = "20260912-cmi-flu-strategy-v3-v02-validation-bank-001"
TARGET = "renta0426/cmi-flu-v3-v02-validation-bank-20260912-001"
PAYLOAD = "payloads/cmi-flu-v3-v02-validation-bank-001/strategy_v3_v02.py"
SCIENCE_BLOB = "e8e87bda474814ecf64b6b9530b46b5d81d1d27b"
SOURCE_B_SHA256 = "0f9df53c3aa8c6e4ac693f6a42dbd2633b4b61d1462df798a9bd767c220be3a5"
SOURCE_B_BYTES = 5926


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def function_names(source: str) -> set[str]:
    return {node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


def load_source(root: Path) -> str:
    raw = (root / PAYLOAD).read_bytes()
    found = git_blob(raw)
    if found != SCIENCE_BLOB:
        raise SystemExit(f"V3-02 exact source blob mismatch:{found}")
    source = raw.decode("utf-8")
    compile(source, "cmi_flu/strategy_v3_v02.py", "exec")
    lowered = source.casefold()
    for token in ("competition_submit(", "kaggle competitions submit", "competitions submit"):
        if token in lowered:
            raise SystemExit("V3-02 source contains Competition submit path")
    required = ("def run_v3_02(", "def write_v3_02_outputs(", "MAX_FITS = 256", "PSEUDO_REPETITIONS = 200")
    if any(token not in source for token in required):
        raise SystemExit("V3-02 source contract token missing")
    return source


def build_runtime(root: Path, output: Path) -> str:
    source = load_source(root)
    with tempfile.TemporaryDirectory(prefix="cmi-v302-parent-") as tmp:
        raw = parent.build_runtime(root, Path(tmp) / "parent.py")
    names = function_names(raw)
    for required in ("load_e12c_v2_module", "load_v3_dependency_closure", "package_bytes", "derive_reference_files", "execute", "main"):
        if required not in names:
            raise SystemExit(f"V3-02 parent runtime entry missing:{required}")
    raw = raw.replace(parent.REQUEST_ID, REQUEST_ID).replace(parent.TARGET, TARGET)
    marker = f'V3_SOURCE_B_SHA256 = "{SOURCE_B_SHA256}"\n'
    if raw.count(marker) != 1:
        raise SystemExit("V3-02 constant insertion anchor changed")
    constants = (
        marker
        + f'V302_REQUEST_ID = {REQUEST_ID!r}\n'
        + f'V302_TARGET_KERNEL = {TARGET!r}\n'
        + f'V302_SCIENCE_BLOB = {SCIENCE_BLOB!r}\n'
        + f'V302_SOURCE = {source!r}\n'
        + 'V302_SOURCE_SEARCH_ROOT = Path("/kaggle/input")\n'
        + 'V302_RUNTIME_SCRATCH_ROOT = Path("/tmp/cmi-flu-v3-v02-runtime")\n'
    )
    raw = raw.replace(marker, constants, 1)

    loader = r'''def load_v3_v02_module() -> object:
    import sys, types
    load_e12c_v2_module()
    load_v3_dependency_closure()
    module = types.ModuleType("cmi_flu.strategy_v3_v02")
    module.__file__ = "<cmi_flu.strategy_v3_v02>"
    module.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_v3_v02"] = module
    exec(compile(V302_SOURCE, "cmi_flu/strategy_v3_v02.py", "exec"), module.__dict__, module.__dict__)
    for name in ("run_v3_02", "write_v3_02_outputs"):
        if not callable(getattr(module, name, None)):
            raise BridgeContractError("v302_entry_missing:" + name)
    if int(getattr(module, "MAX_FITS", -1)) != 256:
        raise BridgeContractError("v302_fit_limit_changed")
    return module
'''
    execute_anchor = "def execute(input_dir: Path, output_dir: Path) -> int:\n"
    if raw.count(execute_anchor) != 1:
        raise SystemExit("V3-02 execute insertion anchor changed")
    raw = raw.replace(execute_anchor, loader + "\n" + execute_anchor, 1)

    execution = r'''def execute(input_dir: Path, output_dir: Path) -> int:
    import contextlib, hashlib, importlib.metadata, io, json, shutil, sys, tempfile
    from dataclasses import replace as dc_replace
    import pandas as pd
    global V302_RUNTIME_SCRATCH_ROOT
    declared = ["v3_v02_oof_bank.csv", "v3_v02_challenge_bank.csv", "v3_v02_summary.json", "v3_v02_bank_manifest.json"]
    published = []
    previous_path = list(sys.path)
    success = False
    stage = "initialize"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.is_symlink() or any(output_dir.iterdir()):
            raise BridgeContractError("v302_final_output_must_be_empty")
        with tempfile.TemporaryDirectory(prefix="cmi-v302-", dir="/tmp") as tmp:
            tmp_root = Path(tmp)
            V302_RUNTIME_SCRATCH_ROOT = tmp_root / "runtime"
            staged = tmp_root / "staged"
            V302_RUNTIME_SCRATCH_ROOT.mkdir()
            package_path = V302_RUNTIME_SCRATCH_ROOT / "cmi_flu_bundle.zip"
            package_path.write_bytes(package_bytes())
            sys.path.insert(0, str(package_path))

            stage = "load_exact_v302_module"
            v302 = load_v3_v02_module()

            stage = "prepare_input_tree"
            data_parent = V302_RUNTIME_SCRATCH_ROOT / "data"
            data_parent.mkdir()
            (data_parent / "raw").symlink_to(input_dir, target_is_directory=True)
            vaccine_n, challenge_n = derive_reference_files(input_dir, V302_RUNTIME_SCRATCH_ROOT / "external")
            if (vaccine_n, challenge_n) != (3, 12):
                raise BridgeContractError("v302_panel_count_mismatch")
            config_dir = V302_RUNTIME_SCRATCH_ROOT / "configs"
            config_dir.mkdir()
            canonical = config_dir / "baseline_b021_robust.yaml"
            canonical.write_text(CONFIG_TEXT, encoding="utf-8")
            if git_blob_sha(canonical.read_bytes()) != CONFIG_BLOB:
                raise BridgeContractError("v302_config_blob_mismatch")
            compat = config_dir / "baseline_b02_transport_compat.yaml"
            if CONFIG_TEXT.count("baseline: b021_taskwise_robust") != 1:
                raise BridgeContractError("v302_config_anchor_changed")
            compat.write_text(CONFIG_TEXT.replace("baseline: b021_taskwise_robust", "baseline: b02_taskwise_compact", 1), encoding="utf-8")
            from cmi_flu.configuration import load_baseline_config
            from cmi_flu.runner import load_inputs
            config = load_baseline_config(compat, repository_root=V302_RUNTIME_SCRATCH_ROOT)
            raw_cfg = dict(config.raw); raw_cfg["baseline"] = "b021_taskwise_robust"
            config = dc_replace(config, source_path=canonical, raw=raw_cfg, baseline="b021_taskwise_robust")
            inputs = load_inputs(config)
            if inputs.checksum_report is None:
                raise BridgeContractError("v302_md5_verification_missing")

            stage = "locate_exact_source_B"
            matches = []
            for candidate in V302_SOURCE_SEARCH_ROOT.rglob("submission.csv"):
                if candidate.is_file():
                    data = candidate.read_bytes()
                    if len(data) == 5926 and hashlib.sha256(data).hexdigest() == V3_SOURCE_B_SHA256:
                        matches.append(candidate)
            if len(matches) != 1:
                raise BridgeContractError("v302_exact_source_B_count:" + str(len(matches)))
            source_b = pd.read_csv(matches[0])

            stage = "run_fixed_validation_bank"
            aggregate, oof_bank, challenge_bank = v302.run_v3_02(config, inputs, source_b_csv=source_b)
            if int(aggregate.get("fit_count", 999999)) > 256:
                raise BridgeContractError("v302_fit_budget_exceeded")
            if aggregate.get("new_candidate_conditions") != 0 or aggregate.get("public_leaderboard_used") is not False:
                raise BridgeContractError("v302_science_boundary_changed")
            if aggregate.get("competition_submission_attempted") is not False:
                raise BridgeContractError("v302_submission_boundary_changed")

            stage = "write_private_bank"
            manifest = v302.write_v3_02_outputs(aggregate, oof_bank, challenge_bank, staged)
            if sorted(p.name for p in staged.iterdir()) != sorted(declared):
                raise BridgeContractError("v302_output_allowlist_mismatch")
            if any(p.is_symlink() or not p.is_file() for p in staged.iterdir()):
                raise BridgeContractError("v302_nonregular_output")
            for item in manifest.get("files", []):
                p = staged / item["filename"]
                if p.stat().st_size != item["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
                    raise BridgeContractError("v302_private_bank_hash_mismatch")
            summary_path = staged / "v3_v02_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["runtime"] = {
                "request_id": V302_REQUEST_ID,
                "science_blob": V302_SCIENCE_BLOB,
                "source_B_sha256": V3_SOURCE_B_SHA256,
                "library_versions": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn")},
                "runtime_terminal_marker": "CMI_FLU_V3_V02_RUNTIME_PASS",
            }
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")

            stage = "publish_validated_outputs"
            for name in declared:
                destination = output_dir / name
                with destination.open("xb") as stream:
                    published.append(destination)
                    stream.write((staged / name).read_bytes())
            success = True
        print(
            "CMI_FLU_V3_V02_RUNTIME_PASS "
            f"fit_count={aggregate.get('fit_count')} oof_rows={len(oof_bank)} challenge_rows={len(challenge_bank)} "
            "new_conditions=0 competition_submit=false public_used=false private_bank=true scratch_cleanup=true"
        )
        return 0
    except Exception as exc:
        code = hashlib.sha256(f"{stage}:{type(exc).__name__}:{exc}".encode()).hexdigest()[:20]
        print(f"CMI_FLU_V3_V02_RUNTIME_FAIL stage={stage} type={type(exc).__name__} code={code}", file=sys.stderr)
        return 1
    finally:
        sys.path[:] = previous_path
        if not success:
            for path in reversed(published):
                path.unlink(missing_ok=True)
'''
    raw = replace_function(raw, "execute", execution)

    self_test = r'''def self_test() -> int:
    import ast, sys, tempfile
    if git_blob_sha(V302_SOURCE.encode()) != V302_SCIENCE_BLOB:
        raise BridgeContractError("v302_blob_mismatch")
    tree = ast.parse(V302_SOURCE)
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and "submit" in n.func.attr.casefold() for n in ast.walk(tree)):
        raise BridgeContractError("v302_submit_call_present")
    with tempfile.TemporaryDirectory(prefix="v302-selftest-") as tmp:
        package = Path(tmp) / "cmi_flu_bundle.zip"; package.write_bytes(package_bytes()); sys.path.insert(0, str(package))
        try:
            module = load_v3_v02_module()
            if module.EXPERIMENT != "strategy_v3_v02_paired_validation_and_private_prediction_store":
                raise BridgeContractError("v302_experiment_identity")
        finally:
            if sys.path and sys.path[0] == str(package): sys.path.pop(0)
    print(f"CMI_FLU_V3_V02_SELF_TEST PASS request_id={V302_REQUEST_ID} blob={V302_SCIENCE_BLOB} fit_limit=256 submission=false")
    return 0
'''
    raw = replace_function(raw, "self_test", self_test)

    entry = r'''def main() -> int:
    import argparse, hashlib, sys
    global V302_SOURCE_SEARCH_ROOT
    parser = argparse.ArgumentParser(description="CMI-Flu Strategy-v3 V3-02 fixed validation/private bank")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    if args.self_test: return self_test()
    try:
        V302_SOURCE_SEARCH_ROOT = args.input_dir.resolve()
        required = ("participants.tsv", "investigations_260821.tsv", "publicData_cytokine.tsv", "publicData_ex_vivo_flow.tsv", "publicData_serology_260821.tsv", "2025LJI_cytokine.tsv", "2025LJI_ex_vivo_flow.tsv", "2025LJI_serology.tsv", "sample_submission_part1.csv", "md5sum")
        roots = {p.parent for p in V302_SOURCE_SEARCH_ROOT.rglob("sample_submission_part1.csv") if all((p.parent / n).is_file() for n in required)}
        if len(roots) != 1: raise BridgeContractError("v302_exact_competition_root_missing_or_ambiguous")
        return execute(next(iter(roots)), args.output_dir.resolve())
    except Exception as exc:
        stage = "locate_competition_input"; code = hashlib.sha256(f"{stage}:{type(exc).__name__}:{exc}".encode()).hexdigest()[:20]
        print(f"CMI_FLU_V3_V02_RUNTIME_FAIL stage={stage} type={type(exc).__name__} code={code}", file=sys.stderr); return 1
'''
    raw = replace_function(raw, "main", entry)
    compile(raw, "v3-v02-runtime.py", "exec")
    if len(raw.encode()) >= 1100000:
        raise SystemExit("V3-02 runtime exceeds reviewed byte budget")
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(raw, encoding="utf-8"); return raw


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--repository-root", required=True, type=Path); p.add_argument("--output", required=True, type=Path); a = p.parse_args()
    raw = build_runtime(a.repository_root.resolve(), a.output.resolve()).encode(); print(f"CMI_FLU_V3_V02_PREPARE PASS bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()} science_blob={SCIENCE_BLOB} submission=false"); return 0


if __name__ == "__main__": raise SystemExit(main())
