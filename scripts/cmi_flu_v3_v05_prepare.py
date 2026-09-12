#!/usr/bin/env python3
"""Build the exact Strategy-v3 V3-05 Task1.3 scale-head Kaggle runtime."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import tempfile

import cmi_flu_v3_v02_prepare as parent
from cmi_flu_v3_batch1_prepare import replace_function

REQUEST_ID = "20260912-cmi-flu-strategy-v3-v05-task13-scale-001"
TARGET = "renta0426/cmi-flu-v3-v05-task13-scale-20260912-001"
PAYLOAD = "payloads/cmi-flu-v3-v05-task13-scale-001/strategy_v3_v05.py"
SCIENCE_COMMIT = "2b8dad2edddfb4df168ecce5f55ad2576d22dc1e"
SCIENCE_BLOB = "136d92de6a5d4f23fb4bb66d4c339aa8f49e79f0"


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def function_names(source: str) -> set[str]:
    return {node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


def load_source(root: Path) -> str:
    raw = (root / PAYLOAD).read_bytes()
    found = git_blob(raw)
    if found != SCIENCE_BLOB:
        raise SystemExit(f"V3-05 exact science blob mismatch:{found}")
    source = raw.decode("utf-8")
    compile(source, "cmi_flu/strategy_v3_v05.py", "exec")
    lowered = source.casefold()
    if any(x in lowered for x in ("competition_submit(", "kaggle competitions submit", "competitions submit")):
        raise SystemExit("V3-05 source contains Competition submit path")
    required = (
        'NEW_CONDITIONS = ("S1", "S2", "S3", "S4")', "MAX_FITS = 96",
        "OUTER_SPLITS = 5", "OUTER_REPEATS = 3", "def run_v3_05_task13(",
        "def write_v3_05_outputs(", '"not_executed_data_limited"',
        '"raw_unit_conversion_performed": False', '"pseudocount_added": False',
    )
    if any(token not in source for token in required):
        raise SystemExit("V3-05 source contract token missing")
    return source


def build_runtime(root: Path, output: Path) -> str:
    source = load_source(root)
    with tempfile.TemporaryDirectory(prefix="cmi-v305-parent-") as tmp:
        raw = parent.build_runtime(root, Path(tmp) / "parent.py")
    for required in (
        "load_v3_v02_module", "load_v3_dependency_closure", "package_bytes",
        "derive_reference_files", "execute", "self_test", "main",
    ):
        if required not in function_names(raw):
            raise SystemExit(f"V3-05 parent runtime entry missing:{required}")
    raw = raw.replace(parent.REQUEST_ID, REQUEST_ID).replace(parent.TARGET, TARGET)
    marker = f'V302_SCIENCE_BLOB = {parent.SCIENCE_BLOB!r}\n'
    if raw.count(marker) != 1:
        raise SystemExit("V3-05 constant insertion anchor changed")
    raw = raw.replace(
        marker,
        marker
        + f'V305_REQUEST_ID = {REQUEST_ID!r}\n'
        + f'V305_TARGET_KERNEL = {TARGET!r}\n'
        + f'V305_SCIENCE_COMMIT = {SCIENCE_COMMIT!r}\n'
        + f'V305_SCIENCE_BLOB = {SCIENCE_BLOB!r}\n'
        + f'V305_SOURCE = {source!r}\n'
        + 'V305_RUNTIME_SCRATCH_ROOT = Path("/tmp/cmi-flu-v3-v05-runtime")\n',
        1,
    )

    loader = r'''def load_v3_v05_module() -> object:
    import sys, types
    load_v3_v02_module()
    # The frozen dependency dataclass predates CandidateEvaluation.raw_oof_predictions.
    # V3-05 calls evaluate_model_spec with aggregate_repeats=False, so the old
    # oof_predictions object is exactly the raw OOF frame. Add only that missing
    # compatibility attribute; no split, fit, target or prediction is changed.
    import cmi_flu.models as _models
    _v302_eval = _models.evaluate_model_spec
    def _v305_eval_compat(*args, **kwargs):
        result = _v302_eval(*args, **kwargs)
        if not hasattr(result, "raw_oof_predictions"):
            if kwargs.get("aggregate_repeats", False):
                raise BridgeContractError("v305_raw_oof_unavailable_after_aggregation")
            result.raw_oof_predictions = result.oof_predictions.copy()
        return result
    _models.evaluate_model_spec = _v305_eval_compat
    module = types.ModuleType("cmi_flu.strategy_v3_v05")
    module.__file__ = "<cmi_flu.strategy_v3_v05>"
    module.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_v3_v05"] = module
    exec(compile(V305_SOURCE, "cmi_flu/strategy_v3_v05.py", "exec"), module.__dict__, module.__dict__)
    for name in ("run_v3_05_task13", "write_v3_05_outputs"):
        if not callable(getattr(module, name, None)):
            raise BridgeContractError("v305_entry_missing:" + name)
    if tuple(getattr(module, "NEW_CONDITIONS", ())) != ("S1", "S2", "S3", "S4"):
        raise BridgeContractError("v305_condition_set_changed")
    if int(getattr(module, "MAX_FITS", -1)) != 96:
        raise BridgeContractError("v305_fit_limit_changed")
    return module
'''
    anchor = "def execute(input_dir: Path, output_dir: Path) -> int:\n"
    if raw.count(anchor) != 1:
        raise SystemExit("V3-05 execute insertion anchor changed")
    raw = raw.replace(anchor, loader + "\n" + anchor, 1)

    execution = r'''def execute(input_dir: Path, output_dir: Path) -> int:
    import hashlib, importlib.metadata, json, sys, tempfile
    from dataclasses import replace as dc_replace
    global V305_RUNTIME_SCRATCH_ROOT
    declared = ["v3_v05_task13_oof_bank.csv", "v3_v05_task13_challenge_bank.csv", "v3_v05_task13_summary.json", "v3_v05_task13_bank_manifest.json"]
    published = []
    previous_path = list(sys.path)
    success = False
    stage = "initialize"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.is_symlink() or any(output_dir.iterdir()):
            raise BridgeContractError("v305_final_output_must_be_empty")
        with tempfile.TemporaryDirectory(prefix="cmi-v305-", dir="/tmp") as tmp:
            tmp_root = Path(tmp)
            V305_RUNTIME_SCRATCH_ROOT = tmp_root / "runtime"
            staged = tmp_root / "staged"
            V305_RUNTIME_SCRATCH_ROOT.mkdir()
            package_path = V305_RUNTIME_SCRATCH_ROOT / "cmi_flu_bundle.zip"
            package_path.write_bytes(package_bytes())
            sys.path.insert(0, str(package_path))
            stage = "load_exact_v305_module"
            v305 = load_v3_v05_module()

            stage = "prepare_input_tree"
            data_parent = V305_RUNTIME_SCRATCH_ROOT / "data"
            data_parent.mkdir()
            (data_parent / "raw").symlink_to(input_dir, target_is_directory=True)
            vaccine_n, challenge_n = derive_reference_files(input_dir, V305_RUNTIME_SCRATCH_ROOT / "external")
            if (vaccine_n, challenge_n) != (3, 12):
                raise BridgeContractError("v305_panel_count_mismatch")
            config_dir = V305_RUNTIME_SCRATCH_ROOT / "configs"
            config_dir.mkdir()
            canonical = config_dir / "baseline_b021_robust.yaml"
            canonical.write_text(CONFIG_TEXT, encoding="utf-8")
            if git_blob_sha(canonical.read_bytes()) != CONFIG_BLOB:
                raise BridgeContractError("v305_config_blob_mismatch")
            compat = config_dir / "baseline_b02_transport_compat.yaml"
            if CONFIG_TEXT.count("baseline: b021_taskwise_robust") != 1:
                raise BridgeContractError("v305_config_anchor_changed")
            compat.write_text(CONFIG_TEXT.replace("baseline: b021_taskwise_robust", "baseline: b02_taskwise_compact", 1), encoding="utf-8")
            from cmi_flu.configuration import load_baseline_config
            from cmi_flu.runner import load_inputs
            config = load_baseline_config(compat, repository_root=V305_RUNTIME_SCRATCH_ROOT)
            raw_cfg = dict(config.raw); raw_cfg["baseline"] = "b021_taskwise_robust"
            config = dc_replace(config, source_path=canonical, raw=raw_cfg, baseline="b021_taskwise_robust")
            inputs = load_inputs(config)
            if inputs.checksum_report is None:
                raise BridgeContractError("v305_md5_verification_missing")

            stage = "run_task13_scale_heads"
            aggregate, oof_bank, challenge_bank = v305.run_v3_05_task13(config, inputs)
            if aggregate.get("new_candidate_conditions") != ["S1", "S2", "S3", "S4"] or int(aggregate.get("new_candidate_condition_count", -1)) != 4:
                raise BridgeContractError("v305_condition_set_changed")
            if int(aggregate.get("fit_count", 999999)) > 96:
                raise BridgeContractError("v305_fit_budget_exceeded")
            if (aggregate.get("Task1.2") or {}).get("state") != "not_executed_data_limited":
                raise BridgeContractError("v305_task12_data_limited_boundary_changed")
            if aggregate.get("raw_unit_conversion_performed") is not False or aggregate.get("pseudocount_added") is not False:
                raise BridgeContractError("v305_measurement_boundary_changed")
            if aggregate.get("public_leaderboard_used") is not False or aggregate.get("competition_submission_attempted") is not False:
                raise BridgeContractError("v305_selection_or_submission_boundary_changed")
            if aggregate.get("aggregate_contains_row_level_values") is not False:
                raise BridgeContractError("v305_aggregate_privacy_boundary_changed")

            stage = "write_private_bank"
            manifest = v305.write_v3_05_outputs(aggregate, oof_bank, challenge_bank, staged)
            if sorted(p.name for p in staged.iterdir()) != sorted(declared) or any(p.is_symlink() or not p.is_file() for p in staged.iterdir()):
                raise BridgeContractError("v305_output_allowlist_or_type_mismatch")
            for item in manifest.get("files", []):
                p = staged / item["filename"]
                if p.stat().st_size != item["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
                    raise BridgeContractError("v305_private_bank_hash_mismatch")
            summary_path = staged / "v3_v05_task13_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["runtime"] = {
                "request_id": V305_REQUEST_ID,
                "science_commit": V305_SCIENCE_COMMIT,
                "science_blob": V305_SCIENCE_BLOB,
                "library_versions": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn")},
                "runtime_terminal_marker": "CMI_FLU_V3_V05_RUNTIME_PASS",
            }
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stage = "publish_validated_outputs"
            for name in declared:
                destination = output_dir / name
                with destination.open("xb") as stream:
                    published.append(destination); stream.write((staged / name).read_bytes())
            success = True
        print(f"CMI_FLU_V3_V05_RUNTIME_PASS fit_count={aggregate.get('fit_count')} oof_rows={len(oof_bank)} challenge_rows={len(challenge_bank)} new_conditions=4 task12=data_limited competition_submit=false public_used=false private_bank=true scratch_cleanup=true")
        return 0
    except Exception as exc:
        code = hashlib.sha256(f"{stage}:{type(exc).__name__}:{exc}".encode()).hexdigest()[:20]
        print(f"CMI_FLU_V3_V05_RUNTIME_FAIL stage={stage} type={type(exc).__name__} code={code}", file=sys.stderr)
        return 1
    finally:
        sys.path[:] = previous_path
        if not success:
            for path in reversed(published): path.unlink(missing_ok=True)
'''
    raw = replace_function(raw, "execute", execution)

    self_test = r'''def self_test() -> int:
    import ast, sys, tempfile
    if git_blob_sha(V305_SOURCE.encode()) != V305_SCIENCE_BLOB:
        raise BridgeContractError("v305_blob_mismatch")
    tree = ast.parse(V305_SOURCE)
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and "submit" in n.func.attr.casefold() for n in ast.walk(tree)):
        raise BridgeContractError("v305_submit_call_present")
    with tempfile.TemporaryDirectory(prefix="v305-selftest-") as tmp:
        package = Path(tmp) / "cmi_flu_bundle.zip"; package.write_bytes(package_bytes()); sys.path.insert(0, str(package))
        try:
            module = load_v3_v05_module()
            if module.EXPERIMENT != "strategy_v3_v05_task13_absolute_scale_heads" or module.NEW_CONDITIONS != ("S1", "S2", "S3", "S4") or module.MAX_FITS != 96:
                raise BridgeContractError("v305_frozen_contract")
        finally:
            if sys.path and sys.path[0] == str(package): sys.path.pop(0)
    print(f"CMI_FLU_V3_V05_SELF_TEST PASS request_id={V305_REQUEST_ID} science_commit={V305_SCIENCE_COMMIT} blob={V305_SCIENCE_BLOB} conditions=4 fit_limit=96 frozen_api_compat=true submission=false")
    return 0
'''
    raw = replace_function(raw, "self_test", self_test)

    entry = r'''def main() -> int:
    import argparse, hashlib, sys
    parser = argparse.ArgumentParser(description="CMI-Flu Strategy-v3 V3-05 Task1.3 scale heads")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    if args.self_test: return self_test()
    try:
        search_root = args.input_dir.resolve()
        required = ("participants.tsv", "investigations_260821.tsv", "publicData_cytokine.tsv", "publicData_ex_vivo_flow.tsv", "publicData_serology_260821.tsv", "2025LJI_aim.tsv", "2025LJI_cytokine.tsv", "2025LJI_ex_vivo_flow.tsv", "2025LJI_serology.tsv", "sample_submission_part1.csv", "md5sum")
        roots = {p.parent for p in search_root.rglob("sample_submission_part1.csv") if all((p.parent / n).is_file() for n in required)}
        if len(roots) != 1:
            raise BridgeContractError("v305_exact_competition_root_missing_or_ambiguous")
        return execute(next(iter(roots)), args.output_dir.resolve())
    except Exception as exc:
        stage = "locate_competition_input"; code = hashlib.sha256(f"{stage}:{type(exc).__name__}:{exc}".encode()).hexdigest()[:20]
        print(f"CMI_FLU_V3_V05_RUNTIME_FAIL stage={stage} type={type(exc).__name__} code={code}", file=sys.stderr); return 1
'''
    raw = replace_function(raw, "main", entry)
    compile(raw, "v3-v05-runtime.py", "exec")
    if len(raw.encode()) >= 1100000:
        raise SystemExit("V3-05 runtime exceeds reviewed byte budget")
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(raw, encoding="utf-8")
    return raw


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    raw = build_runtime(a.repository_root.resolve(), a.output.resolve()).encode()
    print(f"CMI_FLU_V3_V05_PREPARE PASS bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()} science_commit={SCIENCE_COMMIT} science_blob={SCIENCE_BLOB} conditions=4 submission=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
