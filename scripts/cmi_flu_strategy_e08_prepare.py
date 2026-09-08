#!/usr/bin/env python3
"""Build the exact aggregate-only E08 Task1.3 applicability runtime."""
from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REQUEST_ID = "20260909-cmi-flu-strategy-e08-task13-applicability-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-e08-task13-applicability-20260909-001"
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_COMMIT = "1343cb8a0aefc41a4da0ec98d8c606d61c13088d"
REQUEST_PATH = "requests/cmi-flu-strategy-e08-task13-applicability-001.json"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e08-task13-applicability-001"
E08_PATH = f"{PAYLOAD_ROOT}/strategy_e08.py"
E08_V2_PATH = f"{PAYLOAD_ROOT}/strategy_e08_v2.py"
E08_BLOB = "c4afc93e4eaf5eb2784720ba8bb9215de9f43374"
E08_V2_BLOB = "e5542251f94f011f0e99b83d1c1b704efa1ea213"
CONTRACT_BLOB = "3982541febfb4641fbf895438ae1a465bdbb3d5e"
SYNTHETIC_BLOB = "436b6a971622915cc5b335060f9a850feb6108fd"
TASK13_BLOB = "5c6725dc757a5ba9dd21289b1c4f09997e1afdb8"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
E04_BUILDER = "scripts/cmi_flu_strategy_e04_prepare.py"
PINNED_BLOBS = {
    "src/cmi_flu/strategy_e08.py": E08_BLOB,
    "src/cmi_flu/strategy_e08_v2.py": E08_V2_BLOB,
    "src/cmi_flu/strategy_e04_contract.py": CONTRACT_BLOB,
    "src/cmi_flu/strategy_e04_synthetic.py": SYNTHETIC_BLOB,
    "src/cmi_flu/task13_harmonization.py": TASK13_BLOB,
    "configs/baseline_b021_robust.yaml": CONFIG_BLOB,
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def require_blob(path: Path, expected: str, label: str) -> str:
    data = path.read_bytes()
    found = git_blob_sha(data)
    if found != expected:
        raise SystemExit(f"{label} relay blob mismatch:{found}")
    text = data.decode("utf-8")
    compile(text, label, "exec")
    return text


def load_e04_builder(root: Path):
    path = root / E04_BUILDER
    spec = importlib.util.spec_from_file_location("cmi_flu_e08_e04_builder", path)
    if spec is None or spec.loader is None:
        raise SystemExit("E04 base builder unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        "SCIENCE_COMMIT": "6e4d786cddc7b2d02dc4671d1c005db551d19ad6",
        "CONTRACT_BLOB": CONTRACT_BLOB,
        "SYNTHETIC_BLOB": SYNTHETIC_BLOB,
        "TASK13_BLOB": TASK13_BLOB,
        "CONFIG_BLOB": CONFIG_BLOB,
    }
    for key, value in expected.items():
        if getattr(module, key, None) != value:
            raise SystemExit(f"E08 inherited E04 builder contract changed:{key}")
    return module


def validate_request(root: Path) -> None:
    request = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": COMPETITION,
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_repository": SCIENCE_REPOSITORY,
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": "agent_relay_exact_blobs",
        "expected_kernel_version": 1,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"E08 request mismatch:{key}")
    if request.get("pinned_git_blobs") != PINNED_BLOBS:
        raise SystemExit("E08 pinned blob contract mismatch")
    if request.get("resource") != {"accelerator": "cpu", "expected_runtime_minutes": 15, "hard_timeout_minutes": 45, "max_active_runs": 1}:
        raise SystemExit("E08 resource contract mismatch")
    if request.get("allowed_output_paths") != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E08 output allowlist mismatch")
    contract = request.get("experiment_contract") or {}
    if contract.get("experiment") != "strategy_v2_e08_task13_applicability_audit":
        raise SystemExit("E08 experiment contract mismatch")
    if contract.get("compatibility_levels") != ["exact_e04", "material_plural_only", "category_non_gate", "marker_category_upper_bound"]:
        raise SystemExit("E08 compatibility ladder changed")
    if contract.get("minimum_bridge_subjects") != 8 or contract.get("pairing_implementation") != "participant_unique_gate_v2":
        raise SystemExit("E08 threshold/pairing contract changed")


def replace_block(text: str, start: str, end: str, replacement: str) -> str:
    i = text.find(start)
    if i < 0:
        raise SystemExit(f"E08 runtime patch start missing:{start[:50]}")
    j = text.find(end, i + len(start))
    if j < 0:
        raise SystemExit(f"E08 runtime patch end missing:{end[:50]}")
    return text[:i] + replacement.rstrip() + "\n\n" + text[j:]


def apply_e04_v2_runtime_compat(runtime: str) -> str:
    # E04 v2 established these compatibility repairs for the frozen B2.1
    # transport. Reapply them before replacing the E04 execution surface.
    runtime = runtime.replace('"\n"', '"\\n"')
    config_tuple = '        (CONFIG_TEXT, CONFIG_BLOB, "config"),\n'
    if runtime.count(config_tuple) != 1:
        raise SystemExit("E08 inherited config self-test anchor changed")
    runtime = runtime.replace(config_tuple, "", 1)
    adapter_anchor = '    compile(B21_ADAPTER_SOURCE, "cmi_flu_b21_runtime_adapter.py", "exec")\n'
    if runtime.count(adapter_anchor) != 1:
        raise SystemExit("E08 inherited adapter self-test anchor changed")
    runtime = runtime.replace(
        adapter_anchor,
        '    if git_blob_sha(CONFIG_TEXT.encode("utf-8")) != CONFIG_BLOB:\n'
        '        raise BridgeContractError("config_blob_mismatch")\n' + adapter_anchor,
        1,
    )
    install_anchor = '        install()\n        from cmi_flu.configuration import load_baseline_config\n'
    if runtime.count(install_anchor) != 1:
        raise SystemExit("E08 inherited B2.1 adapter install anchor changed")
    runtime = runtime.replace(
        install_anchor,
        '        install()\n'
        '        from cmi_flu import runner as _e08_runner\n'
        '        _e08_frozen_robust_compact = _e08_runner.run_compact_task\n'
        '        def _e08_run_compact_task(dataset, *, specs, splits=None, random_state=42, selection_policy="robust_v1"):\n'
        '            if selection_policy != "robust_v1":\n'
        '                raise BundleContractError("E08 requires robust_v1 selection policy")\n'
        '            return _e08_frozen_robust_compact(dataset, specs=specs, splits=splits, random_state=random_state)\n'
        '        _e08_runner.run_compact_task = _e08_run_compact_task\n'
        '        from cmi_flu.configuration import load_baseline_config\n',
        1,
    )
    return runtime


def build_runtime(root: Path, e08: str, e08_v2: str) -> tuple[str, str]:
    e04 = load_e04_builder(root)
    e04_source, contract, synthetic, task13 = e04.load_exact_e04(root)
    base = e04.load_base(root)
    original_compile = builtins.compile

    def compile_guard(source, filename, mode, *positional, **keywords):
        if filename == "generated_e04_runtime.py":
            return original_compile("pass\n", filename, mode, *positional, **keywords)
        return original_compile(source, filename, mode, *positional, **keywords)

    builtins.compile = compile_guard
    try:
        runtime, package_sha = e04.build_runtime(base, root, e04_source, contract, synthetic, task13)
    finally:
        builtins.compile = original_compile
    runtime = apply_e04_v2_runtime_compat(runtime)

    replacements = {
        'REQUEST_ID = "20260908-cmi-flu-strategy-e04-task13-rescue-001"': f'REQUEST_ID = "{REQUEST_ID}"',
        'SCIENCE_COMMIT = "6e4d786cddc7b2d02dc4671d1c005db551d19ad6"': f'SCIENCE_COMMIT = "{SCIENCE_COMMIT}"',
        'TARGET_KERNEL = "renta0426/cmi-flu-e04-task13-rescue-20260908-001"': f'TARGET_KERNEL = "{TARGET_KERNEL}"',
        '"""CMI-Flu strategy E04 Task1.3 gate rescue. Aggregate outputs only; no submission."""': '"""CMI-Flu strategy E08 Task1.3 applicability audit. Aggregate outputs only; no submission."""',
    }
    for old, new in replacements.items():
        if runtime.count(old) != 1:
            raise SystemExit(f"E08 inherited runtime anchor count changed:{old[:50]}")
        runtime = runtime.replace(old, new, 1)

    source_anchor = "REFERENCE_PLACEHOLDERS = "
    pos = runtime.find(source_anchor)
    if pos < 0:
        raise SystemExit("E08 source injection anchor missing")
    injected = (
        f'E08_BLOB = {E08_BLOB!r}\nE08_V2_BLOB = {E08_V2_BLOB!r}\n'
        f'E08_SOURCE = {e08!r}\nE08_V2_SOURCE = {e08_v2!r}\n'
        'MANAGED_EXTERNAL_MANIFEST_TEXT = ""\n'
    )
    runtime = runtime[:pos] + injected + runtime[pos:]

    runtime = replace_block(runtime, "def self_test() -> int:\n", "def locate_competition_data(", '''def self_test() -> int:
    package = package_bytes()
    for source, expected, label in (
        (E08_SOURCE, E08_BLOB, "strategy_e08"),
        (E08_V2_SOURCE, E08_V2_BLOB, "strategy_e08_v2"),
        (CONTRACT_SOURCE, CONTRACT_BLOB, "strategy_e04_contract"),
        (SYNTHETIC_SOURCE, SYNTHETIC_BLOB, "strategy_e04_synthetic"),
        (TASK13_SOURCE, TASK13_BLOB, "task13_harmonization"),
    ):
        if git_blob_sha(source.encode("utf-8")) != expected:
            raise BridgeContractError(f"{label}_blob_mismatch")
        compile(source, f"cmi_flu/{label}.py", "exec")
    if git_blob_sha(CONFIG_TEXT.encode("utf-8")) != CONFIG_BLOB:
        raise BridgeContractError("config_blob_mismatch")
    compile(B21_ADAPTER_SOURCE, "cmi_flu_b21_runtime_adapter.py", "exec")
    print(
        "CMI_FLU_E08_RUNTIME_SELF_TEST PASS "
        f"request_id={REQUEST_ID} package_bytes={len(package)} package_sha256={PACKAGE_SHA256} "
        f"science_commit={SCIENCE_COMMIT}"
    )
    return 0
''')

    runtime = replace_block(runtime, "def load_e04_modules() -> tuple[object, object]:\n", "def json_safe(value):\n", '''def load_e08_modules() -> tuple[object, object]:
    def install(name: str, source: str) -> object:
        module = types.ModuleType(name)
        module.__file__ = f"<{name}>"
        module.__package__ = "cmi_flu"
        sys.modules[name] = module
        exec(compile(source, name.replace(".", "/") + ".py", "exec"), module.__dict__, module.__dict__)
        return module
    install("cmi_flu.task13_harmonization", TASK13_SOURCE)
    install("cmi_flu.strategy_e04_contract", CONTRACT_SOURCE)
    install("cmi_flu.strategy_e08", E08_SOURCE)
    e08_v2 = install("cmi_flu.strategy_e08_v2", E08_V2_SOURCE)
    synthetic = install("cmi_flu.strategy_e04_synthetic", SYNTHETIC_SOURCE)
    run_e08 = getattr(e08_v2, "run_e08", None)
    make_tables = getattr(synthetic, "make_e04_tables", None)
    if not callable(run_e08) or not callable(make_tables):
        raise BridgeContractError("e08_entry_missing")
    return run_e08, make_tables
''')

    runtime = replace_block(runtime, "def validate_result(result: dict, *, synthetic: bool) -> None:\n", "def render_summary(result: dict) -> str:\n", '''def validate_result(result: dict, *, synthetic: bool) -> None:
    if result.get("schema_version") != 1 or result.get("experiment") != "strategy_v2_e08_task13_applicability_audit" or result.get("task") != "Task1.3":
        raise BridgeContractError("e08_identity_mismatch")
    if result.get("ontology_version") != "e08_task13_ontology_audit_v1":
        raise BridgeContractError("e08_ontology_version_mismatch")
    if result.get("status") != "complete" or result.get("outcomes_accessed") is not False:
        raise BridgeContractError("e08_status_or_outcome_boundary_mismatch")
    if result.get("leaderboard_used_for_selection") is not False or result.get("competition_submission_attempted") is not False or result.get("automatic_compute_retries") != 0:
        raise BridgeContractError("e08_selection_submission_contract_mismatch")
    if result.get("pairing_implementation") != "participant_unique_gate_v2":
        raise BridgeContractError("e08_pairing_implementation_mismatch")
    levels = ["exact_e04", "material_plural_only", "category_non_gate", "marker_category_upper_bound"]
    if result.get("compatibility_levels") != levels or result.get("minimum_bridge_subjects") != 8:
        raise BridgeContractError("e08_compatibility_contract_mismatch")
    bridge = result.get("bridge_counts") or {}
    if list(bridge) != levels:
        raise BridgeContractError("e08_bridge_level_set_mismatch")
    decision = result.get("decision") or {}
    exact = bool((bridge.get("exact_e04") or {}).get("meets_minimum_bridge"))
    alias = bool((bridge.get("material_plural_only") or {}).get("meets_minimum_bridge"))
    expected_ready = alias and not exact
    if decision.get("e04b_measurement_bridge_ready") is not expected_ready or decision.get("material_plural_alias_alone_recovers_bridge") is not expected_ready:
        raise BridgeContractError("e08_decision_not_derived_from_frozen_ladder")
    inventory = result.get("file_inventory") or {}
    if inventory.get("managed_external_manifest_fcs_mentions") != 0:
        raise BridgeContractError("e08_managed_manifest_fcs_contract_changed")
    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"', '"population_definition"', '"subject_group"', '"row_index"', '"oof_predictions"', '"challenge_predictions"')
    if any(token in serialized for token in banned):
        raise BridgeContractError("e08_aggregate_output_contains_private_fields")
    if synthetic:
        if (bridge.get("exact_e04") or {}).get("2024UGA_compatible_subjects") != 0:
            raise BridgeContractError("e08_synthetic_exact_bridge_changed")
        if (bridge.get("material_plural_only") or {}).get("2024UGA_compatible_subjects") != 15:
            raise BridgeContractError("e08_synthetic_material_alias_bridge_changed")
        if decision.get("e04b_measurement_bridge_ready") is not True:
            raise BridgeContractError("e08_synthetic_decision_changed")
''')

    runtime = replace_block(runtime, "def render_summary(result: dict) -> str:\n", "def sha256_file(", '''def render_summary(result: dict) -> str:
    bridge = result.get("bridge_counts") or {}
    decision = result.get("decision") or {}
    inventory = result.get("file_inventory") or {}
    proxy = result.get("sdy272_proxy_intrinsic") or {}
    lines = [
        "# CMI-Flu strategy E08 Task1.3 applicability audit", "",
        "Outcome-independent aggregate-only measurement audit; no Competition submission was attempted.", "",
        f"- science commit: `{SCIENCE_COMMIT}`",
        f"- status: `{result.get('status')}`",
        f"- pairing implementation: `{result.get('pairing_implementation')}`",
        f"- E04b measurement bridge ready: `{decision.get('e04b_measurement_bridge_ready')}`",
        f"- third strict-like domain found: `{decision.get('third_strict_like_domain_found')}`",
        f"- raw FCS available in managed inputs: `{decision.get('raw_fcs_available_in_managed_inputs')}`",
        f"- Competition input file count: `{inventory.get('competition_input_file_count')}`",
        f"- Competition input FCS count: `{inventory.get('competition_input_fcs_count')}`",
        f"- SDY272 paired same-gate subjects: `{proxy.get('paired_same_gate_subjects')}`",
        "",
    ]
    for level in ("exact_e04", "material_plural_only", "category_non_gate", "marker_category_upper_bound"):
        item = bridge.get(level) or {}
        lines.append(f"- {level}: compatible_2024UGA={item.get('2024UGA_compatible_subjects')} minimum={item.get('meets_minimum_bridge')}")
    return "\n".join(lines) + "\n"
''')

    runtime = replace_block(runtime, "def execute(input_dir: Path | None, output_dir: Path, *, synthetic: bool = False) -> int:\n", "def main() -> int:\n", '''def execute(input_dir: Path | None, output_dir: Path, *, synthetic: bool = False) -> int:
    runtime_root = output_dir / "e08-runtime"
    stage = "initialize"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        runtime_root.mkdir()
        stage = "materialize_package"
        package_path = runtime_root / "cmi_flu_bundle.zip"
        package_path.write_bytes(package_bytes())
        sys.path.insert(0, str(package_path))
        stage = "prepare_tree"
        inventory = []
        if not synthetic:
            if input_dir is None:
                raise BridgeContractError("production_input_missing")
            data_parent = runtime_root / "data"
            data_parent.mkdir(parents=True)
            (data_parent / "raw").symlink_to(input_dir, target_is_directory=True)
            vaccine_n, challenge_n = derive_reference_files(input_dir, runtime_root / "external")
            if (vaccine_n, challenge_n) != (3, 12):
                raise BridgeContractError("panel_count_mismatch")
            inventory = sorted(str(path.relative_to(input_dir)) for path in input_dir.rglob("*") if path.is_file())
        config_dir = runtime_root / "configs"
        config_dir.mkdir()
        canonical_config = config_dir / "baseline_b021_robust.yaml"
        canonical_config.write_text(CONFIG_TEXT, encoding="utf-8")
        if git_blob_sha(canonical_config.read_bytes()) != CONFIG_BLOB:
            raise BridgeContractError("runtime_config_blob_mismatch")
        compat = config_dir / "baseline_b02_transport_compat.yaml"
        if CONFIG_TEXT.count("baseline: b021_taskwise_robust") != 1:
            raise BridgeContractError("b021_config_anchor_changed")
        compat.write_text(CONFIG_TEXT.replace("baseline: b021_taskwise_robust", "baseline: b02_taskwise_compact", 1), encoding="utf-8")
        stage = "install_b21_adapter"
        adapter_ns = {}
        exec(compile(B21_ADAPTER_SOURCE, "<b21_runtime_adapter>", "exec"), adapter_ns, adapter_ns)
        install = adapter_ns.get("install")
        if not callable(install):
            raise BridgeContractError("b21_adapter_install_missing")
        install()
        from cmi_flu.configuration import load_baseline_config
        config = load_baseline_config(compat, repository_root=runtime_root)
        raw = dict(config.raw); raw["baseline"] = "b021_taskwise_robust"
        config = replace(config, source_path=canonical_config, raw=raw, baseline="b021_taskwise_robust")
        if not config.verify_md5 or str(config.section("selection").get("policy", "")) != "robust_v1":
            raise BridgeContractError("runtime_config_contract_mismatch")
        stage = "load_e08"
        run_e08, make_tables = load_e08_modules()
        md5_verified = None
        if synthetic:
            from types import SimpleNamespace
            tables = make_tables()
            tables["challenge_flow"] = tables["challenge_flow"].copy()
            tables["challenge_flow"]["material"] = "PBMCs"
            inputs = SimpleNamespace(tables=tables)
            inventory = ["publicData_ex_vivo_flow.tsv", "2025LJI_ex_vivo_flow.tsv"]
        else:
            stage = "load_inputs"
            from cmi_flu.runner import load_inputs
            inputs = load_inputs(config)
            if inputs.checksum_report is None:
                raise BridgeContractError("md5_verification_missing")
            md5_verified = len(inputs.checksum_report.verified)
        stage = "run_e08"
        result = run_e08(inputs, file_inventory=inventory, external_manifest_text=MANAGED_EXTERNAL_MANIFEST_TEXT)
        result = json_safe(dict(result))
        stage = "validate_e08"
        validate_result(result, synthetic=synthetic)
        stage = "write_outputs"
        metrics_path = output_dir / "metrics.json"
        summary_path = output_dir / "summary.md"
        bridge_path = output_dir / "bridge-result.json"
        metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        summary_path.write_text(render_summary(result), encoding="utf-8")
        bridge = {
            "schema_version": 1,
            "request_id": REQUEST_ID,
            "competition": COMPETITION,
            "target_kernel": TARGET_KERNEL,
            "science_commit": SCIENCE_COMMIT,
            "strategy_e08_blob_sha": E08_BLOB,
            "strategy_e08_v2_blob_sha": E08_V2_BLOB,
            "strategy_e04_contract_blob_sha": CONTRACT_BLOB,
            "strategy_e04_synthetic_blob_sha": SYNTHETIC_BLOB,
            "task13_harmonization_blob_sha": TASK13_BLOB,
            "config_blob_sha": CONFIG_BLOB,
            "package_sha256": PACKAGE_SHA256,
            "python_version": platform.python_version(),
            "md5_verified_count": md5_verified,
            "metrics_sha256": sha256_file(metrics_path),
            "summary_sha256": sha256_file(summary_path),
            "competition_submission_attempted": False,
            "leaderboard_used_for_selection": False,
            "contains_participant_identifiers": False,
            "contains_row_level_predictions": False,
            "outcomes_accessed": False,
            "synthetic": bool(synthetic),
        }
        bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.rmtree(runtime_root, ignore_errors=True)
        print(f"CMI_FLU_E08_COMPLETE status={result.get('status')} synthetic={synthetic} md5_verified={md5_verified} submission=false outcomes=false")
        return 0
    except Exception as exc:
        shutil.rmtree(runtime_root, ignore_errors=True)
        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode("utf-8", errors="replace")).hexdigest()[:20]
        detail = f" detail={str(exc)}" if synthetic else ""
        print(f"CMI_FLU_E08_FAILED stage={stage} exception_type={type(exc).__name__} error_code={code}{detail}", file=sys.stderr)
        return 2
''')

    runtime = replace_block(runtime, "def main() -> int:\n", "if __name__ == \"__main__\":\n", '''def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.synthetic:
        return execute(None, args.output_dir, synthetic=True)
    try:
        input_dir = locate_competition_data(args.input_dir)
    except Exception as exc:
        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode()).hexdigest()[:20]
        print(f"CMI_FLU_E08_FAILED stage=locate_competition_data exception_type={type(exc).__name__} error_code={code}", file=sys.stderr)
        return 2
    return execute(input_dir, args.output_dir, synthetic=False)
''')
    original_compile(runtime, "generated_e08_runtime.py", "exec")
    return runtime, package_sha


def main() -> int:
    a = args()
    root = a.repository_root.expanduser().resolve()
    output = a.output.expanduser().resolve()
    validate_request(root)
    e08 = require_blob(root / E08_PATH, E08_BLOB, "cmi_flu/strategy_e08.py")
    e08_v2 = require_blob(root / E08_V2_PATH, E08_V2_BLOB, "cmi_flu/strategy_e08_v2.py")
    required = (
        'EXPERIMENT = "strategy_v2_e08_task13_applicability_audit"',
        'ONTOLOGY_VERSION = "e08_task13_ontology_audit_v1"',
        'MIN_BRIDGE_SUBJECTS = 8',
        'PAIRING_IMPLEMENTATION = "participant_unique_gate_v2"',
    )
    joined = e08 + e08_v2
    if any(token not in joined for token in required):
        raise SystemExit("E08 exact source contract token missing")
    if "kaggle competitions submit" in joined or "competition_submit(" in joined:
        raise SystemExit("E08 relayed source contains submission path")
    runtime, package_sha = build_runtime(root, e08, e08_v2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_E08_PREPARE_PASS "
        f"science_commit={SCIENCE_COMMIT} e08_blob={E08_BLOB} e08_v2_blob={E08_V2_BLOB} "
        f"package_sha256={package_sha} runtime_sha256={sha256(runtime.encode())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
