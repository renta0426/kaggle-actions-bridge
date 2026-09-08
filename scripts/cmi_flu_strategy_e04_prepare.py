#!/usr/bin/env python3
"""Build the exact aggregate-only E04 Task1.3 runtime from relayed Git blobs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

REQUEST_ID = "20260908-cmi-flu-strategy-e04-task13-rescue-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-e04-task13-rescue-20260908-001"
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_COMMIT = "6e4d786cddc7b2d02dc4671d1c005db551d19ad6"
REQUEST_PATH = "requests/cmi-flu-strategy-e04-task13-rescue-001.json"
BASE_BUILDER = "scripts/cmi_flu_strategy_e01_prepare.py"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e04-task13-rescue-001"
E04_PATH = f"{PAYLOAD_ROOT}/strategy_e04.py"
CONTRACT_PATH = f"{PAYLOAD_ROOT}/strategy_e04_contract.py"
SYNTHETIC_PATH = f"{PAYLOAD_ROOT}/strategy_e04_synthetic.py"
TASK13_PATH = f"{PAYLOAD_ROOT}/task13_harmonization.py"
E04_BLOB = "73c112a8bb3b1ee7a6dd5bfbd5645a945bea6ebb"
CONTRACT_BLOB = "3982541febfb4641fbf895438ae1a465bdbb3d5e"
SYNTHETIC_BLOB = "436b6a971622915cc5b335060f9a850feb6108fd"
TASK13_BLOB = "5c6725dc757a5ba9dd21289b1c4f09997e1afdb8"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
PINNED_BLOBS = {
    "src/cmi_flu/strategy_e04.py": E04_BLOB,
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


def load_base(root: Path) -> ModuleType:
    path = root / BASE_BUILDER
    spec = importlib.util.spec_from_file_location("cmi_flu_e04_base_builder", path)
    if spec is None or spec.loader is None:
        raise SystemExit("E04 base builder unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        "SCIENCE_COMMIT": "0b2ecb47eaa09f22450424c9c06dc88cf44bc1fb",
        "E01_BLOB": "dd27aea0cf97d41bad3cec64819c4c4269d94cbd",
        "E01_V2_BLOB": "8cc64dc5ab9483d5957cfada18d445188566c56c",
        "CONFIG_BLOB": CONFIG_BLOB,
    }
    for key, value in expected.items():
        if getattr(module, key, None) != value:
            raise SystemExit(f"E04 frozen B2.1 builder contract changed:{key}")
    return module


def load_exact_e04(root: Path) -> tuple[str, str, str, str]:
    e04 = require_blob(root / E04_PATH, E04_BLOB, "cmi_flu/strategy_e04.py")
    contract = require_blob(root / CONTRACT_PATH, CONTRACT_BLOB, "cmi_flu/strategy_e04_contract.py")
    synthetic = require_blob(root / SYNTHETIC_PATH, SYNTHETIC_BLOB, "cmi_flu/strategy_e04_synthetic.py")
    task13 = require_blob(root / TASK13_PATH, TASK13_BLOB, "cmi_flu/task13_harmonization.py")
    required = (
        'EXPERIMENT = "strategy_v2_e04_task13_anchor_preserving_rescue"',
        'FROZEN_STRICT_MODEL = "pls_1"',
        'FROZEN_HISTORICAL_MODEL = "enet_a0.1_l0.5"',
        'ALPHA = 10.0',
        'SHRINKAGE = 0.25',
        'MAX_CORRECTION = 0.05',
        'PROXY_WEIGHTS = (0.25, 1.0)',
        'strict_reconstructed"] = False',
    )
    joined = e04 + contract
    if any(token not in joined for token in required):
        raise SystemExit("E04 exact source contract token missing")
    for source in (e04, contract, synthetic, task13):
        if "kaggle competitions submit" in source or "competition_submit(" in source:
            raise SystemExit("E04 relayed source contains submission path")
    return e04, contract, synthetic, task13


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
            raise SystemExit(f"E04 request mismatch:{key}")
    if request.get("pinned_git_blobs") != PINNED_BLOBS:
        raise SystemExit("E04 pinned blob contract mismatch")
    if request.get("resource") != {"accelerator": "cpu", "expected_runtime_minutes": 30, "hard_timeout_minutes": 60, "max_active_runs": 1}:
        raise SystemExit("E04 resource contract mismatch")
    if request.get("allowed_output_paths") != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E04 output allowlist mismatch")


def replace_block(text: str, start: str, end: str, replacement: str) -> str:
    i = text.find(start)
    if i < 0:
        raise SystemExit(f"E04 runtime patch start missing:{start[:40]}")
    j = text.find(end, i + len(start))
    if j < 0:
        raise SystemExit(f"E04 runtime patch end missing:{end[:40]}")
    return text[:i] + replacement.rstrip() + "\n\n" + text[j:]


def build_runtime(base: ModuleType, root: Path, e04: str, contract: str, synthetic: str, task13: str) -> tuple[str, str]:
    # Reuse the already-proven B2.1 package/adapter path from E01, but replace
    # the experiment module and aggregate execution surface with exact E04 blobs.
    e01, e01_v2, config = base.load_exact_science(root)
    with tempfile.TemporaryDirectory(prefix="cmi-e04-build-") as tmp:
        package, adapter = base.extract_frozen_runtime(root, Path(tmp))
        runtime = base.build_runtime(package, adapter, e01, e01_v2, config)
    replacements = {
        f'REQUEST_ID = "{base.REQUEST_ID}"': f'REQUEST_ID = "{REQUEST_ID}"',
        f'SCIENCE_COMMIT = "{base.SCIENCE_COMMIT}"': f'SCIENCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'TARGET_KERNEL = "{base.TARGET_KERNEL}"': f'TARGET_KERNEL = "{TARGET_KERNEL}"',
        '"""CMI-Flu strategy E01 paired evaluation. Aggregate outputs only; no submission."""': '"""CMI-Flu strategy E04 Task1.3 gate rescue. Aggregate outputs only; no submission."""',
    }
    for old, new in replacements.items():
        if runtime.count(old) != 1:
            raise SystemExit(f"E04 base runtime anchor count changed:{old[:40]}")
        runtime = runtime.replace(old, new, 1)
    source_anchor = "REFERENCE_PLACEHOLDERS = "
    pos = runtime.find(source_anchor)
    if pos < 0:
        raise SystemExit("E04 source injection anchor missing")
    injected = (
        f'E04_BLOB = {E04_BLOB!r}\nCONTRACT_BLOB = {CONTRACT_BLOB!r}\nSYNTHETIC_BLOB = {SYNTHETIC_BLOB!r}\nTASK13_BLOB = {TASK13_BLOB!r}\n'
        f'E04_SOURCE = {e04!r}\nCONTRACT_SOURCE = {contract!r}\nSYNTHETIC_SOURCE = {synthetic!r}\nTASK13_SOURCE = {task13!r}\n'
    )
    runtime = runtime[:pos] + injected + runtime[pos:]
    parse_anchor = '    p.add_argument("--self-test", action="store_true")\n'
    if runtime.count(parse_anchor) != 1:
        raise SystemExit("E04 parse_args anchor changed")
    runtime = runtime.replace(parse_anchor, parse_anchor + '    p.add_argument("--synthetic", action="store_true")\n', 1)

    runtime = replace_block(runtime, "def self_test() -> int:\n", "def locate_competition_data(", '''def self_test() -> int:
    package = package_bytes()
    for source, expected, label in (
        (E04_SOURCE, E04_BLOB, "e04"),
        (CONTRACT_SOURCE, CONTRACT_BLOB, "e04_contract"),
        (SYNTHETIC_SOURCE, SYNTHETIC_BLOB, "e04_synthetic"),
        (TASK13_SOURCE, TASK13_BLOB, "task13_harmonization"),
        (CONFIG_TEXT, CONFIG_BLOB, "config"),
    ):
        if git_blob_sha(source.encode("utf-8")) != expected:
            raise BridgeContractError(f"{label}_blob_mismatch")
        compile(source, f"cmi_flu/{label}.py", "exec")
    compile(B21_ADAPTER_SOURCE, "cmi_flu_b21_runtime_adapter.py", "exec")
    print(
        "CMI_FLU_E04_RUNTIME_SELF_TEST PASS "
        f"request_id={REQUEST_ID} package_bytes={len(package)} package_sha256={PACKAGE_SHA256} "
        f"science_commit={SCIENCE_COMMIT}"
    )
    return 0
''')

    runtime = replace_block(runtime, "def load_e01_module() -> object:\n", "def json_safe(value):\n", '''def load_e04_modules() -> tuple[object, object]:
    def install(name: str, source: str) -> object:
        module = types.ModuleType(name)
        module.__file__ = f"<{name}>"
        module.__package__ = "cmi_flu"
        sys.modules[name] = module
        exec(compile(source, name.replace(".", "/") + ".py", "exec"), module.__dict__, module.__dict__)
        return module
    install("cmi_flu.task13_harmonization", TASK13_SOURCE)
    install("cmi_flu.strategy_e04_contract", CONTRACT_SOURCE)
    e04 = install("cmi_flu.strategy_e04", E04_SOURCE)
    synthetic = install("cmi_flu.strategy_e04_synthetic", SYNTHETIC_SOURCE)
    run_e04 = getattr(e04, "run_e04", None)
    run_synthetic = getattr(synthetic, "run_synthetic", None)
    if not callable(run_e04) or not callable(run_synthetic):
        raise BridgeContractError("e04_entry_missing")
    return run_e04, run_synthetic
''')

    runtime = replace_block(runtime, "def validate_result(result: dict) -> None:\n", "def metric_value(", '''def validate_result(result: dict, *, synthetic: bool) -> None:
    if result.get("schema_version") != 1 or result.get("experiment") != "strategy_v2_e04_task13_anchor_preserving_rescue" or result.get("task") != "Task1.3":
        raise BridgeContractError("experiment_identity_mismatch")
    if result.get("cv_contract") != "paired_subject_purged_v2":
        raise BridgeContractError("cv_contract_mismatch")
    if result.get("competition_submission_attempted") is not False or result.get("leaderboard_used_for_selection") is not False or result.get("automatic_compute_retries") != 0:
        raise BridgeContractError("selection_submission_contract_mismatch")
    if result.get("incumbent_changed") is not False:
        raise BridgeContractError("automatic_incumbent_change_forbidden")
    expected_frozen = {
        "strict_b21": "pls_1",
        "anchor": "flow_rank__Antibody-secreting_cells_(ASC)",
        "historical_rank": "enet_a0.1_l0.5",
        "residual_alpha": 10.0,
        "residual_shrinkage": 0.25,
        "residual_score_cap": 0.05,
        "proxy_weights": [0.25, 1.0],
    }
    if result.get("frozen_conditions") != expected_frozen:
        raise BridgeContractError("frozen_condition_mismatch")
    audit = result.get("measurement_audit") or {}
    if audit.get("challenge_D7_observed") is not False or audit.get("strict_gate_reconstructed") is not False or audit.get("marker_not_reported_is_negative") is not False or audit.get("cross_study_gate_equivalence_claim") is not False:
        raise BridgeContractError("measurement_boundary_mismatch")
    for group in ("strict_cv", "historical_loso", "strict_cv_plus_proxy"):
        if set((result.get(group) or {})) != {"proxy_weight_0.25", "proxy_weight_1"}:
            raise BridgeContractError(f"condition_set_mismatch_{group}")
    if result.get("status") not in {"complete", "reference_mismatch_review_required"}:
        raise BridgeContractError("status_mismatch")
    frozen = result.get("frozen_control_reproduction") or {}
    if synthetic:
        if frozen.get("status") != "synthetic_not_real_data" or frozen.get("all_pass") is not None:
            raise BridgeContractError("synthetic_frozen_control_contract_mismatch")
    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"', '"subject_group"', '"row_index"', '"oof_predictions"', '"challenge_predictions"')
    if any(token in serialized for token in banned):
        raise BridgeContractError("aggregate_output_contains_row_fields")
''')

    runtime = replace_block(runtime, "def metric_value(", "def sha256_file(", '''def render_summary(result: dict) -> str:
    lines = [
        "# CMI-Flu strategy E04 Task1.3 gate rescue", "",
        "Aggregate-only fixed-condition evaluation; no Competition submission was attempted.", "",
        f"- science commit: `{SCIENCE_COMMIT}`",
        f"- status: `{result.get('status')}`",
        f"- incumbent changed: `{result.get('incumbent_changed')}`",
        "",
    ]
    for label in ("strict_reference", "strict_anchor", "historical_rank_research_control", "availability_only_legacy"):
        item = result.get(label) or {}
        lines.append(f"- {label}: study_equal_spearman={item.get('study_equal_spearman')}")
    for group in ("strict_cv", "historical_loso", "strict_cv_plus_proxy"):
        for name, item in (result.get(group) or {}).items():
            metrics = item.get("metrics") or {}
            pa = item.get("paired_vs_anchor") or {}
            pf = item.get("paired_vs_frozen_reference") or {}
            lines.append(f"- {group}/{name}: study_equal={metrics.get('study_equal_spearman')} delta_anchor={pa.get('paired_study_mean_delta')} delta_frozen={pf.get('paired_study_mean_delta')}")
    for name, item in (result.get("challenge") or {}).items():
        if name.startswith("proxy_weight_"):
            lines.append(f"- challenge/{name}: decision={item.get('decision')} bridge_sources={item.get('strict_bridge_source_subjects')}")
    return "\n".join(lines) + "\n"
''')

    runtime = replace_block(runtime, "def execute(input_dir: Path, output_dir: Path) -> int:\n", "def main() -> int:\n", '''def execute(input_dir: Path | None, output_dir: Path, *, synthetic: bool = False) -> int:
    runtime_root = output_dir / "e04-runtime"
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
        if not synthetic:
            if input_dir is None:
                raise BridgeContractError("production_input_missing")
            data_parent = runtime_root / "data"
            data_parent.mkdir(parents=True)
            (data_parent / "raw").symlink_to(input_dir, target_is_directory=True)
            vaccine_n, challenge_n = derive_reference_files(input_dir, runtime_root / "external")
            if (vaccine_n, challenge_n) != (3, 12):
                raise BridgeContractError("panel_count_mismatch")
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
        stage = "load_e04"
        run_e04, run_synthetic = load_e04_modules()
        md5_verified = None
        if synthetic:
            stage = "run_synthetic"
            result, _ = run_synthetic(config)
        else:
            stage = "load_inputs"
            from cmi_flu.runner import load_inputs
            inputs = load_inputs(config)
            if inputs.checksum_report is None:
                raise BridgeContractError("md5_verification_missing")
            md5_verified = len(inputs.checksum_report.verified)
            stage = "run_e04"
            result = run_e04(config, inputs)
        result = json_safe(dict(result))
        stage = "validate_e04"
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
            "strategy_e04_blob_sha": E04_BLOB,
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
            "synthetic": bool(synthetic),
        }
        bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.rmtree(runtime_root, ignore_errors=True)
        print(f"CMI_FLU_E04_COMPLETE status={result.get('status')} synthetic={synthetic} md5_verified={md5_verified} submission=false")
        return 0
    except Exception as exc:
        shutil.rmtree(runtime_root, ignore_errors=True)
        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode("utf-8", errors="replace")).hexdigest()[:20]
        print(f"CMI_FLU_E04_FAILED stage={stage} exception_type={type(exc).__name__} error_code={code}", file=sys.stderr)
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
        print(f"CMI_FLU_E04_FAILED stage=locate_competition_data exception_type={type(exc).__name__} error_code={code}", file=sys.stderr)
        return 2
    return execute(input_dir, args.output_dir, synthetic=False)
''')
    compile(runtime, "generated_e04_runtime.py", "exec")
    return runtime, sha256(package)


def main() -> int:
    a = args()
    root = a.repository_root.expanduser().resolve()
    output = a.output.expanduser().resolve()
    validate_request(root)
    e04, contract, synthetic, task13 = load_exact_e04(root)
    base = load_base(root)
    runtime, package_sha = build_runtime(base, root, e04, contract, synthetic, task13)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_E04_PREPARE_PASS "
        f"science_commit={SCIENCE_COMMIT} e04_blob={E04_BLOB} contract_blob={CONTRACT_BLOB} "
        f"task13_blob={TASK13_BLOB} package_sha256={package_sha} runtime_sha256={sha256(runtime.encode())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
