#!/usr/bin/env python3
"""Build the one-shot aggregate-only Kaggle runtime for CMI-Flu strategy E02."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REQUEST_ID = "20260907-cmi-flu-strategy-e02-task11-structured-logfc-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-e02-task11-structured-logfc-20260907-001"
SCIENCE_COMMIT = "0d399f89f35145e99fafd10d3061d9295cd749bb"
E01_BLOB = "dd27aea0cf97d41bad3cec64819c4c4269d94cbd"
E01_V2_BLOB = "8cc64dc5ab9483d5957cfada18d445188566c56c"
E02_BLOB = "e6aaba7450a527de1e75ad6e7c0d9ef7e0031d7e"
E02_V2_BLOB = "afdfe373a64444ef10dffcc6f7d9d65a8dd7b27c"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
BASE_BUILDER = "scripts/cmi_flu_strategy_e01_prepare.py"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e02-task11-structured-logfc-001"
E02_PARTS = tuple(f"{PAYLOAD_ROOT}/strategy_e02.part{i:02d}" for i in range(8))
E02_V2_PATH = f"{PAYLOAD_ROOT}/strategy_e02_v2.py"
REQUEST_PATH = "requests/cmi-flu-strategy-e02-task11-structured-logfc-001.json"
COMPAT_MARKER = "e02_frozen_selection_policy_api_compat_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def load_base(root: Path):
    path = root / BASE_BUILDER
    spec = importlib.util.spec_from_file_location("cmi_flu_e01_prepare_for_e02", path)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load frozen E01 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        "SCIENCE_COMMIT": "0b2ecb47eaa09f22450424c9c06dc88cf44bc1fb",
        "E01_BLOB": E01_BLOB,
        "E01_V2_BLOB": E01_V2_BLOB,
        "CONFIG_BLOB": CONFIG_BLOB,
    }
    for key, value in expected.items():
        if getattr(module, key, None) != value:
            raise SystemExit(f"frozen E01 builder contract changed: {key}")
    return module


def load_e02(root: Path) -> tuple[str, str]:
    raw = b"".join((root / part).read_bytes() for part in E02_PARTS)
    if git_blob_sha(raw) != E02_BLOB:
        raise SystemExit(f"strategy E02 relay blob mismatch: {git_blob_sha(raw)}")
    raw_v2 = (root / E02_V2_PATH).read_bytes()
    if git_blob_sha(raw_v2) != E02_V2_BLOB:
        raise SystemExit(f"strategy E02 v2 relay blob mismatch: {git_blob_sha(raw_v2)}")
    e02 = raw.decode("utf-8")
    e02_v2 = raw_v2.decode("utf-8")
    compile(e02, "cmi_flu/strategy_e02.py", "exec")
    compile(e02_v2, "cmi_flu/strategy_e02_v2.py", "exec")
    required = (
        'EXPERIMENT = "strategy_v2_e02_task11_structured_logfc"',
        "RIDGE_ALPHA = 10.0",
        "ANCHOR_RESIDUAL_LAMBDA = 0.25",
        'B21_MODEL = "pls_2"',
        '"competition_submission_attempted": False',
        '"leaderboard_used_for_selection": False',
    )
    if any(token not in e02 for token in required):
        raise SystemExit("strategy E02 fixed-condition token missing")
    if "kaggle competitions submit" in e02 or "competition_submit" in e02:
        raise SystemExit("strategy E02 contains submission path")
    if "explicit one-to-one merge" not in e02_v2:
        raise SystemExit("strategy E02 v2 repeat-pairing correction missing")
    return e02, e02_v2


def validate_request(root: Path) -> None:
    request = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": COMPETITION,
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_repository": "renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": "agent_relay_exact_blobs",
        "strategy_e01_blob_sha": E01_BLOB,
        "strategy_e01_v2_blob_sha": E01_V2_BLOB,
        "strategy_e02_blob_sha": E02_BLOB,
        "strategy_e02_v2_blob_sha": E02_V2_BLOB,
        "config_blob_sha": CONFIG_BLOB,
        "expected_kernel_version": 1,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"E02 request mismatch: {key}")
    if request.get("resource") != {"accelerator": "cpu", "expected_runtime_minutes": 30, "hard_timeout_minutes": 60, "max_active_runs": 1}:
        raise SystemExit("E02 resource contract mismatch")
    if request.get("allowed_output_paths") != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E02 output allowlist mismatch")


def replace_function(text: str, start: str, end: str, replacement: str) -> str:
    a = text.find(start)
    b = text.find(end, a + len(start))
    if a < 0 or b < 0 or b <= a:
        raise SystemExit(f"runtime function anchors changed: {start!r} -> {end!r}")
    return text[:a] + replacement.rstrip() + "\n\n" + text[b:]


def patch_runtime(runtime: str, e02: str, e02_v2: str) -> str:
    # Reuse the already audited robust_v1 compatibility shim required by E01.
    anchor = '        install()\n        stage = "load_inputs"\n'
    if runtime.count(anchor) != 1:
        raise SystemExit("E02 runtime install anchor changed")
    compat = '''        install()\n        stage = "install_e02_api_compat"\n        import cmi_flu.evaluation as _e02_evaluation\n        import cmi_flu.runner as _e02_runner\n        _e02_robust_compact = _e02_runner.run_compact_task\n        _e02_robust_hai = _e02_runner.run_hai_compact_for_panels\n\n        def _e02_compact_compat(dataset, *, specs, splits=None, random_state=42, selection_policy="robust_v1"):\n            if selection_policy != "robust_v1":\n                raise BridgeContractError("e02_compact_selection_policy_not_robust_v1")\n            return _e02_robust_compact(dataset, specs=specs, splits=splits, random_state=random_state)\n\n        def _e02_hai_compat(dataset, *, specs, selection_panels, splits=None, selection_policy="robust_v1"):\n            if selection_policy != "robust_v1":\n                raise BridgeContractError("e02_hai_selection_policy_not_robust_v1")\n            return _e02_robust_hai(dataset, specs=specs, selection_panels=selection_panels, splits=splits)\n\n        _e02_evaluation.run_compact_task = _e02_compact_compat\n        _e02_evaluation.run_hai_compact_for_panels = _e02_hai_compat\n        stage = "load_inputs"\n'''
    runtime = runtime.replace(anchor, compat, 1)

    marker = "REFERENCE_PLACEHOLDERS = "
    pos = runtime.find(marker)
    if pos < 0:
        raise SystemExit("E02 source injection anchor missing")
    injected = (
        f'E02_BLOB = "{E02_BLOB}"\n'
        f'E02_V2_BLOB = "{E02_V2_BLOB}"\n'
        f'E02_SOURCE = {e02!r}\n'
        f'E02_V2_SOURCE = {e02_v2!r}\n'
        f'COMPAT_MARKER = "{COMPAT_MARKER}"\n'
    )
    runtime = runtime[:pos] + injected + runtime[pos:]

    runtime = replace_function(
        runtime,
        "def self_test() -> int:\n",
        "def locate_competition_data(",
        '''def self_test() -> int:\n    package = package_bytes()\n    checks = ((E01_SOURCE, E01_BLOB, "e01"), (E01_V2_SOURCE, E01_V2_BLOB, "e01_v2"), (E02_SOURCE, E02_BLOB, "e02"), (E02_V2_SOURCE, E02_V2_BLOB, "e02_v2"), (CONFIG_TEXT, CONFIG_BLOB, "config"))\n    for source, expected, label in checks:\n        if git_blob_sha(source.encode("utf-8")) != expected:\n            raise BridgeContractError(f"{label}_blob_mismatch")\n    compile(B21_ADAPTER_SOURCE, "cmi_flu_b21_runtime_adapter.py", "exec")\n    compile(E01_SOURCE, "cmi_flu/strategy_e01.py", "exec")\n    compile(E01_V2_SOURCE, "cmi_flu/strategy_e01_v2.py", "exec")\n    compile(E02_SOURCE, "cmi_flu/strategy_e02.py", "exec")\n    compile(E02_V2_SOURCE, "cmi_flu/strategy_e02_v2.py", "exec")\n    print(f"CMI_FLU_E02_RUNTIME_SELF_TEST PASS request_id={REQUEST_ID} package_bytes={len(package)} package_sha256={PACKAGE_SHA256} science_commit={SCIENCE_COMMIT} compat={COMPAT_MARKER}")\n    return 0''',
    )

    runtime = replace_function(
        runtime,
        "def load_e01_module() -> object:\n",
        "def json_safe(value):\n",
        '''def load_e02_module() -> object:\n    base = types.ModuleType("cmi_flu.strategy_e01")\n    base.__file__ = "<cmi_flu.strategy_e01>"\n    base.__package__ = "cmi_flu"\n    sys.modules["cmi_flu.strategy_e01"] = base\n    exec(compile(E01_SOURCE, "cmi_flu/strategy_e01.py", "exec"), base.__dict__, base.__dict__)\n    e01v2 = types.ModuleType("cmi_flu.strategy_e01_v2")\n    e01v2.__file__ = "<cmi_flu.strategy_e01_v2>"\n    e01v2.__package__ = "cmi_flu"\n    sys.modules["cmi_flu.strategy_e01_v2"] = e01v2\n    exec(compile(E01_V2_SOURCE, "cmi_flu/strategy_e01_v2.py", "exec"), e01v2.__dict__, e01v2.__dict__)\n    e02 = types.ModuleType("cmi_flu.strategy_e02")\n    e02.__file__ = "<cmi_flu.strategy_e02>"\n    e02.__package__ = "cmi_flu"\n    sys.modules["cmi_flu.strategy_e02"] = e02\n    exec(compile(E02_SOURCE, "cmi_flu/strategy_e02.py", "exec"), e02.__dict__, e02.__dict__)\n    e02v2 = types.ModuleType("cmi_flu.strategy_e02_v2")\n    e02v2.__file__ = "<cmi_flu.strategy_e02_v2>"\n    e02v2.__package__ = "cmi_flu"\n    sys.modules["cmi_flu.strategy_e02_v2"] = e02v2\n    exec(compile(E02_V2_SOURCE, "cmi_flu/strategy_e02_v2.py", "exec"), e02v2.__dict__, e02v2.__dict__)\n    run = getattr(e02v2, "run_strategy_e02", None)\n    if not callable(run):\n        raise BridgeContractError("e02_entry_missing")\n    return run''',
    )

    runtime = replace_function(
        runtime,
        "def validate_result(result: dict) -> None:\n",
        "def render_summary(result: dict) -> str:\n",
        '''def validate_result(result: dict) -> None:\n    if result.get("schema_version") != 1 or result.get("experiment") != "strategy_v2_e02_task11_structured_logfc":\n        raise BridgeContractError("e02_experiment_identity_mismatch")\n    if result.get("comparison_contract") != "paired_subject_purged_v2" or result.get("random_seed") != 20260907 or result.get("task") != "Task1.1":\n        raise BridgeContractError("e02_evaluation_contract_mismatch")\n    cohort = result.get("cohort") or {}\n    if cohort.get("train_rows") != 127 or cohort.get("train_studies") != 4 or cohort.get("challenge_rows") != 40:\n        raise BridgeContractError("e02_cohort_contract_mismatch")\n    fixed = result.get("fixed_conditions") or {}\n    if fixed.get("b21_model") != "pls_2" or fixed.get("anchor_residual_model") != "pls_1" or fixed.get("anchor_residual_lambda") != 0.25 or fixed.get("structured_ridge_alpha") != 10.0:\n        raise BridgeContractError("e02_fixed_condition_mismatch")\n    conditions = result.get("conditions") or {}\n    for key in ("anchor", "b21", "anchor_residual", "structured_ridge"):\n        metrics = (conditions.get(key) or {}).get("metrics") or {}\n        if metrics.get("rows") != 127:\n            raise BridgeContractError(f"e02_condition_rows_mismatch:{key}")\n    controls = result.get("negative_controls") or {}\n    if "availability_only" not in controls or "structured_subject_block_label_shuffle" not in controls:\n        raise BridgeContractError("e02_negative_controls_missing")\n    repeat = result.get("repeat_baseline_extension") or {}\n    audit = repeat.get("audit") or {}\n    if audit.get("status") not in {"ready", "data_limited"}:\n        raise BridgeContractError("e02_repeat_gate_status_invalid")\n    if result.get("contains_participant_identifiers") is not False or result.get("contains_row_level_predictions") is not False:\n        raise BridgeContractError("e02_row_level_output_flag")\n    if result.get("leaderboard_used_for_selection") is not False or result.get("competition_submission_attempted") is not False:\n        raise BridgeContractError("e02_selection_or_submission_flag")\n    forbidden = {"participant_id", "subject_group", "row_index", "oof_predictions", "challenge_predictions", "predictions"}\n    def walk(value):\n        if isinstance(value, dict):\n            for key, child in value.items():\n                if key in forbidden:\n                    raise BridgeContractError(f"e02_forbidden_output_key:{key}")\n                walk(child)\n        elif isinstance(value, list):\n            for child in value:\n                walk(child)\n    walk(result)''',
    )

    runtime = replace_function(
        runtime,
        "def render_summary(result: dict) -> str:\n",
        "def main() -> int:\n",
        '''def render_summary(result: dict) -> str:\n    lines = ["# CMI-Flu strategy E02 Task1.1 structured logFC", "", "Aggregate-only fixed-condition evaluation; no Competition submission was attempted.", "", f"- science commit: `{SCIENCE_COMMIT}`", "- task: `Task1.1`", "- primary reference: `same_readout_oriented_anchor`", ""]\n    conditions = result.get("conditions") or {}\n    for key in ("anchor", "b21", "anchor_residual", "structured_ridge", "structured_ridge_plus_repeat"):\n        item = conditions.get(key)\n        if not item:\n            continue\n        metrics = item.get("metrics") or {}\n        lines.append(f"- {key}: strict={metrics.get('study_equal_weight_spearman_mean_strict')} pooled_rank={(metrics.get('pooled_within_study_rank_spearman') or {}).get('value')} undefined={metrics.get('undefined_fold_count')}")\n    comp = (result.get("comparisons") or {}).get("structured_ridge_vs_anchor") or {}\n    lines.extend(["", f"- structured vs anchor strict delta: {comp.get('study_mean_delta_strict')}", f"- large-study review: {comp.get('large_studies_requiring_review')}", f"- repeat branch: {(result.get('repeat_baseline_extension') or {}).get('audit', {}).get('status')}", f"- structured candidate: {(result.get('decision') or {}).get('structured_ridge_candidate_over_anchor')}", ""])\n    return "\\n".join(lines)''',
    )

    old = '''        stage = "load_e01"\n        run_e01 = load_e01_module()\n        stage = "run_e01"\n        result = json_safe(dict(run_e01(config, inputs)))\n        stage = "validate_e01"\n'''
    new = '''        stage = "load_e02"\n        run_e02 = load_e02_module()\n        stage = "run_e02"\n        result = json_safe(dict(run_e02(config, inputs)))\n        stage = "validate_e02"\n'''
    if runtime.count(old) != 1:
        raise SystemExit("E02 runtime execution anchor changed")
    runtime = runtime.replace(old, new, 1)
    runtime = runtime.replace("CMI-Flu strategy E01 paired evaluation. Aggregate outputs only; no submission.", "CMI-Flu strategy E02 Task1.1 structured logFC. Aggregate outputs only; no submission.", 1)
    runtime = runtime.replace("CMI_FLU_E01_FAILED", "CMI_FLU_E02_FAILED")
    if "kaggle competitions submit" in runtime or "competition_submit" in runtime:
        raise SystemExit("generated E02 runtime contains submission path")
    for token in ("load_e02_module", "run_strategy_e02", "_e02_evaluation.run_compact_task = _e02_compact_compat", f'COMPAT_MARKER = "{COMPAT_MARKER}"'):
        if token not in runtime:
            raise SystemExit(f"generated E02 runtime missing token: {token}")
    compile(runtime, "generated_e02.py", "exec")
    return runtime


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validate_request(root)
    base = load_base(root)
    e01, e01_v2, config = base.load_exact_science(root)
    if base.git_blob_sha(e01.encode("utf-8")) != E01_BLOB or base.git_blob_sha(e01_v2.encode("utf-8")) != E01_V2_BLOB:
        raise SystemExit("E01 dependency blob mismatch")
    e02, e02_v2 = load_e02(root)
    with tempfile.TemporaryDirectory(prefix="cmi-e02-build-") as tmp:
        package, adapter = base.extract_frozen_runtime(root, Path(tmp))
        base.REQUEST_ID = REQUEST_ID
        base.TARGET_KERNEL = TARGET_KERNEL
        base.SCIENCE_COMMIT = SCIENCE_COMMIT
        runtime = base.build_runtime(package, adapter, e01, e01_v2, config)
    runtime = patch_runtime(runtime, e02, e02_v2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(f"CMI_FLU_E02_BUILD PASS request_id={REQUEST_ID} science_commit={SCIENCE_COMMIT} e02_blob={E02_BLOB} e02_v2_blob={E02_V2_BLOB} runtime_sha256={sha256(runtime.encode('utf-8'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
