#!/usr/bin/env python3
"""Build E01 repair runtime with a narrowly scoped frozen-API compatibility shim."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REQUEST_ID = "20260907-cmi-flu-strategy-e01-paired-evaluation-repair-002"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-e01-paired-eval-repair-20260907-002"
SCIENCE_COMMIT = "0b2ecb47eaa09f22450424c9c06dc88cf44bc1fb"
E01_BLOB = "dd27aea0cf97d41bad3cec64819c4c4269d94cbd"
E01_V2_BLOB = "8cc64dc5ab9483d5957cfada18d445188566c56c"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
BASE_BUILDER = "scripts/cmi_flu_strategy_e01_prepare.py"
REQUEST_PATH = "requests/cmi-flu-strategy-e01-paired-evaluation-repair-002.json"
ROOT_CAUSE_ERROR_CODE = "a117b10075001dd7887e"
ROOT_CAUSE_MESSAGE = "run_compact_task() got an unexpected keyword argument 'selection_policy'"
REPAIR_MARKER = "e01_frozen_selection_policy_api_compat_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_base(root: Path):
    path = root / BASE_BUILDER
    spec = importlib.util.spec_from_file_location("cmi_flu_e01_prepare_base", path)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load E01 base builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        "REQUEST_ID": "20260907-cmi-flu-strategy-e01-paired-evaluation-001",
        "TARGET_KERNEL": "renta0426/cmi-flu-e01-paired-eval-20260907-001",
        "SCIENCE_COMMIT": SCIENCE_COMMIT,
        "E01_BLOB": E01_BLOB,
        "E01_V2_BLOB": E01_V2_BLOB,
        "CONFIG_BLOB": CONFIG_BLOB,
    }
    for name, value in expected.items():
        if getattr(module, name, None) != value:
            raise SystemExit(f"base E01 builder contract changed: {name}")
    return module


def validate_root_cause() -> None:
    code = sha256(f"TypeError:{ROOT_CAUSE_MESSAGE}".encode("utf-8"))[:20]
    if code != ROOT_CAUSE_ERROR_CODE:
        raise SystemExit("E01 root-cause error-code reconstruction mismatch")


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
        "config_blob_sha": CONFIG_BLOB,
        "expected_kernel_version": 1,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
        "repair_marker": REPAIR_MARKER,
        "root_cause_error_code": ROOT_CAUSE_ERROR_CODE,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"E01 repair request mismatch: {key}")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 30,
        "hard_timeout_minutes": 60,
        "max_active_runs": 1,
    }:
        raise SystemExit("E01 repair resource contract mismatch")
    if request.get("allowed_output_paths") != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E01 repair output allowlist mismatch")


def patch_runtime(runtime: str) -> str:
    anchor = '''        install()\n        stage = "load_inputs"\n'''
    if runtime.count(anchor) != 1:
        raise SystemExit("E01 runtime adapter anchor changed")
    replacement = '''        install()\n        stage = "install_e01_api_compat"\n        import cmi_flu.evaluation as _e01_evaluation\n        import cmi_flu.runner as _e01_runner\n        _e01_robust_compact = _e01_runner.run_compact_task\n        _e01_robust_hai = _e01_runner.run_hai_compact_for_panels\n\n        def _e01_compact_compat(dataset, *, specs, splits=None, random_state=42, selection_policy="robust_v1"):\n            if selection_policy != "robust_v1":\n                raise BridgeContractError("e01_compact_selection_policy_not_robust_v1")\n            return _e01_robust_compact(\n                dataset, specs=specs, splits=splits, random_state=random_state\n            )\n\n        def _e01_hai_compat(dataset, *, specs, selection_panels, splits=None, selection_policy="robust_v1"):\n            if selection_policy != "robust_v1":\n                raise BridgeContractError("e01_hai_selection_policy_not_robust_v1")\n            return _e01_robust_hai(\n                dataset, specs=specs, selection_panels=selection_panels, splits=splits\n            )\n\n        _e01_evaluation.run_compact_task = _e01_compact_compat\n        _e01_evaluation.run_hai_compact_for_panels = _e01_hai_compat\n        stage = "load_inputs"\n'''
    runtime = runtime.replace(anchor, replacement, 1)
    marker_anchor = f'TARGET_KERNEL = "{TARGET_KERNEL}"\n'
    if runtime.count(marker_anchor) != 1:
        raise SystemExit("E01 repair target marker missing")
    runtime = runtime.replace(marker_anchor, marker_anchor + f'REPAIR_MARKER = "{REPAIR_MARKER}"\n', 1)
    required = (
        "def _e01_compact_compat(",
        "def _e01_hai_compat(",
        "_e01_evaluation.run_compact_task = _e01_compact_compat",
        "_e01_evaluation.run_hai_compact_for_panels = _e01_hai_compat",
        f'REPAIR_MARKER = "{REPAIR_MARKER}"',
    )
    if any(token not in runtime for token in required):
        raise SystemExit("E01 repair compatibility shim incomplete")
    compile(runtime, "generated_e01_repair.py", "exec")
    return runtime


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validate_root_cause()
    validate_request(root)
    base = load_base(root)
    base.REQUEST_ID = REQUEST_ID
    base.TARGET_KERNEL = TARGET_KERNEL
    base.REQUEST_PATH = REQUEST_PATH
    base.validate_request(root)
    e01, e01_v2, config = base.load_exact_science(root)
    with tempfile.TemporaryDirectory(prefix="cmi-e01-repair-build-") as tmp:
        package, adapter = base.extract_frozen_runtime(root, Path(tmp))
        runtime = base.build_runtime(package, adapter, e01, e01_v2, config)
    runtime = patch_runtime(runtime)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_E01_REPAIR_BUILD PASS "
        f"request_id={REQUEST_ID} repair={REPAIR_MARKER} root_cause={ROOT_CAUSE_ERROR_CODE} "
        f"science_commit={SCIENCE_COMMIT} e01_blob={E01_BLOB} e01_v2_blob={E01_V2_BLOB} "
        f"config_blob={CONFIG_BLOB} runtime_sha256={sha256(runtime.encode('utf-8'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
