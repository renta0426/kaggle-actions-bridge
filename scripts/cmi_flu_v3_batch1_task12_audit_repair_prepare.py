#!/usr/bin/env python3
"""Build fresh V3 batch1 successor with the exact Task1.2 audit repair."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import cmi_flu_v3_batch1_runtime_repair_prepare as parent

REQUEST_ID = "20260912-cmi-flu-strategy-v3-batch1-task12-audit-repair-003"
TARGET = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260912-003"
TITLE = "cmi-flu-v3-batch1-audit-diagnostics-20260912-003"
REQUEST = "requests/cmi-flu-v3-batch1-task12-audit-repair-003.json"
SCIENCE_REPAIR_COMMIT = "06d29b1c2317b9bfb66b5d4364213d3ca6de3d7b"
SCIENCE_REPAIR_BLOB = "6ed220226c7dda80391f1c4700e6acf07d2d9fee"
SCIENCE_REPAIR_PAYLOAD = Path("payloads/cmi-flu-v3-batch1-task12-audit-repair-003/strategy_v3_batch1_v2.py")
PARENT_REQUEST_ID = "20260912-cmi-flu-strategy-v3-batch1-runtime-repair-002"
PARENT_TARGET = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260912-002"
PARENT_RUN = 34620149186
PARENT_JOB = 103331885221
PARENT_FAILURE_CODE = "ebb7c658334065658611"
PARENT_FAILURE_MESSAGE = "Task1.2 dataset has an empty partition"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def validate_request(root: Path) -> None:
    request = json.loads((root / REQUEST).read_text(encoding="utf-8"))
    expected = {
        "request_id": REQUEST_ID,
        "execution_policy": "kaggle_native_capacity_v2",
        "operation": "save_kernel_once",
        "target": TARGET,
        "title": TITLE,
        "science_source_commit": SCIENCE_REPAIR_COMMIT,
        "strategy_v3_batch1_v2_blob_sha": SCIENCE_REPAIR_BLOB,
        "automatic_compute_retries": 0,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "competition_submission_authorized": False,
        "manual_operator_submission_only": True,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"V3 Task1.2 audit-repair request mismatch:{key}")
    parent_failure = request.get("parent_runtime_failure") or {}
    exact_parent = {
        "request_id": PARENT_REQUEST_ID,
        "target": PARENT_TARGET,
        "version": 1,
        "actions_run": PARENT_RUN,
        "actions_job": PARENT_JOB,
        "kernel_terminal_state": "ERROR",
        "failure_classification": "resource_consumed_runtime_failure",
        "failure_stage": "run_no_fit_audit",
        "failure_type": "DataContractError",
        "failure_code": PARENT_FAILURE_CODE,
        "resolved_message": PARENT_FAILURE_MESSAGE,
        "same_target_version_rerun_forbidden": True,
    }
    if parent_failure != exact_parent:
        raise SystemExit("V3 Task1.2 audit-repair parent failure contract mismatch")
    repair = request.get("science_repair_contract") or {}
    if repair.get("repair_scope") != "Task1.2_data_limited_audit_semantics_only":
        raise SystemExit("V3 Task1.2 audit-repair scope changed")
    if repair.get("prediction_logic_changed") is not False or repair.get("reference_expectations_mutated") is not False:
        raise SystemExit("V3 Task1.2 audit-repair science boundary changed")
    if repair.get("empty_strict_baseline_join_policy") != "record_zero_and_data_limited":
        raise SystemExit("V3 Task1.2 audit-repair empty-support policy changed")
    if repair.get("empty_challenge_D1_measurement_policy") != "unresolved_not_compatible":
        raise SystemExit("V3 Task1.2 audit-repair missing-measurement policy changed")
    if any(repair.get(key) is not False for key in ("model_fit_allowed", "refit_allowed", "hpo_allowed")):
        raise SystemExit("V3 Task1.2 audit-repair no-fit boundary changed")


def load_repair_source(root: Path) -> str:
    raw = (root / SCIENCE_REPAIR_PAYLOAD).read_bytes()
    found = git_blob(raw)
    if found != SCIENCE_REPAIR_BLOB:
        raise SystemExit(f"V3 Task1.2 audit-repair exact science blob mismatch:{found}")
    source = raw.decode("utf-8")
    compile(source, "cmi_flu/strategy_v3_batch1_v2.py", "exec")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "fit":
                raise SystemExit("V3 Task1.2 audit-repair science contains fit call")
            if isinstance(func, ast.Attribute) and "submit" in func.attr.casefold():
                raise SystemExit("V3 Task1.2 audit-repair science contains submit call")
    if PARENT_FAILURE_CODE not in source or PARENT_FAILURE_MESSAGE not in source:
        raise SystemExit("V3 Task1.2 audit-repair observed failure proof missing")
    return source


def build_runtime(root: Path, output: Path) -> str:
    validate_request(root)
    repair_source = load_repair_source(root)
    with tempfile.TemporaryDirectory(prefix="cmi-v3-task12-parent-") as tmp:
        parent_runtime = Path(tmp) / "runtime.py"
        parent.build_runtime(root, parent_runtime)
        runtime = parent_runtime.read_text(encoding="utf-8")

    for old, new, label in (
        (PARENT_REQUEST_ID, REQUEST_ID, "request"),
        (PARENT_TARGET, TARGET, "target"),
    ):
        if old not in runtime:
            raise SystemExit(f"V3 Task1.2 audit-repair parent runtime {label} anchor missing")
        runtime = runtime.replace(old, new)
        if old in runtime:
            raise SystemExit(f"V3 Task1.2 audit-repair retained old {label}")

    anchor = f'V3_SOURCE = {parent.base.load_science(root)!r}\n'
    if runtime.count(anchor) != 1:
        raise SystemExit("V3 Task1.2 audit-repair V3 source constant anchor changed")
    runtime = runtime.replace(
        anchor,
        anchor
        + f'V3_V2_SCIENCE_COMMIT = "{SCIENCE_REPAIR_COMMIT}"\n'
        + f'V3_V2_BLOB = "{SCIENCE_REPAIR_BLOB}"\n'
        + f'V3_V2_SOURCE = {repair_source!r}\n',
        1,
    )

    loader = r'''def load_v3_batch1_module() -> object:
    import sys, types
    load_v3_dependency_closure()
    base_module = types.ModuleType("cmi_flu.strategy_v3_batch1")
    base_module.__file__ = "<cmi_flu.strategy_v3_batch1>"
    base_module.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_v3_batch1"] = base_module
    exec(compile(V3_SOURCE, "cmi_flu/strategy_v3_batch1.py", "exec"), base_module.__dict__, base_module.__dict__)
    repair_module = types.ModuleType("cmi_flu.strategy_v3_batch1_v2")
    repair_module.__file__ = "<cmi_flu.strategy_v3_batch1_v2>"
    repair_module.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_v3_batch1_v2"] = repair_module
    exec(compile(V3_V2_SOURCE, "cmi_flu/strategy_v3_batch1_v2.py", "exec"), repair_module.__dict__, repair_module.__dict__)
    for name in ("run_v3_01_audit", "generate_singleton_diagnostics", "write_aggregate_jsons"):
        if not callable(getattr(repair_module, name, None)):
            raise BridgeContractError("v3_batch1_v2_entry_missing:" + name)
    return repair_module
'''
    runtime = parent.parent.base.replace_function(runtime, "load_v3_batch1_module", loader)

    self_test = r'''def self_test() -> int:
    import ast, hashlib, sys, tempfile
    if git_blob_sha(V3_SOURCE.encode("utf-8")) != V3_SCIENCE_BLOB:
        raise BridgeContractError("v3_batch1_blob_mismatch")
    if git_blob_sha(V3_V2_SOURCE.encode("utf-8")) != V3_V2_BLOB:
        raise BridgeContractError("v3_batch1_v2_blob_mismatch")
    expected_code = hashlib.sha256(
        b"run_no_fit_audit:DataContractError:Task1.2 dataset has an empty partition"
    ).hexdigest()[:20]
    if expected_code != "ebb7c658334065658611":
        raise BridgeContractError("v3_task12_parent_failure_fingerprint_mismatch")
    tree = ast.parse(V3_V2_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "fit":
                raise BridgeContractError("v3_batch1_v2_fit_call_present")
            if isinstance(func, ast.Attribute) and "submit" in func.attr.casefold():
                raise BridgeContractError("v3_batch1_v2_submit_call_present")
    with tempfile.TemporaryDirectory(prefix="cmi-v3-task12-smoke-") as tmp:
        package_path = Path(tmp) / "cmi_flu_bundle.zip"
        package_path.write_bytes(package_bytes())
        sys.path.insert(0, str(package_path))
        try:
            module = load_v3_batch1_module()
            if getattr(module, "OBSERVED_RUNTIME_FAILURE_CODE", None) != "ebb7c658334065658611":
                raise BridgeContractError("v3_batch1_v2_observed_failure_contract_missing")
            for name in ("run_v3_01_audit", "generate_singleton_diagnostics", "write_aggregate_jsons"):
                if not callable(getattr(module, name, None)):
                    raise BridgeContractError("v3_batch1_v2_dependency_smoke_entry_missing:" + name)
        finally:
            if sys.path and sys.path[0] == str(package_path):
                sys.path.pop(0)
    print(
        "CMI_FLU_V3_BATCH1_TASK12_AUDIT_REPAIR_SELF_TEST PASS "
        f"request_id={V3_REQUEST_ID} base_science_blob={V3_SCIENCE_BLOB} repair_science_blob={V3_V2_BLOB} "
        "parent_failure_proven=true data_limited_semantics=true model_fit=false competition_submit=false"
    )
    return 0
'''
    runtime = parent.parent.base.replace_function(runtime, "self_test", self_test)

    receipt_anchor = '            "science_blob": V3_SCIENCE_BLOB,\n'
    if runtime.count(receipt_anchor) != 1:
        raise SystemExit("V3 Task1.2 audit-repair receipt anchor changed")
    runtime = runtime.replace(
        receipt_anchor,
        receipt_anchor
        + '            "science_repair_commit": V3_V2_SCIENCE_COMMIT,\n'
        + '            "science_repair_blob": V3_V2_BLOB,\n',
        1,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    return runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    runtime = build_runtime(root, args.output.resolve())
    print(
        "CMI_FLU_V3_BATCH1_TASK12_AUDIT_REPAIR_PREPARE PASS "
        f"request_id={REQUEST_ID} target={TARGET} science_repair_commit={SCIENCE_REPAIR_COMMIT} "
        f"science_repair_blob={SCIENCE_REPAIR_BLOB} runtime_bytes={len(runtime.encode('utf-8'))} "
        "model_fit=false competition_submit=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
