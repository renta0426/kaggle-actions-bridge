#!/usr/bin/env python3
"""Build the V3 batch1 successor runtime with an import-tested dependency closure."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

import cmi_flu_v3_batch1_prepare as base

REQUEST_ID = "20260912-cmi-flu-strategy-v3-batch1-runtime-repair-002"
TARGET = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260912-002"
TITLE = "cmi-flu-v3-batch1-audit-diagnostics-20260912-002"
REQUEST = "requests/cmi-flu-v3-batch1-runtime-repair-002.json"
PAYLOAD_ROOT = Path("payloads/cmi-flu-v3-batch1-runtime-repair-002")
DEPENDENCIES = {
    "strategy_e04_contract": (PAYLOAD_ROOT / "strategy_e04_contract.py", "3982541febfb4641fbf895438ae1a465bdbb3d5e"),
    "strategy_e04b": (PAYLOAD_ROOT / "strategy_e04b.py", "72f936a7248dfa17338932077d17fb43123cb056"),
    "strategy_e09a": (PAYLOAD_ROOT / "strategy_e09a.py", "37270a0a8aafc070d91b5f43fbea3ba169cd61b0"),
}
PARENT_RUN = 34616531727
PARENT_JOB = 103319830217
PARENT_TARGET = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260911-001"
PARENT_FAILURE_CODE = "6e8f8d1003dad8b8feb7"


def _validate_repair_request(root: Path) -> None:
    request = json.loads((root / REQUEST).read_text(encoding="utf-8"))
    if request.get("request_id") != REQUEST_ID or request.get("target") != TARGET or request.get("title") != TITLE:
        raise SystemExit("V3 runtime-repair successor identity mismatch")
    parent = request.get("parent_runtime_failure") or {}
    expected_parent = {
        "request_id": "20260911-cmi-flu-strategy-v3-batch1-001",
        "target": PARENT_TARGET,
        "version": 1,
        "actions_run": PARENT_RUN,
        "actions_job": PARENT_JOB,
        "kernel_terminal_state": "ERROR",
        "failure_classification": "resource_consumed_runtime_failure",
        "failure_stage": "materialize_dependency_package",
        "failure_type": "ModuleNotFoundError",
        "failure_code": PARENT_FAILURE_CODE,
        "missing_module": "cmi_flu.strategy_e04_contract",
        "same_target_version_rerun_forbidden": True,
    }
    if parent != expected_parent:
        raise SystemExit("V3 runtime-repair parent incident mismatch")
    repair = request.get("runtime_repair_contract") or {}
    if repair != {
        "repair_scope": "dependency_import_closure_only",
        "science_logic_changed": False,
        "dependency_import_smoke_required_in_pr_ci": True,
        "automatic_retry": False,
    }:
        raise SystemExit("V3 runtime-repair scope changed")
    if request.get("automatic_compute_retries") != 0 or request.get("competition_submission_authorized") is not False:
        raise SystemExit("V3 runtime-repair retry/submission boundary changed")


def _load_dependency_sources(root: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    for name, (rel, expected_blob) in DEPENDENCIES.items():
        raw = (root / rel).read_bytes()
        found = base.git_blob(raw)
        if found != expected_blob:
            raise SystemExit(f"V3 runtime-repair dependency blob mismatch:{name}:{found}")
        source = raw.decode("utf-8")
        compile(source, f"cmi_flu/{name}.py", "exec")
        sources[name] = source
    return sources


def _validate_v3_dependency_surface(source: str) -> None:
    tree = ast.parse(source)
    imports: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module in {
            "strategy_e04_contract", "strategy_e04b", "strategy_e09a"
        }:
            imports[node.module] = {alias.name for alias in node.names}
    expected = {
        "strategy_e04_contract": {"audit_measurements"},
        "strategy_e04b": {"apply_material_plural_ontology"},
        "strategy_e09a": {"PUBLIC_RNA_FILES", "_scan_rna"},
    }
    if imports != expected:
        raise SystemExit(f"V3 runtime-repair dependency surface changed:{imports}")
    forbidden_calls = {"run_e04", "run_e04b", "bounded_residual"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
            raise SystemExit(f"V3 runtime-repair unexpected modeling call:{node.func.id}")


def build_runtime(root: Path, output: Path) -> str:
    _validate_repair_request(root)
    sources = _load_dependency_sources(root)

    # Reuse the already reviewed V3 builder but point its immutable request/target
    # identity at the fresh successor. The V3 science payload/blob itself is unchanged.
    base.REQUEST_ID = REQUEST_ID
    base.TARGET = TARGET
    base.REQUEST = REQUEST
    source = base.load_science(root)
    _validate_v3_dependency_surface(source)
    base.validate_request(root)
    runtime = base.build_runtime(root, output, source)

    anchor = f"V3_SOURCE = {source!r}\n"
    if runtime.count(anchor) != 1:
        raise SystemExit("V3 runtime-repair source constant anchor changed")
    injected = anchor
    for name in ("strategy_e04_contract", "strategy_e04b", "strategy_e09a"):
        const = name.upper() + "_SOURCE"
        blob = DEPENDENCIES[name][1]
        injected += f'{name.upper()}_BLOB = "{blob}"\n{const} = {sources[name]!r}\n'
    runtime = runtime.replace(anchor, injected, 1)

    loader = r'''def _v3_exec_module(name: str, source: str) -> object:
    import sys, types
    module_name = "cmi_flu." + name
    module = types.ModuleType(module_name)
    module.__file__ = "<" + module_name + ">"
    module.__package__ = "cmi_flu"
    sys.modules[module_name] = module
    exec(compile(source, "cmi_flu/" + name + ".py", "exec"), module.__dict__, module.__dict__)
    return module


def load_v3_dependency_closure() -> None:
    import sys, types
    contract = _v3_exec_module("strategy_e04_contract", STRATEGY_E04_CONTRACT_SOURCE)
    if not callable(getattr(contract, "audit_measurements", None)):
        raise BridgeContractError("v3_dependency_e04_contract_entry_missing")

    # E04b imports run_e04 at module import time, but Strategy-v3 uses only the
    # exact outcome-independent apply_material_plural_ontology function. Fail
    # closed if any path unexpectedly attempts to execute the modeling parent.
    shim = types.ModuleType("cmi_flu.strategy_e04")
    shim.__file__ = "<cmi_flu.strategy_e04:v3-runtime-forbidden-shim>"
    shim.__package__ = "cmi_flu"
    def forbidden_run_e04(*args, **kwargs):
        raise BridgeContractError("v3_runtime_run_e04_forbidden")
    shim.run_e04 = forbidden_run_e04
    sys.modules["cmi_flu.strategy_e04"] = shim

    e04b = _v3_exec_module("strategy_e04b", STRATEGY_E04B_SOURCE)
    if not callable(getattr(e04b, "apply_material_plural_ontology", None)):
        raise BridgeContractError("v3_dependency_e04b_entry_missing")
    e09a = _v3_exec_module("strategy_e09a", STRATEGY_E09A_SOURCE)
    if not callable(getattr(e09a, "_scan_rna", None)) or not isinstance(getattr(e09a, "PUBLIC_RNA_FILES", None), dict):
        raise BridgeContractError("v3_dependency_e09a_entry_missing")


def load_v3_batch1_module() -> object:
    import sys, types
    load_v3_dependency_closure()
    module = types.ModuleType("cmi_flu.strategy_v3_batch1")
    module.__file__ = "<cmi_flu.strategy_v3_batch1>"
    module.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_v3_batch1"] = module
    exec(compile(V3_SOURCE, "cmi_flu/strategy_v3_batch1.py", "exec"), module.__dict__, module.__dict__)
    for name in ("run_v3_01_audit", "generate_singleton_diagnostics", "write_aggregate_jsons"):
        if not callable(getattr(module, name, None)):
            raise BridgeContractError("v3_batch1_entry_missing:" + name)
    return module
'''
    runtime = base.replace_function(runtime, "load_v3_batch1_module", loader)

    self_test = r'''def self_test() -> int:
    import ast, tempfile, sys
    if git_blob_sha(V3_SOURCE.encode("utf-8")) != V3_SCIENCE_BLOB:
        raise BridgeContractError("v3_batch1_blob_mismatch")
    expected = {
        "strategy_e04_contract": (STRATEGY_E04_CONTRACT_SOURCE, STRATEGY_E04_CONTRACT_BLOB),
        "strategy_e04b": (STRATEGY_E04B_SOURCE, STRATEGY_E04B_BLOB),
        "strategy_e09a": (STRATEGY_E09A_SOURCE, STRATEGY_E09A_BLOB),
    }
    for name, (source, blob) in expected.items():
        if git_blob_sha(source.encode("utf-8")) != blob:
            raise BridgeContractError("v3_dependency_blob_mismatch:" + name)
    tree = ast.parse(V3_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "fit":
                raise BridgeContractError("v3_batch1_fit_call_present")
            if isinstance(func, ast.Attribute) and "submit" in func.attr.casefold():
                raise BridgeContractError("v3_batch1_submit_call_present")
    if V3_SOURCE_B_VERSION != 1 or V3_SOURCE_B_BYTES != 5926:
        raise BridgeContractError("v3_batch1_source_B_version_or_size_changed")
    if V3_SOURCE_B_SHA256 != "0f9df53c3aa8c6e4ac693f6a42dbd2633b4b61d1462df798a9bd767c220be3a5":
        raise BridgeContractError("v3_batch1_source_B_hash_changed")

    # This is the regression that the failed parent lacked: use the same old
    # bundle as Kaggle, then actually import the exact relayed dependency
    # closure and the exact V3 module. No competition data or model is touched.
    with tempfile.TemporaryDirectory(prefix="cmi-v3-dependency-smoke-") as tmp:
        package_path = Path(tmp) / "cmi_flu_bundle.zip"
        package_path.write_bytes(package_bytes())
        sys.path.insert(0, str(package_path))
        try:
            module = load_v3_batch1_module()
            for name in ("run_v3_01_audit", "generate_singleton_diagnostics", "write_aggregate_jsons"):
                if not callable(getattr(module, name, None)):
                    raise BridgeContractError("v3_dependency_smoke_entry_missing:" + name)
        finally:
            if sys.path and sys.path[0] == str(package_path):
                sys.path.pop(0)
    print(
        "CMI_FLU_V3_BATCH1_RUNTIME_REPAIR_SELF_TEST PASS "
        f"request_id={V3_REQUEST_ID} science_commit={V3_SCIENCE_COMMIT} science_blob={V3_SCIENCE_BLOB} "
        "dependency_import_smoke=true model_fit=false competition_submit=false diagnostic_files=6"
    )
    return 0
'''
    runtime = base.replace_function(runtime, "self_test", self_test)
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
        "CMI_FLU_V3_BATCH1_RUNTIME_REPAIR_PREPARE PASS "
        f"request_id={REQUEST_ID} target={TARGET} runtime_bytes={len(runtime.encode('utf-8'))} "
        "science_changed=false dependency_closure=exact_relay parent_version_consumed=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
