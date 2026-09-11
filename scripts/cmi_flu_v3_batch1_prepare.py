#!/usr/bin/env python3
"""Build the exact no-fit CMI-Flu Strategy-v3 batch-1 Kaggle runtime."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REQUEST_ID = "20260911-cmi-flu-strategy-v3-batch1-001"
TARGET = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260911-001"
SCIENCE_COMMIT = "270cd6fc5e33579add8177303505e5f744ee5dbc"
SCIENCE_BLOB = "cedd8e2538a06c8b696e74631f0bb2fce427984b"
DEPENDENCY_BASE_COMMIT = "9916c04a3d5510eead8f361960d73b5d94201892"
SOURCE_B = "renta0426/cmi-flu-e12c-manual-submission-20260911-002"
SOURCE_B_VERSION = 1
SOURCE_B_BYTES = 5926
SOURCE_B_SHA256 = "0f9df53c3aa8c6e4ac693f6a42dbd2633b4b61d1462df798a9bd767c220be3a5"
PAYLOAD = "payloads/cmi-flu-v3-batch1-001/strategy_v3_batch1.py"
REQUEST = "requests/cmi-flu-v3-batch1-001.json"
BASE_BUILDER = "scripts/cmi_flu_e12c_v2_historical_backbone_prepare.py"
BASE_BUILDER_BLOB = "7736e76c40eb70e3768d9647b491d13ea91e5ffd"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def replace_function(text: str, name: str, replacement: str) -> str:
    tree = ast.parse(text)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(nodes) != 1:
        raise SystemExit(f"V3 batch1 generated-runtime function binding changed:{name}:{len(nodes)}")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    return "".join(lines[: node.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno :])


def validate_request(root: Path) -> None:
    request = json.loads((root / REQUEST).read_text(encoding="utf-8"))
    exact = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "execution_policy": "kaggle_native_capacity_v2",
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "save_kernel_once",
        "target": TARGET,
        "science_source_commit": SCIENCE_COMMIT,
        "strategy_v3_batch1_blob_sha": SCIENCE_BLOB,
        "automatic_compute_retries": 0,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "competition_submission_authorized": False,
        "manual_operator_submission_only": True,
        "leaderboard_used_for_selection": False,
        "public_artifact_upload": False,
    }
    for key, value in exact.items():
        if request.get(key) != value:
            raise SystemExit(f"V3 batch1 request mismatch:{key}")
    source = request.get("source_B") or {}
    if source.get("source_kernel") != SOURCE_B or source.get("expected_current_version") != SOURCE_B_VERSION:
        raise SystemExit("V3 batch1 source B identity mismatch")
    if source.get("bytes") != SOURCE_B_BYTES or source.get("sha256") != SOURCE_B_SHA256:
        raise SystemExit("V3 batch1 source B byte/hash mismatch")
    audit = request.get("audit_contract") or {}
    if audit.get("model_fit_allowed") is not False or audit.get("refit_allowed") is not False or audit.get("hpo_allowed") is not False:
        raise SystemExit("V3 batch1 no-fit boundary changed")
    diagnostic = request.get("diagnostic_contract") or {}
    if diagnostic.get("tasks") != ["Task1.1", "Task1.2", "Task1.3", "Task1.4", "Task2.1", "Task2.2"]:
        raise SystemExit("V3 batch1 diagnostic task set changed")
    if diagnostic.get("diagnostic_not_final") is not True or diagnostic.get("competition_submission_allowed") is not False:
        raise SystemExit("V3 batch1 diagnostic boundary changed")


def load_science(root: Path) -> str:
    raw = (root / PAYLOAD).read_bytes()
    if git_blob(raw) != SCIENCE_BLOB:
        raise SystemExit("V3 batch1 exact science relay blob mismatch")
    source = raw.decode("utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "fit":
                raise SystemExit("V3 batch1 science source contains model fit call")
            if isinstance(func, ast.Attribute) and "submit" in func.attr.casefold():
                raise SystemExit("V3 batch1 science source contains submit call")
    lowered = source.casefold()
    if "kaggle competitions submit" in lowered or "competition_submit(" in lowered:
        raise SystemExit("V3 batch1 science source contains Competition submit path")
    compile(source, "cmi_flu/strategy_v3_batch1.py", "exec")
    return source


def build_runtime(root: Path, output: Path, source: str) -> str:
    base = root / BASE_BUILDER
    if git_blob(base.read_bytes()) != BASE_BUILDER_BLOB:
        raise SystemExit("V3 batch1 base runtime builder changed")
    with tempfile.TemporaryDirectory(prefix="cmi-v3-batch1-base-") as tmp:
        generated = Path(tmp) / "base_runtime.py"
        subprocess.run([sys.executable, str(base), "--repository-root", str(root), "--output", str(generated)], check=True)
        runtime = generated.read_text(encoding="utf-8")

    marker = f'EXPECTED_TASK13_SHA256 = "de8bc3b6bbd3e3aad4099b83eb63a1ebbd81c4f4eeb60f77808858f4674be5da"\n'
    if runtime.count(marker) != 1:
        raise SystemExit("V3 batch1 insertion anchor changed")
    constants = (
        marker
        + f'V3_REQUEST_ID = "{REQUEST_ID}"\n'
        + f'V3_TARGET_KERNEL = "{TARGET}"\n'
        + f'V3_SCIENCE_COMMIT = "{SCIENCE_COMMIT}"\n'
        + f'V3_SCIENCE_BLOB = "{SCIENCE_BLOB}"\n'
        + f'V3_DEPENDENCY_BASE_COMMIT = "{DEPENDENCY_BASE_COMMIT}"\n'
        + f'V3_SOURCE_B = "{SOURCE_B}"\n'
        + f'V3_SOURCE_B_VERSION = {SOURCE_B_VERSION}\n'
        + f'V3_SOURCE_B_BYTES = {SOURCE_B_BYTES}\n'
        + f'V3_SOURCE_B_SHA256 = "{SOURCE_B_SHA256}"\n'
        + f'V3_SOURCE = {source!r}\n'
    )
    runtime = runtime.replace(marker, constants, 1)

    loader = r'''def load_v3_batch1_module() -> object:
    import sys, types
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
    anchor = "def execute(input_dir: Path, output_dir: Path) -> int:\n"
    if runtime.count(anchor) != 1:
        raise SystemExit("V3 batch1 execute insertion anchor changed")
    runtime = runtime.replace(anchor, loader + "\n" + anchor, 1)

    self_test = r'''def self_test() -> int:
    import ast
    if git_blob_sha(V3_SOURCE.encode("utf-8")) != V3_SCIENCE_BLOB:
        raise BridgeContractError("v3_batch1_blob_mismatch")
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
    print(
        "CMI_FLU_V3_BATCH1_RUNTIME_SELF_TEST PASS "
        f"request_id={V3_REQUEST_ID} science_commit={V3_SCIENCE_COMMIT} science_blob={V3_SCIENCE_BLOB} "
        "model_fit=false competition_submit=false diagnostic_files=6"
    )
    return 0
'''
    runtime = replace_function(runtime, "self_test", self_test)

    execute = r'''def execute(input_dir: Path, output_dir: Path) -> int:
    import hashlib, json, shutil, sys
    runtime_root = Path("/tmp") / "cmi-flu-v3-batch1-runtime"
    stage = "initialize"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        declared = [
            "v3_public_singleton_Task1_1.csv", "v3_public_singleton_Task1_2.csv", "v3_public_singleton_Task1_3.csv",
            "v3_public_singleton_Task1_4.csv", "v3_public_singleton_Task2_1.csv", "v3_public_singleton_Task2_2.csv",
            "teacher_ledger.json", "measurement_contracts.json", "split_support.json", "auxiliary_label_coverage.json",
            "source_alignment_audit.json", "diagnostic_manifest.json", "runtime_receipt.json",
        ]
        if any((output_dir / name).exists() for name in declared):
            raise BridgeContractError("v3_batch1_declared_output_already_exists")
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        runtime_root.mkdir(parents=True)

        stage = "materialize_dependency_package"
        package = package_bytes()
        package_path = runtime_root / "cmi_flu_bundle.zip"
        package_path.write_bytes(package)
        sys.path.insert(0, str(package_path))
        v3 = load_v3_batch1_module()

        stage = "locate_exact_source_B"
        matches = []
        for candidate in Path("/kaggle/input").rglob("submission.csv"):
            if not candidate.is_file():
                continue
            raw = candidate.read_bytes()
            if len(raw) == V3_SOURCE_B_BYTES and hashlib.sha256(raw).hexdigest() == V3_SOURCE_B_SHA256:
                matches.append(candidate)
        if len(matches) != 1:
            raise BridgeContractError(f"v3_batch1_exact_source_B_count:{len(matches)}")
        source_B = matches[0]
        sample_path = input_dir / "sample_submission_part1.csv"
        if not sample_path.is_file():
            raise BridgeContractError("v3_batch1_sample_submission_missing")

        stage = "run_no_fit_audit"
        audit = v3.run_v3_01_audit(input_dir)
        if int(audit.get("model_fit_count", -1)) != 0 or audit.get("competition_submission_attempted") is not False:
            raise BridgeContractError("v3_batch1_no_fit_receipt_failed")
        privacy = audit.get("privacy") or {}
        if privacy.get("aggregate_only") is not True or privacy.get("contains_individual_ids") is not False:
            raise BridgeContractError("v3_batch1_privacy_receipt_failed")

        stage = "generate_all_singletons"
        manifest = v3.generate_singleton_diagnostics(source_B, sample_path, output_dir)
        files = manifest.get("files") or []
        expected_tasks = ["Task1.1", "Task1.2", "Task1.3", "Task1.4", "Task2.1", "Task2.2"]
        if [item.get("task") for item in files] != expected_tasks or len(files) != 6:
            raise BridgeContractError("v3_batch1_diagnostic_file_set_changed")
        if manifest.get("diagnostic_not_final") is not True or manifest.get("all_generated_before_scoring") is not True:
            raise BridgeContractError("v3_batch1_diagnostic_manifest_boundary_changed")

        stage = "write_aggregate_records"
        aggregate_names = v3.write_aggregate_jsons(audit, output_dir)
        if aggregate_names != ["teacher_ledger.json", "measurement_contracts.json", "split_support.json", "auxiliary_label_coverage.json", "source_alignment_audit.json"]:
            raise BridgeContractError("v3_batch1_aggregate_allowlist_changed")
        (output_dir / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipt = {
            "schema_version": 1,
            "request_id": V3_REQUEST_ID,
            "science_commit": V3_SCIENCE_COMMIT,
            "science_blob": V3_SCIENCE_BLOB,
            "dependency_base_commit": V3_DEPENDENCY_BASE_COMMIT,
            "source_B": {"kernel": V3_SOURCE_B, "version": V3_SOURCE_B_VERSION, "bytes": V3_SOURCE_B_BYTES, "sha256": V3_SOURCE_B_SHA256},
            "dataset_version": audit.get("dataset_version"),
            "alias_map_sha256": audit.get("alias_map_sha256"),
            "md5_manifest_verified_count": audit.get("md5_manifest_verified_count"),
            "model_fit_count": 0,
            "competition_submission_attempted": False,
            "diagnostic_not_final": True,
            "diagnostic_file_count": 6,
            "aggregate_file_count": 5,
            "runtime_terminal_marker": "CMI_FLU_V3_BATCH1_RUNTIME_PASS",
        }
        (output_dir / "runtime_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        actual = sorted(path.name for path in output_dir.iterdir() if path.is_file())
        expected = sorted(declared)
        if actual != expected:
            raise BridgeContractError("v3_batch1_output_allowlist_changed")

        print(
            "CMI_FLU_V3_BATCH1_RUNTIME_PASS "
            f"dataset_version={audit.get('dataset_version')} md5_verified={audit.get('md5_manifest_verified_count')} "
            "model_fit_count=0 competition_submit=false diagnostic_files=6 aggregate_files=5 diagnostic_not_final=true"
        )
        return 0
    except Exception as exc:
        code = hashlib.sha256(f"{stage}:{type(exc).__name__}:{exc}".encode()).hexdigest()[:20]
        print(f"CMI_FLU_V3_BATCH1_RUNTIME_FAIL stage={stage} type={type(exc).__name__} code={code}", file=sys.stderr)
        return 1
'''
    runtime = replace_function(runtime, "execute", execute)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    return runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    validate_request(root)
    source = load_science(root)
    runtime = build_runtime(root, args.output.expanduser().resolve(), source)
    required = (REQUEST_ID, TARGET, SCIENCE_COMMIT, SCIENCE_BLOB, SOURCE_B, SOURCE_B_SHA256, "CMI_FLU_V3_BATCH1_RUNTIME_PASS", "model_fit_count=0", "competition_submit=false")
    if any(token not in runtime for token in required):
        raise SystemExit("V3 batch1 generated runtime token contract incomplete")
    print(f"CMI_FLU_V3_BATCH1_PREPARE PASS science_commit={SCIENCE_COMMIT} science_blob={SCIENCE_BLOB} runtime_bytes={len(runtime.encode())} model_fit=false competition_submit=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
