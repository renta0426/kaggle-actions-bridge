#!/usr/bin/env python3
"""Build E12c-v2 Kaggle runtime using the exact historical 0.218 backbone."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REQUEST_ID = "20260911-cmi-flu-e12c-v2-historical-backbone-002"
TARGET = "renta0426/cmi-flu-e12c-manual-submission-20260911-002"
SCIENCE_COMMIT = "9916c04a3d5510eead8f361960d73b5d94201892"
E12C_V2_BLOB = "238d3ad67984fdd8c375bb0aa263bb4717029027"
E12C_V2_CONTRACT_BLOB = "f2835633c83f82fd9f8a9dc1f25598aa098ff56f"
E12C_V2_PATH = "payloads/cmi-flu-e12c-v2-historical-backbone-002/strategy_e12c_v2.py"
E12C_V2_CONTRACT_PATH = "payloads/cmi-flu-e12c-v2-historical-backbone-002/strategy_e12c_v2_historical_backbone.json"
REQUEST_PATH = "requests/cmi-flu-e12c-v2-historical-backbone-002.json"
BASE_BUILDER = "scripts/cmi_flu_e12c_manual_submission_notebook_prepare.py"
BASE_BUILDER_BLOB = "45d5c791401320f566f058487a767c149b0df12b"
OLD_REQUEST = "20260911-cmi-flu-e12c-manual-submission-notebook-001"
OLD_TARGET = "renta0426/cmi-flu-e12c-manual-submission-20260911-001"
OLD_SCIENCE = "bb6f41ed81b0bbd4ecdc397c6abb9df671c6ebf8"
HISTORICAL_SHA256 = "365607d59cd530656b929a1c1c57412cc6d375265a8d1ba10d304c64e012f387"
TASK13_SHA256 = "de8bc3b6bbd3e3aad4099b83eb63a1ebbd81c4f4eeb60f77808858f4674be5da"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def replace_function(text: str, name: str, replacement: str) -> str:
    tree = ast.parse(text)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(nodes) != 1:
        raise SystemExit(f"E12c-v2 generated-runtime function binding changed:{name}:{len(nodes)}")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    return "".join(lines[: node.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno :])


def validate_request(root: Path) -> None:
    request = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    exact = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "execution_policy": "kaggle_native_capacity_v2",
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "save_kernel_once",
        "target": TARGET,
        "science_source_commit": SCIENCE_COMMIT,
        "strategy_e12c_v2_blob_sha": E12C_V2_BLOB,
        "strategy_e12c_v2_contract_blob_sha": E12C_V2_CONTRACT_BLOB,
        "automatic_compute_retries": 0,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "competition_submission_authorized": False,
        "manual_operator_submission_only": True,
        "leaderboard_used_for_selection": False,
    }
    for key, value in exact.items():
        if request.get(key) != value:
            raise SystemExit(f"E12c-v2 request mismatch:{key}")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 10,
        "hard_timeout_minutes": 30,
    }:
        raise SystemExit("E12c-v2 resource contract mismatch")
    hist = request.get("historical_control") or {}
    if hist.get("sha256") != HISTORICAL_SHA256 or hist.get("expected_current_version") != 1:
        raise SystemExit("E12c-v2 historical contract mismatch")
    task13 = request.get("task13_contract") or {}
    if task13.get("expected_prediction_sha256") != TASK13_SHA256:
        raise SystemExit("E12c-v2 Task1.3 contract mismatch")
    assembly = request.get("assembly_contract") or {}
    if assembly.get("changed_tasks") != ["Task1.3"] or assembly.get("refit_unchanged_tasks") is not False:
        raise SystemExit("E12c-v2 assembly contract mismatch")
    parent = request.get("parent_failed_v1") or {}
    if (
        parent.get("version") != 1
        or parent.get("terminal_state") != "ERROR"
        or parent.get("failure_stage") != "verify_frozen_candidate"
        or parent.get("version_consumed") is not True
        or parent.get("same_version_rerun_forbidden") is not True
    ):
        raise SystemExit("E12c-v2 parent incident contract mismatch")


def load_exact_science(root: Path) -> tuple[str, str]:
    source_raw = (root / E12C_V2_PATH).read_bytes()
    if git_blob(source_raw) != E12C_V2_BLOB:
        raise SystemExit("E12c-v2 source relay blob mismatch")
    source = source_raw.decode("utf-8")
    compile(source, "cmi_flu/strategy_e12c_v2.py", "exec")
    contract_raw = (root / E12C_V2_CONTRACT_PATH).read_bytes()
    if git_blob(contract_raw) != E12C_V2_CONTRACT_BLOB:
        raise SystemExit("E12c-v2 contract relay blob mismatch")
    contract = json.loads(contract_raw)
    if contract.get("experiment") != "strategy_v2_e12c_v2_historical_backbone_manual_submission":
        raise SystemExit("E12c-v2 contract identity mismatch")
    if (contract.get("assembly") or {}).get("changed_tasks") != ["Task1.3"]:
        raise SystemExit("E12c-v2 contract changed-task mismatch")
    if (contract.get("task13") or {}).get("expected_prediction_sha256") != TASK13_SHA256:
        raise SystemExit("E12c-v2 contract Task1.3 fingerprint mismatch")
    return source, contract_raw.decode("utf-8")


def build_runtime(root: Path, output: Path, source: str) -> str:
    base = root / BASE_BUILDER
    if git_blob(base.read_bytes()) != BASE_BUILDER_BLOB:
        raise SystemExit("E12c-v2 base builder changed")
    with tempfile.TemporaryDirectory(prefix="cmi-e12c-v2-base-") as tmp:
        generated = Path(tmp) / "e12c-v1.py"
        subprocess.run(
            [sys.executable, str(base), "--repository-root", str(root), "--output", str(generated)],
            check=True,
        )
        runtime = generated.read_text(encoding="utf-8")

    for old, new, label in (
        (OLD_REQUEST, REQUEST_ID, "request"),
        (OLD_TARGET, TARGET, "target"),
        (OLD_SCIENCE, SCIENCE_COMMIT, "science"),
    ):
        if old not in runtime:
            raise SystemExit(f"E12c-v2 base runtime {label} anchor missing")
        runtime = runtime.replace(old, new)
        if old in runtime:
            raise SystemExit(f"E12c-v2 base runtime retained old {label}")

    marker = 'E12C_BLOB = "a495584a7e461478bd1a41d01d2536c4435a8f7f"\n'
    if runtime.count(marker) != 1:
        raise SystemExit("E12c-v2 old E12c constant anchor changed")
    runtime = runtime.replace(
        marker,
        marker
        + f'E12C_V2_BLOB = "{E12C_V2_BLOB}"\n'
        + f'E12C_V2_SOURCE = {source!r}\n'
        + f'HISTORICAL_BACKBONE_SHA256 = "{HISTORICAL_SHA256}"\n'
        + f'EXPECTED_TASK13_SHA256 = "{TASK13_SHA256}"\n',
        1,
    )

    loader = r'''def load_e12c_v2_module() -> object:
    load_e12c_module()
    module = types.ModuleType("cmi_flu.strategy_e12c_v2")
    module.__file__ = "<cmi_flu.strategy_e12c_v2>"
    module.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_e12c_v2"] = module
    exec(compile(E12C_V2_SOURCE, "cmi_flu/strategy_e12c_v2.py", "exec"), module.__dict__, module.__dict__)
    run = getattr(module, "run_e12c_v2", None)
    canonical = getattr(module, "canonical_csv_bytes", None)
    if not callable(run) or not callable(canonical):
        raise BridgeContractError("e12c_v2_entry_missing")
    return module
'''
    execute_marker = "def execute(input_dir: Path, output_dir: Path) -> int:\n"
    if runtime.count(execute_marker) != 1:
        raise SystemExit("E12c-v2 execute insertion anchor changed")
    runtime = runtime.replace(execute_marker, loader + "\n" + execute_marker, 1)

    self_test = r'''def self_test() -> int:
    package = package_bytes()
    if git_blob_sha(E01_SOURCE.encode("utf-8")) != E01_BLOB:
        raise BridgeContractError("e01_blob_mismatch")
    if git_blob_sha(E01_V2_SOURCE.encode("utf-8")) != E01_V2_BLOB:
        raise BridgeContractError("e01_v2_blob_mismatch")
    if git_blob_sha(CONFIG_TEXT.encode("utf-8")) != CONFIG_BLOB:
        raise BridgeContractError("config_blob_mismatch")
    if git_blob_sha(E12A_V2_SOURCE.encode("utf-8")) != E12A_V2_BLOB:
        raise BridgeContractError("e12a_v2_blob_mismatch")
    if git_blob_sha(E12B_SOURCE.encode("utf-8")) != E12B_BLOB:
        raise BridgeContractError("e12b_blob_mismatch")
    if git_blob_sha(E12C_SOURCE.encode("utf-8")) != E12C_BLOB:
        raise BridgeContractError("e12c_v1_blob_mismatch")
    if git_blob_sha(E12C_V2_SOURCE.encode("utf-8")) != E12C_V2_BLOB:
        raise BridgeContractError("e12c_v2_blob_mismatch")
    compile(E12C_V2_SOURCE, "cmi_flu/strategy_e12c_v2.py", "exec")
    if HISTORICAL_BACKBONE_SHA256 != "365607d59cd530656b929a1c1c57412cc6d375265a8d1ba10d304c64e012f387":
        raise BridgeContractError("e12c_v2_historical_hash_changed")
    if EXPECTED_TASK13_SHA256 != "de8bc3b6bbd3e3aad4099b83eb63a1ebbd81c4f4eeb60f77808858f4674be5da":
        raise BridgeContractError("e12c_v2_task13_hash_changed")
    print(
        "CMI_FLU_E12C_V2_RUNTIME_SELF_TEST PASS "
        f"request_id={REQUEST_ID} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} "
        f"e12c_v2_blob={E12C_V2_BLOB} refit_unchanged=false submission_api=false"
    )
    return 0
'''
    runtime = replace_function(runtime, "self_test", self_test)

    execute = r'''def execute(input_dir: Path, output_dir: Path) -> int:
    runtime_root = Path("/tmp") / "cmi-flu-e12c-v2-runtime"
    submission_path = output_dir / "submission.csv"
    manifest_path = output_dir / "manual-submission-manifest.json"
    stage = "initialize"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if submission_path.exists() or manifest_path.exists():
            raise BridgeContractError("e12c_v2_declared_output_already_exists")
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        runtime_root.mkdir(parents=True)

        stage = "materialize_package"
        package = package_bytes()
        package_path = runtime_root / "cmi_flu_bundle.zip"
        package_path.write_bytes(package)
        sys.path.insert(0, str(package_path))

        stage = "prepare_tree"
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
        compat.write_text(
            CONFIG_TEXT.replace("baseline: b021_taskwise_robust", "baseline: b02_taskwise_compact", 1),
            encoding="utf-8",
        )

        stage = "install_b21_adapter"
        adapter_ns = {}
        exec(compile(B21_ADAPTER_SOURCE, "<b21_runtime_adapter>", "exec"), adapter_ns, adapter_ns)
        install = adapter_ns.get("install")
        if not callable(install):
            raise BridgeContractError("b21_adapter_install_missing")
        install()

        stage = "load_inputs"
        from cmi_flu.configuration import load_baseline_config
        from cmi_flu.runner import load_inputs
        config = load_baseline_config(compat, repository_root=runtime_root)
        raw = dict(config.raw)
        raw["baseline"] = "b021_taskwise_robust"
        config = replace(config, source_path=canonical_config, raw=raw, baseline="b021_taskwise_robust")
        if not config.verify_md5 or str(config.section("selection").get("policy", "")) != "robust_v1":
            raise BridgeContractError("runtime_config_contract_mismatch")
        inputs = load_inputs(config)
        if inputs.checksum_report is None:
            raise BridgeContractError("md5_verification_missing")

        stage = "assemble_historical_backbone"
        e12c_v2 = load_e12c_v2_module()
        candidate, aggregate = e12c_v2.run_e12c_v2(config, inputs, Path("/kaggle/input"))
        result = json_safe(dict(aggregate))
        if result.get("experiment") != "strategy_v2_e12c_v2_historical_backbone_manual_submission":
            raise BridgeContractError("e12c_v2_identity")
        if result.get("historical_backbone_sha256") != HISTORICAL_BACKBONE_SHA256:
            raise BridgeContractError("e12c_v2_historical_hash")
        if result.get("changed_tasks_vs_historical_0_218") != ["Task1.3"]:
            raise BridgeContractError("e12c_v2_changed_tasks")
        if result.get("task13_prediction_sha256") != EXPECTED_TASK13_SHA256:
            raise BridgeContractError("e12c_v2_task13_hash")
        if int(result.get("task13_unique_values", 0)) != 36:
            raise BridgeContractError("e12c_v2_task13_unique")
        if int(result.get("submission_rows", 0)) != 40:
            raise BridgeContractError("e12c_v2_rows")
        if result.get("manual_submission_ready") is not True:
            raise BridgeContractError("e12c_v2_ready")
        if result.get("competition_submission_attempted") is not False:
            raise BridgeContractError("e12c_v2_submission_boundary")
        if result.get("leaderboard_used_for_selection") is not False:
            raise BridgeContractError("e12c_v2_leaderboard_boundary")
        if result.get("task13_historical_reproduced") is not True:
            raise BridgeContractError("e12c_v2_task13_reproduction")

        stage = "materialize_submission"
        canonical = e12c_v2.canonical_csv_bytes(candidate)
        generated_sha = hashlib.sha256(canonical).hexdigest()
        generated_bytes = len(canonical)
        if generated_sha != result.get("canonical_csv_sha256") or generated_bytes != int(result.get("canonical_csv_bytes", -1)):
            raise BridgeContractError("e12c_v2_generated_fingerprint_inconsistent")
        submission_path.write_bytes(canonical)
        if sha256_file(submission_path) != generated_sha or submission_path.stat().st_size != generated_bytes:
            raise BridgeContractError("e12c_v2_persisted_submission_fingerprint_mismatch")

        stage = "write_manifest"
        manifest = {
            "schema_version": 1,
            "request_id": REQUEST_ID,
            "experiment": result["experiment"],
            "science_commit": SCIENCE_COMMIT,
            "strategy_e12c_v2_blob_sha": E12C_V2_BLOB,
            "historical_backbone_sha256": HISTORICAL_BACKBONE_SHA256,
            "historical_public_score": 0.218,
            "changed_tasks_vs_historical_0_218": ["Task1.3"],
            "task13_prediction_sha256": EXPECTED_TASK13_SHA256,
            "task13_unique_values": 36,
            "task13_historical_reproduced": True,
            "submission_filename": "submission.csv",
            "submission_sha256": generated_sha,
            "submission_bytes": generated_bytes,
            "submission_rows": 40,
            "semantic_submission_sha256": result["semantic_submission_sha256"],
            "md5_verified_count": len(inputs.checksum_report.verified),
            "manual_submission_ready": True,
            "manual_operator_submission_only": True,
            "competition_submission_attempted": False,
            "competition_submission_authorized": False,
            "leaderboard_used_for_selection": False,
            "refit_unchanged_tasks": False,
            "contains_participant_identifiers": False,
            "contains_row_level_predictions": False,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(runtime_root, ignore_errors=True)
        print(
            "CMI_FLU_E12C_V2_MANUAL_SUBMISSION_READY "
            f"request_id={REQUEST_ID} rows=40 submission_sha256={generated_sha} "
            f"submission_bytes={generated_bytes} changed_vs_0_218=Task1.3 "
            "refit_unchanged=false submission_api=false"
        )
        return 0
    except Exception as exc:
        shutil.rmtree(runtime_root, ignore_errors=True)
        submission_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        code = hashlib.sha256(
            f"{type(exc).__name__}:{str(exc)}".encode("utf-8", errors="replace")
        ).hexdigest()[:20]
        print(
            f"CMI_FLU_E12C_V2_FAILED stage={stage} "
            f"exception_type={type(exc).__name__} error_code={code}",
            file=sys.stderr,
        )
        return 2
'''
    runtime = replace_function(runtime, "execute", execute)
    runtime = runtime.replace("CMI_FLU_E12C_MANUAL_FAILED", "CMI_FLU_E12C_V2_FAILED")
    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'TARGET_KERNEL = "{TARGET}"',
        f'SCIENCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'E12C_V2_BLOB = "{E12C_V2_BLOB}"',
        "def load_e12c_v2_module(",
        "CMI_FLU_E12C_V2_MANUAL_SUBMISSION_READY",
        "refit_unchanged=false",
        "changed_tasks_vs_historical_0_218",
    )
    if any(token not in runtime for token in required):
        raise SystemExit("E12c-v2 generated runtime contract incomplete")
    if OLD_TARGET in runtime or OLD_REQUEST in runtime:
        raise SystemExit("E12c-v2 runtime retained consumed v1 identity")
    lowered = runtime.casefold()
    for token in ("competition_submit(", "kaggle competitions submit"):
        if token in lowered:
            raise SystemExit("E12c-v2 generated runtime contains Competition submission path")
    encoded = runtime.encode("utf-8")
    if len(encoded) >= 950000:
        raise SystemExit(f"E12c-v2 runtime too large:{len(encoded)}")
    compile(runtime, "generated_e12c_v2.py", "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    return runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validate_request(root)
    source, _ = load_exact_science(root)
    runtime = build_runtime(root, output, source)
    encoded = runtime.encode("utf-8")
    print(
        "CMI_FLU_E12C_V2_PREPARE PASS "
        f"request_id={REQUEST_ID} science_commit={SCIENCE_COMMIT} "
        f"e12c_v2_blob={E12C_V2_BLOB} runtime_bytes={len(encoded)} "
        f"runtime_sha256={hashlib.sha256(encoded).hexdigest()} "
        "refit_unchanged=false submission_api=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
