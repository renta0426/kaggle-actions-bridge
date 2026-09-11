#!/usr/bin/env python3
"""Build the exact private Kaggle runtime that materializes the frozen E12c submission.csv."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REQUEST_ID = "20260911-cmi-flu-e12c-manual-submission-notebook-001"
TARGET = "renta0426/cmi-flu-e12c-manual-submission-20260911-001"
TITLE = "CMI Flu E12c Manual Submission 20260911 001"
SCIENCE_COMMIT = "bb6f41ed81b0bbd4ecdc397c6abb9df671c6ebf8"
E12C_BLOB = "a495584a7e461478bd1a41d01d2536c4435a8f7f"
E12B_BLOB = "af5df92ca81abc34a7f7046dba5bc284c98302d4"
E12A_V2_BLOB = "0c4c970c8bacfed61bfbb9587e0a0bfdec7903d9"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
FINAL_SEMANTIC_SHA256 = "5faf93bee4b68aba23c53e09adbb251bdd07d287dd48fb6d94d6fce53ef9a60f"
FINAL_CSV_SHA256 = "983aaf097d04477c4ccf7bf817fdf66e552937e69bceaa48cfb260cb84413f1b"
FINAL_CSV_BYTES = 5933
HISTORICAL_SHA256 = "365607d59cd530656b929a1c1c57412cc6d375265a8d1ba10d304c64e012f387"
HISTORICAL_SOURCE = "renta0426/cmi-flu-manual-probe-files-20260906-001"
HISTORICAL_VERSION = 1
PAYLOAD_PATH = "payloads/cmi-flu-e12c-manual-submission-notebook-001/strategy_e12c.py"
REQUEST_PATH = "requests/cmi-flu-e12c-manual-submission-notebook-001.json"
BASE_BUILDER = "scripts/cmi_flu_strategy_e12b_prepare_v2.py"
BASE_BUILDER_BLOB = "9bee85eec86f919acf860671e97cfd334e704d24"
BASE_REQUEST = "20260911-cmi-flu-strategy-e12b-challenge-freeze-001"
BASE_TARGET = "renta0426/cmi-flu-e12b-challenge-freeze-20260911-001"
BASE_SCIENCE = "a411bf85a4a79fec2e9a2b9c2bc8cbe186adee5f"


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
        raise SystemExit(f"E12c generated-runtime function binding changed:{name}:{len(nodes)}")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    return (
        "".join(lines[: node.lineno - 1])
        + replacement.rstrip()
        + "\n"
        + "".join(lines[node.end_lineno :])
    )


def validate_request(root: Path) -> None:
    request = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "execution_policy": "kaggle_native_capacity_v2",
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "save_kernel_once",
        "target": TARGET,
        "title": TITLE,
        "science_repository": "renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": "connector_verified_exact_blob_relay",
        "strategy_e12c_blob_sha": E12C_BLOB,
        "automatic_compute_retries": 0,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "competition_submission_authorized": False,
        "manual_operator_submission_only": True,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"E12c request mismatch:{key}")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 30,
        "hard_timeout_minutes": 60,
    }:
        raise SystemExit("E12c resource contract mismatch")
    parent = request.get("parent_e12b") or {}
    if parent != {
        "request_id": BASE_REQUEST,
        "target": BASE_TARGET,
        "version": 1,
        "actions_run": 34553673283,
        "actions_job": 103121695346,
        "kernel_terminal_state": "COMPLETE",
        "semantic_submission_sha256": FINAL_SEMANTIC_SHA256,
        "canonical_csv_sha256": FINAL_CSV_SHA256,
        "canonical_csv_bytes": FINAL_CSV_BYTES,
    }:
        raise SystemExit("E12c parent E12b contract mismatch")
    historical = request.get("historical_control") or {}
    if historical != {
        "source_kernel": HISTORICAL_SOURCE,
        "expected_current_version": HISTORICAL_VERSION,
        "expected_terminal_state": "COMPLETE",
        "source_filename": "submission-task12-only.csv",
        "sha256": HISTORICAL_SHA256,
        "public_score": 0.218,
        "required_changed_tasks": ["Task1.3"],
    }:
        raise SystemExit("E12c historical-control contract mismatch")
    output = request.get("output_contract") or {}
    if output != {
        "submission_filename": "submission.csv",
        "submission_sha256": FINAL_CSV_SHA256,
        "submission_bytes": FINAL_CSV_BYTES,
        "manifest_filename": "manual-submission-manifest.json",
        "row_level_submission_may_exist_only_in_private_kaggle_output_and_ephemeral_runner_validation": True,
        "public_artifact_upload": False,
    }:
        raise SystemExit("E12c output contract mismatch")
    if request.get("side_effects") != [
        "create exactly one private Kaggle Notebook version",
        "persist exact validated submission.csv and aggregate manifest in that private Notebook output",
    ]:
        raise SystemExit("E12c side-effect contract mismatch")


def load_exact_e12c(root: Path) -> str:
    path = root / PAYLOAD_PATH
    raw = path.read_bytes()
    found = git_blob(raw)
    if found != E12C_BLOB:
        raise SystemExit(f"E12c science relay blob mismatch:{found}")
    source = raw.decode("utf-8")
    compile(source, "cmi_flu/strategy_e12c.py", "exec")
    required = (
        'EXPECTED_E12B_SEMANTIC_SHA256 = (',
        FINAL_SEMANTIC_SHA256,
        FINAL_CSV_SHA256,
        HISTORICAL_SHA256,
        'EXPECTED_CHANGED_TASKS_VS_HISTORICAL = ("Task1.3",)',
        'float_precision="round_trip"',
        "def verify_frozen_candidate(",
        "def validate_historical_task12_only(",
    )
    if any(token not in source for token in required):
        raise SystemExit("E12c science source contract token missing")
    lowered = source.casefold()
    for token in ("competition_submit(", "kaggle competitions submit", "requests.get(", "urllib.request"):
        if token in lowered:
            raise SystemExit(f"E12c science contains forbidden network/submission path:{token}")
    return source


def build_runtime(root: Path, output: Path, e12c_source: str) -> str:
    base = root / BASE_BUILDER
    raw = base.read_bytes()
    if git_blob(raw) != BASE_BUILDER_BLOB:
        raise SystemExit("E12c base E12b builder changed")
    with tempfile.TemporaryDirectory(prefix="cmi-e12c-base-") as tmp:
        generated = Path(tmp) / "e12b.py"
        subprocess.run(
            [
                sys.executable,
                str(base),
                "--repository-root",
                str(root),
                "--output",
                str(generated),
            ],
            check=True,
        )
        runtime = generated.read_text(encoding="utf-8")

    for old, new, label in (
        (BASE_REQUEST, REQUEST_ID, "request"),
        (BASE_TARGET, TARGET, "target"),
        (BASE_SCIENCE, SCIENCE_COMMIT, "science"),
    ):
        if old not in runtime:
            raise SystemExit(f"E12c base runtime {label} anchor missing")
        runtime = runtime.replace(old, new)
        if old in runtime:
            raise SystemExit(f"E12c base runtime retained old {label}")

    marker = f'E12B_CONTRACT_BLOB = "7e844b2d56bd6739793888bf6306946a0028d966"\n'
    if runtime.count(marker) != 1:
        raise SystemExit("E12c E12b constant anchor changed")
    runtime = runtime.replace(
        marker,
        marker
        + f'E12C_BLOB = "{E12C_BLOB}"\n'
        + f'E12C_SOURCE = {e12c_source!r}\n'
        + f'FINAL_SEMANTIC_SHA256 = "{FINAL_SEMANTIC_SHA256}"\n'
        + f'FINAL_CSV_SHA256 = "{FINAL_CSV_SHA256}"\n'
        + f'FINAL_CSV_BYTES = {FINAL_CSV_BYTES}\n'
        + f'HISTORICAL_TASK12_ONLY_SHA256 = "{HISTORICAL_SHA256}"\n'
        + f'HISTORICAL_SOURCE_KERNEL = "{HISTORICAL_SOURCE}"\n'
        + f'HISTORICAL_SOURCE_VERSION = {HISTORICAL_VERSION}\n',
        1,
    )

    loader = r'''def load_e12c_module() -> object:
    if "cmi_flu.strategy_e12b" not in sys.modules:
        load_e12b_module()
    module = types.ModuleType("cmi_flu.strategy_e12c")
    module.__file__ = "<cmi_flu.strategy_e12c>"
    module.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_e12c"] = module
    exec(compile(E12C_SOURCE, "cmi_flu/strategy_e12c.py", "exec"), module.__dict__, module.__dict__)
    for name in ("verify_frozen_candidate", "validate_historical_task12_only", "canonical_csv_bytes"):
        if not callable(getattr(module, name, None)):
            raise BridgeContractError(f"e12c_entry_missing:{name}")
    return module
'''
    execute_marker = "def execute(input_dir: Path, output_dir: Path) -> int:\n"
    if runtime.count(execute_marker) != 1:
        raise SystemExit("E12c execute insertion anchor changed")
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
        raise BridgeContractError("e12c_blob_mismatch")
    compile(B21_ADAPTER_SOURCE, "cmi_flu_b21_runtime_adapter.py", "exec")
    compile(E01_SOURCE, "cmi_flu/strategy_e01.py", "exec")
    compile(E01_V2_SOURCE, "cmi_flu/strategy_e01_v2.py", "exec")
    compile(E12A_V2_SOURCE, "cmi_flu/strategy_e12a_v2.py", "exec")
    compile(E12B_SOURCE, "cmi_flu/strategy_e12b.py", "exec")
    compile(E12C_SOURCE, "cmi_flu/strategy_e12c.py", "exec")
    if FINAL_CSV_SHA256 != "983aaf097d04477c4ccf7bf817fdf66e552937e69bceaa48cfb260cb84413f1b":
        raise BridgeContractError("e12c_final_csv_hash_changed")
    if HISTORICAL_TASK12_ONLY_SHA256 != "365607d59cd530656b929a1c1c57412cc6d375265a8d1ba10d304c64e012f387":
        raise BridgeContractError("e12c_historical_hash_changed")
    print(
        "CMI_FLU_E12C_MANUAL_RUNTIME_SELF_TEST PASS "
        f"request_id={REQUEST_ID} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} "
        f"e12c_blob={E12C_BLOB} output=submission.csv submission_api=false"
    )
    return 0
'''
    runtime = replace_function(runtime, "self_test", self_test)

    execute = r'''def execute(input_dir: Path, output_dir: Path) -> int:
    runtime_root = Path("/tmp") / "cmi-flu-e12c-manual-runtime"
    submission_path = output_dir / "submission.csv"
    manifest_path = output_dir / "manual-submission-manifest.json"
    stage = "initialize"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if submission_path.exists() or manifest_path.exists():
            raise BridgeContractError("e12c_declared_output_already_exists")
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
        if CONFIG_TEXT.count("baseline: b021_taskwise_robust") != 1:
            raise BridgeContractError("b021_config_anchor_changed")
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
        config = replace(
            config,
            source_path=canonical_config,
            raw=raw,
            baseline="b021_taskwise_robust",
        )
        if not config.verify_md5 or str(config.section("selection").get("policy", "")) != "robust_v1":
            raise BridgeContractError("runtime_config_contract_mismatch")
        inputs = load_inputs(config)
        if inputs.checksum_report is None:
            raise BridgeContractError("md5_verification_missing")

        stage = "run_e12b"
        run_e12b = load_e12b_module()
        candidate, aggregate = run_e12b(config, inputs)
        aggregate_safe = json_safe(dict(aggregate))
        validate_result(aggregate_safe)

        stage = "load_e12c"
        e12c = load_e12c_module()
        sample = inputs.tables["sample_submission"]
        stage = "verify_frozen_candidate"
        frozen = dict(e12c.verify_frozen_candidate(candidate, aggregate, sample))
        if frozen.get("semantic_submission_sha256") != FINAL_SEMANTIC_SHA256:
            raise BridgeContractError("e12c_semantic_hash_mismatch")
        if frozen.get("canonical_csv_sha256") != FINAL_CSV_SHA256:
            raise BridgeContractError("e12c_canonical_hash_mismatch")
        if int(frozen.get("canonical_csv_bytes", -1)) != FINAL_CSV_BYTES:
            raise BridgeContractError("e12c_canonical_size_mismatch")

        stage = "verify_historical_control"
        historical = dict(
            e12c.validate_historical_task12_only(
                Path("/kaggle/input"),
                final_submission=candidate,
                sample_submission=sample,
                expected_sha256=HISTORICAL_TASK12_ONLY_SHA256,
            )
        )
        if historical.get("historical_task12_only_sha256") != HISTORICAL_TASK12_ONLY_SHA256:
            raise BridgeContractError("e12c_historical_hash_mismatch")
        if historical.get("changed_tasks_vs_historical_task12_only") != ["Task1.3"]:
            raise BridgeContractError("e12c_historical_changed_tasks_mismatch")
        if historical.get("task13_only_intervention_proven") is not True:
            raise BridgeContractError("e12c_task13_only_not_proven")

        stage = "materialize_submission"
        canonical = e12c.canonical_csv_bytes(candidate)
        if len(canonical) != FINAL_CSV_BYTES or hashlib.sha256(canonical).hexdigest() != FINAL_CSV_SHA256:
            raise BridgeContractError("e12c_prewrite_submission_fingerprint_mismatch")
        submission_path.write_bytes(canonical)
        if sha256_file(submission_path) != FINAL_CSV_SHA256 or submission_path.stat().st_size != FINAL_CSV_BYTES:
            raise BridgeContractError("e12c_persisted_submission_fingerprint_mismatch")

        stage = "write_manifest"
        manifest = {
            "schema_version": 1,
            "request_id": REQUEST_ID,
            "experiment": "strategy_v2_e12c_manual_submission_generator",
            "science_commit": SCIENCE_COMMIT,
            "strategy_e12c_blob_sha": E12C_BLOB,
            "parent_e12b_semantic_sha256": FINAL_SEMANTIC_SHA256,
            "submission_filename": "submission.csv",
            "submission_sha256": FINAL_CSV_SHA256,
            "submission_bytes": FINAL_CSV_BYTES,
            "submission_rows": int(frozen["submission_rows"]),
            "submission_columns": list(frozen["submission_columns"]),
            "historical_source_kernel": HISTORICAL_SOURCE_KERNEL,
            "historical_source_version": HISTORICAL_SOURCE_VERSION,
            "historical_task12_only_sha256": HISTORICAL_TASK12_ONLY_SHA256,
            "changed_tasks_vs_historical_task12_only": list(
                historical["changed_tasks_vs_historical_task12_only"]
            ),
            "task13_only_intervention_proven": True,
            "md5_verified_count": len(inputs.checksum_report.verified),
            "manual_submission_ready": True,
            "manual_operator_submission_only": True,
            "competition_submission_attempted": False,
            "competition_submission_authorized": False,
            "leaderboard_used_for_selection": False,
            "contains_participant_identifiers": False,
            "contains_row_level_predictions": False,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(runtime_root, ignore_errors=True)
        print(
            "CMI_FLU_E12C_MANUAL_SUBMISSION_READY "
            f"request_id={REQUEST_ID} rows={manifest['submission_rows']} "
            f"submission_sha256={FINAL_CSV_SHA256} submission_bytes={FINAL_CSV_BYTES} "
            "changed_vs_0_218=Task1.3 submission_api=false"
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
            f"CMI_FLU_E12C_MANUAL_FAILED stage={stage} "
            f"exception_type={type(exc).__name__} error_code={code}",
            file=sys.stderr,
        )
        return 2
'''
    runtime = replace_function(runtime, "execute", execute)
    runtime = runtime.replace(
        '"""CMI-Flu strategy E01 paired evaluation. Aggregate outputs only; no submission."""',
        '"""CMI-Flu E12c private manual-submission generator; no Competition submit call."""',
    )
    runtime = runtime.replace("CMI_FLU_E12B_FAILED", "CMI_FLU_E12C_MANUAL_FAILED")

    required_runtime = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'TARGET_KERNEL = "{TARGET}"',
        f'SCIENCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'E12C_BLOB = "{E12C_BLOB}"',
        f'FINAL_CSV_SHA256 = "{FINAL_CSV_SHA256}"',
        f'HISTORICAL_TASK12_ONLY_SHA256 = "{HISTORICAL_SHA256}"',
        "def load_e12c_module(",
        'submission_path = output_dir / "submission.csv"',
        'Path("/kaggle/input")',
        "CMI_FLU_E12C_MANUAL_SUBMISSION_READY",
        '"Task1.3"',
        '"competition_submission_attempted": False',
    )
    if any(token not in runtime for token in required_runtime):
        raise SystemExit("E12c generated runtime contract incomplete")
    if BASE_TARGET in runtime or BASE_REQUEST in runtime:
        raise SystemExit("E12c runtime references consumed E12b target identity")
    lowered = runtime.casefold()
    for token in ("competition_submit(", "kaggle competitions submit"):
        if token in lowered:
            raise SystemExit("E12c generated runtime contains Competition submission path")
    encoded = runtime.encode("utf-8")
    if len(encoded) >= 900000:
        raise SystemExit(f"E12c runtime too large:{len(encoded)}")
    compile(runtime, "generated_e12c_manual.py", "exec")
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
    e12c_source = load_exact_e12c(root)
    runtime = build_runtime(root, output, e12c_source)
    encoded = runtime.encode("utf-8")
    print(
        "CMI_FLU_E12C_MANUAL_PREPARE PASS "
        f"request_id={REQUEST_ID} science_commit={SCIENCE_COMMIT} e12c_blob={E12C_BLOB} "
        f"runtime_bytes={len(encoded)} runtime_sha256={hashlib.sha256(encoded).hexdigest()} "
        f"submission_sha256={FINAL_CSV_SHA256} submission_api=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
