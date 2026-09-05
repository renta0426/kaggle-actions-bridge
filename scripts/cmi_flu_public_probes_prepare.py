#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import py_compile
import subprocess
import sys
from typing import Any

REQUEST_ID = "20260906-cmi-flu-public-probes-001"
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_COMMIT = "7a85abcbf8282bc8bfe047b7db02a72f35222caa"
SCIENCE_PATH = "src/cmi_flu/public_probes.py"
SCIENCE_BLOB = "497a8875d440e1b8a03d4ef938bc3a834213b33d"
SCIENCE_TRANSPORT = "agent_relay_exact_blob"
RELAYED_SCIENCE_PATH = "payloads/cmi-flu-public-probes-001/public_probes.py"
B21_BASE_REQUEST_ID = "20260903-cmi-flu-b21-001"
B21_BASE_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
BASE_SHA256 = "7894a9232e7372950f70a36d35de41bcc8bda90ec4d0320187e82c3b8c76f2db"
BASE_SIZE = 93363
TARGET_COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-public-probes-20260906-001"
TARGET_STAGE = "controlled_public_probes_task12_task13"
EXPECTED_KERNEL_VERSION = 1
BACKBONE_KERNEL = "renta0426/cmi-flu-b21-robust-cv-20260903-001"
BACKBONE_EXPECTED_VERSION = 1
PROBE_NAMES = ("task13_only", "task12_only", "task12_task13")
OUTPUT_FILENAMES = {
    "task13_only": "probe-task13.csv",
    "task12_only": "probe-task12.csv",
    "task12_task13": "probe-task12-task13.csv",
}
BACKBONE_B64_PLACEHOLDER = "__CMI_FLU_BACKBONE_B64__"
BACKBONE_SHA_PLACEHOLDER = "__CMI_FLU_BACKBONE_SHA256__"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def load_science(root: pathlib.Path) -> tuple[str, str]:
    path = root / RELAYED_SCIENCE_PATH
    data = path.read_bytes()
    found = git_blob_sha(data)
    if found != SCIENCE_BLOB:
        raise SystemExit(f"controlled-probe science blob mismatch: {found}")
    source = data.decode("utf-8")
    compile(source, SCIENCE_PATH, "exec")
    required = (
        'TASK12_MODEL_NAME = "et_d5_l5_sqrt"',
        "TASK12_LAMBDA = 0.5",
        'TASK13_MODEL_NAME = "enet_a0.1_l0.5"',
        'PROBE_NAMES = ("task13_only", "task12_only", "task12_task13")',
        "matched_participants",
        "len(harmonized.train) != 68",
        "build_controlled_public_probes",
        '"public_scores_used_to_define_family": False',
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise SystemExit(f"controlled-probe science contract tokens missing: {missing}")
    forbidden = ("competition_submit", "kaggle competitions submit", "leaderboard")
    present = [token for token in forbidden if token in source.casefold()]
    if present:
        raise SystemExit(f"controlled-probe science contains forbidden submit/LB tokens: {present}")
    return source, sha256_bytes(data)


def validate_request(root: pathlib.Path) -> dict[str, Any]:
    request = json.loads(
        (root / "requests/cmi-flu-public-probes-001.json").read_text(encoding="utf-8")
    )
    expected = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": TARGET_COMPETITION,
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_repository": SCIENCE_REPOSITORY,
        "science_source_commit": SCIENCE_COMMIT,
        "science_source_path": SCIENCE_PATH,
        "science_transport": SCIENCE_TRANSPORT,
        "public_probes_blob_sha": SCIENCE_BLOB,
        "relayed_science_path": RELAYED_SCIENCE_PATH,
        "backbone_kernel": BACKBONE_KERNEL,
        "backbone_expected_version": BACKBONE_EXPECTED_VERSION,
        "b21_base_request_id": B21_BASE_REQUEST_ID,
        "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB,
        "expected_kernel_version": EXPECTED_KERNEL_VERSION,
        "probe_family": list(PROBE_NAMES),
        "competition_submission_attempted": False,
        "public_scores_used_to_define_family": False,
        "automatic_compute_retries": 0,
        "enable_internet": False,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"controlled-probe request mismatch: {key}")
    expected_paths = [
        "cmi-flu-public-probes-001/probe-task13.csv",
        "cmi-flu-public-probes-001/probe-task12.csv",
        "cmi-flu-public-probes-001/probe-task12-task13.csv",
        "cmi-flu-public-probes-001/bridge-result.json",
        "cmi-flu-public-probes-001/summary.md",
    ]
    if request.get("allowed_output_paths") != expected_paths:
        raise SystemExit("controlled-probe output allowlist mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 30,
        "hard_timeout_minutes": 90,
        "max_active_runs": 1,
    }:
        raise SystemExit("controlled-probe resource contract mismatch")
    if request.get("api_budget") != {
        "max_calls": 18,
        "initial_poll_schedule_seconds": [30, 60, 120, 300],
        "steady_poll_interval_seconds": 600,
        "max_pages": 2,
    }:
        raise SystemExit("controlled-probe API budget mismatch")
    return request


def injected_helpers(science_source: str, science_sha256: str) -> str:
    return f'''\nTARGET_STAGE = "{TARGET_STAGE}"\nPUBLIC_PROBES_SOURCE_BLOB_SHA = "{SCIENCE_BLOB}"\nPUBLIC_PROBES_SOURCE_SHA256 = "{science_sha256}"\nPUBLIC_PROBES_SOURCE = {science_source!r}\nBACKBONE_KERNEL = "{BACKBONE_KERNEL}"\nBACKBONE_EXPECTED_VERSION = {BACKBONE_EXPECTED_VERSION}\nBACKBONE_SUBMISSION_B64 = "{BACKBONE_B64_PLACEHOLDER}"\nBACKBONE_SUBMISSION_SHA256 = "{BACKBONE_SHA_PLACEHOLDER}"\nPROBE_OUTPUT_FILENAMES = {OUTPUT_FILENAMES!r}\n\ndef probe_json_safe(value: Any) -> Any:\n    if value is None or isinstance(value, (str, bool, int)):\n        return value\n    if isinstance(value, float):\n        return value if math.isfinite(value) else None\n    if isinstance(value, Mapping):\n        return {{str(key): probe_json_safe(item) for key, item in value.items()}}\n    if isinstance(value, (list, tuple)):\n        return [probe_json_safe(item) for item in value]\n    item = getattr(value, "item", None)\n    if callable(item):\n        try:\n            return probe_json_safe(item())\n        except (ValueError, TypeError):\n            pass\n    return str(value)\n'''


def execute_source() -> str:
    return r'''
def execute(input_dir: Path, output_dir: Path) -> Mapping[str, Any]:
    stage = "initialize"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = output_dir / "runtime"
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True)

    try:
        stage = "verify_bound_backbone"
        import base64
        if BACKBONE_SUBMISSION_B64.startswith("__CMI_FLU_"):
            raise BundleContractError("controlled-probe runtime backbone was not bound")
        backbone_bytes = base64.b64decode(BACKBONE_SUBMISSION_B64.encode("ascii"), validate=True)
        if not backbone_bytes or len(backbone_bytes) > 262144:
            raise BundleContractError("B2.1 backbone CSV outside byte budget")
        if sha256_bytes(backbone_bytes) != BACKBONE_SUBMISSION_SHA256:
            raise BundleContractError("B2.1 backbone CSV SHA-256 mismatch")

        stage = "materialize_package"
        bundle = package_bytes()
        package_path = runtime_root / "cmi_flu_bundle.zip"
        package_path.write_bytes(bundle)
        sys.path.insert(0, str(package_path))

        stage = "prepare_runtime_tree"
        data_parent = runtime_root / "data"
        data_parent.mkdir(parents=True, exist_ok=True)
        (data_parent / "raw").symlink_to(input_dir, target_is_directory=True)
        reference_dir = (
            runtime_root / "external" / "google-drive" / "challenge-resources" / "reference_files"
        )
        derive_reference_files(input_dir, reference_dir)

        config_text = CONFIG_TEXT
        if config_text.count("baseline: b02_taskwise_compact") != 1:
            raise BundleContractError("B2 base config baseline anchor mismatch")
        if config_text.count("filename: b02_taskwise_compact.csv") == 1:
            config_text = config_text.replace(
                "filename: b02_taskwise_compact.csv", "filename: b021_taskwise_robust.csv", 1
            )
        if "\nselection:\n" in config_text:
            raise BundleContractError("B2 base config unexpectedly already has selection section")
        config_text = config_text.rstrip() + "\nselection:\n  policy: robust_v1\n"
        config_dir = runtime_root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "baseline_b021_robust.yaml"
        config_path.write_text(config_text, encoding="utf-8")

        stage = "dependency_preflight"
        import io
        import joblib
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        import yaml
        _ = (joblib.__version__, np.__version__, pd.__version__, scipy.__version__, sklearn.__version__, yaml.__version__)

        stage = "install_b21_adapter"
        adapter_namespace: dict[str, Any] = {}
        exec(compile(B21_ADAPTER_SOURCE, "<b21_runtime_adapter>", "exec"), adapter_namespace, adapter_namespace)
        install_adapter = adapter_namespace.get("install")
        if not callable(install_adapter):
            raise BundleContractError("B2.1 runtime adapter lacks install()")
        install_adapter()

        stage = "load_public_probes"
        import types
        probe_module = types.ModuleType("cmi_flu.public_probes")
        probe_module.__file__ = "<cmi_flu.public_probes>"
        probe_module.__package__ = "cmi_flu"
        sys.modules["cmi_flu.public_probes"] = probe_module
        exec(compile(PUBLIC_PROBES_SOURCE, "cmi_flu/public_probes.py", "exec"), probe_module.__dict__, probe_module.__dict__)
        build_probes = getattr(probe_module, "build_controlled_public_probes", None)
        if not callable(build_probes):
            raise BundleContractError("controlled-probe science entry point missing")
        if tuple(getattr(probe_module, "PROBE_NAMES", ())) != ("task13_only", "task12_only", "task12_task13"):
            raise BundleContractError("controlled-probe family identity mismatch")
        if getattr(probe_module, "TASK12_MODEL_NAME", None) != "et_d5_l5_sqrt" or float(getattr(probe_module, "TASK12_LAMBDA", -1)) != 0.5:
            raise BundleContractError("Task1.2 probe contract mismatch")
        if getattr(probe_module, "TASK13_MODEL_NAME", None) != "enet_a0.1_l0.5":
            raise BundleContractError("Task1.3 probe contract mismatch")

        stage = "load_inputs"
        from cmi_flu.configuration import load_baseline_config
        from cmi_flu.runner import load_inputs
        config = load_baseline_config(config_path, repository_root=runtime_root)
        if config.baseline != "b02_taskwise_compact":
            raise BundleContractError("legacy embedded loader returned unexpected baseline")
        raw_compat = dict(config.raw)
        raw_compat["baseline"] = "b021_taskwise_robust"
        object.__setattr__(config, "raw", raw_compat)
        object.__setattr__(config, "baseline", "b021_taskwise_robust")
        inputs = load_inputs(config)

        stage = "load_backbone_submission"
        backbone = pd.read_csv(io.BytesIO(backbone_bytes))
        sample = inputs.tables["sample_submission"]
        from cmi_flu.contracts import TASK_COLUMNS, validate_submission
        validate_submission(backbone, sample, require_nonconstant_public_tasks=True)
        if len(backbone) != 40:
            raise BundleContractError("B2.1 backbone row count mismatch")

        stage = "build_controlled_probes"
        variants, diagnostics = build_probes(config, inputs, backbone)
        if tuple(variants) != ("task13_only", "task12_only", "task12_task13"):
            raise BundleContractError("controlled-probe output family mismatch")
        diagnostics = probe_json_safe(dict(diagnostics))
        if diagnostics.get("public_scores_used_to_define_family") is not False:
            raise BundleContractError("controlled-probe family unexpectedly used Public scores")

        stage = "persist_probe_csvs"
        probe_hashes: dict[str, str] = {}
        for name, filename in PROBE_OUTPUT_FILENAMES.items():
            frame = variants[name]
            validate_submission(frame, sample, require_nonconstant_public_tasks=True)
            path = output_dir / filename
            frame.to_csv(path, index=False)
            probe_hashes[name] = sha256_file(path)

        stage = "validate_backbone_invariance"
        changed = {
            "task13_only": {"Task1.3"},
            "task12_only": {"Task1.2"},
            "task12_task13": {"Task1.2", "Task1.3"},
        }
        for name, frame in variants.items():
            for task in TASK_COLUMNS:
                left = pd.to_numeric(frame[task], errors="raise").to_numpy(dtype=float)
                right = pd.to_numeric(backbone[task], errors="raise").to_numpy(dtype=float)
                equal = np.array_equal(left, right, equal_nan=True)
                if task in changed[name]:
                    if equal:
                        raise BundleContractError(f"probe {name} failed to change intended task {task}")
                elif not equal:
                    raise BundleContractError(f"probe {name} changed forbidden backbone task {task}")

        stage = "write_summary"
        summary = output_dir / "summary.md"
        summary.write_text(
            "# Controlled Public probe family 001\n\n"
            f"- request_id: `{REQUEST_ID}`\n"
            f"- backbone: `{BACKBONE_KERNEL}` current version `{BACKBONE_EXPECTED_VERSION}`\n"
            f"- backbone_sha256: `{BACKBONE_SUBMISSION_SHA256}`\n"
            "- family fixed before Public scores: `true`\n"
            "- Competition submission attempted by generator: `false`\n\n"
            "Generated exactly three probes: Task1.3-only, Task1.2-only, and Task1.2+Task1.3.\n",
            encoding="utf-8",
        )

        stage = "write_manifest"
        payload = {
            "schema_version": 1,
            "request_id": REQUEST_ID,
            "competition": COMPETITION,
            "kernel_stage": TARGET_STAGE,
            "source_commit": SOURCE_COMMIT,
            "base_package_zip_sha256": PACKAGE_ZIP_SHA256,
            "public_probes_source_blob_sha": PUBLIC_PROBES_SOURCE_BLOB_SHA,
            "public_probes_source_sha256": PUBLIC_PROBES_SOURCE_SHA256,
            "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB_SHA,
            "b21_runtime_adapter_sha256": B21_ADAPTER_SHA256,
            "backbone_kernel": BACKBONE_KERNEL,
            "backbone_version": BACKBONE_EXPECTED_VERSION,
            "backbone_submission_sha256": BACKBONE_SUBMISSION_SHA256,
            "probe_sha256": probe_hashes,
            "probe_changed_columns": diagnostics.get("changed_columns"),
            "diagnostics": diagnostics,
            "competition_submission_attempted": False,
            "public_scores_used_to_define_family": False,
        }
        (output_dir / "bridge-result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        shutil.rmtree(runtime_root, ignore_errors=True)
        print(
            "CMI_FLU_PUBLIC_PROBES_COMPLETE "
            f"request_id={REQUEST_ID} backbone_sha256={BACKBONE_SUBMISSION_SHA256} "
            + " ".join(f"{name}_sha256={probe_hashes[name]}" for name in ("task13_only", "task12_only", "task12_task13"))
        )
        return payload
    except Exception as error:
        safe_failure(output_dir, stage=stage, error=error)
        shutil.rmtree(runtime_root, ignore_errors=True)
        traceback.print_exc()
        raise
'''.lstrip("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    validate_request(root)
    science_source, science_sha256 = load_science(root)
    adapter_path = root / "scripts/cmi_flu_b21_runtime_adapter.py"
    if git_blob_sha(adapter_path.read_bytes()) != B21_ADAPTER_BLOB:
        raise SystemExit("B2.1 runtime adapter blob mismatch")

    base_dir = root / "payloads/cmi-flu-b2-broad-004"
    base = b"".join((base_dir / f"part-{index:02d}").read_bytes() for index in range(20))
    if len(base) != BASE_SIZE or sha256_bytes(base) != BASE_SHA256:
        raise SystemExit("B2 004 base payload mismatch")

    work = output.parent
    base_source = work / "base-source.py"
    base004 = work / "base004.py"
    b21 = work / "b21.py"
    base_source.write_bytes(base)
    run(sys.executable, str(root / "scripts/cmi_flu_b2_patch_v4.py"), "--source", str(base_source), "--request", str(root / "requests/cmi-flu-b2-launch-v4.json"), "--output", str(base004))
    run(sys.executable, str(root / "scripts/cmi_flu_b21_patch.py"), "--source", str(base004), "--adapter", str(adapter_path), "--request", str(root / "requests/cmi-flu-b21-launch-v1.json"), "--output", str(b21))

    text = b21.read_text(encoding="utf-8")
    if "B21_ADAPTER_SOURCE" not in text:
        raise SystemExit("controlled-probe source must use materialized B2.1 runtime")
    text = replace_once(text, f'REQUEST_ID = "{B21_BASE_REQUEST_ID}"', f'REQUEST_ID = "{REQUEST_ID}"', label="request id")
    text = replace_once(text, f'SOURCE_COMMIT = "{B21_BASE_SOURCE_COMMIT}"', f'SOURCE_COMMIT = "{SCIENCE_COMMIT}"', label="science source commit")
    text = replace_once(text, 'default=Path("/kaggle/working/cmi-flu-b2")', 'default=Path("/kaggle/working/cmi-flu-public-probes-001")', label="output directory")
    text = text.replace("CMI_FLU_B2_FAILED ", "CMI_FLU_PUBLIC_PROBES_FAILED ")

    marker = "\ndef execute(input_dir: Path, output_dir: Path) -> Mapping[str, Any]:\n"
    text = replace_once(text, marker, injected_helpers(science_source, science_sha256) + marker, label="controlled-probe source insertion")
    execute_start = text.index(marker.lstrip("\n"))
    main_marker = "\ndef main() -> int:\n"
    main_start = text.index(main_marker, execute_start)
    text = text[:execute_start] + execute_source() + text[main_start:]

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)
    namespace: dict[str, Any] = {"__name__": "public_probes_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID or namespace.get("SOURCE_COMMIT") != SCIENCE_COMMIT:
        raise SystemExit("generated controlled-probe runtime identity mismatch")
    if namespace.get("TARGET_STAGE") != TARGET_STAGE:
        raise SystemExit("generated controlled-probe stage mismatch")
    if namespace.get("PUBLIC_PROBES_SOURCE_BLOB_SHA") != SCIENCE_BLOB:
        raise SystemExit("generated controlled-probe science blob mismatch")
    if namespace.get("BACKBONE_SUBMISSION_B64") != BACKBONE_B64_PLACEHOLDER:
        raise SystemExit("generated controlled-probe backbone placeholder mismatch")
    if "competition_submit" in text or "kaggle competitions submit" in text:
        raise SystemExit("generated controlled-probe runtime contains submission path")
    run(sys.executable, str(output), "--self-test")
    print(
        "CMI_FLU_PUBLIC_PROBES_PREPARE PASS "
        f"science_commit={SCIENCE_COMMIT} science_blob={SCIENCE_BLOB} "
        f"backbone_kernel={BACKBONE_KERNEL} expected_backbone_version={BACKBONE_EXPECTED_VERSION}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
