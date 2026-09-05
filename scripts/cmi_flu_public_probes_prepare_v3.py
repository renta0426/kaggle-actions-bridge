#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import py_compile
import subprocess
import sys
from types import ModuleType
from typing import Any

REQUEST_ID = "20260906-cmi-flu-public-probes-003"
PARENT_REQUEST_ID = "20260906-cmi-flu-public-probes-002"
FAILED_WORKFLOW_RUN_ID = 33980093685
TARGET_KERNEL = "renta0426/cmi-flu-public-probes-20260906-003"
TARGET_STAGE = "controlled_public_probes_regenerated_frozen_b21"
OUTPUT_DIR = "/kaggle/working/cmi-flu-public-probes-003"
SCIENCE_COMMIT = "7a85abcbf8282bc8bfe047b7db02a72f35222caa"
SCIENCE_BLOB = "497a8875d440e1b8a03d4ef938bc3a834213b33d"
B21_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
B21_PACKAGE_SHA256 = "48ff4ed2eadec8b059b8d37fb677af1f249b6486bccfbd55dca2274cfc6f3dc3"
PROBE_NAMES = ("task13_only", "task12_only", "task12_task13")
OUTPUT_FILENAMES = {
    "task13_only": "probe-task13.csv",
    "task12_only": "probe-task12.csv",
    "task12_task13": "probe-task12-task13.csv",
}
EXPECTED_HAI_MODELS = {
    "Task2.1": "et_subtype_d3_l5",
    "Task2.2": "et_subtype_d3_l5",
    "Task2.3": "pls_exact_5",
}


def load_base(root: pathlib.Path) -> ModuleType:
    path = root / "scripts/cmi_flu_public_probes_prepare.py"
    spec = importlib.util.spec_from_file_location("cmi_flu_public_probes_prepare_v1", path)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load controlled-probe prepare v1")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def validate_request(root: pathlib.Path) -> dict[str, Any]:
    request = json.loads(
        (root / "requests/cmi-flu-public-probes-003.json").read_text(encoding="utf-8")
    )
    expected = {
        "schema_version": 3,
        "request_id": REQUEST_ID,
        "parent_request_id": PARENT_REQUEST_ID,
        "failed_workflow_run_id": FAILED_WORKFLOW_RUN_ID,
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_source_commit": SCIENCE_COMMIT,
        "public_probes_blob_sha": SCIENCE_BLOB,
        "backbone_mode": "regenerate_frozen_b21_in_same_notebook",
        "b21_base_request_id": "20260903-cmi-flu-b21-001",
        "b21_source_commit": B21_SOURCE_COMMIT,
        "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB,
        "b21_base_package_zip_sha256": B21_PACKAGE_SHA256,
        "b21_verify_competition_md5": True,
        "expected_kernel_version": 1,
        "probe_family": list(PROBE_NAMES),
        "competition_submission_attempted": False,
        "public_scores_used_to_define_family": False,
        "automatic_compute_retries": 0,
        "enable_internet": False,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"Public-probe request 003 mismatch: {key}")
    expected_paths = [
        "cmi-flu-public-probes-003/probe-task13.csv",
        "cmi-flu-public-probes-003/probe-task12.csv",
        "cmi-flu-public-probes-003/probe-task12-task13.csv",
        "cmi-flu-public-probes-003/bridge-result.json",
        "cmi-flu-public-probes-003/summary.md",
    ]
    if request.get("allowed_output_paths") != expected_paths:
        raise SystemExit("Public-probe request 003 output allowlist mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 150,
        "hard_timeout_minutes": 300,
        "max_active_runs": 1,
    }:
        raise SystemExit("Public-probe request 003 resource contract mismatch")
    if request.get("api_budget") != {
        "max_calls": 36,
        "initial_poll_schedule_seconds": [30, 60, 120, 300],
        "steady_poll_interval_seconds": 600,
        "max_pages": 2,
    }:
        raise SystemExit("Public-probe request 003 API budget mismatch")
    repair = request.get("repair", {})
    if repair.get("scientific_contract_changed") is not False:
        raise SystemExit("request 003 must not change the scientific contract")
    if repair.get("generator_science_blob_changed") is not False:
        raise SystemExit("request 003 must retain the exact Public-probe science blob")
    if repair.get("probe_family_changed") is not False:
        raise SystemExit("request 003 must retain the exact three-probe family")
    return request


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
        stage = "materialize_frozen_b21_package"
        bundle = package_bytes()
        if sha256_bytes(bundle) != PACKAGE_ZIP_SHA256:
            raise BundleContractError("frozen B2.1 package SHA-256 mismatch")
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
            raise BundleContractError("frozen B2.1 base config anchor mismatch")
        if config_text.count("filename: b02_taskwise_compact.csv") == 1:
            config_text = config_text.replace(
                "filename: b02_taskwise_compact.csv", "filename: b021_taskwise_robust.csv", 1
            )
        if "\nselection:\n" in config_text:
            raise BundleContractError("frozen base config unexpectedly already has selection section")
        config_text = config_text.rstrip() + "\nselection:\n  policy: robust_v1\n"
        config_dir = runtime_root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "baseline_b021_robust.yaml"
        config_path.write_text(config_text, encoding="utf-8")

        stage = "dependency_preflight"
        import joblib
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        import yaml
        _ = (
            joblib.__version__, np.__version__, pd.__version__, scipy.__version__,
            sklearn.__version__, yaml.__version__
        )

        stage = "install_frozen_b21_adapter"
        adapter_namespace: dict[str, Any] = {}
        exec(
            compile(B21_ADAPTER_SOURCE, "<b21_runtime_adapter>", "exec"),
            adapter_namespace,
            adapter_namespace,
        )
        install_adapter = adapter_namespace.get("install")
        if not callable(install_adapter):
            raise BundleContractError("frozen B2.1 runtime adapter lacks install()")
        install_adapter()

        stage = "regenerate_frozen_b21"
        from cmi_flu.runner import run_from_config
        b21_result = run_from_config(
            config_path,
            repository_root=runtime_root,
            verify_md5=True,
        )
        if b21_result.validation_report.rows != 40:
            raise BundleContractError("regenerated B2.1 submission row contract failed")
        b21_submission_path = Path(b21_result.submission_path)
        b21_metrics_path = Path(b21_result.metrics_path)
        backbone = pd.read_csv(b21_submission_path)
        b21_metrics = json.loads(b21_metrics_path.read_text(encoding="utf-8"))
        selected_models = b21_metrics.get("selected_models")
        expected_tasks = {
            "Task1.1", "Task1.2", "Task1.3", "Task1.4",
            "Task2.1", "Task2.2", "Task2.3",
        }
        if not isinstance(selected_models, dict) or set(selected_models) != expected_tasks:
            raise BundleContractError("regenerated B2.1 selected-model task set mismatch")
        expected_hai = {
            "Task2.1": "et_subtype_d3_l5",
            "Task2.2": "et_subtype_d3_l5",
            "Task2.3": "pls_exact_5",
        }
        for task, expected_model in expected_hai.items():
            if selected_models.get(task) != expected_model:
                raise BundleContractError(
                    f"regenerated B2.1 model mismatch for {task}: {selected_models.get(task)}"
                )
        if selected_models.get("Task1.4") != "raw_pre_vacc_conserved_anchor":
            raise BundleContractError("regenerated B2.1 Task1.4 anchor mismatch")
        checksum = b21_metrics.get("checksum")
        if not isinstance(checksum, dict):
            raise BundleContractError("regenerated B2.1 checksum report missing")
        verified = checksum.get("verified")
        skipped = checksum.get("skipped")
        if not isinstance(verified, list) or len(verified) != 28:
            raise BundleContractError("regenerated B2.1 checksum verified-file count mismatch")
        if set(skipped or []) != {"md5sum"}:
            raise BundleContractError("regenerated B2.1 checksum skipped-file contract mismatch")
        b21_submission_sha = sha256_file(b21_submission_path)
        b21_metrics_sha = sha256_file(b21_metrics_path)

        stage = "load_probe_inputs"
        from cmi_flu.configuration import load_baseline_config
        from cmi_flu.runner import load_inputs
        probe_config = load_baseline_config(config_path, repository_root=runtime_root)
        raw_compat = dict(probe_config.raw)
        raw_compat["baseline"] = "b021_taskwise_robust"
        object.__setattr__(probe_config, "raw", raw_compat)
        object.__setattr__(probe_config, "baseline", "b021_taskwise_robust")
        inputs = load_inputs(probe_config)
        sample = inputs.tables["sample_submission"]
        from cmi_flu.contracts import TASK_COLUMNS, validate_submission
        validate_submission(backbone, sample, require_nonconstant_public_tasks=True)

        stage = "load_fixed_public_probe_science"
        import types
        probe_module = types.ModuleType("cmi_flu.public_probes")
        probe_module.__file__ = "<cmi_flu.public_probes>"
        probe_module.__package__ = "cmi_flu"
        sys.modules["cmi_flu.public_probes"] = probe_module
        exec(
            compile(PUBLIC_PROBES_SOURCE, "cmi_flu/public_probes.py", "exec"),
            probe_module.__dict__,
            probe_module.__dict__,
        )
        build_probes = getattr(probe_module, "build_controlled_public_probes", None)
        if not callable(build_probes):
            raise BundleContractError("controlled-probe science entry point missing")
        if tuple(getattr(probe_module, "PROBE_NAMES", ())) != (
            "task13_only", "task12_only", "task12_task13"
        ):
            raise BundleContractError("controlled-probe family identity mismatch")
        if getattr(probe_module, "TASK12_MODEL_NAME", None) != "et_d5_l5_sqrt":
            raise BundleContractError("Task1.2 probe model mismatch")
        if float(getattr(probe_module, "TASK12_LAMBDA", -1)) != 0.5:
            raise BundleContractError("Task1.2 probe lambda mismatch")
        if getattr(probe_module, "TASK13_MODEL_NAME", None) != "enet_a0.1_l0.5":
            raise BundleContractError("Task1.3 probe model mismatch")

        stage = "build_controlled_probes"
        variants, diagnostics = build_probes(probe_config, inputs, backbone)
        if tuple(variants) != ("task13_only", "task12_only", "task12_task13"):
            raise BundleContractError("controlled-probe output family mismatch")
        diagnostics = probe_json_safe(dict(diagnostics))
        if diagnostics.get("public_scores_used_to_define_family") is not False:
            raise BundleContractError("controlled-probe family unexpectedly used Public scores")

        stage = "validate_backbone_invariance"
        changed = {
            "task13_only": {"Task1.3"},
            "task12_only": {"Task1.2"},
            "task12_task13": {"Task1.2", "Task1.3"},
        }
        probe_hashes: dict[str, str] = {}
        for name, filename in PROBE_OUTPUT_FILENAMES.items():
            frame = variants[name]
            validate_submission(frame, sample, require_nonconstant_public_tasks=True)
            for task in TASK_COLUMNS:
                left = pd.to_numeric(frame[task], errors="raise").to_numpy(dtype=float)
                right = pd.to_numeric(backbone[task], errors="raise").to_numpy(dtype=float)
                equal = np.array_equal(left, right, equal_nan=True)
                if task in changed[name]:
                    if equal:
                        raise BundleContractError(
                            f"probe {name} failed to change intended task {task}"
                        )
                elif not equal:
                    raise BundleContractError(
                        f"probe {name} changed forbidden B2.1 backbone task {task}"
                    )
            path = output_dir / filename
            frame.to_csv(path, index=False)
            probe_hashes[name] = sha256_file(path)

        stage = "write_summary"
        summary = output_dir / "summary.md"
        summary.write_text(
            "# Controlled Public probe family 003\n\n"
            f"- request_id: `{REQUEST_ID}`\n"
            "- backbone_mode: `regenerate_frozen_b21_in_same_notebook`\n"
            f"- regenerated_b21_submission_sha256: `{b21_submission_sha}`\n"
            f"- regenerated_b21_metrics_sha256: `{b21_metrics_sha}`\n"
            "- family fixed before Public scores: `true`\n"
            "- Competition submission attempted by generator: `false`\n\n"
            "Generated exactly three probes: Task1.3-only, Task1.2-only, and Task1.2+Task1.3.\n",
            encoding="utf-8",
        )

        stage = "write_manifest"
        payload = {
            "schema_version": 3,
            "request_id": REQUEST_ID,
            "competition": COMPETITION,
            "kernel_stage": TARGET_STAGE,
            "source_commit": SOURCE_COMMIT,
            "base_package_zip_sha256": PACKAGE_ZIP_SHA256,
            "public_probes_source_blob_sha": PUBLIC_PROBES_SOURCE_BLOB_SHA,
            "public_probes_source_sha256": PUBLIC_PROBES_SOURCE_SHA256,
            "b21_source_commit": B21_SOURCE_COMMIT_FOR_REGEN,
            "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB_SHA,
            "b21_runtime_adapter_sha256": B21_ADAPTER_SHA256,
            "backbone_mode": "regenerate_frozen_b21_in_same_notebook",
            "regenerated_b21_submission_sha256": b21_submission_sha,
            "regenerated_b21_metrics_sha256": b21_metrics_sha,
            "regenerated_b21_selected_models": selected_models,
            "regenerated_b21_checksum_verified_files": len(verified),
            "regenerated_b21_checksum_skipped": sorted(skipped or []),
            "probe_sha256": probe_hashes,
            "probe_changed_columns": diagnostics.get("changed_columns"),
            "diagnostics": diagnostics,
            "competition_submission_attempted": False,
            "public_scores_used_to_define_family": False,
        }
        (output_dir / "bridge-result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(runtime_root, ignore_errors=True)
        print(
            "CMI_FLU_PUBLIC_PROBES_003_COMPLETE "
            f"request_id={REQUEST_ID} b21_submission_sha256={b21_submission_sha} "
            + " ".join(
                f"{name}_sha256={probe_hashes[name]}"
                for name in ("task13_only", "task12_only", "task12_task13")
            )
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
    base = load_base(root)
    template = output.parent / "public-probes-v1-template.py"
    run(
        sys.executable,
        str(root / "scripts/cmi_flu_public_probes_prepare.py"),
        "--repository-root",
        str(root),
        "--output",
        str(template),
    )
    text = template.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'REQUEST_ID = "20260906-cmi-flu-public-probes-001"',
        f'REQUEST_ID = "{REQUEST_ID}"',
        label="request id",
    )
    text = replace_once(
        text,
        'TARGET_STAGE = "controlled_public_probes_task12_task13"',
        f'TARGET_STAGE = "{TARGET_STAGE}"',
        label="kernel stage",
    )
    text = replace_once(
        text,
        'default=Path("/kaggle/working/cmi-flu-public-probes-001")',
        f'default=Path("{OUTPUT_DIR}")',
        label="output directory",
    )
    text = replace_once(
        text,
        'BACKBONE_KERNEL = "renta0426/cmi-flu-b21-robust-cv-20260903-001"',
        'BACKBONE_KERNEL = "regenerated-frozen-b21"',
        label="legacy backbone kernel provenance",
    )
    text = replace_once(
        text,
        'BACKBONE_EXPECTED_VERSION = 1',
        'BACKBONE_EXPECTED_VERSION = 0',
        label="legacy backbone version provenance",
    )
    text = replace_once(
        text,
        'BACKBONE_SUBMISSION_B64 = "__CMI_FLU_BACKBONE_B64__"',
        'BACKBONE_SUBMISSION_B64 = "regenerated-in-process"',
        label="legacy backbone placeholder removal",
    )
    text = replace_once(
        text,
        'BACKBONE_SUBMISSION_SHA256 = "__CMI_FLU_BACKBONE_SHA256__"',
        'BACKBONE_SUBMISSION_SHA256 = "computed-after-regeneration"',
        label="legacy backbone SHA placeholder removal",
    )

    helper_marker = f'PUBLIC_PROBES_SOURCE_BLOB_SHA = "{SCIENCE_BLOB}"\n'
    helper_insertion = (
        helper_marker
        + f'B21_SOURCE_COMMIT_FOR_REGEN = "{B21_SOURCE_COMMIT}"\n'
    )
    text = replace_once(
        text,
        helper_marker,
        helper_insertion,
        label="B2.1 regeneration provenance insertion",
    )

    execute_marker = "def execute(input_dir: Path, output_dir: Path) -> Mapping[str, Any]:\n"
    execute_start = text.index(execute_marker)
    main_marker = "\ndef main() -> int:\n"
    main_start = text.index(main_marker, execute_start)
    text = text[:execute_start] + execute_source() + text[main_start:]

    if "__CMI_FLU_BACKBONE_" in text:
        raise SystemExit("request 003 runtime retains a legacy backbone placeholder")
    forbidden = (
        "kernels_status(",
        "kernels_list(",
        "kaggle kernels output",
        "competition_submit",
        "kaggle competitions submit",
    )
    present = [token for token in forbidden if token in text]
    if present:
        raise SystemExit(f"request 003 runtime contains forbidden remote-read/submit tokens: {present}")
    for token in (
        'B21_SOURCE_COMMIT_FOR_REGEN = "33030746bc7bad02ad2c1e670ac319cc943c524d"',
        '"backbone_mode": "regenerate_frozen_b21_in_same_notebook"',
        'verify_md5=True',
        '"Task2.1": "et_subtype_d3_l5"',
        '"Task2.2": "et_subtype_d3_l5"',
        '"Task2.3": "pls_exact_5"',
        '"Task1.4") != "raw_pre_vacc_conserved_anchor"',
        'len(verified) != 28',
        'set(skipped or []) != {"md5sum"}',
    ):
        if token not in text:
            raise SystemExit(f"request 003 regenerated-B2.1 contract missing: {token}")

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)
    namespace: dict[str, Any] = {"__name__": "public_probes_v3_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("request 003 generated runtime identity mismatch")
    if namespace.get("TARGET_STAGE") != TARGET_STAGE:
        raise SystemExit("request 003 generated runtime stage mismatch")
    if namespace.get("PUBLIC_PROBES_SOURCE_BLOB_SHA") != SCIENCE_BLOB:
        raise SystemExit("request 003 generated runtime science blob mismatch")
    if namespace.get("PACKAGE_ZIP_SHA256") != B21_PACKAGE_SHA256:
        raise SystemExit("request 003 frozen B2.1 package hash mismatch")
    if namespace.get("B21_ADAPTER_BLOB_SHA") != B21_ADAPTER_BLOB:
        raise SystemExit("request 003 frozen B2.1 adapter blob mismatch")
    if namespace.get("B21_SOURCE_COMMIT_FOR_REGEN") != B21_SOURCE_COMMIT:
        raise SystemExit("request 003 frozen B2.1 source provenance mismatch")
    run(sys.executable, str(output), "--self-test")
    print(
        "CMI_FLU_PUBLIC_PROBES_PREPARE_V3 PASS "
        f"request_id={REQUEST_ID} target={TARGET_KERNEL} "
        f"science_blob={SCIENCE_BLOB} b21_package={B21_PACKAGE_SHA256} "
        f"b21_adapter_blob={B21_ADAPTER_BLOB} backbone_mode=regenerated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
