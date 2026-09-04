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

REQUEST_ID = "20260905-cmi-flu-study-similarity-001"
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_COMMIT = "f47ee36ce8933895b522de0ec402c75c7fe517a7"
SCIENCE_PATH = "src/cmi_flu/study_similarity.py"
SCIENCE_BLOB = "27351df3d9187899c4bce2ff1a24b06efc160185"
SCIENCE_TRANSPORT = "agent_relay_exact_blob"
B21_BASE_REQUEST_ID = "20260903-cmi-flu-b21-001"
B21_BASE_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
BASE_SHA256 = "7894a9232e7372950f70a36d35de41bcc8bda90ec4d0320187e82c3b8c76f2db"
BASE_SIZE = 93363
TARGET_COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-study-similarity-20260905-001"
TARGET_STAGE = "phase_a_study_similarity_weighting"
EXPECTED_KERNEL_VERSION = 1
TASKS = ("Task1.1", "Task1.2")
MODEL_NAMES = {"Task1.1": "pls_2", "Task1.2": "enet_a0.001_l0.5"}
EXPECTED_HELD_STUDIES = {"Task1.1": 4, "Task1.2": 3}
ALLOWED_OUTPUT_PATHS = [
    "cmi-flu-study-similarity-001/bridge-result.json",
    "cmi-flu-study-similarity-001/metrics.json",
    "cmi-flu-study-similarity-001/summary.md",
]


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
    path = root / "payloads/cmi-flu-study-similarity-001/study_similarity.py"
    data = path.read_bytes()
    found_blob = git_blob_sha(data)
    if found_blob != SCIENCE_BLOB:
        raise SystemExit(f"study-similarity science blob mismatch: {found_blob}")
    source = data.decode("utf-8")
    compile(source, SCIENCE_PATH, "exec")
    required = (
        '"Task1.1"',
        '"Task1.2"',
        '"pls_2"',
        '"enet_a0.001_l0.5"',
        "purged_leave_one_study_out",
        "fit_final_model",
        "SIMILARITY_SHRINK_TO_UNIFORM = 0.5",
        "WORST_STUDY_TOLERANCE = 0.10",
        "run_study_similarity_experiment",
        '"competition_submission_attempted": False',
        '"leaderboard_used_for_selection": False',
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise SystemExit(f"study-similarity science contract tokens missing: {missing}")
    forbidden = ("sample_weight", "kaggle competitions submit", "competition_submit")
    present = [token for token in forbidden if token in source]
    if present:
        raise SystemExit(f"study-similarity science contains forbidden tokens: {present}")
    return source, sha256_bytes(data)


def validate_request(root: pathlib.Path) -> dict[str, Any]:
    request = json.loads(
        (root / "requests/cmi-flu-study-similarity-001.json").read_text(encoding="utf-8")
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
        "study_similarity_blob_sha": SCIENCE_BLOB,
        "b21_base_request_id": B21_BASE_REQUEST_ID,
        "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB,
        "expected_kernel_version": EXPECTED_KERNEL_VERSION,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
        "enable_internet": False,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"study-similarity request mismatch: {key}")
    if request.get("allowed_output_paths") != ALLOWED_OUTPUT_PATHS:
        raise SystemExit("study-similarity output allowlist mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 60,
        "hard_timeout_minutes": 120,
        "max_active_runs": 1,
    }:
        raise SystemExit("study-similarity resource contract mismatch")
    if request.get("api_budget") != {
        "max_calls": 18,
        "initial_poll_schedule_seconds": [30, 60, 120, 300],
        "steady_poll_interval_seconds": 600,
        "max_pages": 2,
    }:
        raise SystemExit("study-similarity API budget mismatch")
    return request


def injected_helpers(science_source: str, science_sha256: str) -> str:
    return f'''\nTARGET_STAGE = "{TARGET_STAGE}"\nSTUDY_SIMILARITY_SOURCE_BLOB_SHA = "{SCIENCE_BLOB}"\nSTUDY_SIMILARITY_SOURCE_SHA256 = "{science_sha256}"\nSTUDY_SIMILARITY_SOURCE = {science_source!r}\n\ndef similarity_json_safe(value: Any) -> Any:\n    if value is None or isinstance(value, (str, bool, int)):\n        return value\n    if isinstance(value, float):\n        return value if math.isfinite(value) else None\n    if isinstance(value, Mapping):\n        return {{str(key): similarity_json_safe(item) for key, item in value.items()}}\n    if isinstance(value, (list, tuple)):\n        return [similarity_json_safe(item) for item in value]\n    item = getattr(value, "item", None)\n    if callable(item):\n        try:\n            return similarity_json_safe(item())\n        except (ValueError, TypeError):\n            pass\n    return str(value)\n'''


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
            runtime_root
            / "external"
            / "google-drive"
            / "challenge-resources"
            / "reference_files"
        )
        derive_reference_files(input_dir, reference_dir)

        config_text = CONFIG_TEXT
        if config_text.count("baseline: b02_taskwise_compact") != 1:
            raise BundleContractError("B2 base config baseline anchor mismatch")
        if config_text.count("filename: b02_taskwise_compact.csv") == 1:
            config_text = config_text.replace(
                "filename: b02_taskwise_compact.csv",
                "filename: b021_taskwise_robust.csv",
                1,
            )
        if "\nselection:\n" in config_text:
            raise BundleContractError("B2 base config unexpectedly already has selection section")
        config_text = config_text.rstrip() + "\nselection:\n  policy: robust_v1\n"
        config_dir = runtime_root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "baseline_b021_robust.yaml"
        config_path.write_text(config_text, encoding="utf-8")

        stage = "dependency_preflight"
        import inspect
        import joblib
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        import yaml
        _ = (
            joblib.__version__, np.__version__, pd.__version__, scipy.__version__,
            sklearn.__version__, yaml.__version__,
        )

        stage = "install_b21_adapter"
        adapter_namespace: dict[str, Any] = {}
        exec(
            compile(B21_ADAPTER_SOURCE, "<b21_runtime_adapter>", "exec"),
            adapter_namespace,
            adapter_namespace,
        )
        install_adapter = adapter_namespace.get("install")
        if not callable(install_adapter):
            raise BundleContractError("B2.1 runtime adapter lacks install()")
        install_adapter()

        stage = "frozen_api_preflight"
        from cmi_flu.cv import purged_leave_one_study_out
        from cmi_flu.datasets import TaskDataset, build_task_11_dataset, build_task_12_dataset
        from cmi_flu.metrics import percentile_rank, root_mean_squared_error, safe_spearman
        from cmi_flu.models import fit_final_model
        fit_signature = inspect.signature(fit_final_model)
        required_parameters = {"train_frame", "prediction_frame", "target_column", "spec", "excluded_columns"}
        if not required_parameters.issubset(set(fit_signature.parameters)):
            raise BundleContractError(f"frozen fit_final_model signature mismatch: {fit_signature}")
        if not hasattr(TaskDataset, "feature_columns"):
            raise BundleContractError("frozen TaskDataset feature API missing")
        _ = (
            purged_leave_one_study_out,
            build_task_11_dataset,
            build_task_12_dataset,
            percentile_rank,
            root_mean_squared_error,
            safe_spearman,
        )

        stage = "load_study_similarity"
        import types
        study_module = types.ModuleType("cmi_flu.study_similarity")
        study_module.__file__ = "<cmi_flu.study_similarity>"
        study_module.__package__ = "cmi_flu"
        sys.modules["cmi_flu.study_similarity"] = study_module
        exec(
            compile(STUDY_SIMILARITY_SOURCE, "cmi_flu/study_similarity.py", "exec"),
            study_module.__dict__,
            study_module.__dict__,
        )
        run_experiment = getattr(study_module, "run_study_similarity_experiment", None)
        if not callable(run_experiment):
            raise BundleContractError("study-similarity science entry point missing")
        contracts = getattr(study_module, "TASK_CONTRACTS", {})
        if set(contracts) != {"Task1.1", "Task1.2"}:
            raise BundleContractError("study-similarity task contract mismatch")
        if contracts["Task1.1"].get("model_name") != "pls_2":
            raise BundleContractError("Task1.1 locked model mismatch")
        if contracts["Task1.2"].get("model_name") != "enet_a0.001_l0.5":
            raise BundleContractError("Task1.2 locked model mismatch")
        if float(getattr(study_module, "SIMILARITY_SHRINK_TO_UNIFORM", -1)) != 0.5:
            raise BundleContractError("study-similarity shrinkage contract mismatch")

        stage = "load_inputs"
        from cmi_flu.configuration import load_baseline_config
        from cmi_flu.runner import load_inputs
        config = load_baseline_config(config_path, repository_root=runtime_root)
        if config.baseline != "b02_taskwise_compact":
            raise BundleContractError("legacy embedded loader returned unexpected baseline")
        if str(config.section("selection").get("policy", "")) != "robust_v1":
            raise BundleContractError("runtime robust_v1 selection contract missing")
        raw_compat = dict(config.raw)
        raw_compat["baseline"] = "b021_taskwise_robust"
        object.__setattr__(config, "raw", raw_compat)
        object.__setattr__(config, "baseline", "b021_taskwise_robust")
        inputs = load_inputs(config)

        stage = "run_study_similarity"
        metrics = dict(run_experiment(config, inputs))
        metrics.update(
            {
                "run_id": REQUEST_ID,
                "output_policy": "aggregate_only_public_study_names_no_participant_or_row_predictions",
                "checksum": (
                    {
                        "verified_count": len(inputs.checksum_report.verified),
                        "skipped": list(inputs.checksum_report.skipped),
                    }
                    if inputs.checksum_report is not None
                    else None
                ),
            }
        )
        metrics = similarity_json_safe(metrics)

        stage = "validate_study_similarity"
        if metrics.get("experiment") != TARGET_STAGE:
            raise BundleContractError("study-similarity experiment identity mismatch")
        if metrics.get("leaderboard_used_for_selection") is not False:
            raise BundleContractError("study-similarity unexpectedly used leaderboard")
        if metrics.get("competition_submission_attempted") is not False:
            raise BundleContractError("study-similarity unexpectedly attempted Competition submission")
        design = metrics.get("design") or {}
        if design.get("held_target_outcomes_used_for_weights") is not False:
            raise BundleContractError("held outcomes entered study weighting")
        if design.get("model_selection_retuned") is not False:
            raise BundleContractError("study-similarity retuned model selection")
        if float(design.get("worst_study_tolerance", -1)) != 0.10:
            raise BundleContractError("promotion tolerance contract mismatch")

        tasks = metrics.get("tasks")
        if not isinstance(tasks, dict) or set(tasks) != {"Task1.1", "Task1.2"}:
            raise BundleContractError("study-similarity task set mismatch")
        expected_models = {"Task1.1": "pls_2", "Task1.2": "enet_a0.001_l0.5"}
        expected_folds = {"Task1.1": 4, "Task1.2": 3}
        for task, item in tasks.items():
            if item.get("selected_model") != expected_models[task]:
                raise BundleContractError(f"selected model mismatch: {task}")
            if int(item.get("held_study_count", -1)) != expected_folds[task]:
                raise BundleContractError(f"held-study count mismatch: {task}")
            folds = item.get("folds")
            if not isinstance(folds, list) or len(folds) != expected_folds[task]:
                raise BundleContractError(f"held-study fold payload mismatch: {task}")
            summary = item.get("summary") or {}
            if set(summary) != {"reference", "uniform", "similarity_weighted"}:
                raise BundleContractError(f"summary condition mismatch: {task}")
            for condition in ("reference", "uniform", "similarity_weighted"):
                if int((summary[condition] or {}).get("count", -1)) != expected_folds[task]:
                    raise BundleContractError(f"summary count mismatch: {task}/{condition}")
            promotion = item.get("promotion") or {}
            if not isinstance(promotion.get("passed"), bool):
                raise BundleContractError(f"promotion flag missing: {task}")
            for fold in folds:
                if not str(fold.get("held_study", "")):
                    raise BundleContractError(f"held study name missing: {task}")
                source_diag = fold.get("source_diagnostics") or {}
                if int(source_diag.get("source_count", 0)) < 2:
                    raise BundleContractError(f"too few source experts: {task}")
                sources = source_diag.get("sources") or {}
                weights = [float((entry or {}).get("weight")) for entry in sources.values()]
                if len(weights) < 2 or not all(math.isfinite(weight) and weight > 0 for weight in weights):
                    raise BundleContractError(f"invalid source weights: {task}")
                if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
                    raise BundleContractError(f"source weights do not sum to one: {task}")

        challenge = metrics.get("challenge")
        if not isinstance(challenge, dict) or set(challenge) != {"Task1.1", "Task1.2"}:
            raise BundleContractError("challenge task set mismatch")
        for task, item in challenge.items():
            if int(item.get("rows", -1)) != 40:
                raise BundleContractError(f"challenge row count mismatch: {task}")
            for key in (
                "uniform_vs_reference",
                "similarity_weighted_vs_reference",
                "similarity_weighted_vs_uniform",
            ):
                agreement = item.get(key) or {}
                if int(agreement.get("n", -1)) != 40:
                    raise BundleContractError(f"challenge agreement n mismatch: {task}/{key}")
                value = (agreement.get("rank_spearman") or {}).get("value")
                if value is None or not math.isfinite(float(value)):
                    raise BundleContractError(f"challenge agreement non-finite: {task}/{key}")
            source_diag = item.get("source_diagnostics") or {}
            weights = [
                float((entry or {}).get("weight"))
                for entry in (source_diag.get("sources") or {}).values()
            ]
            if len(weights) < 2 or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise BundleContractError(f"challenge source weights invalid: {task}")

        serialized = json.dumps(metrics, sort_keys=True, ensure_ascii=False)
        banned = (
            '"participant_id"',
            '"subject_group"',
            '"row_index"',
            '"oof_predictions"',
            '"challenge_predictions"',
        )
        hits = [token for token in banned if token in serialized]
        if hits:
            raise BundleContractError(f"study-similarity aggregate output leaked row-level fields: {hits}")

        final_metrics = output_dir / "metrics.json"
        final_summary = output_dir / "summary.md"
        final_metrics.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        lines = [
            "# Phase A pseudo-challenge study-similarity weighting",
            "",
            f"- request_id: `{REQUEST_ID}`",
            f"- science_source_commit: `{SOURCE_COMMIT}`",
            "- held-target outcomes used for weights: `false`",
            "- Competition submission: none",
            "",
        ]
        for task in ("Task1.1", "Task1.2"):
            item = tasks[task]
            lines.extend([f"## {task}", ""])
            for condition in ("reference", "uniform", "similarity_weighted"):
                summary = item["summary"][condition]
                lines.append(
                    f"- {condition}: mean=`{summary.get('mean')}`, median=`{summary.get('median')}`, "
                    f"min=`{summary.get('min')}`, max=`{summary.get('max')}`"
                )
            promotion = item["promotion"]
            lines.append(
                f"- promotion: `{promotion.get('passed')}`, wins=`{promotion.get('wins')}`/`{promotion.get('required_wins')}`"
            )
            lines.append("")
        lines.append("Output is aggregate-only; public study names/weights are retained, but no participant IDs or row-level predictions are written.")
        final_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

        stage = "finalize_manifest"
        payload = {
            "schema_version": 1,
            "request_id": REQUEST_ID,
            "competition": COMPETITION,
            "kernel_stage": TARGET_STAGE,
            "source_commit": SOURCE_COMMIT,
            "base_package_zip_sha256": PACKAGE_ZIP_SHA256,
            "study_similarity_source_blob_sha": STUDY_SIMILARITY_SOURCE_BLOB_SHA,
            "study_similarity_source_sha256": STUDY_SIMILARITY_SOURCE_SHA256,
            "b21_runtime_adapter_sha256": B21_ADAPTER_SHA256,
            "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB_SHA,
            "python_version": platform.python_version(),
            "metrics_sha256": sha256_file(final_metrics),
            "promotion": metrics["promotion"],
            "task_summaries": {
                task: {
                    "selected_model": item["selected_model"],
                    "held_study_count": item["held_study_count"],
                    "summary": item["summary"],
                    "promotion": item["promotion"],
                }
                for task, item in tasks.items()
            },
            "challenge": challenge,
            "checksum": metrics.get("checksum"),
            "competition_submission_attempted": False,
            "leaderboard_used_for_selection": False,
        }
        (output_dir / "bridge-result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(runtime_root, ignore_errors=True)
        promoted = [task for task, evidence in metrics["promotion"].items() if evidence.get("passed")]
        print(
            "CMI_FLU_STUDY_SIMILARITY_COMPLETE "
            f"request_id={REQUEST_ID} promoted={','.join(promoted) if promoted else 'none'}"
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
    run(
        sys.executable,
        str(root / "scripts/cmi_flu_b2_patch_v4.py"),
        "--source",
        str(base_source),
        "--request",
        str(root / "requests/cmi-flu-b2-launch-v4.json"),
        "--output",
        str(base004),
    )
    run(
        sys.executable,
        str(root / "scripts/cmi_flu_b21_patch.py"),
        "--source",
        str(base004),
        "--adapter",
        str(adapter_path),
        "--request",
        str(root / "requests/cmi-flu-b21-launch-v1.json"),
        "--output",
        str(b21),
    )

    text = b21.read_text(encoding="utf-8")
    if "B21_ADAPTER_SOURCE" not in text:
        raise SystemExit("study-similarity source must be materialized B2.1 runtime")
    text = replace_once(
        text,
        f'REQUEST_ID = "{B21_BASE_REQUEST_ID}"',
        f'REQUEST_ID = "{REQUEST_ID}"',
        label="request id",
    )
    text = replace_once(
        text,
        f'SOURCE_COMMIT = "{B21_BASE_SOURCE_COMMIT}"',
        f'SOURCE_COMMIT = "{SCIENCE_COMMIT}"',
        label="science source commit",
    )
    text = replace_once(
        text,
        'default=Path("/kaggle/working/cmi-flu-b2")',
        'default=Path("/kaggle/working/cmi-flu-study-similarity-001")',
        label="output directory",
    )
    text = text.replace("CMI_FLU_B2_FAILED ", "CMI_FLU_STUDY_SIMILARITY_FAILED ")

    marker = "\ndef execute(input_dir: Path, output_dir: Path) -> Mapping[str, Any]:\n"
    text = replace_once(
        text,
        marker,
        injected_helpers(science_source, science_sha256) + marker,
        label="study-similarity source insertion",
    )
    execute_start = text.index(marker.lstrip("\n"))
    main_marker = "\ndef main() -> int:\n"
    main_start = text.index(main_marker, execute_start)
    text = text[:execute_start] + execute_source() + text[main_start:]

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)

    namespace: dict[str, Any] = {"__name__": "study_similarity_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("generated study-similarity runtime request mismatch")
    if namespace.get("SOURCE_COMMIT") != SCIENCE_COMMIT:
        raise SystemExit("generated study-similarity runtime science commit mismatch")
    if namespace.get("TARGET_STAGE") != TARGET_STAGE:
        raise SystemExit("generated study-similarity runtime stage mismatch")
    if namespace.get("STUDY_SIMILARITY_SOURCE_BLOB_SHA") != SCIENCE_BLOB:
        raise SystemExit("generated study-similarity runtime science blob mismatch")
    if "kaggle competitions submit" in text or "api.competition_submit" in text:
        raise SystemExit("generated study-similarity runtime contains forbidden submission path")
    if text.count("shutil.rmtree(runtime_root, ignore_errors=True)") != 2:
        raise SystemExit("runtime scratch cleanup must cover success and failure paths")

    run(sys.executable, str(output), "--self-test")
    print(
        "CMI_FLU_STUDY_SIMILARITY_PREPARE PASS "
        f"science_commit={SCIENCE_COMMIT} science_blob={SCIENCE_BLOB} "
        f"science_sha256={science_sha256} b21_adapter_blob={B21_ADAPTER_BLOB} "
        f"target_kernel={TARGET_KERNEL} expected_version=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
