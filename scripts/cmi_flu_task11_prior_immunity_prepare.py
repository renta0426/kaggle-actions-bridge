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

REQUEST_ID = "20260905-cmi-flu-task11-prior-immunity-001"
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_COMMIT = "06c85e6263b59cd6ac97b7087779e7a9fb1cbdae"
SCIENCE_PATH = "src/cmi_flu/task11_prior_immunity.py"
SCIENCE_URL = (
    "https://raw.githubusercontent.com/renta0426/CMI-Flu-Invited-Prediction-Challenge/"
    "06c85e6263b59cd6ac97b7087779e7a9fb1cbdae/src/cmi_flu/task11_prior_immunity.py"
)
SCIENCE_BLOB = "50d9a43604d2b75479b8f873a86a8daf9d5bd7a9"
SCIENCE_TRANSPORT = "pinned_public_raw_presecret"
B21_BASE_REQUEST_ID = "20260903-cmi-flu-b21-001"
B21_BASE_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
BASE_SHA256 = "7894a9232e7372950f70a36d35de41bcc8bda90ec4d0320187e82c3b8c76f2db"
BASE_SIZE = 93363
TARGET_COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-phase-b-task1-1-prior-immunity-20260905-001"
TARGET_STAGE = "phase_b_task11_prior_immunity_late_fusion"
EXPECTED_KERNEL_VERSION = 1
EXPECTED_STUDIES = ("SDY180", "SDY515", "SDY519", "SDY56")
BASE_CONDITIONS = ("b1", "b21", "anchor_residual")
FUSION_CONDITIONS = (
    "b1_plus_prior_w0.25",
    "b1_plus_prior_w0.5",
    "b21_plus_prior_w0.25",
    "b21_plus_prior_w0.5",
    "anchor_residual_plus_prior_w0.25",
    "anchor_residual_plus_prior_w0.5",
)
ALLOWED_OUTPUT_PATHS = [
    "cmi-flu-task11-prior-immunity-001/bridge-result.json",
    "cmi-flu-task11-prior-immunity-001/metrics.json",
    "cmi-flu-task11-prior-immunity-001/summary.md",
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


def load_science(path: pathlib.Path) -> tuple[str, str]:
    data = path.read_bytes()
    if git_blob_sha(data) != SCIENCE_BLOB:
        raise SystemExit("Task1.1 prior-immunity science blob mismatch")
    source = data.decode("utf-8")
    compile(source, SCIENCE_PATH, "exec")
    required = (
        'TASK = "Task1.1"',
        'ANCHOR_COLUMN = "cytokine_rank__CXCL10"',
        'B21_MODEL_NAME = "pls_2"',
        'ANCHOR_RESIDUAL_MODEL_NAME = "pls_1"',
        "ANCHOR_RESIDUAL_LAMBDA = 0.25",
        'name="prior_immunity_ridge_a10"',
        'params={"alpha": 10.0}',
        "FUSION_WEIGHTS = (0.25, 0.5)",
        "WORST_STUDY_TOLERANCE = 0.10",
        "build_hai_baseline_long",
        "build_hai_panel_summaries",
        "fit_final_model",
        "run_task11_prior_immunity_experiment",
        '"leaderboard_used_for_selection": False',
        '"competition_submission_attempted": False',
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise SystemExit(f"Task1.1 prior-immunity science contract tokens missing: {missing}")
    forbidden = (
        "kaggle competitions submit",
        "competition_submit",
        "sample_weight=",
        '"virus_in_vaccine"',
    )
    present = [token for token in forbidden if token in source]
    if present:
        raise SystemExit(f"Task1.1 prior-immunity science contains forbidden tokens: {present}")
    return source, sha256_bytes(data)


def validate_request(root: pathlib.Path) -> dict[str, Any]:
    request = json.loads(
        (root / "requests/cmi-flu-task11-prior-immunity-001.json").read_text(
            encoding="utf-8"
        )
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
        "science_source_raw_url": SCIENCE_URL,
        "science_transport": SCIENCE_TRANSPORT,
        "task11_prior_immunity_blob_sha": SCIENCE_BLOB,
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
            raise SystemExit(f"Task1.1 prior-immunity request mismatch: {key}")
    if request.get("allowed_output_paths") != ALLOWED_OUTPUT_PATHS:
        raise SystemExit("Task1.1 prior-immunity output allowlist mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 60,
        "hard_timeout_minutes": 120,
        "max_active_runs": 1,
    }:
        raise SystemExit("Task1.1 prior-immunity resource contract mismatch")
    if request.get("api_budget") != {
        "max_calls": 18,
        "initial_poll_schedule_seconds": [30, 60, 120, 300],
        "steady_poll_interval_seconds": 600,
        "max_pages": 2,
    }:
        raise SystemExit("Task1.1 prior-immunity API budget mismatch")
    return request


def injected_helpers(science_source: str, science_sha256: str) -> str:
    return f'''\nTARGET_STAGE = "{TARGET_STAGE}"\nTASK11_PRIOR_IMMUNITY_SOURCE_BLOB_SHA = "{SCIENCE_BLOB}"\nTASK11_PRIOR_IMMUNITY_SOURCE_SHA256 = "{science_sha256}"\nTASK11_PRIOR_IMMUNITY_SOURCE = {science_source!r}\n\ndef prior_json_safe(value: Any) -> Any:\n    if value is None or isinstance(value, (str, bool, int)):\n        return value\n    if isinstance(value, float):\n        return value if math.isfinite(value) else None\n    if isinstance(value, Mapping):\n        return {{str(key): prior_json_safe(item) for key, item in value.items()}}\n    if isinstance(value, (list, tuple)):\n        return [prior_json_safe(item) for item in value]\n    item = getattr(value, "item", None)\n    if callable(item):\n        try:\n            return prior_json_safe(item())\n        except (ValueError, TypeError):\n            pass\n    return str(value)\n'''


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
        from cmi_flu.evaluation import default_splits_for_task
        from cmi_flu.features.serology import build_hai_baseline_long, build_hai_panel_summaries
        from cmi_flu.metrics import percentile_rank, safe_spearman
        from cmi_flu.models import ModelSpec, fit_final_model
        from cmi_flu.runner import build_b02_datasets
        fit_signature = inspect.signature(fit_final_model)
        required_fit = {"train_frame", "prediction_frame", "target_column", "spec", "excluded_columns"}
        if not required_fit.issubset(set(fit_signature.parameters)):
            raise BundleContractError(f"frozen fit_final_model signature mismatch: {fit_signature}")
        panel_signature = inspect.signature(build_hai_panel_summaries)
        if "group_columns" not in panel_signature.parameters:
            raise BundleContractError(f"frozen HAI summary signature mismatch: {panel_signature}")
        _ = (
            default_splits_for_task,
            build_hai_baseline_long,
            percentile_rank,
            safe_spearman,
            ModelSpec,
            build_b02_datasets,
        )

        stage = "load_task11_prior_immunity"
        import types
        prior_module = types.ModuleType("cmi_flu.task11_prior_immunity")
        prior_module.__file__ = "<cmi_flu.task11_prior_immunity>"
        prior_module.__package__ = "cmi_flu"
        sys.modules["cmi_flu.task11_prior_immunity"] = prior_module
        exec(
            compile(
                TASK11_PRIOR_IMMUNITY_SOURCE,
                "cmi_flu/task11_prior_immunity.py",
                "exec",
            ),
            prior_module.__dict__,
            prior_module.__dict__,
        )
        run_experiment = getattr(prior_module, "run_task11_prior_immunity_experiment", None)
        if not callable(run_experiment):
            raise BundleContractError("Task1.1 prior-immunity science entry point missing")
        if getattr(prior_module, "TASK", None) != "Task1.1":
            raise BundleContractError("Task1.1 prior-immunity task contract mismatch")
        if getattr(prior_module, "B21_MODEL_NAME", None) != "pls_2":
            raise BundleContractError("Task1.1 frozen B2.1 model mismatch")
        if getattr(prior_module, "ANCHOR_RESIDUAL_MODEL_NAME", None) != "pls_1":
            raise BundleContractError("Task1.1 anchor-residual model mismatch")
        if float(getattr(prior_module, "ANCHOR_RESIDUAL_LAMBDA", -1)) != 0.25:
            raise BundleContractError("Task1.1 anchor-residual lambda mismatch")
        if tuple(getattr(prior_module, "FUSION_WEIGHTS", ())) != (0.25, 0.5):
            raise BundleContractError("Task1.1 prior-immunity fusion-weight mismatch")
        summary_columns = tuple(getattr(prior_module, "SEROLOGY_SUMMARY_COLUMNS", ()))
        if len(summary_columns) != 9 or "hai_panel_count" in summary_columns:
            raise BundleContractError("Task1.1 prior-immunity feature contract mismatch")
        prior_model = getattr(prior_module, "SERology_MODEL", None)
        if prior_model is None or prior_model.name != "prior_immunity_ridge_a10":
            raise BundleContractError("Task1.1 prior-immunity Ridge contract missing")
        if prior_model.family != "ridge" or float(prior_model.params.get("alpha", -1)) != 10.0:
            raise BundleContractError("Task1.1 prior-immunity Ridge parameters mismatch")

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

        stage = "run_task11_prior_immunity"
        metrics = dict(run_experiment(config, inputs))
        metrics.update(
            {
                "run_id": REQUEST_ID,
                "output_policy": "aggregate_only_public_study_names_no_row_predictions",
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
        metrics = prior_json_safe(metrics)

        stage = "validate_task11_prior_immunity"
        if metrics.get("experiment") != TARGET_STAGE or metrics.get("task") != "Task1.1":
            raise BundleContractError("Task1.1 prior-immunity experiment identity mismatch")
        if metrics.get("leaderboard_used_for_selection") is not False:
            raise BundleContractError("Task1.1 prior-immunity unexpectedly used leaderboard")
        if metrics.get("competition_submission_attempted") is not False:
            raise BundleContractError("Task1.1 prior-immunity unexpectedly attempted submission")
        if metrics.get("held_target_outcomes_used_for_serology_features") is not False:
            raise BundleContractError("held target entered prior-immunity feature construction")
        if metrics.get("missing_serology_policy") != "fallback_to_base_without_complete_case_filtering":
            raise BundleContractError("Task1.1 missing-serology fallback contract mismatch")
        if metrics.get("fusion_weights") != [0.25, 0.5]:
            raise BundleContractError("Task1.1 fusion-weight output mismatch")
        base_contract = metrics.get("base_contract") or {}
        if base_contract != {
            "b21_model": "pls_2",
            "anchor_residual_model": "pls_1",
            "anchor_residual_lambda": 0.25,
        }:
            raise BundleContractError("Task1.1 base-contract output mismatch")
        serology_model = metrics.get("serology_model") or {}
        if serology_model.get("name") != "prior_immunity_ridge_a10":
            raise BundleContractError("Task1.1 serology model output mismatch")
        if serology_model.get("family") != "ridge" or float((serology_model.get("params") or {}).get("alpha", -1)) != 10.0:
            raise BundleContractError("Task1.1 serology model parameter output mismatch")
        features = metrics.get("serology_summary_columns") or []
        if len(features) != 9 or "hai_panel_count" in features:
            raise BundleContractError("Task1.1 serology feature output mismatch")

        folds = metrics.get("folds")
        if not isinstance(folds, list) or len(folds) != 4:
            raise BundleContractError("Task1.1 held-study fold count mismatch")
        held_names = {str(fold.get("held_study", "")) for fold in folds}
        if held_names != {"SDY180", "SDY515", "SDY519", "SDY56"}:
            raise BundleContractError(f"Task1.1 held-study set mismatch: {sorted(held_names)}")
        for fold in folds:
            if int(fold.get("validation_rows", 0)) <= 0:
                raise BundleContractError("Task1.1 held-study row count missing")
            if int(fold.get("held_serology_rows", 0)) < 5:
                raise BundleContractError("Task1.1 held-study serology coverage below contract")
            fraction = float(fold.get("held_serology_fraction", -1))
            if not (0.0 < fraction <= 1.0):
                raise BundleContractError("Task1.1 held-study serology fraction invalid")
            conditions = fold.get("conditions") or {}
            required_conditions = set(BASE_CONDITIONS) | set(FUSION_CONDITIONS) | {"prior_only_overlap"}
            if set(conditions) != required_conditions:
                raise BundleContractError("Task1.1 held-study condition set mismatch")
            for condition in BASE_CONDITIONS + FUSION_CONDITIONS:
                metric = conditions.get(condition) or {}
                if int(metric.get("n", 0)) != int(fold["validation_rows"]):
                    raise BundleContractError(f"Task1.1 full-row metric n mismatch: {condition}")

        summary = metrics.get("summary") or {}
        if set(summary) != set(BASE_CONDITIONS) | set(FUSION_CONDITIONS):
            raise BundleContractError("Task1.1 summary condition set mismatch")
        for condition, item in summary.items():
            count = int((item or {}).get("count", 0))
            if count < 1 or count > 4:
                raise BundleContractError(f"Task1.1 summary finite-fold count invalid: {condition}")

        promotion = metrics.get("promotion") or {}
        if set(promotion) != set(FUSION_CONDITIONS):
            raise BundleContractError("Task1.1 promotion condition set mismatch")
        for condition, evidence in promotion.items():
            if not isinstance((evidence or {}).get("passed"), bool):
                raise BundleContractError(f"Task1.1 promotion flag missing: {condition}")
        selected = metrics.get("selected_promoted_condition")
        if selected is not None:
            if selected not in promotion or promotion[selected].get("passed") is not True:
                raise BundleContractError("Task1.1 selected promoted condition is invalid")

        coverage = metrics.get("historical_coverage_by_study") or {}
        if set(coverage) != {"SDY180", "SDY515", "SDY519", "SDY56"}:
            raise BundleContractError("Task1.1 historical coverage study set mismatch")
        if sum(int(item.get("target_rows", 0)) for item in coverage.values()) != 127:
            raise BundleContractError("Task1.1 historical target row total mismatch")

        challenge = metrics.get("challenge") or {}
        if int(challenge.get("rows", -1)) != 40:
            raise BundleContractError("Task1.1 challenge row count mismatch")
        challenge_serology_rows = int(challenge.get("serology_rows", -1))
        if not (1 <= challenge_serology_rows <= 40):
            raise BundleContractError("Task1.1 challenge serology coverage invalid")
        challenge_conditions = challenge.get("conditions") or {}
        if set(challenge_conditions) != set(FUSION_CONDITIONS):
            raise BundleContractError("Task1.1 challenge fusion condition set mismatch")
        for condition, evidence in challenge_conditions.items():
            for key in ("agreement_vs_base", "agreement_vs_anchor_residual"):
                agreement = (evidence or {}).get(key) or {}
                if int(agreement.get("n", -1)) != 40:
                    raise BundleContractError(f"Task1.1 challenge agreement n mismatch: {condition}/{key}")
                value = (agreement.get("rank_spearman") or {}).get("value")
                if value is None or not math.isfinite(float(value)):
                    raise BundleContractError(f"Task1.1 challenge agreement non-finite: {condition}/{key}")

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
            raise BundleContractError(f"Task1.1 prior-immunity output leaked row-level fields: {hits}")

        final_metrics = output_dir / "metrics.json"
        final_summary = output_dir / "summary.md"
        final_metrics.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        lines = [
            "# Phase B Task1.1 prior-immunity late fusion",
            "",
            f"- request_id: `{REQUEST_ID}`",
            f"- science_source_commit: `{SOURCE_COMMIT}`",
            f"- challenge serology coverage: `{challenge_serology_rows}/40`",
            "- held-target outcomes used for serology features: `false`",
            "- Competition submission: none",
            "",
            "## References",
            "",
        ]
        for condition in BASE_CONDITIONS:
            item = summary[condition]
            lines.append(
                f"- {condition}: mean=`{item.get('mean')}`, median=`{item.get('median')}`, "
                f"min=`{item.get('min')}`, max=`{item.get('max')}`"
            )
        lines.extend(["", "## Prior-immunity fusion", ""])
        for condition in FUSION_CONDITIONS:
            item = summary[condition]
            gate = promotion[condition]
            lines.append(
                f"- {condition}: mean=`{item.get('mean')}`, min=`{item.get('min')}`, "
                f"promotion=`{gate.get('passed')}`, wins=`{gate.get('wins')}`/`{gate.get('required_wins')}`"
            )
        lines.extend(
            [
                "",
                f"Selected promoted condition: `{selected}`",
                "",
                "Output is aggregate-only; public study names and coverage counts are retained, but no participant IDs or row-level predictions are written.",
            ]
        )
        final_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

        stage = "finalize_manifest"
        payload = {
            "schema_version": 1,
            "request_id": REQUEST_ID,
            "competition": COMPETITION,
            "kernel_stage": TARGET_STAGE,
            "source_commit": SOURCE_COMMIT,
            "base_package_zip_sha256": PACKAGE_ZIP_SHA256,
            "task11_prior_immunity_source_blob_sha": TASK11_PRIOR_IMMUNITY_SOURCE_BLOB_SHA,
            "task11_prior_immunity_source_sha256": TASK11_PRIOR_IMMUNITY_SOURCE_SHA256,
            "b21_runtime_adapter_sha256": B21_ADAPTER_SHA256,
            "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB_SHA,
            "python_version": platform.python_version(),
            "metrics_sha256": sha256_file(final_metrics),
            "summary": summary,
            "promotion": promotion,
            "selected_promoted_condition": selected,
            "historical_coverage_by_study": coverage,
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
        print(
            "CMI_FLU_TASK11_PRIOR_IMMUNITY_COMPLETE "
            f"request_id={REQUEST_ID} promoted={selected or 'none'}"
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
    parser.add_argument("--science-source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    science_path = args.science_source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    validate_request(root)
    science_source, science_sha256 = load_science(science_path)

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
        raise SystemExit("Task1.1 prior-immunity source must use materialized B2.1 runtime")
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
        'default=Path("/kaggle/working/cmi-flu-task11-prior-immunity-001")',
        label="output directory",
    )
    text = text.replace("CMI_FLU_B2_FAILED ", "CMI_FLU_TASK11_PRIOR_IMMUNITY_FAILED ")

    marker = "\ndef execute(input_dir: Path, output_dir: Path) -> Mapping[str, Any]:\n"
    text = replace_once(
        text,
        marker,
        injected_helpers(science_source, science_sha256) + marker,
        label="Task1.1 prior-immunity source insertion",
    )
    execute_start = text.index(marker.lstrip("\n"))
    main_marker = "\ndef main() -> int:\n"
    main_start = text.index(main_marker, execute_start)
    text = text[:execute_start] + execute_source() + text[main_start:]

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)

    namespace: dict[str, Any] = {"__name__": "task11_prior_immunity_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("generated Task1.1 prior-immunity runtime request mismatch")
    if namespace.get("SOURCE_COMMIT") != SCIENCE_COMMIT:
        raise SystemExit("generated Task1.1 prior-immunity runtime science commit mismatch")
    if namespace.get("TARGET_STAGE") != TARGET_STAGE:
        raise SystemExit("generated Task1.1 prior-immunity runtime stage mismatch")
    if namespace.get("TASK11_PRIOR_IMMUNITY_SOURCE_BLOB_SHA") != SCIENCE_BLOB:
        raise SystemExit("generated Task1.1 prior-immunity runtime science blob mismatch")
    if "kaggle competitions submit" in text or "api.competition_submit" in text:
        raise SystemExit("generated Task1.1 prior-immunity runtime contains forbidden submission path")
    if text.count("shutil.rmtree(runtime_root, ignore_errors=True)") != 2:
        raise SystemExit("runtime scratch cleanup must cover success and failure paths")

    run(sys.executable, str(output), "--self-test")
    print(
        "CMI_FLU_TASK11_PRIOR_IMMUNITY_PREPARE PASS "
        f"science_commit={SCIENCE_COMMIT} science_blob={SCIENCE_BLOB} "
        f"science_sha256={science_sha256} b21_adapter_blob={B21_ADAPTER_BLOB} "
        f"target_kernel={TARGET_KERNEL} expected_version=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
