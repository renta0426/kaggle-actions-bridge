#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASE_REQUEST_ID = "20260903-cmi-flu-b21-001"
BASE_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"
REQUEST_ID = "20260904-cmi-flu-task13-sdy272-001"
SCIENCE_SOURCE_COMMIT = "baae9fb40329057a316935d0d1285adc64c948b0"
TASK13_BLOB = "0f1e728fe2e5ea0f3713c1442c3beeba21b8d347"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
TARGET_STAGE = "phase_a_task13_sdy272_harmonization"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-task13-sdy272-20260904-001"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def load_task13_source(path: Path) -> str:
    data = path.read_bytes()
    found = git_blob_sha(data)
    if found != TASK13_BLOB:
        raise SystemExit(f"Task1.3 harmonization source blob mismatch: {found}")
    source = data.decode("utf-8")
    compile(source, "cmi_flu/task13_harmonization.py", "exec")
    return source


def injected_helpers(task13_source: str, task13_sha256: str) -> str:
    return f'''\nTARGET_STAGE = "{TARGET_STAGE}"\nTASK13_HARMONIZATION_SOURCE_BLOB_SHA = "{TASK13_BLOB}"\nTASK13_HARMONIZATION_SOURCE_SHA256 = "{task13_sha256}"\nTASK13_HARMONIZATION_SOURCE = {task13_source!r}\n\ndef task13_json_safe(value: Any) -> Any:\n    if value is None or isinstance(value, (str, bool, int)):\n        return value\n    if isinstance(value, float):\n        return value if math.isfinite(value) else None\n    if isinstance(value, Mapping):\n        return {{str(key): task13_json_safe(item) for key, item in value.items()}}\n    if isinstance(value, (list, tuple)):\n        return [task13_json_safe(item) for item in value]\n    item = getattr(value, "item", None)\n    if callable(item):\n        try:\n            return task13_json_safe(item())\n        except (ValueError, TypeError):\n            pass\n    return str(value)\n'''


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
        vaccine_panel, challenge_panel = derive_reference_files(input_dir, reference_dir)

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

        stage = "install_task13_compatibility"
        import cmi_flu.evaluation as evaluation
        import cmi_flu.runner as runner

        robust_compact = runner.run_compact_task

        def compatible_compact(
            dataset: Any,
            *,
            specs: Any,
            splits: Any = None,
            random_state: int = 42,
            selection_policy: str = "robust_v1",
        ) -> Any:
            if selection_policy != "robust_v1":
                raise BundleContractError("Task1.3 compatibility requires robust_v1")
            return robust_compact(
                dataset,
                specs=specs,
                splits=splits,
                random_state=random_state,
            )

        evaluation.run_compact_task = compatible_compact

        stage = "load_task13_harmonization"
        import types
        task13_module = types.ModuleType("cmi_flu.task13_harmonization")
        task13_module.__file__ = "<cmi_flu.task13_harmonization>"
        task13_module.__package__ = "cmi_flu"
        sys.modules["cmi_flu.task13_harmonization"] = task13_module
        exec(
            compile(
                TASK13_HARMONIZATION_SOURCE,
                "cmi_flu/task13_harmonization.py",
                "exec",
            ),
            task13_module.__dict__,
            task13_module.__dict__,
        )
        run_experiment = getattr(
            task13_module,
            "run_task13_sdy272_harmonization_experiment",
            None,
        )
        if not callable(run_experiment):
            raise BundleContractError("Task1.3 source lacks experiment entry point")
        if str(getattr(task13_module, "PROXY_STUDY", "")) != "SDY272":
            raise BundleContractError("Task1.3 proxy-study contract mismatch")
        if tuple(getattr(task13_module, "GATE_CONFIDENCE_WEIGHTS", ())) != (0.5, 0.75, 1.0):
            raise BundleContractError("Task1.3 gate-confidence contract mismatch")

        stage = "load_inputs"
        from cmi_flu.configuration import load_baseline_config
        from cmi_flu.runner import load_inputs
        config = load_baseline_config(config_path, repository_root=runtime_root)
        if config.baseline != "b02_taskwise_compact":
            raise BundleContractError("legacy embedded loader returned unexpected baseline")
        if str(config.section("selection").get("policy", "")) != "robust_v1":
            raise BundleContractError("runtime robust_v1 selection contract missing")
        if str(config.section("flow").get("task_13_mode", "")) != "broad":
            raise BundleContractError("runtime Task1.3 broad-flow contract missing")
        raw_compat = dict(config.raw)
        raw_compat["baseline"] = "b021_taskwise_robust"
        object.__setattr__(config, "raw", raw_compat)
        object.__setattr__(config, "baseline", "b021_taskwise_robust")
        inputs = load_inputs(config)

        stage = "run_task13_sdy272"
        metrics = dict(run_experiment(config, inputs))
        metrics["run_id"] = REQUEST_ID
        metrics["checksum"] = (
            {
                "verified_count": len(inputs.checksum_report.verified),
                "skipped": list(inputs.checksum_report.skipped),
            }
            if inputs.checksum_report is not None
            else None
        )
        metrics = task13_json_safe(metrics)

        stage = "validate_task13_sdy272"
        if metrics.get("experiment") != TARGET_STAGE:
            raise BundleContractError("Task1.3 experiment identity mismatch")
        if metrics.get("selection_policy") != "robust_v1":
            raise BundleContractError("Task1.3 selection policy mismatch")
        if metrics.get("flow_mode") != "broad" or metrics.get("proxy_study") != "SDY272":
            raise BundleContractError("Task1.3 flow/proxy contract mismatch")
        if metrics.get("leaderboard_used_for_selection") is not False:
            raise BundleContractError("Task1.3 unexpectedly used leaderboard")
        if metrics.get("competition_submission_attempted") is not False:
            raise BundleContractError("Task1.3 unexpectedly attempted submission")
        if metrics.get("output_policy") != "aggregate_only_no_participant_ids_or_row_level_predictions":
            raise BundleContractError("Task1.3 output-policy mismatch")

        gate = metrics.get("gate_diagnostics")
        if not isinstance(gate, dict) or int(gate.get("matched_rows", 0)) <= 0 or int(gate.get("matched_participants", 0)) <= 0:
            raise BundleContractError("Task1.3 SDY272 gate matched no data")

        conditions = metrics.get("conditions")
        expected_conditions = {
            "strict",
            "proxy_target_only",
            "gate_harmonized_raw",
            "gate_harmonized_rank",
        }
        if not isinstance(conditions, dict) or set(conditions) != expected_conditions:
            raise BundleContractError("Task1.3 condition set mismatch")
        for name, item in conditions.items():
            if int(item.get("training_rows", 0)) <= 0 or int(item.get("training_subjects", 0)) <= 0:
                raise BundleContractError(f"{name} training cohort is empty")
            if int(item.get("training_studies", 0)) <= 0:
                raise BundleContractError(f"{name} study count is invalid")
            selected = item.get("selected_model")
            if not isinstance(selected, dict) or not str(selected.get("name", "")):
                raise BundleContractError(f"{name} selected model missing")
            pooled = item.get("pooled") or {}
            spearman = (pooled.get("spearman") or {}).get("value")
            rmse = (pooled.get("rmse") or {}).get("value")
            if spearman is None or not math.isfinite(float(spearman)):
                raise BundleContractError(f"{name} pooled Spearman is non-finite")
            if rmse is None or not math.isfinite(float(rmse)):
                raise BundleContractError(f"{name} pooled RMSE is non-finite")
            rank_rmse = item.get("rank_rmse")
            if rank_rmse is None or not math.isfinite(float(rank_rmse)):
                raise BundleContractError(f"{name} rank RMSE is non-finite")
            fold_summary = item.get("fold_summary") or {}
            if int(fold_summary.get("count", 0)) <= 0:
                raise BundleContractError(f"{name} fold summary is empty")
        if int(conditions["gate_harmonized_rank"].get("training_studies", 0)) < 2:
            raise BundleContractError("rank-harmonized condition is not cross-study")

        confidence = metrics.get("gate_confidence_selection")
        if not isinstance(confidence, dict) or confidence.get("weights") != [0.5, 0.75, 1.0]:
            raise BundleContractError("Task1.3 gate-confidence weights mismatch")
        confidence_results = confidence.get("results")
        if not isinstance(confidence_results, list) or len(confidence_results) != 3:
            raise BundleContractError("Task1.3 gate-confidence result count mismatch")
        if not isinstance(confidence.get("selected_model_stable"), bool):
            raise BundleContractError("Task1.3 gate-confidence stability flag missing")
        if not isinstance(confidence.get("weight_1_matches_robust_v1"), bool):
            raise BundleContractError("Task1.3 weight-1 consistency flag missing")

        agreements = metrics.get("challenge_rank_agreements")
        expected_agreements = {
            "proxy_target_only_vs_strict",
            "gate_harmonized_raw_vs_strict",
            "gate_harmonized_rank_vs_strict",
            "gate_harmonized_rank_vs_raw",
        }
        if not isinstance(agreements, dict) or set(agreements) != expected_agreements:
            raise BundleContractError("Task1.3 challenge agreement set mismatch")
        for name, item in agreements.items():
            if int(item.get("n", -1)) != 40:
                raise BundleContractError(f"{name} challenge row count mismatch")
            value = (item.get("rank_spearman") or {}).get("value")
            if value is None or not math.isfinite(float(value)):
                raise BundleContractError(f"{name} challenge rank agreement is non-finite")

        promotion = metrics.get("promotion_gate")
        if not isinstance(promotion, dict) or not isinstance(promotion.get("passed"), bool):
            raise BundleContractError("Task1.3 promotion gate missing")
        fold_spearman = promotion.get("fold_spearman")
        if not isinstance(fold_spearman, dict) or len(fold_spearman) < 2:
            raise BundleContractError("Task1.3 promotion fold evidence incomplete")

        final_metrics = output_dir / "metrics.json"
        final_summary = output_dir / "summary.md"
        final_metrics.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        lines = [
            "# Phase A Task1.3 SDY272 harmonization diagnostics",
            "",
            f"- request_id: `{REQUEST_ID}`",
            f"- science_source_commit: `{SOURCE_COMMIT}`",
            f"- matched SDY272 gate rows: `{gate['matched_rows']}`",
            f"- matched SDY272 gate participants: `{gate['matched_participants']}`",
            "- Competition submission: none",
            "",
        ]
        for name in ("strict", "proxy_target_only", "gate_harmonized_raw", "gate_harmonized_rank"):
            item = conditions[name]
            summary = item["fold_summary"]
            lines.extend(
                [
                    f"## {name}",
                    "",
                    f"- selected_model: `{item['selected_model']['name']}`",
                    f"- training_rows: `{item['training_rows']}`",
                    f"- training_subjects: `{item['training_subjects']}`",
                    f"- training_studies: `{item['training_studies']}`",
                    f"- held_spearman_mean: `{summary.get('spearman_mean')}`",
                    f"- held_spearman_median: `{summary.get('spearman_median')}`",
                    f"- held_spearman_min: `{summary.get('spearman_min')}`",
                    f"- pooled_spearman: `{item['pooled']['spearman']['value']}`",
                    f"- pooled_rmse: `{item['pooled']['rmse']['value']}`",
                    f"- within_study_rank_spearman: `{item['within_study_rank_spearman']['value']}`",
                    f"- rank_rmse: `{item['rank_rmse']}`",
                    "",
                ]
            )
        lines.extend(
            [
                "## Promotion gate",
                "",
                f"- passed: `{promotion['passed']}`",
                f"- rule: `{promotion['rule']}`",
                f"- gate-confidence selected model stable: `{confidence['selected_model_stable']}`",
                f"- weight 1.00 matches robust_v1: `{confidence['weight_1_matches_robust_v1']}`",
                "",
                "Output is aggregate-only; no participant IDs or row-level predictions are written.",
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
            "task13_harmonization_source_blob_sha": TASK13_HARMONIZATION_SOURCE_BLOB_SHA,
            "task13_harmonization_source_sha256": TASK13_HARMONIZATION_SOURCE_SHA256,
            "b21_runtime_adapter_sha256": B21_ADAPTER_SHA256,
            "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB_SHA,
            "python_version": platform.python_version(),
            "metrics_sha256": sha256_file(final_metrics),
            "derived_panel_sizes": {
                "vaccine": len(vaccine_panel),
                "challenge": len(challenge_panel),
            },
            "gate_diagnostics": gate,
            "condition_summaries": {
                name: {
                    "selected_model": item["selected_model"]["name"],
                    "training_rows": item["training_rows"],
                    "training_subjects": item["training_subjects"],
                    "training_studies": item["training_studies"],
                    "pooled": item["pooled"],
                    "fold_summary": item["fold_summary"],
                    "within_study_rank_spearman": item["within_study_rank_spearman"],
                    "rank_rmse": item["rank_rmse"],
                }
                for name, item in conditions.items()
            },
            "gate_confidence_selection": confidence,
            "challenge_rank_agreements": agreements,
            "promotion_gate": promotion,
            "checksum": metrics.get("checksum"),
            "competition_submission_attempted": False,
        }
        (output_dir / "bridge-result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(runtime_root, ignore_errors=True)
        print(
            "CMI_FLU_TASK13_SDY272_COMPLETE "
            f"request_id={REQUEST_ID} promotion={promotion['passed']} "
            f"matched_rows={gate['matched_rows']} matched_participants={gate['matched_participants']}"
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task13-source", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("request_id") != REQUEST_ID:
        raise SystemExit("Task1.3 request identity mismatch")
    if request.get("target") != TARGET_KERNEL:
        raise SystemExit("Task1.3 target mismatch")
    if request.get("science_source_commit") != SCIENCE_SOURCE_COMMIT:
        raise SystemExit("Task1.3 science commit mismatch")
    if request.get("task13_harmonization_blob_sha") != TASK13_BLOB:
        raise SystemExit("Task1.3 source manifest mismatch")
    if request.get("b21_base_request_id") != BASE_REQUEST_ID:
        raise SystemExit("B2.1 lineage mismatch")
    if request.get("b21_runtime_adapter_blob_sha") != B21_ADAPTER_BLOB:
        raise SystemExit("B2.1 adapter provenance mismatch")
    if request.get("competition_submission_attempted") is not False:
        raise SystemExit("Task1.3 must not attempt Competition submission")
    if request.get("enable_internet") is not False or request.get("automatic_compute_retries") != 0:
        raise SystemExit("Task1.3 safety contract mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 90,
        "hard_timeout_minutes": 180,
        "max_active_runs": 1,
    }:
        raise SystemExit("Task1.3 resource contract mismatch")

    task13_source = load_task13_source(args.task13_source)
    task13_sha256 = hashlib.sha256(task13_source.encode("utf-8")).hexdigest()
    text = args.source.read_text(encoding="utf-8")
    if "B21_ADAPTER_SOURCE" not in text:
        raise SystemExit("Task1.3 source must be materialized B2.1 runtime")
    text = replace_once(
        text,
        f'REQUEST_ID = "{BASE_REQUEST_ID}"',
        f'REQUEST_ID = "{REQUEST_ID}"',
        label="request id",
    )
    text = replace_once(
        text,
        f'SOURCE_COMMIT = "{BASE_SOURCE_COMMIT}"',
        f'SOURCE_COMMIT = "{SCIENCE_SOURCE_COMMIT}"',
        label="science source commit",
    )
    text = replace_once(
        text,
        'default=Path("/kaggle/working/cmi-flu-b2")',
        'default=Path("/kaggle/working/cmi-flu-task13-sdy272")',
        label="output directory",
    )
    text = text.replace("CMI_FLU_B2_FAILED ", "CMI_FLU_TASK13_SDY272_FAILED ")

    marker = "\ndef execute(input_dir: Path, output_dir: Path) -> Mapping[str, Any]:\n"
    text = replace_once(
        text,
        marker,
        injected_helpers(task13_source, task13_sha256) + marker,
        label="Task1.3 source insertion",
    )
    execute_start = text.index(marker.lstrip("\n"))
    main_marker = "\ndef main() -> int:\n"
    main_start = text.index(main_marker, execute_start)
    text = text[:execute_start] + execute_source() + text[main_start:]

    compile(text, str(args.output), "exec")
    args.output.write_text(text, encoding="utf-8")
    print(
        "CMI_FLU_TASK13_SDY272_PATCH PASS "
        f"task13_blob={TASK13_BLOB} task13_sha256={task13_sha256} "
        f"source_commit={SCIENCE_SOURCE_COMMIT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
