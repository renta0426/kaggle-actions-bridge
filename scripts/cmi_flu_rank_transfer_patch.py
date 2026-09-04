#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

B21_REQUEST_ID = "20260903-cmi-flu-b21-001"
B21_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"
REQUEST_ID = "20260904-cmi-flu-rank-transfer-001"
SCIENCE_SOURCE_COMMIT = "9e9922806209241121274f9242a6cf09bf140d75"
RANK_TRANSFER_BLOB = "d5e07cdd09d2eabdc935eb1733ec238e26ab4c17"
TARGET_STAGE = "phase_a_rank_transfer_task11_task12"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-rank-transfer-20260904-001"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def load_rank_source(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    data = source.encode("utf-8")
    found = git_blob_sha(data)
    if found != RANK_TRANSFER_BLOB:
        raise SystemExit(f"rank-transfer source blob mismatch: {found}")
    compile(source, "cmi_flu/rank_transfer.py", "exec")
    return source


def execute_source() -> str:
    return r'''
def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        import numpy as np
        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
        config_text = config_text.replace(
            "baseline: b02_taskwise_compact",
            "baseline: b021_taskwise_robust",
            1,
        )
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

        stage = "install_rank_transfer_compatibility"
        import cmi_flu.evaluation as evaluation
        import cmi_flu.models as models
        import cmi_flu.runner as runner

        metric_summary = adapter_namespace.get("_metric_summary")
        if not callable(metric_summary):
            raise BundleContractError("B2.1 adapter lacks metric-summary compatibility")
        models.summarize_metric_frame = metric_summary

        robust_compact = runner.run_compact_task
        robust_hai = runner.run_hai_compact_for_panels

        def compatible_compact(
            dataset: Any,
            *,
            specs: Any,
            splits: Any = None,
            random_state: int = 42,
            selection_policy: str = "robust_v1",
        ) -> Any:
            if selection_policy != "robust_v1":
                raise BundleContractError("rank-transfer compact compatibility requires robust_v1")
            return robust_compact(
                dataset,
                specs=specs,
                splits=splits,
                random_state=random_state,
            )

        def compatible_hai(
            dataset: Any,
            *,
            specs: Any,
            selection_panels: Any,
            splits: Any = None,
            selection_policy: str = "robust_v1",
        ) -> Any:
            if selection_policy != "robust_v1":
                raise BundleContractError("rank-transfer HAI compatibility requires robust_v1")
            return robust_hai(
                dataset,
                specs=specs,
                selection_panels=selection_panels,
                splits=splits,
            )

        evaluation.run_compact_task = compatible_compact
        evaluation.run_hai_compact_for_panels = compatible_hai

        stage = "load_rank_transfer"
        import types
        rank_module = types.ModuleType("cmi_flu.rank_transfer")
        rank_module.__file__ = "<cmi_flu.rank_transfer>"
        rank_module.__package__ = "cmi_flu"
        sys.modules["cmi_flu.rank_transfer"] = rank_module
        exec(
            compile(RANK_TRANSFER_SOURCE, "cmi_flu/rank_transfer.py", "exec"),
            rank_module.__dict__,
            rank_module.__dict__,
        )
        run_experiment = getattr(rank_module, "run_rank_transfer_experiment", None)
        variants = tuple(getattr(rank_module, "VARIANTS", ()))
        if not callable(run_experiment):
            raise BundleContractError("rank-transfer source lacks run_rank_transfer_experiment()")
        expected_variants = (
            "target_rank_raw",
            "target_rank_rank_only",
            "target_rank_raw_plus_rank",
        )
        if variants != expected_variants:
            raise BundleContractError("rank-transfer source variant contract mismatch")

        stage = "load_inputs"
        from cmi_flu.configuration import load_baseline_config
        from cmi_flu.runner import load_inputs
        config = load_baseline_config(config_path, repository_root=runtime_root)
        if config.baseline != "b021_taskwise_robust":
            raise BundleContractError("runtime B2.1 config identity mismatch")
        inputs = load_inputs(config)

        stage = "run_rank_transfer"
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
        metrics = json_safe(metrics)

        stage = "validate_rank_transfer"
        if metrics.get("experiment") != TARGET_STAGE:
            raise BundleContractError("rank-transfer experiment identity mismatch")
        if metrics.get("selection_policy") != "robust_v1":
            raise BundleContractError("rank-transfer selection policy mismatch")
        if metrics.get("target_estimand") != "within_study_percentile_rank":
            raise BundleContractError("rank-transfer target estimand mismatch")
        tasks = metrics.get("tasks")
        if not isinstance(tasks, dict) or set(tasks) != {"Task1.1", "Task1.2"}:
            raise BundleContractError("rank-transfer task set mismatch")
        for task in ("Task1.1", "Task1.2"):
            item = tasks[task]
            raw = item.get("raw_b21_reference")
            rank_variants = item.get("rank_variants")
            agreement = item.get("challenge_agreement_vs_raw_b21")
            if not isinstance(raw, dict) or not isinstance(rank_variants, dict) or not isinstance(agreement, dict):
                raise BundleContractError(f"{task} result shape mismatch")
            if set(rank_variants) != set(variants) or set(agreement) != set(variants):
                raise BundleContractError(f"{task} variant set mismatch")
            raw_value = raw.get("within_group_rank_spearman", {}).get("value")
            if raw_value is None or not math.isfinite(float(raw_value)):
                raise BundleContractError(f"{task} raw reference is non-finite")
            for variant in variants:
                value = rank_variants[variant].get("within_group_rank_spearman", {}).get("value")
                if value is None or not math.isfinite(float(value)):
                    raise BundleContractError(f"{task}/{variant} metric is non-finite")
                if int(agreement[variant].get("rows", -1)) != 40:
                    raise BundleContractError(f"{task}/{variant} challenge agreement row mismatch")

        final_metrics = output_dir / "metrics.json"
        final_summary = output_dir / "summary.md"
        final_metrics.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# Phase A rank-transfer diagnostics",
            "",
            f"- request_id: `{REQUEST_ID}`",
            f"- science_source_commit: `{SOURCE_COMMIT}`",
            "- Competition submission: none",
            "",
        ]
        for task in ("Task1.1", "Task1.2"):
            item = tasks[task]
            raw_value = item["raw_b21_reference"]["within_group_rank_spearman"]["value"]
            lines.extend([f"## {task}", "", f"- raw_b21: `{raw_value}`"])
            for variant in variants:
                result = item["rank_variants"][variant]
                lines.append(
                    f"- {variant}: `{result['within_group_rank_spearman']['value']}` "
                    f"using `{result['selected_model']}`"
                )
            lines.append("")
        lines.append("Output is aggregate-only; no participant IDs or row-level predictions are written.")
        final_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

        stage = "finalize_manifest"
        metrics_sha = sha256_file(final_metrics)
        payload = {
            "schema_version": 1,
            "request_id": REQUEST_ID,
            "competition": COMPETITION,
            "kernel_stage": TARGET_STAGE,
            "source_commit": SOURCE_COMMIT,
            "base_package_zip_sha256": PACKAGE_ZIP_SHA256,
            "rank_transfer_source_blob_sha": RANK_TRANSFER_SOURCE_BLOB_SHA,
            "rank_transfer_source_sha256": RANK_TRANSFER_SOURCE_SHA256,
            "b21_runtime_adapter_sha256": B21_ADAPTER_SHA256,
            "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB_SHA,
            "python_version": platform.python_version(),
            "metrics_sha256": metrics_sha,
            "derived_panel_sizes": {
                "vaccine": len(vaccine_panel),
                "challenge": len(challenge_panel),
            },
            "tasks": tasks,
            "checksum": metrics.get("checksum"),
            "competition_submission_attempted": False,
        }
        (output_dir / "bridge-result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            "CMI_FLU_RANK_TRANSFER_COMPLETE "
            f"request_id={REQUEST_ID} tasks=2 variants={len(variants)}"
        )
        return payload
    except Exception as error:
        safe_failure(output_dir, stage=stage, error=error)
        traceback.print_exc()
        raise
'''.lstrip("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rank-source", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("request_id") != REQUEST_ID:
        raise SystemExit("rank-transfer request identity mismatch")
    if request.get("target") != TARGET_KERNEL:
        raise SystemExit("rank-transfer target mismatch")
    if request.get("science_source_commit") != SCIENCE_SOURCE_COMMIT:
        raise SystemExit("rank-transfer science commit mismatch")
    if request.get("rank_transfer_blob_sha") != RANK_TRANSFER_BLOB:
        raise SystemExit("rank-transfer source manifest mismatch")
    if request.get("competition_submission_attempted") is not False:
        raise SystemExit("rank-transfer must not attempt Competition submission")
    if request.get("enable_internet") is not False or request.get("automatic_compute_retries") != 0:
        raise SystemExit("rank-transfer safety contract mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 60,
        "hard_timeout_minutes": 180,
        "max_active_runs": 1,
    }:
        raise SystemExit("rank-transfer resource contract mismatch")

    rank_source = load_rank_source(args.rank_source)
    rank_sha = hashlib.sha256(rank_source.encode("utf-8")).hexdigest()
    text = args.source.read_text(encoding="utf-8")
    if "B21_ADAPTER_SOURCE" not in text or "B21_ADAPTER_SHA256" not in text:
        raise SystemExit("rank-transfer source must be the materialized B2.1 runtime")
    text = replace_once(
        text,
        f'REQUEST_ID = "{B21_REQUEST_ID}"',
        f'REQUEST_ID = "{REQUEST_ID}"',
        label="request id",
    )
    text = replace_once(
        text,
        f'SOURCE_COMMIT = "{B21_SOURCE_COMMIT}"',
        f'SOURCE_COMMIT = "{SCIENCE_SOURCE_COMMIT}"',
        label="science source commit",
    )
    if text.count('default=Path("/kaggle/working/cmi-flu-b2")') == 1:
        text = text.replace(
            'default=Path("/kaggle/working/cmi-flu-b2")',
            'default=Path("/kaggle/working/cmi-flu-rank-transfer")',
            1,
        )
    text = text.replace("CMI_FLU_B2_FAILED ", "CMI_FLU_RANK_TRANSFER_FAILED ")

    marker = "\ndef execute(input_dir: Path, output_dir: Path) -> Mapping[str, Any]:\n"
    injected = (
        f'\nTARGET_STAGE = "{TARGET_STAGE}"\n'
        f'RANK_TRANSFER_SOURCE_BLOB_SHA = "{RANK_TRANSFER_BLOB}"\n'
        f'RANK_TRANSFER_SOURCE_SHA256 = "{rank_sha}"\n'
        f'RANK_TRANSFER_SOURCE = {rank_source!r}\n'
        + marker
    )
    text = replace_once(text, marker, injected, label="rank-transfer source insertion")
    execute_start = text.index(marker.lstrip("\n"))
    main_marker = "\ndef main() -> int:\n"
    main_start = text.index(main_marker, execute_start)
    text = text[:execute_start] + execute_source() + text[main_start:]

    compile(text, str(args.output), "exec")
    args.output.write_text(text, encoding="utf-8")
    print(
        "CMI_FLU_RANK_TRANSFER_PATCH PASS "
        f"rank_blob={RANK_TRANSFER_BLOB} rank_sha256={rank_sha} "
        f"source_commit={SCIENCE_SOURCE_COMMIT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
