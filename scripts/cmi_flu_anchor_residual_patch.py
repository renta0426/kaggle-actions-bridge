#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

BASE_REQUEST_ID = "20260904-cmi-flu-rank-transfer-003"
BASE_SOURCE_COMMIT = "9e9922806209241121274f9242a6cf09bf140d75"
REQUEST_ID = "20260904-cmi-flu-anchor-residual-001"
SCIENCE_SOURCE_COMMIT = "23ab4ff53d65eeb8b8e5582f5442081f245f03b3"
ANCHOR_RESIDUAL_BLOB = "9a7814ecdf70e0b38b2740ade21e9db588869379"
RANK_TRANSFER_BLOB = "d5e07cdd09d2eabdc935eb1733ec238e26ab4c17"
TARGET_STAGE = "phase_a_anchor_residual_task11_task12"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-anchor-residual-20260904-001"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def load_anchor_source(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    found = git_blob_sha(source.encode("utf-8"))
    if found != ANCHOR_RESIDUAL_BLOB:
        raise SystemExit(f"anchor-residual source blob mismatch: {found}")
    compile(source, "cmi_flu/anchor_residual.py", "exec")
    return source


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

        stage = "install_anchor_residual_compatibility"
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
                raise BundleContractError("anchor-residual compact compatibility requires robust_v1")
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
                raise BundleContractError("anchor-residual HAI compatibility requires robust_v1")
            return robust_hai(
                dataset,
                specs=specs,
                selection_panels=selection_panels,
                splits=splits,
            )

        evaluation.run_compact_task = compatible_compact
        evaluation.run_hai_compact_for_panels = compatible_hai

        stage = "load_rank_transfer_dependency"
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
        if not callable(getattr(rank_module, "within_group_rank_series", None)):
            raise BundleContractError("rank-transfer dependency lacks within_group_rank_series()")

        stage = "load_anchor_residual"
        anchor_module = types.ModuleType("cmi_flu.anchor_residual")
        anchor_module.__file__ = "<cmi_flu.anchor_residual>"
        anchor_module.__package__ = "cmi_flu"
        sys.modules["cmi_flu.anchor_residual"] = anchor_module
        exec(
            compile(ANCHOR_RESIDUAL_SOURCE, "cmi_flu/anchor_residual.py", "exec"),
            anchor_module.__dict__,
            anchor_module.__dict__,
        )
        run_experiment = getattr(anchor_module, "run_anchor_residual_experiment", None)
        weights = tuple(getattr(anchor_module, "SHRINKAGE_WEIGHTS", ()))
        if not callable(run_experiment):
            raise BundleContractError("anchor-residual source lacks run_anchor_residual_experiment()")
        if weights != (0.25, 0.5, 1.0):
            raise BundleContractError("anchor-residual shrinkage contract mismatch")

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
        if config.baseline != "b021_taskwise_robust":
            raise BundleContractError("runtime B2.1 compatibility promotion failed")
        inputs = load_inputs(config)

        stage = "run_anchor_residual"
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

        stage = "validate_anchor_residual"
        if metrics.get("experiment") != TARGET_STAGE:
            raise BundleContractError("anchor-residual experiment identity mismatch")
        if metrics.get("selection_policy") != "robust_anchor_residual_v1":
            raise BundleContractError("anchor-residual selection policy mismatch")
        if metrics.get("target_estimand") != "within_study_percentile_rank_residual_from_b1_anchor":
            raise BundleContractError("anchor-residual target estimand mismatch")
        if metrics.get("shrinkage_weights") != [0.0, 0.25, 0.5, 1.0]:
            raise BundleContractError("anchor-residual shrinkage weights mismatch")
        tasks = metrics.get("tasks")
        if not isinstance(tasks, dict) or set(tasks) != {"Task1.1", "Task1.2"}:
            raise BundleContractError("anchor-residual task set mismatch")
        expected_anchors = {
            "Task1.1": "cytokine_rank__CXCL10",
            "Task1.2": "flow_rank__Classical_monocytes",
        }
        for task in ("Task1.1", "Task1.2"):
            item = tasks[task]
            if item.get("anchor_column") != expected_anchors[task]:
                raise BundleContractError(f"{task} anchor column mismatch")
            b1 = item.get("b1_reference")
            raw = item.get("raw_b21_reference")
            selected = item.get("selected")
            candidates = item.get("candidate_summaries")
            if not isinstance(b1, dict) or not isinstance(raw, dict) or not isinstance(selected, dict):
                raise BundleContractError(f"{task} result shape mismatch")
            if not isinstance(candidates, list) or not candidates:
                raise BundleContractError(f"{task} candidate summaries missing")
            if int(b1.get("challenge_rows", -1)) != 40:
                raise BundleContractError(f"{task} B1 challenge row mismatch")
            for label, result in (("b1", b1), ("raw_b21", raw), ("selected", selected)):
                value = result.get("within_group_rank_spearman", {}).get("value")
                if value is None or not math.isfinite(float(value)):
                    raise BundleContractError(f"{task}/{label} metric is non-finite")
            selected_lambda = float(selected.get("lambda", -1.0))
            if selected_lambda not in (0.0, 0.25, 0.5, 1.0):
                raise BundleContractError(f"{task} selected lambda outside locked family")
            if not str(selected.get("model", "")):
                raise BundleContractError(f"{task} selected model missing")
            for key in ("challenge_agreement_vs_b1", "challenge_agreement_vs_raw_b21"):
                agreement = selected.get(key)
                if not isinstance(agreement, dict) or int(agreement.get("rows", -1)) != 40:
                    raise BundleContractError(f"{task}/{key} row mismatch")

        final_metrics = output_dir / "metrics.json"
        final_summary = output_dir / "summary.md"
        final_metrics.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# Phase A anchor-residual diagnostics",
            "",
            f"- request_id: `{REQUEST_ID}`",
            f"- science_source_commit: `{SOURCE_COMMIT}`",
            "- Competition submission: none",
            "",
        ]
        for task in ("Task1.1", "Task1.2"):
            item = tasks[task]
            selected = item["selected"]
            lines.extend(
                [
                    f"## {task}",
                    "",
                    f"- anchor: `{item['anchor_column']}`",
                    f"- b1_within: `{item['b1_reference']['within_group_rank_spearman']['value']}`",
                    f"- raw_b21_within: `{item['raw_b21_reference']['within_group_rank_spearman']['value']}`",
                    f"- selected: `{selected['model']}` with lambda `{selected['lambda']}`",
                    f"- selected_within: `{selected['within_group_rank_spearman']['value']}`",
                    f"- selected_rank_rmse: `{selected['rank_rmse']['value']}`",
                    f"- agreement_vs_b1: `{selected['challenge_agreement_vs_b1']['rank_spearman']['value']}`",
                    f"- agreement_vs_raw_b21: `{selected['challenge_agreement_vs_raw_b21']['rank_spearman']['value']}`",
                    "",
                ]
            )
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
            "anchor_residual_source_blob_sha": ANCHOR_RESIDUAL_SOURCE_BLOB_SHA,
            "anchor_residual_source_sha256": ANCHOR_RESIDUAL_SOURCE_SHA256,
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
            "CMI_FLU_ANCHOR_RESIDUAL_COMPLETE "
            f"request_id={REQUEST_ID} tasks=2 weights={len(weights) + 1}"
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
    parser.add_argument("--anchor-source", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("request_id") != REQUEST_ID:
        raise SystemExit("anchor-residual request identity mismatch")
    if request.get("target") != TARGET_KERNEL:
        raise SystemExit("anchor-residual target mismatch")
    if request.get("science_source_commit") != SCIENCE_SOURCE_COMMIT:
        raise SystemExit("anchor-residual science commit mismatch")
    if request.get("anchor_residual_blob_sha") != ANCHOR_RESIDUAL_BLOB:
        raise SystemExit("anchor-residual source manifest mismatch")
    if request.get("rank_transfer_base_request_id") != BASE_REQUEST_ID:
        raise SystemExit("rank-transfer base lineage mismatch")
    if request.get("rank_transfer_blob_sha") != RANK_TRANSFER_BLOB:
        raise SystemExit("rank-transfer dependency provenance mismatch")
    if request.get("b21_base_request_id") != "20260903-cmi-flu-b21-001":
        raise SystemExit("B2.1 lineage mismatch")
    if request.get("competition_submission_attempted") is not False:
        raise SystemExit("anchor-residual must not attempt Competition submission")
    if request.get("enable_internet") is not False or request.get("automatic_compute_retries") != 0:
        raise SystemExit("anchor-residual safety contract mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 90,
        "hard_timeout_minutes": 180,
        "max_active_runs": 1,
    }:
        raise SystemExit("anchor-residual resource contract mismatch")

    anchor_source = load_anchor_source(args.anchor_source)
    anchor_sha = hashlib.sha256(anchor_source.encode("utf-8")).hexdigest()
    text = args.source.read_text(encoding="utf-8")
    if "RANK_TRANSFER_SOURCE" not in text or "B21_ADAPTER_SOURCE" not in text:
        raise SystemExit("anchor-residual source must be materialized rank-transfer B2.1 runtime")
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
    if text.count('default=Path("/kaggle/working/cmi-flu-rank-transfer")') == 1:
        text = text.replace(
            'default=Path("/kaggle/working/cmi-flu-rank-transfer")',
            'default=Path("/kaggle/working/cmi-flu-anchor-residual")',
            1,
        )
    text = text.replace("CMI_FLU_RANK_TRANSFER_FAILED ", "CMI_FLU_ANCHOR_RESIDUAL_FAILED ")

    marker = "\ndef execute(input_dir: Path, output_dir: Path) -> Mapping[str, Any]:\n"
    injected = (
        f'\nANCHOR_RESIDUAL_SOURCE_BLOB_SHA = "{ANCHOR_RESIDUAL_BLOB}"\n'
        f'ANCHOR_RESIDUAL_SOURCE_SHA256 = "{anchor_sha}"\n'
        f'ANCHOR_RESIDUAL_SOURCE = {anchor_source!r}\n'
        + marker
    )
    text = replace_once(text, marker, injected, label="anchor-residual source insertion")
    execute_start = text.index(marker.lstrip("\n"))
    main_marker = "\ndef main() -> int:\n"
    main_start = text.index(main_marker, execute_start)
    text = text[:execute_start] + execute_source() + text[main_start:]

    compile(text, str(args.output), "exec")
    args.output.write_text(text, encoding="utf-8")
    print(
        "CMI_FLU_ANCHOR_RESIDUAL_PATCH PASS "
        f"anchor_blob={ANCHOR_RESIDUAL_BLOB} anchor_sha256={anchor_sha} "
        f"source_commit={SCIENCE_SOURCE_COMMIT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
