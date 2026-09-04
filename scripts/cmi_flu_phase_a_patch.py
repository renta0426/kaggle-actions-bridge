#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

B21_REQUEST_ID = "20260903-cmi-flu-b21-001"
B21_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"
PHASE_A_REQUEST_ID = "20260904-cmi-flu-phase-a-001"
PHASE_A_SOURCE_COMMIT = "0282262e049683135ec01a56f71b44a46356b194"
PHASE_A_SOURCE_BLOB = "bd9351a2857bc0964309ca117f60c766354066a0"
TARGET_STAGE = "phase_a_b1_vs_b21_same_contract"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def load_phase_a_source(parts_dir: Path) -> str:
    part0 = (parts_dir / "part-00").read_text(encoding="utf-8").rstrip("\n")
    part1 = (parts_dir / "part-01").read_text(encoding="utf-8").lstrip("\n").rstrip("\n")
    source = part0 + "\n" + part1 + "\n"
    data = source.encode("utf-8")
    blob = git_blob_sha(data)
    if blob != PHASE_A_SOURCE_BLOB:
        raise SystemExit(f"Phase A source blob mismatch: {blob}")
    compile(source, "cmi_flu/phase_a.py", "exec")
    return source


def phase_a_execute_source() -> str:
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
        data_link = data_parent / "raw"
        data_link.symlink_to(input_dir, target_is_directory=True)
        reference_dir = (
            runtime_root
            / "external"
            / "google-drive"
            / "challenge-resources"
            / "reference_files"
        )
        vaccine_panel, challenge_panel = derive_reference_files(input_dir, reference_dir)
        config_dir = runtime_root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "baseline_b02_taskwise.yaml"
        config_text = CONFIG_TEXT.rstrip() + "\nselection:\n  policy: robust_v1\n"
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

        stage = "install_phase_a_compatibility"
        import cmi_flu.evaluation as evaluation
        import cmi_flu.runner as runner

        robust_compact = runner.run_compact_task
        robust_hai = runner.run_hai_compact_for_panels

        def phase_a_compact(
            dataset: Any,
            *,
            specs: Any,
            splits: Any = None,
            random_state: int = 42,
            selection_policy: str = "robust_v1",
        ) -> Any:
            if selection_policy != "robust_v1":
                raise BundleContractError("Phase A compact compatibility requires robust_v1")
            return robust_compact(
                dataset,
                specs=specs,
                splits=splits,
                random_state=random_state,
            )

        def phase_a_hai(
            dataset: Any,
            *,
            specs: Any,
            selection_panels: Any,
            splits: Any = None,
            selection_policy: str = "robust_v1",
        ) -> Any:
            if selection_policy != "robust_v1":
                raise BundleContractError("Phase A HAI compatibility requires robust_v1")
            return robust_hai(
                dataset,
                specs=specs,
                selection_panels=selection_panels,
                splits=splits,
            )

        evaluation.run_compact_task = phase_a_compact
        evaluation.run_hai_compact_for_panels = phase_a_hai

        stage = "load_phase_a"
        import types
        phase_module = types.ModuleType("cmi_flu.phase_a")
        phase_module.__file__ = "<cmi_flu.phase_a>"
        phase_module.__package__ = "cmi_flu"
        sys.modules["cmi_flu.phase_a"] = phase_module
        exec(
            compile(PHASE_A_SOURCE, "cmi_flu/phase_a.py", "exec"),
            phase_module.__dict__,
            phase_module.__dict__,
        )
        run_phase_a = getattr(phase_module, "run_phase_a_anchor_comparison", None)
        if not callable(run_phase_a):
            raise BundleContractError("Phase A source lacks run_phase_a_anchor_comparison()")

        stage = "load_inputs"
        from cmi_flu.configuration import load_baseline_config
        from cmi_flu.runner import load_inputs
        config = load_baseline_config(config_path, repository_root=runtime_root)
        inputs = load_inputs(config)

        stage = "run_phase_a"
        metrics = dict(run_phase_a(config, inputs))
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

        stage = "validate_phase_a"
        if metrics.get("experiment") != TARGET_STAGE:
            raise BundleContractError("Phase A experiment identity mismatch")
        if metrics.get("selection_policy") != "robust_v1" or metrics.get("flow_contract") != "broad":
            raise BundleContractError("Phase A evaluation contract mismatch")
        macro = metrics.get("five_task_within_group_rank_macro")
        if not isinstance(macro, dict) or set(macro) != {"b1", "b21", "fixed_equal_rank_blend"}:
            raise BundleContractError("Phase A macro contract mismatch")
        for key, value in macro.items():
            if value is None or not math.isfinite(float(value)):
                raise BundleContractError(f"Phase A macro is not finite: {key}")
        tasks = metrics.get("task_comparisons")
        task_names = {"Task1.1", "Task1.2", "Task1.3", "Task1.4", "Task2.1", "Task2.2", "Task2.3"}
        if not isinstance(tasks, dict) or set(tasks) != task_names:
            raise BundleContractError("Phase A task comparison set mismatch")
        if tasks["Task1.4"].get("status") != "not_evaluable":
            raise BundleContractError("Phase A Task1.4 must remain not_evaluable")
        agreement = metrics.get("challenge_prediction_agreement")
        if not isinstance(agreement, dict) or set(agreement) != task_names:
            raise BundleContractError("Phase A challenge agreement set mismatch")
        if any(int(agreement[task].get("rows", -1)) != 40 for task in task_names):
            raise BundleContractError("Phase A challenge agreement must cover 40 donors per task")
        selected = metrics.get("selected_b21_models")
        if not isinstance(selected, dict) or set(selected) != task_names:
            raise BundleContractError("Phase A selected-model set mismatch")

        final_metrics = output_dir / "metrics.json"
        final_summary = output_dir / "summary.md"
        final_metrics.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        final_summary.write_text(
            "# Phase A: B1 vs B2.1 same-contract comparison\n\n"
            f"- B1 five-task within-group-rank proxy: `{macro['b1']}`\n"
            f"- B2.1 five-task within-group-rank proxy: `{macro['b21']}`\n"
            f"- Fixed 50/50 rank-blend proxy: `{macro['fixed_equal_rank_blend']}`\n"
            "- Task1.4: not supervised/evaluable\n"
            "- Kaggle competition submission: none\n",
            encoding="utf-8",
        )

        stage = "finalize_manifest"
        metrics_sha = sha256_file(final_metrics)
        payload = {
            "schema_version": 1,
            "request_id": REQUEST_ID,
            "competition": COMPETITION,
            "kernel_stage": TARGET_STAGE,
            "source_commit": SOURCE_COMMIT,
            "base_package_zip_sha256": PACKAGE_ZIP_SHA256,
            "phase_a_source_blob_sha": PHASE_A_SOURCE_BLOB_SHA,
            "phase_a_source_sha256": PHASE_A_SOURCE_SHA256,
            "b21_runtime_adapter_sha256": B21_ADAPTER_SHA256,
            "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB_SHA,
            "python_version": platform.python_version(),
            "metrics_sha256": metrics_sha,
            "derived_panel_sizes": {
                "vaccine": len(vaccine_panel),
                "challenge": len(challenge_panel),
            },
            "five_task_within_group_rank_macro": macro,
            "selected_b21_models": selected,
            "challenge_prediction_agreement": agreement,
            "checksum": metrics.get("checksum"),
            "competition_submission_attempted": False,
        }
        (output_dir / "bridge-result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            "CMI_FLU_PHASE_A_COMPLETE "
            f"b1={float(macro['b1']):.12g} "
            f"b21={float(macro['b21']):.12g} "
            f"blend={float(macro['fixed_equal_rank_blend']):.12g}"
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
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("request_id") != PHASE_A_REQUEST_ID:
        raise SystemExit("Phase A request identity mismatch")
    if request.get("science_source_commit") != PHASE_A_SOURCE_COMMIT:
        raise SystemExit("Phase A science commit mismatch")
    if request.get("phase_a_source_blob_sha") != PHASE_A_SOURCE_BLOB:
        raise SystemExit("Phase A source blob manifest mismatch")
    if request.get("b21_base_request_id") != B21_REQUEST_ID:
        raise SystemExit("Phase A B2.1 base request mismatch")
    if request.get("competition_submission_attempted") is not False:
        raise SystemExit("Phase A must not attempt competition submission")
    if request.get("enable_internet") is not False or request.get("automatic_compute_retries") != 0:
        raise SystemExit("Phase A safety contract mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 120,
        "hard_timeout_minutes": 240,
        "max_active_runs": 1,
    }:
        raise SystemExit("Phase A resource contract mismatch")

    phase_source = load_phase_a_source(args.parts_dir)
    phase_sha = hashlib.sha256(phase_source.encode("utf-8")).hexdigest()

    text = args.source.read_text(encoding="utf-8")
    if "B21_ADAPTER_SOURCE" not in text or "B21_ADAPTER_SHA256" not in text:
        raise SystemExit("Phase A source must be the materialized B2.1 runtime")
    text = replace_once(
        text,
        f'REQUEST_ID = "{B21_REQUEST_ID}"',
        f'REQUEST_ID = "{PHASE_A_REQUEST_ID}"',
        label="request id",
    )
    text = replace_once(
        text,
        f'SOURCE_COMMIT = "{B21_SOURCE_COMMIT}"',
        f'SOURCE_COMMIT = "{PHASE_A_SOURCE_COMMIT}"',
        label="science source commit",
    )

    marker = "\ndef execute(input_dir: Path, output_dir: Path) -> Mapping[str, Any]:\n"
    injected = (
        f'\nTARGET_STAGE = "{TARGET_STAGE}"\n'
        f'PHASE_A_SOURCE_BLOB_SHA = "{PHASE_A_SOURCE_BLOB}"\n'
        f'PHASE_A_SOURCE_SHA256 = "{phase_sha}"\n'
        f'PHASE_A_SOURCE = {phase_source!r}\n'
        + marker
    )
    text = replace_once(text, marker, injected, label="Phase A source insertion")

    execute_start = text.index(marker.lstrip("\n"))
    main_marker = "\ndef main() -> int:\n"
    main_start = text.index(main_marker, execute_start)
    text = text[:execute_start] + phase_a_execute_source() + text[main_start:]

    compile(text, str(args.output), "exec")
    args.output.write_text(text, encoding="utf-8")
    print(
        "CMI_FLU_PHASE_A_PATCH PASS "
        f"phase_a_blob={PHASE_A_SOURCE_BLOB} phase_a_sha256={phase_sha} "
        f"source_commit={PHASE_A_SOURCE_COMMIT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
