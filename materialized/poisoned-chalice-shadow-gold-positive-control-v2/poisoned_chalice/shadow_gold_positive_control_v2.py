"""Narrow protocol-manifest compatibility repair for Gold positive control v1.

The first Kaggle execution of ``shadow-gold-positive-control-v1`` failed before
any model forward pass because its freshly written protocol manifest omitted two
sentinel fields required by the unchanged Gold training loader:
``optimiser_steps_performed`` and ``accelerator_selected``.  The loader treats a
missing sentinel as a failed no-compute guard.

This module keeps the frozen positive-control corpus, split, probe, exposure
schedule, model pair, optimizer, scoring primitives, and sensitivity gate
unchanged.  It only restores the exact four no-compute sentinel values used by
the original Gold protocol manifest before either GPU child is launched.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from . import shadow_gold_kaggle as gold_v1
from . import shadow_gold_kaggle_v2 as gold_v2
from . import shadow_gold_positive_control as v1
from .shadow_training_dual_gpu import validate_gpu_inventory, visible_cuda_device_names


SHADOW_GOLD_POSITIVE_CONTROL_MANIFEST_REPAIR_VERSION = (
    "shadow-gold-positive-control-no-compute-manifest-v2"
)
NO_COMPUTE_GUARD = {
    "optimiser_steps_performed": 0,
    "model_compute_started": False,
    "accelerator_selected": False,
    "kaggle_operation_performed": False,
}


class ShadowGoldPositiveControlManifestRepairError(RuntimeError):
    """Raised when the narrow manifest repair cannot be applied safely."""


def repair_protocol_manifest(protocol_directory: str | Path) -> dict[str, Any]:
    """Restore the original Gold no-compute sentinel without changing science."""

    root = Path(protocol_directory).resolve()
    path = root / "shadow_protocol_manifest.json"
    if not path.is_file():
        raise ShadowGoldPositiveControlManifestRepairError(
            "positive-control protocol manifest does not exist"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise ShadowGoldPositiveControlManifestRepairError(
            "positive-control protocol manifest is not frozen"
        )
    if manifest.get("operation") != "freeze_gold_positive_control_protocol":
        raise ShadowGoldPositiveControlManifestRepairError(
            "positive-control protocol operation changed"
        )
    if manifest.get("experiment_id") != v1.EXPERIMENT_ID:
        raise ShadowGoldPositiveControlManifestRepairError(
            "positive-control experiment identity changed"
        )
    if manifest.get("positive_control_version") != v1.SHADOW_GOLD_POSITIVE_CONTROL_VERSION:
        raise ShadowGoldPositiveControlManifestRepairError(
            "positive-control version changed"
        )
    if manifest.get("probe_version") != v1.PROBE_VERSION:
        raise ShadowGoldPositiveControlManifestRepairError(
            "positive-control probe version changed"
        )
    protocol = manifest.get("protocol") or {}
    if protocol.get("pretrained_weights_used") is not False:
        raise ShadowGoldPositiveControlManifestRepairError(
            "positive-control pretrained-weight guard changed"
        )
    if protocol.get("target_evaluation_labels_used_for_training") is not False:
        raise ShadowGoldPositiveControlManifestRepairError(
            "positive-control label guard changed"
        )
    if protocol.get("membership_labels_exact_for_emitted_training_corpus") is not True:
        raise ShadowGoldPositiveControlManifestRepairError(
            "positive-control exact-membership assertion changed"
        )

    for key, expected in NO_COMPUTE_GUARD.items():
        if key in manifest and manifest[key] != expected:
            raise ShadowGoldPositiveControlManifestRepairError(
                f"refusing to overwrite conflicting no-compute sentinel: {key}"
            )
        manifest[key] = expected
    manifest["protocol_manifest_repair"] = (
        SHADOW_GOLD_POSITIVE_CONTROL_MANIFEST_REPAIR_VERSION
    )
    manifest["scientific_protocol_changed"] = False
    gold_v1._write_json(path, manifest)

    repaired = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in NO_COMPUTE_GUARD.items():
        if repaired.get(key) != expected:
            raise ShadowGoldPositiveControlManifestRepairError(
                f"positive-control sentinel repair did not persist: {key}"
            )
    if repaired.get("scientific_protocol_changed") is not False:
        raise ShadowGoldPositiveControlManifestRepairError(
            "positive-control repair changed scientific protocol marker"
        )
    return repaired


def prepare_positive_control_runtime(
    root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the frozen v1 preparation and repair only its protocol sentinel."""

    prepared = v1.prepare_positive_control_runtime(root, config)
    repaired = repair_protocol_manifest(prepared["protocol_directory"])
    result = dict(prepared)
    result["protocol_manifest_repair"] = (
        SHADOW_GOLD_POSITIVE_CONTROL_MANIFEST_REPAIR_VERSION
    )
    result["scientific_protocol_changed"] = False
    result["protocol_no_compute_guard"] = {
        key: repaired[key] for key in NO_COMPUTE_GUARD
    }
    return result


def run_positive_control_benchmark(
    *,
    scratch_directory: str | Path,
    config: Mapping[str, Any],
    timeout_seconds: int = 5400,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Run the unchanged v1 calibration after the narrow manifest repair."""

    v1.validate_positive_control_config(config)
    names = visible_cuda_device_names()
    validate_gpu_inventory(names, expected_count=2, required_name_fragment="T4")
    scratch = Path(scratch_directory).resolve()
    if scratch.exists() and any(scratch.iterdir()):
        raise v1.ShadowGoldPositiveControlError(
            "positive-control scratch directory is not empty"
        )
    scratch.mkdir(parents=True, exist_ok=True)
    prepared = prepare_positive_control_runtime(scratch / "prepared", config)
    dual = gold_v1.run_gold_dual_gpu(
        protocol_directory=prepared["protocol_directory"],
        scoring_bundle_directory=prepared["scoring_bundle_directory"],
        output_directory=scratch / "dual",
        expected_steps=int(config["protocol"]["expected_optimizer_steps_per_architecture"]),
        timeout_seconds=timeout_seconds,
    )
    attack, metrics, predictions = gold_v2.finalize_gold_results(
        dual_output_directory=scratch / "dual",
        label_directory=prepared["label_directory"],
        false_positive_rate=float(config["scoring"]["false_positive_rate"]),
    )
    sensitivity = v1.evaluate_sensitivity_gate(metrics, config)
    metrics = dict(metrics)
    metrics["positive_control_version"] = v1.SHADOW_GOLD_POSITIVE_CONTROL_VERSION
    metrics["positive_control_sensitivity_gate"] = sensitivity
    metrics["protocol_manifest_repair"] = (
        SHADOW_GOLD_POSITIVE_CONTROL_MANIFEST_REPAIR_VERSION
    )
    metrics["scientific_protocol_changed"] = False
    metrics["stage2_v3_selection_allowed"] = False
    metrics["external_model_holdout_consumed"] = False
    manifest = {
        "status": "complete",
        "experiment_id": v1.EXPERIMENT_ID,
        "execution_id": "shadow-gold-positive-control-v2",
        "version": v1.SHADOW_GOLD_POSITIVE_CONTROL_VERSION,
        "protocol_manifest_repair": SHADOW_GOLD_POSITIVE_CONTROL_MANIFEST_REPAIR_VERSION,
        "scientific_protocol_changed": False,
        "gpu_names": names,
        "prepared": prepared,
        "dual_gpu": dual,
        "selected_attack_secondary_diagnostic": attack["selected_candidate"],
        "primary_sensitivity_predictor": "loss",
        "sensitivity_gate_status": sensitivity["status"],
        "sensitivity_gate_passed": sensitivity["status"] == "pass",
        "scientific_result_valid_even_if_gate_fails": True,
        "evaluation_labels_passed_to_children": False,
        "competition_rows_used": 0,
        "external_rows_used": 0,
        "pretrained_weights_used": False,
        "automatic_compute_retries": 0,
        "stage2_v3_selection_allowed": False,
        "external_model_holdout_consumed": False,
        "fourth_external_holdout_consumed": False,
    }
    return attack, metrics, predictions, manifest


__all__ = [
    "NO_COMPUTE_GUARD",
    "SHADOW_GOLD_POSITIVE_CONTROL_MANIFEST_REPAIR_VERSION",
    "ShadowGoldPositiveControlManifestRepairError",
    "prepare_positive_control_runtime",
    "repair_protocol_manifest",
    "run_positive_control_benchmark",
]
