"""Probe-boundary-free aggregation ablation for the controlled Gold benchmark.

This module is downstream of the probe-local calibration.  It deliberately does
not use the known canary text or boundary.  Instead it repeats the frozen
positive-control training regime and evaluates a small predeclared matrix of
features that the generic Stage2 scorer can compute for arbitrary code.

Gold labels may rank these candidates only as controlled mechanistic evidence.
Nothing selected here is promoted to Stage2-v3 or a competition feature without
separate external-model validation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import shadow_gold_kaggle as gold_v1
from . import shadow_gold_kaggle_v2 as gold_v2
from . import shadow_gold_positive_control as positive_v1
from . import shadow_gold_positive_control_v2 as positive_v2
from .shadow_gold_transfer import (
    attach_exact_labels,
    evaluate_frozen_gold_holdout,
    split_development_holdout,
)
from .shadow_protocol import ByteCodeTokenizer
from .shadow_scoring import load_scoring_bundle, load_shadow_checkpoint
from .shadow_training_dual_gpu import (
    run_dual_gpu_shadow,
    validate_gpu_inventory,
    visible_cuda_device_names,
)
from .stage2_api import STAGE2_METHOD_VERSION, Stage2RuntimeConfig, score_samples_detailed


SHADOW_GOLD_GENERIC_AGGREGATION_VERSION = "shadow-gold-generic-aggregation-v1"
EXECUTION_ID = "shadow-gold-generic-aggregation-v1"
_METADATA_COLUMNS = ["benchmark_id", "language", "length_bin", "character_count"]


class ShadowGoldGenericAggregationError(RuntimeError):
    """Raised when a frozen generic-aggregation invariant fails."""


def _candidate_contract(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = config.get("candidate_features")
    if not isinstance(raw, list) or not raw:
        raise ShadowGoldGenericAggregationError("generic aggregation candidate list is empty")
    candidates: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_columns: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ShadowGoldGenericAggregationError("generic aggregation candidate is not a mapping")
        name = str(item.get("name") or "")
        column = str(item.get("source_column") or "")
        try:
            orientation = float(item.get("orientation"))
        except (TypeError, ValueError) as error:
            raise ShadowGoldGenericAggregationError("generic aggregation orientation is invalid") from error
        if not name or not column or orientation not in {-1.0, 1.0}:
            raise ShadowGoldGenericAggregationError("generic aggregation candidate identity changed")
        if name in seen_names or column in seen_columns:
            raise ShadowGoldGenericAggregationError("generic aggregation candidate names/columns must be unique")
        if "probe" in name.casefold() or "probe" in column.casefold():
            raise ShadowGoldGenericAggregationError("probe-specific candidate leaked into generic aggregation")
        seen_names.add(name)
        seen_columns.add(column)
        candidates.append({"name": name, "source_column": column, "orientation": orientation})
    return candidates


def validate_generic_aggregation_config(config: Mapping[str, Any]) -> None:
    if config.get("execution_id") != EXECUTION_ID or config.get("role") != "gold_probe_boundary_free_aggregation_ablation":
        raise ShadowGoldGenericAggregationError("generic aggregation identity changed")
    base = config.get("base_evidence") or {}
    if base.get("probe_local_result_commit") != "844d83992ff364ee7bc43ae3560fb06540e2bd99":
        raise ShadowGoldGenericAggregationError("generic aggregation base evidence changed")
    if base.get("training_protocol_must_be_identical") is not True:
        raise ShadowGoldGenericAggregationError("generic aggregation training identity guard changed")
    training = config.get("training_contract") or {}
    expected_training = {
        "candidate_rows": 10240,
        "training_corpus_rows": 5120,
        "evaluation_rows": 5120,
        "evaluation_rows_per_language_per_class": 512,
        "probe_version": positive_v1.PROBE_VERSION,
        "probe_applied_to_members": True,
        "probe_applied_to_nonmembers": True,
        "membership_used_to_construct_probe": False,
        "benchmark_id_used_to_construct_probe": False,
        "exposure_repeats": 16,
        "forced_training_window": "suffix",
        "training_sequences": 81920,
        "global_batch_size": 64,
        "optimizer_steps_per_architecture": 1280,
        "architecture_pair": ["shadow-gpt2-byte-5m", "shadow-llama-byte-5m"],
        "learning_rate": 0.0003,
        "weight_decay": 0.1,
        "warmup_fraction": 0.05,
        "gradient_clip_norm": 1.0,
        "random_initialisation": True,
        "pretrained_weights_used": False,
    }
    for key, value in expected_training.items():
        if training.get(key) != value:
            raise ShadowGoldGenericAggregationError(f"generic aggregation training contract changed: {key}")
    scoring = config.get("generic_scoring") or {}
    expected_scoring = {
        "stage2_method_version": STAGE2_METHOD_VERSION,
        "max_length": 256,
        "min_k_percents": [1, 2, 5, 10],
        "local_widths": [8, 16, 32, 64],
        "max_batch_tokens": 4096,
        "vocab_chunk_tokens": 64,
        "rank_vocab_block_size": 256,
        "fidelity_tokens": 24,
        "fidelity_atol": 1e-5,
        "probe_boundary_used": False,
        "probe_text_used_to_select_tokens": False,
        "benchmark_id_used_as_model_input": False,
        "membership_used_for_scoring": False,
    }
    for key, value in expected_scoring.items():
        if scoring.get(key) != value:
            raise ShadowGoldGenericAggregationError(f"generic aggregation scorer changed: {key}")
    candidates = _candidate_contract(config)
    expected_candidates = [
        ("loss_mean", "score_logp_mean__mean", 1.0),
        ("loss_max", "score_logp_mean__max", 1.0),
        ("loss_std_neg", "score_logp_mean__std", -1.0),
        ("min_k_01_mean", "min_k_01__mean", 1.0),
        ("min_k_02_mean", "min_k_02__mean", 1.0),
        ("min_k_05_mean", "min_k_05__mean", 1.0),
        ("min_k_10_mean", "min_k_10__mean", 1.0),
        ("min_kpp_01_mean", "min_kpp_zselect_01__mean", 1.0),
        ("min_kpp_02_mean", "min_kpp_zselect_02__mean", 1.0),
        ("min_kpp_05_mean", "min_kpp_zselect_05__mean", 1.0),
        ("min_kpp_10_mean", "min_kpp_zselect_10__mean", 1.0),
        ("min_kpp_05_max", "min_kpp_zselect_05__max", 1.0),
        ("local_08_max", "best_local_8__max", 1.0),
        ("local_16_max", "best_local_16__max", 1.0),
        ("local_32_max", "best_local_32__max", 1.0),
        ("local_64_max", "best_local_64__max", 1.0),
        ("correct_z_mean", "correct_z_mean__mean", 1.0),
        ("log_rank_mean", "mean_log_rank__mean", -1.0),
    ]
    observed = [(item["name"], item["source_column"], item["orientation"]) for item in candidates]
    if observed != expected_candidates:
        raise ShadowGoldGenericAggregationError("generic aggregation candidate matrix changed")
    selection = config.get("selection_protocol") or {}
    expected_selection = {
        "fit_architecture_slot": "left",
        "blind_transfer_architecture_slot": "right",
        "matched_pair_split_seed": 2027,
        "development_fraction": 0.5,
        "selected_candidate_applied_unchanged_to_left_holdout": True,
        "selected_candidate_applied_unchanged_to_right_holdout": True,
        "no_logistic_fit": True,
    }
    for key, value in expected_selection.items():
        if selection.get(key) != value:
            raise ShadowGoldGenericAggregationError(f"generic aggregation selection changed: {key}")
    recovery = config.get("recovery_criterion") or {}
    if recovery.get("baseline_candidate") != "loss_max" or float(recovery.get("minimum_auc_gain_vs_baseline_each_architecture", -1)) != 0.03:
        raise ShadowGoldGenericAggregationError("generic aggregation recovery criterion changed")
    if recovery.get("both_architectures_required") is not True or recovery.get("does_not_promote_stage2_v3") is not True:
        raise ShadowGoldGenericAggregationError("generic aggregation recovery guard changed")
    runtime = config.get("runtime") or {}
    if runtime.get("backend") != "cuda" or runtime.get("kaggle_machine_shape") != "NvidiaTeslaT4":
        raise ShadowGoldGenericAggregationError("generic aggregation CUDA resource changed")
    if runtime.get("expected_visible_gpu_count") != 2 or runtime.get("per_architecture_world_size") != 1:
        raise ShadowGoldGenericAggregationError("generic aggregation GPU/world size changed")
    for key in ("distributed_training", "ddp_used", "nccl_used"):
        if runtime.get(key) is not False:
            raise ShadowGoldGenericAggregationError(f"generic aggregation distributed guard changed: {key}")
    if runtime.get("automatic_compute_retries") != 0 or runtime.get("notebook_internet") is not False:
        raise ShadowGoldGenericAggregationError("generic aggregation retry/internet contract changed")
    guards = config.get("scientific_guards") or {}
    false_keys = (
        "probe_local_score_used_as_candidate",
        "known_probe_boundary_used",
        "stage2_v3_selection_allowed",
        "competition_feature_promotion_allowed",
        "external_model_holdout_consumed",
        "fourth_external_holdout_consumed",
        "smollm2_labels_used",
        "public_leaderboard_feedback_used",
        "hidden_stage1_validation_labels_used",
    )
    for key in false_keys:
        if guards.get(key) is not False:
            raise ShadowGoldGenericAggregationError(f"generic aggregation scientific guard changed: {key}")
    if guards.get("generic_stage2_api_only") is not True:
        raise ShadowGoldGenericAggregationError("generic Stage2 API guard changed")
    if guards.get("competition_rows_used") != 0 or guards.get("external_rows_used") != 0:
        raise ShadowGoldGenericAggregationError("generic aggregation external rows changed")


def score_generic_candidate_features(
    training_output_directory: str | Path,
    scoring_bundle_directory: str | Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score the predeclared generic feature matrix without reading labels."""

    import torch

    validate_generic_aggregation_config(config)
    scoring = config["generic_scoring"]
    bundle = load_scoring_bundle(scoring_bundle_directory)
    checkpoint = load_shadow_checkpoint(training_output_directory, backend="cuda")
    if int(checkpoint.training_manifest.get("completed_steps", -1)) != 1280:
        raise ShadowGoldGenericAggregationError("generic aggregation checkpoint step count changed")
    stage2 = Stage2RuntimeConfig(
        max_length=int(scoring["max_length"]),
        max_batch_tokens=int(scoring["max_batch_tokens"]),
        vocab_chunk_tokens=int(scoring["vocab_chunk_tokens"]),
        rank_vocab_block_size=int(scoring["rank_vocab_block_size"]),
        min_k_percents=tuple(int(value) for value in scoring["min_k_percents"]),
        local_widths=tuple(int(value) for value in scoring["local_widths"]),
        language_calibration_min_rows=20,
        length_calibration_min_rows=12,
        fidelity_gate=True,
        fidelity_tokens=int(scoring["fidelity_tokens"]),
        fidelity_atol=float(scoring["fidelity_atol"]),
        device="auto",
        move_model=False,
    )
    source = bundle.frame.reset_index(drop=True)
    detailed = score_samples_detailed(
        model=checkpoint.model,
        tokenizer=ByteCodeTokenizer(),
        samples=source.content.astype(str).tolist(),
        languages=source.language.astype(str).tolist(),
        runtime_config=stage2,
    )
    matrix = detailed.features.sort_values("sample_index").reset_index(drop=True)
    if matrix.sample_index.tolist() != list(range(len(source))):
        raise ShadowGoldGenericAggregationError("generic aggregation Stage2 row order changed")
    output = source[_METADATA_COLUMNS].copy()
    contract = _candidate_contract(config)
    missing = [item["source_column"] for item in contract if item["source_column"] not in matrix.columns]
    if missing:
        raise ShadowGoldGenericAggregationError(f"generic aggregation source features missing: {missing}")
    for item in contract:
        values = matrix[item["source_column"]].to_numpy(dtype=np.float64) * float(item["orientation"])
        if not np.isfinite(values).all():
            raise ShadowGoldGenericAggregationError(f"generic aggregation candidate is non-finite: {item['name']}")
        output[item["name"]] = values
    if output.benchmark_id.duplicated().any():
        raise ShadowGoldGenericAggregationError("generic aggregation feature IDs are duplicated")
    manifest = {
        "status": "sealed",
        "version": SHADOW_GOLD_GENERIC_AGGREGATION_VERSION,
        "architecture_slot": checkpoint.training_manifest.get("architecture_slot"),
        "rows": len(output),
        "candidate_contract": contract,
        "stage2_method_version": detailed.manifest.get("method_version"),
        "stage2_manifest": detailed.manifest,
        "evaluation_input_sha256": bundle.input_sha256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "model_state_sha256": checkpoint.model_state_sha256,
        "probe_boundary_used": False,
        "probe_text_used_to_select_tokens": False,
        "membership_labels_read": False,
        "benchmark_ids_passed_to_model": False,
        "stage2_v3_selection_allowed": False,
    }
    del checkpoint, detailed, matrix
    torch.cuda.empty_cache()
    return output, manifest


def _load_labels(label_directory: str | Path) -> pd.DataFrame:
    root = Path(label_directory).resolve()
    path = root / "evaluation_labels.jsonl"
    manifest_path = root / "label_manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise ShadowGoldGenericAggregationError("generic aggregation parent label seal is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("passed_to_gpu_children") is not False:
        raise ShadowGoldGenericAggregationError("generic aggregation label boundary failed")
    if manifest.get("label_sha256") != gold_v1._sha256_file(path):
        raise ShadowGoldGenericAggregationError("generic aggregation parent label hash mismatch")
    labels = pd.read_json(path, lines=True)
    return gold_v2.canonicalize_exact_schema(labels, gold_v1._LABEL_COLUMNS, artifact_name="Gold parent label")


def _select_candidate(
    development: pd.DataFrame,
    candidate_names: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    metrics: dict[str, Any] = {}
    for name in candidate_names:
        metrics[name] = evaluate_frozen_gold_holdout(
            development,
            development[name].to_numpy(dtype=np.float64),
            false_positive_rate=0.01,
        )["overall"]
    order = {name: index for index, name in enumerate(candidate_names)}
    selected = max(
        candidate_names,
        key=lambda name: (
            metrics[name]["partial_auc_standardized"],
            metrics[name]["tpr_at_fpr"],
            metrics[name]["auc"],
            -order[name],
        ),
    )
    return selected, metrics


def evaluate_generic_aggregation(
    left_features: pd.DataFrame,
    right_features: pd.DataFrame,
    label_directory: str | Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select on Shadow-A development pairs and transfer unchanged to both holdouts."""

    validate_generic_aggregation_config(config)
    contract = _candidate_contract(config)
    names = [item["name"] for item in contract]
    expected_columns = [*_METADATA_COLUMNS, *names]
    if list(left_features.columns) != expected_columns or list(right_features.columns) != expected_columns:
        raise ShadowGoldGenericAggregationError("generic aggregation feature schema changed")
    if left_features.benchmark_id.astype(str).tolist() != right_features.benchmark_id.astype(str).tolist():
        raise ShadowGoldGenericAggregationError("generic aggregation left/right row order differs")
    labels = _load_labels(label_directory)
    left_labeled = attach_exact_labels(left_features, labels)
    right_labeled = attach_exact_labels(right_features, labels)
    left_development, left_holdout = split_development_holdout(left_labeled, seed=2027)
    _, right_holdout = split_development_holdout(right_labeled, seed=2027)
    holdout_ids = left_holdout.benchmark_id.astype(str).tolist()
    right_holdout = right_holdout.set_index(right_holdout.benchmark_id.astype(str), drop=False).loc[holdout_ids].reset_index(drop=True)
    if right_holdout.benchmark_id.astype(str).tolist() != holdout_ids:
        raise ShadowGoldGenericAggregationError("generic aggregation paired holdout order changed")
    selected, development_metrics = _select_candidate(left_development, names)
    all_holdout: dict[str, Any] = {"left": {}, "right": {}}
    for name in names:
        all_holdout["left"][name] = evaluate_frozen_gold_holdout(
            left_holdout,
            left_holdout[name].to_numpy(dtype=np.float64),
            false_positive_rate=0.01,
        )["overall"]
        all_holdout["right"][name] = evaluate_frozen_gold_holdout(
            right_holdout,
            right_holdout[name].to_numpy(dtype=np.float64),
            false_positive_rate=0.01,
        )["overall"]
    selected_left = all_holdout["left"][selected]
    selected_right = all_holdout["right"][selected]
    baseline_left = all_holdout["left"]["loss_max"]
    baseline_right = all_holdout["right"]["loss_max"]
    left_gain = float(selected_left["auc"] - baseline_left["auc"])
    right_gain = float(selected_right["auc"] - baseline_right["auc"])
    minimum_gain = float(config["recovery_criterion"]["minimum_auc_gain_vs_baseline_each_architecture"])
    recovery_passed = left_gain >= minimum_gain and right_gain >= minimum_gain
    result = {
        "status": "complete",
        "version": SHADOW_GOLD_GENERIC_AGGREGATION_VERSION,
        "selected_candidate": selected,
        "development_candidate_metrics": development_metrics,
        "holdout_candidate_metrics": all_holdout,
        "selected_holdout": {"left": selected_left, "right": selected_right},
        "baseline_loss_max_holdout": {"left": baseline_left, "right": baseline_right},
        "selected_auc_gain_vs_loss_max": {"left": left_gain, "right": right_gain},
        "recovery_criterion": {
            "minimum_auc_gain_each_architecture": minimum_gain,
            "passed": recovery_passed,
            "both_architectures_required": True,
        },
        "selected_on_shadow_a_development_only": True,
        "applied_unchanged_to_shadow_b": True,
        "probe_boundary_used": False,
        "probe_local_score_used": False,
        "stage2_v3_selection_allowed": False,
        "competition_feature_promotion_allowed": False,
    }
    predictions = left_holdout[["benchmark_id", "language", "length_bin", "character_count", "membership", "matched_pair_id"]].copy()
    predictions["selected_candidate"] = selected
    predictions["left_selected_score"] = left_holdout[selected].to_numpy(dtype=np.float64)
    predictions["right_selected_score"] = right_holdout[selected].to_numpy(dtype=np.float64)
    predictions["left_loss_max"] = left_holdout.loss_max.to_numpy(dtype=np.float64)
    predictions["right_loss_max"] = right_holdout.loss_max.to_numpy(dtype=np.float64)
    return result, predictions


def run_generic_aggregation_benchmark(
    *,
    scratch_directory: str | Path,
    positive_control_config: Mapping[str, Any],
    aggregation_config: Mapping[str, Any],
    timeout_seconds: int = 5400,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Repeat frozen training, seal generic candidates, then reveal Gold labels."""

    positive_v1.validate_positive_control_config(positive_control_config)
    validate_generic_aggregation_config(aggregation_config)
    names = visible_cuda_device_names()
    validate_gpu_inventory(names, expected_count=2, required_name_fragment="T4")
    scratch = Path(scratch_directory).resolve()
    if scratch.exists() and any(scratch.iterdir()):
        raise ShadowGoldGenericAggregationError("generic aggregation scratch directory is not empty")
    scratch.mkdir(parents=True, exist_ok=True)
    prepared = positive_v2.prepare_positive_control_runtime(scratch / "prepared", positive_control_config)
    dual = run_dual_gpu_shadow(
        prepared["protocol_directory"],
        scratch / "training",
        max_steps=1280,
        checkpoint_reload_atol=1e-5,
        parameter_sync_atol=1e-5,
        expected_gpu_count=2,
        required_gpu_name_fragment="T4",
        timeout_seconds=timeout_seconds,
    )
    left_features, left_manifest = score_generic_candidate_features(
        scratch / "training" / "left",
        prepared["scoring_bundle_directory"],
        aggregation_config,
    )
    right_features, right_manifest = score_generic_candidate_features(
        scratch / "training" / "right",
        prepared["scoring_bundle_directory"],
        aggregation_config,
    )
    seal_dir = scratch / "generic_feature_seal"
    seal_dir.mkdir()
    left_path = seal_dir / "left_generic_features.jsonl"
    right_path = seal_dir / "right_generic_features.jsonl"
    feature_columns = list(left_features.columns)
    gold_v1._write_jsonl(left_features, left_path, feature_columns)
    gold_v1._write_jsonl(right_features, right_path, feature_columns)
    seal = {
        "status": "sealed",
        "left_sha256": gold_v1._sha256_file(left_path),
        "right_sha256": gold_v1._sha256_file(right_path),
        "left_manifest": left_manifest,
        "right_manifest": right_manifest,
        "membership_labels_read": False,
        "probe_boundary_used": False,
        "stage2_v3_selection_allowed": False,
    }
    gold_v1._write_json(seal_dir / "generic_feature_seal.json", seal)
    evaluation, predictions = evaluate_generic_aggregation(
        left_features,
        right_features,
        prepared["label_directory"],
        aggregation_config,
    )
    manifest = {
        "status": "complete",
        "execution_id": EXECUTION_ID,
        "version": SHADOW_GOLD_GENERIC_AGGREGATION_VERSION,
        "gpu_names": names,
        "prepared": prepared,
        "dual_gpu_training": dual,
        "generic_feature_seal": seal,
        "selected_candidate": evaluation["selected_candidate"],
        "recovery_criterion_passed": evaluation["recovery_criterion"]["passed"],
        "training_protocol_changed_from_positive_control_v2": False,
        "probe_boundary_used": False,
        "probe_local_score_used": False,
        "features_sealed_before_label_reveal": True,
        "competition_rows_used": 0,
        "external_rows_used": 0,
        "pretrained_weights_used": False,
        "automatic_compute_retries": 0,
        "stage2_v3_selection_allowed": False,
        "competition_feature_promotion_allowed": False,
        "external_model_holdout_consumed": False,
        "fourth_external_holdout_consumed": False,
    }
    return evaluation, predictions, manifest


__all__ = [
    "EXECUTION_ID",
    "SHADOW_GOLD_GENERIC_AGGREGATION_VERSION",
    "ShadowGoldGenericAggregationError",
    "evaluate_generic_aggregation",
    "run_generic_aggregation_benchmark",
    "score_generic_candidate_features",
    "validate_generic_aggregation_config",
]
