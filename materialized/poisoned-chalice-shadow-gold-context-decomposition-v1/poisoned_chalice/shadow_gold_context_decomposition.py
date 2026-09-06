"""Decompose the Gold training-context alignment effect.

The preceding context-alignment ablation recovered a large probe-boundary-free
membership signal, but one scorer change bundled several mechanisms: forcing the
known training region (suffix), changing 256 raw payload tokens to 254 payload
bytes, inserting the BOS boundary used during training, and removing Stage2's
best-of-prefix/middle/suffix window aggregation.

This module freezes a no-selection mechanistic decomposition. It repeats the
positive-control training unchanged and evaluates fixed mean-log-likelihood
conditions only. It never receives the known probe text/boundary or labels while
scoring. Results are mechanistic and cannot directly promote Stage2-v3.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import shadow_gold_context_alignment as context_v1
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

SHADOW_GOLD_CONTEXT_DECOMPOSITION_VERSION = "suffix-context-decomposition-v1"
EXECUTION_ID = "shadow-gold-context-decomposition-v1"
CONTEXT_ALIGNMENT_RESULT_COMMIT = "5efad09708be2ee3645726365aa02afda73c7179"
_CONDITIONS = [
    "raw_stage2_loss_max",
    "raw_suffix256_mean",
    "raw_suffix254_mean",
    "bos_suffix254_mean",
    "bos_prefix254_mean",
    "bos_middle254_mean",
]
_METADATA_COLUMNS = ["benchmark_id", "language", "length_bin", "character_count"]
_FEATURE_COLUMNS = [*_METADATA_COLUMNS, *_CONDITIONS]


class ShadowGoldContextDecompositionError(RuntimeError):
    """Raised when a frozen context-decomposition invariant fails."""


def validate_context_decomposition_config(config: Mapping[str, Any]) -> None:
    if config.get("execution_id") != EXECUTION_ID or config.get("role") != "gold_suffix_context_decomposition":
        raise ShadowGoldContextDecompositionError("context-decomposition identity changed")
    evidence = config.get("base_evidence") or {}
    if evidence.get("context_alignment_result_commit") != CONTEXT_ALIGNMENT_RESULT_COMMIT:
        raise ShadowGoldContextDecompositionError("context-decomposition evidence changed")
    if evidence.get("formal_context_alignment_gate_failed") is not True:
        raise ShadowGoldContextDecompositionError("context-decomposition formal-result guard changed")
    if evidence.get("mechanistic_context_alignment_effect_positive") is not True:
        raise ShadowGoldContextDecompositionError("context-decomposition mechanistic-result guard changed")
    if evidence.get("next_scalars_must_use_no_selection") is not True:
        raise ShadowGoldContextDecompositionError("context-decomposition no-selection guard changed")

    training = config.get("training_contract") or {}
    expected_training = {
        "candidate_rows": 10240,
        "training_corpus_rows": 5120,
        "evaluation_rows": 5120,
        "probe_version": positive_v1.PROBE_VERSION,
        "probe_applied_to_members": True,
        "probe_applied_to_nonmembers": True,
        "exposure_repeats": 16,
        "forced_training_window": "suffix",
        "training_sequences": 81920,
        "global_batch_size": 64,
        "optimizer_steps_per_architecture": 1280,
        "architecture_pair": ["shadow-gpt2-byte-5m", "shadow-llama-byte-5m"],
        "random_initialisation": True,
        "pretrained_weights_used": False,
        "scientific_protocol_changed_from_positive_control_v2": False,
    }
    for key, value in expected_training.items():
        if training.get(key) != value:
            raise ShadowGoldContextDecompositionError(f"context-decomposition training contract changed: {key}")

    scoring = config.get("scoring") or {}
    expected_scoring = {
        "version": SHADOW_GOLD_CONTEXT_DECOMPOSITION_VERSION,
        "tokenizer": "lossless_utf8_bytes",
        "max_sequence_tokens": 256,
        "mean_log_likelihood_only": True,
        "candidate_selection_used": False,
        "probe_boundary_used": False,
        "probe_text_used": False,
        "membership_used_for_scoring": False,
        "benchmark_id_used_as_model_input": False,
        "batch_size": 64,
    }
    for key, value in expected_scoring.items():
        if scoring.get(key) != value:
            raise ShadowGoldContextDecompositionError(f"context-decomposition scorer changed: {key}")
    if list(config.get("conditions") or []) != _CONDITIONS:
        raise ShadowGoldContextDecompositionError("context-decomposition conditions changed")

    definitions = scoring.get("condition_definitions") or {}
    expected_definitions = {
        "raw_stage2_loss_max": "stage2-model-independent-v1 score_logp_mean__max",
        "raw_suffix256_mean": "last_256_payload_tokens_no_special_tokens_mean_predicted_logp",
        "raw_suffix254_mean": "last_254_payload_tokens_no_special_tokens_mean_predicted_logp",
        "bos_suffix254_mean": "bos_plus_last_254_payload_plus_eos_score_payload_mean",
        "bos_prefix254_mean": "bos_plus_first_254_payload_plus_eos_score_payload_mean",
        "bos_middle254_mean": "bos_plus_middle_254_payload_plus_eos_score_payload_mean",
    }
    if definitions != expected_definitions:
        raise ShadowGoldContextDecompositionError("context-decomposition condition definitions changed")

    decision = config.get("mechanistic_decision") or {}
    expected_decision = {
        "aligned_min_auc_each_architecture": 0.56,
        "aligned_min_gain_vs_raw_stage2_each_architecture": 0.03,
        "aligned_min_tpr_at_1pct_fpr_each_architecture": 0.015,
        "location_explains_most_auc_tolerance": 0.01,
        "crop_explains_most_auc_tolerance": 0.01,
        "bos_context_material_min_auc_gain": 0.02,
        "suffix_specificity_min_auc_margin": 0.03,
        "both_architectures_required_for_named_mechanism": True,
        "criterion_is_mechanistic_only": True,
        "does_not_promote_stage2_v3": True,
    }
    for key, value in expected_decision.items():
        if decision.get(key) != value:
            raise ShadowGoldContextDecompositionError(f"context-decomposition decision changed: {key}")

    runtime = config.get("runtime") or {}
    if runtime.get("backend") != "cuda" or runtime.get("kaggle_machine_shape") != "NvidiaTeslaT4":
        raise ShadowGoldContextDecompositionError("context-decomposition CUDA resource changed")
    if runtime.get("expected_visible_gpu_count") != 2 or runtime.get("per_architecture_world_size") != 1:
        raise ShadowGoldContextDecompositionError("context-decomposition GPU/world-size changed")
    for key in ("distributed_training", "ddp_used", "nccl_used"):
        if runtime.get(key) is not False:
            raise ShadowGoldContextDecompositionError(f"context-decomposition distributed guard changed: {key}")
    if runtime.get("automatic_compute_retries") != 0 or runtime.get("notebook_internet") is not False:
        raise ShadowGoldContextDecompositionError("context-decomposition retry/internet contract changed")

    guards = config.get("scientific_guards") or {}
    for key in (
        "known_probe_boundary_used",
        "probe_text_used",
        "probe_local_score_used",
        "candidate_selection_used",
        "evaluation_labels_passed_to_training_children",
        "stage2_v3_selection_allowed",
        "competition_feature_promotion_allowed",
        "external_model_holdout_consumed",
        "fourth_external_holdout_consumed",
        "smollm2_labels_used",
        "public_leaderboard_feedback_used",
        "hidden_stage1_validation_labels_used",
    ):
        if guards.get(key) is not False:
            raise ShadowGoldContextDecompositionError(f"context-decomposition scientific guard changed: {key}")
    if guards.get("features_sealed_before_label_reveal") is not True:
        raise ShadowGoldContextDecompositionError("context-decomposition feature-seal guard changed")
    if guards.get("competition_rows_used") != 0 or guards.get("external_rows_used") != 0:
        raise ShadowGoldContextDecompositionError("context-decomposition external-row guard changed")


def _region(payload: list[int], name: str, capacity: int) -> list[int]:
    if len(payload) <= capacity:
        return list(payload)
    if name == "prefix":
        return list(payload[:capacity])
    if name == "suffix":
        return list(payload[-capacity:])
    if name == "middle":
        start = (len(payload) - capacity) // 2
        return list(payload[start : start + capacity])
    raise ShadowGoldContextDecompositionError(f"unknown region: {name}")


def build_context_decomposition_plan(
    content: str,
    condition: str,
    *,
    tokenizer: ByteCodeTokenizer | None = None,
) -> dict[str, Any]:
    """Build one fixed context condition without probe information."""

    runtime_tokenizer = tokenizer or ByteCodeTokenizer()
    payload = runtime_tokenizer.encode(str(content), add_special_tokens=False)
    if len(payload) < 2:
        raise ShadowGoldContextDecompositionError("context-decomposition sample has fewer than two payload tokens")
    if condition == "raw_suffix256_mean":
        selected = _region(payload, "suffix", 256)
        input_ids = list(selected)
        attention_mask = [1] * len(input_ids)
        score_positions = list(range(max(0, len(selected) - 1)))
        scored_payload_tokens = max(0, len(selected) - 1)
        wrapper = "raw_suffix_256_no_special_tokens"
    elif condition == "raw_suffix254_mean":
        selected = _region(payload, "suffix", 254)
        input_ids = list(selected)
        attention_mask = [1] * len(input_ids)
        score_positions = list(range(max(0, len(selected) - 1)))
        scored_payload_tokens = max(0, len(selected) - 1)
        wrapper = "raw_suffix_254_no_special_tokens"
    elif condition in {"bos_suffix254_mean", "bos_prefix254_mean", "bos_middle254_mean"}:
        region_name = condition.split("_")[1].replace("254", "")
        selected = _region(payload, region_name, 254)
        input_ids = [runtime_tokenizer.bos_token_id, *selected, runtime_tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        score_positions = list(range(len(selected)))
        scored_payload_tokens = len(selected)
        wrapper = f"bos_{region_name}_254_plus_eos"
    else:
        raise ShadowGoldContextDecompositionError(f"condition has no explicit plan: {condition}")
    if not score_positions:
        raise ShadowGoldContextDecompositionError("context-decomposition condition has no scored payload tokens")
    if len(input_ids) > 256:
        raise ShadowGoldContextDecompositionError("context-decomposition plan exceeds 256 tokens")
    padding = 256 - len(input_ids)
    input_ids += [runtime_tokenizer.pad_token_id] * padding
    attention_mask += [0] * padding
    return {
        "condition": condition,
        "wrapper": wrapper,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "score_logit_positions": score_positions,
        "selected_payload_tokens": len(selected),
        "scored_payload_tokens": scored_payload_tokens,
        "selected_payload_ids": selected,
    }


def score_context_decomposition_features(
    training_output_directory: str | Path,
    scoring_bundle_directory: str | Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score every fixed context condition before membership labels are opened."""

    import torch

    validate_context_decomposition_config(config)
    bundle = load_scoring_bundle(scoring_bundle_directory)
    checkpoint = load_shadow_checkpoint(training_output_directory, backend="cuda")
    if int(checkpoint.training_manifest.get("completed_steps", -1)) != 1280:
        raise ShadowGoldContextDecompositionError("context-decomposition checkpoint step count changed")
    if checkpoint.training_manifest.get("evaluation_labels_read") is not False:
        raise ShadowGoldContextDecompositionError("context-decomposition checkpoint label guard failed")
    if checkpoint.training_manifest.get("pretrained_weights_used") is not False:
        raise ShadowGoldContextDecompositionError("context-decomposition pretrained-weight guard failed")

    source = bundle.frame.reset_index(drop=True)
    tokenizer = ByteCodeTokenizer()
    model = checkpoint.model
    model.eval()
    device = next(model.parameters()).device
    batch_size = int(config["scoring"]["batch_size"])

    features = source[_METADATA_COLUMNS].copy()
    features["raw_stage2_loss_max"] = context_v1._raw_stage2_loss_max(model, source)

    explicit_conditions = [name for name in _CONDITIONS if name != "raw_stage2_loss_max"]
    for condition in explicit_conditions:
        plans = [build_context_decomposition_plan(content, condition, tokenizer=tokenizer) for content in source.content.astype(str)]
        values_out = np.full(len(source), np.nan, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(plans), batch_size):
                stop = min(start + batch_size, len(plans))
                batch = plans[start:stop]
                input_ids = torch.as_tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
                attention_mask = torch.as_tensor([row["attention_mask"] for row in batch], dtype=torch.long, device=device)
                output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = output.logits if hasattr(output, "logits") else output[0]
                shifted_logits = logits[:, :-1].float()
                targets = input_ids[:, 1:]
                correct = torch.log_softmax(shifted_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                for local_index, plan in enumerate(batch):
                    positions = torch.as_tensor(plan["score_logit_positions"], dtype=torch.long, device=device)
                    row_values = correct[local_index].index_select(0, positions)
                    values_out[start + local_index] = float(row_values.mean().detach().cpu())
                del input_ids, attention_mask, output, logits, shifted_logits, targets, correct
        if not np.isfinite(values_out).all():
            raise ShadowGoldContextDecompositionError(f"context-decomposition condition is non-finite: {condition}")
        features[condition] = values_out

    features = features[_FEATURE_COLUMNS]
    if features.benchmark_id.duplicated().any():
        raise ShadowGoldContextDecompositionError("context-decomposition feature IDs are duplicated")
    manifest = {
        "status": "sealed",
        "version": SHADOW_GOLD_CONTEXT_DECOMPOSITION_VERSION,
        "architecture_slot": checkpoint.training_manifest.get("architecture_slot"),
        "rows": len(features),
        "conditions": list(_CONDITIONS),
        "mean_log_likelihood_only": True,
        "candidate_selection_used": False,
        "evaluation_input_sha256": bundle.input_sha256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "model_state_sha256": checkpoint.model_state_sha256,
        "membership_labels_read": False,
        "probe_boundary_used": False,
        "probe_text_used": False,
        "benchmark_ids_passed_to_model": False,
        "stage2_v3_selection_allowed": False,
    }
    del checkpoint, model
    torch.cuda.empty_cache()
    return features, manifest


def _load_labels(label_directory: str | Path) -> pd.DataFrame:
    root = Path(label_directory).resolve()
    path = root / "evaluation_labels.jsonl"
    manifest_path = root / "label_manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise ShadowGoldContextDecompositionError("context-decomposition parent label seal is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("passed_to_gpu_children") is not False:
        raise ShadowGoldContextDecompositionError("context-decomposition label boundary failed")
    if manifest.get("label_sha256") != gold_v1._sha256_file(path):
        raise ShadowGoldContextDecompositionError("context-decomposition parent label hash mismatch")
    labels = pd.read_json(path, lines=True)
    return gold_v2.canonicalize_exact_schema(labels, gold_v1._LABEL_COLUMNS, artifact_name="Gold parent label")


def evaluate_context_decomposition(
    left_features: pd.DataFrame,
    right_features: pd.DataFrame,
    label_directory: str | Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate fixed conditions on the frozen paired holdout; no candidate selection."""

    validate_context_decomposition_config(config)
    if list(left_features.columns) != _FEATURE_COLUMNS or list(right_features.columns) != _FEATURE_COLUMNS:
        raise ShadowGoldContextDecompositionError("context-decomposition feature schema changed")
    if left_features.benchmark_id.astype(str).tolist() != right_features.benchmark_id.astype(str).tolist():
        raise ShadowGoldContextDecompositionError("context-decomposition left/right row order differs")
    labels = _load_labels(label_directory)
    left_labeled = attach_exact_labels(left_features, labels)
    right_labeled = attach_exact_labels(right_features, labels)
    _, left_holdout = split_development_holdout(left_labeled, seed=2027)
    _, right_holdout = split_development_holdout(right_labeled, seed=2027)
    holdout_ids = left_holdout.benchmark_id.astype(str).tolist()
    right_holdout = right_holdout.set_index(right_holdout.benchmark_id.astype(str), drop=False).loc[holdout_ids].reset_index(drop=True)
    if right_holdout.benchmark_id.astype(str).tolist() != holdout_ids:
        raise ShadowGoldContextDecompositionError("context-decomposition paired holdout order changed")

    condition_metrics: dict[str, dict[str, Any]] = {"left": {}, "right": {}}
    for condition in _CONDITIONS:
        condition_metrics["left"][condition] = evaluate_frozen_gold_holdout(
            left_holdout, left_holdout[condition].to_numpy(dtype=np.float64), false_positive_rate=0.01
        )["overall"]
        condition_metrics["right"][condition] = evaluate_frozen_gold_holdout(
            right_holdout, right_holdout[condition].to_numpy(dtype=np.float64), false_positive_rate=0.01
        )["overall"]

    decision = config["mechanistic_decision"]
    checks: dict[str, dict[str, bool]] = {}
    mechanism_per_side: dict[str, str] = {}
    for side in ("left", "right"):
        m = condition_metrics[side]
        aligned = m["bos_suffix254_mean"]
        raw = m["raw_stage2_loss_max"]
        raw256 = m["raw_suffix256_mean"]
        raw254 = m["raw_suffix254_mean"]
        prefix = m["bos_prefix254_mean"]
        middle = m["bos_middle254_mean"]
        side_checks = {
            "aligned_min_auc": aligned["auc"] >= float(decision["aligned_min_auc_each_architecture"]),
            "aligned_min_gain": aligned["auc"] - raw["auc"] >= float(decision["aligned_min_gain_vs_raw_stage2_each_architecture"]),
            "aligned_min_tpr": aligned["tpr_at_fpr"] >= float(decision["aligned_min_tpr_at_1pct_fpr_each_architecture"]),
            "location_explains_most": abs(aligned["auc"] - raw256["auc"]) <= float(decision["location_explains_most_auc_tolerance"]),
            "crop_explains_most": abs(aligned["auc"] - raw254["auc"]) <= float(decision["crop_explains_most_auc_tolerance"]),
            "bos_context_material": aligned["auc"] - raw254["auc"] >= float(decision["bos_context_material_min_auc_gain"]),
            "suffix_specificity": aligned["auc"] - max(prefix["auc"], middle["auc"]) >= float(decision["suffix_specificity_min_auc_margin"]),
        }
        checks[side] = side_checks
        if side_checks["location_explains_most"]:
            mechanism = "suffix_region_selection_dominant"
        elif side_checks["crop_explains_most"]:
            mechanism = "suffix_region_plus_254_crop_dominant"
        elif side_checks["bos_context_material"]:
            mechanism = "bos_start_context_material"
        else:
            mechanism = "mixed_or_unresolved"
        mechanism_per_side[side] = mechanism

    aligned_recovery = all(
        checks[side][name]
        for side in ("left", "right")
        for name in ("aligned_min_auc", "aligned_min_gain", "aligned_min_tpr")
    )
    named_mechanism = (
        mechanism_per_side["left"]
        if mechanism_per_side["left"] == mechanism_per_side["right"]
        else "architecture_dependent"
    )
    result = {
        "status": "complete",
        "version": SHADOW_GOLD_CONTEXT_DECOMPOSITION_VERSION,
        "condition_metrics": condition_metrics,
        "mechanistic_checks": checks,
        "aligned_recovery": aligned_recovery,
        "mechanism_per_architecture": mechanism_per_side,
        "named_mechanism": named_mechanism,
        "candidate_selection_used": False,
        "mean_log_likelihood_only": True,
        "probe_boundary_used": False,
        "probe_text_used": False,
        "stage2_v3_selection_allowed": False,
        "competition_feature_promotion_allowed": False,
    }
    predictions = left_holdout[
        ["benchmark_id", "language", "length_bin", "character_count", "membership", "matched_pair_id"]
    ].copy()
    for condition in _CONDITIONS:
        predictions[f"left_{condition}"] = left_holdout[condition].to_numpy(dtype=np.float64)
        predictions[f"right_{condition}"] = right_holdout[condition].to_numpy(dtype=np.float64)
    return result, predictions


def run_context_decomposition_benchmark(
    *,
    scratch_directory: str | Path,
    positive_control_config: Mapping[str, Any],
    context_decomposition_config: Mapping[str, Any],
    timeout_seconds: int = 5400,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Repeat frozen positive-control training and evaluate the fixed decomposition."""

    positive_v1.validate_positive_control_config(positive_control_config)
    validate_context_decomposition_config(context_decomposition_config)
    names = visible_cuda_device_names()
    validate_gpu_inventory(names, expected_count=2, required_name_fragment="T4")
    scratch = Path(scratch_directory).resolve()
    if scratch.exists() and any(scratch.iterdir()):
        raise ShadowGoldContextDecompositionError("context-decomposition scratch directory is not empty")
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
    left_features, left_manifest = score_context_decomposition_features(
        scratch / "training" / "left", prepared["scoring_bundle_directory"], context_decomposition_config
    )
    right_features, right_manifest = score_context_decomposition_features(
        scratch / "training" / "right", prepared["scoring_bundle_directory"], context_decomposition_config
    )
    seal_dir = scratch / "context_decomposition_feature_seal"
    seal_dir.mkdir()
    left_path = seal_dir / "left_context_decomposition_features.jsonl"
    right_path = seal_dir / "right_context_decomposition_features.jsonl"
    gold_v1._write_jsonl(left_features, left_path, list(left_features.columns))
    gold_v1._write_jsonl(right_features, right_path, list(right_features.columns))
    seal = {
        "status": "sealed",
        "left_sha256": gold_v1._sha256_file(left_path),
        "right_sha256": gold_v1._sha256_file(right_path),
        "left_manifest": left_manifest,
        "right_manifest": right_manifest,
        "candidate_selection_used": False,
        "mean_log_likelihood_only": True,
        "membership_labels_read": False,
        "probe_boundary_used": False,
        "probe_text_used": False,
        "stage2_v3_selection_allowed": False,
    }
    gold_v1._write_json(seal_dir / "context_decomposition_feature_seal.json", seal)
    evaluation, predictions = evaluate_context_decomposition(
        left_features, right_features, prepared["label_directory"], context_decomposition_config
    )
    manifest = {
        "status": "complete",
        "execution_id": EXECUTION_ID,
        "version": SHADOW_GOLD_CONTEXT_DECOMPOSITION_VERSION,
        "context_alignment_result_commit": CONTEXT_ALIGNMENT_RESULT_COMMIT,
        "gpu_names": names,
        "prepared": prepared,
        "dual_gpu_training": dual,
        "context_decomposition_feature_seal": seal,
        "named_mechanism": evaluation["named_mechanism"],
        "aligned_recovery": evaluation["aligned_recovery"],
        "training_protocol_changed_from_positive_control_v2": False,
        "features_sealed_before_label_reveal": True,
        "candidate_selection_used": False,
        "probe_boundary_used": False,
        "probe_text_used": False,
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
    "CONTEXT_ALIGNMENT_RESULT_COMMIT",
    "EXECUTION_ID",
    "SHADOW_GOLD_CONTEXT_DECOMPOSITION_VERSION",
    "ShadowGoldContextDecompositionError",
    "build_context_decomposition_plan",
    "evaluate_context_decomposition",
    "run_context_decomposition_benchmark",
    "score_context_decomposition_features",
    "validate_context_decomposition_config",
]
