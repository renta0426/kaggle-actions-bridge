"""Probe-local sensitivity calibration for the controlled Gold benchmark.

This module does not define a Stage2 attack.  It reuses the frozen positive-
control training protocol unchanged and asks only whether the model assigns
higher autoregressive likelihood to the deliberately repeated, content-derived
high-entropy probe itself.  Probe boundaries are derived from source content and
language before labels are revealed.  Membership labels are joined only after
both architectures have sealed content-free probe-local features.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import shadow_gold_kaggle as gold_v1
from . import shadow_gold_kaggle_v2 as gold_v2
from . import shadow_gold_positive_control as positive_v1
from . import shadow_gold_positive_control_v2 as positive_v2
from .shadow_gold_transfer import attach_exact_labels, evaluate_frozen_gold_holdout, split_development_holdout
from .shadow_protocol import ByteCodeTokenizer
from .shadow_scoring import load_scoring_bundle, load_shadow_checkpoint
from .shadow_training_dual_gpu import validate_gpu_inventory, visible_cuda_device_names


SHADOW_GOLD_PROBE_LOCAL_VERSION = "gold-probe-local-byte-logp-v1"
EXECUTION_ID = "shadow-gold-probe-local-v1"
_FEATURE_COLUMNS = [
    "benchmark_id",
    "language",
    "length_bin",
    "character_count",
    "probe_digest_logp_mean",
    "probe_full_logp_mean",
]


class ShadowGoldProbeLocalError(RuntimeError):
    """Raised when a frozen probe-local calibration invariant fails."""


def validate_probe_local_config(config: Mapping[str, Any]) -> None:
    if config.get("execution_id") != EXECUTION_ID or config.get("role") != "gold_positive_control_probe_local_calibration":
        raise ShadowGoldProbeLocalError("probe-local identity changed")
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
            raise ShadowGoldProbeLocalError(f"probe-local training contract changed: {key}")
    scoring = config.get("probe_local_scoring") or {}
    expected_scoring = {
        "version": SHADOW_GOLD_PROBE_LOCAL_VERSION,
        "tokenizer": "lossless_utf8_bytes",
        "training_window_capacity_payload_tokens": 254,
        "primary_predictor": "probe_digest_logp_mean",
        "secondary_predictor": "probe_full_logp_mean",
        "digest_hex_characters": 16,
        "digest_token_count": 16,
        "full_probe_must_be_suffix": True,
        "probe_boundary_derived_from_content_and_language_only": True,
        "membership_used_for_scoring": False,
        "benchmark_id_used_as_model_input": False,
        "score_eos": False,
        "batch_size": 64,
    }
    for key, value in expected_scoring.items():
        if scoring.get(key) != value:
            raise ShadowGoldProbeLocalError(f"probe-local scoring contract changed: {key}")
    gate = config.get("sensitivity_gate") or {}
    expected_gate = {
        "primary_predictor": "probe_digest_logp_mean",
        "shadow_a_holdout_min_auc": 0.60,
        "shadow_b_blind_holdout_min_auc": 0.60,
        "shadow_a_holdout_min_tpr_at_1pct_fpr": 0.03,
        "shadow_b_blind_holdout_min_tpr_at_1pct_fpr": 0.03,
        "all_four_thresholds_required": True,
        "secondary_predictor_may_rescue_failed_primary_gate": False,
        "failed_gate_is_valid_scientific_result_not_runtime_failure": True,
    }
    for key, value in expected_gate.items():
        if gate.get(key) != value:
            raise ShadowGoldProbeLocalError(f"probe-local sensitivity gate changed: {key}")
    runtime = config.get("runtime") or {}
    if runtime.get("backend") != "cuda" or runtime.get("kaggle_machine_shape") != "NvidiaTeslaT4":
        raise ShadowGoldProbeLocalError("probe-local CUDA resource contract changed")
    if runtime.get("expected_visible_gpu_count") != 2 or runtime.get("per_architecture_world_size") != 1:
        raise ShadowGoldProbeLocalError("probe-local GPU/world-size contract changed")
    for key in ("distributed_training", "ddp_used", "nccl_used"):
        if runtime.get(key) is not False:
            raise ShadowGoldProbeLocalError(f"probe-local distributed-training guard changed: {key}")
    if runtime.get("automatic_compute_retries") != 0 or runtime.get("notebook_internet") is not False:
        raise ShadowGoldProbeLocalError("probe-local retry/internet contract changed")
    guards = config.get("scientific_guards") or {}
    for key in (
        "generic_stage2_primitive_suite_changed",
        "stage2_v3_selection_allowed",
        "probe_local_score_may_become_stage2_feature",
        "external_model_holdout_consumed",
        "fourth_external_holdout_consumed",
        "public_leaderboard_feedback_used",
        "hidden_stage1_validation_labels_used",
    ):
        if guards.get(key) is not False:
            raise ShadowGoldProbeLocalError(f"probe-local scientific guard changed: {key}")
    if guards.get("competition_rows_used") != 0 or guards.get("external_rows_used") != 0:
        raise ShadowGoldProbeLocalError("probe-local external/competition rows changed")


def _probe_parts(content: str, language: str) -> tuple[str, str, str, str]:
    prefix = "\n# gold_probe_" if language in {"Python", "Ruby"} else "\n// gold_probe_"
    suffix_width = len(prefix) + 16 + 1
    if len(content) < suffix_width:
        raise ShadowGoldProbeLocalError("positive-control content is shorter than probe suffix")
    probe = content[-suffix_width:]
    base = content[:-suffix_width]
    expected = positive_v1.positive_control_probe(base, language)
    if probe != expected:
        raise ShadowGoldProbeLocalError("positive-control probe is not the expected content-derived suffix")
    digest = probe[len(prefix) : len(prefix) + 16]
    if len(digest) != 16 or any(character not in "0123456789abcdef" for character in digest):
        raise ShadowGoldProbeLocalError("positive-control digest format changed")
    return base, prefix, digest, probe


def build_probe_token_plan(
    content: str,
    language: str,
    *,
    tokenizer: ByteCodeTokenizer | None = None,
    max_sequence_tokens: int = 256,
) -> dict[str, Any]:
    """Build the exact suffix window and model-logit positions for the known probe."""

    runtime_tokenizer = tokenizer or ByteCodeTokenizer()
    if max_sequence_tokens != 256:
        raise ShadowGoldProbeLocalError("probe-local calibration requires the frozen 256-token context")
    _, prefix, digest, probe = _probe_parts(str(content), str(language))
    payload = runtime_tokenizer.encode(str(content), add_special_tokens=False)
    probe_tokens = runtime_tokenizer.encode(probe, add_special_tokens=False)
    prefix_tokens = runtime_tokenizer.encode(prefix, add_special_tokens=False)
    digest_tokens = runtime_tokenizer.encode(digest, add_special_tokens=False)
    if len(digest_tokens) != 16:
        raise ShadowGoldProbeLocalError("byte-tokenized probe digest is not 16 tokens")
    capacity = max_sequence_tokens - 2
    selected_start = max(0, len(payload) - capacity)
    selected = payload[selected_start:]
    global_probe_start = len(payload) - len(probe_tokens)
    local_probe_start = global_probe_start - selected_start
    if local_probe_start < 0:
        raise ShadowGoldProbeLocalError("probe suffix does not fit in the frozen suffix window")
    local_digest_start = local_probe_start + len(prefix_tokens)
    full_positions = list(range(local_probe_start, local_probe_start + len(probe_tokens)))
    digest_positions = list(range(local_digest_start, local_digest_start + len(digest_tokens)))
    if not digest_positions or digest_positions[-1] >= len(selected):
        raise ShadowGoldProbeLocalError("probe digest positions exceed selected suffix payload")
    input_ids = [runtime_tokenizer.bos_token_id, *selected, runtime_tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    padding = max_sequence_tokens - len(input_ids)
    input_ids += [runtime_tokenizer.pad_token_id] * padding
    attention_mask += [0] * padding
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "probe_logit_positions": full_positions,
        "digest_logit_positions": digest_positions,
        "probe_token_count": len(probe_tokens),
        "digest_token_count": len(digest_tokens),
    }


def score_probe_local_features(
    training_output_directory: str | Path,
    scoring_bundle_directory: str | Path,
    *,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score probe bytes only, without loading or receiving membership labels."""

    import torch

    if batch_size != 64:
        raise ShadowGoldProbeLocalError("probe-local batch size changed")
    bundle = load_scoring_bundle(scoring_bundle_directory)
    checkpoint = load_shadow_checkpoint(training_output_directory, backend="cuda")
    training_manifest = checkpoint.training_manifest
    if int(training_manifest.get("completed_steps", -1)) != 1280:
        raise ShadowGoldProbeLocalError("probe-local checkpoint does not contain 1,280 frozen steps")
    if training_manifest.get("pretrained_weights_used") is not False:
        raise ShadowGoldProbeLocalError("probe-local checkpoint pretrained-weight guard failed")
    if training_manifest.get("evaluation_labels_read") is not False:
        raise ShadowGoldProbeLocalError("probe-local checkpoint label guard failed")
    source = bundle.frame.reset_index(drop=True)
    tokenizer = ByteCodeTokenizer()
    plans = [
        build_probe_token_plan(content, language, tokenizer=tokenizer, max_sequence_tokens=256)
        for content, language in zip(source.content.astype(str), source.language.astype(str))
    ]
    digest_scores = np.full(len(source), np.nan, dtype=np.float64)
    full_scores = np.full(len(source), np.nan, dtype=np.float64)
    model = checkpoint.model
    model.eval()
    device = next(model.parameters()).device
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
                digest_positions = torch.as_tensor(plan["digest_logit_positions"], dtype=torch.long, device=device)
                probe_positions = torch.as_tensor(plan["probe_logit_positions"], dtype=torch.long, device=device)
                digest_scores[start + local_index] = float(correct[local_index].index_select(0, digest_positions).mean().cpu())
                full_scores[start + local_index] = float(correct[local_index].index_select(0, probe_positions).mean().cpu())
            del input_ids, attention_mask, output, logits, shifted_logits, targets, correct
    if not np.isfinite(digest_scores).all() or not np.isfinite(full_scores).all():
        raise ShadowGoldProbeLocalError("probe-local score coverage or finiteness failed")
    features = source[["benchmark_id", "language", "length_bin", "character_count"]].copy()
    features["probe_digest_logp_mean"] = digest_scores
    features["probe_full_logp_mean"] = full_scores
    features = features[_FEATURE_COLUMNS]
    if features.benchmark_id.duplicated().any():
        raise ShadowGoldProbeLocalError("probe-local feature IDs are duplicated")
    manifest = {
        "status": "sealed",
        "version": SHADOW_GOLD_PROBE_LOCAL_VERSION,
        "architecture_slot": training_manifest.get("architecture_slot"),
        "rows": len(features),
        "primary_predictor": "probe_digest_logp_mean",
        "secondary_predictor": "probe_full_logp_mean",
        "digest_token_count": 16,
        "probe_boundary_derived_from_content_and_language_only": True,
        "membership_labels_read": False,
        "benchmark_ids_passed_to_model": False,
        "evaluation_input_sha256": bundle.input_sha256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "model_state_sha256": checkpoint.model_state_sha256,
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
        raise ShadowGoldProbeLocalError("parent-only label seal is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("passed_to_gpu_children") is not False:
        raise ShadowGoldProbeLocalError("probe-local parent label boundary failed")
    if manifest.get("label_sha256") != gold_v1._sha256_file(path):
        raise ShadowGoldProbeLocalError("probe-local label SHA mismatch")
    labels = pd.read_json(path, lines=True)
    return gold_v2.canonicalize_exact_schema(labels, gold_v1._LABEL_COLUMNS, artifact_name="Gold parent label")


def evaluate_probe_local_gate(
    left_features: pd.DataFrame,
    right_features: pd.DataFrame,
    label_directory: str | Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Reveal labels only after both probe-local feature tables are sealed."""

    validate_probe_local_config(config)
    if list(left_features.columns) != _FEATURE_COLUMNS or list(right_features.columns) != _FEATURE_COLUMNS:
        raise ShadowGoldProbeLocalError("probe-local feature schema changed")
    if left_features.benchmark_id.astype(str).tolist() != right_features.benchmark_id.astype(str).tolist():
        raise ShadowGoldProbeLocalError("probe-local left/right row order differs")
    labels = _load_labels(label_directory)
    left_labeled = attach_exact_labels(left_features, labels)
    right_labeled = attach_exact_labels(right_features, labels)
    _, left_holdout = split_development_holdout(left_labeled, seed=2027)
    _, right_holdout = split_development_holdout(right_labeled, seed=2027)
    holdout_ids = left_holdout.benchmark_id.astype(str).tolist()
    right_holdout = right_holdout.set_index(right_holdout.benchmark_id.astype(str), drop=False).loc[holdout_ids].reset_index(drop=True)
    if right_holdout.benchmark_id.astype(str).tolist() != holdout_ids:
        raise ShadowGoldProbeLocalError("probe-local paired holdout order differs")
    fpr = 0.01
    metrics: dict[str, Any] = {"left": {}, "right": {}}
    for predictor in ("probe_digest_logp_mean", "probe_full_logp_mean"):
        metrics["left"][predictor] = evaluate_frozen_gold_holdout(
            left_holdout,
            left_holdout[predictor].to_numpy(dtype=np.float64),
            false_positive_rate=fpr,
        )["overall"]
        metrics["right"][predictor] = evaluate_frozen_gold_holdout(
            right_holdout,
            right_holdout[predictor].to_numpy(dtype=np.float64),
            false_positive_rate=fpr,
        )["overall"]
    gate = config["sensitivity_gate"]
    primary_left = metrics["left"]["probe_digest_logp_mean"]
    primary_right = metrics["right"]["probe_digest_logp_mean"]
    checks = {
        "shadow_a_auc": float(primary_left["auc"]) >= float(gate["shadow_a_holdout_min_auc"]),
        "shadow_b_auc": float(primary_right["auc"]) >= float(gate["shadow_b_blind_holdout_min_auc"]),
        "shadow_a_tpr_at_1pct_fpr": float(primary_left["tpr_at_fpr"]) >= float(gate["shadow_a_holdout_min_tpr_at_1pct_fpr"]),
        "shadow_b_tpr_at_1pct_fpr": float(primary_right["tpr_at_fpr"]) >= float(gate["shadow_b_blind_holdout_min_tpr_at_1pct_fpr"]),
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "version": SHADOW_GOLD_PROBE_LOCAL_VERSION,
        "primary_predictor": "probe_digest_logp_mean",
        "secondary_predictor": "probe_full_logp_mean",
        "checks": checks,
        "holdout_rows_per_architecture": len(left_holdout),
        "metrics": metrics,
        "secondary_predictor_can_rescue": False,
        "failed_gate_is_valid_scientific_result": True,
        "stage2_v3_selection_allowed": False,
    }
    predictions = left_holdout[["benchmark_id", "language", "length_bin", "character_count", "membership", "matched_pair_id"]].copy()
    predictions["left_probe_digest_logp_mean"] = left_holdout.probe_digest_logp_mean.to_numpy(dtype=np.float64)
    predictions["right_probe_digest_logp_mean"] = right_holdout.probe_digest_logp_mean.to_numpy(dtype=np.float64)
    predictions["left_probe_full_logp_mean"] = left_holdout.probe_full_logp_mean.to_numpy(dtype=np.float64)
    predictions["right_probe_full_logp_mean"] = right_holdout.probe_full_logp_mean.to_numpy(dtype=np.float64)
    return result, predictions


def run_probe_local_benchmark(
    *,
    scratch_directory: str | Path,
    positive_control_config: Mapping[str, Any],
    probe_local_config: Mapping[str, Any],
    timeout_seconds: int = 5400,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Repeat frozen v2 training and add calibration-only probe-local scoring."""

    positive_v1.validate_positive_control_config(positive_control_config)
    validate_probe_local_config(probe_local_config)
    names = visible_cuda_device_names()
    validate_gpu_inventory(names, expected_count=2, required_name_fragment="T4")
    scratch = Path(scratch_directory).resolve()
    if scratch.exists() and any(scratch.iterdir()):
        raise ShadowGoldProbeLocalError("probe-local scratch directory is not empty")
    scratch.mkdir(parents=True, exist_ok=True)
    prepared = positive_v2.prepare_positive_control_runtime(scratch / "prepared", positive_control_config)
    dual = gold_v1.run_gold_dual_gpu(
        protocol_directory=prepared["protocol_directory"],
        scoring_bundle_directory=prepared["scoring_bundle_directory"],
        output_directory=scratch / "dual",
        expected_steps=1280,
        timeout_seconds=timeout_seconds,
    )
    left_features, left_manifest = score_probe_local_features(
        scratch / "dual" / "left" / "training",
        prepared["scoring_bundle_directory"],
        batch_size=64,
    )
    right_features, right_manifest = score_probe_local_features(
        scratch / "dual" / "right" / "training",
        prepared["scoring_bundle_directory"],
        batch_size=64,
    )
    probe_dir = scratch / "probe_sealed"
    probe_dir.mkdir()
    left_path = probe_dir / "left_probe_features.jsonl"
    right_path = probe_dir / "right_probe_features.jsonl"
    gold_v1._write_jsonl(left_features, left_path, _FEATURE_COLUMNS)
    gold_v1._write_jsonl(right_features, right_path, _FEATURE_COLUMNS)
    seal = {
        "status": "sealed",
        "left_sha256": gold_v1._sha256_file(left_path),
        "right_sha256": gold_v1._sha256_file(right_path),
        "left_manifest": left_manifest,
        "right_manifest": right_manifest,
        "membership_labels_read": False,
        "stage2_v3_selection_allowed": False,
    }
    gold_v1._write_json(probe_dir / "probe_feature_seal.json", seal)

    generic_attack, generic_metrics, _ = gold_v2.finalize_gold_results(
        dual_output_directory=scratch / "dual",
        label_directory=prepared["label_directory"],
        false_positive_rate=0.01,
    )
    probe_gate, predictions = evaluate_probe_local_gate(
        left_features,
        right_features,
        prepared["label_directory"],
        probe_local_config,
    )
    metrics = {
        "status": "complete",
        "execution_id": EXECUTION_ID,
        "probe_local_gate": probe_gate,
        "generic_positive_control_metrics": generic_metrics,
        "generic_positive_control_selected_attack_secondary": generic_attack["selected_candidate"],
        "training_protocol_changed_from_positive_control_v2": False,
        "stage2_v3_selection_allowed": False,
        "probe_local_score_may_become_stage2_feature": False,
        "external_model_holdout_consumed": False,
    }
    manifest = {
        "status": "complete",
        "execution_id": EXECUTION_ID,
        "version": SHADOW_GOLD_PROBE_LOCAL_VERSION,
        "gpu_names": names,
        "prepared": prepared,
        "dual_gpu": dual,
        "probe_feature_seal": seal,
        "probe_local_gate_status": probe_gate["status"],
        "primary_predictor": "probe_digest_logp_mean",
        "training_protocol_changed_from_positive_control_v2": False,
        "evaluation_labels_passed_to_training_children": False,
        "probe_scores_sealed_before_label_reveal": True,
        "competition_rows_used": 0,
        "external_rows_used": 0,
        "pretrained_weights_used": False,
        "automatic_compute_retries": 0,
        "stage2_v3_selection_allowed": False,
        "probe_local_score_may_become_stage2_feature": False,
        "external_model_holdout_consumed": False,
        "fourth_external_holdout_consumed": False,
    }
    return generic_attack, metrics, predictions, manifest


__all__ = [
    "EXECUTION_ID",
    "SHADOW_GOLD_PROBE_LOCAL_VERSION",
    "ShadowGoldProbeLocalError",
    "build_probe_token_plan",
    "evaluate_probe_local_gate",
    "run_probe_local_benchmark",
    "score_probe_local_features",
    "validate_probe_local_config",
]
