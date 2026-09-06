"""Attribute the Gold training-start boundary effect to first-token vs tail context.

The preceding suffix-context decomposition showed that the generic Stage2 scorer
already effectively selected the repeatedly trained suffix and that the 256->254
crop was not the dominant recovery mechanism.  Exact BOS + suffix-254 framing
recovered a large membership signal, but that comparison also made the first
suffix payload token scoreable.  This module freezes a no-selection attribution:
score t1|BOS separately and compare the identical downstream targets t2..t254
with and without BOS/start-position alignment.
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

SHADOW_GOLD_START_BOUNDARY_ATTRIBUTION_VERSION = "start-boundary-attribution-v1"
EXECUTION_ID = "shadow-gold-start-boundary-attribution-v1"
CONTEXT_DECOMPOSITION_RESULT_COMMIT = "cf5da228ac0c0ffd7a1e13edfb2835f5a8655015"
_CONDITIONS = [
    "raw_stage2_loss_max",
    "raw_suffix254_tail_mean",
    "bos_suffix254_all_mean",
    "bos_suffix254_first_logp",
    "bos_suffix254_tail_mean",
]
_METADATA_COLUMNS = ["benchmark_id", "language", "length_bin", "character_count"]
_FEATURE_COLUMNS = [*_METADATA_COLUMNS, *_CONDITIONS]


class ShadowGoldStartBoundaryAttributionError(RuntimeError):
    """Raised when a frozen start-boundary attribution invariant fails."""


def validate_start_boundary_attribution_config(config: Mapping[str, Any]) -> None:
    if config.get("execution_id") != EXECUTION_ID or config.get("role") != "gold_start_boundary_attribution":
        raise ShadowGoldStartBoundaryAttributionError("start-boundary attribution identity changed")

    evidence = config.get("base_evidence") or {}
    expected_evidence = {
        "context_decomposition_result_commit": CONTEXT_DECOMPOSITION_RESULT_COMMIT,
        "context_decomposition_formal_pass": True,
        "frozen_named_mechanism": "bos_start_context_material",
        "publication_interpretation_requires_first_vs_tail_attribution": True,
    }
    for key, value in expected_evidence.items():
        if evidence.get(key) != value:
            raise ShadowGoldStartBoundaryAttributionError(f"start-boundary evidence changed: {key}")

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
        "learning_rate": 0.0003,
        "weight_decay": 0.1,
        "warmup_fraction": 0.05,
        "gradient_clip_norm": 1.0,
        "seed": 2027,
        "random_initialisation": True,
        "pretrained_weights_used": False,
        "scientific_protocol_changed_from_positive_control_v2": False,
    }
    for key, value in expected_training.items():
        if training.get(key) != value:
            raise ShadowGoldStartBoundaryAttributionError(f"start-boundary training contract changed: {key}")

    scoring = config.get("scoring") or {}
    expected_scoring = {
        "version": SHADOW_GOLD_START_BOUNDARY_ATTRIBUTION_VERSION,
        "tokenizer": "lossless_utf8_bytes",
        "max_sequence_tokens": 256,
        "payload_capacity_tokens": 254,
        "candidate_selection_used": False,
        "probe_boundary_used": False,
        "probe_text_used": False,
        "membership_used_for_scoring": False,
        "benchmark_id_used_as_model_input": False,
        "batch_size": 64,
        "raw_tail_and_bos_tail_target_tokens_identical": True,
    }
    for key, value in expected_scoring.items():
        if scoring.get(key) != value:
            raise ShadowGoldStartBoundaryAttributionError(f"start-boundary scoring contract changed: {key}")
    expected_conditions = {
        "raw_stage2_loss_max": "stage2-model-independent-v1 score_logp_mean__max",
        "raw_suffix254_tail_mean": "input t1..t254 without BOS; score targets t2..t254",
        "bos_suffix254_all_mean": "input BOS+t1..t254+EOS; score targets t1..t254",
        "bos_suffix254_first_logp": "input BOS+t1..t254+EOS; score only target t1 conditioned on BOS",
        "bos_suffix254_tail_mean": "input BOS+t1..t254+EOS; score targets t2..t254",
    }
    if scoring.get("conditions") != expected_conditions or list(config.get("condition_order") or []) != _CONDITIONS:
        raise ShadowGoldStartBoundaryAttributionError("start-boundary condition matrix changed")

    decision = config.get("mechanistic_decision") or {}
    expected_decision = {
        "all_mean_min_auc_each_architecture": 0.56,
        "all_mean_min_gain_vs_raw_stage2_each_architecture": 0.03,
        "all_mean_min_tpr_at_1pct_fpr_each_architecture": 0.015,
        "tail_min_auc_each_architecture": 0.56,
        "tail_min_gain_vs_raw_tail_each_architecture": 0.03,
        "tail_min_tpr_at_1pct_fpr_each_architecture": 0.015,
        "first_min_auc_each_architecture": 0.56,
        "first_min_tpr_at_1pct_fpr_each_architecture": 0.015,
        "both_architectures_required_for_named_mechanism": True,
        "criterion_is_mechanistic_only": True,
        "does_not_promote_stage2_v3": True,
    }
    for key, value in expected_decision.items():
        if decision.get(key) != value:
            raise ShadowGoldStartBoundaryAttributionError(f"start-boundary decision changed: {key}")

    runtime = config.get("runtime") or {}
    if runtime.get("backend") != "cuda" or runtime.get("kaggle_machine_shape") != "NvidiaTeslaT4":
        raise ShadowGoldStartBoundaryAttributionError("start-boundary CUDA resource changed")
    if runtime.get("expected_visible_gpu_count") != 2 or runtime.get("per_architecture_world_size") != 1:
        raise ShadowGoldStartBoundaryAttributionError("start-boundary GPU/world-size changed")
    for key in ("distributed_training", "ddp_used", "nccl_used"):
        if runtime.get(key) is not False:
            raise ShadowGoldStartBoundaryAttributionError(f"start-boundary distributed guard changed: {key}")
    if runtime.get("automatic_compute_retries") != 0 or runtime.get("notebook_internet") is not False:
        raise ShadowGoldStartBoundaryAttributionError("start-boundary retry/internet contract changed")

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
            raise ShadowGoldStartBoundaryAttributionError(f"start-boundary scientific guard changed: {key}")
    if guards.get("features_sealed_before_label_reveal") is not True:
        raise ShadowGoldStartBoundaryAttributionError("start-boundary feature-seal guard changed")
    if guards.get("competition_rows_used") != 0 or guards.get("external_rows_used") != 0:
        raise ShadowGoldStartBoundaryAttributionError("start-boundary external-row guard changed")


def build_start_boundary_plans(
    content: str,
    *,
    tokenizer: ByteCodeTokenizer | None = None,
) -> dict[str, Any]:
    """Build raw/BOS suffix plans with an exact shared downstream target set."""

    runtime_tokenizer = tokenizer or ByteCodeTokenizer()
    payload = runtime_tokenizer.encode(str(content), add_special_tokens=False)
    if len(payload) < 2:
        raise ShadowGoldStartBoundaryAttributionError("start-boundary sample has fewer than two payload tokens")
    selected = list(payload[-254:])
    if len(selected) < 2:
        raise ShadowGoldStartBoundaryAttributionError("start-boundary suffix has fewer than two payload tokens")

    raw_ids = list(selected)
    raw_mask = [1] * len(raw_ids)
    raw_ids += [runtime_tokenizer.pad_token_id] * (256 - len(raw_ids))
    raw_mask += [0] * (256 - len(raw_mask))

    bos_ids = [runtime_tokenizer.bos_token_id, *selected, runtime_tokenizer.eos_token_id]
    bos_mask = [1] * len(bos_ids)
    bos_ids += [runtime_tokenizer.pad_token_id] * (256 - len(bos_ids))
    bos_mask += [0] * (256 - len(bos_mask))

    # For raw [t1..tN], shifted-logit position 0 predicts t2, so positions
    # 0..N-2 target t2..tN.  For BOS+[t1..tN]+EOS, position 0 predicts t1,
    # positions 1..N-1 target the identical downstream set t2..tN.
    raw_tail_positions = list(range(len(selected) - 1))
    bos_all_positions = list(range(len(selected)))
    bos_first_positions = [0]
    bos_tail_positions = list(range(1, len(selected)))
    if len(raw_tail_positions) != len(bos_tail_positions):
        raise ShadowGoldStartBoundaryAttributionError("raw/BOS tail target count differs")

    return {
        "selected_payload_ids": selected,
        "raw_input_ids": raw_ids,
        "raw_attention_mask": raw_mask,
        "bos_input_ids": bos_ids,
        "bos_attention_mask": bos_mask,
        "raw_tail_logit_positions": raw_tail_positions,
        "bos_all_logit_positions": bos_all_positions,
        "bos_first_logit_positions": bos_first_positions,
        "bos_tail_logit_positions": bos_tail_positions,
        "raw_tail_target_payload_ids": selected[1:],
        "bos_tail_target_payload_ids": selected[1:],
        "bos_first_target_payload_id": selected[0],
    }


def _correct_logp(model: Any, input_ids: Any, attention_mask: Any, torch: Any) -> Any:
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits if hasattr(output, "logits") else output[0]
    shifted_logits = logits[:, :-1].float()
    targets = input_ids[:, 1:]
    correct = torch.log_softmax(shifted_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return correct, output, logits, shifted_logits, targets


def score_start_boundary_features(
    training_output_directory: str | Path,
    scoring_bundle_directory: str | Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score fixed first/tail boundary conditions before labels are opened."""

    import torch

    validate_start_boundary_attribution_config(config)
    bundle = load_scoring_bundle(scoring_bundle_directory)
    checkpoint = load_shadow_checkpoint(training_output_directory, backend="cuda")
    if int(checkpoint.training_manifest.get("completed_steps", -1)) != 1280:
        raise ShadowGoldStartBoundaryAttributionError("start-boundary checkpoint step count changed")
    if checkpoint.training_manifest.get("evaluation_labels_read") is not False:
        raise ShadowGoldStartBoundaryAttributionError("start-boundary checkpoint label guard failed")
    if checkpoint.training_manifest.get("pretrained_weights_used") is not False:
        raise ShadowGoldStartBoundaryAttributionError("start-boundary pretrained-weight guard failed")

    source = bundle.frame.reset_index(drop=True)
    tokenizer = ByteCodeTokenizer()
    plans = [build_start_boundary_plans(content, tokenizer=tokenizer) for content in source.content.astype(str)]
    for plan in plans:
        if plan["raw_tail_target_payload_ids"] != plan["bos_tail_target_payload_ids"]:
            raise ShadowGoldStartBoundaryAttributionError("raw/BOS downstream target identity failed")

    model = checkpoint.model
    model.eval()
    device = next(model.parameters()).device
    batch_size = int(config["scoring"]["batch_size"])

    features = source[_METADATA_COLUMNS].copy()
    features["raw_stage2_loss_max"] = context_v1._raw_stage2_loss_max(model, source)
    raw_tail = np.full(len(source), np.nan, dtype=np.float64)
    bos_all = np.full(len(source), np.nan, dtype=np.float64)
    bos_first = np.full(len(source), np.nan, dtype=np.float64)
    bos_tail = np.full(len(source), np.nan, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(plans), batch_size):
            stop = min(start + batch_size, len(plans))
            batch = plans[start:stop]

            raw_ids = torch.as_tensor([row["raw_input_ids"] for row in batch], dtype=torch.long, device=device)
            raw_mask = torch.as_tensor([row["raw_attention_mask"] for row in batch], dtype=torch.long, device=device)
            raw_correct, raw_output, raw_logits, raw_shifted, raw_targets = _correct_logp(model, raw_ids, raw_mask, torch)
            for local_index, plan in enumerate(batch):
                positions = torch.as_tensor(plan["raw_tail_logit_positions"], dtype=torch.long, device=device)
                raw_tail[start + local_index] = float(raw_correct[local_index].index_select(0, positions).mean().detach().cpu())
            del raw_ids, raw_mask, raw_correct, raw_output, raw_logits, raw_shifted, raw_targets

            bos_ids = torch.as_tensor([row["bos_input_ids"] for row in batch], dtype=torch.long, device=device)
            bos_mask = torch.as_tensor([row["bos_attention_mask"] for row in batch], dtype=torch.long, device=device)
            bos_correct, bos_output, bos_logits, bos_shifted, bos_targets = _correct_logp(model, bos_ids, bos_mask, torch)
            for local_index, plan in enumerate(batch):
                all_pos = torch.as_tensor(plan["bos_all_logit_positions"], dtype=torch.long, device=device)
                tail_pos = torch.as_tensor(plan["bos_tail_logit_positions"], dtype=torch.long, device=device)
                all_values = bos_correct[local_index].index_select(0, all_pos)
                tail_values = bos_correct[local_index].index_select(0, tail_pos)
                row_index = start + local_index
                bos_all[row_index] = float(all_values.mean().detach().cpu())
                bos_first[row_index] = float(bos_correct[local_index, 0].detach().cpu())
                bos_tail[row_index] = float(tail_values.mean().detach().cpu())
            del bos_ids, bos_mask, bos_correct, bos_output, bos_logits, bos_shifted, bos_targets

    values = {
        "raw_suffix254_tail_mean": raw_tail,
        "bos_suffix254_all_mean": bos_all,
        "bos_suffix254_first_logp": bos_first,
        "bos_suffix254_tail_mean": bos_tail,
    }
    for name, array in values.items():
        if not np.isfinite(array).all():
            raise ShadowGoldStartBoundaryAttributionError(f"start-boundary score is non-finite: {name}")
        features[name] = array
    features = features[_FEATURE_COLUMNS]
    if features.benchmark_id.duplicated().any():
        raise ShadowGoldStartBoundaryAttributionError("start-boundary feature IDs are duplicated")

    manifest = {
        "status": "sealed",
        "version": SHADOW_GOLD_START_BOUNDARY_ATTRIBUTION_VERSION,
        "architecture_slot": checkpoint.training_manifest.get("architecture_slot"),
        "rows": len(features),
        "conditions": list(_CONDITIONS),
        "candidate_selection_used": False,
        "raw_tail_and_bos_tail_target_tokens_identical": True,
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
        raise ShadowGoldStartBoundaryAttributionError("start-boundary parent label seal is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("passed_to_gpu_children") is not False:
        raise ShadowGoldStartBoundaryAttributionError("start-boundary label boundary failed")
    if manifest.get("label_sha256") != gold_v1._sha256_file(path):
        raise ShadowGoldStartBoundaryAttributionError("start-boundary parent label hash mismatch")
    labels = pd.read_json(path, lines=True)
    return gold_v2.canonicalize_exact_schema(labels, gold_v1._LABEL_COLUMNS, artifact_name="Gold parent label")


def _mechanism_name(tail_material: bool, first_material: bool) -> str:
    if tail_material and first_material:
        return "both_first_and_downstream_material"
    if tail_material:
        return "downstream_context_dominant"
    if first_material:
        return "first_boundary_token_dominant"
    return "mixed_or_unresolved"


def evaluate_start_boundary_attribution(
    left_features: pd.DataFrame,
    right_features: pd.DataFrame,
    label_directory: str | Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate all fixed conditions on the paired holdout; no candidate selection."""

    validate_start_boundary_attribution_config(config)
    if list(left_features.columns) != _FEATURE_COLUMNS or list(right_features.columns) != _FEATURE_COLUMNS:
        raise ShadowGoldStartBoundaryAttributionError("start-boundary feature schema changed")
    if left_features.benchmark_id.astype(str).tolist() != right_features.benchmark_id.astype(str).tolist():
        raise ShadowGoldStartBoundaryAttributionError("start-boundary left/right row order differs")

    labels = _load_labels(label_directory)
    left_labeled = attach_exact_labels(left_features, labels)
    right_labeled = attach_exact_labels(right_features, labels)
    _, left_holdout = split_development_holdout(left_labeled, seed=2027)
    _, right_holdout = split_development_holdout(right_labeled, seed=2027)
    holdout_ids = left_holdout.benchmark_id.astype(str).tolist()
    right_holdout = right_holdout.set_index(right_holdout.benchmark_id.astype(str), drop=False).loc[holdout_ids].reset_index(drop=True)
    if right_holdout.benchmark_id.astype(str).tolist() != holdout_ids:
        raise ShadowGoldStartBoundaryAttributionError("start-boundary paired holdout order changed")

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
        raw_stage2 = m["raw_stage2_loss_max"]
        raw_tail = m["raw_suffix254_tail_mean"]
        all_mean = m["bos_suffix254_all_mean"]
        first = m["bos_suffix254_first_logp"]
        tail = m["bos_suffix254_tail_mean"]
        side_checks = {
            "all_mean_min_auc": all_mean["auc"] >= float(decision["all_mean_min_auc_each_architecture"]),
            "all_mean_min_gain": all_mean["auc"] - raw_stage2["auc"] >= float(decision["all_mean_min_gain_vs_raw_stage2_each_architecture"]),
            "all_mean_min_tpr": all_mean["tpr_at_fpr"] >= float(decision["all_mean_min_tpr_at_1pct_fpr_each_architecture"]),
            "tail_min_auc": tail["auc"] >= float(decision["tail_min_auc_each_architecture"]),
            "tail_min_gain": tail["auc"] - raw_tail["auc"] >= float(decision["tail_min_gain_vs_raw_tail_each_architecture"]),
            "tail_min_tpr": tail["tpr_at_fpr"] >= float(decision["tail_min_tpr_at_1pct_fpr_each_architecture"]),
            "first_min_auc": first["auc"] >= float(decision["first_min_auc_each_architecture"]),
            "first_min_tpr": first["tpr_at_fpr"] >= float(decision["first_min_tpr_at_1pct_fpr_each_architecture"]),
        }
        checks[side] = side_checks
        tail_material = all(side_checks[name] for name in ("tail_min_auc", "tail_min_gain", "tail_min_tpr"))
        first_material = all(side_checks[name] for name in ("first_min_auc", "first_min_tpr"))
        mechanism_per_side[side] = _mechanism_name(tail_material, first_material)

    all_mean_recovery = all(
        checks[side][name]
        for side in ("left", "right")
        for name in ("all_mean_min_auc", "all_mean_min_gain", "all_mean_min_tpr")
    )
    downstream_context_material = all(
        checks[side][name]
        for side in ("left", "right")
        for name in ("tail_min_auc", "tail_min_gain", "tail_min_tpr")
    )
    first_token_material = all(
        checks[side][name]
        for side in ("left", "right")
        for name in ("first_min_auc", "first_min_tpr")
    )
    named_mechanism = (
        mechanism_per_side["left"]
        if mechanism_per_side["left"] == mechanism_per_side["right"]
        else "architecture_dependent"
    )

    result = {
        "status": "complete",
        "version": SHADOW_GOLD_START_BOUNDARY_ATTRIBUTION_VERSION,
        "condition_metrics": condition_metrics,
        "mechanistic_checks": checks,
        "all_mean_recovery": all_mean_recovery,
        "downstream_context_material": downstream_context_material,
        "first_token_material": first_token_material,
        "mechanism_per_architecture": mechanism_per_side,
        "named_mechanism": named_mechanism,
        "candidate_selection_used": False,
        "raw_tail_and_bos_tail_target_tokens_identical": True,
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


def run_start_boundary_attribution_benchmark(
    *,
    scratch_directory: str | Path,
    positive_control_config: Mapping[str, Any],
    attribution_config: Mapping[str, Any],
    timeout_seconds: int = 5400,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Repeat frozen training, seal fixed boundary scores, then reveal labels."""

    positive_v1.validate_positive_control_config(positive_control_config)
    validate_start_boundary_attribution_config(attribution_config)
    names = visible_cuda_device_names()
    validate_gpu_inventory(names, expected_count=2, required_name_fragment="T4")

    scratch = Path(scratch_directory).resolve()
    if scratch.exists() and any(scratch.iterdir()):
        raise ShadowGoldStartBoundaryAttributionError("start-boundary scratch directory is not empty")
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
    left_features, left_manifest = score_start_boundary_features(
        scratch / "training" / "left", prepared["scoring_bundle_directory"], attribution_config
    )
    right_features, right_manifest = score_start_boundary_features(
        scratch / "training" / "right", prepared["scoring_bundle_directory"], attribution_config
    )

    seal_dir = scratch / "start_boundary_feature_seal"
    seal_dir.mkdir()
    left_path = seal_dir / "left_start_boundary_features.jsonl"
    right_path = seal_dir / "right_start_boundary_features.jsonl"
    gold_v1._write_jsonl(left_features, left_path, list(left_features.columns))
    gold_v1._write_jsonl(right_features, right_path, list(right_features.columns))
    seal = {
        "status": "sealed",
        "left_sha256": gold_v1._sha256_file(left_path),
        "right_sha256": gold_v1._sha256_file(right_path),
        "left_manifest": left_manifest,
        "right_manifest": right_manifest,
        "candidate_selection_used": False,
        "raw_tail_and_bos_tail_target_tokens_identical": True,
        "membership_labels_read": False,
        "probe_boundary_used": False,
        "probe_text_used": False,
        "stage2_v3_selection_allowed": False,
    }
    gold_v1._write_json(seal_dir / "start_boundary_feature_seal.json", seal)

    evaluation, predictions = evaluate_start_boundary_attribution(
        left_features, right_features, prepared["label_directory"], attribution_config
    )
    manifest = {
        "status": "complete",
        "execution_id": EXECUTION_ID,
        "version": SHADOW_GOLD_START_BOUNDARY_ATTRIBUTION_VERSION,
        "gpu_names": names,
        "prepared": prepared,
        "dual_gpu_training": dual,
        "start_boundary_feature_seal": seal,
        "all_mean_recovery": evaluation["all_mean_recovery"],
        "downstream_context_material": evaluation["downstream_context_material"],
        "first_token_material": evaluation["first_token_material"],
        "named_mechanism": evaluation["named_mechanism"],
        "training_protocol_changed_from_positive_control_v2": False,
        "features_sealed_before_label_reveal": True,
        "raw_tail_and_bos_tail_target_tokens_identical": True,
        "probe_boundary_used": False,
        "probe_text_used": False,
        "competition_rows_used": 0,
        "external_rows_used": 0,
        "pretrained_weights_used": False,
        "automatic_compute_retries": 0,
        "candidate_selection_used": False,
        "stage2_v3_selection_allowed": False,
        "competition_feature_promotion_allowed": False,
        "external_model_holdout_consumed": False,
        "fourth_external_holdout_consumed": False,
    }
    return evaluation, predictions, manifest


__all__ = [
    "EXECUTION_ID",
    "SHADOW_GOLD_START_BOUNDARY_ATTRIBUTION_VERSION",
    "ShadowGoldStartBoundaryAttributionError",
    "build_start_boundary_plans",
    "evaluate_start_boundary_attribution",
    "run_start_boundary_attribution_benchmark",
    "score_start_boundary_features",
    "validate_start_boundary_attribution_config",
]
