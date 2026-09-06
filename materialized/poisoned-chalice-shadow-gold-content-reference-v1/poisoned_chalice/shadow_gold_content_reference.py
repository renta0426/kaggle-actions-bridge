"""Content-difficulty calibrated local membership diagnostic for Gold shadows.

The preceding Gold experiments establish two facts: the deliberately repeated
content-derived probe is memorised when scored at its known boundary, while raw
file/window scalar aggregations fail to recover that signal.  This module tests
one mechanistic explanation without using the probe text or boundary: intrinsic
content difficulty confounds raw target-model likelihood.

A label-free, per-language byte n-gram reference is fitted to the evaluation
batch.  Every file is scored leave-one-file-out, so the reference never trains
on the file it is evaluating.  Target token log-probability is then residualised
against this content-only reference and aggregated over generic local spans.
Membership labels are revealed only after both architecture feature tables are
sealed.  Gold labels may select a candidate only for this controlled experiment;
no result here directly promotes a Stage2-v3 or competition feature.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import shadow_gold_kaggle as gold_v1
from . import shadow_gold_kaggle_v2 as gold_v2
from . import shadow_gold_positive_control as positive_v1
from . import shadow_gold_positive_control_v2 as positive_v2
from . import stage2_api as stage2_core
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


SHADOW_GOLD_CONTENT_REFERENCE_VERSION = "loo-byte-ngram-difficulty-v1"
EXECUTION_ID = "shadow-gold-content-reference-v1"
_CANDIDATES = [
    "residual_mean",
    "residual_top05_mean",
    "residual_top10_mean",
    "residual_local08_max",
    "residual_local16_max",
    "residual_local32_max",
    "residual_local64_max",
]
_METADATA_COLUMNS = ["benchmark_id", "language", "length_bin", "character_count"]
_FEATURE_COLUMNS = [*_METADATA_COLUMNS, *_CANDIDATES, "loss_max", "reference_logp_mean"]


class ShadowGoldContentReferenceError(RuntimeError):
    """Raised when a frozen content-reference invariant fails."""


def validate_content_reference_config(config: Mapping[str, Any]) -> None:
    if config.get("execution_id") != EXECUTION_ID or config.get("role") != "gold_intrinsic_difficulty_calibration":
        raise ShadowGoldContentReferenceError("content-reference identity changed")
    base = config.get("base_evidence") or {}
    if base.get("generic_aggregation_result_commit") != "0a773f252cd89de50eb9ff9b67a94ed2e9baf426":
        raise ShadowGoldContentReferenceError("content-reference base result changed")
    if base.get("probe_local_result_commit") != "844d83992ff364ee7bc43ae3560fb06540e2bd99":
        raise ShadowGoldContentReferenceError("content-reference probe-local evidence changed")
    if base.get("training_protocol_must_be_identical") is not True:
        raise ShadowGoldContentReferenceError("content-reference training identity guard changed")

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
            raise ShadowGoldContentReferenceError(f"content-reference training contract changed: {key}")

    reference = config.get("content_reference") or {}
    expected_reference = {
        "version": SHADOW_GOLD_CONTENT_REFERENCE_VERSION,
        "fit_scope": "evaluation_batch_content_only",
        "per_language": True,
        "leave_one_file_out": True,
        "ngram_order": 5,
        "max_context_bytes": 4,
        "additive_alpha": 0.5,
        "minimum_context_count": 4,
        "byte_vocabulary_size": 256,
        "membership_labels_used": False,
        "benchmark_id_used": False,
        "probe_text_used": False,
        "probe_boundary_used": False,
        "external_model_used": False,
    }
    for key, value in expected_reference.items():
        if reference.get(key) != value:
            raise ShadowGoldContentReferenceError(f"content-reference reference contract changed: {key}")

    target = config.get("target_scoring") or {}
    expected_target = {
        "max_length": 256,
        "max_batch_tokens": 4096,
        "window_policy": "stage2_prefix_middle_suffix_v1",
        "local_widths": [8, 16, 32, 64],
        "top_residual_percents": [5, 10],
        "residual_definition": "target_correct_token_logp_minus_loo_reference_logp",
        "higher_is_more_member_like": True,
        "benchmark_id_used_as_model_input": False,
        "membership_used_for_scoring": False,
    }
    for key, value in expected_target.items():
        if target.get(key) != value:
            raise ShadowGoldContentReferenceError(f"content-reference target scorer changed: {key}")

    if list(config.get("candidate_features") or []) != _CANDIDATES:
        raise ShadowGoldContentReferenceError("content-reference candidate matrix changed")
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
            raise ShadowGoldContentReferenceError(f"content-reference selection changed: {key}")

    recovery = config.get("recovery_criterion") or {}
    expected_recovery = {
        "baseline_candidate": "loss_max",
        "minimum_holdout_auc_each_architecture": 0.56,
        "minimum_auc_gain_vs_baseline_each_architecture": 0.03,
        "minimum_tpr_at_1pct_fpr_each_architecture": 0.015,
        "reference_only_auc_lower_bound": 0.45,
        "reference_only_auc_upper_bound": 0.55,
        "both_architectures_required": True,
        "all_conditions_required": True,
        "criterion_is_mechanistic_only": True,
        "does_not_promote_stage2_v3": True,
    }
    for key, value in expected_recovery.items():
        if recovery.get(key) != value:
            raise ShadowGoldContentReferenceError(f"content-reference recovery criterion changed: {key}")

    runtime = config.get("runtime") or {}
    if runtime.get("backend") != "cuda" or runtime.get("kaggle_machine_shape") != "NvidiaTeslaT4":
        raise ShadowGoldContentReferenceError("content-reference CUDA resource changed")
    if runtime.get("expected_visible_gpu_count") != 2 or runtime.get("per_architecture_world_size") != 1:
        raise ShadowGoldContentReferenceError("content-reference GPU/world size changed")
    for key in ("distributed_training", "ddp_used", "nccl_used"):
        if runtime.get(key) is not False:
            raise ShadowGoldContentReferenceError(f"content-reference distributed guard changed: {key}")
    if runtime.get("automatic_compute_retries") != 0 or runtime.get("notebook_internet") is not False:
        raise ShadowGoldContentReferenceError("content-reference retry/internet contract changed")

    guards = config.get("scientific_guards") or {}
    false_keys = (
        "known_probe_boundary_used",
        "probe_text_used",
        "probe_local_score_used_as_candidate",
        "reference_fit_uses_labels",
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
            raise ShadowGoldContentReferenceError(f"content-reference scientific guard changed: {key}")
    if guards.get("competition_rows_used") != 0 or guards.get("external_rows_used") != 0:
        raise ShadowGoldContentReferenceError("content-reference external rows changed")


def _file_ngram_counts(payload: bytes, max_context: int) -> tuple[list[Counter], list[Counter]]:
    contexts = [Counter() for _ in range(max_context + 1)]
    joints = [Counter() for _ in range(max_context + 1)]
    for index, target in enumerate(payload):
        for width in range(min(max_context, index) + 1):
            context = payload[index - width : index] if width else b""
            contexts[width][context] += 1
            joints[width][(context, int(target))] += 1
    return contexts, joints


def build_content_reference(
    contents: Sequence[str],
    languages: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit global per-language byte n-gram counts without labels or IDs."""

    validate_content_reference_config(config)
    content_values = [str(value) for value in contents]
    language_values = [str(value) for value in languages]
    if not content_values or len(content_values) != len(language_values):
        raise ShadowGoldContentReferenceError("content-reference fit inputs are empty or misaligned")
    max_context = int(config["content_reference"]["max_context_bytes"])
    global_contexts: dict[str, list[Counter]] = {}
    global_joints: dict[str, list[Counter]] = {}
    rows_per_language: Counter = Counter()
    for content, language in zip(content_values, language_values):
        payload = content.encode("utf-8")
        if len(payload) < 2:
            raise ShadowGoldContentReferenceError("content-reference sample is shorter than two bytes")
        if language not in global_contexts:
            global_contexts[language] = [Counter() for _ in range(max_context + 1)]
            global_joints[language] = [Counter() for _ in range(max_context + 1)]
        local_contexts, local_joints = _file_ngram_counts(payload, max_context)
        for width in range(max_context + 1):
            global_contexts[language][width].update(local_contexts[width])
            global_joints[language][width].update(local_joints[width])
        rows_per_language[language] += 1
    if any(count < 2 for count in rows_per_language.values()):
        raise ShadowGoldContentReferenceError("content-reference requires at least two rows per language")
    return {
        "version": SHADOW_GOLD_CONTENT_REFERENCE_VERSION,
        "max_context_bytes": max_context,
        "global_contexts": global_contexts,
        "global_joints": global_joints,
        "rows_per_language": dict(sorted(rows_per_language.items())),
        "labels_used": False,
        "benchmark_ids_used": False,
        "probe_boundary_used": False,
    }


def score_loo_reference_logp(
    content: str,
    language: str,
    reference: Mapping[str, Any],
    config: Mapping[str, Any],
) -> np.ndarray:
    """Score every byte using counts from all same-language files except itself."""

    validate_content_reference_config(config)
    payload = str(content).encode("utf-8")
    language = str(language)
    max_context = int(config["content_reference"]["max_context_bytes"])
    alpha = float(config["content_reference"]["additive_alpha"])
    minimum_context = int(config["content_reference"]["minimum_context_count"])
    vocabulary = int(config["content_reference"]["byte_vocabulary_size"])
    if reference.get("version") != SHADOW_GOLD_CONTENT_REFERENCE_VERSION:
        raise ShadowGoldContentReferenceError("content-reference runtime version changed")
    if language not in reference["global_contexts"]:
        raise ShadowGoldContentReferenceError(f"content-reference language missing: {language}")
    local_contexts, local_joints = _file_ngram_counts(payload, max_context)
    output = np.empty(len(payload), dtype=np.float64)
    for index, target in enumerate(payload):
        selected_probability: float | None = None
        for width in range(min(max_context, index), -1, -1):
            context = payload[index - width : index] if width else b""
            global_context = int(reference["global_contexts"][language][width][context])
            local_context = int(local_contexts[width][context])
            loo_context = global_context - local_context
            if loo_context < 0:
                raise ShadowGoldContentReferenceError("content-reference leave-one-out context underflow")
            if width and loo_context < minimum_context:
                continue
            global_joint = int(reference["global_joints"][language][width][(context, int(target))])
            local_joint = int(local_joints[width][(context, int(target))])
            loo_joint = global_joint - local_joint
            if loo_joint < 0 or loo_joint > loo_context:
                raise ShadowGoldContentReferenceError("content-reference leave-one-out joint underflow")
            selected_probability = (loo_joint + alpha) / (loo_context + alpha * vocabulary)
            break
        if selected_probability is None or not 0.0 < selected_probability <= 1.0:
            raise ShadowGoldContentReferenceError("content-reference probability is invalid")
        output[index] = math.log(selected_probability)
    if not np.isfinite(output).all():
        raise ShadowGoldContentReferenceError("content-reference score is non-finite")
    return output


def _best_local(values: np.ndarray, width: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ShadowGoldContentReferenceError("content-reference local window is empty")
    if len(values) <= width:
        return float(values.mean())
    sums = np.convolve(values, np.ones(width, dtype=np.float64), mode="valid")
    return float((sums / width).max())


def _top_mean(values: np.ndarray, percent: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ShadowGoldContentReferenceError("content-reference top-tail input is empty")
    count = max(1, math.ceil(len(values) * percent / 100.0))
    indices = np.argpartition(values, len(values) - count)[-count:]
    return float(values[indices].mean())


def _make_target_records(source: pd.DataFrame, tokenizer: ByteCodeTokenizer, max_length: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample_index, content in enumerate(source.content.astype(str)):
        ids = tokenizer.encode(content, add_special_tokens=False)
        if len(ids) < 2:
            raise ShadowGoldContentReferenceError("content-reference target sample is shorter than two tokens")
        for position, start in stage2_core._window_starts(len(ids), max_length):
            window = ids[start : start + max_length]
            records.append(
                {
                    "sample_index": sample_index,
                    "position": position,
                    "window_start": start,
                    "file_token_count": len(ids),
                    "window_token_count": len(window),
                    "input_ids": window,
                }
            )
    return records


def score_content_reference_features(
    training_output_directory: str | Path,
    source: pd.DataFrame,
    reference: Mapping[str, Any],
    reference_logp: Sequence[np.ndarray],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score target/reference residuals without reading membership labels."""

    import torch

    validate_content_reference_config(config)
    required_source = ["benchmark_id", "language", "length_bin", "character_count", "content"]
    if any(column not in source.columns for column in required_source):
        raise ShadowGoldContentReferenceError("content-reference scoring bundle schema changed")
    if len(source) != len(reference_logp):
        raise ShadowGoldContentReferenceError("content-reference score rows are misaligned")
    checkpoint = load_shadow_checkpoint(training_output_directory, backend="cuda")
    if int(checkpoint.training_manifest.get("completed_steps", -1)) != 1280:
        raise ShadowGoldContentReferenceError("content-reference checkpoint step count changed")
    if checkpoint.training_manifest.get("evaluation_labels_read") is not False:
        raise ShadowGoldContentReferenceError("content-reference checkpoint label guard failed")
    if checkpoint.training_manifest.get("pretrained_weights_used") is not False:
        raise ShadowGoldContentReferenceError("content-reference checkpoint pretrained-weight guard failed")

    tokenizer = ByteCodeTokenizer()
    max_length = int(config["target_scoring"]["max_length"])
    records = _make_target_records(source, tokenizer, max_length)
    per_sample: list[list[dict[str, Any]]] = [[] for _ in range(len(source))]
    model = checkpoint.model
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in stage2_core._dynamic_batches(records, int(config["target_scoring"]["max_batch_tokens"])):
            width = max(int(row["window_token_count"]) for row in batch)
            input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(batch), width), dtype=torch.long, device=device)
            for row_index, record in enumerate(batch):
                ids = torch.as_tensor(record["input_ids"], dtype=torch.long, device=device)
                input_ids[row_index, : len(ids)] = ids
                attention_mask[row_index, : len(ids)] = 1
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = output.logits if hasattr(output, "logits") else output[0]
            shifted = logits[:, :-1].float()
            targets = input_ids[:, 1:]
            correct = torch.log_softmax(shifted, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            for row_index, record in enumerate(batch):
                valid = int(record["window_token_count"]) - 1
                if valid <= 0:
                    raise ShadowGoldContentReferenceError("content-reference target window has no predictable token")
                target_values = correct[row_index, :valid].detach().cpu().numpy().astype(np.float64, copy=False)
                sample_index = int(record["sample_index"])
                start = int(record["window_start"])
                ref_values = np.asarray(reference_logp[sample_index][start + 1 : start + 1 + valid], dtype=np.float64)
                if len(ref_values) != valid:
                    raise ShadowGoldContentReferenceError("content-reference target/reference token alignment failed")
                residual = target_values - ref_values
                if not np.isfinite(residual).all():
                    raise ShadowGoldContentReferenceError("content-reference residual is non-finite")
                per_sample[sample_index].append(
                    {
                        "position": record["position"],
                        "target": target_values,
                        "reference": ref_values,
                        "residual": residual,
                    }
                )
            del input_ids, attention_mask, output, logits, shifted, targets, correct

    features = source[_METADATA_COLUMNS].copy().reset_index(drop=True)
    for name in [*_CANDIDATES, "loss_max", "reference_logp_mean"]:
        features[name] = np.nan
    local_widths = [int(value) for value in config["target_scoring"]["local_widths"]]
    for sample_index, windows in enumerate(per_sample):
        if not windows:
            raise ShadowGoldContentReferenceError("content-reference sample has no scored windows")
        residual_all = np.concatenate([row["residual"] for row in windows])
        reference_all = np.concatenate([row["reference"] for row in windows])
        target_window_means = [float(row["target"].mean()) for row in windows]
        features.at[sample_index, "residual_mean"] = float(residual_all.mean())
        features.at[sample_index, "residual_top05_mean"] = _top_mean(residual_all, 5)
        features.at[sample_index, "residual_top10_mean"] = _top_mean(residual_all, 10)
        for width_value in local_widths:
            name = f"residual_local{width_value:02d}_max"
            features.at[sample_index, name] = max(_best_local(row["residual"], width_value) for row in windows)
        features.at[sample_index, "loss_max"] = max(target_window_means)
        features.at[sample_index, "reference_logp_mean"] = float(reference_all.mean())
    features = features[_FEATURE_COLUMNS]
    numeric = features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or features.benchmark_id.duplicated().any():
        raise ShadowGoldContentReferenceError("content-reference feature coverage/finiteness failed")
    manifest = {
        "status": "sealed",
        "version": SHADOW_GOLD_CONTENT_REFERENCE_VERSION,
        "architecture_slot": checkpoint.training_manifest.get("architecture_slot"),
        "rows": len(features),
        "candidate_features": list(_CANDIDATES),
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "model_state_sha256": checkpoint.model_state_sha256,
        "reference_version": reference.get("version"),
        "reference_rows_per_language": reference.get("rows_per_language"),
        "reference_fit_uses_labels": False,
        "reference_leave_one_file_out": True,
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
        raise ShadowGoldContentReferenceError("content-reference parent label seal is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("passed_to_gpu_children") is not False:
        raise ShadowGoldContentReferenceError("content-reference label boundary failed")
    if manifest.get("label_sha256") != gold_v1._sha256_file(path):
        raise ShadowGoldContentReferenceError("content-reference parent label hash mismatch")
    labels = pd.read_json(path, lines=True)
    return gold_v2.canonicalize_exact_schema(labels, gold_v1._LABEL_COLUMNS, artifact_name="Gold parent label")


def _select_candidate(development: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    metrics: dict[str, Any] = {}
    for name in _CANDIDATES:
        metrics[name] = evaluate_frozen_gold_holdout(
            development,
            development[name].to_numpy(dtype=np.float64),
            false_positive_rate=0.01,
        )["overall"]
    order = {name: index for index, name in enumerate(_CANDIDATES)}
    selected = max(
        _CANDIDATES,
        key=lambda name: (
            metrics[name]["partial_auc_standardized"],
            metrics[name]["tpr_at_fpr"],
            metrics[name]["auc"],
            -order[name],
        ),
    )
    return selected, metrics


def evaluate_content_reference(
    left_features: pd.DataFrame,
    right_features: pd.DataFrame,
    label_directory: str | Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Reveal labels after feature sealing, then select only on Shadow-A development."""

    validate_content_reference_config(config)
    if list(left_features.columns) != _FEATURE_COLUMNS or list(right_features.columns) != _FEATURE_COLUMNS:
        raise ShadowGoldContentReferenceError("content-reference feature schema changed")
    if left_features.benchmark_id.astype(str).tolist() != right_features.benchmark_id.astype(str).tolist():
        raise ShadowGoldContentReferenceError("content-reference left/right row order differs")
    labels = _load_labels(label_directory)
    left_labeled = attach_exact_labels(left_features, labels)
    right_labeled = attach_exact_labels(right_features, labels)
    left_development, left_holdout = split_development_holdout(left_labeled, seed=2027)
    _, right_holdout = split_development_holdout(right_labeled, seed=2027)
    holdout_ids = left_holdout.benchmark_id.astype(str).tolist()
    right_holdout = right_holdout.set_index(right_holdout.benchmark_id.astype(str), drop=False).loc[holdout_ids].reset_index(drop=True)
    selected, development_metrics = _select_candidate(left_development)

    holdout_candidates: dict[str, Any] = {"left": {}, "right": {}}
    for name in _CANDIDATES:
        holdout_candidates["left"][name] = evaluate_frozen_gold_holdout(
            left_holdout, left_holdout[name].to_numpy(dtype=np.float64), false_positive_rate=0.01
        )["overall"]
        holdout_candidates["right"][name] = evaluate_frozen_gold_holdout(
            right_holdout, right_holdout[name].to_numpy(dtype=np.float64), false_positive_rate=0.01
        )["overall"]
    baseline = {
        "left": evaluate_frozen_gold_holdout(left_holdout, left_holdout.loss_max.to_numpy(dtype=np.float64), false_positive_rate=0.01)["overall"],
        "right": evaluate_frozen_gold_holdout(right_holdout, right_holdout.loss_max.to_numpy(dtype=np.float64), false_positive_rate=0.01)["overall"],
    }
    reference_only = {
        "left": evaluate_frozen_gold_holdout(left_holdout, left_holdout.reference_logp_mean.to_numpy(dtype=np.float64), false_positive_rate=0.01)["overall"],
        "right": evaluate_frozen_gold_holdout(right_holdout, right_holdout.reference_logp_mean.to_numpy(dtype=np.float64), false_positive_rate=0.01)["overall"],
    }
    selected_holdout = {"left": holdout_candidates["left"][selected], "right": holdout_candidates["right"][selected]}
    gains = {
        side: float(selected_holdout[side]["auc"] - baseline[side]["auc"])
        for side in ("left", "right")
    }
    recovery = config["recovery_criterion"]
    lower = float(recovery["reference_only_auc_lower_bound"])
    upper = float(recovery["reference_only_auc_upper_bound"])
    checks = {
        "left_min_auc": selected_holdout["left"]["auc"] >= float(recovery["minimum_holdout_auc_each_architecture"]),
        "right_min_auc": selected_holdout["right"]["auc"] >= float(recovery["minimum_holdout_auc_each_architecture"]),
        "left_min_gain": gains["left"] >= float(recovery["minimum_auc_gain_vs_baseline_each_architecture"]),
        "right_min_gain": gains["right"] >= float(recovery["minimum_auc_gain_vs_baseline_each_architecture"]),
        "left_min_tpr": selected_holdout["left"]["tpr_at_fpr"] >= float(recovery["minimum_tpr_at_1pct_fpr_each_architecture"]),
        "right_min_tpr": selected_holdout["right"]["tpr_at_fpr"] >= float(recovery["minimum_tpr_at_1pct_fpr_each_architecture"]),
        "left_reference_control": lower <= reference_only["left"]["auc"] <= upper,
        "right_reference_control": lower <= reference_only["right"]["auc"] <= upper,
    }
    passed = all(checks.values())
    result = {
        "status": "complete",
        "version": SHADOW_GOLD_CONTENT_REFERENCE_VERSION,
        "selected_candidate": selected,
        "development_candidate_metrics": development_metrics,
        "holdout_candidate_metrics": holdout_candidates,
        "selected_holdout": selected_holdout,
        "baseline_loss_max_holdout": baseline,
        "reference_only_holdout": reference_only,
        "selected_auc_gain_vs_loss_max": gains,
        "recovery_checks": checks,
        "recovery_passed": passed,
        "selected_on_shadow_a_development_only": True,
        "applied_unchanged_to_shadow_b": True,
        "reference_fit_uses_labels": False,
        "reference_leave_one_file_out": True,
        "probe_boundary_used": False,
        "probe_text_used": False,
        "stage2_v3_selection_allowed": False,
        "competition_feature_promotion_allowed": False,
    }
    predictions = left_holdout[["benchmark_id", "language", "length_bin", "character_count", "membership", "matched_pair_id"]].copy()
    predictions["selected_candidate"] = selected
    predictions["left_selected_score"] = left_holdout[selected].to_numpy(dtype=np.float64)
    predictions["right_selected_score"] = right_holdout[selected].to_numpy(dtype=np.float64)
    predictions["left_loss_max"] = left_holdout.loss_max.to_numpy(dtype=np.float64)
    predictions["right_loss_max"] = right_holdout.loss_max.to_numpy(dtype=np.float64)
    predictions["left_reference_logp_mean"] = left_holdout.reference_logp_mean.to_numpy(dtype=np.float64)
    predictions["right_reference_logp_mean"] = right_holdout.reference_logp_mean.to_numpy(dtype=np.float64)
    return result, predictions


def run_content_reference_benchmark(
    *,
    scratch_directory: str | Path,
    positive_control_config: Mapping[str, Any],
    content_reference_config: Mapping[str, Any],
    timeout_seconds: int = 5400,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Repeat frozen target training, seal calibrated features, then reveal Gold labels."""

    positive_v1.validate_positive_control_config(positive_control_config)
    validate_content_reference_config(content_reference_config)
    names = visible_cuda_device_names()
    validate_gpu_inventory(names, expected_count=2, required_name_fragment="T4")
    scratch = Path(scratch_directory).resolve()
    if scratch.exists() and any(scratch.iterdir()):
        raise ShadowGoldContentReferenceError("content-reference scratch directory is not empty")
    scratch.mkdir(parents=True, exist_ok=True)
    prepared = positive_v2.prepare_positive_control_runtime(scratch / "prepared", positive_control_config)
    dual = run_dual_gpu_shadow(
        prepared["protocol_directory"],
        scratch / "training",
        max_steps=1280,
        checkpoint_reload_atol=1e-5,
        parameter_sync_atol=1e-5,
        expected_gpu_count=2,
        required_name_fragment="T4",
        timeout_seconds=timeout_seconds,
    )

    bundle = load_scoring_bundle(prepared["scoring_bundle_directory"])
    source = bundle.frame.reset_index(drop=True)
    reference = build_content_reference(source.content.astype(str).tolist(), source.language.astype(str).tolist(), content_reference_config)
    reference_logp = [
        score_loo_reference_logp(content, language, reference, content_reference_config)
        for content, language in zip(source.content.astype(str), source.language.astype(str))
    ]
    left_features, left_manifest = score_content_reference_features(
        scratch / "training" / "left", source, reference, reference_logp, content_reference_config
    )
    right_features, right_manifest = score_content_reference_features(
        scratch / "training" / "right", source, reference, reference_logp, content_reference_config
    )
    seal_dir = scratch / "content_reference_feature_seal"
    seal_dir.mkdir()
    left_path = seal_dir / "left_content_reference_features.jsonl"
    right_path = seal_dir / "right_content_reference_features.jsonl"
    gold_v1._write_jsonl(left_features, left_path, list(left_features.columns))
    gold_v1._write_jsonl(right_features, right_path, list(right_features.columns))
    seal = {
        "status": "sealed",
        "left_sha256": gold_v1._sha256_file(left_path),
        "right_sha256": gold_v1._sha256_file(right_path),
        "left_manifest": left_manifest,
        "right_manifest": right_manifest,
        "reference_version": reference["version"],
        "reference_rows_per_language": reference["rows_per_language"],
        "reference_fit_uses_labels": False,
        "reference_leave_one_file_out": True,
        "membership_labels_read": False,
        "probe_boundary_used": False,
        "probe_text_used": False,
        "stage2_v3_selection_allowed": False,
    }
    gold_v1._write_json(seal_dir / "content_reference_feature_seal.json", seal)
    evaluation, predictions = evaluate_content_reference(
        left_features, right_features, prepared["label_directory"], content_reference_config
    )
    manifest = {
        "status": "complete",
        "execution_id": EXECUTION_ID,
        "version": SHADOW_GOLD_CONTENT_REFERENCE_VERSION,
        "gpu_names": names,
        "prepared": prepared,
        "dual_gpu_training": dual,
        "content_reference_feature_seal": seal,
        "selected_candidate": evaluation["selected_candidate"],
        "recovery_passed": evaluation["recovery_passed"],
        "training_protocol_changed_from_positive_control_v2": False,
        "reference_fit_uses_labels": False,
        "reference_leave_one_file_out": True,
        "features_sealed_before_label_reveal": True,
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
    "EXECUTION_ID",
    "SHADOW_GOLD_CONTENT_REFERENCE_VERSION",
    "ShadowGoldContentReferenceError",
    "build_content_reference",
    "evaluate_content_reference",
    "run_content_reference_benchmark",
    "score_content_reference_features",
    "score_loo_reference_logp",
    "validate_content_reference_config",
]
