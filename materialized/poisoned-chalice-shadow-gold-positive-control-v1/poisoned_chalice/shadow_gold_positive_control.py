"""Sensitivity calibration for the controlled Gold membership benchmark.

The architecture-transfer Gold run was a valid negative result: the frozen
primitive suite remained near chance despite exact sample-level membership.  This
module therefore does not tune an attack.  Instead it asks a narrower question:
can the controlled infrastructure detect deliberately stronger memorisation when
all samples receive a membership-independent, high-entropy suffix probe and
training repeatedly exposes that suffix?

The original controlled split is constructed before probe injection.  Probe
values depend only on original source content, never membership or benchmark
IDs, and are appended to both member and non-member rows.  Only member rows enter
training, as defined by the pre-existing controlled split.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .controlled_transfer import ControlledTransferConfig, build_controlled_transfer_benchmark, make_exposure_schedule
from .shadow_gold_corpus import SHADOW_GOLD_LANGUAGES, build_shadow_gold_corpus, shadow_gold_corpus_sha256
from . import shadow_gold_kaggle as gold_v1
from . import shadow_gold_kaggle_v2 as gold_v2
from .shadow_protocol import (
    ByteCodeTokenizer,
    ShadowSequenceConfig,
    ShadowTrainingSpec,
    build_shadow_protocol_manifest,
    build_shadow_training_sequences,
    default_shadow_pair,
)
from .shadow_training_dual_gpu import validate_gpu_inventory, visible_cuda_device_names


SHADOW_GOLD_POSITIVE_CONTROL_VERSION = "shadow-gold-positive-control-v1"
EXPERIMENT_ID = "shadow-gold-positive-control-v1"
PROBE_VERSION = "content-hash-comment-probe-v1"
_PROBE_HEX_CHARACTERS = 16


class ShadowGoldPositiveControlError(RuntimeError):
    """Raised when a frozen positive-control invariant fails."""


def _probe_digest(content: str) -> str:
    return hashlib.blake2b(content.encode("utf-8"), digest_size=8).hexdigest()


def positive_control_probe(content: str, language: str) -> str:
    """Return a fixed-width, membership-independent suffix comment."""

    digest = _probe_digest(str(content))
    if len(digest) != _PROBE_HEX_CHARACTERS:
        raise RuntimeError("positive-control digest width changed")
    prefix = "# gold_probe_" if str(language) in {"Python", "Ruby"} else "// gold_probe_"
    return f"\n{prefix}{digest}\n"


def _length_bins(character_count: pd.Series, edges: tuple[int, ...]) -> np.ndarray:
    values = character_count.to_numpy(dtype=np.int64)
    return np.searchsorted(np.asarray(edges, dtype=np.int64), values, side="right") - 1


def inject_positive_control_probe(frame: pd.DataFrame, *, length_bin_edges: tuple[int, ...]) -> pd.DataFrame:
    """Append the content-derived probe while preserving the frozen length stratum."""

    required = {"benchmark_id", "content", "language", "character_count", "length_bin"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ShadowGoldPositiveControlError(f"positive-control frame missing columns: {missing}")
    result = frame.copy()
    original_content = result.content.astype(str).tolist()
    languages = result.language.astype(str).tolist()
    result["content"] = [
        content + positive_control_probe(content, language)
        for content, language in zip(original_content, languages)
    ]
    result["character_count"] = result.content.str.len().astype(int)
    post_bins = _length_bins(result.character_count, length_bin_edges)
    pre_bins = frame.length_bin.to_numpy(dtype=np.int64)
    if not np.array_equal(post_bins, pre_bins):
        changed = int(np.sum(post_bins != pre_bins))
        raise ShadowGoldPositiveControlError(
            f"positive-control probe changed frozen length strata for {changed} rows"
        )
    result["length_bin"] = post_bins.astype(int)
    if result.benchmark_id.astype(str).tolist() != frame.benchmark_id.astype(str).tolist():
        raise ShadowGoldPositiveControlError("positive-control benchmark IDs changed")
    return result


def build_suffix_exposure_schedule(
    train_corpus: pd.DataFrame,
    *,
    repeats: int = 16,
    seed: int = 2027,
) -> pd.DataFrame:
    """Build balanced repeats whose window-policy round always resolves to suffix."""

    base = make_exposure_schedule(train_corpus, repeats, seed=seed)
    repeat_index = base.exposure_round.to_numpy(dtype=np.int64)
    base = base.copy()
    base["exposure_round"] = 2 + 4 * repeat_index
    if not (base.exposure_round.to_numpy(dtype=np.int64) % 4 == 2).all():
        raise ShadowGoldPositiveControlError("positive-control schedule is not suffix-only")
    counts = base.groupby("benchmark_id").size()
    if not counts.eq(repeats).all():
        raise ShadowGoldPositiveControlError("positive-control exposure counts are not balanced")
    return base


def validate_positive_control_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("role") != "gold_positive_control_calibration":
        raise ShadowGoldPositiveControlError("positive-control identity changed")
    data = config.get("data") or {}
    expected_data = {
        "base_corpus": "shadow-gold-code-like-corpus-v1",
        "languages": list(SHADOW_GOLD_LANGUAGES),
        "candidate_rows_per_language": 2048,
        "candidate_rows_total": 10240,
        "synthetic_family_size": 4,
        "competition_rows_used": 0,
        "external_rows_used": 0,
        "preexisting_membership_labels_present": False,
    }
    for key, value in expected_data.items():
        if data.get(key) != value:
            raise ShadowGoldPositiveControlError(f"positive-control data contract changed: {key}")
    split = config.get("controlled_split") or {}
    expected_split = {
        "seed": 2027,
        "group_column": "synthetic_family_id",
        "member_fraction": 0.5,
        "eval_rows_per_language_per_class": 512,
        "min_eval_rows_per_language_per_class": 512,
        "expected_evaluation_rows": 5120,
        "expected_evaluation_member_rows": 2560,
        "expected_evaluation_nonmember_rows": 2560,
        "expected_training_corpus_rows": 5120,
        "split_is_built_before_probe_injection": True,
    }
    for key, value in expected_split.items():
        if split.get(key) != value:
            raise ShadowGoldPositiveControlError(f"positive-control split contract changed: {key}")
    probe = config.get("positive_control_probe") or {}
    expected_probe = {
        "version": PROBE_VERSION,
        "digest": "blake2b-64bit-of-original-content",
        "hex_characters": 16,
        "placement": "suffix_comment",
        "applied_to_members": True,
        "applied_to_nonmembers": True,
        "membership_used_to_construct_probe": False,
        "benchmark_id_used_to_construct_probe": False,
        "source_content_used_to_construct_probe": True,
        "post_injection_length_bin_must_equal_preinjection_length_bin": True,
    }
    for key, value in expected_probe.items():
        if probe.get(key) != value:
            raise ShadowGoldPositiveControlError(f"positive-control probe contract changed: {key}")
    protocol = config.get("protocol") or {}
    expected_protocol = {
        "max_sequence_tokens": 256,
        "seed": 2027,
        "global_batch_size": 64,
        "learning_rate": 0.0003,
        "weight_decay": 0.1,
        "warmup_fraction": 0.05,
        "gradient_clip_norm": 1.0,
        "exposure_repeats": 16,
        "forced_training_window": "suffix",
        "exposure_round_mapping": "2_plus_4_times_repeat_index",
        "expected_training_sequences": 81920,
        "expected_optimizer_steps_per_architecture": 1280,
        "architecture_pair": ["shadow-gpt2-byte-5m", "shadow-llama-byte-5m"],
        "random_initialisation": True,
        "pretrained_weights_used": False,
    }
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise ShadowGoldPositiveControlError(f"positive-control protocol changed: {key}")
    runtime = config.get("runtime") or {}
    expected_runtime = {
        "backend": "cuda",
        "kaggle_machine_shape": "NvidiaTeslaT4",
        "expected_visible_gpu_count": 2,
        "architecture_gpu_assignment": {"left": 0, "right": 1},
        "per_architecture_world_size": 1,
        "distributed_training": False,
        "ddp_used": False,
        "nccl_used": False,
        "run_architectures_concurrently": True,
        "checkpoint_reload_atol": 1e-5,
        "automatic_compute_retries": 0,
        "notebook_internet": False,
    }
    for key, value in expected_runtime.items():
        if runtime.get(key) != value:
            raise ShadowGoldPositiveControlError(f"positive-control runtime changed: {key}")
    scoring = config.get("scoring") or {}
    if scoring.get("primary_sensitivity_predictor") != "loss" or scoring.get("primitive_suite_unchanged_from_gold_v2") is not True:
        raise ShadowGoldPositiveControlError("positive-control scoring contract changed")
    gate = config.get("sensitivity_gate") or {}
    expected_gate = {
        "primary_predictor": "loss",
        "shadow_a_holdout_min_auc": 0.60,
        "shadow_b_blind_holdout_min_auc": 0.60,
        "shadow_a_holdout_min_tpr_at_1pct_fpr": 0.03,
        "shadow_b_blind_holdout_min_tpr_at_1pct_fpr": 0.03,
        "all_four_thresholds_required": True,
        "secondary_predictors_may_rescue_failed_primary_gate": False,
        "failed_gate_is_valid_scientific_result_not_runtime_failure": True,
    }
    for key, value in expected_gate.items():
        if gate.get(key) != value:
            raise ShadowGoldPositiveControlError(f"positive-control sensitivity gate changed: {key}")
    guards = config.get("scientific_guards") or {}
    for key in ("stage2_v3_selection_allowed", "external_model_holdout_consumed", "fourth_external_holdout_consumed"):
        if guards.get(key) is not False:
            raise ShadowGoldPositiveControlError(f"positive-control scientific guard changed: {key}")


def prepare_positive_control_runtime(root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare a label-free suffix-probe training/scoring runtime."""

    validate_positive_control_config(config)
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ShadowGoldPositiveControlError("positive-control runtime root is not empty")
    root.mkdir(parents=True, exist_ok=True)
    protocol_dir = root / "protocol"
    scoring_dir = root / "scoring_bundle"
    label_dir = root / "labels_parent_only"
    protocol_dir.mkdir()
    scoring_dir.mkdir()
    label_dir.mkdir()

    data_cfg = config["data"]
    split_cfg = config["controlled_split"]
    protocol_cfg = config["protocol"]
    edges = tuple(int(value) for value in split_cfg["length_bin_edges"])
    corpus = build_shadow_gold_corpus(
        rows_per_language=int(data_cfg["candidate_rows_per_language"]),
        family_size=int(data_cfg["synthetic_family_size"]),
        seed=int(split_cfg["seed"]),
    )
    base_corpus_hash = shadow_gold_corpus_sha256(corpus)
    controlled = build_controlled_transfer_benchmark(
        corpus,
        ControlledTransferConfig(
            content_column="content",
            language_column="language",
            group_column=str(split_cfg["group_column"]),
            seed=int(split_cfg["seed"]),
            member_fraction=float(split_cfg["member_fraction"]),
            eval_rows_per_language_per_class=int(split_cfg["eval_rows_per_language_per_class"]),
            min_eval_rows_per_language_per_class=int(split_cfg["min_eval_rows_per_language_per_class"]),
            min_characters=int(split_cfg["min_characters"]),
            max_characters=int(split_cfg["max_characters"]),
            length_bin_edges=edges,
            near_duplicate_threshold=float(split_cfg["near_duplicate_threshold"]),
        ),
    )
    if len(controlled.train_corpus) != int(split_cfg["expected_training_corpus_rows"]):
        raise ShadowGoldPositiveControlError("positive-control training row count changed")
    if len(controlled.evaluation) != int(split_cfg["expected_evaluation_rows"]):
        raise ShadowGoldPositiveControlError("positive-control evaluation row count changed")

    train_corpus = inject_positive_control_probe(controlled.train_corpus, length_bin_edges=edges)
    evaluation = inject_positive_control_probe(controlled.evaluation, length_bin_edges=edges)
    train_lookup = train_corpus.set_index(train_corpus.benchmark_id.astype(str)).content.astype(str)
    eval_members = evaluation[evaluation.membership.astype(int) == 1].copy()
    eval_nonmembers = evaluation[evaluation.membership.astype(int) == 0].copy()
    if not set(eval_members.benchmark_id.astype(str)) <= set(train_lookup.index):
        raise ShadowGoldPositiveControlError("positive-control member IDs left the training corpus")
    for row in eval_members.itertuples(index=False):
        if train_lookup.loc[str(row.benchmark_id)] != str(row.content):
            raise ShadowGoldPositiveControlError("positive-control member content differs between training and evaluation")
    if set(eval_nonmembers.benchmark_id.astype(str)) & set(train_lookup.index):
        raise ShadowGoldPositiveControlError("positive-control nonmember leaked into training")

    schedule = build_suffix_exposure_schedule(
        train_corpus,
        repeats=int(protocol_cfg["exposure_repeats"]),
        seed=int(protocol_cfg["seed"]),
    )
    if len(schedule) != int(protocol_cfg["expected_training_sequences"]):
        raise ShadowGoldPositiveControlError("positive-control training sequence count changed")
    sequence_config = ShadowSequenceConfig(max_sequence_tokens=int(protocol_cfg["max_sequence_tokens"]))
    tokenizer = ByteCodeTokenizer()
    training_spec = ShadowTrainingSpec(
        seed=int(protocol_cfg["seed"]),
        learning_rate=float(protocol_cfg["learning_rate"]),
        weight_decay=float(protocol_cfg["weight_decay"]),
        warmup_fraction=float(protocol_cfg["warmup_fraction"]),
        global_batch_size=int(protocol_cfg["global_batch_size"]),
        gradient_clip_norm=float(protocol_cfg["gradient_clip_norm"]),
    )
    sequences = build_shadow_training_sequences(
        train_corpus,
        schedule,
        tokenizer=tokenizer,
        sequence_config=sequence_config,
        seed=int(protocol_cfg["seed"]),
    )
    if set(sequences.window_name.astype(str)) != {"suffix"}:
        raise ShadowGoldPositiveControlError("positive-control materialized sequence windows are not suffix-only")
    left, right = default_shadow_pair(max_position_embeddings=int(protocol_cfg["max_sequence_tokens"]))
    protocol_manifest = build_shadow_protocol_manifest(
        left=left,
        right=right,
        sequence_config=sequence_config,
        training_spec=training_spec,
        training_sequences=sequences,
        tokenizer=tokenizer,
        max_parameter_ratio=1.15,
    )
    input_ids = np.asarray(sequences.input_ids.tolist(), dtype=np.uint16)
    expected_shape = (
        int(protocol_cfg["expected_training_sequences"]),
        int(protocol_cfg["max_sequence_tokens"]),
    )
    if input_ids.shape != expected_shape:
        raise ShadowGoldPositiveControlError(f"positive-control protocol array shape changed: {input_ids.shape}")
    input_path = protocol_dir / "training_input_ids.npy"
    with input_path.open("wb") as handle:
        np.save(handle, input_ids, allow_pickle=False)
    metadata_path = protocol_dir / "training_sequence_metadata.jsonl"
    gold_v1._write_jsonl(
        sequences,
        metadata_path,
        ["benchmark_id", "exposure_round", "sequence_index", "source_token_count", "window_start", "window_name", "selected_payload_tokens"],
    )
    tokenizer.save_pretrained(protocol_dir)
    protocol_record = {
        "status": "frozen",
        "operation": "freeze_gold_positive_control_protocol",
        "experiment_id": EXPERIMENT_ID,
        "positive_control_version": SHADOW_GOLD_POSITIVE_CONTROL_VERSION,
        "probe_version": PROBE_VERSION,
        "base_corpus_sha256": base_corpus_hash,
        "protocol": protocol_manifest,
        "training_input_shape": list(input_ids.shape),
        "training_input_dtype": str(input_ids.dtype),
        "forced_training_window": "suffix",
        "exposure_repeats": int(protocol_cfg["exposure_repeats"]),
        "output_sha256": {
            "training_input_ids.npy": gold_v1._sha256_file(input_path),
            "training_sequence_metadata.jsonl": gold_v1._sha256_file(metadata_path),
            "byte_tokenizer.json": gold_v1._sha256_file(protocol_dir / "byte_tokenizer.json"),
        },
        "model_compute_started": False,
        "kaggle_operation_performed": False,
    }
    gold_v1._write_json(protocol_dir / "shadow_protocol_manifest.json", protocol_record)

    input_columns = ["benchmark_id", "content", "language", "character_count", "length_bin", "cluster_id"]
    label_columns = list(gold_v1._LABEL_COLUMNS)
    evaluation_input_path = scoring_dir / "evaluation_inputs.jsonl"
    label_path = label_dir / "evaluation_labels.jsonl"
    gold_v1._write_jsonl(evaluation, evaluation_input_path, input_columns)
    gold_v1._write_jsonl(evaluation, label_path, label_columns)
    input_hash = gold_v1._sha256_file(evaluation_input_path)
    label_hash = gold_v1._sha256_file(label_path)
    gold_v1._write_json(
        scoring_dir / "scoring_bundle_manifest.json",
        {
            "status": "sealed",
            "operation": "export_label_free_gold_positive_control_scoring_bundle",
            "evaluation_input_file": evaluation_input_path.name,
            "evaluation_input_sha256": input_hash,
            "withheld_label_sha256": label_hash,
            "payload_files_present": [evaluation_input_path.name],
            "membership_labels_present": False,
            "matched_pair_ids_present": False,
            "probe_version": PROBE_VERSION,
            "model_compute_started": False,
            "kaggle_operation_performed": False,
        },
    )
    gold_v1._write_json(
        label_dir / "label_manifest.json",
        {
            "status": "sealed_parent_only",
            "rows": len(evaluation),
            "membership_rows": int((evaluation.membership.astype(int) == 1).sum()),
            "nonmembership_rows": int((evaluation.membership.astype(int) == 0).sum()),
            "label_sha256": label_hash,
            "scoring_input_sha256": input_hash,
            "passed_to_gpu_children": False,
        },
    )
    expected_steps = len(schedule) // int(protocol_cfg["global_batch_size"])
    if expected_steps != int(protocol_cfg["expected_optimizer_steps_per_architecture"]):
        raise ShadowGoldPositiveControlError("positive-control optimizer-step count changed")
    return {
        "status": "prepared",
        "version": SHADOW_GOLD_POSITIVE_CONTROL_VERSION,
        "base_corpus_sha256": base_corpus_hash,
        "probe_version": PROBE_VERSION,
        "candidate_rows": len(corpus),
        "train_rows": len(train_corpus),
        "evaluation_rows": len(evaluation),
        "training_sequences": len(schedule),
        "optimizer_steps_per_architecture": expected_steps,
        "forced_training_window": "suffix",
        "evaluation_input_sha256": input_hash,
        "evaluation_label_sha256": label_hash,
        "protocol_directory": str(protocol_dir),
        "scoring_bundle_directory": str(scoring_dir),
        "label_directory": str(label_dir),
        "evaluation_labels_passed_to_children": False,
        "competition_rows_used": 0,
        "external_rows_used": 0,
    }


def evaluate_sensitivity_gate(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the predeclared plain-Loss sensitivity gate without tuning."""

    gate = config["sensitivity_gate"]
    primitive = metrics["primitive_holdout_overall"]
    left = primitive["left"]["loss"]
    right = primitive["right"]["loss"]
    checks = {
        "shadow_a_auc": float(left["auc"]) >= float(gate["shadow_a_holdout_min_auc"]),
        "shadow_b_auc": float(right["auc"]) >= float(gate["shadow_b_blind_holdout_min_auc"]),
        "shadow_a_tpr_at_1pct_fpr": float(left["tpr_at_fpr"]) >= float(gate["shadow_a_holdout_min_tpr_at_1pct_fpr"]),
        "shadow_b_tpr_at_1pct_fpr": float(right["tpr_at_fpr"]) >= float(gate["shadow_b_blind_holdout_min_tpr_at_1pct_fpr"]),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "primary_predictor": "loss",
        "checks": checks,
        "observed": {
            "shadow_a_auc": float(left["auc"]),
            "shadow_b_auc": float(right["auc"]),
            "shadow_a_tpr_at_1pct_fpr": float(left["tpr_at_fpr"]),
            "shadow_b_tpr_at_1pct_fpr": float(right["tpr_at_fpr"]),
        },
        "thresholds": {
            "shadow_a_auc": float(gate["shadow_a_holdout_min_auc"]),
            "shadow_b_auc": float(gate["shadow_b_blind_holdout_min_auc"]),
            "shadow_a_tpr_at_1pct_fpr": float(gate["shadow_a_holdout_min_tpr_at_1pct_fpr"]),
            "shadow_b_tpr_at_1pct_fpr": float(gate["shadow_b_blind_holdout_min_tpr_at_1pct_fpr"]),
        },
        "secondary_predictors_can_rescue": False,
        "failed_gate_is_valid_scientific_result": True,
    }


def run_positive_control_benchmark(
    *,
    scratch_directory: str | Path,
    config: Mapping[str, Any],
    timeout_seconds: int = 5400,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Run the frozen Gold sensitivity calibration on two T4s."""

    validate_positive_control_config(config)
    names = visible_cuda_device_names()
    validate_gpu_inventory(names, expected_count=2, required_name_fragment="T4")
    scratch = Path(scratch_directory).resolve()
    if scratch.exists() and any(scratch.iterdir()):
        raise ShadowGoldPositiveControlError("positive-control scratch directory is not empty")
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
    sensitivity = evaluate_sensitivity_gate(metrics, config)
    metrics = dict(metrics)
    metrics["positive_control_version"] = SHADOW_GOLD_POSITIVE_CONTROL_VERSION
    metrics["positive_control_sensitivity_gate"] = sensitivity
    metrics["stage2_v3_selection_allowed"] = False
    metrics["external_model_holdout_consumed"] = False
    manifest = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "version": SHADOW_GOLD_POSITIVE_CONTROL_VERSION,
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
    "EXPERIMENT_ID",
    "PROBE_VERSION",
    "SHADOW_GOLD_POSITIVE_CONTROL_VERSION",
    "ShadowGoldPositiveControlError",
    "build_suffix_exposure_schedule",
    "evaluate_sensitivity_gate",
    "inject_positive_control_probe",
    "positive_control_probe",
    "prepare_positive_control_runtime",
    "run_positive_control_benchmark",
    "validate_positive_control_config",
]
