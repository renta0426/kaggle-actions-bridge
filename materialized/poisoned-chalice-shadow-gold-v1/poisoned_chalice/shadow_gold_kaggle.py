"""End-to-end isolated T4 x2 runtime for the Gold shadow benchmark.

The parent process constructs the exact controlled benchmark and keeps membership
labels outside the child-GPU boundary.  Each child receives only the frozen
training protocol plus a label-free scoring bundle, trains one architecture on
one visible T4, and emits content-free primitive features.  The parent joins
labels only after both feature artifacts have been sealed, freezes an attack on
Shadow-A development pairs, and applies it unchanged to paired A/B holdouts.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .controlled_transfer import ControlledTransferConfig, build_controlled_transfer_benchmark, make_exposure_schedule
from .shadow_gold_corpus import SHADOW_GOLD_LANGUAGES, build_shadow_gold_corpus, shadow_gold_corpus_sha256
from .shadow_gold_transfer import (
    GOLD_PRIMITIVES,
    GoldScoringConfig,
    apply_frozen_gold_attack,
    attach_exact_labels,
    evaluate_frozen_gold_holdout,
    fit_frozen_gold_attack,
    frozen_attack_to_dict,
    score_gold_shadow_features,
)
from .shadow_protocol import (
    ByteCodeTokenizer,
    ShadowSequenceConfig,
    ShadowTrainingSpec,
    build_shadow_protocol_manifest,
    build_shadow_training_sequences,
    default_shadow_pair,
)
from .shadow_training import ShadowRuntimeConfig, train_shadow_model
from .shadow_training_dual_gpu import validate_gpu_inventory, visible_cuda_device_names


SHADOW_GOLD_KAGGLE_RUNTIME_VERSION = "shadow-gold-kaggle-t4x2-v1"
EXPERIMENT_ID = "shadow-gold-architecture-transfer-v1"
_FEATURE_COLUMNS = [
    "benchmark_id",
    "language",
    "length_bin",
    "character_count",
    "loss",
    "min_kpp",
    "local_64",
    "log_rank",
    "stage2_v1",
]
_LABEL_COLUMNS = ["benchmark_id", "membership", "matched_pair_id"]


class ShadowGoldKaggleError(RuntimeError):
    """Raised when a frozen Gold execution or clean-room invariant fails."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ShadowGoldKaggleError(f"JSONL output columns missing: {missing}")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in frame[columns].itertuples(index=False, name=None):
            record = {column: value for column, value in zip(columns, row)}
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=_json_default,
                )
                + "\n"
            )


def validate_gold_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("role") != "gold_controlled_benchmark":
        raise ShadowGoldKaggleError("Gold experiment identity changed")
    data = config.get("data") or {}
    if data.get("languages") != list(SHADOW_GOLD_LANGUAGES):
        raise ShadowGoldKaggleError("Gold language set/order changed")
    if data.get("candidate_rows_per_language") != 2048 or data.get("candidate_rows_total") != 10240:
        raise ShadowGoldKaggleError("Gold candidate scale changed")
    if data.get("synthetic_family_size") != 4 or data.get("competition_rows_used") != 0:
        raise ShadowGoldKaggleError("Gold corpus family/competition guard changed")
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
    }
    for key, value in expected_split.items():
        if split.get(key) != value:
            raise ShadowGoldKaggleError(f"Gold controlled split changed: {key}")
    protocol = config.get("protocol") or {}
    expected_protocol = {
        "max_sequence_tokens": 256,
        "seed": 2027,
        "global_batch_size": 64,
        "learning_rate": 0.0003,
        "weight_decay": 0.1,
        "warmup_fraction": 0.05,
        "gradient_clip_norm": 1.0,
        "exposure_repeats": 4,
        "expected_training_sequences": 20480,
        "expected_optimizer_steps_per_architecture": 320,
    }
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise ShadowGoldKaggleError(f"Gold training protocol changed: {key}")
    runtime = config.get("runtime") or {}
    if runtime.get("backend") != "cuda" or runtime.get("expected_visible_gpu_count") != 2:
        raise ShadowGoldKaggleError("Gold CUDA inventory contract changed")
    if runtime.get("architecture_gpu_assignment") != {"left": 0, "right": 1}:
        raise ShadowGoldKaggleError("Gold GPU assignment changed")
    if runtime.get("per_architecture_world_size") != 1:
        raise ShadowGoldKaggleError("Gold per-architecture world size changed")
    if any(runtime.get(key) is not False for key in ("distributed_training", "ddp_used", "nccl_used")):
        raise ShadowGoldKaggleError("distributed training became enabled")
    if runtime.get("checkpoint_reload_atol") != 1e-5 or runtime.get("automatic_compute_retries") != 0:
        raise ShadowGoldKaggleError("Gold fidelity/retry contract changed")
    transfer = config.get("transfer_protocol") or {}
    if transfer.get("fit_architecture_slot") != "left" or transfer.get("blind_transfer_architecture_slot") != "right":
        raise ShadowGoldKaggleError("Gold transfer direction changed")
    guards = config.get("scientific_guards") or {}
    if guards.get("stage2_v3_selection_allowed_from_this_benchmark_alone") is not False:
        raise ShadowGoldKaggleError("Gold Stage2-v3 guard changed")


def prepare_gold_runtime(root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Build protocol, label-free scoring bundle, and parent-only exact labels."""

    validate_gold_config(config)
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ShadowGoldKaggleError("Gold runtime root already exists and is not empty")
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
    corpus = build_shadow_gold_corpus(
        rows_per_language=int(data_cfg["candidate_rows_per_language"]),
        family_size=int(data_cfg["synthetic_family_size"]),
        seed=int(split_cfg["seed"]),
    )
    corpus_hash = shadow_gold_corpus_sha256(corpus)
    controlled_config = ControlledTransferConfig(
        content_column="content",
        language_column="language",
        group_column=str(split_cfg["group_column"]),
        seed=int(split_cfg["seed"]),
        member_fraction=float(split_cfg["member_fraction"]),
        eval_rows_per_language_per_class=int(split_cfg["eval_rows_per_language_per_class"]),
        min_eval_rows_per_language_per_class=int(split_cfg["min_eval_rows_per_language_per_class"]),
        min_characters=int(split_cfg["min_characters"]),
        max_characters=int(split_cfg["max_characters"]),
        length_bin_edges=tuple(int(value) for value in split_cfg["length_bin_edges"]),
        near_duplicate_threshold=float(split_cfg["near_duplicate_threshold"]),
    )
    benchmark = build_controlled_transfer_benchmark(corpus, controlled_config)
    if len(benchmark.train_corpus) != int(split_cfg["expected_training_corpus_rows"]):
        raise ShadowGoldKaggleError("Gold training corpus row count changed")
    evaluation = benchmark.evaluation.reset_index(drop=True)
    if len(evaluation) != int(split_cfg["expected_evaluation_rows"]):
        raise ShadowGoldKaggleError("Gold evaluation row count changed")
    counts = evaluation.groupby(["language", "membership"]).size().to_dict()
    expected_per_class = int(split_cfg["eval_rows_per_language_per_class"])
    if counts != {
        (language, membership): expected_per_class
        for language in SHADOW_GOLD_LANGUAGES
        for membership in (0, 1)
    }:
        raise ShadowGoldKaggleError("Gold language/class balance changed")

    schedule = make_exposure_schedule(
        benchmark.train_corpus,
        int(protocol_cfg["exposure_repeats"]),
        seed=int(protocol_cfg["seed"]),
    )
    if len(schedule) != int(protocol_cfg["expected_training_sequences"]):
        raise ShadowGoldKaggleError("Gold exposure sequence count changed")
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
        benchmark.train_corpus,
        schedule,
        tokenizer=tokenizer,
        sequence_config=sequence_config,
        seed=int(protocol_cfg["seed"]),
    )
    left, right = default_shadow_pair(max_position_embeddings=int(protocol_cfg["max_sequence_tokens"]))
    protocol = build_shadow_protocol_manifest(
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
        raise ShadowGoldKaggleError(f"Gold protocol array shape changed: {input_ids.shape}")
    input_path = protocol_dir / "training_input_ids.npy"
    with input_path.open("wb") as handle:
        np.save(handle, input_ids, allow_pickle=False)
    metadata_path = protocol_dir / "training_sequence_metadata.jsonl"
    meta_columns = [
        "benchmark_id",
        "exposure_round",
        "sequence_index",
        "source_token_count",
        "window_start",
        "window_name",
        "selected_payload_tokens",
    ]
    _write_jsonl(sequences, metadata_path, meta_columns)
    tokenizer.save_pretrained(protocol_dir)
    protocol_manifest = {
        "status": "frozen",
        "operation": "freeze_gold_controlled_shadow_protocol",
        "experiment_id": EXPERIMENT_ID,
        "synthetic_gold_controlled_only": True,
        "corpus_sha256": corpus_hash,
        "protocol": protocol,
        "training_input_shape": list(input_ids.shape),
        "training_input_dtype": str(input_ids.dtype),
        "pad_token_id": tokenizer.pad_token_id,
        "attention_mask_rule": "training_input_ids != pad_token_id",
        "label_rule": "input id where attention=1, otherwise -100",
        "output_sha256": {
            "training_input_ids.npy": _sha256_file(input_path),
            "training_sequence_metadata.jsonl": _sha256_file(metadata_path),
            "byte_tokenizer.json": _sha256_file(protocol_dir / "byte_tokenizer.json"),
        },
        "optimiser_steps_performed": 0,
        "model_compute_started": False,
        "accelerator_selected": False,
        "kaggle_operation_performed": False,
    }
    _write_json(protocol_dir / "shadow_protocol_manifest.json", protocol_manifest)

    input_columns = ["benchmark_id", "content", "language", "character_count", "length_bin", "cluster_id"]
    label_columns = list(_LABEL_COLUMNS)
    evaluation_input_path = scoring_dir / "evaluation_inputs.jsonl"
    label_path = label_dir / "evaluation_labels.jsonl"
    _write_jsonl(evaluation, evaluation_input_path, input_columns)
    _write_jsonl(evaluation, label_path, label_columns)
    input_hash = _sha256_file(evaluation_input_path)
    label_hash = _sha256_file(label_path)
    scoring_manifest = {
        "status": "sealed",
        "operation": "export_label_free_gold_scoring_bundle",
        "evaluation_input_file": evaluation_input_path.name,
        "evaluation_input_sha256": input_hash,
        "withheld_label_sha256": label_hash,
        "payload_files_present": [evaluation_input_path.name],
        "membership_labels_present": False,
        "matched_pair_ids_present": False,
        "model_compute_started": False,
        "kaggle_operation_performed": False,
    }
    _write_json(scoring_dir / "scoring_bundle_manifest.json", scoring_manifest)
    label_manifest = {
        "status": "sealed_parent_only",
        "rows": len(evaluation),
        "membership_rows": int((evaluation.membership.astype(int) == 1).sum()),
        "nonmembership_rows": int((evaluation.membership.astype(int) == 0).sum()),
        "label_sha256": label_hash,
        "scoring_input_sha256": input_hash,
        "passed_to_gpu_children": False,
    }
    _write_json(label_dir / "label_manifest.json", label_manifest)

    expected_steps = len(schedule) // int(protocol_cfg["global_batch_size"])
    if expected_steps != int(protocol_cfg["expected_optimizer_steps_per_architecture"]):
        raise ShadowGoldKaggleError("Gold optimizer-step count changed")
    return {
        "status": "prepared",
        "runtime_version": SHADOW_GOLD_KAGGLE_RUNTIME_VERSION,
        "corpus_sha256": corpus_hash,
        "candidate_rows": len(corpus),
        "train_rows": len(benchmark.train_corpus),
        "evaluation_rows": len(evaluation),
        "evaluation_input_sha256": input_hash,
        "evaluation_label_sha256": label_hash,
        "training_sequences": len(schedule),
        "optimizer_steps_per_architecture": expected_steps,
        "protocol_directory": str(protocol_dir),
        "scoring_bundle_directory": str(scoring_dir),
        "label_directory": str(label_dir),
        "evaluation_labels_passed_to_children": False,
        "competition_rows_used": 0,
    }


def _child_output_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "gold_features.jsonl", output_dir / "gold_child_manifest.json"


def run_gold_child(
    *,
    protocol_directory: str | Path,
    scoring_bundle_directory: str | Path,
    output_directory: str | Path,
    slot: str,
    expected_steps: int = 320,
) -> dict[str, Any]:
    """Train and label-free-score one architecture on its single visible T4."""

    if slot not in {"left", "right"}:
        raise ValueError("slot must be left or right")
    names = visible_cuda_device_names()
    validate_gpu_inventory(names, expected_count=1, required_name_fragment="T4")
    output_dir = Path(output_directory).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ShadowGoldKaggleError("Gold child output directory already exists and is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir = output_dir / "training"
    scoring_output = output_dir / "scoring"
    runtime = ShadowRuntimeConfig(
        backend="cuda",
        architecture_slot=slot,
        max_steps=None,
        checkpoint_reload_atol=1e-5,
        parameter_sync_atol=1e-5,
        save_optimizer_state=True,
    )
    training = train_shadow_model(protocol_directory, training_dir, runtime)
    if not isinstance(training, dict) or training.get("status") != "complete":
        raise ShadowGoldKaggleError("Gold child training did not complete")
    if int(training.get("completed_steps", -1)) != expected_steps:
        raise ShadowGoldKaggleError("Gold child completed-step count changed")
    if training.get("backend") != "cuda" or int(training.get("world_size", -1)) != 1:
        raise ShadowGoldKaggleError("Gold child CUDA/world-size contract failed")
    if training.get("checkpoint_reload_passed") is not True:
        raise ShadowGoldKaggleError("Gold child checkpoint reload failed")
    if float(training.get("checkpoint_reload_max_absolute_logit_difference", math.inf)) > 1e-5:
        raise ShadowGoldKaggleError("Gold child checkpoint reload fidelity exceeded tolerance")
    for key in ("pretrained_weights_used", "evaluation_inputs_read", "evaluation_labels_read", "sample_ids_used_as_features"):
        if training.get(key) is not False:
            raise ShadowGoldKaggleError(f"Gold child training clean-room guard failed: {key}")
    if training.get("automatic_compute_retries") != 0:
        raise ShadowGoldKaggleError("Gold child automatic retry guard failed")

    features, scoring_manifest = score_gold_shadow_features(
        training_dir,
        scoring_bundle_directory,
        backend="cuda",
        config=GoldScoringConfig(),
    )
    if list(features.columns) != _FEATURE_COLUMNS:
        raise ShadowGoldKaggleError(f"Gold child feature columns changed: {list(features.columns)}")
    feature_path, manifest_path = _child_output_paths(output_dir)
    _write_jsonl(features, feature_path, _FEATURE_COLUMNS)
    feature_hash = _sha256_file(feature_path)
    child_manifest = {
        "status": "sealed",
        "runtime_version": SHADOW_GOLD_KAGGLE_RUNTIME_VERSION,
        "slot": slot,
        "visible_gpu_name": names[0],
        "expected_steps": expected_steps,
        "training_manifest": training,
        "scoring_manifest": scoring_manifest,
        "feature_rows": len(features),
        "feature_sha256": feature_hash,
        "feature_columns": _FEATURE_COLUMNS,
        "membership_labels_read": False,
        "label_file_received": False,
        "automatic_compute_retries": 0,
    }
    _write_json(manifest_path, child_manifest)
    return child_manifest


def _tail(path: Path, maximum: int) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return data[-maximum:].decode("utf-8", errors="replace")


def run_gold_dual_gpu(
    *,
    protocol_directory: str | Path,
    scoring_bundle_directory: str | Path,
    output_directory: str | Path,
    expected_steps: int = 320,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run left/right train+score children concurrently on physical GPU0/GPU1."""

    names = visible_cuda_device_names()
    validate_gpu_inventory(names, expected_count=2, required_name_fragment="T4")
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise ShadowGoldKaggleError("Gold dual-GPU output directory already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "_logs"
    logs.mkdir()
    protocol = str(Path(protocol_directory).resolve())
    scoring = str(Path(scoring_bundle_directory).resolve())
    children: dict[str, tuple[subprocess.Popen[Any], Any, Any, Path, int]] = {}
    for slot, gpu_index in (("left", 0), ("right", 1)):
        child_output = output / slot
        command = [
            sys.executable,
            "-m",
            "poisoned_chalice.shadow_gold_kaggle",
            "--child",
            "--slot",
            slot,
            "--protocol-directory",
            protocol,
            "--scoring-bundle-directory",
            scoring,
            "--output-directory",
            str(child_output),
            "--expected-steps",
            str(expected_steps),
        ]
        env = os.environ.copy()
        for key in ("PJRT_DEVICE", "TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID", "XRT_TPU_CONFIG", "LOCAL_RANK", "RANK", "WORLD_SIZE"):
            env.pop(key, None)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env["TOKENIZERS_PARALLELISM"] = "false"
        stdout_handle = (logs / f"{slot}.stdout").open("wb")
        stderr_handle = (logs / f"{slot}.stderr").open("wb")
        process = subprocess.Popen(command, env=env, stdout=stdout_handle, stderr=stderr_handle)
        children[slot] = (process, stdout_handle, stderr_handle, child_output, gpu_index)

    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            if all(process.poll() is not None for process, _, _, _, _ in children.values()):
                break
            if time.monotonic() >= deadline:
                for process, _, _, _, _ in children.values():
                    if process.poll() is None:
                        process.kill()
                raise ShadowGoldKaggleError("Gold dual-GPU timeout; automatic retry forbidden")
            time.sleep(0.5)
    finally:
        for process, stdout_handle, stderr_handle, _, _ in children.values():
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            stdout_handle.close()
            stderr_handle.close()

    failures = []
    manifests: dict[str, dict[str, Any]] = {}
    for slot, (process, _, _, child_output, gpu_index) in children.items():
        if process.returncode != 0:
            failures.append(
                {
                    "slot": slot,
                    "physical_gpu_index": gpu_index,
                    "returncode": process.returncode,
                    "stdout_tail": _tail(logs / f"{slot}.stdout", 2000),
                    "stderr_tail": _tail(logs / f"{slot}.stderr", 8000),
                }
            )
            continue
        manifest_path = child_output / "gold_child_manifest.json"
        feature_path = child_output / "gold_features.jsonl"
        if not manifest_path.is_file() or not feature_path.is_file():
            failures.append({"slot": slot, "physical_gpu_index": gpu_index, "returncode": 0, "stderr_tail": "child artifacts missing"})
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "sealed" or manifest.get("slot") != slot:
            raise ShadowGoldKaggleError(f"Gold child manifest invalid: {slot}")
        if manifest.get("feature_sha256") != _sha256_file(feature_path):
            raise ShadowGoldKaggleError(f"Gold child feature SHA mismatch: {slot}")
        if manifest.get("membership_labels_read") is not False or manifest.get("label_file_received") is not False:
            raise ShadowGoldKaggleError(f"Gold child label boundary failed: {slot}")
        manifests[slot] = manifest
    if failures:
        raise ShadowGoldKaggleError("one or more Gold GPU children failed: " + json.dumps(failures, sort_keys=True))
    if set(manifests) != {"left", "right"}:
        raise ShadowGoldKaggleError("Gold child manifest coverage incomplete")
    return {
        "status": "complete",
        "runtime_version": SHADOW_GOLD_KAGGLE_RUNTIME_VERSION,
        "gpu_names": names,
        "architecture_gpu_assignment": {"left": 0, "right": 1},
        "distributed_training": False,
        "automatic_compute_retries": 0,
        "left": manifests["left"],
        "right": manifests["right"],
    }


def _load_child_features(path: Path) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True)
    if list(frame.columns) != _FEATURE_COLUMNS:
        raise ShadowGoldKaggleError("Gold child persisted feature columns changed")
    if frame.benchmark_id.duplicated().any():
        raise ShadowGoldKaggleError("Gold child persisted feature IDs are duplicated")
    for column in [*GOLD_PRIMITIVES, "stage2_v1"]:
        if not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
            raise ShadowGoldKaggleError(f"Gold child persisted feature is non-finite: {column}")
    return frame


def _spearman_score_length(frame: pd.DataFrame, score: np.ndarray) -> float:
    left = pd.Series(np.asarray(score, dtype=np.float64)).rank(method="average")
    right = pd.Series(frame.character_count.to_numpy(dtype=np.float64)).rank(method="average")
    value = float(left.corr(right))
    return value if math.isfinite(value) else 0.0


def finalize_gold_results(
    *,
    dual_output_directory: str | Path,
    label_directory: str | Path,
    false_positive_rate: float = 0.01,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Reveal exact synthetic labels only after both children have sealed features."""

    dual = Path(dual_output_directory).resolve()
    labels_root = Path(label_directory).resolve()
    left_features = _load_child_features(dual / "left" / "gold_features.jsonl")
    right_features = _load_child_features(dual / "right" / "gold_features.jsonl")
    if left_features.benchmark_id.tolist() != right_features.benchmark_id.tolist():
        raise ShadowGoldKaggleError("left/right Gold feature row order differs")
    label_path = labels_root / "evaluation_labels.jsonl"
    label_manifest_path = labels_root / "label_manifest.json"
    if not label_path.is_file() or not label_manifest_path.is_file():
        raise ShadowGoldKaggleError("parent-only Gold label seal is incomplete")
    label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
    if label_manifest.get("passed_to_gpu_children") is not False:
        raise ShadowGoldKaggleError("Gold parent label boundary manifest failed")
    if label_manifest.get("label_sha256") != _sha256_file(label_path):
        raise ShadowGoldKaggleError("Gold parent label SHA mismatch")
    labels = pd.read_json(label_path, lines=True)
    if list(labels.columns) != _LABEL_COLUMNS:
        raise ShadowGoldKaggleError("Gold parent label columns changed")

    left_labeled = attach_exact_labels(left_features, labels)
    attack, left_holdout = fit_frozen_gold_attack(
        left_labeled,
        seed=2027,
        false_positive_rate=false_positive_rate,
    )
    holdout_ids = left_holdout.benchmark_id.astype(str).tolist()
    right_lookup = right_features.set_index(right_features.benchmark_id.astype(str), drop=False)
    try:
        right_holdout_features = right_lookup.loc[holdout_ids].reset_index(drop=True)
    except KeyError as error:
        raise ShadowGoldKaggleError("right Gold features lack frozen holdout IDs") from error
    right_holdout = attach_exact_labels(right_holdout_features, labels[labels.benchmark_id.astype(str).isin(holdout_ids)].copy())
    right_holdout = right_holdout.set_index(right_holdout.benchmark_id.astype(str), drop=False).loc[holdout_ids].reset_index(drop=True)
    if left_holdout.benchmark_id.astype(str).tolist() != right_holdout.benchmark_id.astype(str).tolist():
        raise ShadowGoldKaggleError("paired left/right Gold holdout order differs")

    left_score = apply_frozen_gold_attack(left_holdout, attack)
    right_score = apply_frozen_gold_attack(right_holdout, attack)
    left_metrics = evaluate_frozen_gold_holdout(left_holdout, left_score, false_positive_rate=false_positive_rate)
    right_metrics = evaluate_frozen_gold_holdout(right_holdout, right_score, false_positive_rate=false_positive_rate)
    primitive_holdout: dict[str, Any] = {"left": {}, "right": {}}
    for name in [*GOLD_PRIMITIVES, "stage2_v1"]:
        primitive_holdout["left"][name] = evaluate_frozen_gold_holdout(
            left_holdout,
            left_holdout[name].to_numpy(dtype=np.float64),
            false_positive_rate=false_positive_rate,
        )["overall"]
        primitive_holdout["right"][name] = evaluate_frozen_gold_holdout(
            right_holdout,
            right_holdout[name].to_numpy(dtype=np.float64),
            false_positive_rate=false_positive_rate,
        )["overall"]

    left_overall = left_metrics["overall"]
    right_overall = right_metrics["overall"]
    metrics = {
        "status": "complete",
        "version": SHADOW_GOLD_KAGGLE_RUNTIME_VERSION,
        "selected_attack": attack.selected_candidate,
        "development_candidate_metrics": attack.development_metrics,
        "development_rows": attack.development_rows,
        "holdout_rows_per_architecture": attack.holdout_rows,
        "left_holdout": left_metrics,
        "right_holdout": right_metrics,
        "primitive_holdout_overall": primitive_holdout,
        "left_to_right_delta": {
            key: float(right_overall[key] - left_overall[key])
            for key in ("auc", "partial_auc_standardized", "tpr_at_fpr")
        },
        "score_length_spearman": {
            "left": _spearman_score_length(left_holdout, left_score),
            "right": _spearman_score_length(right_holdout, right_score),
        },
        "shadow_a_labels_used_for_fit": True,
        "shadow_a_holdout_labels_used_for_fit": False,
        "shadow_b_labels_used_for_fit": False,
        "frozen_attack_applied_unchanged_to_right": True,
        "stage2_v3_selection_allowed": False,
        "external_model_holdout_consumed": False,
    }
    attack_dict = frozen_attack_to_dict(attack)
    predictions = left_holdout[["benchmark_id", "language", "length_bin", "character_count", "membership", "matched_pair_id"]].copy()
    predictions["left_frozen_score"] = left_score
    predictions["right_frozen_score"] = right_score
    return attack_dict, metrics, predictions


def run_gold_benchmark(
    *,
    scratch_directory: str | Path,
    config: Mapping[str, Any],
    timeout_seconds: int = 1800,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Run the full Gold benchmark in one parent process after GPU inventory validation."""

    names = visible_cuda_device_names()
    validate_gpu_inventory(names, expected_count=2, required_name_fragment="T4")
    scratch = Path(scratch_directory).resolve()
    if scratch.exists() and any(scratch.iterdir()):
        raise ShadowGoldKaggleError("Gold scratch directory already exists and is not empty")
    scratch.mkdir(parents=True, exist_ok=True)
    prepared = prepare_gold_runtime(scratch / "prepared", config)
    dual = run_gold_dual_gpu(
        protocol_directory=prepared["protocol_directory"],
        scoring_bundle_directory=prepared["scoring_bundle_directory"],
        output_directory=scratch / "dual",
        expected_steps=int(config["protocol"]["expected_optimizer_steps_per_architecture"]),
        timeout_seconds=timeout_seconds,
    )
    attack, metrics, predictions = finalize_gold_results(
        dual_output_directory=scratch / "dual",
        label_directory=prepared["label_directory"],
        false_positive_rate=float(config["scoring"]["false_positive_rate"]),
    )
    manifest = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "runtime_version": SHADOW_GOLD_KAGGLE_RUNTIME_VERSION,
        "gpu_names": names,
        "prepared": prepared,
        "dual_gpu": dual,
        "selected_attack": attack["selected_candidate"],
        "metrics_status": metrics["status"],
        "evaluation_labels_passed_to_children": False,
        "competition_rows_used": 0,
        "pretrained_weights_used": False,
        "automatic_compute_retries": 0,
        "stage2_v3_selection_allowed": False,
    }
    return attack, metrics, predictions, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--slot", choices=("left", "right"))
    parser.add_argument("--protocol-directory", type=Path)
    parser.add_argument("--scoring-bundle-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--expected-steps", type=int, default=320)
    args = parser.parse_args(argv)
    if not args.child:
        parser.error("module CLI is reserved for isolated Gold child execution")
    if args.slot is None or args.protocol_directory is None or args.scoring_bundle_directory is None or args.output_directory is None:
        parser.error("Gold child requires slot, protocol, scoring bundle, and output directory")
    result = run_gold_child(
        protocol_directory=args.protocol_directory,
        scoring_bundle_directory=args.scoring_bundle_directory,
        output_directory=args.output_directory,
        slot=args.slot,
        expected_steps=args.expected_steps,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "slot": result["slot"],
                "feature_rows": result["feature_rows"],
                "membership_labels_read": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPERIMENT_ID",
    "SHADOW_GOLD_KAGGLE_RUNTIME_VERSION",
    "ShadowGoldKaggleError",
    "finalize_gold_results",
    "prepare_gold_runtime",
    "run_gold_benchmark",
    "run_gold_child",
    "run_gold_dual_gpu",
    "validate_gold_config",
]
