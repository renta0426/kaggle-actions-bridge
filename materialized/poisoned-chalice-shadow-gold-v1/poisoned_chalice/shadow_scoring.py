"""Seal label-free predictions from a trained controlled shadow model.

The scorer accepts a training-output directory and a scoring bundle that contains
only controlled evaluation inputs. It rejects any label file, verifies the
checkpoint and input SHA chains, calls the generic Stage 2 API without passing
benchmark IDs, and atomically writes content-free predictions plus a manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .shadow_protocol import ByteCodeTokenizer, ShadowArchitectureSpec
from .shadow_training import _state_dict_sha256
from .stage2_api import (
    STAGE2_METHOD_VERSION,
    Stage2RuntimeConfig,
    score_samples_detailed,
)


SHADOW_SCORING_VERSION = "shadow-scoring-v1"
_EVALUATION_INPUT_COLUMNS = [
    "benchmark_id",
    "content",
    "language",
    "character_count",
    "length_bin",
    "cluster_id",
]
_PREDICTION_COLUMNS = [
    "benchmark_id",
    "membership_score",
    "language",
    "length_bin",
    "character_count",
]


class ShadowScoringError(RuntimeError):
    """Raised when a checkpoint/input/scoring invariant is violated."""


@dataclass(frozen=True)
class ShadowScoringConfig:
    backend: str = "cpu"
    max_batch_tokens: int = 4_096
    vocab_chunk_tokens: int = 64
    rank_vocab_block_size: int = 256
    language_calibration_min_rows: int = 20
    length_calibration_min_rows: int = 12
    fidelity_gate: bool = True
    fidelity_tokens: int = 24
    fidelity_atol: float = 1e-5
    time_budget_seconds: float | None = None
    max_peak_vram_bytes: int | None = None
    cpu_threads: int | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "cuda"}:
            raise ValueError("backend must be 'cpu' or 'cuda'")
        for name in (
            "max_batch_tokens",
            "vocab_chunk_tokens",
            "rank_vocab_block_size",
            "language_calibration_min_rows",
            "length_calibration_min_rows",
            "fidelity_tokens",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.fidelity_atol < 0:
            raise ValueError("fidelity_atol must be non-negative")
        if self.time_budget_seconds is not None and self.time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds must be positive when set")
        if self.max_peak_vram_bytes is not None and self.max_peak_vram_bytes <= 0:
            raise ValueError("max_peak_vram_bytes must be positive when set")
        if self.cpu_threads is not None and self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive when set")


@dataclass(frozen=True)
class LoadedShadowCheckpoint:
    model: Any
    architecture: ShadowArchitectureSpec
    training_manifest: Mapping[str, Any]
    training_manifest_sha256: str
    checkpoint_sha256: str
    model_state_sha256: str


@dataclass(frozen=True)
class LoadedScoringBundle:
    frame: pd.DataFrame
    manifest: Mapping[str, Any]
    input_sha256: str
    bundle_manifest_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _write_jsonl(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ShadowScoringError(f"prediction frame is missing columns: {missing}")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in frame[columns].itertuples(index=False, name=None):
            handle.write(_canonical_json(dict(zip(columns, row))))
            handle.write("\n")


def export_scoring_bundle(
    sealed_evaluation_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Copy only label-free evaluation inputs into an independently sealed bundle."""

    source = Path(sealed_evaluation_directory).resolve()
    destination = Path(output_directory).resolve()
    if not source.is_dir():
        raise ShadowScoringError(f"sealed evaluation directory does not exist: {source}")
    if destination.exists() and any(destination.iterdir()):
        raise ShadowScoringError("output directory already exists and is not empty")
    seal_path = source / "evaluation_seal_manifest.json"
    input_path = source / "evaluation_inputs.jsonl"
    label_path = source / "evaluation_labels.jsonl"
    if not all(path.is_file() for path in (seal_path, input_path, label_path)):
        raise ShadowScoringError("sealed evaluation directory is incomplete")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("status") != "sealed":
        raise ShadowScoringError("evaluation seal manifest is not sealed")
    hashes = seal.get("output_sha256") or {}
    input_sha256 = _sha256_file(input_path)
    label_sha256 = _sha256_file(label_path)
    if hashes.get(input_path.name) != input_sha256:
        raise ShadowScoringError("sealed evaluation input SHA-256 mismatch")
    if hashes.get(label_path.name) != label_sha256:
        raise ShadowScoringError("sealed evaluation label SHA-256 mismatch")
    if seal.get("scoring_job_allowed_files") != [input_path.name]:
        raise ShadowScoringError("evaluation seal scoring allowlist mismatch")
    if seal.get("input_contains_membership_label") is not False:
        raise ShadowScoringError("evaluation seal input-label guard failed")

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        copied_input = temporary / input_path.name
        shutil.copyfile(input_path, copied_input)
        if _sha256_file(copied_input) != input_sha256:
            raise ShadowScoringError("scoring bundle copy SHA-256 mismatch")
        bundle = {
            "status": "sealed",
            "operation": "export_label_free_shadow_scoring_bundle",
            "source_evaluation_seal_manifest_sha256": _sha256_file(seal_path),
            "evaluation_input_file": copied_input.name,
            "evaluation_input_sha256": input_sha256,
            "withheld_label_sha256": label_sha256,
            "payload_files_present": [copied_input.name],
            "membership_labels_present": False,
            "matched_pair_ids_present": False,
            "model_compute_started": False,
            "kaggle_operation_performed": False,
        }
        (temporary / "scoring_bundle_manifest.json").write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            destination.rmdir()
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return bundle


def load_scoring_bundle(directory: str | Path) -> LoadedScoringBundle:
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ShadowScoringError(f"scoring bundle does not exist: {root}")
    expected_files = {"evaluation_inputs.jsonl", "scoring_bundle_manifest.json"}
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ShadowScoringError(
            "scoring bundle file allowlist mismatch: "
            f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
        )
    manifest_path = root / "scoring_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "sealed":
        raise ShadowScoringError("scoring bundle manifest is not sealed")
    if manifest.get("membership_labels_present") is not False:
        raise ShadowScoringError("scoring bundle membership-label guard failed")
    if manifest.get("matched_pair_ids_present") is not False:
        raise ShadowScoringError("scoring bundle matched-pair guard failed")
    if manifest.get("payload_files_present") != ["evaluation_inputs.jsonl"]:
        raise ShadowScoringError("scoring bundle manifest file allowlist mismatch")
    input_path = root / "evaluation_inputs.jsonl"
    input_sha256 = _sha256_file(input_path)
    if manifest.get("evaluation_input_sha256") != input_sha256:
        raise ShadowScoringError("scoring bundle input SHA-256 mismatch")
    frame = pd.read_json(input_path, lines=True)
    if set(frame.columns) != set(_EVALUATION_INPUT_COLUMNS):
        raise ShadowScoringError(
            "scoring input columns mismatch: "
            f"expected={sorted(_EVALUATION_INPUT_COLUMNS)}, actual={sorted(frame.columns)}"
        )
    frame = frame[_EVALUATION_INPUT_COLUMNS].copy()
    forbidden = {"membership", "label", "matched_pair_id", "is_member"}
    present = sorted(forbidden.intersection(frame.columns))
    if present:
        raise ShadowScoringError(f"labels leaked into scoring input: {present}")
    if frame.empty or frame.benchmark_id.duplicated().any():
        raise ShadowScoringError("scoring input IDs are empty or non-unique")
    if frame.content.isna().any() or frame.language.isna().any():
        raise ShadowScoringError("scoring input content/language contains missing values")
    if frame.language.astype(str).str.strip().eq("").any():
        raise ShadowScoringError("scoring input contains an empty language")
    if not np.isfinite(frame.character_count.to_numpy(dtype=float)).all():
        raise ShadowScoringError("scoring input character counts are non-finite")
    if (frame.character_count.astype(int) < 0).any():
        raise ShadowScoringError("scoring input character counts are negative")
    return LoadedScoringBundle(
        frame=frame,
        manifest=manifest,
        input_sha256=input_sha256,
        bundle_manifest_sha256=_sha256_file(manifest_path),
    )


def load_shadow_checkpoint(
    training_output_directory: str | Path,
    *,
    backend: str,
) -> LoadedShadowCheckpoint:
    import torch

    root = Path(training_output_directory).resolve()
    if not root.is_dir():
        raise ShadowScoringError(f"training output directory does not exist: {root}")
    manifest_path = root / "training_manifest.json"
    if not manifest_path.is_file():
        raise ShadowScoringError("training manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ShadowScoringError("training manifest is not complete")
    for key in (
        "pretrained_weights_used",
        "evaluation_inputs_read",
        "evaluation_labels_read",
        "sample_ids_used_as_features",
    ):
        if manifest.get(key) is not False:
            raise ShadowScoringError(f"training clean-room guard failed: {key}")
    if manifest.get("automatic_compute_retries") != 0:
        raise ShadowScoringError("training automatic-retry guard failed")
    if manifest.get("checkpoint_reload_passed") is not True:
        raise ShadowScoringError("training checkpoint-reload gate did not pass")
    checkpoint_name = manifest.get("checkpoint_file")
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise ShadowScoringError("training checkpoint file name is invalid")
    checkpoint_path = root / checkpoint_name
    if not checkpoint_path.is_file():
        raise ShadowScoringError("training checkpoint file is missing")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if manifest.get("checkpoint_sha256") != checkpoint_sha256:
        raise ShadowScoringError("training checkpoint SHA-256 mismatch")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol_manifest_sha256") != manifest.get(
        "protocol_manifest_sha256"
    ):
        raise ShadowScoringError("checkpoint/training protocol SHA mismatch")
    if checkpoint.get("architecture_slot") != manifest.get("architecture_slot"):
        raise ShadowScoringError("checkpoint/training architecture slot mismatch")
    architecture_value = checkpoint.get("architecture")
    if architecture_value != manifest.get("architecture"):
        raise ShadowScoringError("checkpoint/training architecture spec mismatch")
    if not isinstance(architecture_value, dict):
        raise ShadowScoringError("checkpoint architecture spec is missing")
    architecture = ShadowArchitectureSpec(**architecture_value)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ShadowScoringError("checkpoint model state is missing")
    logical_sha256 = _state_dict_sha256(state)
    if logical_sha256 != checkpoint.get("model_state_sha256"):
        raise ShadowScoringError("checkpoint logical model-state SHA mismatch")
    if logical_sha256 != manifest.get("model_state_sha256"):
        raise ShadowScoringError("training manifest logical model-state SHA mismatch")

    training_spec = checkpoint.get("training_spec")
    if not isinstance(training_spec, dict) or "seed" not in training_spec:
        raise ShadowScoringError("checkpoint training seed is missing")
    try:
        seed = int(training_spec["seed"])
    except (TypeError, ValueError) as error:
        raise ShadowScoringError("checkpoint training seed is invalid") from error
    model = architecture.build_model(seed=seed)
    model.load_state_dict(state, strict=True)
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise ShadowScoringError("CUDA scoring requested but unavailable")
        model.to(torch.device("cuda", 0))
    elif backend == "cpu":
        model.to(torch.device("cpu"))
    else:
        raise ValueError("backend must be 'cpu' or 'cuda'")
    model.eval()
    del checkpoint, state
    return LoadedShadowCheckpoint(
        model=model,
        architecture=architecture,
        training_manifest=manifest,
        training_manifest_sha256=_sha256_file(manifest_path),
        checkpoint_sha256=checkpoint_sha256,
        model_state_sha256=logical_sha256,
    )


def score_shadow_checkpoint(
    training_output_directory: str | Path,
    scoring_bundle_directory: str | Path,
    output_directory: str | Path,
    runtime_config: ShadowScoringConfig | None = None,
) -> dict[str, Any]:
    """Score the complete controlled target batch and seal predictions atomically."""

    import torch

    config = runtime_config or ShadowScoringConfig()
    if config.cpu_threads is not None and config.backend == "cpu":
        torch.set_num_threads(config.cpu_threads)
    bundle = load_scoring_bundle(scoring_bundle_directory)
    checkpoint = load_shadow_checkpoint(
        training_output_directory,
        backend=config.backend,
    )
    stage2_config = Stage2RuntimeConfig(
        max_length=checkpoint.architecture.max_position_embeddings,
        max_batch_tokens=config.max_batch_tokens,
        vocab_chunk_tokens=config.vocab_chunk_tokens,
        rank_vocab_block_size=config.rank_vocab_block_size,
        language_calibration_min_rows=config.language_calibration_min_rows,
        length_calibration_min_rows=config.length_calibration_min_rows,
        fidelity_gate=config.fidelity_gate,
        fidelity_tokens=config.fidelity_tokens,
        fidelity_atol=config.fidelity_atol,
        device="auto",
        move_model=False,
        time_budget_seconds=config.time_budget_seconds,
        max_peak_vram_bytes=config.max_peak_vram_bytes,
    )
    frame = bundle.frame
    detailed = score_samples_detailed(
        model=checkpoint.model,
        tokenizer=ByteCodeTokenizer(),
        samples=frame.content.astype(str).tolist(),
        languages=frame.language.astype(str).tolist(),
        runtime_config=stage2_config,
    )
    scores = np.asarray(detailed.scores, dtype=np.float64)
    if scores.shape != (len(frame),) or not np.isfinite(scores).all():
        raise ShadowScoringError("shadow score coverage or finiteness check failed")
    predictions = frame[
        ["benchmark_id", "language", "length_bin", "character_count"]
    ].copy()
    predictions.insert(1, "membership_score", scores)
    predictions = predictions[_PREDICTION_COLUMNS]
    if predictions.benchmark_id.duplicated().any():
        raise ShadowScoringError("prediction benchmark IDs are non-unique")

    destination = Path(output_directory).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ShadowScoringError("output directory already exists and is not empty")
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        prediction_path = temporary / "shadow_predictions.jsonl"
        _write_jsonl(predictions, prediction_path, _PREDICTION_COLUMNS)
        prediction_sha256 = _sha256_file(prediction_path)
        manifest = {
            "status": "sealed",
            "scoring_version": SHADOW_SCORING_VERSION,
            "stage2_method_version": STAGE2_METHOD_VERSION,
            "training_manifest_sha256": checkpoint.training_manifest_sha256,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "model_state_sha256": checkpoint.model_state_sha256,
            "protocol_manifest_sha256": checkpoint.training_manifest.get(
                "protocol_manifest_sha256"
            ),
            "architecture_slot": checkpoint.training_manifest.get(
                "architecture_slot"
            ),
            "architecture": asdict(checkpoint.architecture),
            "scoring_bundle_manifest_sha256": bundle.bundle_manifest_sha256,
            "evaluation_input_sha256": bundle.input_sha256,
            "withheld_label_sha256": bundle.manifest.get("withheld_label_sha256"),
            "prediction_file": prediction_path.name,
            "prediction_sha256": prediction_sha256,
            "prediction_rows": len(predictions),
            "prediction_columns": _PREDICTION_COLUMNS,
            "score_minimum": float(scores.min()),
            "score_maximum": float(scores.max()),
            "score_mean": float(scores.mean()),
            "score_standard_deviation": float(scores.std()),
            "runtime_config": asdict(config),
            "stage2_manifest": detailed.manifest,
            "membership_labels_read": False,
            "matched_pair_ids_read": False,
            "benchmark_ids_passed_to_scorer": False,
            "sample_ids_used_as_features": False,
            "code_content_written_to_predictions": False,
            "public_leaderboard_tuning_used": False,
            "automatic_compute_retries": 0,
        }
        (temporary / "prediction_manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            destination.rmdir()
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


__all__ = [
    "SHADOW_SCORING_VERSION",
    "LoadedScoringBundle",
    "LoadedShadowCheckpoint",
    "ShadowScoringConfig",
    "ShadowScoringError",
    "export_scoring_bundle",
    "load_scoring_bundle",
    "load_shadow_checkpoint",
    "score_shadow_checkpoint",
]
