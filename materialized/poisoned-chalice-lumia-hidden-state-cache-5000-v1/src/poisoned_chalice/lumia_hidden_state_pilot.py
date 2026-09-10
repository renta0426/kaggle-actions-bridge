"""Faithful LUMIA hidden-state extraction helpers and runtime gate.

The public SERSEM/LUMIA implementation at author commit
413f56040e5b4805bcf15ed794dec56bc4e16b41 concatenates an ordinary sequence
mean with a heuristic-weighted sequence mean for every transformer layer.  This
module reproduces that activation mechanism while keeping two structural-weight
branches explicit:

* caller-literal: current released caller omits ``language`` and therefore uses
  regex AST fallback + generic linter;
* helper-language-aware: supplies the known competition language to the
  released helper semantics (Tree-sitter/PyEnchant + Python flake8).

This module contains no top-level torch/transformers imports so its cohort and
contract logic can be unit-tested on CPU-only CI.  The first execution gate is a
50-row 8192-token runtime/fidelity pilot and computes no membership metric.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Sequence
import json
import math
import random
import time

import numpy as np
import pandas as pd

from poisoned_chalice.sersem_author_faithful import (
    build_author_character_weights,
    sigmoid_z,
    token_weights_from_character_weights,
    weighted_output_score,
)

AUTHOR_REPOSITORY = "Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026"
AUTHOR_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
MODEL_ID = "bigcode/starcoder2-3b"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
DATASET_ID = "Poisoned-Chalice/ICSE-2027-public"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
EXPECTED_TRAIN_ROWS = {"Go": 10_000, "Java": 10_000, "Python": 10_000,
                       "Ruby": 10_000, "Rust": 9_040}


@dataclass(frozen=True)
class LumiaRuntimePilotConfig:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    dataset_id: str = DATASET_ID
    dataset_revision: str = DATASET_REVISION
    rows: int = 50
    rows_per_language_label: int = 5
    seed: int = 20260909
    max_length: int = 8192
    z_chunk_tokens: int = 64
    expected_hidden_state_variants: int = 3
    output_dir: str = "/kaggle/working/lumia_hidden_state_runtime_pilot_50"
    performance_metrics_computed: bool = False


def membership_labels(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower().str.replace("_", "-", regex=False)
    labels = normalized.map({"member": 1, "non-member": 0, "nonmember": 0})
    if labels.isna().any():
        raise ValueError("unrecognized public membership label")
    return labels.astype(int)


def validate_public_train(train: pd.DataFrame) -> None:
    required = {"sample_id", "content", "membership", "language"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"public train missing columns: {sorted(missing)}")
    if len(train) != 49_040 or train.sample_id.duplicated().any():
        raise ValueError("unexpected public train identity contract")
    if train.groupby("language").size().to_dict() != EXPECTED_TRAIN_ROWS:
        raise ValueError("unexpected public train language counts")
    if train.membership.isna().any():
        raise ValueError("public train membership must be complete")
    expected_hash = train.content.map(lambda value: sha256(value.encode("utf-8")).hexdigest())
    observed_hash = train.sample_id.astype(str).str.rsplit("-", n=1).str[-1]
    if not np.array_equal(expected_hash.to_numpy(), observed_hash.to_numpy()):
        raise ValueError("public sample_id/content SHA mismatch")


def select_runtime_pilot(train: pd.DataFrame, config: LumiaRuntimePilotConfig) -> pd.DataFrame:
    """Select a deterministic language×label-balanced runtime cohort.

    Membership is used only for deterministic cohort balancing.  The returned
    model-input frame deliberately excludes membership/label so the runtime
    scoring loop cannot compute a performance metric.
    """
    validate_public_train(train)
    work = train.copy()
    work["_label"] = membership_labels(work.membership)
    pieces: list[pd.DataFrame] = []
    for language_index, language in enumerate(LANGUAGES):
        for label in (0, 1):
            pool = work[(work.language == language) & (work._label == label)].copy()
            # A distinct but deterministic seed avoids identical index draws
            # from the independently ordered member/non-member pools.
            sampled = pool.sample(
                n=config.rows_per_language_label,
                random_state=config.seed + 100 * language_index + label,
                replace=False,
            )
            pieces.append(sampled)
    cohort = pd.concat(pieces, ignore_index=True)
    if len(cohort) != config.rows or cohort.sample_id.duplicated().any():
        raise ValueError("runtime pilot cohort contract failed")
    balance = cohort.groupby(["language", "_label"]).size()
    if not balance.eq(config.rows_per_language_label).all() or len(balance) != 10:
        raise ValueError("runtime pilot balance contract failed")
    # Stable execution order is label-independent and directly auditable.
    cohort = cohort.sort_values("sample_id").reset_index(drop=True)
    return cohort[["sample_id", "language", "content"]].copy()


def literal_character_weights(text: str):
    """Current released caller semantics: language is omitted."""
    weights, diagnostics = build_author_character_weights(
        text,
        "",
        parser=None,
        word_is_known=None,
        flake8_errors=None,
    )
    return weights, diagnostics


def helper_character_weights(text: str, language: str, runtime: Any):
    flake8_errors = runtime.flake8_errors(text, language)
    return build_author_character_weights(
        text,
        language,
        parser=runtime.parsers.get(language),
        word_is_known=runtime.word_is_known,
        flake8_errors=flake8_errors,
    )


def normalized_token_weights(
    text: str,
    offsets: Sequence[Sequence[int]],
    character_weights: Sequence[float],
) -> np.ndarray:
    """Project author character weights and apply LumiaAttack's max normalization."""
    pairs = [(int(pair[0]), int(pair[1])) for pair in offsets]
    weights = token_weights_from_character_weights(text, pairs, character_weights).astype(np.float32)
    if weights.ndim != 1 or len(weights) != len(pairs) or len(weights) == 0:
        raise ValueError("token weight shape failure")
    maximum = float(weights.max())
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("token weight maximum must be positive finite")
    normalized = weights / (maximum + 1e-8)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized token weights are nonfinite")
    return normalized


def validate_activation_triplet(
    mean: np.ndarray,
    caller_weighted: np.ndarray,
    helper_weighted: np.ndarray,
    *,
    expected_hidden_dim: int,
) -> None:
    expected = (expected_hidden_dim,)
    for name, values in (
        ("mean", mean),
        ("caller", caller_weighted),
        ("helper", helper_weighted),
    ):
        if values.shape != expected:
            raise ValueError(f"{name} activation shape mismatch: {values.shape} != {expected}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} activation contains nonfinite values")


def correct_z_from_logits(logits: Any, target_ids: Any, *, chunk_tokens: int = 64) -> np.ndarray:
    """Compute the historical correct-logit uniform-vocabulary z-score in bounded chunks."""
    import torch

    if logits.ndim != 3 or target_ids.ndim != 2 or logits.shape[:2] != target_ids.shape:
        raise ValueError("logit/target shape mismatch")
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, logits.shape[1], chunk_tokens):
            stop = min(start + chunk_tokens, logits.shape[1])
            chunk = logits[:, start:stop].float()
            target = target_ids[:, start:stop].to(chunk.device)
            correct = chunk.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            mean = chunk.mean(dim=-1)
            std = chunk.std(dim=-1, correction=0).clamp_min(1e-6)
            z = (correct - mean) / std
            rows.append(z.detach().cpu().numpy())
            del chunk, target, correct, mean, std, z
    return np.concatenate(rows, axis=1)[0].astype(np.float32, copy=False)


def _layer_modules(model: Any) -> list[Any]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    modules: list[Any] = []
    import torch.nn as nn
    for name, module in model.named_modules():
        if ("layer" in name.lower() or "block" in name.lower()) and isinstance(module, nn.Module) and not isinstance(module, nn.ModuleList):
            modules.append(module)
    return modules


def extract_one_sample(
    model: Any,
    tokenizer: Any,
    text: str,
    language: str,
    runtime: Any,
    config: LumiaRuntimePilotConfig,
) -> dict[str, Any]:
    """Run one faithful 8192-token forward and return pooled all-layer activations.

    The hook computes mean, caller-literal weighted mean, and helper-aware
    weighted mean in the same model forward.  Full token-level hidden states are
    never persisted.
    """
    import torch

    encodings = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_offsets_mapping=True,
    )
    if encodings.input_ids.size(0) != 1 or encodings.input_ids.size(1) < 2:
        raise ValueError("runtime sample must contain at least two tokens")
    offsets_raw = encodings.pop("offset_mapping")[0].cpu().tolist()
    offsets = [(int(start), int(stop)) for start, stop in offsets_raw]
    seq_len = int(encodings.input_ids.size(1))
    if len(offsets) != seq_len:
        raise ValueError("offset/token length mismatch")

    caller_chars, caller_diag = literal_character_weights(text)
    helper_chars, helper_diag = helper_character_weights(text, language, runtime)
    caller_weights_np = normalized_token_weights(text, offsets, caller_chars)
    helper_weights_np = normalized_token_weights(text, offsets, helper_chars)

    layers = _layer_modules(model)
    if not layers:
        raise RuntimeError("could not resolve transformer layers")
    hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0))
    if hidden_size <= 0:
        raise RuntimeError("hidden size unavailable")

    activation_storage: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != seq_len:
                raise RuntimeError("unexpected layer output shape")
            caller_weights = torch.as_tensor(
                caller_weights_np,
                dtype=torch.float32,
                device=value.device,
            ).view(1, seq_len, 1)
            helper_weights = torch.as_tensor(
                helper_weights_np,
                dtype=torch.float32,
                device=value.device,
            ).view(1, seq_len, 1)
            # Current author implementation uses an ordinary unmasked mean; with
            # one unpadded sample this is identical to a valid-token mean.
            mean = value.mean(dim=1)
            caller_weighted = (value * caller_weights).sum(dim=1) / (caller_weights.sum(dim=1) + 1e-8)
            helper_weighted = (value * helper_weights).sum(dim=1) / (helper_weights.sum(dim=1) + 1e-8)
            triplet = tuple(
                tensor[0].detach().float().cpu().numpy().astype(np.float32, copy=False)
                for tensor in (mean, caller_weighted, helper_weighted)
            )
            validate_activation_triplet(*triplet, expected_hidden_dim=hidden_size)
            activation_storage[layer_index] = triplet
        return hook

    for layer_index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(layer_index)))

    try:
        device = next(model.parameters()).device
        model_inputs = {key: value.to(device) for key, value in encodings.items()}
        started = time.perf_counter()
        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=False)
        forward_seconds = time.perf_counter() - started
        if len(activation_storage) != len(layers):
            raise RuntimeError("not every transformer layer produced an activation")
        logits = outputs.logits[:, :-1]
        targets = model_inputs["input_ids"][:, 1:].to(logits.device)
        z = correct_z_from_logits(logits, targets, chunk_tokens=config.z_chunk_tokens)
        if len(z) != seq_len - 1:
            raise RuntimeError("correct-z target length mismatch")
        target_caller = caller_weights_np[1:]
        target_helper = helper_weights_np[1:]
        output_scores = {
            "zsigmoid_single_sequence": float(sigmoid_z(z).mean()),
            "caller_literal_output_weighted_single_sequence": weighted_output_score(z, target_caller),
            "helper_language_aware_output_weighted_single_sequence": weighted_output_score(z, target_helper),
        }
        layer_arrays = {
            layer_index: {
                "mean": triplet[0],
                "caller_weighted": triplet[1],
                "helper_weighted": triplet[2],
            }
            for layer_index, triplet in activation_storage.items()
        }
        return {
            "seq_len": seq_len,
            "forward_seconds": float(forward_seconds),
            "hidden_size": hidden_size,
            "num_layers": len(layers),
            "activations": layer_arrays,
            "output_scores": output_scores,
            "caller_diagnostics": asdict(caller_diag),
            "helper_diagnostics": asdict(helper_diag),
        }
    finally:
        for handle in handles:
            handle.remove()


def run_runtime_pilot(
    model: Any,
    tokenizer: Any,
    cohort: pd.DataFrame,
    runtime: Any,
    config: LumiaRuntimePilotConfig,
) -> dict[str, Any]:
    """Execute the no-performance 50-row fidelity/runtime gate."""
    import torch

    if list(cohort.columns) != ["sample_id", "language", "content"]:
        raise ValueError("model-scoring cohort must physically exclude labels")
    if len(cohort) != config.rows or cohort.sample_id.duplicated().any():
        raise ValueError("runtime cohort identity mismatch")

    sample_seconds: list[float] = []
    token_counts: list[int] = []
    expected_layers: int | None = None
    expected_hidden: int | None = None
    helper_tree_sitter = 0
    helper_flake8 = 0
    caller_generic = 0
    output_score_finite = 0
    started = time.perf_counter()

    for row in cohort.itertuples(index=False):
        result = extract_one_sample(model, tokenizer, row.content, row.language, runtime, config)
        token_counts.append(int(result["seq_len"]))
        sample_seconds.append(float(result["forward_seconds"]))
        expected_layers = result["num_layers"] if expected_layers is None else expected_layers
        expected_hidden = result["hidden_size"] if expected_hidden is None else expected_hidden
        if result["num_layers"] != expected_layers or result["hidden_size"] != expected_hidden:
            raise RuntimeError("activation dimensions changed across samples")
        caller_generic += int(result["caller_diagnostics"]["linter_backend"].startswith("generic"))
        helper_tree_sitter += int(result["helper_diagnostics"]["ast_backend"] == "tree_sitter")
        helper_flake8 += int(result["helper_diagnostics"]["linter_backend"] == "flake8")
        if all(math.isfinite(float(value)) for value in result["output_scores"].values()):
            output_score_finite += 1
        # Runtime gate intentionally discards all activations and scores after
        # shape/finite validation; nothing sample-level is persisted.
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wall_seconds = time.perf_counter() - started
    if caller_generic != config.rows:
        raise RuntimeError("caller-literal path was not generic for every runtime sample")
    if output_score_finite != config.rows:
        raise RuntimeError("same-input output baseline contained a nonfinite sample")

    token_array = np.asarray(token_counts, dtype=np.int64)
    forward_array = np.asarray(sample_seconds, dtype=np.float64)
    estimated_1000_minutes_wall = float(wall_seconds * (1000.0 / config.rows) / 60.0)
    estimated_1000_minutes_forward = float(forward_array.sum() * (1000.0 / config.rows) / 60.0)
    recommendation = (
        "direct_1000_row_t4_feasible_under_120m_gate"
        if estimated_1000_minutes_wall <= 105.0
        else "do_not_launch_1000_rows_on_same_single_t4_protocol_without_resource_or_scope_revision"
    )
    return {
        "schema_version": 1,
        "task_id": "P1-03-lumia-hidden-state-runtime-pilot-50-v1",
        "author_repository": AUTHOR_REPOSITORY,
        "author_commit": AUTHOR_COMMIT,
        "config": asdict(config),
        "rows": config.rows,
        "labels_present_during_model_scoring": False,
        "performance_metrics_computed": False,
        "num_layers": int(expected_layers or 0),
        "hidden_size": int(expected_hidden or 0),
        "token_count": {
            "min": int(token_array.min()),
            "median": float(np.median(token_array)),
            "max": int(token_array.max()),
            "sum": int(token_array.sum()),
            "truncated_at_max_length": int((token_array == config.max_length).sum()),
        },
        "timing": {
            "wall_seconds": float(wall_seconds),
            "forward_seconds_sum": float(forward_array.sum()),
            "forward_seconds_median": float(np.median(forward_array)),
            "forward_seconds_max": float(forward_array.max()),
            "estimated_1000_minutes_wall_linear": estimated_1000_minutes_wall,
            "estimated_1000_minutes_forward_linear": estimated_1000_minutes_forward,
        },
        "diagnostics": {
            "caller_generic_linter_samples": int(caller_generic),
            "helper_tree_sitter_samples": int(helper_tree_sitter),
            "helper_flake8_samples": int(helper_flake8),
            "finite_output_baseline_samples": int(output_score_finite),
        },
        "runtime_recommendation": recommendation,
        "persistent_sample_level_outputs": False,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
