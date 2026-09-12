"""Label-free, model-agnostic Stage 2 membership scoring.

The public entry point deliberately receives an already loaded causal language
model and tokenizer. It has no dataset, model-id, Kaggle-path, sample-id, or
label dependency. The implementation extracts the strongest attack families
that survived the StarCoder2-3B -> 7B transfer check (local span likelihood,
standard Min-K++, multi-window likelihood, and rank statistics), calibrates them
with label-free mid-ranks, and returns scores in the caller's original order.

This module does not claim that a fixed unsupervised ensemble is optimal for an
unknown target. It is the conservative Stage 2 baseline against which any
architecture-specific adaptation must be justified without target labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


STAGE2_METHOD_VERSION = "stage2-model-independent-v1"
_TOKEN_STAT_NAMES = ("logp", "correct_z", "rank", "margin", "entropy")


class Stage2ScoringError(RuntimeError):
    """Base exception for deterministic Stage 2 scoring failures."""


class Stage2BudgetExceeded(Stage2ScoringError):
    """Raised before starting another batch when the fixed runtime budget expired."""


class Stage2FidelityError(Stage2ScoringError):
    """Raised when blocked/chunked statistics differ from the exact pilot."""


@dataclass(frozen=True)
class Stage2RuntimeConfig:
    """Runtime and fidelity limits for :func:`score_samples`.

    ``device='auto'`` respects the device on the model's input embeddings. This
    also works for Accelerate/Hugging Face sharded models because inputs are sent
    to the embedding device and targets are moved to the logits device. An
    explicit device only moves an unsharded model when ``move_model=True``.
    """

    max_length: int = 768
    max_batch_tokens: int = 4_096
    vocab_chunk_tokens: int = 64
    rank_vocab_block_size: int = 8_192
    min_k_percents: tuple[int, ...] = (1, 2, 5, 10, 20, 30)
    local_widths: tuple[int, ...] = (32, 64, 128)
    language_calibration_min_rows: int = 20
    length_calibration_min_rows: int = 12
    fidelity_gate: bool = True
    fidelity_tokens: int = 24
    fidelity_atol: float = 1e-5
    device: str = "auto"
    move_model: bool = False
    time_budget_seconds: float | None = None
    max_peak_vram_bytes: int | None = None

    def __post_init__(self) -> None:
        positive = {
            "max_length": self.max_length,
            "max_batch_tokens": self.max_batch_tokens,
            "vocab_chunk_tokens": self.vocab_chunk_tokens,
            "rank_vocab_block_size": self.rank_vocab_block_size,
            "language_calibration_min_rows": self.language_calibration_min_rows,
            "length_calibration_min_rows": self.length_calibration_min_rows,
            "fidelity_tokens": self.fidelity_tokens,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.min_k_percents or any(not 0 < int(value) <= 100 for value in self.min_k_percents):
            raise ValueError("min_k_percents must contain values in [1, 100]")
        if len(set(self.min_k_percents)) != len(self.min_k_percents):
            raise ValueError("min_k_percents must be unique")
        if not self.local_widths or any(int(value) <= 0 for value in self.local_widths):
            raise ValueError("local_widths must contain positive values")
        if len(set(self.local_widths)) != len(self.local_widths):
            raise ValueError("local_widths must be unique")
        if self.fidelity_atol < 0:
            raise ValueError("fidelity_atol must be non-negative")
        if self.time_budget_seconds is not None and self.time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds must be positive when set")
        if self.max_peak_vram_bytes is not None and self.max_peak_vram_bytes <= 0:
            raise ValueError("max_peak_vram_bytes must be positive when set")


@dataclass(frozen=True)
class Stage2ScoreResult:
    """Detailed deterministic result; ``score_samples`` returns only ``scores``."""

    scores: np.ndarray
    features: pd.DataFrame
    manifest: Mapping[str, Any]


def _normalise_inputs(samples: Sequence[str], languages: Sequence[str]) -> tuple[list[str], list[str]]:
    if isinstance(samples, (str, bytes)) or isinstance(languages, (str, bytes)):
        raise TypeError("samples and languages must be sequences, not scalar strings")
    content = list(samples)
    language = [str(value) for value in languages]
    if not content:
        raise ValueError("samples must not be empty")
    if len(content) != len(language):
        raise ValueError("samples and languages must have equal length")
    for index, value in enumerate(content):
        if not isinstance(value, str):
            raise TypeError(f"sample {index} is not a string")
    for index, value in enumerate(language):
        if not value.strip():
            raise ValueError(f"language {index} is empty")
    return content, language


def _input_ids(tokenizer: Any, content: str) -> list[int]:
    encoded = tokenizer(
        content,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise Stage2ScoringError("tokenizer returned an unexpected batched value")
        values = values[0]
    result = [int(value) for value in values]
    if len(result) < 2:
        raise Stage2ScoringError("every sample must contain at least two tokens")
    return result


def _window_starts(token_count: int, max_length: int) -> list[tuple[str, int]]:
    if token_count <= max_length:
        return [("whole", 0)]
    candidates = (
        ("prefix", 0),
        ("middle", (token_count - max_length) // 2),
        ("suffix", token_count - max_length),
    )
    result: list[tuple[str, int]] = []
    seen: set[int] = set()
    for name, start in candidates:
        if start not in seen:
            result.append((name, start))
            seen.add(start)
    return result


def _make_windows(
    samples: Sequence[str],
    languages: Sequence[str],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample_index, (content, language) in enumerate(zip(samples, languages)):
        ids = _input_ids(tokenizer, content)
        for position, start in _window_starts(len(ids), max_length):
            window = ids[start : start + max_length]
            records.append(
                {
                    "sample_index": sample_index,
                    "language": language,
                    "position": position,
                    "window_start": start,
                    "file_token_count": len(ids),
                    "window_token_count": len(window),
                    "input_ids": window,
                }
            )
    return records


def _dynamic_batches(records: Sequence[dict[str, Any]], max_batch_tokens: int):
    ordered = sorted(records, key=lambda row: (row["window_token_count"], row["sample_index"], row["window_start"]))
    batch: list[dict[str, Any]] = []
    width = 0
    for record in ordered:
        candidate_width = max(width, int(record["window_token_count"]))
        if batch and candidate_width * (len(batch) + 1) > max_batch_tokens:
            yield batch
            batch = []
            width = 0
        batch.append(record)
        width = max(width, int(record["window_token_count"]))
    if batch:
        yield batch


def _embedding_device(model: Any, torch: Any):
    if not hasattr(model, "get_input_embeddings"):
        raise Stage2ScoringError("model must expose get_input_embeddings()")
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is None or not hasattr(weight, "device"):
        raise Stage2ScoringError("model input embeddings do not expose a device")
    device = weight.device
    if getattr(device, "type", None) == "meta":
        raise Stage2ScoringError("model input embeddings are still on the meta device")
    return torch.device(device)


def _resolve_device(model: Any, config: Stage2RuntimeConfig, torch: Any):
    current = _embedding_device(model, torch)
    if config.device == "auto":
        return current
    desired = torch.device(config.device)
    if desired == current:
        return current
    if not config.move_model:
        raise Stage2ScoringError(
            f"model is on {current}, requested {desired}; set move_model=True or load the model on the target device"
        )
    if getattr(model, "hf_device_map", None):
        raise Stage2ScoringError("cannot move an already sharded model")
    model.to(desired)
    return _embedding_device(model, torch)


def _extract_logits(output: Any):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise Stage2ScoringError("model output does not contain logits")


def _blocked_rank(chunk: Any, correct: Any, block_size: int, torch: Any):
    rank = torch.ones_like(correct, dtype=torch.int64)
    threshold = correct.unsqueeze(-1)
    for start in range(0, chunk.shape[-1], block_size):
        stop = min(start + block_size, chunk.shape[-1])
        rank += (chunk[..., start:stop] > threshold).sum(dim=-1)
    return rank


def _token_statistics_fast(logits: Any, targets: Any, config: Stage2RuntimeConfig, torch: Any):
    statistics = torch.empty(
        (*logits.shape[:2], len(_TOKEN_STAT_NAMES)),
        dtype=torch.float32,
        device=logits.device,
    )
    for start in range(0, logits.shape[1], config.vocab_chunk_tokens):
        stop = min(start + config.vocab_chunk_tokens, logits.shape[1])
        chunk = logits[:, start:stop].float()
        target = targets[:, start:stop]
        correct = chunk.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        log_norm = torch.logsumexp(chunk, dim=-1)
        logp = correct - log_norm
        z = (correct - chunk.mean(dim=-1)) / chunk.std(dim=-1, correction=0).clamp_min(1e-6)
        rank = _blocked_rank(chunk, correct, config.rank_vocab_block_size, torch)
        top2 = chunk.topk(k=2, dim=-1).values
        best_other = torch.where(top2[..., 0] == correct, top2[..., 1], top2[..., 0])
        margin = correct - best_other
        probabilities = torch.softmax(chunk, dim=-1)
        entropy = log_norm - (probabilities * chunk).sum(dim=-1)
        statistics[:, start:stop] = torch.stack(
            (logp, z, rank.float(), margin, entropy), dim=-1
        )
        del chunk, probabilities
    return statistics


def _token_statistics_exact(logits: Any, targets: Any, torch: Any):
    chunk = logits.float()
    correct = chunk.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    log_norm = torch.logsumexp(chunk, dim=-1)
    logp = correct - log_norm
    z = (correct - chunk.mean(dim=-1)) / chunk.std(dim=-1, correction=0).clamp_min(1e-6)
    rank = 1 + (chunk > correct.unsqueeze(-1)).sum(dim=-1)
    top2 = chunk.topk(k=2, dim=-1).values
    best_other = torch.where(top2[..., 0] == correct, top2[..., 1], top2[..., 0])
    margin = correct - best_other
    probabilities = torch.softmax(chunk, dim=-1)
    entropy = log_norm - (probabilities * chunk).sum(dim=-1)
    return torch.stack((logp, z, rank.float(), margin, entropy), dim=-1)


def _fidelity_check(
    logits: Any,
    targets: Any,
    fast: Any,
    config: Stage2RuntimeConfig,
    torch: Any,
) -> dict[str, Any]:
    token_count = min(int(logits.shape[1]), int(config.fidelity_tokens))
    exact = _token_statistics_exact(logits[:1, :token_count], targets[:1, :token_count], torch)
    candidate = fast[:1, :token_count]
    difference = (exact - candidate).abs()
    per_stat = {
        name: float(difference[..., index].max().detach().cpu())
        for index, name in enumerate(_TOKEN_STAT_NAMES)
    }
    rank_equal = bool(torch.equal(exact[..., 2].to(torch.int64), candidate[..., 2].to(torch.int64)))
    maximum = max(per_stat.values(), default=0.0)
    passed = rank_equal and maximum <= config.fidelity_atol
    result = {
        "passed": passed,
        "atol": config.fidelity_atol,
        "tokens": token_count,
        "max_absolute_difference": maximum,
        "rank_exact": rank_equal,
        "per_stat": per_stat,
    }
    if not passed:
        raise Stage2FidelityError(f"fast token statistics failed fidelity gate: {result}")
    return result


def _lowest_mean(values: np.ndarray, percent: int) -> float:
    k = max(1, math.ceil(len(values) * percent / 100))
    indices = np.argpartition(values, k - 1)[:k]
    return float(values[indices].mean())


def _best_local(values: np.ndarray, width: int) -> float:
    if len(values) <= width:
        return float(values.mean())
    sums = np.convolve(values.astype(np.float64, copy=False), np.ones(width, dtype=np.float64), mode="valid")
    return float((sums / width).max())


def _summarise_tokens(arrays: Mapping[str, np.ndarray], config: Stage2RuntimeConfig) -> dict[str, float]:
    logp = np.asarray(arrays["logp"], dtype=np.float64)
    correct_z = np.asarray(arrays["correct_z"], dtype=np.float64)
    rank = np.asarray(arrays["rank"], dtype=np.float64)
    margin = np.asarray(arrays["margin"], dtype=np.float64)
    entropy = np.asarray(arrays["entropy"], dtype=np.float64)
    result = {
        "score_logp_mean": float(logp.mean()),
        "score_logp_median": float(np.median(logp)),
        "correct_z_mean": float(correct_z.mean()),
        "mean_log_rank": float(np.log1p(rank).mean()),
        "top1_rate": float((rank == 1).mean()),
        "top5_rate": float((rank <= 5).mean()),
        "margin_mean": float(margin.mean()),
        "entropy_mean": float(entropy.mean()),
    }
    for percent in config.min_k_percents:
        result[f"min_k_{percent:02d}"] = _lowest_mean(logp, int(percent))
        result[f"min_kpp_zselect_{percent:02d}"] = _lowest_mean(correct_z, int(percent))
    for width in config.local_widths:
        result[f"best_local_{width}"] = _best_local(logp, int(width))
    return result


def _aggregate_windows(window_features: pd.DataFrame) -> pd.DataFrame:
    identity = ["sample_index", "language"]
    metadata = {"window_start", "window_token_count", "file_token_count"}
    feature_columns = [
        column
        for column in window_features.select_dtypes(include=[np.number]).columns
        if column not in metadata and column != "sample_index"
    ]
    grouped = window_features.groupby(identity, sort=False)
    parts = [
        grouped["file_token_count"].max().rename("token_count"),
        grouped.size().rename("window_count"),
    ]
    for column in feature_columns:
        parts.extend(
            [
                grouped[column].mean().rename(f"{column}__mean"),
                grouped[column].max().rename(f"{column}__max"),
                grouped[column].std(ddof=0).fillna(0).rename(f"{column}__std"),
            ]
        )
    return pd.concat(parts, axis=1).reset_index().sort_values("sample_index").reset_index(drop=True)


def _midrank(values: pd.Series) -> pd.Series:
    count = len(values)
    if count == 0:
        return pd.Series(index=values.index, dtype=float)
    return (values.rank(method="average") - 0.5) / count


def _calibrated_percentile(
    frame: pd.DataFrame,
    column: str,
    *,
    higher_is_member: bool,
    config: Stage2RuntimeConfig,
) -> np.ndarray:
    values = frame[column].astype(float)
    if not higher_is_member:
        values = -values
    global_rank = _midrank(values)

    language_rank = pd.Series(index=frame.index, dtype=float)
    for _, indices in frame.groupby("language", sort=False).groups.items():
        indices = list(indices)
        if len(indices) >= config.language_calibration_min_rows:
            language_rank.loc[indices] = _midrank(values.loc[indices])
        else:
            language_rank.loc[indices] = global_rank.loc[indices]

    length_bin = np.floor(np.log2(frame["token_count"].clip(lower=1).astype(float))).astype(int)
    length_rank = pd.Series(index=frame.index, dtype=float)
    calibration_groups = pd.DataFrame({"language": frame.language, "length_bin": length_bin})
    for _, indices in calibration_groups.groupby(["language", "length_bin"], sort=False).groups.items():
        indices = list(indices)
        if len(indices) >= config.length_calibration_min_rows:
            length_rank.loc[indices] = _midrank(values.loc[indices])
        else:
            length_rank.loc[indices] = language_rank.loc[indices]

    calibrated = 0.25 * global_rank + 0.50 * language_rank + 0.25 * length_rank
    return calibrated.to_numpy(dtype=float)


def _closest_feature(frame: pd.DataFrame, prefix: str, requested: int, suffix: str) -> str:
    candidates: list[tuple[int, str]] = []
    for column in frame.columns:
        if not column.startswith(prefix) or not column.endswith(suffix):
            continue
        middle = column[len(prefix) : len(column) - len(suffix)]
        try:
            value = int(middle)
        except ValueError:
            continue
        candidates.append((value, column))
    if not candidates:
        raise Stage2ScoringError(f"no feature matches {prefix}*{suffix}")
    return min(candidates, key=lambda item: (abs(item[0] - requested), item[0]))[1]


def _label_free_ensemble(frame: pd.DataFrame, config: Stage2RuntimeConfig) -> tuple[np.ndarray, dict[str, Any]]:
    local_64 = _closest_feature(frame, "best_local_", 64, "__max")
    minkpp_10 = _closest_feature(frame, "min_kpp_zselect_", 10, "__max")
    min_k_10 = _closest_feature(frame, "min_k_", 10, "__max")
    branches = {
        "local_64": _calibrated_percentile(frame, local_64, higher_is_member=True, config=config),
        "standard_minkpp_10": _calibrated_percentile(frame, minkpp_10, higher_is_member=True, config=config),
        "multi_window_logp": _calibrated_percentile(
            frame, "score_logp_mean__mean", higher_is_member=True, config=config
        ),
        "rank": _calibrated_percentile(
            frame, "mean_log_rank__mean", higher_is_member=False, config=config
        ),
    }
    core = np.mean(np.column_stack(list(branches.values())), axis=1)

    local_novelty = [
        _calibrated_percentile(
            frame, f"best_local_{width}__max", higher_is_member=True, config=config
        )
        for width in config.local_widths
    ]
    tail_percents = [percent for percent in config.min_k_percents if percent <= 10]
    if not tail_percents:
        tail_percents = [min(config.min_k_percents)]
    tail_novelty = [
        _calibrated_percentile(
            frame,
            f"min_kpp_zselect_{percent:02d}__max",
            higher_is_member=True,
            config=config,
        )
        for percent in tail_percents
    ]
    novelty = 0.5 * np.max(np.column_stack(local_novelty), axis=1) + 0.5 * np.max(
        np.column_stack(tail_novelty), axis=1
    )
    score = np.clip(0.85 * core + 0.15 * novelty, 0.0, 1.0)
    details = {
        "core_weights": {name: 0.85 / len(branches) for name in branches},
        "novelty_weight": 0.15,
        "resolved_features": {
            "local_64": local_64,
            "standard_minkpp_10": minkpp_10,
            "legacy_min_k_10_diagnostic": min_k_10,
            "multi_window_logp": "score_logp_mean__mean",
            "rank": "mean_log_rank__mean",
        },
        "calibration": "0.25 global + 0.50 language + 0.25 language-length-bin midrank",
    }
    return score, details


def _device_manifest(model: Any, device: Any) -> dict[str, Any]:
    raw_map = getattr(model, "hf_device_map", None)
    devices: set[str] = set()
    if isinstance(raw_map, Mapping):
        for value in raw_map.values():
            text = str(value)
            if text not in {"cpu", "disk", "meta"}:
                devices.add(text)
    if not devices:
        devices.add(str(device))
    return {
        "input_device": str(device),
        "model_devices": sorted(devices),
        "sharded": bool(isinstance(raw_map, Mapping) and len(raw_map) > 1),
    }


def score_samples_detailed(
    model: Any,
    tokenizer: Any,
    samples: Sequence[str],
    languages: Sequence[str],
    runtime_config: Stage2RuntimeConfig | None = None,
) -> Stage2ScoreResult:
    """Return label-free scores, features, and an audit manifest.

    The function never consumes labels or sample identifiers. Scores are
    transductively calibrated over the supplied target batch; call it once for
    the complete Stage 2 sample set rather than independently per row/shard.
    """

    import torch

    config = runtime_config or Stage2RuntimeConfig()
    content, language = _normalise_inputs(samples, languages)
    started = time.perf_counter()
    windows = _make_windows(content, language, tokenizer, config.max_length)
    batches = list(_dynamic_batches(windows, config.max_batch_tokens))
    device = _resolve_device(model, config, torch)
    if hasattr(model, "eval"):
        model.eval()

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise Stage2ScoringError("tokenizer must define pad_token_id or eos_token_id")

    if device.type == "cuda":
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)

    rows: list[dict[str, Any]] = []
    fidelity: dict[str, Any] | None = None
    with torch.inference_mode():
        for batch_index, batch in enumerate(batches):
            elapsed = time.perf_counter() - started
            if config.time_budget_seconds is not None and elapsed >= config.time_budget_seconds:
                raise Stage2BudgetExceeded(
                    f"runtime budget expired before batch {batch_index}/{len(batches)}"
                )
            width = max(int(record["window_token_count"]) for record in batch)
            input_ids = torch.full(
                (len(batch), width), int(pad_id), dtype=torch.long, device=device
            )
            attention = torch.zeros_like(input_ids)
            valid_lengths: list[int] = []
            for row_index, record in enumerate(batch):
                ids = torch.as_tensor(record["input_ids"], dtype=torch.long, device=device)
                input_ids[row_index, : len(ids)] = ids
                attention[row_index, : len(ids)] = 1
                valid_lengths.append(len(ids) - 1)

            output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
            logits = _extract_logits(output)[:, :-1]
            targets = input_ids[:, 1:].to(logits.device)
            fast = _token_statistics_fast(logits, targets, config, torch)
            if config.fidelity_gate and fidelity is None:
                fidelity = _fidelity_check(logits, targets, fast, config, torch)
            host = fast.detach().cpu().numpy()
            for row_index, (record, valid_length) in enumerate(zip(batch, valid_lengths)):
                values = host[row_index, :valid_length]
                arrays = {
                    name: values[:, offset]
                    for offset, name in enumerate(_TOKEN_STAT_NAMES)
                }
                summary = _summarise_tokens(arrays, config)
                rows.append(
                    {
                        key: value
                        for key, value in record.items()
                        if key != "input_ids"
                    }
                    | summary
                )
            del output, logits, targets, fast, host, input_ids, attention

            if device.type == "cuda" and config.max_peak_vram_bytes is not None:
                peak = max(
                    int(torch.cuda.max_memory_allocated(index))
                    for index in range(torch.cuda.device_count())
                )
                if peak > config.max_peak_vram_bytes:
                    raise Stage2BudgetExceeded(
                        f"peak CUDA allocation {peak} exceeded limit {config.max_peak_vram_bytes}"
                    )

    window_features = pd.DataFrame(rows)
    features = _aggregate_windows(window_features)
    if len(features) != len(content) or not features.sample_index.is_unique:
        raise Stage2ScoringError("sample/window aggregation coverage failed")
    scores, ensemble = _label_free_ensemble(features, config)
    if len(scores) != len(content) or not np.isfinite(scores).all():
        raise Stage2ScoringError("non-finite or incomplete Stage 2 scores")

    peak_vram = 0
    if device.type == "cuda":
        peak_vram = max(
            int(torch.cuda.max_memory_allocated(index))
            for index in range(torch.cuda.device_count())
        )
    manifest = {
        "status": "complete",
        "method_version": STAGE2_METHOD_VERSION,
        "samples": len(content),
        "windows": len(windows),
        "batches": len(batches),
        "runtime_seconds": time.perf_counter() - started,
        "runtime_config": asdict(config),
        "device": _device_manifest(model, device),
        "peak_vram_bytes": peak_vram,
        "fidelity": fidelity or {"passed": not config.fidelity_gate, "skipped": True},
        "ensemble": ensemble,
        "labels_used": False,
        "sample_ids_used": False,
        "dataset_id_used": False,
        "model_id_used": False,
        "target_batch_calibration": True,
    }
    return Stage2ScoreResult(
        scores=np.asarray(scores, dtype=np.float64),
        features=features,
        manifest=manifest,
    )


def score_samples(
    model: Any,
    tokenizer: Any,
    samples: Sequence[str],
    languages: Sequence[str],
    runtime_config: Stage2RuntimeConfig | None = None,
) -> np.ndarray:
    """Return membership scores in input order with no target-label dependency."""

    return score_samples_detailed(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        languages=languages,
        runtime_config=runtime_config,
    ).scores


__all__ = [
    "STAGE2_METHOD_VERSION",
    "Stage2BudgetExceeded",
    "Stage2FidelityError",
    "Stage2RuntimeConfig",
    "Stage2ScoreResult",
    "Stage2ScoringError",
    "score_samples",
    "score_samples_detailed",
]
