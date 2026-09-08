"""Paper-faithful Min-K++ token scoring.

This module intentionally keeps the historical uniform-vocabulary z-score out of
the implementation.  ``prob_weighted_minkpp_v1`` is a new schema: the model's
own next-token distribution weights the mean and variance, matching the
Min-K++ definition.  Functions have no top-level torch import so the base
package remains CPU-light.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math

import numpy as np


MINKPP_PAPER_SCHEMA_VERSION = "prob_weighted_minkpp_v1"
DEFAULT_VARIANCE_FLOOR = 1e-12
DEFAULT_VOCAB_BLOCK_SIZE = 8_192
MINK_FRACTIONS = (0.01, 0.05, 0.10, 0.20)


@dataclass(frozen=True)
class MinkppDiagnostics:
    schema_version: str
    reduction: str
    valid_token_count: int
    variance_clamp_count: int
    min_variance_before_clamp: float
    max_abs_z: float
    accumulation_dtype: str
    vocab_block_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MinkppTokenStatistics:
    standardized_log_prob: Any
    probability_weighted_variance: Any
    diagnostics: MinkppDiagnostics


def _validate_shapes(logits: Any, targets: Any, valid_mask: Any | None) -> None:
    if logits.ndim < 2:
        raise ValueError("logits must have shape [..., tokens, vocabulary]")
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"target shape {tuple(targets.shape)} does not match logits prefix "
            f"{tuple(logits.shape[:-1])}"
        )
    if valid_mask is not None and valid_mask.shape != targets.shape:
        raise ValueError("valid_mask must have the same shape as targets")
    if logits.shape[-1] < 2:
        raise ValueError("vocabulary dimension must contain at least two entries")


def _finalize(
    *,
    target_shifted: Any,
    mean_shifted: Any,
    variance: Any,
    valid_mask: Any | None,
    variance_floor: float,
    reduction: str,
    accumulation_dtype: str,
    vocab_block_size: int | None,
) -> MinkppTokenStatistics:
    import torch

    if not math.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be finite and positive")

    if valid_mask is None:
        valid = torch.ones_like(target_shifted, dtype=torch.bool)
    else:
        valid = valid_mask.to(device=target_shifted.device, dtype=torch.bool)

    variance = torch.clamp(variance, min=0.0)
    clamped = variance < variance_floor
    safe_variance = variance.clamp_min(variance_floor)
    z = (target_shifted - mean_shifted) / safe_variance.sqrt()

    # Invalid/padded positions are not scores.  Setting them to zero keeps the
    # returned tensors finite while the explicit mask remains the authority.
    z = torch.where(valid, z, torch.zeros_like(z))
    out_variance = torch.where(valid, variance, torch.zeros_like(variance))

    valid_variance = variance[valid]
    valid_z = z[valid]
    diagnostics = MinkppDiagnostics(
        schema_version=MINKPP_PAPER_SCHEMA_VERSION,
        reduction=reduction,
        valid_token_count=int(valid.sum().item()),
        variance_clamp_count=int((clamped & valid).sum().item()),
        min_variance_before_clamp=(
            float(valid_variance.min().item()) if valid_variance.numel() else float("nan")
        ),
        max_abs_z=float(valid_z.abs().max().item()) if valid_z.numel() else 0.0,
        accumulation_dtype=accumulation_dtype,
        vocab_block_size=vocab_block_size,
    )
    return MinkppTokenStatistics(z, out_variance, diagnostics)


def probability_weighted_token_statistics_dense(
    logits: Any,
    targets: Any,
    *,
    valid_mask: Any | None = None,
    variance_floor: float = DEFAULT_VARIANCE_FLOOR,
    accumulation_dtype: Any | None = None,
) -> MinkppTokenStatistics:
    """Reference-like dense GPU/CPU implementation.

    The centered standardized score is invariant to the log-softmax additive
    normalizer, so probability-weighted moments may be taken over shifted
    logits.  The weights are still the model softmax probabilities.
    """
    import torch

    _validate_shapes(logits, targets, valid_mask)
    dtype = accumulation_dtype or torch.float32
    values = logits.to(dtype=dtype)
    shifted = values - values.amax(dim=-1, keepdim=True)
    probabilities = torch.softmax(shifted, dim=-1)
    mean = (probabilities * shifted).sum(dim=-1)
    variance = (probabilities * (shifted - mean.unsqueeze(-1)).square()).sum(dim=-1)
    target_shifted = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return _finalize(
        target_shifted=target_shifted,
        mean_shifted=mean,
        variance=variance,
        valid_mask=valid_mask,
        variance_floor=variance_floor,
        reduction="dense",
        accumulation_dtype=str(dtype),
        vocab_block_size=None,
    )


def probability_weighted_token_statistics_blocked(
    logits: Any,
    targets: Any,
    *,
    valid_mask: Any | None = None,
    variance_floor: float = DEFAULT_VARIANCE_FLOOR,
    vocab_block_size: int = DEFAULT_VOCAB_BLOCK_SIZE,
    accumulation_dtype: Any | None = None,
) -> MinkppTokenStatistics:
    """Memory-bounded paper-faithful implementation.

    The vocabulary is traversed in blocks.  Float32 (by default) accumulation
    is used, and no full-vocabulary float32 softmax tensor is materialized.
    """
    import torch

    _validate_shapes(logits, targets, valid_mask)
    if vocab_block_size <= 0:
        raise ValueError("vocab_block_size must be positive")
    dtype = accumulation_dtype or torch.float32
    shape = logits.shape[:-1]
    device = logits.device

    max_logits = torch.full(shape, -torch.inf, dtype=dtype, device=device)
    for start in range(0, logits.shape[-1], vocab_block_size):
        block = logits[..., start : start + vocab_block_size].to(dtype=dtype)
        max_logits = torch.maximum(max_logits, block.amax(dim=-1))

    normalizer = torch.zeros(shape, dtype=dtype, device=device)
    first_moment_numerator = torch.zeros_like(normalizer)
    for start in range(0, logits.shape[-1], vocab_block_size):
        block = logits[..., start : start + vocab_block_size].to(dtype=dtype)
        shifted = block - max_logits.unsqueeze(-1)
        weight = shifted.exp()
        normalizer += weight.sum(dim=-1)
        first_moment_numerator += (weight * shifted).sum(dim=-1)
    mean = first_moment_numerator / normalizer

    variance_numerator = torch.zeros_like(normalizer)
    for start in range(0, logits.shape[-1], vocab_block_size):
        block = logits[..., start : start + vocab_block_size].to(dtype=dtype)
        shifted = block - max_logits.unsqueeze(-1)
        weight = shifted.exp()
        variance_numerator += (weight * (shifted - mean.unsqueeze(-1)).square()).sum(dim=-1)
    variance = variance_numerator / normalizer

    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).to(dtype=dtype)
    target_shifted = target_logits - max_logits
    return _finalize(
        target_shifted=target_shifted,
        mean_shifted=mean,
        variance=variance,
        valid_mask=valid_mask,
        variance_floor=variance_floor,
        reduction="blocked",
        accumulation_dtype=str(dtype),
        vocab_block_size=int(vocab_block_size),
    )


def bottom_fraction_mean(values: np.ndarray, fraction: float) -> float:
    """Mean of the lowest fraction, using the existing ceil convention."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, math.ceil(array.size * fraction))
    indices = np.argpartition(array, count - 1)[:count]
    return float(array[indices].mean())


def summarize_minkpp(values: np.ndarray) -> dict[str, float]:
    """Return the frozen pilot Min-K++ fractions."""
    return {
        f"minkpp_{int(round(fraction * 100)):02d}": bottom_fraction_mean(values, fraction)
        for fraction in MINK_FRACTIONS
    }
