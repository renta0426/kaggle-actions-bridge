"""Label-free matched-target context-contrast planning and reductions.

The module deliberately does not accept membership labels, sample identifiers,
dataset identifiers, or model identifiers.  It selects at most three target spans
from UTF-8 byte anchors, tokenizes the full source exactly once, and constructs
32/128/512-token left-context conditions by slicing the same token-id sequence.
Consequently every condition for a span scores the exact same target token IDs.

Model inference is intentionally outside this CPU reference.  A later scorer may
consume :class:`ContextWindow` records, obtain per-target log-p and paper-Min-K++
normalized values, then use the fixed reductions below.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ContextContrastConfig:
    """Frozen planning/reduction contract for the P2-01 CPU reference."""

    context_budgets: tuple[int, ...] = (32, 128, 512)
    target_tokens: int = 256
    min_target_tokens: int = 64
    spans_per_file: int = 3
    anchor_fractions: tuple[float, ...] = (0.25, 0.50, 0.75)
    local_width: int = 64
    paper_minkpp_fraction: float = 0.10

    def __post_init__(self) -> None:
        if not self.context_budgets:
            raise ValueError("context_budgets must be non-empty")
        if tuple(sorted(set(self.context_budgets))) != self.context_budgets:
            raise ValueError("context_budgets must be strictly increasing and unique")
        if any(type(value) is not int or value < 1 for value in self.context_budgets):
            raise ValueError("context budgets must be positive integers")
        if type(self.target_tokens) is not int or self.target_tokens < 1:
            raise ValueError("target_tokens must be positive")
        if type(self.min_target_tokens) is not int or not 1 <= self.min_target_tokens <= self.target_tokens:
            raise ValueError("min_target_tokens must be in [1, target_tokens]")
        if type(self.spans_per_file) is not int or self.spans_per_file < 1:
            raise ValueError("spans_per_file must be positive")
        if len(self.anchor_fractions) < self.spans_per_file:
            raise ValueError("anchor_fractions must cover spans_per_file")
        if any(not 0.0 < float(value) < 1.0 for value in self.anchor_fractions):
            raise ValueError("anchor fractions must be in (0, 1)")
        if tuple(sorted(self.anchor_fractions)) != self.anchor_fractions:
            raise ValueError("anchor_fractions must be ordered")
        if type(self.local_width) is not int or self.local_width < 1:
            raise ValueError("local_width must be positive")
        if not 0.0 < float(self.paper_minkpp_fraction) <= 1.0:
            raise ValueError("paper_minkpp_fraction must be in (0, 1]")


@dataclass(frozen=True)
class TokenizedDocument:
    input_ids: tuple[int, ...]
    byte_offsets: tuple[tuple[int, int], ...]
    utf8_bytes: int


@dataclass(frozen=True)
class TargetSpan:
    span_index: int
    anchor_fraction: float
    anchor_byte: int
    target_start_token: int
    target_end_token: int
    target_start_byte: int
    target_end_byte: int
    available_context_budgets: tuple[int, ...]

    @property
    def target_token_count(self) -> int:
        return self.target_end_token - self.target_start_token


@dataclass(frozen=True)
class ContextWindow:
    span_index: int
    context_budget: int
    context_start_token: int
    source_target_start_token: int
    source_target_end_token: int
    target_start_in_window: int
    target_end_in_window: int
    input_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]


def _char_to_byte_boundaries(content: str) -> list[int]:
    boundaries = [0]
    total = 0
    for character in content:
        total += len(character.encode("utf-8"))
        boundaries.append(total)
    return boundaries


def _flatten_single(value: Any, *, field: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"tokenizer returned batched {field}")
        value = value[0]
    return list(value)


def tokenize_with_byte_offsets(tokenizer: Any, content: str) -> TokenizedDocument:
    """Tokenize once and convert fast-tokenizer character offsets to UTF-8 bytes."""

    if not isinstance(content, str) or not content:
        raise ValueError("content must be a non-empty string")
    encoded = tokenizer(
        content,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    if "input_ids" not in encoded or "offset_mapping" not in encoded:
        raise ValueError("tokenizer must return input_ids and offset_mapping")
    input_ids = [int(value) for value in _flatten_single(encoded["input_ids"], field="input_ids")]
    offsets = _flatten_single(encoded["offset_mapping"], field="offset_mapping")
    if len(input_ids) != len(offsets) or not input_ids:
        raise ValueError("token IDs and offset mapping must be non-empty and aligned")

    boundaries = _char_to_byte_boundaries(content)
    byte_offsets: list[tuple[int, int]] = []
    for index, pair in enumerate(offsets):
        if len(pair) != 2:
            raise ValueError(f"invalid offset pair at token {index}")
        start, end = int(pair[0]), int(pair[1])
        if not 0 <= start <= end <= len(content):
            raise ValueError(f"offset outside source at token {index}")
        if start == end:
            raise ValueError("zero-width token offset is unsupported without special tokens")
        byte_offsets.append((boundaries[start], boundaries[end]))
    return TokenizedDocument(
        input_ids=tuple(input_ids),
        byte_offsets=tuple(byte_offsets),
        utf8_bytes=boundaries[-1],
    )


def _token_index_for_anchor(byte_offsets: Sequence[tuple[int, int]], anchor_byte: int) -> int:
    for index, (start, end) in enumerate(byte_offsets):
        if start <= anchor_byte < end or end > anchor_byte:
            return index
    return len(byte_offsets) - 1


def select_target_spans(
    document: TokenizedDocument,
    config: ContextContrastConfig | None = None,
) -> list[TargetSpan]:
    """Select at most three deterministic spans from byte-position anchors.

    Files unable to provide even the smallest context plus ``min_target_tokens``
    return no spans.  Larger context budgets are included only when the target
    start has that many preceding tokens; they are never silently imputed.
    """

    cfg = config or ContextContrastConfig()
    token_count = len(document.input_ids)
    smallest_context = cfg.context_budgets[0]
    latest_start = token_count - cfg.min_target_tokens
    if latest_start < smallest_context:
        return []

    selected: list[TargetSpan] = []
    seen_starts: set[int] = set()
    for fraction in cfg.anchor_fractions[: cfg.spans_per_file]:
        anchor = min(document.utf8_bytes - 1, max(0, int(math.floor(document.utf8_bytes * fraction))))
        raw_start = _token_index_for_anchor(document.byte_offsets, anchor)
        target_start = min(max(raw_start, smallest_context), latest_start)
        if target_start in seen_starts:
            continue
        target_end = min(token_count, target_start + cfg.target_tokens)
        if target_end - target_start < cfg.min_target_tokens:
            continue
        available = tuple(value for value in cfg.context_budgets if value <= target_start)
        if not available:
            continue
        target_byte_start = document.byte_offsets[target_start][0]
        target_byte_end = document.byte_offsets[target_end - 1][1]
        selected.append(
            TargetSpan(
                span_index=len(selected),
                anchor_fraction=float(fraction),
                anchor_byte=anchor,
                target_start_token=target_start,
                target_end_token=target_end,
                target_start_byte=target_byte_start,
                target_end_byte=target_byte_end,
                available_context_budgets=available,
            )
        )
        seen_starts.add(target_start)
    return selected


def build_context_windows(
    document: TokenizedDocument,
    spans: Sequence[TargetSpan],
) -> list[ContextWindow]:
    """Create prefix-truncated conditions while keeping target IDs identical."""

    windows: list[ContextWindow] = []
    for span in spans:
        target_ids = document.input_ids[span.target_start_token : span.target_end_token]
        if not target_ids:
            raise ValueError("target span is empty")
        for budget in span.available_context_budgets:
            context_start = span.target_start_token - budget
            window_ids = document.input_ids[context_start : span.target_end_token]
            target_start_in_window = budget
            target_end_in_window = budget + len(target_ids)
            if window_ids[target_start_in_window:target_end_in_window] != target_ids:
                raise RuntimeError("shared target token identity failure")
            windows.append(
                ContextWindow(
                    span_index=span.span_index,
                    context_budget=budget,
                    context_start_token=context_start,
                    source_target_start_token=span.target_start_token,
                    source_target_end_token=span.target_end_token,
                    target_start_in_window=target_start_in_window,
                    target_end_in_window=target_end_in_window,
                    input_ids=tuple(window_ids),
                    target_token_ids=tuple(target_ids),
                )
            )
    return windows


def _best_local(values: np.ndarray, width: int) -> float:
    width = min(width, len(values))
    if width == len(values):
        return float(values.mean())
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    means = (prefix[width:] - prefix[:-width]) / width
    return float(means.max())


def summarize_shared_target_scores(
    logp: Sequence[float],
    paper_z: Sequence[float],
    config: ContextContrastConfig | None = None,
) -> dict[str, float]:
    """Reduce one condition's exact shared-target token scores to fixed scalars."""

    cfg = config or ContextContrastConfig()
    logp_values = np.asarray(logp, dtype=float)
    z_values = np.asarray(paper_z, dtype=float)
    if logp_values.ndim != 1 or z_values.ndim != 1 or len(logp_values) != len(z_values) or len(logp_values) == 0:
        raise ValueError("logp and paper_z must be aligned non-empty vectors")
    if not np.isfinite(logp_values).all() or not np.isfinite(z_values).all():
        raise ValueError("target scores must be finite")
    k = max(1, int(math.ceil(len(z_values) * cfg.paper_minkpp_fraction)))
    bottom = np.argpartition(z_values, k - 1)[:k]
    return {
        "mean_logp": float(logp_values.mean()),
        "paper_minkpp10": float(z_values[bottom].mean()),
        "local64": _best_local(logp_values, cfg.local_width),
        "target_tokens": float(len(logp_values)),
    }


def contrast_deltas(condition_summaries: Mapping[int, Mapping[str, float]]) -> dict[str, float]:
    """Compute fixed positive-orientation context deltas for available pairs.

    Positive values mean the identical target becomes easier under the longer
    left context.  Missing 512-token context is not replaced with a 128-token
    score; the 128-vs-32 fallback is emitted under distinct feature names.
    """

    out: dict[str, float] = {}
    for long_budget, short_budget in ((512, 32), (128, 32)):
        if long_budget not in condition_summaries or short_budget not in condition_summaries:
            continue
        left = condition_summaries[long_budget]
        right = condition_summaries[short_budget]
        if int(left["target_tokens"]) != int(right["target_tokens"]):
            raise ValueError("context conditions do not score the same number of target tokens")
        for metric in ("mean_logp", "paper_minkpp10", "local64"):
            out[f"delta_{long_budget}_{short_budget}_{metric}"] = float(left[metric] - right[metric])
    return out


def aggregate_span_contrasts(span_contrasts: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Aggregate up to three span deltas without mixing fallback feature names."""

    keys = sorted({key for row in span_contrasts for key in row})
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in span_contrasts if key in row], dtype=float)
        if not len(values):
            continue
        if not np.isfinite(values).all():
            raise ValueError("non-finite span contrast")
        out[f"{key}__max_span"] = float(values.max())
        out[f"{key}__mean_span"] = float(values.mean())
        out[f"{key}__span_coverage"] = float(len(values))
    return out
