"""Controlled cross-architecture shadow-model protocol.

The protocol removes three confounders from the first Stage 2 transfer test:
training membership, sample exposure, and tokenization. Both random-initialised
models receive the exact same byte-token sequences in the exact same order.
Only the causal-LM architecture changes between the GPT-2-style and Llama-style
shadows.

This module prepares data/configuration and can instantiate small random models.
It does not start training or select a compute backend. TPU execution is kept in
a separate, auditable runtime layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SHADOW_PROTOCOL_VERSION = "controlled-shadow-protocol-v1"


class ByteCodeTokenizer:
    """Lossless UTF-8 byte tokenizer with a fixed 259-token vocabulary."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    byte_offset = 3
    vocab_size = 259
    is_fast = False
    model_max_length = 1_000_000_000

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("ByteCodeTokenizer input must be a string")
        values = [self.byte_offset + value for value in text.encode("utf-8")]
        if add_special_tokens:
            return [self.bos_token_id, *values, self.eos_token_id]
        return values

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        values: list[int] = []
        for token_id in token_ids:
            token = int(token_id)
            if token in {self.pad_token_id, self.bos_token_id, self.eos_token_id}:
                if skip_special_tokens:
                    continue
                raise ValueError("special tokens cannot be decoded as UTF-8 bytes")
            byte_value = token - self.byte_offset
            if not 0 <= byte_value <= 255:
                raise ValueError(f"token id is outside the byte vocabulary: {token}")
            values.append(byte_value)
        return bytes(values).decode("utf-8")

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool | str = False,
        return_attention_mask: bool = True,
        return_tensors: str | None = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        scalar = isinstance(text, str)
        values = [text] if scalar else list(text)
        encoded = [self.encode(value, add_special_tokens=add_special_tokens) for value in values]
        if truncation:
            if max_length is None or max_length <= 0:
                raise ValueError("positive max_length is required when truncation=True")
            encoded = [row[:max_length] for row in encoded]
        should_pad = bool(padding)
        if should_pad:
            target = max_length if padding == "max_length" else max(len(row) for row in encoded)
            if target is None or target <= 0:
                raise ValueError("positive padding length required")
            if any(len(row) > target for row in encoded):
                raise ValueError("sequence exceeds requested padding length without truncation")
            attention = [[1] * len(row) + [0] * (target - len(row)) for row in encoded]
            encoded = [row + [self.pad_token_id] * (target - len(row)) for row in encoded]
        else:
            attention = [[1] * len(row) for row in encoded]
        result: dict[str, Any] = {
            "input_ids": encoded[0] if scalar else encoded,
        }
        if return_attention_mask:
            result["attention_mask"] = attention[0] if scalar else attention
        if return_tensors is not None:
            if not should_pad and len({len(row) for row in encoded}) > 1:
                raise ValueError("tensor output requires equal-length sequences")
            if return_tensors == "np":
                result = {key: np.asarray(value, dtype=np.int64) for key, value in result.items()}
            elif return_tensors == "pt":
                import torch

                result = {key: torch.as_tensor(value, dtype=torch.long) for key, value in result.items()}
            else:
                raise ValueError("return_tensors must be None, 'np', or 'pt'")
        return result

    @property
    def specification(self) -> dict[str, Any]:
        return {
            "type": "lossless_utf8_bytes",
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "byte_offset": self.byte_offset,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.specification, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save_pretrained(self, directory: str | Path) -> tuple[str]:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "byte_tokenizer.json"
        path.write_text(
            json.dumps(self.specification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return (str(path),)

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "ByteCodeTokenizer":
        path = Path(directory) / "byte_tokenizer.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        instance = cls()
        if value != instance.specification:
            raise ValueError("byte tokenizer specification mismatch")
        return instance


@dataclass(frozen=True)
class ShadowArchitectureSpec:
    name: str
    architecture: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int | None
    max_position_embeddings: int
    vocab_size: int = ByteCodeTokenizer.vocab_size
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.architecture not in {"gpt2", "llama"}:
            raise ValueError("architecture must be 'gpt2' or 'llama'")
        positive = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.architecture == "llama":
            if self.num_key_value_heads is None or self.num_key_value_heads <= 0:
                raise ValueError("Llama requires positive num_key_value_heads")
            if self.num_attention_heads % self.num_key_value_heads:
                raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        elif self.num_key_value_heads is not None:
            raise ValueError("GPT-2 does not use num_key_value_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def transformers_config(self):
        if self.architecture == "gpt2":
            from transformers import GPT2Config

            return GPT2Config(
                vocab_size=self.vocab_size,
                n_positions=self.max_position_embeddings,
                n_ctx=self.max_position_embeddings,
                n_embd=self.hidden_size,
                n_layer=self.num_hidden_layers,
                n_head=self.num_attention_heads,
                n_inner=self.intermediate_size,
                resid_pdrop=self.dropout,
                embd_pdrop=self.dropout,
                attn_pdrop=self.dropout,
                bos_token_id=ByteCodeTokenizer.bos_token_id,
                eos_token_id=ByteCodeTokenizer.eos_token_id,
                pad_token_id=ByteCodeTokenizer.pad_token_id,
                use_cache=False,
            )
        from transformers import LlamaConfig

        return LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            attention_dropout=self.dropout,
            hidden_dropout=self.dropout,
            bos_token_id=ByteCodeTokenizer.bos_token_id,
            eos_token_id=ByteCodeTokenizer.eos_token_id,
            pad_token_id=ByteCodeTokenizer.pad_token_id,
            tie_word_embeddings=False,
            use_cache=False,
        )

    def build_model(self, *, seed: int):
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.architecture == "gpt2":
            from transformers import GPT2LMHeadModel

            return GPT2LMHeadModel(self.transformers_config())
        from transformers import LlamaForCausalLM

        return LlamaForCausalLM(self.transformers_config())

    def parameter_count(self) -> int:
        model = self.build_model(seed=0)
        count = sum(int(parameter.numel()) for parameter in model.parameters())
        del model
        return count


@dataclass(frozen=True)
class ShadowSequenceConfig:
    max_sequence_tokens: int = 256
    window_policy: str = "prefix-middle-suffix-hashed-v1"

    def __post_init__(self) -> None:
        if self.max_sequence_tokens < 4:
            raise ValueError("max_sequence_tokens must be at least 4")
        if self.window_policy != "prefix-middle-suffix-hashed-v1":
            raise ValueError("unsupported window_policy")


@dataclass(frozen=True)
class ShadowTrainingSpec:
    seed: int = 2027
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_fraction: float = 0.05
    global_batch_size: int = 64
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")


def default_shadow_pair(*, max_position_embeddings: int = 256) -> tuple[ShadowArchitectureSpec, ShadowArchitectureSpec]:
    """Return similarly sized GPT-2- and Llama-style random architectures."""

    return (
        ShadowArchitectureSpec(
            name="shadow-gpt2-byte-5m",
            architecture="gpt2",
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=None,
            max_position_embeddings=max_position_embeddings,
        ),
        ShadowArchitectureSpec(
            name="shadow-llama-byte-5m",
            architecture="llama",
            hidden_size=256,
            intermediate_size=688,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=max_position_embeddings,
        ),
    )


def validate_shadow_pair(
    left: ShadowArchitectureSpec,
    right: ShadowArchitectureSpec,
    *,
    max_parameter_ratio: float = 1.15,
) -> dict[str, Any]:
    if left.architecture == right.architecture:
        raise ValueError("shadow architectures must be different")
    if left.vocab_size != right.vocab_size:
        raise ValueError("shadow vocabularies must be identical in phase 1")
    if left.max_position_embeddings != right.max_position_embeddings:
        raise ValueError("shadow position limits must be identical in phase 1")
    if max_parameter_ratio < 1:
        raise ValueError("max_parameter_ratio must be at least 1")
    left_count = left.parameter_count()
    right_count = right.parameter_count()
    ratio = max(left_count, right_count) / min(left_count, right_count)
    if ratio > max_parameter_ratio:
        raise ValueError(
            f"shadow parameter ratio {ratio:.4f} exceeds {max_parameter_ratio:.4f}"
        )
    return {
        "left": {"spec": asdict(left), "parameters": left_count},
        "right": {"spec": asdict(right), "parameters": right_count},
        "parameter_ratio": ratio,
        "maximum_parameter_ratio": max_parameter_ratio,
        "same_tokenizer_required": True,
        "same_exposure_schedule_required": True,
    }


def _stable_uint64(seed: int, value: str) -> int:
    payload = f"{seed}\0{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _window_start(
    token_count: int,
    capacity: int,
    *,
    benchmark_id: str,
    exposure_round: int,
    seed: int,
) -> tuple[str, int]:
    if token_count <= capacity:
        return "whole", 0
    maximum = token_count - capacity
    policy_index = exposure_round % 4
    if policy_index == 0:
        return "prefix", 0
    if policy_index == 1:
        return "middle", maximum // 2
    if policy_index == 2:
        return "suffix", maximum
    hashed = _stable_uint64(seed + exposure_round, benchmark_id) % (maximum + 1)
    return "hashed", int(hashed)


def encode_exposure(
    *,
    content: str,
    benchmark_id: str,
    exposure_round: int,
    tokenizer: ByteCodeTokenizer,
    sequence_config: ShadowSequenceConfig,
    seed: int,
) -> dict[str, Any]:
    payload = tokenizer.encode(content, add_special_tokens=False)
    capacity = sequence_config.max_sequence_tokens - 2
    window_name, start = _window_start(
        len(payload),
        capacity,
        benchmark_id=benchmark_id,
        exposure_round=exposure_round,
        seed=seed,
    )
    selected = payload[start : start + capacity]
    input_ids = [tokenizer.bos_token_id, *selected, tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    padding = sequence_config.max_sequence_tokens - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding
    attention_mask += [0] * padding
    labels = [token if mask else -100 for token, mask in zip(input_ids, attention_mask)]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "source_token_count": len(payload),
        "window_start": start,
        "window_name": window_name,
        "selected_payload_tokens": len(selected),
    }


def build_shadow_training_sequences(
    train_corpus: pd.DataFrame,
    exposure_schedule: pd.DataFrame,
    *,
    tokenizer: ByteCodeTokenizer | None = None,
    sequence_config: ShadowSequenceConfig | None = None,
    seed: int = 2027,
) -> pd.DataFrame:
    """Materialise the architecture-independent token sequence schedule."""

    required_train = {"benchmark_id", "content"}
    required_schedule = {"benchmark_id", "exposure_round", "sequence_index"}
    missing_train = sorted(required_train.difference(train_corpus.columns))
    missing_schedule = sorted(required_schedule.difference(exposure_schedule.columns))
    if missing_train:
        raise ValueError(f"train_corpus is missing columns: {missing_train}")
    if missing_schedule:
        raise ValueError(f"exposure_schedule is missing columns: {missing_schedule}")
    if train_corpus.benchmark_id.duplicated().any():
        raise ValueError("train_corpus benchmark_id must be unique")
    if exposure_schedule.duplicated(["exposure_round", "sequence_index"]).any():
        raise ValueError("exposure schedule positions must be unique")
    train_ids = set(train_corpus.benchmark_id.astype(str))
    schedule_ids = set(exposure_schedule.benchmark_id.astype(str))
    if train_ids != schedule_ids:
        missing = len(train_ids - schedule_ids)
        unexpected = len(schedule_ids - train_ids)
        raise ValueError(
            "exposure schedule benchmark IDs must exactly match train_corpus: "
            f"missing={missing}, unexpected={unexpected}"
        )

    runtime_tokenizer = tokenizer or ByteCodeTokenizer()
    runtime_sequence = sequence_config or ShadowSequenceConfig()
    lookup = train_corpus.set_index("benchmark_id").content.astype(str).to_dict()
    schedule = exposure_schedule.copy()
    schedule["benchmark_id"] = schedule.benchmark_id.astype(str)
    schedule = schedule.sort_values(
        ["exposure_round", "sequence_index", "benchmark_id"], kind="mergesort"
    ).reset_index(drop=True)
    rows = []
    for row in schedule.itertuples(index=False):
        encoded = encode_exposure(
            content=lookup[row.benchmark_id],
            benchmark_id=row.benchmark_id,
            exposure_round=int(row.exposure_round),
            tokenizer=runtime_tokenizer,
            sequence_config=runtime_sequence,
            seed=seed,
        )
        rows.append(
            {
                "benchmark_id": row.benchmark_id,
                "exposure_round": int(row.exposure_round),
                "sequence_index": int(row.sequence_index),
                **encoded,
            }
        )
    result = pd.DataFrame(rows)
    expected_width = runtime_sequence.max_sequence_tokens
    if not all(len(values) == expected_width for values in result.input_ids):
        raise RuntimeError("shadow input sequence width mismatch")
    if not all(len(values) == expected_width for values in result.labels):
        raise RuntimeError("shadow label sequence width mismatch")
    if not all(len(values) == expected_width for values in result.attention_mask):
        raise RuntimeError("shadow attention sequence width mismatch")
    return result


def training_sequence_sha256(frame: pd.DataFrame) -> str:
    required = [
        "benchmark_id",
        "exposure_round",
        "sequence_index",
        "input_ids",
        "attention_mask",
        "labels",
        "window_start",
        "window_name",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"training sequence frame is missing columns: {missing}")
    digest = hashlib.sha256()
    for row in frame[required].itertuples(index=False, name=None):
        value = {column: item for column, item in zip(required, row)}
        digest.update(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def build_shadow_protocol_manifest(
    *,
    left: ShadowArchitectureSpec,
    right: ShadowArchitectureSpec,
    sequence_config: ShadowSequenceConfig,
    training_spec: ShadowTrainingSpec,
    training_sequences: pd.DataFrame,
    tokenizer: ByteCodeTokenizer | None = None,
    max_parameter_ratio: float = 1.15,
) -> dict[str, Any]:
    runtime_tokenizer = tokenizer or ByteCodeTokenizer()
    pair = validate_shadow_pair(
        left,
        right,
        max_parameter_ratio=max_parameter_ratio,
    )
    return {
        "status": "frozen",
        "protocol_version": SHADOW_PROTOCOL_VERSION,
        "architecture_pair": pair,
        "tokenizer": runtime_tokenizer.specification,
        "tokenizer_sha256": runtime_tokenizer.fingerprint,
        "sequence_config": asdict(sequence_config),
        "training_spec": asdict(training_spec),
        "training_sequence_rows": len(training_sequences),
        "training_sequence_sha256": training_sequence_sha256(training_sequences),
        "random_initialisation": True,
        "pretrained_weights_used": False,
        "membership_labels_exact_for_emitted_training_corpus": True,
        "target_evaluation_labels_used_for_training": False,
        "compute_backend_selected": False,
        "model_compute_started": False,
    }


__all__ = [
    "SHADOW_PROTOCOL_VERSION",
    "ByteCodeTokenizer",
    "ShadowArchitectureSpec",
    "ShadowSequenceConfig",
    "ShadowTrainingSpec",
    "build_shadow_protocol_manifest",
    "build_shadow_training_sequences",
    "default_shadow_pair",
    "encode_exposure",
    "training_sequence_sha256",
    "validate_shadow_pair",
]
