"""Deterministic controlled shadow-model training runtime.

The runtime consumes only the frozen byte-token array and protocol manifest. It
never reads the controlled evaluation inputs or labels. CPU/CUDA and PyTorch/XLA
workers use the same global sequence order, contiguous rank shards, per-sequence
causal loss normalisation, optimiser schedule and checkpoint contract.

The public launcher does not retry failures. A Kaggle TPU request remains a
separate protected operation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, Mapping

import numpy as np

from .shadow_protocol import (
    ByteCodeTokenizer,
    ShadowArchitectureSpec,
    ShadowTrainingSpec,
)


SHADOW_TRAINING_RUNTIME_VERSION = "shadow-training-runtime-v1"


class ShadowTrainingError(RuntimeError):
    """Raised when a frozen protocol or runtime invariant is violated."""


@dataclass(frozen=True)
class ShadowRuntimeConfig:
    backend: str
    architecture_slot: str
    max_steps: int | None = None
    checkpoint_reload_atol: float = 1e-5
    parameter_sync_atol: float = 1e-5
    save_optimizer_state: bool = True
    cpu_threads: int | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "cuda", "xla"}:
            raise ValueError("backend must be 'cpu', 'cuda', or 'xla'")
        if self.architecture_slot not in {"left", "right"}:
            raise ValueError("architecture_slot must be 'left' or 'right'")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when set")
        if self.checkpoint_reload_atol < 0:
            raise ValueError("checkpoint_reload_atol must be non-negative")
        if self.parameter_sync_atol < 0:
            raise ValueError("parameter_sync_atol must be non-negative")
        if self.cpu_threads is not None and self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive when set")


@dataclass(frozen=True)
class FrozenShadowProtocol:
    directory: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    input_ids: np.ndarray
    architecture: ShadowArchitectureSpec
    training_spec: ShadowTrainingSpec
    global_batch_size: int
    full_steps: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            raise ShadowTrainingError(f"state value is not a tensor: {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_json(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def _to_cpu(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def load_frozen_shadow_protocol(
    directory: str | Path,
    architecture_slot: str,
) -> FrozenShadowProtocol:
    root = Path(directory).resolve()
    if architecture_slot not in {"left", "right"}:
        raise ValueError("architecture_slot must be 'left' or 'right'")
    required = {
        "training_input_ids.npy",
        "training_sequence_metadata.jsonl",
        "byte_tokenizer.json",
        "shadow_protocol_manifest.json",
    }
    if not root.is_dir():
        raise ShadowTrainingError(f"shadow protocol directory does not exist: {root}")
    actual = {path.name for path in root.iterdir() if path.is_file()}
    missing = sorted(required.difference(actual))
    if missing:
        raise ShadowTrainingError(f"shadow protocol directory is missing files: {missing}")

    manifest_path = root / "shadow_protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise ShadowTrainingError("shadow protocol manifest is not frozen")
    for key in (
        "optimiser_steps_performed",
        "model_compute_started",
        "accelerator_selected",
        "kaggle_operation_performed",
    ):
        expected = 0 if key == "optimiser_steps_performed" else False
        if manifest.get(key) != expected:
            raise ShadowTrainingError(f"shadow protocol no-compute guard failed: {key}")
    protocol = manifest.get("protocol") or {}
    if protocol.get("pretrained_weights_used") is not False:
        raise ShadowTrainingError("pretrained weight guard failed")
    if protocol.get("target_evaluation_labels_used_for_training") is not False:
        raise ShadowTrainingError("target evaluation label guard failed")
    if protocol.get("membership_labels_exact_for_emitted_training_corpus") is not True:
        raise ShadowTrainingError("exact membership assertion is missing")

    hashes = manifest.get("output_sha256") or {}
    for name in ("training_input_ids.npy", "training_sequence_metadata.jsonl", "byte_tokenizer.json"):
        if hashes.get(name) != _sha256_file(root / name):
            raise ShadowTrainingError(f"shadow protocol SHA-256 mismatch: {name}")
    tokenizer = ByteCodeTokenizer.from_pretrained(root)
    tokenizer_manifest = protocol.get("tokenizer")
    if tokenizer_manifest != tokenizer.specification:
        raise ShadowTrainingError("byte tokenizer manifest mismatch")
    if protocol.get("tokenizer_sha256") != tokenizer.fingerprint:
        raise ShadowTrainingError("byte tokenizer fingerprint mismatch")

    input_ids = np.load(root / "training_input_ids.npy", mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(int(value) for value in manifest.get("training_input_shape", []))
    if input_ids.shape != expected_shape:
        raise ShadowTrainingError(
            f"training input shape mismatch: {input_ids.shape} != {expected_shape}"
        )
    if input_ids.dtype != np.uint16:
        raise ShadowTrainingError(f"training input dtype mismatch: {input_ids.dtype}")
    if input_ids.ndim != 2 or input_ids.shape[0] <= 0 or input_ids.shape[1] < 4:
        raise ShadowTrainingError("training input array has invalid dimensions")
    if int(input_ids.max()) >= tokenizer.vocab_size:
        raise ShadowTrainingError("training input contains token outside frozen vocabulary")
    if not np.all(input_ids[:, 0] == tokenizer.bos_token_id):
        raise ShadowTrainingError("training input does not begin with BOS")

    metadata_rows = sum(
        1 for line in (root / "training_sequence_metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if metadata_rows != input_ids.shape[0]:
        raise ShadowTrainingError("training metadata row count mismatch")

    pair = protocol.get("architecture_pair") or {}
    architecture_value = ((pair.get(architecture_slot) or {}).get("spec"))
    if not isinstance(architecture_value, dict):
        raise ShadowTrainingError(f"architecture spec is missing: {architecture_slot}")
    architecture = ShadowArchitectureSpec(**architecture_value)
    if architecture.vocab_size != tokenizer.vocab_size:
        raise ShadowTrainingError("architecture/tokenizer vocabulary mismatch")
    if architecture.max_position_embeddings != input_ids.shape[1]:
        raise ShadowTrainingError("architecture/sequence position limit mismatch")

    training_value = protocol.get("training_spec")
    if not isinstance(training_value, dict):
        raise ShadowTrainingError("training specification is missing")
    training_spec = ShadowTrainingSpec(**training_value)
    global_batch_size = int(training_spec.global_batch_size)
    if input_ids.shape[0] % global_batch_size:
        raise ShadowTrainingError(
            "training sequence rows must be divisible by the frozen global batch size"
        )
    full_steps = input_ids.shape[0] // global_batch_size
    if full_steps <= 0:
        raise ShadowTrainingError("frozen protocol has no optimiser step")
    if protocol.get("training_sequence_rows") != input_ids.shape[0]:
        raise ShadowTrainingError("protocol sequence row count mismatch")
    return FrozenShadowProtocol(
        directory=root,
        manifest=manifest,
        manifest_sha256=_sha256_file(manifest_path),
        input_ids=input_ids,
        architecture=architecture,
        training_spec=training_spec,
        global_batch_size=global_batch_size,
        full_steps=full_steps,
    )


def _seed_everything(seed: int, *, deterministic: bool = True) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _per_sequence_causal_loss(logits: Any, labels: Any):
    import torch
    import torch.nn.functional as functional

    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    token_count = valid.sum(dim=1)
    if bool(token_count.eq(0).any().detach().cpu()):
        raise ShadowTrainingError("a training sequence has no target token")
    token_loss = functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(shifted_labels)
    sequence_loss = (token_loss * valid).sum(dim=1) / token_count
    return sequence_loss.mean(), sequence_loss.detach()


def _learning_rate_factor(step: int, *, total_steps: int, warmup_fraction: float) -> float:
    warmup_steps = int(math.floor(total_steps * warmup_fraction))
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / warmup_steps
    remaining = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / remaining, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class _TorchBackend:
    def __init__(self, name: str):
        import torch

        if name == "cpu":
            self.device = torch.device("cpu")
        elif name == "cuda":
            if not torch.cuda.is_available():
                raise ShadowTrainingError("CUDA backend requested but unavailable")
            self.device = torch.device("cuda", 0)
        else:
            raise ValueError(name)
        self.name = name
        self.rank = 0
        self.world_size = 1
        self.master = True

    def optimizer_step(self, optimizer: Any) -> None:
        optimizer.step()

    def mark_step(self) -> None:
        return None

    def rendezvous(self, tag: str) -> None:
        del tag

    def reduce_mean(self, value: float, tag: str) -> float:
        del tag
        return float(value)

    def reduce_range(self, value: float, tag: str) -> tuple[float, float]:
        del tag
        return float(value), float(value)


class _XlaBackend:
    def __init__(self):
        import torch_xla.core.xla_model as xm

        self.xm = xm
        self.name = "xla"
        self.device = xm.xla_device()
        self.rank = int(xm.get_ordinal())
        self.world_size = int(xm.xrt_world_size())
        self.master = bool(xm.is_master_ordinal())

    def optimizer_step(self, optimizer: Any) -> None:
        self.xm.optimizer_step(optimizer, barrier=False)

    def mark_step(self) -> None:
        self.xm.mark_step()

    def rendezvous(self, tag: str) -> None:
        self.xm.rendezvous(tag)

    def reduce_mean(self, value: float, tag: str) -> float:
        return float(
            self.xm.mesh_reduce(
                tag,
                float(value),
                lambda values: sum(float(item) for item in values) / len(values),
            )
        )

    def reduce_range(self, value: float, tag: str) -> tuple[float, float]:
        result = self.xm.mesh_reduce(
            tag,
            float(value),
            lambda values: (
                min(float(item) for item in values),
                max(float(item) for item in values),
            ),
        )
        return float(result[0]), float(result[1])


def _parameter_scalar_checksum(model: Any) -> float:
    import torch

    with torch.no_grad():
        checksum = torch.zeros((), dtype=torch.float64, device=next(model.parameters()).device)
        for index, parameter in enumerate(model.parameters(), start=1):
            value = parameter.detach().double()
            checksum = checksum + value.sum() * index + value.square().sum() * (index + 0.5)
    return float(checksum.detach().cpu())


def _fixed_local_batch(
    protocol: FrozenShadowProtocol,
    backend: _TorchBackend | _XlaBackend,
):
    local_batch_size = protocol.global_batch_size // backend.world_size
    start = backend.rank * local_batch_size
    stop = start + local_batch_size
    values = np.asarray(protocol.input_ids[start:stop], dtype=np.int64)
    return values


def _local_batch(
    protocol: FrozenShadowProtocol,
    backend: _TorchBackend | _XlaBackend,
    step: int,
):
    if protocol.global_batch_size % backend.world_size:
        raise ShadowTrainingError(
            "global batch size must be divisible by backend world size"
        )
    local_batch_size = protocol.global_batch_size // backend.world_size
    global_start = step * protocol.global_batch_size
    local_start = global_start + backend.rank * local_batch_size
    local_stop = local_start + local_batch_size
    return np.asarray(protocol.input_ids[local_start:local_stop], dtype=np.int64)


def _tensor_batch(values: np.ndarray, device: Any):
    import torch

    input_ids = torch.as_tensor(values, dtype=torch.long, device=device)
    attention_mask = input_ids.ne(ByteCodeTokenizer.pad_token_id).long()
    labels = input_ids.clone()
    labels.masked_fill_(attention_mask.eq(0), -100)
    return input_ids, attention_mask, labels


def _forward_logits(model: Any, values: np.ndarray, device: Any):
    import torch

    input_ids, attention_mask, _ = _tensor_batch(values, device)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = output.logits if hasattr(output, "logits") else output[0]
        result = logits.detach().float().cpu()
    del input_ids, attention_mask, output, logits
    return result


def _write_checkpoint(
    *,
    path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    protocol: FrozenShadowProtocol,
    runtime: ShadowRuntimeConfig,
    completed_steps: int,
) -> tuple[str, str]:
    import torch

    state_dict = _to_cpu(model.state_dict())
    state_sha256 = _state_dict_sha256(state_dict)
    payload = {
        "runtime_version": SHADOW_TRAINING_RUNTIME_VERSION,
        "protocol_manifest_sha256": protocol.manifest_sha256,
        "architecture_slot": runtime.architecture_slot,
        "architecture": asdict(protocol.architecture),
        "training_spec": asdict(protocol.training_spec),
        "completed_steps": completed_steps,
        "model_state_dict": state_dict,
        "model_state_sha256": state_sha256,
    }
    if runtime.save_optimizer_state:
        payload["optimizer_state_dict"] = _to_cpu(optimizer.state_dict())
        payload["scheduler_state_dict"] = scheduler.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return state_sha256, _sha256_file(path)


def _reload_and_compare(
    *,
    checkpoint_path: Path,
    protocol: FrozenShadowProtocol,
    backend: _TorchBackend | _XlaBackend,
    reference_logits: Any,
    fixed_values: np.ndarray,
) -> tuple[float, str]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol_manifest_sha256") != protocol.manifest_sha256:
        raise ShadowTrainingError("checkpoint protocol SHA-256 mismatch")
    if checkpoint.get("architecture_slot") not in {"left", "right"}:
        raise ShadowTrainingError("checkpoint architecture slot missing")
    model = protocol.architecture.build_model(seed=protocol.training_spec.seed)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ShadowTrainingError("checkpoint model state is missing")
    state_sha256 = _state_dict_sha256(state)
    if state_sha256 != checkpoint.get("model_state_sha256"):
        raise ShadowTrainingError("checkpoint logical model-state hash mismatch")
    model.load_state_dict(state, strict=True)
    model.to(backend.device).eval()
    reloaded = _forward_logits(model, fixed_values, backend.device)
    difference = float((reference_logits - reloaded).abs().max())
    del model, reloaded, checkpoint, state
    return difference, state_sha256


def _train_worker(
    protocol_directory: str,
    output_directory: str,
    runtime_value: Mapping[str, Any],
    *,
    xla: bool,
) -> dict[str, Any] | None:
    import torch

    runtime = ShadowRuntimeConfig(**dict(runtime_value))
    backend: _TorchBackend | _XlaBackend = _XlaBackend() if xla else _TorchBackend(runtime.backend)
    if runtime.cpu_threads is not None and backend.name == "cpu":
        torch.set_num_threads(runtime.cpu_threads)
    protocol = load_frozen_shadow_protocol(
        protocol_directory,
        runtime.architecture_slot,
    )
    if protocol.global_batch_size % backend.world_size:
        raise ShadowTrainingError(
            f"global batch size {protocol.global_batch_size} is not divisible by world size {backend.world_size}"
        )
    steps = protocol.full_steps if runtime.max_steps is None else runtime.max_steps
    if steps > protocol.full_steps:
        raise ShadowTrainingError(
            f"requested {steps} steps but frozen protocol contains {protocol.full_steps}"
        )

    output = Path(output_directory).resolve()
    if backend.master:
        if output.exists() and any(output.iterdir()):
            raise ShadowTrainingError("output directory already exists and is not empty")
        output.mkdir(parents=True, exist_ok=True)
    backend.rendezvous("shadow-output-ready")

    _seed_everything(protocol.training_spec.seed)
    model = protocol.architecture.build_model(seed=protocol.training_spec.seed)
    model.to(backend.device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.training_spec.learning_rate,
        weight_decay=protocol.training_spec.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _learning_rate_factor(
            step,
            total_steps=protocol.full_steps,
            warmup_fraction=protocol.training_spec.warmup_fraction,
        ),
    )

    losses: list[float] = []
    learning_rates: list[float] = []
    for step in range(steps):
        values = _local_batch(protocol, backend, step)
        input_ids, attention_mask, labels = _tensor_batch(values, backend.device)
        optimizer.zero_grad(set_to_none=True)
        output_value = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = output_value.logits if hasattr(output_value, "logits") else output_value[0]
        loss, _ = _per_sequence_causal_loss(logits, labels)
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise ShadowTrainingError(f"non-finite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), protocol.training_spec.gradient_clip_norm
        )
        if not bool(torch.isfinite(gradient_norm).detach().cpu()):
            raise ShadowTrainingError(f"non-finite gradient norm at step {step}")
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        backend.optimizer_step(optimizer)
        scheduler.step()
        backend.mark_step()
        loss_value = backend.reduce_mean(
            float(loss.detach().cpu()), f"shadow-loss-{step}"
        )
        losses.append(loss_value)
        del values, input_ids, attention_mask, labels, output_value, logits, loss, gradient_norm

    model.eval()
    checksum = _parameter_scalar_checksum(model)
    checksum_min, checksum_max = backend.reduce_range(
        checksum, "shadow-parameter-checksum"
    )
    checksum_spread = checksum_max - checksum_min
    if checksum_spread > runtime.parameter_sync_atol:
        raise ShadowTrainingError(
            f"cross-replica parameter checksum spread {checksum_spread} exceeds {runtime.parameter_sync_atol}"
        )

    fixed_values = _fixed_local_batch(protocol, backend)
    reference_logits = _forward_logits(model, fixed_values, backend.device)
    checkpoint_path = output / f"{protocol.architecture.name}.pt"
    state_sha256: str | None = None
    checkpoint_sha256: str | None = None
    if backend.master:
        state_sha256, checkpoint_sha256 = _write_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            protocol=protocol,
            runtime=runtime,
            completed_steps=steps,
        )
    backend.rendezvous("shadow-checkpoint-written")
    reload_difference, reloaded_state_sha256 = _reload_and_compare(
        checkpoint_path=checkpoint_path,
        protocol=protocol,
        backend=backend,
        reference_logits=reference_logits,
        fixed_values=fixed_values,
    )
    reload_max = backend.reduce_range(
        reload_difference, "shadow-reload-difference"
    )[1]
    if reload_max > runtime.checkpoint_reload_atol:
        raise ShadowTrainingError(
            f"checkpoint reload logits difference {reload_max} exceeds {runtime.checkpoint_reload_atol}"
        )

    result: dict[str, Any] | None = None
    if backend.master:
        if state_sha256 != reloaded_state_sha256:
            raise ShadowTrainingError("saved and reloaded logical state hashes differ")
        result = {
            "status": "complete",
            "runtime_version": SHADOW_TRAINING_RUNTIME_VERSION,
            "protocol_manifest_sha256": protocol.manifest_sha256,
            "architecture_slot": runtime.architecture_slot,
            "architecture": asdict(protocol.architecture),
            "training_spec": asdict(protocol.training_spec),
            "runtime_config": asdict(runtime),
            "backend": backend.name,
            "world_size": backend.world_size,
            "global_batch_size": protocol.global_batch_size,
            "local_batch_size": protocol.global_batch_size // backend.world_size,
            "full_protocol_steps": protocol.full_steps,
            "completed_steps": steps,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "minimum_loss": min(losses),
            "maximum_loss": max(losses),
            "initial_learning_rate": learning_rates[0],
            "final_learning_rate": learning_rates[-1],
            "parameter_checksum_min": checksum_min,
            "parameter_checksum_max": checksum_max,
            "parameter_checksum_spread": checksum_spread,
            "model_state_sha256": state_sha256,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_reload_max_absolute_logit_difference": reload_max,
            "checkpoint_reload_passed": True,
            "random_initialisation": True,
            "pretrained_weights_used": False,
            "evaluation_inputs_read": False,
            "evaluation_labels_read": False,
            "sample_ids_used_as_features": False,
            "automatic_compute_retries": 0,
        }
        manifest_path = output / "training_manifest.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
    backend.rendezvous("shadow-training-manifest-written")
    del model, optimizer, scheduler, reference_logits
    return result


def train_shadow_model(
    protocol_directory: str | Path,
    output_directory: str | Path,
    runtime_config: ShadowRuntimeConfig,
) -> dict[str, Any] | None:
    """Train one frozen shadow architecture exactly once.

    For ``backend='xla'`` this function launches one worker per available XLA
    device. The caller must invoke it from a normal Python entry point, not from
    each already-spawned worker.
    """

    protocol_value = str(Path(protocol_directory).resolve())
    output_value = str(Path(output_directory).resolve())
    runtime_value = asdict(runtime_config)
    if runtime_config.backend != "xla":
        return _train_worker(
            protocol_value,
            output_value,
            runtime_value,
            xla=False,
        )

    def worker(index: int, protocol_path: str, output_path: str, config_value: Mapping[str, Any]):
        del index
        _train_worker(
            protocol_path,
            output_path,
            config_value,
            xla=True,
        )

    try:
        import torch_xla
    except ImportError as error:
        raise ShadowTrainingError("XLA backend requested but torch_xla is unavailable") from error
    launch = getattr(torch_xla, "launch", None)
    if launch is not None:
        launch(worker, args=(protocol_value, output_value, runtime_value))
    else:
        import torch_xla.distributed.xla_multiprocessing as xmp

        xmp.spawn(
            worker,
            args=(protocol_value, output_value, runtime_value),
            nprocs=None,
            start_method="fork",
        )
    manifest_path = Path(output_value) / "training_manifest.json"
    if not manifest_path.is_file():
        raise ShadowTrainingError("XLA workers did not produce a training manifest")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


__all__ = [
    "SHADOW_TRAINING_RUNTIME_VERSION",
    "FrozenShadowProtocol",
    "ShadowRuntimeConfig",
    "ShadowTrainingError",
    "load_frozen_shadow_protocol",
    "train_shadow_model",
]
