"""Spawn-safe PyTorch/XLA entrypoint for controlled shadow training.

PyTorch/XLA 2.8 uses Python multiprocessing with a spawn-style start method for
``torch_xla.launch``.  Spawn requires the worker callable to be importable from
a module top level; a nested function such as ``train_shadow_model.<locals>.worker``
is not pickleable.  This module changes only the XLA process-entry boundary and
reuses the frozen runtime worker unchanged.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

from .shadow_training import (
    ShadowRuntimeConfig,
    ShadowTrainingError,
    _train_worker,
    train_shadow_model,
)


SHADOW_XLA_SPAWN_COMPAT_VERSION = "shadow-xla-spawn-safe-v1"


def xla_shadow_worker(
    index: int,
    protocol_path: str,
    output_path: str,
    config_value: Mapping[str, Any],
) -> None:
    """Top-level pickleable worker passed to ``torch_xla.launch``."""

    del index
    _train_worker(
        protocol_path,
        output_path,
        config_value,
        xla=True,
    )


def train_shadow_model_spawn_safe(
    protocol_directory: str | Path,
    output_directory: str | Path,
    runtime_config: ShadowRuntimeConfig,
) -> dict[str, Any] | None:
    """Train one frozen shadow architecture with a spawn-safe XLA boundary.

    CPU/CUDA behavior is delegated to the unchanged runtime.  Only the XLA
    launcher path differs from ``shadow_training.train_shadow_model``.
    """

    if runtime_config.backend != "xla":
        return train_shadow_model(
            protocol_directory,
            output_directory,
            runtime_config,
        )

    protocol_value = str(Path(protocol_directory).resolve())
    output_value = str(Path(output_directory).resolve())
    runtime_value = asdict(runtime_config)

    try:
        import torch_xla
    except ImportError as error:
        raise ShadowTrainingError(
            "XLA backend requested but torch_xla is unavailable"
        ) from error

    launch = getattr(torch_xla, "launch", None)
    if launch is not None:
        launch(
            xla_shadow_worker,
            args=(protocol_value, output_value, runtime_value),
        )
    else:
        import torch_xla.distributed.xla_multiprocessing as xmp

        xmp.spawn(
            xla_shadow_worker,
            args=(protocol_value, output_value, runtime_value),
            nprocs=None,
            start_method="fork",
        )

    manifest_path = Path(output_value) / "training_manifest.json"
    if not manifest_path.is_file():
        raise ShadowTrainingError("XLA workers did not produce a training manifest")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


__all__ = [
    "SHADOW_XLA_SPAWN_COMPAT_VERSION",
    "train_shadow_model_spawn_safe",
    "xla_shadow_worker",
]
