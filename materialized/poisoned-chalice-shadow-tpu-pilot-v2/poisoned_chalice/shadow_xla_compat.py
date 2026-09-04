"""Narrow PyTorch/XLA compatibility shim for the controlled-shadow TPU pilot.

PyTorch/XLA removed ``xla_model.get_ordinal`` and ``xla_model.xrt_world_size``
in release 2.7 in favor of ``torch_xla.runtime.global_ordinal`` and
``torch_xla.runtime.world_size``.  The frozen shadow runtime still calls the old
names.  This module restores only those two aliases before the XLA backend is
constructed, without changing CPU/CUDA behavior or the scientific protocol.
"""

from __future__ import annotations

from typing import Any


def install_shadow_xla_compat() -> dict[str, Any]:
    """Install only the two removed XLA model aliases required by runtime v1.

    Returns a small non-sensitive record suitable for the TPU pilot manifest.
    Existing attributes are never replaced, so older PyTorch/XLA releases keep
    their native implementation.
    """

    import torch_xla
    import torch_xla.core.xla_model as xm
    from torch_xla import runtime as xr

    patched: dict[str, str] = {}
    if not hasattr(xm, "get_ordinal"):
        xm.get_ordinal = xr.global_ordinal
        patched["get_ordinal"] = "torch_xla.runtime.global_ordinal"
    if not hasattr(xm, "xrt_world_size"):
        xm.xrt_world_size = xr.world_size
        patched["xrt_world_size"] = "torch_xla.runtime.world_size"

    return {
        "compatibility_layer": "shadow-xla-compat-v1",
        "torch_xla_version": str(getattr(torch_xla, "__version__", "unknown")),
        "patched_aliases": patched,
    }
