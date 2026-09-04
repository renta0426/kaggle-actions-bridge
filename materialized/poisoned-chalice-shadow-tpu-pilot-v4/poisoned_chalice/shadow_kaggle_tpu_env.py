"""Kaggle TPU environment compatibility for PyTorch/XLA PJRT.

Kaggle TPU notebook environments can expose topology variables intended for the
platform runtime.  PyTorch/XLA PJRT may interpret those variables as a complete
multi-worker topology and fail before device discovery when only one address is
present.  This compatibility layer removes only the two Kaggle/Cloud-TPU
variables observed in that failure mode, before importing torch_xla.

The values are deliberately never returned or logged because they may contain
worker-address information.  Only the variable names and count are reported.
"""

from __future__ import annotations

import os
from typing import MutableMapping


SHADOW_KAGGLE_TPU_ENV_COMPAT_VERSION = "shadow-kaggle-tpu-pjrt-env-v1"
_TOPOLOGY_VARIABLES = ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID")


def sanitize_kaggle_tpu_pjrt_environment(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, object]:
    """Remove stale Kaggle TPU topology hints before torch_xla/PJRT import.

    The operation is intentionally narrow and idempotent.  No other environment
    variables are changed.  Returned metadata contains no variable values.
    """

    env = os.environ if environ is None else environ
    removed: list[str] = []
    for name in _TOPOLOGY_VARIABLES:
        if name in env:
            env.pop(name)
            removed.append(name)
    return {
        "compatibility_layer": SHADOW_KAGGLE_TPU_ENV_COMPAT_VERSION,
        "removed_variables": removed,
        "removed_count": len(removed),
    }


__all__ = [
    "SHADOW_KAGGLE_TPU_ENV_COMPAT_VERSION",
    "sanitize_kaggle_tpu_pjrt_environment",
]
