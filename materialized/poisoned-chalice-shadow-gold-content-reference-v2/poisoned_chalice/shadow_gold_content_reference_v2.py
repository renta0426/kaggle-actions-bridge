"""Narrow runtime compatibility repair for Gold content-reference calibration v1.

The first Kaggle execution of ``shadow-gold-content-reference-v1`` failed before
model training because the content-reference runtime called the unchanged
``run_dual_gpu_shadow`` helper with ``required_name_fragment`` instead of the
helper's public keyword ``required_gpu_name_fragment``.

This module changes only that keyword boundary.  It does not change the frozen
Gold corpus, split, positive-control probe, exposures, optimizer, target scoring,
content reference, candidate matrix, selection protocol, recovery gate, or
clean-room rules.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping

from . import shadow_gold_content_reference as content_reference_v1


SHADOW_GOLD_CONTENT_REFERENCE_REPAIR_VERSION = "content-reference-dual-gpu-kwarg-repair-v1"


class ShadowGoldContentReferenceRepairError(RuntimeError):
    """Raised when the narrow runtime adapter can no longer be applied safely."""


def make_dual_gpu_keyword_adapter(delegate: Callable[..., Any]) -> Callable[..., Any]:
    """Map the v1 typo to the unchanged dual-GPU helper keyword.

    The adapter fails closed if the delegate API drifts or if callers supply both
    spellings.  All other positional/keyword arguments are forwarded unchanged.
    """

    signature = inspect.signature(delegate)
    parameters = signature.parameters
    if "required_gpu_name_fragment" not in parameters:
        raise ShadowGoldContentReferenceRepairError(
            "dual-GPU helper no longer exposes required_gpu_name_fragment"
        )
    if "required_name_fragment" in parameters:
        raise ShadowGoldContentReferenceRepairError(
            "dual-GPU helper unexpectedly exposes the historical typo"
        )

    def adapted(*args: Any, **kwargs: Any) -> Any:
        if "required_name_fragment" in kwargs and "required_gpu_name_fragment" in kwargs:
            raise ShadowGoldContentReferenceRepairError(
                "both dual-GPU name-fragment spellings were supplied"
            )
        if "required_name_fragment" in kwargs:
            kwargs = dict(kwargs)
            kwargs["required_gpu_name_fragment"] = kwargs.pop("required_name_fragment")
        return delegate(*args, **kwargs)

    return adapted


def run_content_reference_benchmark(
    *,
    scratch_directory: str,
    positive_control_config: Mapping[str, Any],
    content_reference_config: Mapping[str, Any],
    timeout_seconds: int = 5400,
):
    """Run v1 science with only the dual-GPU keyword compatibility repair."""

    original = content_reference_v1.run_dual_gpu_shadow
    adapted = make_dual_gpu_keyword_adapter(original)
    content_reference_v1.run_dual_gpu_shadow = adapted
    try:
        return content_reference_v1.run_content_reference_benchmark(
            scratch_directory=scratch_directory,
            positive_control_config=positive_control_config,
            content_reference_config=content_reference_config,
            timeout_seconds=timeout_seconds,
        )
    finally:
        content_reference_v1.run_dual_gpu_shadow = original


__all__ = [
    "SHADOW_GOLD_CONTENT_REFERENCE_REPAIR_VERSION",
    "ShadowGoldContentReferenceRepairError",
    "make_dual_gpu_keyword_adapter",
    "run_content_reference_benchmark",
]
