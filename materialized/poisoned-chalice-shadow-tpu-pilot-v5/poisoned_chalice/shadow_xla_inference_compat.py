"""Narrow PyTorch/XLA compatibility for fixed-logit shadow forwards.

PyTorch/XLA 2.8 can raise ``Cannot set version_counter for inference tensor``
when Hugging Face model code mutates tensor metadata under
``torch.inference_mode()``. The controlled-shadow runtime uses inference mode
only for fixed reference/checkpoint-parity forwards; optimiser steps themselves
run with normal autograd.

This module replaces only the XLA branch of ``shadow_training._forward_logits``
with an equivalent ``torch.no_grad()`` forward. CPU/CUDA calls continue to use
the original runtime implementation unchanged.
"""

from __future__ import annotations

from typing import Any


SHADOW_XLA_INFERENCE_COMPAT_VERSION = "shadow-xla-no-grad-forward-v1"


def install_shadow_xla_inference_compat() -> dict[str, Any]:
    """Use ``no_grad`` instead of ``inference_mode`` for XLA fixed forwards."""

    import torch

    from . import shadow_training as runtime

    current = runtime._forward_logits
    if getattr(current, "_shadow_xla_inference_compat", None) == SHADOW_XLA_INFERENCE_COMPAT_VERSION:
        return {
            "compatibility_layer": SHADOW_XLA_INFERENCE_COMPAT_VERSION,
            "installed": False,
            "xla_forward_context": "torch.no_grad",
            "non_xla_forward_context": "unchanged",
        }

    original = current

    def xla_safe_forward_logits(model: Any, values: Any, device: Any):
        if getattr(device, "type", None) != "xla":
            return original(model, values, device)

        input_ids, attention_mask, _ = runtime._tensor_batch(values, device)
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = output.logits if hasattr(output, "logits") else output[0]
            result = logits.detach().float().cpu()
        del input_ids, attention_mask, output, logits
        return result

    xla_safe_forward_logits._shadow_xla_inference_compat = SHADOW_XLA_INFERENCE_COMPAT_VERSION
    xla_safe_forward_logits._shadow_xla_inference_original = original
    runtime._forward_logits = xla_safe_forward_logits
    return {
        "compatibility_layer": SHADOW_XLA_INFERENCE_COMPAT_VERSION,
        "installed": True,
        "xla_forward_context": "torch.no_grad",
        "non_xla_forward_context": "unchanged",
    }


__all__ = [
    "SHADOW_XLA_INFERENCE_COMPAT_VERSION",
    "install_shadow_xla_inference_compat",
]
