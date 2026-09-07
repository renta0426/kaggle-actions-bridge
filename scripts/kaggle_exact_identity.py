"""Exact, fail-closed Kaggle identity checks; no search, writes or retries.

Imports of Kaggle itself are lazy so pure contracts can be tested without a
credential. Search results are discovery aids, not identity/existence proofs.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

REF_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
TERMINAL = frozenset({"COMPLETE", "ERROR", "CANCELLED", "CANCELED"})
ACTIVE = frozenset({"QUEUED", "RUNNING", "PENDING"})


class IdentityError(RuntimeError):
    """Messages are fixed categories, never remote exception text."""


def status_name(value: Any) -> str:
    name = getattr(value, "name", None)
    text = str(name if name is not None else value).strip().upper()
    # Support the SDK enum rendering without substring acceptance.
    text = text.rsplit(".", 1)[-1]
    if text not in TERMINAL | ACTIVE:
        raise IdentityError("unknown_kernel_status")
    return text


def validate_metadata(metadata: Any, kernel: str, version: int, *, cpu: bool = False) -> None:
    if not REF_RE.fullmatch(kernel) or type(version) is not int or version < 1:
        raise IdentityError("invalid_expected_identity")
    if getattr(metadata, "ref", None) != kernel:
        raise IdentityError("kernel_ref_mismatch")
    if getattr(metadata, "is_private", None) is not True:
        raise IdentityError("private_flag_not_proven")
    observed = getattr(metadata, "current_version_number", None)
    if isinstance(observed, bool) or observed is None or int(observed) != version:
        raise IdentityError("current_version_mismatch")
    if cpu:
        for field in ("enable_gpu", "enable_tpu", "enable_internet"):
            if getattr(metadata, field, None) is not False:
                raise IdentityError("cpu_offline_contract_not_proven")


def exact_metadata(api: Any, kernel: str) -> Any:
    if not REF_RE.fullmatch(kernel):
        raise IdentityError("invalid_kernel_ref")
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    owner, slug = kernel.split("/", 1)
    request = ApiGetKernelRequest()
    request.user_name = owner
    request.kernel_slug = slug
    with api.build_kaggle_client() as client:
        return client.kernels.kernels_api_client.get_kernel(request).metadata


def verify_current(api: Any, kernel: str, version: int, *, allow_failed: bool = False, cpu: bool = False) -> str:
    metadata = exact_metadata(api, kernel)
    validate_metadata(metadata, kernel, version, cpu=cpu)
    state = status_name(getattr(api.kernels_status(kernel), "status", None))
    allowed = TERMINAL if allow_failed else {"COMPLETE"}
    if state not in allowed:
        raise IdentityError("kernel_not_in_allowed_terminal_state")
    return state


def safe_exception(exc: BaseException) -> dict[str, Any]:
    """Do not leak refs, response bodies, tokens, paths or signed URLs."""
    text = f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
    status = getattr(exc, "status", getattr(exc, "status_code", None))
    status = status if type(status) is int and 100 <= status <= 599 else None
    return {"exception_type": type(exc).__name__, "error_sha256": hashlib.sha256(text).hexdigest(), "http_status": status}
