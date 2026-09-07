"""Exact, fail-closed Kaggle identity checks; never use search as authority.

The primitive ``exact_metadata`` performs exactly one direct read.  The opt-in
``exact_metadata_eventually`` helper exists for one observed Kaggle integration
failure class: immediately after a successful private Notebook push, the exact
GetKernel endpoint can transiently return HTTP 403/404 even though the Notebook
exists and runs.  It retries only that same read-only exact endpoint, with a
small fixed bound.  HTTP 429 and every other failure are never retried here.

Imports of Kaggle itself are lazy so pure contracts can be tested without a
credential. Search/list results remain discovery aids, not identity/existence
proofs. No function in this module performs a write.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Callable

REF_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
TERMINAL = frozenset({"COMPLETE", "ERROR", "CANCELLED", "CANCELED"})
ACTIVE = frozenset({"QUEUED", "RUNNING", "PENDING"})
TRANSIENT_EXACT_METADATA_HTTP = frozenset({403, 404})
MAX_EXACT_METADATA_ATTEMPTS = 12
MAX_EXACT_METADATA_DELAY_SECONDS = 30.0


class IdentityError(RuntimeError):
    """Messages are fixed categories, never remote exception text."""


def status_name(value: Any) -> str:
    name = getattr(value, "name", None)
    text = str(name if name is not None else value).strip().upper()
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


def _http_status(exc: BaseException) -> int | None:
    candidates = (
        getattr(exc, "status", None),
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    )
    for value in candidates:
        if type(value) is int and 100 <= value <= 599:
            return value
    return None


def exact_metadata_eventually(
    api: Any,
    kernel: str,
    *,
    attempts: int = 6,
    delay_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Bounded reconciliation for transient exact GetKernel 403/404 only.

    Identity is still proven solely by the exact metadata response.  This does
    not fall back to search/list and never retries writes.  A 429, unknown
    transport error, or exhausted 403/404 sequence fails closed with a fixed
    non-sensitive error category.
    """

    if type(attempts) is not int or not 1 <= attempts <= MAX_EXACT_METADATA_ATTEMPTS:
        raise IdentityError("invalid_exact_metadata_attempt_bound")
    if isinstance(delay_seconds, bool) or not 0.0 <= float(delay_seconds) <= MAX_EXACT_METADATA_DELAY_SECONDS:
        raise IdentityError("invalid_exact_metadata_delay_bound")
    for index in range(attempts):
        try:
            return exact_metadata(api, kernel)
        except IdentityError:
            raise
        except Exception as exc:
            status = _http_status(exc)
            if status not in TRANSIENT_EXACT_METADATA_HTTP:
                raise IdentityError("exact_metadata_nontransient_failure") from exc
            if index + 1 >= attempts:
                raise IdentityError("exact_metadata_transient_exhausted") from exc
            sleep(float(delay_seconds))
    raise IdentityError("exact_metadata_transient_exhausted")


def verify_current(api: Any, kernel: str, version: int, *, allow_failed: bool = False, cpu: bool = False) -> str:
    metadata = exact_metadata(api, kernel)
    validate_metadata(metadata, kernel, version, cpu=cpu)
    state = status_name(getattr(api.kernels_status(kernel), "status", None))
    allowed = TERMINAL if allow_failed else {"COMPLETE"}
    if state not in allowed:
        raise IdentityError("kernel_not_in_allowed_terminal_state")
    return state


def verify_current_eventually(
    api: Any,
    kernel: str,
    version: int,
    *,
    allow_failed: bool = False,
    cpu: bool = False,
    attempts: int = 6,
    delay_seconds: float = 2.0,
) -> str:
    metadata = exact_metadata_eventually(
        api,
        kernel,
        attempts=attempts,
        delay_seconds=delay_seconds,
    )
    validate_metadata(metadata, kernel, version, cpu=cpu)
    state = status_name(getattr(api.kernels_status(kernel), "status", None))
    allowed = TERMINAL if allow_failed else {"COMPLETE"}
    if state not in allowed:
        raise IdentityError("kernel_not_in_allowed_terminal_state")
    return state


def safe_exception(exc: BaseException) -> dict[str, Any]:
    """Do not leak refs, response bodies, tokens, paths or signed URLs."""
    text = f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
    status = _http_status(exc)
    return {"exception_type": type(exc).__name__, "error_sha256": hashlib.sha256(text).hexdigest(), "http_status": status}
