#!/usr/bin/env python3
"""One-shot Kaggle Kernel write with read-only reconciliation.

This helper intentionally has no quota_view(), kernels_list(), active-session
classification, or bridge-local remote-capacity admission.  Kaggle decides
whether the requested Notebook can be accepted/run.

The helper does not automatically retry writes.  It performs exactly one
``kernels_push`` call and then uses exact read-only metadata/status reads to
classify the observed outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from pathlib import Path
from typing import Any, Callable

from kaggle_exact_identity import exact_metadata, safe_exception, status_name


@dataclass(frozen=True)
class KernelWriteOutcome:
    classification: str
    target: str
    before_version: int
    observed_version: int
    observed_state: str
    write_calls: int
    client_failure_kind: str | None
    client_http_status: int | None
    client_error_sha256: str | None


class KernelWriteError(RuntimeError):
    """Fixed local categories only; remote bodies/refs are not interpolated."""


def _version(metadata: Any) -> int:
    value = getattr(metadata, "current_version_number", None)
    if isinstance(value, bool) or value is None:
        raise KernelWriteError("current_version_unavailable")
    try:
        version = int(value)
    except Exception as exc:
        raise KernelWriteError("current_version_invalid") from exc
    if version < 1:
        raise KernelWriteError("current_version_invalid")
    return version


def _client_failure(response: Any = None, exc: BaseException | None = None) -> tuple[str | None, dict[str, Any]]:
    if exc is not None:
        return "exception", safe_exception(exc)
    error = str(getattr(response, "error", "") or "")
    if not error:
        return None, {"http_status": None, "error_sha256": None}
    return "response_error", {
        "http_status": None,
        "error_sha256": hashlib.sha256(error.encode("utf-8", errors="replace")).hexdigest(),
    }


def _observe_after_write(
    api: Any,
    target: str,
    *,
    before_version: int,
    expected_after_version: int,
    attempts: int,
    delay_seconds: float,
    sleep: Callable[[float], None],
) -> tuple[int, str, bool]:
    if type(attempts) is not int or not 1 <= attempts <= 12:
        raise KernelWriteError("invalid_reconciliation_attempts")
    if isinstance(delay_seconds, bool) or not 0.0 <= float(delay_seconds) <= 30.0:
        raise KernelWriteError("invalid_reconciliation_delay")

    last_version = before_version
    last_state = "UNKNOWN"
    saw_exact = False
    for index in range(attempts):
        try:
            metadata = exact_metadata(api, target)
            last_version = _version(metadata)
            saw_exact = True
            try:
                last_state = status_name(getattr(api.kernels_status(target), "status", None))
            except Exception:
                last_state = "UNKNOWN"
        except Exception:
            # Reconciliation is read-only.  A transient metadata failure must
            # not trigger another write; keep observing within the fixed bound.
            pass
        if last_version == expected_after_version:
            return last_version, last_state, True
        if index + 1 < attempts:
            sleep(float(delay_seconds))

    return last_version, last_state, saw_exact


def push_kernel_version_once(
    api: Any,
    kernel_dir: str | Path,
    target: str,
    *,
    expected_before_version: int,
    expected_after_version: int,
    expected_before_state: str | None = None,
    reconcile_attempts: int = 8,
    reconcile_delay_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> KernelWriteOutcome:
    """Perform exactly one Kaggle kernels_push and reconcile the result.

    ``expected_before_state`` is optional and should be used only when the
    version/state semantics are part of the approved operation, e.g. replacing
    an explicitly failed version.  No unrelated capacity/session state is read.
    """

    if type(expected_before_version) is not int or type(expected_after_version) is not int:
        raise KernelWriteError("invalid_expected_versions")
    if expected_after_version != expected_before_version + 1:
        raise KernelWriteError("expected_version_increment_must_be_one")

    before = exact_metadata(api, target)
    if getattr(before, "ref", None) != target:
        raise KernelWriteError("target_ref_mismatch")
    before_version = _version(before)
    if before_version != expected_before_version:
        raise KernelWriteError("before_version_mismatch")

    if expected_before_state is not None:
        state = status_name(getattr(api.kernels_status(target), "status", None))
        if state != expected_before_state.upper():
            raise KernelWriteError("before_state_mismatch")

    response = None
    failure_exc: BaseException | None = None
    try:
        response = api.kernels_push(str(kernel_dir))
    except Exception as exc:  # exactly one write attempt
        failure_exc = exc

    failure_kind, failure_info = _client_failure(response=response, exc=failure_exc)

    observed_version, observed_state, saw_exact = _observe_after_write(
        api,
        target,
        before_version=before_version,
        expected_after_version=expected_after_version,
        attempts=reconcile_attempts,
        delay_seconds=reconcile_delay_seconds,
        sleep=sleep,
    )

    if observed_version == expected_after_version:
        return KernelWriteOutcome(
            classification="write_observed",
            target=target,
            before_version=before_version,
            observed_version=observed_version,
            observed_state=observed_state,
            write_calls=1,
            client_failure_kind=failure_kind,
            client_http_status=failure_info.get("http_status"),
            client_error_sha256=failure_info.get("error_sha256"),
        )

    if failure_kind is not None and saw_exact and observed_version == before_version:
        return KernelWriteOutcome(
            classification="platform_rejected_no_side_effect",
            target=target,
            before_version=before_version,
            observed_version=observed_version,
            observed_state=observed_state,
            write_calls=1,
            client_failure_kind=failure_kind,
            client_http_status=failure_info.get("http_status"),
            client_error_sha256=failure_info.get("error_sha256"),
        )

    # Never infer that it is safe to send a second write when the first call
    # returned success/unknown but the expected version has not been proven.
    raise KernelWriteError("ambiguous_write_reconciliation_required")
