#!/usr/bin/env python3
"""Regression tests for bounded exact GetKernel reconciliation.

No Kaggle credential, Kaggle package, or network access is used. These tests
encode the observed recurrent failure class: a successful write can be followed
by a transient exact-metadata HTTP 403/404. Only those read failures may be
retried; 429 and all other errors fail immediately, and exact identity is never
replaced by a search/list result.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import kaggle_exact_identity as identity

KERNEL = "renta0426/example-private-kernel"
ROOT = Path(__file__).resolve().parent


class FakeHttpError(RuntimeError):
    def __init__(self, status: int):
        super().__init__("remote detail intentionally ignored")
        self.response = SimpleNamespace(status_code=status)


def metadata(*, version: int = 1, private: bool = True):
    return SimpleNamespace(
        ref=KERNEL,
        is_private=private,
        current_version_number=version,
        enable_gpu=False,
        enable_tpu=False,
        enable_internet=False,
    )


def expect_identity_error(category: str, fn) -> None:
    try:
        fn()
    except identity.IdentityError as exc:
        assert str(exc) == category, (category, exc)
    else:
        raise AssertionError(f"expected IdentityError:{category}")


def test_transient_403_then_exact_metadata_succeeds() -> None:
    original = identity.exact_metadata
    calls = []
    sleeps = []
    sequence = [FakeHttpError(403), metadata()]
    try:
        def fake(api, kernel):
            assert kernel == KERNEL
            calls.append(kernel)
            value = sequence.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value
        identity.exact_metadata = fake
        observed = identity.exact_metadata_eventually(
            object(), KERNEL, attempts=3, delay_seconds=0.25, sleep=sleeps.append
        )
        identity.validate_metadata(observed, KERNEL, 1, cpu=True)
    finally:
        identity.exact_metadata = original
    assert len(calls) == 2
    assert sleeps == [0.25]


def test_transient_404_then_exact_metadata_succeeds() -> None:
    original = identity.exact_metadata
    calls = []
    try:
        def fake(api, kernel):
            calls.append(kernel)
            if len(calls) == 1:
                raise FakeHttpError(404)
            return metadata()
        identity.exact_metadata = fake
        observed = identity.exact_metadata_eventually(
            object(), KERNEL, attempts=2, delay_seconds=0.0, sleep=lambda _: None
        )
        identity.validate_metadata(observed, KERNEL, 1, cpu=True)
    finally:
        identity.exact_metadata = original
    assert len(calls) == 2


def test_persistent_403_fails_closed_without_unbounded_retry() -> None:
    original = identity.exact_metadata
    calls = []
    sleeps = []
    try:
        def fake(api, kernel):
            calls.append(kernel)
            raise FakeHttpError(403)
        identity.exact_metadata = fake
        expect_identity_error(
            "exact_metadata_transient_exhausted",
            lambda: identity.exact_metadata_eventually(
                object(), KERNEL, attempts=3, delay_seconds=0.1, sleep=sleeps.append
            ),
        )
    finally:
        identity.exact_metadata = original
    assert len(calls) == 3
    assert sleeps == [0.1, 0.1]


def test_429_is_never_retried() -> None:
    original = identity.exact_metadata
    calls = []
    sleeps = []
    try:
        def fake(api, kernel):
            calls.append(kernel)
            raise FakeHttpError(429)
        identity.exact_metadata = fake
        expect_identity_error(
            "exact_metadata_nontransient_failure",
            lambda: identity.exact_metadata_eventually(
                object(), KERNEL, attempts=6, delay_seconds=1.0, sleep=sleeps.append
            ),
        )
    finally:
        identity.exact_metadata = original
    assert len(calls) == 1
    assert sleeps == []


def test_identity_mismatch_after_transient_visibility_still_fails() -> None:
    original = identity.exact_metadata
    calls = []
    try:
        def fake(api, kernel):
            calls.append(kernel)
            if len(calls) == 1:
                raise FakeHttpError(403)
            return metadata(version=2)
        identity.exact_metadata = fake
        observed = identity.exact_metadata_eventually(
            object(), KERNEL, attempts=3, delay_seconds=0.0, sleep=lambda _: None
        )
        expect_identity_error(
            "current_version_mismatch",
            lambda: identity.validate_metadata(observed, KERNEL, 1, cpu=True),
        )
    finally:
        identity.exact_metadata = original
    assert len(calls) == 2


def test_bounds_fail_closed() -> None:
    expect_identity_error(
        "invalid_exact_metadata_attempt_bound",
        lambda: identity.exact_metadata_eventually(object(), KERNEL, attempts=0),
    )
    expect_identity_error(
        "invalid_exact_metadata_delay_bound",
        lambda: identity.exact_metadata_eventually(object(), KERNEL, delay_seconds=31.0),
    )


def test_e05_executor_is_wired_to_common_bounded_exact_helper() -> None:
    text = (ROOT / "cmi_flu_strategy_e05_execute_v2.py").read_text(encoding="utf-8")
    assert "from kaggle_exact_identity import exact_metadata_eventually" in text
    assert "return exact_metadata_eventually(api, ref, attempts=8, delay_seconds=3.0)" in text
    # Old direct SDK GetKernel implementation must not remain in the reusable E05 base executor.
    assert "ApiGetKernelRequest" not in text
    assert "kernels_api_client.get_kernel" not in text


def test_current_output_reader_uses_bounded_exact_helper_before_and_after_download() -> None:
    text = (ROOT / "kaggle_current_output_read.py").read_text(encoding="utf-8")
    assert "verify_current_eventually" in text
    assert "exact_metadata_eventually(api, kernel)" in text
    assert "exact_metadata(api, kernel)" not in text


def main() -> int:
    for test in (
        test_transient_403_then_exact_metadata_succeeds,
        test_transient_404_then_exact_metadata_succeeds,
        test_persistent_403_fails_closed_without_unbounded_retry,
        test_429_is_never_retried,
        test_identity_mismatch_after_transient_visibility_still_fails,
        test_bounds_fail_closed,
        test_e05_executor_is_wired_to_common_bounded_exact_helper,
        test_current_output_reader_uses_bounded_exact_helper_before_and_after_download,
    ):
        test()
    print(
        "KAGGLE_EXACT_IDENTITY_EVENTUAL_PASS "
        "transient_403=true transient_404=true persistent_fail_closed=true "
        "rate_limit_retry=false identity_fallback=false e05_common_helper=true output_common_helper=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
