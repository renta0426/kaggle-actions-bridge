#!/usr/bin/env python3
"""One exact metadata observation; no output download or automatic recovery."""
from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

from kaggle_exact_identity import IdentityError, exact_metadata, safe_exception, validate_metadata

KERNEL = "renta0426/codeparrot-fresh-full-v1"
EXPECTED_VERSION = 1


def safe_projection(metadata) -> dict:
    if getattr(metadata, "ref", None) != KERNEL:
        raise IdentityError("kernel_ref_mismatch")
    if getattr(metadata, "is_private", None) is not True:
        raise IdentityError("private_flag_not_proven")
    value = getattr(metadata, "current_version_number", None)
    observed = None
    if type(value) is int and 0 <= value <= 10**12:
        observed = value
    elif type(value) is str and value.isascii() and value.isdecimal() and len(value) <= 13:
        observed = value
    elif type(value) is float and value.is_integer() and 0 <= value <= 10**12:
        observed = value
    try:
        validate_metadata(metadata, KERNEL, EXPECTED_VERSION)
        matches, category = True, "frozen_version_matches"
    except IdentityError as exc:
        matches, category = False, str(exc)
    except (TypeError, ValueError, OverflowError):
        matches, category = False, "version_field_not_integer_compatible"
    return {
        "schema_version": "codeparrot_current_identity_diagnostic_v1",
        "status": "metadata_observation_complete_not_evaluation",
        "expected_kernel": KERNEL,
        "ref_matches": True,
        "private_flag_proven": True,
        "expected_current_version": EXPECTED_VERSION,
        "observed_current_version_number": observed,
        "observed_version_type": type(value).__name__[:40],
        "frozen_version_matches": matches,
        "version_validation_category": category,
        "exact_get_kernel_invocations": 1,
        "output_download_invocations": 0,
        "kaggle_write_calls": 0,
        "kaggle_compute_started": False,
        "membership_source_opened": False,
        "performance_metrics_computed": False,
        "automatic_recovery_or_version_substitution": False,
    }


def diagnose(api, read=exact_metadata) -> dict:
    return safe_projection(read(api, KERNEL))


def self_test() -> None:
    for value, expected in ((1, True), (2, False), (None, False), (True, False), ("2", False), ("private-canary", False)):
        calls = []
        def read(api, kernel):
            calls.append(kernel)
            return SimpleNamespace(ref=KERNEL, is_private=True, current_version_number=value)
        result = diagnose(object(), read=read)
        assert calls == [KERNEL]
        assert result["frozen_version_matches"] is expected
        assert result["output_download_invocations"] == 0
        assert "private-canary" not in json.dumps(result)
    for metadata in (SimpleNamespace(ref="other/notebook", is_private=True, current_version_number=1), SimpleNamespace(ref=KERNEL, is_private=False, current_version_number=1)):
        try:
            safe_projection(metadata)
        except IdentityError:
            pass
        else:
            raise AssertionError("identity_boundary_not_enforced")
    calls = []
    def failed_read(api, kernel):
        calls.append(kernel)
        raise RuntimeError("synthetic failure")
    try:
        diagnose(object(), read=failed_read)
    except RuntimeError:
        pass
    else:
        raise AssertionError("read_error_not_propagated")
    assert calls == [KERNEL]
    print("CODEPARROT_IDENTITY_DIAGNOSTIC_SELF_TEST PASS version_1_2_missing_bool=1 safe_projection=1 exact_read_once=1 no_retry=1 no_download=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        result = diagnose(api)
    except Exception as exc:
        print("CODEPARROT_IDENTITY_DIAGNOSTIC_FAILED " + json.dumps(safe_exception(exc), sort_keys=True))
        raise SystemExit(1) from None
    print("CODEPARROT_IDENTITY_DIAGNOSTIC " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
