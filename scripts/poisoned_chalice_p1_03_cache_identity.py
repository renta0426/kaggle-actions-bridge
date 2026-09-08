#!/usr/bin/env python3
"""Observe exactly two historical cache producer identities; no output read/write."""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from kaggle_exact_identity import IdentityError, exact_metadata, safe_exception, validate_metadata

TARGETS = (
    ("renta0426/starter-plus-10k-v2", 1),
    ("renta0426/starter-plus-10k-v2-continuation", 1),
)


def safe_projection(metadata, kernel: str, expected_version: int) -> dict:
    if getattr(metadata, "ref", None) != kernel:
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
        validate_metadata(metadata, kernel, expected_version)
        matches, category = True, "frozen_version_matches"
    except IdentityError as exc:
        matches, category = False, str(exc)
    except (TypeError, ValueError, OverflowError):
        matches, category = False, "version_field_not_integer_compatible"
    return {
        "expected_kernel": kernel,
        "expected_current_version": expected_version,
        "observed_current_version_number": observed,
        "observed_version_type": type(value).__name__[:40],
        "ref_matches": True,
        "private_flag_proven": True,
        "frozen_version_matches": matches,
        "version_validation_category": category,
    }


def diagnose(api, read=exact_metadata) -> dict:
    observations = []
    for kernel, version in TARGETS:
        observations.append(safe_projection(read(api, kernel), kernel, version))
    return {
        "schema_version": "poisoned_chalice_p1_03_cache_identity_v1",
        "status": "metadata_observation_complete_not_output_read",
        "observations": observations,
        "all_frozen_versions_match": all(item["frozen_version_matches"] for item in observations),
        "exact_get_kernel_invocations": 2,
        "output_download_invocations": 0,
        "kaggle_write_calls": 0,
        "kaggle_compute_started": False,
        "membership_source_opened": False,
        "performance_metrics_computed": False,
        "automatic_recovery_or_version_substitution": False,
    }


def self_test() -> None:
    calls = []
    versions = {TARGETS[0][0]: 1, TARGETS[1][0]: 1}
    def read(api, kernel):
        calls.append(kernel)
        return SimpleNamespace(ref=kernel, is_private=True, current_version_number=versions[kernel])
    result = diagnose(object(), read=read)
    assert calls == [item[0] for item in TARGETS]
    assert result["all_frozen_versions_match"] is True
    assert result["exact_get_kernel_invocations"] == 2
    assert result["output_download_invocations"] == 0

    versions[TARGETS[1][0]] = 2
    result = diagnose(object(), read=read)
    assert result["all_frozen_versions_match"] is False
    assert result["observations"][1]["observed_current_version_number"] == 2
    assert result["observations"][1]["frozen_version_matches"] is False

    try:
        safe_projection(SimpleNamespace(ref="other/kernel", is_private=True, current_version_number=1), TARGETS[0][0], 1)
    except IdentityError:
        pass
    else:
        raise AssertionError("ref boundary not enforced")
    print("P1_03_CACHE_IDENTITY_SELF_TEST PASS exact_reads=2 no_output=1 no_write=1 no_compute=1 mismatch_visible=1")


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
        print("P1_03_CACHE_IDENTITY_FAILED " + json.dumps(safe_exception(exc), sort_keys=True))
        raise SystemExit(1) from None
    print("P1_03_CACHE_IDENTITY " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
