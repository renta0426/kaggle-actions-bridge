#!/usr/bin/env python3
"""Observe exact current identities for the two historical P1-03 cache producers.

This diagnostic is deliberately metadata-only.  The unknown continuation version
is observed but cannot be used for output access in this run; it must first be
copied into a separate immutable request.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from kaggle_exact_identity import IdentityError, exact_metadata, safe_exception, status_name


EXPECTED_OPERATION = "observe_exact_current_identity_only"


def _load_request(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("operation") != EXPECTED_OPERATION:
        raise IdentityError("invalid_identity_request")
    if data.get("environment") != "kaggle-readonry":
        raise IdentityError("invalid_identity_environment")
    exact_counts = {
        "exact_get_kernel_invocations": 2,
        "exact_status_invocations": 2,
        "output_list_invocations": 0,
        "output_download_invocations": 0,
        "automatic_retries": 0,
        "kaggle_write_calls": 0,
    }
    for key, expected in exact_counts.items():
        if data.get(key) != expected:
            raise IdentityError("invalid_identity_call_budget")
    false_flags = (
        "kaggle_compute_started",
        "performance_evaluation_allowed",
        "membership_sources_allowed",
        "version_substitution_allowed",
        "downstream_use_of_observed_identity_allowed_in_this_run",
    )
    if any(data.get(key) is not False for key in false_flags):
        raise IdentityError("invalid_identity_side_effect_contract")

    targets = data.get("targets")
    if not isinstance(targets, dict) or set(targets) != {"primary", "continuation"}:
        raise IdentityError("invalid_identity_targets")
    primary = targets["primary"]
    continuation = targets["continuation"]
    if primary != {
        "kernel": "renta0426/starter-plus-10k-v2",
        "expected_current_version": 1,
        "expected_status": "ERROR",
        "identity_mode": "verify_known_identity",
    }:
        raise IdentityError("invalid_primary_identity_contract")
    if continuation != {
        "kernel": "renta0426/starter-plus-10k-v2-continuation",
        "identity_mode": "observe_only_then_freeze_in_separate_request",
    }:
        raise IdentityError("invalid_continuation_identity_contract")
    return data


def _observe(
    api: Any,
    target: dict[str, Any],
    *,
    exact_reader: Callable[[Any, str], Any] = exact_metadata,
) -> dict[str, Any]:
    kernel = target["kernel"]
    metadata = exact_reader(api, kernel)
    if getattr(metadata, "ref", None) != kernel:
        raise IdentityError("kernel_ref_mismatch")
    if getattr(metadata, "is_private", None) is not True:
        raise IdentityError("private_flag_not_proven")
    observed_raw = getattr(metadata, "current_version_number", None)
    if isinstance(observed_raw, bool) or observed_raw is None:
        raise IdentityError("current_version_not_proven")
    try:
        version = int(observed_raw)
    except (TypeError, ValueError) as exc:
        raise IdentityError("current_version_not_proven") from exc
    if version < 1:
        raise IdentityError("current_version_not_proven")

    state = status_name(getattr(api.kernels_status(kernel), "status", None))
    if target["identity_mode"] == "verify_known_identity":
        if version != target["expected_current_version"]:
            raise IdentityError("current_version_mismatch")
        if state != target["expected_status"]:
            raise IdentityError("kernel_status_mismatch")
    elif target["identity_mode"] != "observe_only_then_freeze_in_separate_request":
        raise IdentityError("invalid_identity_mode")

    return {
        "kernel": kernel,
        "current_version": version,
        "status": state,
        "private": True,
        "observation_only": target["identity_mode"] != "verify_known_identity",
    }


def _self_test() -> None:
    class FakeApi:
        def __init__(self) -> None:
            self.states = {
                "renta0426/starter-plus-10k-v2": "ERROR",
                "renta0426/starter-plus-10k-v2-continuation": "COMPLETE",
            }

        def kernels_status(self, kernel: str) -> Any:
            return SimpleNamespace(status=self.states[kernel])

    metadata = {
        "renta0426/starter-plus-10k-v2": SimpleNamespace(
            ref="renta0426/starter-plus-10k-v2",
            is_private=True,
            current_version_number=1,
        ),
        "renta0426/starter-plus-10k-v2-continuation": SimpleNamespace(
            ref="renta0426/starter-plus-10k-v2-continuation",
            is_private=True,
            current_version_number=7,
        ),
    }

    def fake_reader(api: Any, kernel: str) -> Any:
        del api
        return metadata[kernel]

    primary = {
        "kernel": "renta0426/starter-plus-10k-v2",
        "expected_current_version": 1,
        "expected_status": "ERROR",
        "identity_mode": "verify_known_identity",
    }
    continuation = {
        "kernel": "renta0426/starter-plus-10k-v2-continuation",
        "identity_mode": "observe_only_then_freeze_in_separate_request",
    }
    first = _observe(FakeApi(), primary, exact_reader=fake_reader)
    second = _observe(FakeApi(), continuation, exact_reader=fake_reader)
    assert first["current_version"] == 1 and first["status"] == "ERROR"
    assert first["observation_only"] is False
    assert second["current_version"] == 7 and second["status"] == "COMPLETE"
    assert second["observation_only"] is True
    print("SERSEM_CACHE_IDENTITY_SELF_TEST PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if args.request is None:
        raise SystemExit("--request is required")

    try:
        request = _load_request(args.request)
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        observations = {
            name: _observe(api, request["targets"][name])
            for name in ("primary", "continuation")
        }
    except Exception as exc:
        print(
            "SERSEM_CACHE_IDENTITY_FAILURE "
            + json.dumps(safe_exception(exc), sort_keys=True),
            flush=True,
        )
        return 1

    result = {
        "request_id": request["request_id"],
        "science_sha": request["science_sha"],
        "observations": observations,
        "outputs_listed": 0,
        "outputs_downloaded": 0,
        "kaggle_writes": 0,
        "compute_started": False,
        "downstream_use_permitted": False,
    }
    print("SERSEM_CACHE_IDENTITY_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
