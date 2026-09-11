#!/usr/bin/env python3
"""Read-only successor for the failed P1-02 version-1 recovery.

The predecessor GitHub Actions run 34240887808 failed before any write or
compute request with an HTTP 403 during the remote identity read.  This
successor preserves the exact target, version, source-identity proof, output
allowlist, and scientific validation.  Its only repair is bounded reconciliation
of the two exact read-only identity/status endpoints on HTTP 403/404.

This program has no Kaggle write, run-start, submission, target substitution, or
version substitution path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import poisoned_chalice_p1_02_start_boundary_recover_v1 as base
from kaggle_exact_identity import (
    TERMINAL,
    exact_metadata_eventually,
    safe_exception,
    status_name,
    validate_metadata,
)

REQUEST_ID = "20260912-poisoned-chalice-p1-02-start-boundary-readonly-recovery-v2-001"
PREDECESSOR_REQUEST_ID = "20260908-poisoned-chalice-p1-02-start-boundary-readonly-recovery-v1-001"
PREDECESSOR_RUN_ID = 34240887808
ORIGINAL_REQUEST_ID = "20260906-poisoned-chalice-shadow-gold-start-boundary-attribution-v1-001"
TARGET = "renta0426/shadow-gold-start-boundary-attribution-v1"
VERSION = 1
RESEARCH_COMMIT = "6b6f388627dfcf16ab0140337cabe7e93f4a8891"
EXECUTION_ID = "shadow-gold-start-boundary-attribution-v1"
READ_ATTEMPTS = 6
READ_DELAY_SECONDS = 2.0
TRANSIENT_READ_HTTP = frozenset({403, 404})


def _validate_base_contract() -> None:
    expected = {
        "TARGET": TARGET,
        "VERSION": VERSION,
        "RESEARCH_COMMIT": RESEARCH_COMMIT,
        "EXECUTION_ID": EXECUTION_ID,
        "EXPECTED_PREDICTION_ROWS": 2560,
    }
    for name, value in expected.items():
        if getattr(base, name, None) != value:
            raise RuntimeError(f"predecessor_contract_drift:{name}")
    if set(base.EXPECTED_OUTPUTS) != {
        "start_boundary_decision.json",
        "start_boundary_metrics.json",
        "start_boundary_predictions.jsonl",
        "start_boundary_runtime_manifest.json",
    }:
        raise RuntimeError("predecessor_output_contract_drift")


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


def kernels_status_eventually(
    api: Any,
    *,
    attempts: int = READ_ATTEMPTS,
    delay_seconds: float = READ_DELAY_SECONDS,
) -> str:
    """Retry only the same read-only status call on HTTP 403/404."""
    if type(attempts) is not int or not 1 <= attempts <= READ_ATTEMPTS:
        raise ValueError("invalid_status_attempt_bound")
    if isinstance(delay_seconds, bool) or not 0.0 <= float(delay_seconds) <= READ_DELAY_SECONDS:
        raise ValueError("invalid_status_delay_bound")
    for index in range(attempts):
        try:
            return status_name(getattr(api.kernels_status(TARGET), "status", None))
        except Exception as exc:
            if _http_status(exc) not in TRANSIENT_READ_HTTP:
                raise
            if index + 1 >= attempts:
                raise RuntimeError("kernel_status_transient_exhausted") from exc
            time.sleep(float(delay_seconds))
    raise RuntimeError("kernel_status_transient_exhausted")


def validate_remote_identity(api: Any) -> str:
    # Fixed stage markers identify the failing read without exposing private data.
    print("P1_02_RECOVERY_READ_STAGE exact_get_kernel")
    metadata = exact_metadata_eventually(
        api,
        TARGET,
        attempts=READ_ATTEMPTS,
        delay_seconds=READ_DELAY_SECONDS,
    )
    validate_metadata(metadata, TARGET, VERSION, cpu=False)
    if getattr(metadata, "enable_gpu", None) is not True:
        raise RuntimeError("gpu_contract_not_proven")
    if getattr(metadata, "enable_tpu", None) is not False:
        raise RuntimeError("tpu_contract_not_proven")
    if getattr(metadata, "enable_internet", None) is not False:
        raise RuntimeError("offline_contract_not_proven")

    print("P1_02_RECOVERY_READ_STAGE kernel_status")
    state = kernels_status_eventually(api)
    if state not in TERMINAL:
        raise RuntimeError("kernel_not_in_allowed_terminal_state")
    return state


def recover(bridge_root: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError("recovery_output_exists")
    _validate_base_contract()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    state = validate_remote_identity(api)
    with tempfile.TemporaryDirectory(prefix="pc-p1-02-recovery-v2-") as tmp:
        root = Path(tmp)
        source_sha256 = base.prove_source_identity(bridge_root, root)
        decision, metrics, manifest, predictions = base.load_outputs(root)

        # Re-check the exact same target/version after the reads so concurrent
        # mutation cannot pass.  This is still read-only and bounded.
        second_state = validate_remote_identity(api)
        if second_state != state:
            raise RuntimeError("remote_state_changed_during_recovery")

        aggregates = base.validate_science(decision, metrics, manifest, predictions)
        receipt = {
            "request_id": REQUEST_ID,
            "predecessor_request_id": PREDECESSOR_REQUEST_ID,
            "predecessor_run_id": PREDECESSOR_RUN_ID,
            "original_request_id": ORIGINAL_REQUEST_ID,
            "target": TARGET,
            "version": VERSION,
            "remote_status": state,
            "research_commit": RESEARCH_COMMIT,
            "notebook_code_sha256": source_sha256,
            "new_write_attempted": False,
            "new_compute_requested": False,
            "competition_submission_attempted": False,
            "automatic_compute_retries": 0,
            "bounded_read_reconciliation": {
                "exact_get_kernel_attempts": READ_ATTEMPTS,
                "kernel_status_attempts": READ_ATTEMPTS,
                "delay_seconds": READ_DELAY_SECONDS,
                "retry_http_statuses": sorted(TRANSIENT_READ_HTTP),
            },
            "scientific_outputs_validated": True,
            "prediction_sha256": base.sha256_file(predictions),
        }
        output_dir.mkdir(parents=True)
        (output_dir / "recovery.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("P1_02_START_BOUNDARY_RECOVERY_V2_VERIFIED " + json.dumps(receipt, sort_keys=True))
        print("P1_02_START_BOUNDARY_AGGREGATES_BEGIN")
        print(json.dumps(aggregates, sort_keys=True))
        print("P1_02_START_BOUNDARY_AGGREGATES_END")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        recover(args.bridge_root.resolve(), args.output_dir.resolve())
    except Exception as exc:
        print(
            "P1_02_START_BOUNDARY_RECOVERY_V2_FAILED "
            + json.dumps(
                {
                    "request_id": REQUEST_ID,
                    "predecessor_run_id": PREDECESSOR_RUN_ID,
                    "new_write_attempted": False,
                    "new_compute_requested": False,
                    **safe_exception(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
