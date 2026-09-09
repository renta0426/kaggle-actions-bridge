#!/usr/bin/env python3
"""Validate aggregate-only output from the P1-03 50-row LUMIA runtime gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_REQUEST_ID = "20260909-poisoned-chalice-lumia-hs-runtime-50-readout-v1"
EXPECTED_TARGET = "renta0426/poisoned-chalice-lumia-hs-runtime-50-v1"
EXPECTED_TASK = "P1-03-lumia-hidden-state-runtime-pilot-50-v1"
EXPECTED_MODEL_ID = "bigcode/starcoder2-3b"
EXPECTED_MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
EXPECTED_DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
EXPECTED_AUTHOR_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
EXPECTED_ROWS = 50
EXPECTED_MAX_LENGTH = 8192
GATE_MINUTES = 105.0


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_request(request: dict[str, Any]) -> None:
    assert request == {
        "schema_version": 1,
        "request_id": EXPECTED_REQUEST_ID,
        "competition": "poisoned-chalice-icse27",
        "operation": "current_kernel_runtime_readout",
        "target_kernel": EXPECTED_TARGET,
        "kernel_version": 1,
        "expected_outputs": [
            "lumia_hidden_state_runtime_pilot_50/runtime.json",
            "lumia_hidden_state_runtime_pilot_50/run_manifest.json",
        ],
        "side_effects": [],
        "automatic_retries": 0,
        "publish_private_outputs": False,
        "publish_only_aggregate_runtime": True,
        "scientific_contract": {
            "task_id": EXPECTED_TASK,
            "rows": EXPECTED_ROWS,
            "max_length": EXPECTED_MAX_LENGTH,
            "direct_1000_gate_estimated_wall_minutes_lte": 105,
            "performance_metrics_computed": False,
            "sample_level_outputs_persisted": False,
            "automatic_promotion": False,
        },
    }


def _finite_number(value: Any, name: str, *, nonnegative: bool = True) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"non-finite {name}: {value!r}")
    if nonnegative and number < 0:
        raise RuntimeError(f"negative {name}: {number}")
    return number


def evaluate(runtime: dict[str, Any], manifest: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    validate_request(request)
    if runtime.get("schema_version") != 1 or runtime.get("task_id") != EXPECTED_TASK:
        raise RuntimeError("runtime identity mismatch")
    if runtime.get("author_commit") != EXPECTED_AUTHOR_COMMIT:
        raise RuntimeError("author commit mismatch")
    if runtime.get("rows") != EXPECTED_ROWS:
        raise RuntimeError("runtime row count mismatch")
    if runtime.get("labels_present_during_model_scoring") is not False:
        raise RuntimeError("labels were present during scoring")
    if runtime.get("performance_metrics_computed") is not False:
        raise RuntimeError("performance metrics unexpectedly computed")
    if runtime.get("persistent_sample_level_outputs") is not False:
        raise RuntimeError("sample-level outputs unexpectedly persisted")

    config = runtime.get("config") or {}
    if config.get("model_id") != EXPECTED_MODEL_ID or config.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise RuntimeError("model identity mismatch")
    if config.get("dataset_revision") != EXPECTED_DATASET_REVISION:
        raise RuntimeError("dataset revision mismatch")
    if int(config.get("max_length", -1)) != EXPECTED_MAX_LENGTH or int(config.get("rows", -1)) != EXPECTED_ROWS:
        raise RuntimeError("runtime config changed")

    layers = int(runtime.get("num_layers", 0))
    hidden = int(runtime.get("hidden_size", 0))
    if layers <= 0 or hidden <= 0:
        raise RuntimeError("invalid activation dimensions")

    token_count = runtime.get("token_count") or {}
    token_min = int(token_count.get("min", 0))
    token_max = int(token_count.get("max", 0))
    token_sum = int(token_count.get("sum", 0))
    truncated = int(token_count.get("truncated_at_max_length", -1))
    if token_min <= 0 or token_max > EXPECTED_MAX_LENGTH or token_sum <= 0 or not (0 <= truncated <= EXPECTED_ROWS):
        raise RuntimeError("token-count diagnostics invalid")

    diagnostics = runtime.get("diagnostics") or {}
    if int(diagnostics.get("caller_generic_linter_samples", -1)) != EXPECTED_ROWS:
        raise RuntimeError("caller-literal generic-linter gate failed")
    if int(diagnostics.get("finite_output_baseline_samples", -1)) != EXPECTED_ROWS:
        raise RuntimeError("finite output-score gate failed")

    timing = runtime.get("timing") or {}
    wall_seconds = _finite_number(timing.get("wall_seconds"), "wall_seconds")
    forward_seconds_sum = _finite_number(timing.get("forward_seconds_sum"), "forward_seconds_sum")
    estimated_wall = _finite_number(timing.get("estimated_1000_minutes_wall_linear"), "estimated_1000_minutes_wall_linear")
    estimated_forward = _finite_number(timing.get("estimated_1000_minutes_forward_linear"), "estimated_1000_minutes_forward_linear")
    gate_pass = estimated_wall <= GATE_MINUTES
    expected_recommendation = (
        "direct_1000_row_t4_feasible_under_120m_gate"
        if gate_pass
        else "do_not_launch_1000_rows_on_same_single_t4_protocol_without_resource_or_scope_revision"
    )
    if runtime.get("runtime_recommendation") != expected_recommendation:
        raise RuntimeError("runtime recommendation is inconsistent with frozen gate")

    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete" or manifest.get("task_id") != EXPECTED_TASK:
        raise RuntimeError("run manifest identity/status mismatch")
    if manifest.get("model_id") != EXPECTED_MODEL_ID or manifest.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise RuntimeError("run manifest model mismatch")
    if manifest.get("dataset_revision") != EXPECTED_DATASET_REVISION or manifest.get("author_commit") != EXPECTED_AUTHOR_COMMIT:
        raise RuntimeError("run manifest provenance mismatch")
    if manifest.get("rows") != EXPECTED_ROWS or manifest.get("max_length") != EXPECTED_MAX_LENGTH:
        raise RuntimeError("run manifest scope mismatch")
    if manifest.get("labels_present_during_model_scoring") is not False or manifest.get("performance_metrics_computed") is not False:
        raise RuntimeError("run manifest clean-room boundary mismatch")
    if manifest.get("persistent_sample_level_outputs") is not False:
        raise RuntimeError("run manifest sample-output boundary mismatch")
    if manifest.get("persistent_outputs") != ["runtime.json", "run_manifest.json"]:
        raise RuntimeError("run manifest persistent-output allowlist mismatch")
    if manifest.get("runtime_recommendation") != expected_recommendation:
        raise RuntimeError("runtime/manifest recommendation mismatch")

    return {
        "task_id": EXPECTED_TASK,
        "rows": EXPECTED_ROWS,
        "max_length": EXPECTED_MAX_LENGTH,
        "num_layers": layers,
        "hidden_size": hidden,
        "token_min": token_min,
        "token_median": float(token_count.get("median")),
        "token_max": token_max,
        "token_sum": token_sum,
        "truncated_at_max_length": truncated,
        "wall_seconds": wall_seconds,
        "wall_minutes": wall_seconds / 60.0,
        "forward_seconds_sum": forward_seconds_sum,
        "estimated_1000_minutes_wall_linear": estimated_wall,
        "estimated_1000_minutes_forward_linear": estimated_forward,
        "caller_generic_linter_samples": int(diagnostics.get("caller_generic_linter_samples")),
        "helper_tree_sitter_samples": int(diagnostics.get("helper_tree_sitter_samples", 0)),
        "helper_flake8_samples": int(diagnostics.get("helper_flake8_samples", 0)),
        "finite_output_baseline_samples": int(diagnostics.get("finite_output_baseline_samples")),
        "gate_threshold_minutes": GATE_MINUTES,
        "direct_1000_gate_pass": gate_pass,
        "runtime_recommendation": expected_recommendation,
        "performance_metrics_computed": False,
        "sample_level_outputs_persisted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--runtime")
    parser.add_argument("--manifest")
    parser.add_argument("--static", action="store_true")
    args = parser.parse_args()
    request = _load(args.request)
    if args.static:
        validate_request(request)
        print("LUMIA_RUNTIME_READOUT_STATIC PASS version=1 side_effects=0 aggregate_only=true")
        return 0
    if not args.runtime or not args.manifest:
        parser.error("--runtime and --manifest are required unless --static is used")
    summary = evaluate(_load(args.runtime), _load(args.manifest), request)
    print("LUMIA_RUNTIME_READOUT_RESULT " + json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
