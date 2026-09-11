#!/usr/bin/env python3
"""Read-only P1-02 recovery after fixing a validator-only JSON key-order bug.

Recovery v3 proved the canonical private Notebook exists and that exact
GetKernel/status, source pull, and the frozen four-output download all succeed.
It then failed with the fixed error category ``condition_order_contract:left``.
The persisted metrics were written by the frozen Notebook with
``json.dumps(..., sort_keys=True)``, so dictionary serialization order cannot be
a scientific invariant.

This successor changes only that validator invariant: each architecture must
contain exactly the frozen condition *set*, while metrics are still validated in
the frozen semantic order. No Kaggle write, compute, submission, recomputation,
target substitution, version substitution, or scientific threshold change is
possible here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import poisoned_chalice_p1_02_start_boundary_recover_v1 as science_v1
import poisoned_chalice_p1_02_start_boundary_recover_v3 as recovery_v3
from kaggle_exact_identity import safe_exception

REQUEST_ID = "20260912-poisoned-chalice-p1-02-start-boundary-order-independent-recovery-v4-001"
PREDECESSOR_REQUEST_ID = recovery_v3.REQUEST_ID
PREDECESSOR_RUN_ID = 34623427221
PREDECESSOR_ERROR_SHA256 = "6b512e031236ce5c4f6a4917f0927ceea5b40adee4b2e058d89e7454cc8d5182"
TARGET = recovery_v3.TARGET
VERSION = recovery_v3.VERSION
EXPECTED_CONDITIONS = tuple(science_v1.EXPECTED_CONDITIONS)
EXPECTED_PREDICTION_ROWS = science_v1.EXPECTED_PREDICTION_ROWS


def validate_condition_metrics(condition_metrics: dict) -> None:
    if set(condition_metrics) != {"left", "right"}:
        raise RuntimeError("architecture_key_contract")
    expected = set(EXPECTED_CONDITIONS)
    for side in ("left", "right"):
        rows = condition_metrics[side]
        if not isinstance(rows, dict) or set(rows) != expected:
            raise RuntimeError(f"condition_key_contract:{side}")
        # Iterate in the frozen semantic order. Persistence order is irrelevant.
        for condition in EXPECTED_CONDITIONS:
            row = rows[condition]
            if not isinstance(row, dict):
                raise RuntimeError(f"condition_metric_row_contract:{side}:{condition}")
            for key in ("auc", "tpr_at_1pct_fpr"):
                value = row.get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise RuntimeError(f"condition_metric_missing:{side}:{condition}:{key}")


def validate_science(decision: dict, metrics: dict, manifest: dict, predictions: Path) -> dict:
    if metrics.get("status") != "complete" or manifest.get("status") != "complete":
        raise RuntimeError("scientific_output_not_complete")
    if manifest.get("research_commit") != science_v1.RESEARCH_COMMIT:
        raise RuntimeError("research_commit_mismatch")
    if manifest.get("training_protocol_changed_from_positive_control_v2") is not False:
        raise RuntimeError("training_protocol_changed")
    frozen_guards = {
        "features_sealed_before_label_reveal": True,
        "candidate_selection_used": False,
        "raw_tail_and_bos_tail_target_tokens_identical": True,
        "probe_boundary_used": False,
        "probe_text_used": False,
        "competition_rows_used": 0,
        "external_rows_used": 0,
        "pretrained_weights_used": False,
        "automatic_compute_retries": 0,
        "stage2_v3_selection_allowed": False,
        "competition_feature_promotion_allowed": False,
    }
    for key, expected in frozen_guards.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"manifest_guard_mismatch:{key}")
    if metrics.get("raw_tail_and_bos_tail_target_tokens_identical") is not True:
        raise RuntimeError("metrics_target_identity_mismatch")

    condition_metrics = metrics.get("condition_metrics") or {}
    validate_condition_metrics(condition_metrics)

    expected_decision = {
        "all_mean_recovery": metrics.get("all_mean_recovery"),
        "downstream_context_material": metrics.get("downstream_context_material"),
        "first_token_material": metrics.get("first_token_material"),
        "named_mechanism": metrics.get("named_mechanism"),
        "mechanism_per_architecture": metrics.get("mechanism_per_architecture"),
        "mechanistic_checks": metrics.get("mechanistic_checks"),
        "raw_tail_and_bos_tail_target_tokens_identical": metrics.get("raw_tail_and_bos_tail_target_tokens_identical"),
    }
    if decision != expected_decision:
        raise RuntimeError("decision_metrics_mismatch")

    rows = 0
    with predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError("prediction_row_contract")
            rows += 1
    if rows != EXPECTED_PREDICTION_ROWS:
        raise RuntimeError("prediction_count_changed")

    return {
        "execution_id": science_v1.EXECUTION_ID,
        "status": "complete",
        "named_mechanism": metrics.get("named_mechanism"),
        "all_mean_recovery": metrics.get("all_mean_recovery"),
        "downstream_context_material": metrics.get("downstream_context_material"),
        "first_token_material": metrics.get("first_token_material"),
        "mechanism_per_architecture": metrics.get("mechanism_per_architecture"),
        "mechanistic_checks": metrics.get("mechanistic_checks"),
        "condition_metrics": {side: {condition: condition_metrics[side][condition] for condition in EXPECTED_CONDITIONS} for side in ("left", "right")},
        "prediction_rows": rows,
    }


def self_test() -> None:
    # The exact persistence pattern that v3 incorrectly rejected: alphabetical
    # JSON key order rather than the semantic experiment order.
    persisted_order = sorted(EXPECTED_CONDITIONS)
    if persisted_order == list(EXPECTED_CONDITIONS):
        raise RuntimeError("self_test_requires_distinct_orders")
    row = {"auc": 0.5, "tpr_at_1pct_fpr": 0.01}
    condition_metrics = {
        side: {condition: dict(row) for condition in persisted_order}
        for side in ("left", "right")
    }
    validate_condition_metrics(condition_metrics)
    if list(condition_metrics["left"]) == list(EXPECTED_CONDITIONS):
        raise RuntimeError("self_test_did_not_exercise_order_bug")
    print("P1_02_RECOVERY_V4_SELF_TEST PASS persisted_order_independent=true conditions=5 architectures=2")


def recover(bridge_root: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError("recovery_output_exists")
    recovery_v3._validate_historical_contract(bridge_root)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    state = recovery_v3.validate_remote_identity(api)
    with tempfile.TemporaryDirectory(prefix="pc-p1-02-recovery-v4-") as tmp:
        root = Path(tmp)
        source_sha256 = recovery_v3.prove_source_identity(bridge_root, root)
        decision, metrics, manifest, predictions = recovery_v3.load_outputs(root)
        second_state = recovery_v3.validate_remote_identity(api)
        if second_state != state:
            raise RuntimeError("remote_state_changed_during_recovery")
        aggregates = validate_science(decision, metrics, manifest, predictions)

        receipt = {
            "request_id": REQUEST_ID,
            "predecessor_request_id": PREDECESSOR_REQUEST_ID,
            "predecessor_run_id": PREDECESSOR_RUN_ID,
            "predecessor_error_sha256": PREDECESSOR_ERROR_SHA256,
            "original_request_id": recovery_v3.ORIGINAL_REQUEST_ID,
            "original_launch_run_id": recovery_v3.ORIGINAL_LAUNCH_RUN_ID,
            "historical_declared_target": recovery_v3.DECLARED_TARGET,
            "historical_title": recovery_v3.EXPECTED_TITLE,
            "canonical_target": TARGET,
            "version": VERSION,
            "remote_status": state,
            "research_commit": recovery_v3.RESEARCH_COMMIT,
            "notebook_code_sha256": source_sha256,
            "root_cause_chain": [
                "historical_kernel_title_id_slug_mismatch",
                "recovery_validator_required_serialized_dict_order",
            ],
            "validator_repair": "condition_key_set_exact_order_independent",
            "new_write_attempted": False,
            "new_compute_requested": False,
            "competition_submission_attempted": False,
            "automatic_compute_retries": 0,
            "scientific_outputs_validated": True,
            "prediction_sha256": science_v1.sha256_file(predictions),
        }
        output_dir.mkdir(parents=True)
        (output_dir / "recovery.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("P1_02_START_BOUNDARY_RECOVERY_V4_VERIFIED " + json.dumps(receipt, sort_keys=True))
        print("P1_02_START_BOUNDARY_AGGREGATES_BEGIN")
        print(json.dumps(aggregates, sort_keys=True))
        print("P1_02_START_BOUNDARY_AGGREGATES_END")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.bridge_root is None or args.output_dir is None:
        raise SystemExit("--bridge-root and --output-dir are required")
    try:
        recover(args.bridge_root.resolve(), args.output_dir.resolve())
    except Exception as exc:
        print(
            "P1_02_START_BOUNDARY_RECOVERY_V4_FAILED "
            + json.dumps(
                {
                    "request_id": REQUEST_ID,
                    "predecessor_run_id": PREDECESSOR_RUN_ID,
                    "canonical_target": TARGET,
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
