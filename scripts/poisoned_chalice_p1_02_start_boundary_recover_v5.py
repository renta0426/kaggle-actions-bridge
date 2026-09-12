#!/usr/bin/env python3
"""Read-only P1-02 recovery with the frozen producer metric schema.

Recovery v4 reached scientific validation after proving the canonical private
Notebook identity, source identity and frozen four-output download. It failed
because the recovery validator expected ``tpr_at_1pct_fpr`` while the frozen
producer serializes the 1% FPR metric as ``tpr_at_fpr``. The producer calls
``evaluate_frozen_gold_holdout(..., false_positive_rate=0.01)`` and its frozen
metric helper returns exactly ``auc``, ``partial_auc_standardized`` and
``tpr_at_fpr``.

This successor changes only the recovery-side metric-field contract. It cannot
write to Kaggle, start compute, submit, recompute, change target/version, alter
scientific conditions, or alter thresholds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import poisoned_chalice_p1_02_start_boundary_recover_v1 as science_v1
import poisoned_chalice_p1_02_start_boundary_recover_v3 as recovery_v3
import poisoned_chalice_p1_02_start_boundary_recover_v4 as recovery_v4
from kaggle_exact_identity import safe_exception

REQUEST_ID = "20260912-poisoned-chalice-p1-02-start-boundary-metric-schema-recovery-v5-001"
PREDECESSOR_REQUEST_ID = recovery_v4.REQUEST_ID
PREDECESSOR_RUN_ID = 34659847414
PREDECESSOR_ERROR_SHA256 = "af93c27cd8f5ba62cf79b87622524ddeb932313296d2bda87141a4a988ab41c5"
TARGET = recovery_v3.TARGET
VERSION = recovery_v3.VERSION
EXPECTED_CONDITIONS = tuple(science_v1.EXPECTED_CONDITIONS)
EXPECTED_PREDICTION_ROWS = science_v1.EXPECTED_PREDICTION_ROWS
EXPECTED_METRIC_KEYS = frozenset({"auc", "partial_auc_standardized", "tpr_at_fpr"})
ATTRIBUTION_RUNTIME_BLOB = "a7e26a3f256ad792891ff70509a080b046d7226b"
TRANSFER_RUNTIME_BLOB = "1ee969fe852f8a18d00a5c0249f4a8a90d61338a"


def _validate_frozen_producer_schema(bridge_root: Path) -> None:
    attribution = bridge_root / "materialized/poisoned-chalice-shadow-gold-start-boundary-attribution-v1/poisoned_chalice/shadow_gold_start_boundary_attribution.py"
    transfer = bridge_root / "materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_gold_transfer.py"
    if recovery_v3.git_blob_sha(attribution) != ATTRIBUTION_RUNTIME_BLOB:
        raise RuntimeError("attribution_runtime_blob_drift")
    if recovery_v3.git_blob_sha(transfer) != TRANSFER_RUNTIME_BLOB:
        raise RuntimeError("transfer_runtime_blob_drift")
    attribution_text = attribution.read_text(encoding="utf-8")
    transfer_text = transfer.read_text(encoding="utf-8")
    required_attribution = (
        "evaluate_frozen_gold_holdout(",
        "false_positive_rate=0.01",
        '["tpr_at_fpr"]',
    )
    required_transfer = (
        'return {"auc": auc, "partial_auc_standardized": pauc, "tpr_at_fpr": tpr_at}',
        "def _metrics(y: np.ndarray, score: np.ndarray, *, false_positive_rate: float = 0.01)",
    )
    if not all(marker in attribution_text for marker in required_attribution):
        raise RuntimeError("attribution_metric_contract_drift")
    if not all(marker in transfer_text for marker in required_transfer):
        raise RuntimeError("transfer_metric_contract_drift")


def validate_condition_metrics(condition_metrics: dict) -> None:
    if set(condition_metrics) != {"left", "right"}:
        raise RuntimeError("architecture_key_contract")
    expected_conditions = set(EXPECTED_CONDITIONS)
    for side in ("left", "right"):
        rows = condition_metrics[side]
        if not isinstance(rows, dict) or set(rows) != expected_conditions:
            raise RuntimeError(f"condition_key_contract:{side}")
        for condition in EXPECTED_CONDITIONS:
            row = rows[condition]
            if not isinstance(row, dict) or set(row) != EXPECTED_METRIC_KEYS:
                raise RuntimeError(f"condition_metric_schema:{side}:{condition}")
            for key in ("auc", "partial_auc_standardized", "tpr_at_fpr"):
                value = row[key]
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
        "condition_metrics": {
            side: {condition: condition_metrics[side][condition] for condition in EXPECTED_CONDITIONS}
            for side in ("left", "right")
        },
        "prediction_rows": rows,
    }


def self_test() -> None:
    row = {"auc": 0.5, "partial_auc_standardized": 0.5, "tpr_at_fpr": 0.01}
    metrics = {
        side: {condition: dict(row) for condition in sorted(EXPECTED_CONDITIONS)}
        for side in ("left", "right")
    }
    validate_condition_metrics(metrics)
    wrong = json.loads(json.dumps(metrics))
    wrong["left"][EXPECTED_CONDITIONS[0]]["tpr_at_1pct_fpr"] = wrong["left"][EXPECTED_CONDITIONS[0]].pop("tpr_at_fpr")
    try:
        validate_condition_metrics(wrong)
    except RuntimeError as exc:
        if str(exc) != f"condition_metric_schema:left:{EXPECTED_CONDITIONS[0]}":
            raise
    else:
        raise RuntimeError("self_test_failed_to_reject_wrong_metric_name")
    print("P1_02_RECOVERY_V5_SELF_TEST PASS metric_schema=tpr_at_fpr fpr=0.01 order_independent=true")


def recover(bridge_root: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError("recovery_output_exists")
    recovery_v3._validate_historical_contract(bridge_root)
    _validate_frozen_producer_schema(bridge_root)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    state = recovery_v3.validate_remote_identity(api)
    with tempfile.TemporaryDirectory(prefix="pc-p1-02-recovery-v5-") as tmp:
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
                "recovery_validator_used_nonexistent_tpr_metric_field",
            ],
            "validator_repair": "exact_frozen_metric_schema_tpr_at_fpr_at_0p01_fpr",
            "new_write_attempted": False,
            "new_compute_requested": False,
            "competition_submission_attempted": False,
            "automatic_compute_retries": 0,
            "scientific_outputs_validated": True,
            "prediction_sha256": science_v1.sha256_file(predictions),
        }
        output_dir.mkdir(parents=True)
        (output_dir / "recovery.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("P1_02_START_BOUNDARY_RECOVERY_V5_VERIFIED " + json.dumps(receipt, sort_keys=True))
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
            "P1_02_START_BOUNDARY_RECOVERY_V5_FAILED "
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
