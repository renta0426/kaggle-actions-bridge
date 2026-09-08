#!/usr/bin/env python3
"""Recover the already-launched P1-02 Gold start-boundary result read-only.

This program is deliberately incapable of creating a Kaggle version, starting
compute, or submitting to a competition. It accepts only the exact current
private kernel version 1, proves that the pulled Notebook source matches the
frozen launcher materialization, downloads the four declared outputs, and
prints aggregate scientific results. Any identity drift fails closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from kaggle_exact_identity import exact_metadata, safe_exception, verify_current

REQUEST_ID = "20260908-poisoned-chalice-p1-02-start-boundary-readonly-recovery-v1-001"
ORIGINAL_REQUEST_ID = "20260906-poisoned-chalice-shadow-gold-start-boundary-attribution-v1-001"
TARGET = "renta0426/shadow-gold-start-boundary-attribution-v1"
VERSION = 1
RESEARCH_COMMIT = "6b6f388627dfcf16ab0140337cabe7e93f4a8891"
EXECUTION_ID = "shadow-gold-start-boundary-attribution-v1"
LAUNCHER = "scripts/poisoned_chalice_shadow_gold_start_boundary_attribution_v1.py"
EXPECTED_OUTPUTS = {
    "start_boundary_decision.json": 1_048_576,
    "start_boundary_metrics.json": 8_388_608,
    "start_boundary_predictions.jsonl": 67_108_864,
    "start_boundary_runtime_manifest.json": 2_097_152,
}
EXPECTED_CONDITIONS = [
    "raw_stage2_loss_max",
    "raw_suffix254_tail_mean",
    "bos_suffix254_all_mean",
    "bos_suffix254_first_logp",
    "bos_suffix254_tail_mean",
]
EXPECTED_PREDICTION_ROWS = 2560


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def kaggle_cli() -> str:
    found = shutil.which("kaggle")
    if found:
        return found
    adjacent = Path(sys.executable).with_name("kaggle")
    if adjacent.is_file():
        return str(adjacent)
    raise RuntimeError("kaggle_cli_missing")


def cli_read(verb: str, target_dir: Path) -> None:
    if verb not in {"pull", "output"}:
        raise ValueError("read_verb_not_allowed")
    completed = subprocess.run(
        [kaggle_cli(), "kernels", verb, TARGET, "-p", str(target_dir)],
        capture_output=True,
        check=False,
        timeout=240,
        env=os.environ.copy(),
    )
    if completed.returncode:
        digest = sha256_bytes(completed.stdout + completed.stderr)
        raise RuntimeError(f"read_cli_failed:{verb}:{completed.returncode}:{digest}")


def notebook_code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells") or []
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    if len(code_cells) != 1:
        raise RuntimeError("notebook_code_cell_contract")
    source = code_cells[0].get("source")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    if isinstance(source, str):
        return source
    raise RuntimeError("notebook_source_contract")


def prove_source_identity(bridge_root: Path, root: Path) -> str:
    expected_dir = root / "expected"
    launcher = bridge_root / LAUNCHER
    completed = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "--materialize",
            "--snapshot-root",
            str(bridge_root),
            "--kernel-dir",
            str(expected_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode:
        digest = sha256_bytes((completed.stdout + completed.stderr).encode("utf-8", errors="replace"))
        raise RuntimeError(f"materialization_failed:{digest}")
    expected_files = list(expected_dir.glob("*.ipynb"))
    if len(expected_files) != 1:
        raise RuntimeError("expected_notebook_contract")

    pulled_dir = root / "pulled"
    pulled_dir.mkdir()
    cli_read("pull", pulled_dir)
    pulled_files = list(pulled_dir.rglob("*.ipynb"))
    if len(pulled_files) != 1:
        raise RuntimeError("pulled_notebook_contract")

    expected_code = notebook_code(expected_files[0])
    pulled_code = notebook_code(pulled_files[0])
    expected_digest = sha256_bytes(expected_code.encode("utf-8"))
    observed_digest = sha256_bytes(pulled_code.encode("utf-8"))
    if observed_digest != expected_digest:
        raise RuntimeError("pulled_notebook_source_mismatch")
    return expected_digest


def validate_remote_identity(api) -> str:
    state = verify_current(api, TARGET, VERSION, allow_failed=True, cpu=False)
    metadata = exact_metadata(api, TARGET)
    if getattr(metadata, "enable_gpu", None) is not True:
        raise RuntimeError("gpu_contract_not_proven")
    if getattr(metadata, "enable_tpu", None) is not False:
        raise RuntimeError("tpu_contract_not_proven")
    if getattr(metadata, "enable_internet", None) is not False:
        raise RuntimeError("offline_contract_not_proven")
    return state


def load_outputs(root: Path) -> tuple[dict, dict, dict, Path]:
    downloaded = root / "outputs"
    downloaded.mkdir()
    cli_read("output", downloaded)
    paths = list(downloaded.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("output_symlink")
    files = [path for path in paths if path.is_file()]
    selected: dict[str, Path] = {}
    for path in files:
        if path.suffix == ".log":
            continue
        if path.name not in EXPECTED_OUTPUTS or path.name in selected:
            raise RuntimeError("unexpected_output")
        if not 0 < path.stat().st_size <= EXPECTED_OUTPUTS[path.name]:
            raise RuntimeError("output_size_contract")
        selected[path.name] = path
    if set(selected) != set(EXPECTED_OUTPUTS):
        raise RuntimeError("declared_output_set_incomplete")

    decision = json.loads(selected["start_boundary_decision.json"].read_text(encoding="utf-8"))
    metrics = json.loads(selected["start_boundary_metrics.json"].read_text(encoding="utf-8"))
    manifest = json.loads(selected["start_boundary_runtime_manifest.json"].read_text(encoding="utf-8"))
    predictions = selected["start_boundary_predictions.jsonl"]
    return decision, metrics, manifest, predictions


def validate_science(decision: dict, metrics: dict, manifest: dict, predictions: Path) -> dict:
    if metrics.get("status") != "complete" or manifest.get("status") != "complete":
        raise RuntimeError("scientific_output_not_complete")
    if manifest.get("research_commit") != RESEARCH_COMMIT:
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
    if set(condition_metrics) != {"left", "right"}:
        raise RuntimeError("architecture_key_contract")
    for side in ("left", "right"):
        if list(condition_metrics[side]) != EXPECTED_CONDITIONS:
            raise RuntimeError(f"condition_order_contract:{side}")
        for condition in EXPECTED_CONDITIONS:
            row = condition_metrics[side][condition]
            for key in ("auc", "tpr_at_1pct_fpr"):
                value = row.get(key)
                if not isinstance(value, (int, float)):
                    raise RuntimeError(f"condition_metric_missing:{side}:{condition}:{key}")

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
        "execution_id": EXECUTION_ID,
        "status": "complete",
        "named_mechanism": metrics.get("named_mechanism"),
        "all_mean_recovery": metrics.get("all_mean_recovery"),
        "downstream_context_material": metrics.get("downstream_context_material"),
        "first_token_material": metrics.get("first_token_material"),
        "mechanism_per_architecture": metrics.get("mechanism_per_architecture"),
        "mechanistic_checks": metrics.get("mechanistic_checks"),
        "condition_metrics": condition_metrics,
        "prediction_rows": rows,
    }


def recover(bridge_root: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError("recovery_output_exists")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    state = validate_remote_identity(api)
    with tempfile.TemporaryDirectory(prefix="pc-p1-02-recovery-") as tmp:
        root = Path(tmp)
        source_sha256 = prove_source_identity(bridge_root, root)
        decision, metrics, manifest, predictions = load_outputs(root)
        # Re-check exact version after reads so a concurrent mutation cannot pass.
        second_state = validate_remote_identity(api)
        if second_state != state:
            raise RuntimeError("remote_state_changed_during_recovery")
        aggregates = validate_science(decision, metrics, manifest, predictions)
        receipt = {
            "request_id": REQUEST_ID,
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
            "scientific_outputs_validated": True,
            "prediction_sha256": sha256_file(predictions),
        }
        output_dir.mkdir(parents=True)
        (output_dir / "recovery.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("P1_02_START_BOUNDARY_RECOVERY_VERIFIED " + json.dumps(receipt, sort_keys=True))
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
            "P1_02_START_BOUNDARY_RECOVERY_FAILED "
            + json.dumps(
                {
                    "request_id": REQUEST_ID,
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
