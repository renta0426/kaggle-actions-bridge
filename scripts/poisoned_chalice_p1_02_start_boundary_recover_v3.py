#!/usr/bin/env python3
"""Recover the already-created P1-02 Gold run from its canonical Kaggle slug.

The historical launcher declared the id
``renta0426/shadow-gold-start-boundary-attribution-v1`` while its frozen title
was ``Gold start-boundary attribution v1``. Kaggle kernel titles and slugs are
linked; the canonical slug for that title is ``gold-start-boundary-attribution-v1``.
The failed v1/v2 recoveries therefore queried the wrong exact ref.

This successor changes no scientific input and performs no Kaggle write,
run-start, submission, recompute, or version substitution. It accepts only the
canonical title-derived target at current version 1, proves the pulled source is
identical to the frozen original launcher materialization, downloads only the
frozen output allowlist, and validates the existing scientific results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import poisoned_chalice_p1_02_start_boundary_recover_v1 as base
from kaggle_exact_identity import TERMINAL, exact_metadata, safe_exception, status_name, validate_metadata

REQUEST_ID = "20260912-poisoned-chalice-p1-02-start-boundary-canonical-slug-recovery-v3-001"
PREDECESSOR_REQUEST_ID = "20260912-poisoned-chalice-p1-02-start-boundary-readonly-recovery-v2-001"
PREDECESSOR_RUN_ID = 34619164085
ORIGINAL_REQUEST_ID = "20260906-poisoned-chalice-shadow-gold-start-boundary-attribution-v1-001"
ORIGINAL_LAUNCH_RUN_ID = 34038595367
ORIGINAL_LAUNCH_COMMIT = "0448206e5b4650204e3ec28d10c472bd376d1a0c"
ORIGINAL_LAUNCHER_BLOB_SHA = "16cf5028a6d09f2aed4c590b69c42c248365a265"
DECLARED_TARGET = "renta0426/shadow-gold-start-boundary-attribution-v1"
EXPECTED_TITLE = "Gold start-boundary attribution v1"
TARGET = "renta0426/gold-start-boundary-attribution-v1"
VERSION = 1
RESEARCH_COMMIT = "6b6f388627dfcf16ab0140337cabe7e93f4a8891"
EXECUTION_ID = "shadow-gold-start-boundary-attribution-v1"
EXPECTED_PREDICTION_ROWS = 2560
EXPECTED_OUTPUTS = dict(base.EXPECTED_OUTPUTS)


def canonical_slug(title: str) -> str:
    """Canonicalize this frozen ASCII title using Kaggle's title/slug rule."""
    if title != EXPECTED_TITLE:
        raise ValueError("unexpected_historical_title")
    return re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _validate_historical_contract(bridge_root: Path) -> None:
    """Prove the root cause from the frozen launcher materialization."""
    if base.TARGET != DECLARED_TARGET:
        raise RuntimeError("historical_declared_target_drift")
    if base.VERSION != VERSION or base.RESEARCH_COMMIT != RESEARCH_COMMIT or base.EXECUTION_ID != EXECUTION_ID:
        raise RuntimeError("historical_science_contract_drift")
    if base.EXPECTED_PREDICTION_ROWS != EXPECTED_PREDICTION_ROWS:
        raise RuntimeError("historical_prediction_contract_drift")
    if set(EXPECTED_OUTPUTS) != {
        "start_boundary_decision.json",
        "start_boundary_metrics.json",
        "start_boundary_predictions.jsonl",
        "start_boundary_runtime_manifest.json",
    }:
        raise RuntimeError("historical_output_contract_drift")

    launcher = bridge_root / base.LAUNCHER
    if not launcher.is_file():
        raise RuntimeError("historical_launcher_missing")
    if git_blob_sha(launcher) != ORIGINAL_LAUNCHER_BLOB_SHA:
        raise RuntimeError("historical_launcher_blob_mismatch")

    with tempfile.TemporaryDirectory(prefix="pc-p1-02-v3-history-") as tmp:
        kernel_dir = Path(tmp) / "kernel"
        completed = subprocess.run(
            [
                sys.executable,
                str(launcher),
                "--materialize",
                "--snapshot-root",
                str(bridge_root),
                "--kernel-dir",
                str(kernel_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if completed.returncode:
            raise RuntimeError("historical_materialization_failed")
        metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        if metadata.get("id") != DECLARED_TARGET:
            raise RuntimeError("historical_metadata_id_drift")
        if metadata.get("title") != EXPECTED_TITLE:
            raise RuntimeError("historical_metadata_title_drift")
        canonical = "renta0426/" + canonical_slug(EXPECTED_TITLE)
        if canonical != TARGET:
            raise RuntimeError("canonical_target_derivation_drift")
        if metadata["id"] == canonical:
            raise RuntimeError("historical_title_id_mismatch_not_present")


def _run_read(command: list[str], stage: str, timeout: int = 240) -> None:
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        timeout=timeout,
        env=os.environ.copy(),
    )
    if completed.returncode:
        digest = base.sha256_bytes(completed.stdout + completed.stderr)
        raise RuntimeError(f"read_cli_failed:{stage}:{completed.returncode}:{digest}")


def validate_remote_identity(api: Any) -> str:
    print("P1_02_RECOVERY_V3_READ_STAGE exact_get_kernel_canonical_slug")
    metadata = exact_metadata(api, TARGET)
    validate_metadata(metadata, TARGET, VERSION, cpu=False)
    if getattr(metadata, "title", None) != EXPECTED_TITLE:
        raise RuntimeError("canonical_target_title_mismatch")
    if getattr(metadata, "enable_gpu", None) is not True:
        raise RuntimeError("gpu_contract_not_proven")
    if getattr(metadata, "enable_tpu", None) is not False:
        raise RuntimeError("tpu_contract_not_proven")
    if getattr(metadata, "enable_internet", None) is not False:
        raise RuntimeError("offline_contract_not_proven")

    print("P1_02_RECOVERY_V3_READ_STAGE kernel_status_canonical_slug")
    state = status_name(getattr(api.kernels_status(TARGET), "status", None))
    if state not in TERMINAL:
        raise RuntimeError("kernel_not_in_allowed_terminal_state")
    return state


def prove_source_identity(bridge_root: Path, root: Path) -> str:
    expected_dir = root / "expected"
    launcher = bridge_root / base.LAUNCHER
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
        raise RuntimeError("expected_materialization_failed")
    expected_files = list(expected_dir.glob("*.ipynb"))
    if len(expected_files) != 1:
        raise RuntimeError("expected_notebook_contract")

    pulled_dir = root / "pulled"
    pulled_dir.mkdir()
    _run_read([base.kaggle_cli(), "kernels", "pull", TARGET, "-p", str(pulled_dir), "-m"], "pull")
    paths = list(pulled_dir.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("pulled_source_symlink")
    pulled_files = list(pulled_dir.rglob("*.ipynb"))
    if len(pulled_files) != 1:
        raise RuntimeError("pulled_notebook_contract")
    metadata_path = pulled_dir / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("pulled_metadata_missing")
    pulled_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if pulled_metadata.get("id") != TARGET or pulled_metadata.get("title") != EXPECTED_TITLE:
        raise RuntimeError("pulled_canonical_identity_mismatch")

    expected_code = base.notebook_code(expected_files[0])
    pulled_code = base.notebook_code(pulled_files[0])
    expected_digest = base.sha256_bytes(expected_code.encode("utf-8"))
    observed_digest = base.sha256_bytes(pulled_code.encode("utf-8"))
    if observed_digest != expected_digest:
        raise RuntimeError("pulled_notebook_source_mismatch")
    return expected_digest


def load_outputs(root: Path) -> tuple[dict, dict, dict, Path]:
    downloaded = root / "outputs"
    downloaded.mkdir()
    _run_read([base.kaggle_cli(), "kernels", "output", TARGET, "-p", str(downloaded)], "output")
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


def recover(bridge_root: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError("recovery_output_exists")
    _validate_historical_contract(bridge_root)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    state = validate_remote_identity(api)
    with tempfile.TemporaryDirectory(prefix="pc-p1-02-recovery-v3-") as tmp:
        root = Path(tmp)
        source_sha256 = prove_source_identity(bridge_root, root)
        decision, metrics, manifest, predictions = load_outputs(root)

        second_state = validate_remote_identity(api)
        if second_state != state:
            raise RuntimeError("remote_state_changed_during_recovery")
        aggregates = base.validate_science(decision, metrics, manifest, predictions)

        receipt = {
            "request_id": REQUEST_ID,
            "predecessor_request_id": PREDECESSOR_REQUEST_ID,
            "predecessor_run_id": PREDECESSOR_RUN_ID,
            "original_request_id": ORIGINAL_REQUEST_ID,
            "original_launch_run_id": ORIGINAL_LAUNCH_RUN_ID,
            "historical_declared_target": DECLARED_TARGET,
            "historical_title": EXPECTED_TITLE,
            "canonical_target": TARGET,
            "version": VERSION,
            "remote_status": state,
            "research_commit": RESEARCH_COMMIT,
            "notebook_code_sha256": source_sha256,
            "root_cause": "historical_kernel_title_id_slug_mismatch",
            "new_write_attempted": False,
            "new_compute_requested": False,
            "competition_submission_attempted": False,
            "automatic_compute_retries": 0,
            "scientific_outputs_validated": True,
            "prediction_sha256": base.sha256_file(predictions),
        }
        output_dir.mkdir(parents=True)
        (output_dir / "recovery.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("P1_02_START_BOUNDARY_RECOVERY_V3_VERIFIED " + json.dumps(receipt, sort_keys=True))
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
            "P1_02_START_BOUNDARY_RECOVERY_V3_FAILED "
            + json.dumps(
                {
                    "request_id": REQUEST_ID,
                    "predecessor_run_id": PREDECESSOR_RUN_ID,
                    "historical_declared_target": DECLARED_TARGET,
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
