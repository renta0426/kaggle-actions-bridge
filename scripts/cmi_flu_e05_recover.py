#!/usr/bin/env python3
"""Read-only recovery of the already-written E05 version 1.

No push, execution, submission, or version mutation is allowed. The downloaded
Notebook source is hashed but never imported or executed. Public logs contain
only aggregate metrics or the runtime's sanitized failure signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from kaggle_exact_identity import exact_metadata, validate_metadata, verify_current, safe_exception

REQUEST = "20260907-cmi-flu-e05-current-recovery-001"
ORIGINAL_REQUEST = "20260907-cmi-flu-strategy-e05-hai-donor-strain-001"
TARGET = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-001"
VERSION = 1
RUNTIME_SHA = "bb295cce54a89cd2b2181629b1c47f0474e044a250c26a3048fc4f11f435162b"
SCIENCE = "e02a601f38250480b526e548a48cf7f2526c00ca"
LIMITS = {"metrics.json": 12582912, "summary.md": 1048576, "bridge-result.json": 1048576}
SAFE_STAGES = {
    "locate_competition_data", "initialize", "materialize_package", "prepare_tree",
    "install_b21_adapter", "install_e02_api_compat", "load_inputs", "load_e05",
    "load_locked_references", "run_e05", "validate_e05", "write_outputs",
}
SIG_RE = re.compile(
    r"CMI_FLU_E05_FAILED stage=([a-z0-9_]{1,64}) "
    r"exception_type=([A-Za-z][A-Za-z0-9_]{0,63}) error_code=([0-9a-f]{20})"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_kaggle_cli() -> str:
    """Resolve the CLI from PATH or the active venv used to execute recovery."""
    cli = shutil.which("kaggle")
    if cli:
        return cli
    adjacent = Path(sys.executable).resolve().with_name("kaggle")
    if adjacent.is_file():
        return str(adjacent)
    raise RuntimeError("kaggle_cli_missing")


def cli_read(verb: str, folder: Path) -> None:
    if verb not in {"pull", "output"}:
        raise ValueError("read_verb_not_allowed")
    cli = resolve_kaggle_cli()
    completed = subprocess.run(
        [cli, "kernels", verb, TARGET, "-p", str(folder)],
        capture_output=True, timeout=180, check=False,
    )
    if completed.returncode:
        info = {
            "stage": verb,
            "return_code": completed.returncode,
            "stdout_bytes": len(completed.stdout),
            "stderr_bytes": len(completed.stderr),
            "diagnostic_sha256": hashlib.sha256(completed.stdout + completed.stderr).hexdigest(),
        }
        print("E05_RECOVERY_CLI " + json.dumps(info, sort_keys=True))
        raise RuntimeError("read_cli_failed")


def project(metrics: dict) -> dict:
    tasks = metrics.get("tasks") or {}
    if set(tasks) != {"Task2.1", "Task2.2"}:
        raise ValueError("recovery_task_set")
    safe = {"experiment": metrics.get("experiment"), "tasks": {}}
    for task, payload in tasks.items():
        routing = payload.get("routing") or {}
        item = {
            "routing": {
                key: routing.get(key)
                for key in (
                    "routing",
                    "positive_e05_signal",
                    "historical_proxy_study_mean_delta_vs_anchor",
                    "challenge_rank_changed_by_at_least_one_position",
                )
            }
        }
        comparisons = payload.get("comparisons") or {}
        selected = {}
        for name in (
            "ridge_donor_by_strain_interactions_vs_anchor",
            "ridge_donor_by_strain_interactions_vs_ridge_main_effects",
        ):
            if name in comparisons:
                selected[name] = comparisons[name]
        if selected:
            item["comparisons"] = selected
        challenge = payload.get("challenge_rank_agreement")
        if challenge is not None:
            item["challenge_rank_agreement"] = challenge
        safe["tasks"][task] = item
    serialized = json.dumps(safe, sort_keys=True, ensure_ascii=False)
    banned = ('participant_id', 'subject_group', 'row_index', 'oof_predictions', 'challenge_predictions')
    if any(token in serialized for token in banned):
        raise ValueError("recovery_projection_privacy")
    return safe


def recover(runtime: Path, output: Path) -> None:
    if output.exists() or digest(runtime) != RUNTIME_SHA:
        raise ValueError("recovery_local_contract")
    trusted = runpy.run_path(str(runtime), run_name="approved_e05_validator")
    if trusted.get("SCIENCE_COMMIT") != SCIENCE or trusted.get("REQUEST_ID") != ORIGINAL_REQUEST:
        raise ValueError("trusted_runtime_identity")

    from kaggle.api.kaggle_api_extended import KaggleApi
    from cmi_flu_strategy_e05_execute import live_rules

    api = KaggleApi(); api.authenticate(); live_rules(api)
    state = verify_current(api, TARGET, VERSION, allow_failed=True, cpu=True)
    receipt = {
        "request_id": REQUEST,
        "original_request_id": ORIGINAL_REQUEST,
        "version": VERSION,
        "remote_status": state,
        "new_write_attempted": False,
        "new_compute_requested": False,
        "source_sha256": RUNTIME_SHA,
        "science_commit": SCIENCE,
    }

    with tempfile.TemporaryDirectory(prefix="cmi-e05-recovery-") as tmp:
        root = Path(tmp)
        code_dir = root / "code"; code_dir.mkdir()
        cli_read("pull", code_dir)
        code_files = [p for p in code_dir.rglob("*") if p.is_file()]
        if len(code_files) != 1 or any(p.is_symlink() for p in code_dir.rglob("*")):
            raise RuntimeError("current_code_file_contract")
        if code_files[0].stat().st_size > 4_194_304 or digest(code_files[0]) != RUNTIME_SHA:
            raise RuntimeError("current_code_hash_mismatch")
        validate_metadata(exact_metadata(api, TARGET), TARGET, VERSION, cpu=True)

        download = root / "outputs"; download.mkdir()
        cli_read("output", download)
        validate_metadata(exact_metadata(api, TARGET), TARGET, VERSION, cpu=True)
        paths = list(download.rglob("*"))
        if any(p.is_symlink() for p in paths):
            raise RuntimeError("output_symlink")
        files = [p for p in paths if p.is_file()]
        if sum(p.stat().st_size for p in files) > 33_554_432:
            raise RuntimeError("recovery_download_size")

        logs = [p for p in files if p.suffix == ".log"]
        if len(logs) > 1:
            raise RuntimeError("transport_log_count")
        selected = {}
        for p in files:
            if p in logs:
                continue
            if p.name not in LIMITS or p.name in selected or not 0 < p.stat().st_size <= LIMITS[p.name]:
                raise RuntimeError("unexpected_or_oversize_output")
            selected[p.name] = p

        signatures = []
        if logs:
            text = logs[0].read_text(encoding="utf-8", errors="replace")
            for stage, kind, code in SIG_RE.findall(text):
                if stage in SAFE_STAGES:
                    signatures.append({"stage": stage, "exception_type": kind, "error_code": code})
        receipt["failure_signatures"] = signatures[:2]
        receipt["output_names"] = sorted(selected)
        receipt["transport_log_present"] = bool(logs)
        if logs:
            receipt["transport_log_sha256"] = digest(logs[0])
            receipt["transport_log_bytes"] = logs[0].stat().st_size

        safe = None
        if "metrics.json" in selected:
            metrics = json.loads(selected["metrics.json"].read_text(encoding="utf-8"))
            trusted["validate_result"](metrics)
            safe = project(metrics)
            receipt["metrics_sha256"] = digest(selected["metrics.json"])
            receipt["scientific_metrics_validated"] = True
        else:
            receipt["scientific_metrics_validated"] = False

        output.mkdir(parents=True)
        (output / "recovery.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        if safe is not None:
            (output / "aggregate.json").write_text(json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print("E05_RECOVERY_VERIFIED " + json.dumps(receipt, sort_keys=True))
        if safe is not None:
            print("E05_AGGREGATES_BEGIN\n" + json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\nE05_AGGREGATES_END")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        recover(a.runtime, a.output)
    except Exception as exc:
        print("E05_RECOVERY_FAILED " + json.dumps({"request_id": REQUEST, "new_write_attempted": False, **safe_exception(exc)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
