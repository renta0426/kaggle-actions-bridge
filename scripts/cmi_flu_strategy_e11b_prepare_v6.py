#!/usr/bin/env python3
"""E11b 005 repair: retarget the frozen 004 runtime; science/dependency transport unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PRIOR = Path(__file__).with_name("cmi_flu_strategy_e11b_prepare_v5.py")
PRIOR_BLOB = "2eec8e7237d4c8cd49ad1f3397aae1b7b29735f5"
OLD_REQUEST = "20260910-cmi-flu-strategy-e11b-tabpfn3-004"
OLD_TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-004"
NEW_REQUEST = "20260910-cmi-flu-strategy-e11b-tabpfn3-005"
NEW_TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-005"
REQUEST_PATH = "requests/cmi-flu-strategy-e11b-tabpfn3-005.json"
PARENT_REQUEST_PATH = "requests/cmi-flu-strategy-e11b-tabpfn3-004.json"
MAX_RUNTIME_BYTES = 900_000


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--tabpfn-wheel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def validate_request(root: Path) -> None:
    current = json.loads((root / REQUEST_PATH).read_text())
    parent = json.loads((root / PARENT_REQUEST_PATH).read_text())
    if current.get("request_id") != NEW_REQUEST or current.get("parent_request_id") != OLD_REQUEST:
        raise SystemExit("E11b 005 request identity mismatch")
    if current.get("target") != NEW_TARGET:
        raise SystemExit("E11b 005 target mismatch")
    for key in (
        "competition", "science_repository", "science_source_commit", "science_transport",
        "strategy_e11b_blob_sha", "strategy_e11b_synthetic_blob_sha", "dependency_blobs",
        "expected_kernel_version", "enable_internet", "resource", "api_budget",
        "model_access_preflight", "external_model", "experiment_contract", "dependency_transport",
        "allowed_output_paths", "automatic_compute_retries", "competition_submission_attempted",
        "leaderboard_used_for_selection", "publish_only_sanitized_aggregate",
    ):
        if current.get(key) != parent.get(key):
            raise SystemExit(f"E11b 005 frozen-contract drift:{key}")
    provenance = current.get("repair_provenance") or {}
    expected = {
        "failed_request_id": OLD_REQUEST,
        "failed_target": OLD_TARGET,
        "failed_actions_run": 34444519009,
        "failed_actions_job": 102766461385,
        "failure_stage": "prewrite_exact_absence_guard",
        "failure_api_operation": "GetKernel",
        "failure_http_status": 500,
        "model_access_succeeded": True,
        "kernel_write_attempted": False,
        "kernel_version_consumed": False,
        "kaggle_compute_started": False,
        "root_cause": "Kaggle GetKernel returned HTTP 500 while checking a nonexistent exact target; the fail-closed guard therefore stopped before SaveKernel",
        "repair_scope": "prewrite_guard_only_kernels_list_exact_ref_match",
    }
    if provenance != expected:
        raise SystemExit("E11b 005 repair provenance mismatch")
    transport = current.get("transport") or {}
    if transport != {
        "kernel_write_method": "KaggleApi.kernels_push",
        "cli_kernels_push_used": False,
        "structured_failure_receipt_required": True,
        "raw_server_error_persisted": False,
        "prewrite_absence_method": "kernels_list_user_search_exact_ref_match",
        "confirm_failed_003_target_absent_before_write": True,
        "confirm_failed_004_target_absent_before_write": True,
    }:
        raise SystemExit("E11b 005 transport mismatch")


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    data = PRIOR.read_bytes()
    if git_blob_sha(data) != PRIOR_BLOB:
        raise SystemExit("E11b prepare-v6 ancestry changed")
    validate_request(root)
    with tempfile.TemporaryDirectory(prefix="cmi-flu-e11b-v6-") as tmp:
        prior_runtime = Path(tmp) / "runtime.py"
        subprocess.run(
            [sys.executable, str(PRIOR), "--repository-root", str(root),
             "--reference-dir", str(args.reference_dir.resolve()),
             "--tabpfn-wheel", str(args.tabpfn_wheel.resolve()),
             "--output", str(prior_runtime)],
            check=True,
        )
        runtime = prior_runtime.read_text()
    if runtime.count(OLD_REQUEST) < 1 or runtime.count(OLD_TARGET) < 1:
        raise SystemExit("E11b 005 runtime identity anchors missing")
    runtime = runtime.replace(OLD_REQUEST, NEW_REQUEST).replace(OLD_TARGET, NEW_TARGET)
    if OLD_REQUEST in runtime or OLD_TARGET in runtime:
        raise SystemExit("E11b 005 old identity remained")
    raw = runtime.encode("utf-8")
    if len(raw) >= MAX_RUNTIME_BYTES:
        raise SystemExit(f"E11b 005 runtime exceeds source budget:{len(raw)}")
    compile(runtime, "generated_e11b_005_runtime.py", "exec")
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime)
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E11B_PREPARE_V6_PASS repair=prewrite_list_guard_only "
        f"request_id={NEW_REQUEST} target={NEW_TARGET} science_change=false dependency_change=false "
        f"runtime_bytes={len(raw)} runtime_sha256={hashlib.sha256(raw).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
