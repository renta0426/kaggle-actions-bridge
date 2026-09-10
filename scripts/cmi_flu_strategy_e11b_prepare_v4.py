#!/usr/bin/env python3
"""E11b transport repair: retarget the reviewed 002 runtime to request/target 003."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PRIOR = Path(__file__).with_name("cmi_flu_strategy_e11b_prepare_v3.py")
PRIOR_BLOB = "c01c75052efcba1035d4b4f6b628a13cbb76840e"
OLD_REQUEST = "20260910-cmi-flu-strategy-e11b-tabpfn3-002"
OLD_TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-002"
NEW_REQUEST = "20260910-cmi-flu-strategy-e11b-tabpfn3-003"
NEW_TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-003"
REQUEST_PATH = "requests/cmi-flu-strategy-e11b-tabpfn3-003.json"
PARENT_REQUEST_PATH = "requests/cmi-flu-strategy-e11b-tabpfn3-002.json"


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
        raise SystemExit("E11b 003 request identity mismatch")
    if current.get("target") != NEW_TARGET:
        raise SystemExit("E11b 003 target mismatch")
    for key in (
        "competition", "science_repository", "science_source_commit", "science_transport",
        "strategy_e11b_blob_sha", "strategy_e11b_synthetic_blob_sha", "dependency_blobs",
        "expected_kernel_version", "enable_internet", "resource", "api_budget", "external_model",
        "experiment_contract", "allowed_output_paths", "automatic_compute_retries",
        "competition_submission_attempted", "leaderboard_used_for_selection", "publish_only_sanitized_aggregate",
    ):
        if current.get(key) != parent.get(key):
            raise SystemExit(f"E11b 003 science/runtime drift:{key}")
    if current.get("model_access_preflight") != parent.get("model_access_preflight"):
        raise SystemExit("E11b 003 model preflight drift")
    provenance = current.get("repair_provenance") or {}
    expected = {
        "failed_request_id": OLD_REQUEST,
        "failed_target": OLD_TARGET,
        "failed_actions_run": 34430166816,
        "failed_actions_job": 102733973098,
        "failure_stage": "kernel_push_transport",
        "failure_return_code": 1,
        "failure_stderr_bytes": 102,
        "failure_stderr_sha256": "ef812bc9f178b83f556046dfa3dc61b4d05a90e4003989ea212e7552cf7b8f2c",
        "model_access_succeeded": True,
        "target_created": False,
        "kernel_version_consumed": False,
        "kaggle_compute_consumed": False,
        "diagnostic_actions_run": 34434976964,
        "diagnostic_actions_job": 102738087104,
        "repair_scope": "bridge_transport_observability_direct_kaggleapi_kernels_push_only",
    }
    if provenance != expected:
        raise SystemExit("E11b 003 repair provenance mismatch")
    transport = current.get("transport") or {}
    if transport != {
        "kernel_write_method": "KaggleApi.kernels_push",
        "cli_kernels_push_used": False,
        "structured_failure_receipt_required": True,
        "raw_server_error_persisted": False,
    }:
        raise SystemExit("E11b 003 transport contract mismatch")


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    data = PRIOR.read_bytes()
    if git_blob_sha(data) != PRIOR_BLOB:
        raise SystemExit("E11b prepare-v4 ancestry changed")
    validate_request(root)
    with tempfile.TemporaryDirectory(prefix="cmi-flu-e11b-v4-") as tmp:
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
        raise SystemExit("E11b 003 runtime identity anchors missing")
    runtime = runtime.replace(OLD_REQUEST, NEW_REQUEST).replace(OLD_TARGET, NEW_TARGET)
    if OLD_REQUEST in runtime or OLD_TARGET in runtime:
        raise SystemExit("E11b 003 old identity remained")
    compile(runtime, "generated_e11b_003_runtime.py", "exec")
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime)
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E11B_PREPARE_V4_PASS repair=direct_savekernel_transport_observability "
        f"request_id={NEW_REQUEST} target={NEW_TARGET} science_change=false "
        f"runtime_sha256={hashlib.sha256(runtime.encode()).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
