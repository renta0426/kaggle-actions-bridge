#!/usr/bin/env python3
"""E11b no-write repair: retarget reviewed runtime to request/target 002 only."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PRIOR = Path(__file__).with_name("cmi_flu_strategy_e11b_prepare_v2.py")
PRIOR_BLOB = "ca6f7a8b121a424228f92ebfc7d69f198afec875"
OLD_REQUEST = "20260910-cmi-flu-strategy-e11b-tabpfn3-001"
OLD_TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-001"
NEW_REQUEST = "20260910-cmi-flu-strategy-e11b-tabpfn3-002"
NEW_TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-002"
REQUEST_PATH = "requests/cmi-flu-strategy-e11b-tabpfn3-002.json"
PARENT_REQUEST_PATH = "requests/cmi-flu-strategy-e11b-tabpfn3-001.json"


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
        raise SystemExit("E11b repair request identity mismatch")
    if current.get("target") != NEW_TARGET:
        raise SystemExit("E11b repair target mismatch")
    for key in (
        "competition", "science_repository", "science_source_commit", "science_transport",
        "strategy_e11b_blob_sha", "strategy_e11b_synthetic_blob_sha", "dependency_blobs",
        "expected_kernel_version", "enable_internet", "resource", "api_budget", "external_model",
        "experiment_contract", "side_effects", "allowed_output_paths", "automatic_compute_retries",
        "competition_submission_attempted", "leaderboard_used_for_selection", "publish_only_sanitized_aggregate",
    ):
        if current.get(key) != parent.get(key):
            raise SystemExit(f"E11b repair scientific/runtime contract drift:{key}")
    preflight = current.get("model_access_preflight") or {}
    parent_preflight = parent.get("model_access_preflight") or {}
    for key in ("required_before_kernel_write", "official_kaggle_model_source", "failure_classification", "write_on_failure"):
        if preflight.get(key) != parent_preflight.get(key):
            raise SystemExit(f"E11b repair preflight contract drift:{key}")
    if preflight.get("cli_output_format") != "--format json":
        raise SystemExit("E11b repair CLI output format mismatch")
    provenance = current.get("repair_provenance") or {}
    expected = {
        "failed_request_id": OLD_REQUEST,
        "failed_target": OLD_TARGET,
        "failed_actions_run": 34428419698,
        "failed_actions_job": 102718671668,
        "failure_stage": "model_access_preflight",
        "failure_return_code": 2,
        "failure_stderr_sha256": "84a7c134e2e40b3e983060472fc11d524bb82523408563cad44313780a0fc3ed",
        "kernel_write_attempted": False,
        "kaggle_compute_consumed": False,
        "root_cause": "Kaggle CLI 2.2.4 model-version file listing supports --format json, not --json",
        "repair_scope": "bridge_model_access_preflight_cli_output_flag_only",
    }
    if provenance != expected:
        raise SystemExit("E11b repair provenance mismatch")


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    data = PRIOR.read_bytes()
    if git_blob_sha(data) != PRIOR_BLOB:
        raise SystemExit("E11b prepare-v3 ancestry changed")
    validate_request(root)
    with tempfile.TemporaryDirectory(prefix="cmi-flu-e11b-v3-") as tmp:
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
        raise SystemExit("E11b repair identity anchors missing")
    runtime = runtime.replace(OLD_REQUEST, NEW_REQUEST).replace(OLD_TARGET, NEW_TARGET)
    if OLD_REQUEST in runtime or OLD_TARGET in runtime:
        raise SystemExit("E11b repair old identity remained")
    compile(runtime, "generated_e11b_repair_runtime.py", "exec")
    out = args.output.resolve(); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(runtime)
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E11B_PREPARE_V3_PASS repair=model_access_cli_format_only "
        f"request_id={NEW_REQUEST} target={NEW_TARGET} science_change=false runtime_sha256={hashlib.sha256(runtime.encode()).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
