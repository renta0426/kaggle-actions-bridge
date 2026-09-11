#!/usr/bin/env python3
"""Credential-free regression for V3 Task1.2 data-limited audit repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REQUEST = "requests/cmi-flu-v3-batch1-task12-audit-repair-003.json"
PREPARE = "scripts/cmi_flu_v3_batch1_task12_audit_repair_prepare.py"
EXECUTOR = "scripts/cmi_flu_v3_batch1_task12_audit_repair_execute.py"
SANITIZER = "scripts/cmi_flu_v3_batch1_sanitize.py"
STUB_DIR = "scripts/v3_runtime_smoke_stubs"
REPAIR_PAYLOAD = "payloads/cmi-flu-v3-batch1-task12-audit-repair-003/strategy_v3_batch1_v2.py"
REPAIR_BLOB = "6ed220226c7dda80391f1c4700e6acf07d2d9fee"
BASE_V3_PAYLOAD = "payloads/cmi-flu-v3-batch1-001/strategy_v3_batch1.py"
BASE_V3_BLOB = "cedd8e2538a06c8b696e74631f0bb2fce427984b"
REQUEST_ID = "20260912-cmi-flu-strategy-v3-batch1-task12-audit-repair-003"
TARGET = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260912-003"
TITLE = "cmi-flu-v3-batch1-audit-diagnostics-20260912-003"
SCIENCE_COMMIT = "06d29b1c2317b9bfb66b5d4364213d3ca6de3d7b"
PARENT_FAILURE_CODE = "ebb7c658334065658611"
PARENT_FAILURE_MESSAGE = "Task1.2 dataset has an empty partition"
SOURCE_B_SHA = "0f9df53c3aa8c6e4ac693f6a42dbd2633b4b61d1462df798a9bd767c220be3a5"
SOURCE_C_SHA = "983aaf097d04477c4ccf7bf817fdf66e552937e69bceaa48cfb260cb84413f1b"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    work = args.workdir.resolve(); work.mkdir(parents=True, exist_ok=True)

    if git_blob((root / REPAIR_PAYLOAD).read_bytes()) != REPAIR_BLOB:
        raise SystemExit("V3 Task1.2 repair exact science relay mismatch")
    if git_blob((root / BASE_V3_PAYLOAD).read_bytes()) != BASE_V3_BLOB:
        raise SystemExit("V3 Task1.2 repair base V3 relay mismatch")
    if not (root / STUB_DIR / "sitecustomize.py").is_file():
        raise SystemExit("V3 Task1.2 repair CI external stub bootstrap missing")
    for rel in (PREPARE, EXECUTOR, SANITIZER, f"{STUB_DIR}/sitecustomize.py", REPAIR_PAYLOAD):
        subprocess.run([sys.executable, "-W", "error::SyntaxWarning", "-m", "py_compile", str(root / rel)], check=True)

    request = json.loads((root / REQUEST).read_text(encoding="utf-8"))
    if request.get("request_id") != REQUEST_ID or request.get("target") != TARGET or request.get("title") != TITLE:
        raise SystemExit("V3 Task1.2 repair request identity mismatch")
    if TITLE != TARGET.split("/", 1)[1]:
        raise SystemExit("V3 Task1.2 repair title-target slug mismatch")
    if request.get("science_source_commit") != SCIENCE_COMMIT or request.get("strategy_v3_batch1_v2_blob_sha") != REPAIR_BLOB:
        raise SystemExit("V3 Task1.2 repair science provenance mismatch")
    if request.get("automatic_compute_retries") != 0 or request.get("competition_submission_authorized") is not False:
        raise SystemExit("V3 Task1.2 repair retry/submission boundary changed")
    parent = request.get("parent_runtime_failure") or {}
    if parent.get("target") != "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260912-002" or parent.get("version") != 1:
        raise SystemExit("V3 Task1.2 repair consumed parent identity changed")
    if parent.get("kernel_terminal_state") != "ERROR" or parent.get("same_target_version_rerun_forbidden") is not True:
        raise SystemExit("V3 Task1.2 repair parent version-consumption boundary changed")
    if parent.get("failure_code") != PARENT_FAILURE_CODE or parent.get("resolved_message") != PARENT_FAILURE_MESSAGE:
        raise SystemExit("V3 Task1.2 repair parent root cause changed")
    observed = hashlib.sha256(
        f"run_no_fit_audit:DataContractError:{PARENT_FAILURE_MESSAGE}".encode()
    ).hexdigest()[:20]
    if observed != PARENT_FAILURE_CODE:
        raise SystemExit("V3 Task1.2 repair failure fingerprint proof mismatch")
    repair = request.get("science_repair_contract") or {}
    expected_counts = {"SDY296": 36, "SDY301": 40, "SDY416": 5}
    if repair.get("Task1.2_target_by_study_expected") != expected_counts:
        raise SystemExit("V3 Task1.2 repair fixed target expectations changed")
    if repair.get("reference_expectations_mutated") is not False or repair.get("prediction_logic_changed") is not False:
        raise SystemExit("V3 Task1.2 repair scientific boundary changed")
    if (request.get("source_B") or {}).get("sha256") != SOURCE_B_SHA or (request.get("forbidden_source_C") or {}).get("sha256") != SOURCE_C_SHA:
        raise SystemExit("V3 Task1.2 repair source B/C identity changed")

    science = (root / REPAIR_PAYLOAD).read_text(encoding="utf-8")
    if ".fit(" in science or "kaggle competitions submit" in science.casefold():
        raise SystemExit("V3 Task1.2 repair science introduced fit/submit path")
    for token in (
        PARENT_FAILURE_CODE,
        PARENT_FAILURE_MESSAGE,
        '"SDY296": 36',
        '"SDY301": 40',
        '"SDY416": 5',
        '"data_limited" if baseline_join_rows == 0 else "observed"',
        '"unresolved_not_compatible"',
        '"reference_expectations_mutated": False',
    ):
        if token not in science:
            raise SystemExit(f"V3 Task1.2 repair science contract token missing:{token}")

    runtime = work / "runtime.py"
    subprocess.run([sys.executable, str(root / PREPARE), "--repository-root", str(root), "--output", str(runtime)], check=True)
    smoke_env = dict(os.environ)
    stub_path = str((root / STUB_DIR).resolve())
    smoke_env["PYTHONPATH"] = stub_path + (os.pathsep + smoke_env["PYTHONPATH"] if smoke_env.get("PYTHONPATH") else "")
    subprocess.run([sys.executable, str(runtime), "--self-test"], check=True, env=smoke_env)
    text = runtime.read_text(encoding="utf-8")
    for token in (
        REQUEST_ID, TARGET, BASE_V3_BLOB, REPAIR_BLOB, SCIENCE_COMMIT, SOURCE_B_SHA,
        "CMI_FLU_V3_BATCH1_TASK12_AUDIT_REPAIR_SELF_TEST PASS",
        "parent_failure_proven=true", "data_limited_semantics=true",
        "model_fit=false", "competition_submit=false",
    ):
        if token not in text:
            raise SystemExit(f"V3 Task1.2 repair generated runtime token missing:{token}")
    if len(runtime.read_bytes()) >= 1_750_000:
        raise SystemExit("V3 Task1.2 repair runtime byte budget exceeded")

    print(
        "CMI_FLU_V3_BATCH1_TASK12_AUDIT_REPAIR_CI PASS exact_science=true parent_failure_proven=true "
        "data_limited_semantics=true fixed_expectations=true fresh_target=true auth=false write=false compute=false "
        "model_fit=false competition_submit=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
