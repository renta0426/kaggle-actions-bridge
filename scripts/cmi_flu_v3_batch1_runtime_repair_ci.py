#!/usr/bin/env python3
"""Credential-free regression for the V3 batch1 runtime-closure successor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REQUEST = "requests/cmi-flu-v3-batch1-runtime-repair-002.json"
PREPARE = "scripts/cmi_flu_v3_batch1_runtime_repair_prepare.py"
EXECUTOR = "scripts/cmi_flu_v3_batch1_runtime_repair_execute.py"
SANITIZER = "scripts/cmi_flu_v3_batch1_sanitize.py"
V3_PAYLOAD = "payloads/cmi-flu-v3-batch1-001/strategy_v3_batch1.py"
V3_BLOB = "cedd8e2538a06c8b696e74631f0bb2fce427984b"
DEPENDENCIES = {
    "payloads/cmi-flu-v3-batch1-runtime-repair-002/strategy_e04_contract.py": "3982541febfb4641fbf895438ae1a465bdbb3d5e",
    "payloads/cmi-flu-v3-batch1-runtime-repair-002/strategy_e04b.py": "72f936a7248dfa17338932077d17fb43123cb056",
    "payloads/cmi-flu-v3-batch1-runtime-repair-002/strategy_e09a.py": "37270a0a8aafc070d91b5f43fbea3ba169cd61b0",
}
REQUEST_ID = "20260912-cmi-flu-strategy-v3-batch1-runtime-repair-002"
TARGET = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260912-002"
TITLE = "cmi-flu-v3-batch1-audit-diagnostics-20260912-002"
SOURCE_B_SHA = "0f9df53c3aa8c6e4ac693f6a42dbd2633b4b61d1462df798a9bd767c220be3a5"
SOURCE_C_SHA = "983aaf097d04477c4ccf7bf817fdf66e552937e69bceaa48cfb260cb84413f1b"
FAILURE_CODE = "6e8f8d1003dad8b8feb7"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    work = args.workdir.resolve(); work.mkdir(parents=True, exist_ok=True)

    if git_blob((root / V3_PAYLOAD).read_bytes()) != V3_BLOB:
        raise SystemExit("V3 runtime-repair exact science relay mismatch")
    for rel, expected in DEPENDENCIES.items():
        found = git_blob((root / rel).read_bytes())
        if found != expected:
            raise SystemExit(f"V3 runtime-repair dependency relay mismatch:{rel}:{found}")

    for rel in (PREPARE, EXECUTOR, SANITIZER):
        subprocess.run([sys.executable, "-W", "error::SyntaxWarning", "-m", "py_compile", str(root / rel)], check=True)

    request = json.loads((root / REQUEST).read_text(encoding="utf-8"))
    if request.get("request_id") != REQUEST_ID or request.get("target") != TARGET or request.get("title") != TITLE:
        raise SystemExit("V3 runtime-repair request identity mismatch")
    if TITLE != TARGET.split("/", 1)[1]:
        raise SystemExit("V3 runtime-repair title-target slug mismatch")
    if request.get("automatic_compute_retries") != 0 or request.get("competition_submission_authorized") is not False:
        raise SystemExit("V3 runtime-repair retry/submission boundary changed")
    parent = request.get("parent_runtime_failure") or {}
    if parent.get("kernel_terminal_state") != "ERROR" or parent.get("same_target_version_rerun_forbidden") is not True:
        raise SystemExit("V3 runtime-repair parent version-consumption boundary changed")
    if parent.get("failure_code") != FAILURE_CODE or parent.get("missing_module") != "cmi_flu.strategy_e04_contract":
        raise SystemExit("V3 runtime-repair parent root cause changed")
    observed_code = hashlib.sha256(
        b"materialize_dependency_package:ModuleNotFoundError:No module named 'cmi_flu.strategy_e04_contract'"
    ).hexdigest()[:20]
    if observed_code != FAILURE_CODE:
        raise SystemExit("V3 runtime-repair failure-code proof mismatch")
    deps = request.get("dependency_blob_shas") or {}
    if set(deps.values()) != set(DEPENDENCIES.values()):
        raise SystemExit("V3 runtime-repair request dependency blobs changed")
    if (request.get("source_B") or {}).get("sha256") != SOURCE_B_SHA or (request.get("forbidden_source_C") or {}).get("sha256") != SOURCE_C_SHA:
        raise SystemExit("V3 runtime-repair source identity changed")

    runtime = work / "runtime.py"
    subprocess.run([sys.executable, str(root / PREPARE), "--repository-root", str(root), "--output", str(runtime)], check=True)
    subprocess.run([sys.executable, str(runtime), "--self-test"], check=True)
    text = runtime.read_text(encoding="utf-8")
    for token in (
        REQUEST_ID, TARGET, V3_BLOB, SOURCE_B_SHA,
        "CMI_FLU_V3_BATCH1_RUNTIME_REPAIR_SELF_TEST PASS",
        "dependency_import_smoke=true", "model_fit_count=0", "competition_submit=false",
    ):
        if token not in text:
            raise SystemExit(f"V3 runtime-repair generated runtime token missing:{token}")
    if "kaggle competitions submit" in (root / V3_PAYLOAD).read_text(encoding="utf-8").casefold():
        raise SystemExit("V3 runtime-repair science source contains Competition submit path")
    if ".fit(" in (root / V3_PAYLOAD).read_text(encoding="utf-8"):
        raise SystemExit("V3 runtime-repair science source contains fit path")
    if len(runtime.read_bytes()) >= 1_500_000:
        raise SystemExit("V3 runtime-repair runtime byte budget exceeded")

    print(
        "CMI_FLU_V3_BATCH1_RUNTIME_REPAIR_CI PASS exact_science=true exact_dependencies=3 "
        "dependency_import_smoke=true parent_failure_proven=true fresh_target=true auth=false write=false "
        "compute=false model_fit=false competition_submit=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
