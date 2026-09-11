#!/usr/bin/env python3
"""Credential-free regression for E12c-v2 historical-backbone Notebook."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

SCIENCE_BLOB = "238d3ad67984fdd8c375bb0aa263bb4717029027"
CONTRACT_BLOB = "f2835633c83f82fd9f8a9dc1f25598aa098ff56f"
SCIENCE_PATH = "payloads/cmi-flu-e12c-v2-historical-backbone-002/strategy_e12c_v2.py"
CONTRACT_PATH = "payloads/cmi-flu-e12c-v2-historical-backbone-002/strategy_e12c_v2_historical_backbone.json"
PREPARE = "scripts/cmi_flu_e12c_v2_historical_backbone_prepare.py"
EXECUTOR = "scripts/cmi_flu_e12c_v2_historical_backbone_execute.py"
SANITIZER = "scripts/cmi_flu_e12c_v2_historical_backbone_sanitize.py"
REQUEST = "requests/cmi-flu-e12c-v2-historical-backbone-002.json"
WORKFLOW = ".github/workflows/cmi-flu-e12c-v2-historical-backbone-002.yml"
OLD_TARGET = "renta0426/cmi-flu-e12c-manual-submission-20260911-001"
NEW_TARGET = "renta0426/cmi-flu-e12c-manual-submission-20260911-002"
HISTORICAL_SHA = "365607d59cd530656b929a1c1c57412cc6d375265a8d1ba10d304c64e012f387"
TASK13_SHA = "de8bc3b6bbd3e3aad4099b83eb63a1ebbd81c4f4eeb60f77808858f4674be5da"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    work = args.workdir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    for rel, expected in ((SCIENCE_PATH, SCIENCE_BLOB), (CONTRACT_PATH, CONTRACT_BLOB)):
        raw = (root / rel).read_bytes()
        if git_blob(raw) != expected:
            raise SystemExit(f"E12c-v2 exact relay mismatch:{rel}")

    for rel in (PREPARE, EXECUTOR, SANITIZER):
        subprocess.run(
            [sys.executable, "-W", "error::SyntaxWarning", "-m", "py_compile", str(root / rel)],
            check=True,
        )

    runtime = work / "runtime.py"
    subprocess.run(
        [
            sys.executable,
            str(root / PREPARE),
            "--repository-root",
            str(root),
            "--output",
            str(runtime),
        ],
        check=True,
    )
    subprocess.run([sys.executable, str(runtime), "--self-test"], check=True)
    text = runtime.read_text(encoding="utf-8")
    required = (
        NEW_TARGET,
        'changed_tasks_vs_historical_0_218',
        'refit_unchanged=false',
        HISTORICAL_SHA,
        TASK13_SHA,
        'CMI_FLU_E12C_V2_MANUAL_SUBMISSION_READY',
    )
    if any(token not in text for token in required):
        raise SystemExit("E12c-v2 generated runtime token contract incomplete")
    if OLD_TARGET in text:
        raise SystemExit("E12c-v2 runtime references consumed v1 target")
    lowered = text.casefold()
    for token in ("competition_submit(", "kaggle competitions submit"):
        if token in lowered:
            raise SystemExit("E12c-v2 runtime contains Competition submission path")
    if len(runtime.read_bytes()) >= 950000:
        raise SystemExit("E12c-v2 runtime byte budget exceeded")

    request_text = (root / REQUEST).read_text(encoding="utf-8")
    if '"execution_policy": "kaggle_native_capacity_v2"' not in request_text:
        raise SystemExit("E12c-v2 request policy marker missing")
    if any(token in request_text for token in ("max_active_runs", "min_remaining_quota_hours")):
        raise SystemExit("E12c-v2 request contains bridge capacity field")

    workflow_path = root / WORKFLOW
    if workflow_path.exists():
        workflow = workflow_path.read_text(encoding="utf-8")
        for token in ("quota_view(", "GPU admission", "CPU admission", "TPU admission", "max_active_runs"):
            if token in workflow:
                raise SystemExit(f"E12c-v2 workflow contains bridge capacity gate:{token}")

    print(
        "CMI_FLU_E12C_V2_CI PASS exact_relay=true runtime_self_test=true "
        "fresh_target=true refit_unchanged=false auth=false write=false compute=false submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
