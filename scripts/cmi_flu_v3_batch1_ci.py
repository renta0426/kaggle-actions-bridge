#!/usr/bin/env python3
"""Credential-free regression for CMI-Flu Strategy-v3 batch 1."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import re
import subprocess
import sys

SCIENCE_BLOB = "cedd8e2538a06c8b696e74631f0bb2fce427984b"
SCIENCE_PATH = "payloads/cmi-flu-v3-batch1-001/strategy_v3_batch1.py"
PREPARE = "scripts/cmi_flu_v3_batch1_prepare.py"
EXECUTOR = "scripts/cmi_flu_v3_batch1_execute.py"
SANITIZER = "scripts/cmi_flu_v3_batch1_sanitize.py"
REQUEST = "requests/cmi-flu-v3-batch1-001.json"
WORKFLOW = ".github/workflows/cmi-flu-v3-batch1-001.yml"
TARGET = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260911-001"
SCIENCE_COMMIT = "270cd6fc5e33579add8177303505e5f744ee5dbc"
SOURCE_B_SHA = "0f9df53c3aa8c6e4ac693f6a42dbd2633b4b61d1462df798a9bd767c220be3a5"
SOURCE_C_SHA = "983aaf097d04477c4ccf7bf817fdf66e552937e69bceaa48cfb260cb84413f1b"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def assignment_string(tree: ast.AST, name: str) -> str:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    raise SystemExit(f"V3 batch1 executor string assignment missing:{name}")


def normalized_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args(); root = args.repository_root.resolve(); work = args.workdir.resolve(); work.mkdir(parents=True, exist_ok=True)
    raw = (root / SCIENCE_PATH).read_bytes()
    if git_blob(raw) != SCIENCE_BLOB:
        raise SystemExit("V3 batch1 exact science relay mismatch")
    for rel in (PREPARE, EXECUTOR, SANITIZER):
        subprocess.run([sys.executable, "-W", "error::SyntaxWarning", "-m", "py_compile", str(root / rel)], check=True)

    executor_text = (root / EXECUTOR).read_text(encoding="utf-8")
    executor_tree = ast.parse(executor_text, filename=EXECUTOR)
    executor_target = assignment_string(executor_tree, "TARGET")
    executor_title = assignment_string(executor_tree, "TITLE")
    if executor_target != TARGET:
        raise SystemExit("V3 batch1 executor target identity changed")
    if normalized_slug(executor_title) != TARGET.split("/", 1)[1]:
        raise SystemExit("V3 batch1 executor title does not normalize to target slug")
    if "validate_title_target_identity()" not in executor_text:
        raise SystemExit("V3 batch1 runtime title-target guard missing")

    runtime = work / "runtime.py"
    subprocess.run([sys.executable, str(root / PREPARE), "--repository-root", str(root), "--output", str(runtime)], check=True)
    subprocess.run([sys.executable, str(runtime), "--self-test"], check=True)
    text = runtime.read_text(encoding="utf-8")
    required = (TARGET, SCIENCE_COMMIT, SCIENCE_BLOB, SOURCE_B_SHA, "CMI_FLU_V3_BATCH1_RUNTIME_PASS", "model_fit_count=0", "competition_submit=false")
    if any(token not in text for token in required):
        raise SystemExit("V3 batch1 generated runtime token contract incomplete")
    if len(runtime.read_bytes()) >= 1100000:
        raise SystemExit("V3 batch1 runtime byte budget exceeded")
    request = (root / REQUEST).read_text(encoding="utf-8")
    for token in ('"execution_policy": "kaggle_native_capacity_v2"', SOURCE_B_SHA, SOURCE_C_SHA, '"automatic_compute_retries": 0', '"competition_submission_authorized": false'):
        if token not in request:
            raise SystemExit(f"V3 batch1 request token missing:{token}")
    for token in ("max_active_runs", "min_remaining_quota_hours"):
        if token in request:
            raise SystemExit("V3 batch1 request contains forbidden bridge capacity gate")
    if (root / WORKFLOW).exists():
        workflow = (root / WORKFLOW).read_text(encoding="utf-8")
        for token in ("quota_view(", "GPU admission", "CPU admission", "TPU admission", "max_active_runs"):
            if token in workflow:
                raise SystemExit(f"V3 batch1 workflow contains bridge capacity gate:{token}")
    source = raw.decode("utf-8").casefold()
    if ".fit(" in source or "kaggle competitions submit" in source or "competition_submit(" in source:
        raise SystemExit("V3 batch1 science source violates no-fit/no-submit static contract")
    print("CMI_FLU_V3_BATCH1_CI PASS exact_relay=true runtime_self_test=true title_target_identity=true source_B_exact=true source_C_rejected=true auth=false write=false compute=false model_fit=false competition_submit=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
