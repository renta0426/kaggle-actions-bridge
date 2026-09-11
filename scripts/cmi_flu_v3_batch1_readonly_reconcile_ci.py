#!/usr/bin/env python3
"""Credential-free regression for the V3 batch-1 read-only reconciliation path."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

REQUEST = "requests/cmi-flu-v3-batch1-readonly-reconcile-001.json"
SCRIPT = "scripts/cmi_flu_v3_batch1_readonly_reconcile.py"
WORKFLOW = ".github/workflows/cmi-flu-v3-batch1-readonly-reconcile-001.yml"
OLD_WRITER_WORKFLOW = ".github/workflows/cmi-flu-v3-batch1-001.yml"
INTENDED = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260911-001"
TITLE_DERIVED = "renta0426/cmi-flu-strategy-v3-batch1-audit-diagnostics-20260911-001"
SOURCE_B_SHA = "0f9df53c3aa8c6e4ac693f6a42dbd2633b4b61d1462df798a9bd767c220be3a5"
FORBIDDEN = (
    "kernels_push(", "kernels_push_cli", "competitions_submit", "competition_submit(",
    "kaggle competitions submit", "dataset_create", "dataset_version_create",
    "model_create", "model_instance_version_create",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args(); root = args.repository_root.resolve()
    source = (root / SCRIPT).read_text(encoding="utf-8")
    ast.parse(source, filename=SCRIPT)
    lowered = source.lower()
    bad = [token for token in FORBIDDEN if token.lower() in lowered]
    if bad:
        raise SystemExit(f"V3 reconciliation write marker forbidden:{bad}")
    for token in (INTENDED, TITLE_DERIVED, SOURCE_B_SHA, "outcome=none_found", "outcome=collision", "outcome=one_complete", "writes=0"):
        if token not in source:
            raise SystemExit(f"V3 reconciliation source token missing:{token}")

    request = json.loads((root / REQUEST).read_text(encoding="utf-8"))
    if request.get("operation") != "kernel_read_reconcile" or request.get("read_only") is not True or request.get("side_effects") != []:
        raise SystemExit("V3 reconciliation request is not strictly read-only")
    if request.get("candidate_kernel_refs") != [INTENDED, TITLE_DERIVED]:
        raise SystemExit("V3 reconciliation candidate refs changed")
    auth = request.get("authorization_boundary") or {}
    for key in ("kaggle_writes_authorized", "competition_submission_authorized", "kernel_push_authorized", "kernel_delete_authorized"):
        if auth.get(key) is not False:
            raise SystemExit(f"V3 reconciliation write authorization must be false:{key}")
    if auth.get("output_read_authorized") is not True or auth.get("aggregate_safe_recovery_authorized") is not True:
        raise SystemExit("V3 reconciliation read authorization missing")

    workflow = (root / WORKFLOW).read_text(encoding="utf-8")
    required = ("permissions: {}", "environment: kaggle-readonry", "CMI_FLU_V3_BATCH1_RECONCILE", "cmi_flu_v3_batch1_sanitize.py")
    if any(token not in workflow for token in required):
        raise SystemExit("V3 reconciliation workflow boundary incomplete")
    if any(token.lower() in workflow.lower() for token in FORBIDDEN):
        raise SystemExit("V3 reconciliation workflow contains write marker")

    old_writer = (root / OLD_WRITER_WORKFLOW).read_text(encoding="utf-8")
    for rel in (REQUEST, SCRIPT, WORKFLOW):
        if rel in old_writer:
            raise SystemExit(f"V3 reconciliation path unexpectedly triggers old writer workflow:{rel}")
    print("CMI_FLU_V3_BATCH1_RECONCILE_CI PASS read_only=true candidates=2 old_writer_not_triggered=true writes=0 competition_submit=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
