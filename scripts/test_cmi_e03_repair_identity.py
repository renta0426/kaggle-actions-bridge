#!/usr/bin/env python3
"""Prove E03 request 002 is a one-field-class repair of the rejected request 001."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

OLD_REQUEST = "requests/cmi-flu-strategy-e03-task11-optional-view-fusion-001.json"
NEW_REQUEST = "requests/cmi-flu-strategy-e03-task11-optional-view-fusion-002.json"
NEW_REQUEST_ID = "20260908-cmi-flu-strategy-e03-task11-optional-view-fusion-002"
NEW_TARGET = "renta0426/cmi-flu-e03-task11-optview-20260908-002"
NEW_TITLE = "CMI Flu E03 Task11 Optview 20260908 002"
NEW_SLUG = "cmi-flu-e03-task11-optview-20260908-002"
CONSERVATIVE_IDENTITY_MAX_CHARS = 48
PRIOR_STDERR_SHA256 = "ef812bc9f178b83f556046dfa3dc61b4d05a90e4003989ea212e7552cf7b8f2c"
PRIOR_STDERR_BYTES = 102
PRIOR_SAVEKERNEL_400 = (
    b"400 Client Error: Bad Request for url: "
    b"https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel\n"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    old = json.loads((root / OLD_REQUEST).read_text(encoding="utf-8"))
    new = json.loads((root / NEW_REQUEST).read_text(encoding="utf-8"))

    if len(PRIOR_SAVEKERNEL_400) != PRIOR_STDERR_BYTES:
        raise SystemExit("prior SaveKernel 400 byte receipt regression")
    if hashlib.sha256(PRIOR_SAVEKERNEL_400).hexdigest() != PRIOR_STDERR_SHA256:
        raise SystemExit("prior SaveKernel 400 digest receipt regression")

    expected_slug = re.sub(r"[^a-z0-9]+", "-", NEW_TITLE.lower()).strip("-")
    if expected_slug != NEW_SLUG or NEW_TARGET != f"renta0426/{NEW_SLUG}":
        raise SystemExit("fresh E03 title/slug resolution mismatch")
    if not (5 <= len(NEW_TITLE) <= CONSERVATIVE_IDENTITY_MAX_CHARS):
        raise SystemExit("fresh E03 title outside conservative length contract")
    if not (5 <= len(NEW_SLUG) <= CONSERVATIVE_IDENTITY_MAX_CHARS):
        raise SystemExit("fresh E03 slug outside conservative length contract")
    if new.get("request_id") != NEW_REQUEST_ID or new.get("target") != NEW_TARGET:
        raise SystemExit("fresh E03 request identity mismatch")

    old_core = {k: v for k, v in old.items() if k not in {"request_id", "target"}}
    new_core = {k: v for k, v in new.items() if k not in {"request_id", "target", "repair_provenance"}}
    if old_core != new_core:
        raise SystemExit("E03 repair changed science/runtime/resource contract")

    provenance = new.get("repair_provenance") or {}
    expected_provenance = {
        "repair_of_request_id": old.get("request_id"),
        "prior_workflow_run_id": 34193867697,
        "prior_target": old.get("target"),
        "prior_failure_class": "kaggle_save_kernel_http_400",
        "prior_stderr_bytes": PRIOR_STDERR_BYTES,
        "prior_stderr_sha256": PRIOR_STDERR_SHA256,
        "single_repair": "shorten_kernel_title_and_slug",
        "old_title_chars": 52,
        "old_slug_chars": 52,
        "new_title_chars": len(NEW_TITLE),
        "new_slug_chars": len(NEW_SLUG),
    }
    if provenance != expected_provenance:
        raise SystemExit("E03 repair provenance mismatch")

    print(
        "CMI_FLU_E03_REPAIR_IDENTITY PASS "
        f"prior_failure=savekernel_http_400 title_chars={len(NEW_TITLE)} "
        f"slug_chars={len(NEW_SLUG)} science_runtime_unchanged=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
