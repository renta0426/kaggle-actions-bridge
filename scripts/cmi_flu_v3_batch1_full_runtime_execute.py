#!/usr/bin/env python3
"""Exactly one fresh private V3 operation; no automatic retry or submission."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys

import cmi_flu_v3_batch1_execute as base

REQUEST_ID = "20260912-cmi-flu-strategy-v3-batch1-full-runtime-repair-004"
TARGET = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260912-004"


def main() -> int:
    base.REQUEST_ID = REQUEST_ID
    base.TARGET = TARGET
    base.TITLE = TARGET.split("/", 1)[1]
    base.EXPECTED_VERSION = 1
    if "--path-self-test" not in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--runtime", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--approved-runtime-sha256", required=True)
        args = parser.parse_args()
        if not re.fullmatch(r"[0-9a-f]{64}", args.approved_runtime_sha256):
            raise SystemExit("V3 approved runtime digest is missing")
        if hashlib.sha256(args.runtime.read_bytes()).hexdigest() != args.approved_runtime_sha256:
            raise SystemExit("V3 approved runtime digest mismatch before any Kaggle call")
        sys.argv = [sys.argv[0], "--runtime", str(args.runtime), "--output-dir", str(args.output_dir)]
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
