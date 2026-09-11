#!/usr/bin/env python3
"""Execute exactly one fresh V3 batch1 runtime-repair successor Notebook."""
from __future__ import annotations

import cmi_flu_v3_batch1_execute as base

REQUEST_ID = "20260912-cmi-flu-strategy-v3-batch1-runtime-repair-002"
TARGET = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260912-002"
TITLE = "cmi-flu-v3-batch1-audit-diagnostics-20260912-002"


def main() -> int:
    # Reuse the reviewed one-write executor with only fresh successor identity.
    # Every function in the base module resolves these globals at call time.
    base.REQUEST_ID = REQUEST_ID
    base.TARGET = TARGET
    base.TITLE = TITLE
    base.EXPECTED_VERSION = 1
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
