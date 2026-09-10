#!/usr/bin/env python3
"""Identity-only sanitizer wrapper for manual E11c-002."""
from __future__ import annotations

import cmi_flu_strategy_e11c_sanitize as base

REQUEST_ID = "20260910-cmi-flu-strategy-e11c-robust-tabpfn-002"
TARGET = "renta0426/cmi-flu-e11c-robust-tabpfn-20260910-002"

if base.REQUEST_ID != "20260910-cmi-flu-strategy-e11c-robust-tabpfn-001":
    raise SystemExit("E11c-002 sanitizer parent request changed")
if base.TARGET != "renta0426/cmi-flu-e11c-robust-tabpfn-20260910-001":
    raise SystemExit("E11c-002 sanitizer parent target changed")
if base.SCIENCE_COMMIT != "0815f4517345e127dbcbcae0f380a54f3a3d15bd":
    raise SystemExit("E11c-002 sanitizer science commit changed")
if base.E11C_BLOB != "acf876996b5849f7ba794999d45ecb862d1be507":
    raise SystemExit("E11c-002 sanitizer science blob changed")

base.REQUEST_ID = REQUEST_ID
base.TARGET = TARGET


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
