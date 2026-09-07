#!/usr/bin/env python3
"""Aggregate-only sanitizer for E05 003."""
from __future__ import annotations

import cmi_flu_strategy_e05_sanitize_v2 as base

REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-003"
if base.REQUEST_ID != "20260907-cmi-flu-strategy-e05-hai-donor-strain-002":
    raise SystemExit("E05 v2 sanitizer contract changed")
if base.SCIENCE_COMMIT != "e02a601f38250480b526e548a48cf7f2526c00ca" or base.E05_BLOB != "78cde9e3a6b6e3b332352218c4b4545f769aa6c4":
    raise SystemExit("E05 v2 sanitizer provenance changed")
base.REQUEST_ID = REQUEST_ID


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
