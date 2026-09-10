#!/usr/bin/env python3
"""Aggregate-only sanitizer for E11b 004; science/output schema unchanged."""
from __future__ import annotations

import cmi_flu_strategy_e11b_sanitize as prior

REQUEST_ID = "20260910-cmi-flu-strategy-e11b-tabpfn3-004"
TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-004"

if prior.REQUEST_ID != "20260910-cmi-flu-strategy-e11b-tabpfn3-001":
    raise SystemExit("E11b sanitizer-v4 request ancestry changed")
if prior.TARGET != "renta0426/cmi-flu-e11b-tabpfn3-20260910-001":
    raise SystemExit("E11b sanitizer-v4 target ancestry changed")

prior.REQUEST_ID = REQUEST_ID
prior.TARGET = TARGET


def main() -> int:
    return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
