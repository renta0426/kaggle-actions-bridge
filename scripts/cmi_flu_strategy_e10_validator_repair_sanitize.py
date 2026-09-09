#!/usr/bin/env python3
"""Aggregate-only sanitizer for fresh E10 validator-runtime repair 003."""
from __future__ import annotations

import cmi_flu_strategy_e10_repair_sanitize as prior

REQUEST_ID = "20260910-cmi-flu-strategy-e10-pooled-domain-correction-003"
TARGET = "renta0426/cmi-flu-e10-pooled-domain-correction-20260910-003"

if prior.REQUEST_ID != "20260909-cmi-flu-strategy-e10-pooled-domain-correction-002":
    raise SystemExit("E10 validator-repair sanitizer request ancestry changed")
if prior.TARGET != "renta0426/cmi-flu-e10-pooled-domain-correction-20260909-002":
    raise SystemExit("E10 validator-repair sanitizer target ancestry changed")
if prior.SCIENCE_COMMIT != "3a8728879179115487f5ce0ba4a4a0467e1910bb":
    raise SystemExit("E10 validator-repair sanitizer science ancestry changed")
if prior.MIN_SOURCE_SUBJECTS != 5:
    raise SystemExit("E10 validator-repair sanitizer source-size ancestry changed")

prior.REQUEST_ID = REQUEST_ID
prior.TARGET = TARGET


def main() -> int:
    return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
