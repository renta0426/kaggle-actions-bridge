#!/usr/bin/env python3
"""Identity-only wrapper around the audited E11b-005 metadata sanitizer."""
from __future__ import annotations

import cmi_flu_e11b_005_source_metadata_audit_sanitize as base

REQUEST_ID = "20260910-cmi-flu-e11b-005-source-metadata-audit-002"
TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-005"

if base.REQUEST_ID != "20260910-cmi-flu-e11b-005-source-metadata-audit-001":
    raise SystemExit("E11b-005 audit sanitizer-v1 identity changed")
if base.TARGET != TARGET or base.OUTPUT_NAME != "source-metadata-audit.json" or base.MAX_BYTES != 65_536:
    raise SystemExit("E11b-005 audit sanitizer-v1 contract changed")

base.REQUEST_ID = REQUEST_ID


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
