#!/usr/bin/env python3
"""Repair of E11b-005 source metadata audit using one proven exact GetKernel read.

Audit 001 failed locally before an exact Kaggle metadata request because it
called a nonexistent KaggleApi.kernel_metadata method under kaggle==2.2.4.
This repair reuses the already-audited projection/output code unchanged and
replaces only that retrieval primitive with kaggle_exact_identity.exact_metadata.
"""
from __future__ import annotations

import os

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle_exact_identity import exact_metadata

import cmi_flu_e11b_005_source_metadata_audit as base

REQUEST_ID = "20260910-cmi-flu-e11b-005-source-metadata-audit-002"
TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-005"
EXPECTED_VERSION = 1

_EXPECTED_BASE = {
    "REQUEST_ID": "20260910-cmi-flu-e11b-005-source-metadata-audit-001",
    "TARGET": TARGET,
    "EXPECTED_VERSION": EXPECTED_VERSION,
    "OUTPUT_NAME": "source-metadata-audit.json",
    "MAX_OUTPUT_BYTES": 65_536,
}
for key, value in _EXPECTED_BASE.items():
    if getattr(base, key, None) != value:
        raise SystemExit(f"E11b-005 audit-v1 contract changed:{key}")

# The reused projection/validator functions resolve REQUEST_ID dynamically.
# Change identity only; their allowlists, redaction policy and byte bounds stay
# exactly as audited in 001.
base.REQUEST_ID = REQUEST_ID


def main() -> int:
    parsed = base.parse_args()
    if parsed.self_test:
        return base.self_test()
    if parsed.output_dir is None:
        raise SystemExit("--output-dir is required outside --self-test")

    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_") or token != token.strip():
        raise SystemExit("KAGGLE_API_TOKEN contract failed")

    api = KaggleApi()
    api.authenticate()

    # Exactly one direct, read-only GetKernel metadata request. Do not fall back
    # to search/list and do not retry: this is an old, known-existing consumed
    # target, not a just-created Notebook requiring eventual reconciliation.
    metadata = exact_metadata(api, TARGET)

    payload = base.build_audit(metadata)
    payload["operation"] = "kernel_exact_metadata_source_projection_read"
    base.validate_audit(payload)
    path = base.write_audit(payload, parsed.output_dir.resolve())
    print(
        "CMI_FLU_E11B_005_SOURCE_METADATA_AUDIT_002 PASS "
        f"version={payload['metadata']['current_version_number']} "
        f"model_sources={payload['model_data_sources']['count']} "
        f"competition_sources={payload['competition_data_sources']['count']} "
        f"bytes={path.stat().st_size} exact_reads=1 write=false compute=false submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
