#!/usr/bin/env python3
"""Fail-closed sanitizer for the E11b-005 source metadata audit artifact."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

REQUEST_ID = "20260910-cmi-flu-e11b-005-source-metadata-audit-001"
TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-005"
OUTPUT_NAME = "source-metadata-audit.json"
MAX_BYTES = 65_536
ALLOWED_TOP_LEVEL = {
    "schema_version",
    "request_id",
    "operation",
    "target",
    "expected_kernel_version",
    "metadata",
    "expected_sources",
    "model_data_sources",
    "competition_data_sources",
    "projection_diagnostics",
    "side_effects",
    "output_policy",
}
ALLOWED_ITEM_KEYS = {
    "python_type",
    "python_module",
    "direct_string",
    "string_value",
    "allowlisted_attributes",
    "mapping_key_count",
    "mapping_allowlisted_key_count",
}
BANNED_RAW_TOKENS = (
    "kgat_",
    "authorization",
    "bearer ",
    "cookie",
    "access_token",
    "api_token",
    "password",
    "oof_predictions",
    "challenge_predictions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _validate_collection(name: str, value: object) -> None:
    if not isinstance(value, Mapping):
        raise SystemExit(f"{name} must be an object")
    count = value.get("count")
    items = value.get("items")
    if not isinstance(count, int) or count < 0 or count > 32:
        raise SystemExit(f"{name} count invalid")
    if not isinstance(items, list) or len(items) != count:
        raise SystemExit(f"{name} items/count mismatch")
    for item in items:
        if not isinstance(item, Mapping):
            raise SystemExit(f"{name} item must be an object")
        unexpected = set(item) - ALLOWED_ITEM_KEYS
        if unexpected:
            raise SystemExit(f"{name} item contains unexpected keys")
        attrs = item.get("allowlisted_attributes")
        if not isinstance(attrs, Mapping):
            raise SystemExit(f"{name} item allowlisted_attributes invalid")
        if len(attrs) > 48:
            raise SystemExit(f"{name} item attribute count too large")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    if files != [OUTPUT_NAME]:
        raise SystemExit(f"unexpected audit output file set:{files}")
    path = output_dir / OUTPUT_NAME
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise SystemExit("metadata audit artifact exceeds byte limit")
    lowered = raw.decode("utf-8", errors="strict").casefold()
    for token in BANNED_RAW_TOKENS:
        if token in lowered:
            raise SystemExit("metadata audit artifact contains forbidden token")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise SystemExit("metadata audit top level must be an object")
    if set(payload) != ALLOWED_TOP_LEVEL:
        raise SystemExit("metadata audit top-level schema changed")
    if payload.get("schema_version") != 1:
        raise SystemExit("metadata audit schema version changed")
    if payload.get("request_id") != REQUEST_ID or payload.get("target") != TARGET:
        raise SystemExit("metadata audit identity changed")
    if payload.get("expected_kernel_version") != 1:
        raise SystemExit("metadata audit expected version changed")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise SystemExit("metadata audit metadata object missing")
    if metadata.get("ref") != TARGET or metadata.get("current_version_number") != 1:
        raise SystemExit("metadata audit consumed target/version mismatch")
    _validate_collection("model_data_sources", payload.get("model_data_sources"))
    _validate_collection(
        "competition_data_sources", payload.get("competition_data_sources")
    )
    side_effects = payload.get("side_effects")
    if not isinstance(side_effects, Mapping) or any(bool(v) for v in side_effects.values()):
        raise SystemExit("metadata audit side-effect flag changed")
    policy = payload.get("output_policy")
    if not isinstance(policy, Mapping) or any(bool(v) for v in policy.values()):
        raise SystemExit("metadata audit output policy flag changed")
    projection = payload.get("projection_diagnostics")
    if not isinstance(projection, Mapping) or projection.get("validator_repair_selected") is not False:
        raise SystemExit("metadata audit unexpectedly selected a repair")
    print(
        "CMI_FLU_E11B_005_SOURCE_METADATA_SANITIZE PASS "
        f"bytes={len(raw)} files=1 write=false compute=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
