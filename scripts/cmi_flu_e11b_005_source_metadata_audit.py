#!/usr/bin/env python3
"""Read-only source-projection audit for consumed CMI-Flu E11b-005.

This diagnostic exists only to observe how Kaggle API 2.2.4 represents the
model_data_sources and competition_data_sources collections for the already
consumed E11b-005 version.  It performs no kernel write, output download,
submission, rerun, or compute operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

REQUEST_ID = "20260910-cmi-flu-e11b-005-source-metadata-audit-001"
TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-005"
EXPECTED_VERSION = 1
EXPECTED_MODEL_SOURCE = "prior-labsai/tabpfn-3/pytorch/default/1"
EXPECTED_COMPETITION = "cmi-flu-first-prediction-challenge"
OUTPUT_NAME = "source-metadata-audit.json"
MAX_OUTPUT_BYTES = 65_536

# Public metadata names that are useful for identifying an attached Kaggle
# source.  Only scalar values under these exact names are serialized.
ALLOWED_ITEM_ATTRIBUTES = (
    "ref",
    "id",
    "source",
    "source_ref",
    "source_type",
    "source_name",
    "source_version",
    "source_version_number",
    "name",
    "owner",
    "owner_slug",
    "user_name",
    "slug",
    "version",
    "version_number",
    "model_ref",
    "model_id",
    "model_slug",
    "model_owner",
    "model_owner_slug",
    "model_instance",
    "model_instance_slug",
    "model_instance_version",
    "model_instance_version_number",
    "model_variation",
    "model_variation_slug",
    "model_version",
    "model_version_number",
    "competition_ref",
    "competition_id",
    "competition_name",
    "competition_slug",
    "dataset_ref",
    "dataset_id",
    "dataset_slug",
    "url",
)

BANNED_VALUE_TOKENS = (
    "kgat_",
    "authorization",
    "bearer ",
    "cookie",
    "access_token",
    "api_token",
    "password",
    "participant_id",
    "subject_group",
    "oof_predictions",
    "challenge_predictions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _safe_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        lowered = value.casefold()
        if any(token in lowered for token in BANNED_VALUE_TOKENS):
            return "<redacted>"
        if len(value.encode("utf-8", errors="replace")) > 2_048:
            digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
            return f"<oversize-string sha256={digest}>"
        return value
    raise TypeError(type(value).__name__)


def _allowed_mapping_values(item: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for name in ALLOWED_ITEM_ATTRIBUTES:
        if name not in item:
            continue
        value = item[name]
        try:
            projected[name] = _safe_scalar(value)
        except TypeError:
            projected[name] = {
                "value_type": type(value).__name__,
                "scalar_serialized": False,
            }
    return projected


def _allowed_object_values(item: Any) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for name in ALLOWED_ITEM_ATTRIBUTES:
        try:
            value = getattr(item, name)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            continue
        try:
            projected[name] = _safe_scalar(value)
        except TypeError:
            projected[name] = {
                "value_type": type(value).__name__,
                "scalar_serialized": False,
            }
    return projected


def project_source_item(item: Any) -> dict[str, Any]:
    """Project only type information and allowlisted public scalar values."""
    descriptor: dict[str, Any] = {
        "python_type": type(item).__name__,
        "python_module": type(item).__module__,
        "direct_string": isinstance(item, str),
    }
    if isinstance(item, str):
        descriptor["string_value"] = _safe_scalar(item)
        descriptor["allowlisted_attributes"] = {}
    elif isinstance(item, Mapping):
        descriptor["allowlisted_attributes"] = _allowed_mapping_values(item)
        descriptor["mapping_key_count"] = len(item)
        descriptor["mapping_allowlisted_key_count"] = len(
            descriptor["allowlisted_attributes"]
        )
    else:
        descriptor["allowlisted_attributes"] = _allowed_object_values(item)
    return descriptor


def project_source_collection(value: Any) -> dict[str, Any]:
    if value is None:
        items: list[Any] = []
        collection_type = "NoneType"
    elif isinstance(value, (str, bytes, bytearray)):
        # A source collection becoming a scalar is itself useful schema evidence.
        items = [value.decode("utf-8", errors="replace") if isinstance(value, (bytes, bytearray)) else value]
        collection_type = type(value).__name__
    elif isinstance(value, Sequence):
        items = list(value)
        collection_type = type(value).__name__
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
        collection_type = type(value).__name__
    if len(items) > 32:
        raise RuntimeError(f"unexpectedly large metadata source collection:{len(items)}")
    return {
        "collection_type": collection_type,
        "count": len(items),
        "items": [project_source_item(item) for item in items],
    }


def _projected_strings(collection: Mapping[str, Any]) -> list[str]:
    strings: list[str] = []
    for item in collection.get("items", []):
        value = item.get("string_value")
        if isinstance(value, str) and value != "<redacted>":
            strings.append(value)
        attrs = item.get("allowlisted_attributes", {})
        if isinstance(attrs, Mapping):
            for attr_value in attrs.values():
                if isinstance(attr_value, str) and attr_value != "<redacted>":
                    strings.append(attr_value)
    return strings


def build_audit(metadata: Any) -> dict[str, Any]:
    model_collection = project_source_collection(
        getattr(metadata, "model_data_sources", None)
    )
    competition_collection = project_source_collection(
        getattr(metadata, "competition_data_sources", None)
    )
    model_strings = _projected_strings(model_collection)
    competition_strings = _projected_strings(competition_collection)

    version_value = getattr(metadata, "current_version_number", None)
    try:
        version = int(version_value or 0)
    except (TypeError, ValueError):
        version = 0

    return {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "operation": "kernel_metadata_source_projection_read",
        "target": TARGET,
        "expected_kernel_version": EXPECTED_VERSION,
        "metadata": {
            "python_type": type(metadata).__name__,
            "python_module": type(metadata).__module__,
            "ref": _safe_scalar(str(getattr(metadata, "ref", "") or "")),
            "current_version_number": version,
            "is_private": bool(getattr(metadata, "is_private", False)),
            "enable_gpu": bool(getattr(metadata, "enable_gpu", False)),
            "enable_tpu": bool(getattr(metadata, "enable_tpu", False)),
            "enable_internet": bool(getattr(metadata, "enable_internet", False)),
        },
        "expected_sources": {
            "model_source": EXPECTED_MODEL_SOURCE,
            "competition": EXPECTED_COMPETITION,
        },
        "model_data_sources": model_collection,
        "competition_data_sources": competition_collection,
        "projection_diagnostics": {
            "expected_model_source_seen_in_allowlisted_scalar_projection": (
                EXPECTED_MODEL_SOURCE in model_strings
            ),
            "expected_competition_seen_in_allowlisted_scalar_projection": (
                EXPECTED_COMPETITION in competition_strings
            ),
            "model_projected_string_count": len(model_strings),
            "competition_projected_string_count": len(competition_strings),
            "validator_repair_selected": False,
        },
        "side_effects": {
            "write_attempted": False,
            "compute_requested": False,
            "output_download_requested": False,
            "competition_submission_attempted": False,
            "kernel_rerun_attempted": False,
        },
        "output_policy": {
            "arbitrary_object_dict_dumped": False,
            "arbitrary_repr_dumped": False,
            "participant_identifiers_included": False,
            "row_level_predictions_included": False,
        },
    }


def validate_audit(payload: Mapping[str, Any]) -> None:
    if payload.get("request_id") != REQUEST_ID or payload.get("target") != TARGET:
        raise RuntimeError("audit identity changed")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("audit metadata missing")
    if metadata.get("ref") != TARGET:
        raise RuntimeError("consumed target metadata ref mismatch")
    if int(metadata.get("current_version_number", 0) or 0) != EXPECTED_VERSION:
        raise RuntimeError("consumed target version mismatch")
    side_effects = payload.get("side_effects")
    if not isinstance(side_effects, Mapping) or any(bool(v) for v in side_effects.values()):
        raise RuntimeError("read-only audit side-effect contract changed")
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    lowered = serialized.casefold()
    for token in BANNED_VALUE_TOKENS:
        if token in lowered and token not in {
            "participant_id",
            "subject_group",
            "oof_predictions",
            "challenge_predictions",
        }:
            raise RuntimeError("audit output contains forbidden secret-like material")
    if len(serialized.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise RuntimeError("audit output exceeds byte limit")


def write_audit(payload: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / OUTPUT_NAME
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise RuntimeError("audit output exceeds byte limit")
    path.write_bytes(encoded)
    return path


def self_test() -> int:
    class FakeSource:
        def __init__(self) -> None:
            self.ref = "owner/model/framework/variation/1"
            self.source_type = "MODEL"
            self.version_number = 1
            self.secret = "KGAT_SHOULD_NEVER_APPEAR"

    class FakeMetadata:
        ref = TARGET
        current_version_number = 1
        is_private = True
        enable_gpu = False
        enable_tpu = False
        enable_internet = True
        model_data_sources = [FakeSource(), EXPECTED_MODEL_SOURCE]
        competition_data_sources = [
            {"ref": EXPECTED_COMPETITION, "secret": "KGAT_NOT_ALLOWLISTED"}
        ]

    payload = build_audit(FakeMetadata())
    validate_audit(payload)
    serialized = json.dumps(payload, sort_keys=True)
    if "KGAT_SHOULD_NEVER_APPEAR" in serialized or "KGAT_NOT_ALLOWLISTED" in serialized:
        raise RuntimeError("self-test leaked non-allowlisted values")
    if payload["model_data_sources"]["count"] != 2:
        raise RuntimeError("self-test model source count changed")
    if not payload["projection_diagnostics"][
        "expected_model_source_seen_in_allowlisted_scalar_projection"
    ]:
        raise RuntimeError("self-test direct string projection failed")
    if not payload["projection_diagnostics"][
        "expected_competition_seen_in_allowlisted_scalar_projection"
    ]:
        raise RuntimeError("self-test mapping projection failed")
    print("CMI_FLU_E11B_005_SOURCE_METADATA_AUDIT_SELF_TEST PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.output_dir is None:
        raise SystemExit("--output-dir is required outside --self-test")

    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_") or token != token.strip():
        raise SystemExit("KAGGLE_API_TOKEN contract failed")

    api = KaggleApi()
    api.authenticate()
    metadata = api.kernel_metadata(TARGET)
    payload = build_audit(metadata)
    validate_audit(payload)
    path = write_audit(payload, args.output_dir.resolve())
    print(
        "CMI_FLU_E11B_005_SOURCE_METADATA_AUDIT PASS "
        f"version={payload['metadata']['current_version_number']} "
        f"model_sources={payload['model_data_sources']['count']} "
        f"competition_sources={payload['competition_data_sources']['count']} "
        f"bytes={path.stat().st_size} write=false compute=false submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
