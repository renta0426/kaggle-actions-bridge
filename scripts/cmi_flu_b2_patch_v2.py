#!/usr/bin/env python3
"""Deterministically patch the reviewed B2 bundle for Kaggle mount discovery.

Request 001 failed before any modeling because its locator inspected only direct
children of /kaggle/input. B1 already proved that locating
sample_submission_part1.csv recursively is robust to Kaggle's nested mount
layout. This patch changes only the request id and that locator function.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

BASE_SOURCE_SHA256 = "a039b56868e5f9e2f10ae8016466a02fe4a6e5342854487ca6c6a198a21abd5e"
REQUEST_ID = "20260903-cmi-flu-b2-002"
TARGET = "renta0426/cmi-flu-b2-taskwise-compact-20260903-002"
OLD_REQUEST = 'REQUEST_ID = "__REQUEST_ID__"'
NEW_REQUEST = f'REQUEST_ID = "{REQUEST_ID}"'
OLD_LOCATOR = '''def locate_competition_data(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.append(Path("/kaggle/input") / COMPETITION)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(sorted(path for path in kaggle_input.iterdir() if path.is_dir()))
    candidates.extend((Path.cwd(), Path.cwd() / "data" / "raw"))

    seen: set[Path] = set()
    valid: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        if all((resolved / name).is_file() for name in CORE_FILES):
            valid.append(resolved)
    if not valid:
        raise BundleContractError("CMI-Flu Competition Data mount was not found")
    exact = (Path("/kaggle/input") / COMPETITION).resolve()
    if exact in valid:
        return exact
    if explicit is not None:
        resolved_explicit = explicit.expanduser().resolve()
        if resolved_explicit in valid:
            return resolved_explicit
    if len(valid) != 1:
        raise BundleContractError("multiple candidate Competition Data directories were found")
    return valid[0]
'''
NEW_LOCATOR = '''def locate_competition_data(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(
            path.parent
            for path in kaggle_input.rglob("sample_submission_part1.csv")
        )

    candidates.extend((Path.cwd(), Path.cwd() / "data" / "raw"))

    seen: set[Path] = set()
    valid: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        if all((resolved / name).is_file() for name in CORE_FILES):
            valid.append(resolved)

    if explicit is not None:
        resolved_explicit = explicit.expanduser().resolve()
        if resolved_explicit in valid:
            return resolved_explicit

    if not valid:
        raise BundleContractError("CMI-Flu Competition Data mount was not found")
    if len(valid) != 1:
        raise BundleContractError("multiple candidate Competition Data directories were found")
    return valid[0]
'''
OLD_LOCATOR_SHA256 = "ec14e5dcb5fc8d33d01a3751dc81589290d32ddd79415665c25c56cf45987d21"
NEW_LOCATOR_SHA256 = "fdd1248fcea6527c4897c887a3a51200a4aa26f3aebf55e368b297bb94a7201e"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_request(request: dict) -> None:
    if request.get("schema_version") != 2:
        raise SystemExit("B2 v2 request schema mismatch")
    if request.get("request_id") != REQUEST_ID:
        raise SystemExit("B2 v2 request id mismatch")
    if request.get("parent_request_id") != "20260903-cmi-flu-b2-001":
        raise SystemExit("B2 v2 parent request mismatch")
    if request.get("competition") != "cmi-flu-first-prediction-challenge":
        raise SystemExit("B2 v2 competition mismatch")
    if request.get("operation") != "kernel_run" or request.get("target") != TARGET:
        raise SystemExit("B2 v2 operation/target mismatch")
    payload = request.get("payload", {})
    if payload.get("source_sha256") != BASE_SOURCE_SHA256:
        raise SystemExit("B2 v2 base source SHA mismatch")
    if payload.get("part_count") != 20 or payload.get("size_bytes") != 92839:
        raise SystemExit("B2 v2 payload inventory mismatch")
    request_patch = payload.get("request_id_replacement", {})
    if request_patch != {"old": OLD_REQUEST, "new": NEW_REQUEST, "count": 1}:
        raise SystemExit("B2 v2 request-id patch contract mismatch")
    locator_patch = payload.get("mount_locator_patch", {})
    if locator_patch.get("old_sha256") != OLD_LOCATOR_SHA256:
        raise SystemExit("B2 v2 old locator SHA mismatch")
    if locator_patch.get("new_sha256") != NEW_LOCATOR_SHA256:
        raise SystemExit("B2 v2 new locator SHA mismatch")
    if locator_patch.get("count") != 1:
        raise SystemExit("B2 v2 locator patch count mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 120,
        "hard_timeout_minutes": 240,
        "max_active_runs": 1,
    }:
        raise SystemExit("B2 v2 resource contract mismatch")
    if request.get("automatic_compute_retries") != 0 or request.get("enable_internet") is not False:
        raise SystemExit("B2 v2 retry/internet contract mismatch")


def patch_source(source: bytes, request: dict) -> bytes:
    if len(source) != 92839 or sha256_bytes(source) != BASE_SOURCE_SHA256:
        raise SystemExit("B2 v2 reconstructed base source mismatch")
    if sha256_bytes(OLD_LOCATOR.encode()) != OLD_LOCATOR_SHA256:
        raise SystemExit("embedded old locator SHA mismatch")
    if sha256_bytes(NEW_LOCATOR.encode()) != NEW_LOCATOR_SHA256:
        raise SystemExit("embedded new locator SHA mismatch")
    text = source.decode("utf-8")
    if text.count(OLD_REQUEST) != 1:
        raise SystemExit("B2 v2 request placeholder occurrence mismatch")
    if text.count(OLD_LOCATOR) != 1:
        raise SystemExit("B2 v2 old locator occurrence mismatch")
    text = text.replace(OLD_REQUEST, NEW_REQUEST, 1)
    text = text.replace(OLD_LOCATOR, NEW_LOCATOR, 1)
    if "__REQUEST_ID__" in text or OLD_LOCATOR in text:
        raise SystemExit("B2 v2 stale source remains after patch")
    compile(text, "script.py", "exec")

    tree = ast.parse(text)
    locator_nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "locate_competition_data"
    ]
    if len(locator_nodes) != 1:
        raise SystemExit("B2 v2 patched locator definition count mismatch")
    locator = ast.get_source_segment(text, locator_nodes[0]) or ""
    if '.rglob("sample_submission_part1.csv")' not in locator:
        raise SystemExit("B2 v2 patched locator lacks recursive sample discovery")
    if '.iterdir()' in locator:
        raise SystemExit("B2 v2 patched locator still uses top-level-only discovery")
    return text.encode("utf-8")


def main() -> int:
    args = parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    validate_request(request)
    patched = patch_source(args.source.read_bytes(), request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(patched)
    print(
        "CMI_FLU_B2_PATCH_V2 PASS "
        f"base_sha256={BASE_SOURCE_SHA256} patched_sha256={sha256_bytes(patched)} "
        f"request_id={REQUEST_ID} locator=recursive_sample_parent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
