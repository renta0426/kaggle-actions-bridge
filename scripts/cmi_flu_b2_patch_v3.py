#!/usr/bin/env python3
"""Deterministically patch the reviewed B2 bundle for request 003.

The base payload remains byte-for-byte identical to request 001. Request 003
applies only two reviewed corrections before Kaggle execution:

1. recursively discover the Competition mount, matching the successful B1 path;
2. extract the pinned embedded package and apply the CMI-main flow-label fix so
   strict harmonization filters predictors, not Task1.2/Task1.3 absolute labels.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

BASE_SOURCE_SHA256 = "a039b56868e5f9e2f10ae8016466a02fe4a6e5342854487ca6c6a198a21abd5e"
BASE_PACKAGE_SHA256 = "390a84bfc4393960ad1dfeab6a92a56cc8db57b1ae7c8063d8c2bf8b072b8ce0"
CMI_FLOW_FIX_COMMIT = "7b2e6c1fe59299518e9952ef92cb2d06e8508a39"
REQUEST_ID = "20260903-cmi-flu-b2-003"
TARGET = "renta0426/cmi-flu-b2-taskwise-compact-20260903-003"
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

OLD_RUNTIME_BLOCK = '''        stage = "materialize_package"
        bundle = package_bytes()
        package_path = runtime_root / "cmi_flu_bundle.zip"
        package_path.write_bytes(bundle)
        sys.path.insert(0, str(package_path))
'''

NEW_RUNTIME_BLOCK = '''        stage = "materialize_package"
        bundle = package_bytes()
        package_path = runtime_root / "cmi_flu_bundle.zip"
        package_path.write_bytes(bundle)
        package_dir = runtime_root / "package"
        with zipfile.ZipFile(package_path) as archive:
            archive.extractall(package_dir)
        datasets_path = package_dir / "cmi_flu" / "datasets.py"
        datasets_text = datasets_path.read_text(encoding="utf-8")
        old_flow_target = '    target_source = _strict_flow_rows(public_flow) if mode == "strict" else public_flow\\n'
        new_flow_target = (
            '    # Harmonization mode controls baseline predictor construction only. The\\n'
            '    # absolute D1/D7 response labels must use the original public flow table.\\n'
            '    target_source = public_flow\\n'
        )
        if datasets_text.count(old_flow_target) != 1:
            raise BundleContractError("embedded flow target patch occurrence mismatch")
        datasets_path.write_text(
            datasets_text.replace(old_flow_target, new_flow_target, 1),
            encoding="utf-8",
        )
        sys.path.insert(0, str(package_dir))
'''

OLD_RUNTIME_BLOCK_SHA256 = "c1e5e48ea8f9ef5d3b63d9970cd7e241f0c149057f223932cbefd5e239e4ef53"
NEW_RUNTIME_BLOCK_SHA256 = "ed40c75f73967fee40d9380a9585abc7b626576deafcd4b0320ab42919c050ef"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_request(request: dict) -> None:
    if request.get("schema_version") != 3:
        raise SystemExit("B2 v3 request schema mismatch")
    if request.get("request_id") != REQUEST_ID:
        raise SystemExit("B2 v3 request id mismatch")
    if request.get("parent_request_id") != "20260903-cmi-flu-b2-002":
        raise SystemExit("B2 v3 parent request mismatch")
    if request.get("competition") != "cmi-flu-first-prediction-challenge":
        raise SystemExit("B2 v3 competition mismatch")
    if request.get("operation") != "kernel_run" or request.get("target") != TARGET:
        raise SystemExit("B2 v3 operation/target mismatch")

    payload = request.get("payload", {})
    if payload.get("source_sha256") != BASE_SOURCE_SHA256:
        raise SystemExit("B2 v3 base source SHA mismatch")
    if payload.get("package_zip_sha256") != BASE_PACKAGE_SHA256:
        raise SystemExit("B2 v3 package SHA mismatch")
    if payload.get("part_count") != 20 or payload.get("size_bytes") != 92839:
        raise SystemExit("B2 v3 payload inventory mismatch")

    request_patch = payload.get("request_id_replacement", {})
    if request_patch != {"old": OLD_REQUEST, "new": NEW_REQUEST, "count": 1}:
        raise SystemExit("B2 v3 request-id patch contract mismatch")

    locator_patch = payload.get("mount_locator_patch", {})
    if locator_patch != {
        "old_sha256": OLD_LOCATOR_SHA256,
        "new_sha256": NEW_LOCATOR_SHA256,
        "count": 1,
    }:
        raise SystemExit("B2 v3 locator patch contract mismatch")

    flow_patch = payload.get("flow_target_patch", {})
    if flow_patch.get("cmi_main_commit") != CMI_FLOW_FIX_COMMIT:
        raise SystemExit("B2 v3 CMI flow-fix commit mismatch")
    if flow_patch.get("old_runtime_block_sha256") != OLD_RUNTIME_BLOCK_SHA256:
        raise SystemExit("B2 v3 old runtime-block SHA mismatch")
    if flow_patch.get("new_runtime_block_sha256") != NEW_RUNTIME_BLOCK_SHA256:
        raise SystemExit("B2 v3 new runtime-block SHA mismatch")
    if flow_patch.get("count") != 1:
        raise SystemExit("B2 v3 flow patch count mismatch")

    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 120,
        "hard_timeout_minutes": 240,
        "max_active_runs": 1,
    }:
        raise SystemExit("B2 v3 resource contract mismatch")
    if request.get("automatic_compute_retries") != 0 or request.get("enable_internet") is not False:
        raise SystemExit("B2 v3 retry/internet contract mismatch")
    if request.get("side_effects") != ["create one private Notebook version and start one CPU run"]:
        raise SystemExit("B2 v3 side-effect contract mismatch")


def patch_source(source: bytes, request: dict) -> bytes:
    if len(source) != 92839 or sha256_bytes(source) != BASE_SOURCE_SHA256:
        raise SystemExit("B2 v3 reconstructed base source mismatch")
    checks = (
        (OLD_LOCATOR, OLD_LOCATOR_SHA256, "old locator"),
        (NEW_LOCATOR, NEW_LOCATOR_SHA256, "new locator"),
        (OLD_RUNTIME_BLOCK, OLD_RUNTIME_BLOCK_SHA256, "old runtime block"),
        (NEW_RUNTIME_BLOCK, NEW_RUNTIME_BLOCK_SHA256, "new runtime block"),
    )
    for text, expected, label in checks:
        if sha256_bytes(text.encode()) != expected:
            raise SystemExit(f"embedded {label} SHA mismatch")

    text = source.decode("utf-8")
    for old, label in (
        (OLD_REQUEST, "request placeholder"),
        (OLD_LOCATOR, "old locator"),
        (OLD_RUNTIME_BLOCK, "old runtime block"),
    ):
        if text.count(old) != 1:
            raise SystemExit(f"B2 v3 {label} occurrence mismatch")

    text = text.replace(OLD_REQUEST, NEW_REQUEST, 1)
    text = text.replace(OLD_LOCATOR, NEW_LOCATOR, 1)
    text = text.replace(OLD_RUNTIME_BLOCK, NEW_RUNTIME_BLOCK, 1)
    if "__REQUEST_ID__" in text or OLD_LOCATOR in text or OLD_RUNTIME_BLOCK in text:
        raise SystemExit("B2 v3 stale source remains after patch")
    compile(text, "script.py", "exec")

    tree = ast.parse(text)
    locator_nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "locate_competition_data"
    ]
    if len(locator_nodes) != 1:
        raise SystemExit("B2 v3 patched locator definition count mismatch")
    locator = ast.get_source_segment(text, locator_nodes[0]) or ""
    if '.rglob("sample_submission_part1.csv")' not in locator or '.iterdir()' in locator:
        raise SystemExit("B2 v3 locator is not recursive-only")

    if 'datasets_text.replace(old_flow_target, new_flow_target, 1)' not in text:
        raise SystemExit("B2 v3 flow-target runtime patch is absent")
    if 'sys.path.insert(0, str(package_dir))' not in text:
        raise SystemExit("B2 v3 extracted package is not the import source")
    return text.encode("utf-8")


def main() -> int:
    args = parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    validate_request(request)
    patched = patch_source(args.source.read_bytes(), request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(patched)
    print(
        "CMI_FLU_B2_PATCH_V3 PASS "
        f"base_sha256={BASE_SOURCE_SHA256} patched_sha256={sha256_bytes(patched)} "
        f"request_id={REQUEST_ID} locator=recursive flow_labels=unfiltered_public_flow "
        f"cmi_flow_fix_commit={CMI_FLOW_FIX_COMMIT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
