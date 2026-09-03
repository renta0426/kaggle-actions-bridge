#!/usr/bin/env python3
"""Validate the regenerated broad-flow B2 bundle and patch Kaggle mount discovery."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

BASE_SOURCE_SHA256 = "7894a9232e7372950f70a36d35de41bcc8bda90ec4d0320187e82c3b8c76f2db"
BASE_SIZE = 93363
GENERATED_SOURCE_COMMIT = "d6297c36366ab5c3ef49b9077c2357277f82a708"
MERGED_CMI_COMMIT = "802d93bac61b97844adf846199863c7ca9604ea1"
PACKAGE_ZIP_SHA256 = "48ff4ed2eadec8b059b8d37fb677af1f249b6486bccfbd55dca2274cfc6f3dc3"
REQUEST_ID = "20260903-cmi-flu-b2-004"
TARGET = "renta0426/cmi-flu-b2-taskwise-compact-broad-20260903-004"
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


def literal_assignments(text: str) -> dict[str, Any]:
    tree = ast.parse(text)
    output: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id not in {"SOURCE_COMMIT", "PACKAGE_ZIP_SHA256", "PACKAGE_B64", "CONFIG_TEXT"}:
            continue
        output[target.id] = ast.literal_eval(node.value)
    return output


def validate_embedded_scientific_contract(text: str) -> None:
    values = literal_assignments(text)
    if values.get("SOURCE_COMMIT") != GENERATED_SOURCE_COMMIT:
        raise SystemExit("B2 broad generated source commit mismatch")
    if values.get("PACKAGE_ZIP_SHA256") != PACKAGE_ZIP_SHA256:
        raise SystemExit("B2 broad package SHA declaration mismatch")
    config = values.get("CONFIG_TEXT")
    if not isinstance(config, str):
        raise SystemExit("B2 broad CONFIG_TEXT missing")
    required_config = (
        "  task_12_mode: broad\n",
        "  task_13_mode: broad\n",
    )
    if not all(token in config for token in required_config):
        raise SystemExit("B2 broad flow config contract mismatch")
    if "  task_12_mode: strict\n" in config or "  task_13_mode: strict\n" in config:
        raise SystemExit("B2 broad config retains a strict flow task")

    encoded = values.get("PACKAGE_B64")
    if not isinstance(encoded, str):
        raise SystemExit("B2 broad embedded package missing")
    package = base64.b64decode("".join(encoded.split()), validate=True)
    if sha256_bytes(package) != PACKAGE_ZIP_SHA256:
        raise SystemExit("B2 broad embedded package bytes mismatch")
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        if archive.testzip() is not None:
            raise SystemExit("B2 broad embedded package is corrupt")
        flow = archive.read("cmi_flu/features/flow.py").decode("utf-8")
        datasets = archive.read("cmi_flu/datasets.py").decode("utf-8")
    for token in (
        "def _broad_rank_baseline(",
        'ranked_source["_unit_key"]',
        'ranked_source["_material_key"]',
        '["study_accession", "name", "_unit_key", "_material_key"]',
        "Combining already-unitless ranks is safe",
    ):
        if token not in flow:
            raise SystemExit(f"B2 broad flow implementation contract missing: {token}")
    if "target_source = public_flow" not in datasets:
        raise SystemExit("B2 absolute flow target-source correction missing")
    if "target_source = _strict_flow_rows(public_flow)" in datasets:
        raise SystemExit("B2 stale strict target-source logic remains")


def validate_request(request: dict[str, Any]) -> None:
    expected_top = {
        "schema_version": 4,
        "request_id": REQUEST_ID,
        "parent_request_id": "20260903-cmi-flu-b2-003",
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "kernel_run",
        "target": TARGET,
        "automatic_compute_retries": 0,
        "enable_internet": False,
    }
    for key, value in expected_top.items():
        if request.get(key) != value:
            raise SystemExit(f"B2 request 004 mismatch: {key}")
    payload = request.get("payload", {})
    if payload.get("parts_dir") != "payloads/cmi-flu-b2-broad-004":
        raise SystemExit("B2 request 004 payload directory mismatch")
    if payload.get("part_count") != 20 or payload.get("size_bytes") != BASE_SIZE:
        raise SystemExit("B2 request 004 payload inventory mismatch")
    if payload.get("source_sha256") != BASE_SOURCE_SHA256:
        raise SystemExit("B2 request 004 source SHA mismatch")
    if payload.get("generated_source_commit") != GENERATED_SOURCE_COMMIT:
        raise SystemExit("B2 request 004 generated source commit mismatch")
    if payload.get("merged_cmi_commit") != MERGED_CMI_COMMIT:
        raise SystemExit("B2 request 004 merged CMI commit mismatch")
    if payload.get("package_zip_sha256") != PACKAGE_ZIP_SHA256:
        raise SystemExit("B2 request 004 package ZIP SHA mismatch")
    if payload.get("request_id_replacement") != {
        "old": OLD_REQUEST,
        "new": NEW_REQUEST,
        "count": 1,
    }:
        raise SystemExit("B2 request 004 request-id patch mismatch")
    locator = payload.get("mount_locator_patch", {})
    if locator.get("old_sha256") != OLD_LOCATOR_SHA256 or locator.get("new_sha256") != NEW_LOCATOR_SHA256 or locator.get("count") != 1:
        raise SystemExit("B2 request 004 locator patch mismatch")
    if payload.get("scientific_contract") != {
        "task_12_mode": "broad",
        "task_13_mode": "broad",
        "flow_target_source": "public_flow",
        "broad_rank_strata": ["study_accession", "name", "_unit_key", "_material_key"],
        "broad_raw_features": False,
        "broad_log1p_features": False,
    }:
        raise SystemExit("B2 request 004 scientific contract mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 120,
        "hard_timeout_minutes": 240,
        "max_active_runs": 1,
    }:
        raise SystemExit("B2 request 004 resource contract mismatch")
    if request.get("side_effects") != ["create one private Notebook version and start one CPU run"]:
        raise SystemExit("B2 request 004 side-effect contract mismatch")


def patch_source(source: bytes, request: dict[str, Any]) -> bytes:
    if len(source) != BASE_SIZE or sha256_bytes(source) != BASE_SOURCE_SHA256:
        raise SystemExit("B2 request 004 reconstructed source mismatch")
    if sha256_bytes(OLD_LOCATOR.encode()) != OLD_LOCATOR_SHA256:
        raise SystemExit("B2 request 004 old locator constant mismatch")
    if sha256_bytes(NEW_LOCATOR.encode()) != NEW_LOCATOR_SHA256:
        raise SystemExit("B2 request 004 new locator constant mismatch")
    text = source.decode("utf-8")
    validate_embedded_scientific_contract(text)
    if text.count(OLD_REQUEST) != 1 or text.count(OLD_LOCATOR) != 1:
        raise SystemExit("B2 request 004 patch occurrence mismatch")
    text = text.replace(OLD_REQUEST, NEW_REQUEST, 1)
    text = text.replace(OLD_LOCATOR, NEW_LOCATOR, 1)
    if "__REQUEST_ID__" in text or OLD_LOCATOR in text:
        raise SystemExit("B2 request 004 stale runtime source remains")
    compile(text, "script.py", "exec")
    locator_nodes = [
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef) and node.name == "locate_competition_data"
    ]
    if len(locator_nodes) != 1:
        raise SystemExit("B2 request 004 locator definition count mismatch")
    locator_text = ast.get_source_segment(text, locator_nodes[0]) or ""
    if '.rglob("sample_submission_part1.csv")' not in locator_text or ".iterdir()" in locator_text:
        raise SystemExit("B2 request 004 recursive mount correction missing")
    return text.encode("utf-8")


def main() -> int:
    args = parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    validate_request(request)
    patched = patch_source(args.source.read_bytes(), request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(patched)
    print(
        "CMI_FLU_B2_PATCH_V4 PASS "
        f"base_sha256={BASE_SOURCE_SHA256} patched_sha256={sha256_bytes(patched)} "
        f"request_id={REQUEST_ID} flow=broad rank_strata=study+population+unit+material locator=recursive"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
