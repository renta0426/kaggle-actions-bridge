#!/usr/bin/env python3
"""Reconstruct the reviewed CMI-Flu B2.1 Kaggle runner from the public B2 payload.

The private CMI-Flu repository is intentionally not fetched from the public bridge.
Instead this builder starts from the already-approved public B2 004 source package,
applies the reviewed B2.1 scientific patch, adds the inert diagnostics module that
was present in the reviewed source tree, verifies the exact package SHA256, and
then applies only the previously reviewed recursive Kaggle mount correction plus
a request-id replacement to the runtime shell.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

BASE_REQUEST_ID = "20260903-cmi-flu-b2-004"
BASE_PACKAGE_SHA256 = "48ff4ed2eadec8b059b8d37fb677af1f249b6486bccfbd55dca2274cfc6f3dc3"
B21_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"
B21_PACKAGE_SHA256 = "87c317789b3b8fcbd5fce2ff8f74663d23c19d0fcdce0e05ba1e809ea7728cb2"
B21_PLACEHOLDER_RUNNER_BLOB = "f14179a9b82aa01ab1bb4731a0112a0aafce7c20"
DIAGNOSTICS_BLOB = "6dd5e9f69095b5db63fded3913f22c0041a30f2a"
FIXED_ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)
OLD_REQUEST = f'REQUEST_ID = "{BASE_REQUEST_ID}"'
PLACEHOLDER_REQUEST = 'REQUEST_ID = "__REQUEST_ID__"'

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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def literal_assignments(text: str) -> dict[str, Any]:
    tree = ast.parse(text)
    output: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in {"SOURCE_COMMIT", "PACKAGE_ZIP_SHA256", "PACKAGE_B64", "CONFIG_TEXT"}:
            output[target.id] = ast.literal_eval(node.value)
    return output


def deterministic_package_zip(package_root: Path) -> bytes:
    paths = sorted((package_root / "cmi_flu").rglob("*.py"))
    if not paths:
        raise SystemExit("no CMI-Flu package files found")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            relative = path.relative_to(package_root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def chunk_base64(data: bytes, width: int = 96) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return "\n".join(encoded[i : i + width] for i in range(0, len(encoded), width))


def assignment_source(text: str, name: str) -> str:
    tree = ast.parse(text)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1:
        raise SystemExit(f"assignment count mismatch for {name}: {len(matches)}")
    segment = ast.get_source_segment(text, matches[0])
    if not segment:
        raise SystemExit(f"cannot recover assignment source for {name}")
    return segment


def replace_assignment(text: str, name: str, replacement: str) -> str:
    old = assignment_source(text, name)
    if text.count(old) != 1:
        raise SystemExit(f"assignment source is not unique for {name}")
    return text.replace(old, replacement, 1)


def reconstruct_package(
    base_source: str,
    *,
    patch_script: Path,
    diagnostics_module: Path,
) -> tuple[bytes, str]:
    values = literal_assignments(base_source)
    if values.get("PACKAGE_ZIP_SHA256") != BASE_PACKAGE_SHA256:
        raise SystemExit("base B2 package declaration mismatch")
    encoded = values.get("PACKAGE_B64")
    config_text = values.get("CONFIG_TEXT")
    if not isinstance(encoded, str) or not isinstance(config_text, str):
        raise SystemExit("base B2 package/config assignments missing")
    package = base64.b64decode("".join(encoded.split()), validate=True)
    if sha256_bytes(package) != BASE_PACKAGE_SHA256:
        raise SystemExit("base B2 embedded package bytes mismatch")

    with tempfile.TemporaryDirectory(prefix="cmi-flu-b21-") as temporary:
        root = Path(temporary)
        src_root = root / "src"
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            if archive.testzip() is not None:
                raise SystemExit("base B2 package is corrupt")
            for member in archive.infolist():
                if member.is_dir():
                    continue
                relative = Path(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise SystemExit("unsafe member in B2 package")
                destination = src_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))

        configs = root / "configs"
        configs.mkdir(parents=True, exist_ok=True)
        (configs / "baseline_b02_taskwise.yaml").write_text(config_text, encoding="utf-8")
        diagnostics = diagnostics_module.read_bytes()
        if git_blob_sha(diagnostics) != DIAGNOSTICS_BLOB:
            raise SystemExit("diagnostics module blob mismatch")
        (src_root / "cmi_flu" / "diagnostics.py").write_bytes(diagnostics)

        patch = load_module(patch_script, "cmi_flu_b21_scientific_patch")
        patch.patch_models(root)
        patch.patch_evaluation(root)
        patch.patch_runner(root)
        patch.patch_configuration(root)
        patch.create_config(root)
        patch.validate_target_blobs(root)

        rebuilt = deterministic_package_zip(src_root)
        actual = sha256_bytes(rebuilt)
        if actual != B21_PACKAGE_SHA256:
            raise SystemExit(
                f"B2.1 package SHA mismatch: expected={B21_PACKAGE_SHA256} actual={actual}"
            )
        robust_config = (configs / "baseline_b021_robust.yaml").read_text(encoding="utf-8")
        return rebuilt, robust_config


def build_placeholder_runner(base_source: str, *, package: bytes, config_text: str) -> str:
    text = base_source
    if text.count(OLD_REQUEST) != 1:
        raise SystemExit("base B2 request-id occurrence mismatch")
    text = text.replace(OLD_REQUEST, PLACEHOLDER_REQUEST, 1)

    text = replace_assignment(text, "SOURCE_COMMIT", f'SOURCE_COMMIT = "{B21_SOURCE_COMMIT}"')
    text = replace_assignment(text, "PACKAGE_ZIP_SHA256", f'PACKAGE_ZIP_SHA256 = "{B21_PACKAGE_SHA256}"')
    text = replace_assignment(text, "PACKAGE_B64", f'PACKAGE_B64 = """{chunk_base64(package)}"""')
    text = replace_assignment(text, "CONFIG_TEXT", f"CONFIG_TEXT = {config_text!r}")

    replacements = {
        "Self-contained Kaggle runner for the CMI-Flu B2 taskwise compact baseline.":
            "Self-contained Kaggle runner for the CMI-Flu B2.1 robust-CV baseline.",
        'default=Path("/kaggle/working/cmi-flu-b2"),':
            'default=Path("/kaggle/working/cmi-flu-b21"),',
        'config_path = config_dir / "baseline_b02_taskwise.yaml"':
            'config_path = config_dir / "baseline_b021_robust.yaml"',
        '"kernel_stage": "b2_taskwise_compact",':
            '"kernel_stage": "b21_taskwise_robust",',
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise SystemExit(f"runtime compatibility anchor mismatch: {old}")
        text = text.replace(old, new, 1)
    compile(text, "b21-placeholder.py", "exec")
    return text


def apply_bridge_runtime_patches(text: str, *, request_id: str) -> str:
    if text.count(PLACEHOLDER_REQUEST) != 1:
        raise SystemExit("B2.1 placeholder request-id occurrence mismatch")
    if text.count(OLD_LOCATOR) != 1:
        raise SystemExit("B2.1 old Kaggle locator occurrence mismatch")
    text = text.replace(PLACEHOLDER_REQUEST, f'REQUEST_ID = "{request_id}"', 1)
    text = text.replace(OLD_LOCATOR, NEW_LOCATOR, 1)
    if "__REQUEST_ID__" in text or OLD_LOCATOR in text:
        raise SystemExit("stale B2.1 runtime patch anchor remains")
    compile(text, "b21-runtime.py", "exec")
    locator_nodes = [
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef) and node.name == "locate_competition_data"
    ]
    if len(locator_nodes) != 1:
        raise SystemExit("B2.1 locator definition count mismatch")
    locator = ast.get_source_segment(text, locator_nodes[0]) or ""
    if '.rglob("sample_submission_part1.csv")' not in locator or ".iterdir()" in locator:
        raise SystemExit("B2.1 recursive Kaggle mount correction missing")
    return text


def validate_request(request: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "request_id",
        "parent_request_id",
        "competition",
        "operation",
        "target",
        "purpose",
        "payload",
        "resource",
        "api_budget",
        "side_effects",
        "automatic_compute_retries",
        "enable_internet",
        "rules_checked_at_utc",
    }
    if set(request) != required:
        raise SystemExit(f"B2.1 request key set mismatch: {sorted(set(request) ^ required)}")
    if request["schema_version"] != 4:
        raise SystemExit("B2.1 request schema mismatch")
    if request["competition"] != "cmi-flu-first-prediction-challenge":
        raise SystemExit("B2.1 competition mismatch")
    if request["operation"] != "kernel_run":
        raise SystemExit("B2.1 operation mismatch")
    if request["automatic_compute_retries"] != 0 or request["enable_internet"] is not False:
        raise SystemExit("B2.1 retry/internet contract mismatch")
    payload = request["payload"]
    expected_payload = {
        "base_request_id": BASE_REQUEST_ID,
        "base_package_zip_sha256": BASE_PACKAGE_SHA256,
        "b21_source_commit": B21_SOURCE_COMMIT,
        "b21_package_zip_sha256": B21_PACKAGE_SHA256,
        "b21_placeholder_runner_blob": B21_PLACEHOLDER_RUNNER_BLOB,
        "scientific_patch": "scripts/cmi_flu_b21_scientific_patch.py",
        "diagnostics_module": "payloads/cmi-flu-b21-runtime/diagnostics.py",
        "selection_policy": "robust_v1",
        "submission_attempted": False,
    }
    if payload != expected_payload:
        raise SystemExit("B2.1 payload contract mismatch")
    if request["resource"] != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 120,
        "hard_timeout_minutes": 240,
        "max_active_runs": 1,
    }:
        raise SystemExit("B2.1 resource contract mismatch")
    if request["side_effects"] != ["create one private Notebook version and start one CPU run"]:
        raise SystemExit("B2.1 side-effect contract mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--scientific-patch", type=Path, required=True)
    parser.add_argument("--diagnostics-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    validate_request(request)
    base_source = args.base_source.read_text(encoding="utf-8")
    package, config = reconstruct_package(
        base_source,
        patch_script=args.scientific_patch,
        diagnostics_module=args.diagnostics_module,
    )
    placeholder = build_placeholder_runner(base_source, package=package, config_text=config)
    placeholder_blob = git_blob_sha(placeholder.encode("utf-8"))
    if placeholder_blob != B21_PLACEHOLDER_RUNNER_BLOB:
        raise SystemExit(
            f"B2.1 placeholder runner blob mismatch: expected={B21_PLACEHOLDER_RUNNER_BLOB} actual={placeholder_blob}"
        )
    final = apply_bridge_runtime_patches(placeholder, request_id=request["request_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(final, encoding="utf-8")
    print(
        "CMI_FLU_B21_BUILDER PASS "
        f"request_id={request['request_id']} package_sha256={B21_PACKAGE_SHA256} "
        f"placeholder_blob={placeholder_blob} runtime_sha256={sha256_bytes(final.encode())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
