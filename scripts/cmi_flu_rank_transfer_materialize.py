#!/usr/bin/env python3
"""Materialize and validate the exact CMI-Flu rank-transfer Kaggle runner.

This helper is intentionally narrow. It reads two bridge files at an exact bridge
commit using the workflow's read-only GITHUB_TOKEN, reads the fixed public science
commit without forwarding that token cross-repository, verifies exact Git blobs,
and deterministically builds the native science runner twice.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import py_compile
import re
import ssl
import subprocess
import sys
import urllib.parse
import urllib.request
from typing import Any

BRIDGE_REPOSITORY = "renta0426/kaggle-actions-bridge"
REQUEST_PATH = "requests/cmi-flu-rank-transfer-001.json"
LOCK_PATH = "requirements/kaggle-2.2.4.lock"
REQUEST_ID = "20260904-cmi-flu-rank-transfer-001"
TARGET_COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-rank-transfer-20260904-001"
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_SOURCE_COMMIT = "9e9922806209241121274f9242a6cf09bf140d75"
EXPECTED_RANK_TRANSFER_BLOB = "d5e07cdd09d2eabdc935eb1733ec238e26ab4c17"
EXPECTED_BUNDLE_BUILDER_BLOB = "bad454639d9e95331961272df4d0f13a0d2bcb87"
EXPECTED_PHASE_BUILDER_BLOB = "9afdb52abfadb8e634cdb9864581aff1dfe29726"
EXPECTED_B21_CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
MAX_SCIENCE_FILES = 80
MIN_SCIENCE_FILES = 10
MAX_SCIENCE_BYTES = 2_000_000
MAX_HTTP_BYTES = 5_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-ref", required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    return parser.parse_args()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def _read_response(response: Any, *, label: str) -> bytes:
    data = response.read(MAX_HTTP_BYTES + 1)
    if len(data) > MAX_HTTP_BYTES:
        raise SystemExit(f"HTTP response exceeds byte budget: {label}")
    return data


def _json_get(url: str, *, token: str | None, label: str) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "kaggle-actions-bridge/1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(
            request, timeout=45, context=ssl.create_default_context()
        ) as response:
            return json.loads(_read_response(response, label=label))
    except Exception as error:
        raise SystemExit(f"GitHub read failed at {label}: {type(error).__name__}") from error


def _bridge_contents(path: str, ref: str, *, token: str) -> bytes:
    quoted = urllib.parse.quote(path, safe="/")
    payload = _json_get(
        f"https://api.github.com/repos/{BRIDGE_REPOSITORY}/contents/{quoted}?ref={ref}",
        token=token,
        label=f"bridge:{path}",
    )
    if not isinstance(payload, dict) or payload.get("encoding") != "base64":
        raise SystemExit(f"bridge content encoding mismatch: {path}")
    return base64.b64decode(payload["content"], validate=False)


def _public_science_json(url: str, *, label: str) -> Any:
    # Deliberately omit the bridge-scoped GITHUB_TOKEN. GitHub returns 404 when a
    # repository-scoped workflow token is forwarded to an unrelated public repo.
    return _json_get(url, token=None, label=label)


def _validate_request(request: dict[str, Any]) -> None:
    allowed = {
        "schema_version",
        "request_id",
        "competition",
        "operation",
        "target",
        "purpose",
        "science_repository",
        "science_source_commit",
        "rank_transfer_blob_sha",
        "bundle_builder_blob_sha",
        "phase_builder_blob_sha",
        "b21_config_blob_sha",
        "resource",
        "api_budget",
        "side_effects",
        "competition_submission_attempted",
        "automatic_compute_retries",
        "enable_internet",
        "rules_checked_at_utc",
    }
    if set(request) != allowed:
        raise SystemExit(f"manifest fields mismatch: {sorted(set(request) ^ allowed)}")
    if request["schema_version"] != 1 or request["request_id"] != REQUEST_ID:
        raise SystemExit("request identity mismatch")
    if (
        request["competition"] != TARGET_COMPETITION
        or request["target"] != TARGET_KERNEL
        or request["operation"] != "kernel_run"
    ):
        raise SystemExit("request target/operation mismatch")
    if (
        request["science_repository"] != SCIENCE_REPOSITORY
        or request["science_source_commit"] != SCIENCE_SOURCE_COMMIT
    ):
        raise SystemExit("science provenance mismatch")
    if request["resource"] != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 60,
        "hard_timeout_minutes": 180,
        "max_active_runs": 1,
    }:
        raise SystemExit("resource contract mismatch")
    if request["api_budget"] != {
        "max_calls": 20,
        "poll_interval_seconds": 900,
        "max_pages": 2,
    }:
        raise SystemExit("API budget mismatch")
    if (
        request["competition_submission_attempted"] is not False
        or request["automatic_compute_retries"] != 0
        or request["enable_internet"] is not False
    ):
        raise SystemExit("safety contract mismatch")


def _materialize_science(science_root: pathlib.Path) -> dict[str, str]:
    commit = _public_science_json(
        f"https://api.github.com/repos/{SCIENCE_REPOSITORY}/commits/{SCIENCE_SOURCE_COMMIT}",
        label="science:commit",
    )
    if not isinstance(commit, dict) or commit.get("sha") != SCIENCE_SOURCE_COMMIT:
        raise SystemExit("science commit identity mismatch")
    tree_sha = str(commit["commit"]["tree"]["sha"])
    tree = _public_science_json(
        f"https://api.github.com/repos/{SCIENCE_REPOSITORY}/git/trees/{tree_sha}?recursive=1",
        label="science:tree",
    )
    if not isinstance(tree, dict) or tree.get("truncated"):
        raise SystemExit("science tree unavailable or truncated")

    extras = {
        "configs/baseline_b021_robust.yaml",
        "scripts/build_phase_a_kaggle_bundle.py",
        "scripts/build_rank_transfer_kaggle_bundle.py",
    }
    wanted: list[tuple[str, str, int]] = []
    for item in tree.get("tree", []):
        if item.get("type") != "blob":
            continue
        path = str(item.get("path") or "")
        if (path.startswith("src/cmi_flu/") and path.endswith(".py")) or path in extras:
            wanted.append((path, str(item["sha"]), int(item.get("size") or 0)))

    total_bytes = sum(size for _, _, size in wanted)
    if not MIN_SCIENCE_FILES <= len(wanted) <= MAX_SCIENCE_FILES:
        raise SystemExit(f"unexpected science snapshot file count: {len(wanted)}")
    if total_bytes > MAX_SCIENCE_BYTES:
        raise SystemExit(f"science snapshot exceeds byte budget: {total_bytes}")

    blobs: dict[str, str] = {}
    for path, sha, size in wanted:
        payload = _public_science_json(
            f"https://api.github.com/repos/{SCIENCE_REPOSITORY}/git/blobs/{sha}",
            label=f"science:blob:{path}",
        )
        if not isinstance(payload, dict) or payload.get("encoding") != "base64":
            raise SystemExit(f"science blob encoding mismatch: {path}")
        data = base64.b64decode(payload["content"], validate=False)
        if len(data) != size or git_blob_sha(data) != sha:
            raise SystemExit(f"science blob content mismatch: {path}")
        output = science_root / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        blobs[path] = sha

    print(f"SCIENCE_SNAPSHOT files={len(wanted)} bytes={total_bytes}")
    return blobs


def _verify_science(request: dict[str, Any], blobs: dict[str, str]) -> None:
    checks = {
        "src/cmi_flu/rank_transfer.py": (
            "rank_transfer_blob_sha",
            EXPECTED_RANK_TRANSFER_BLOB,
        ),
        "scripts/build_rank_transfer_kaggle_bundle.py": (
            "bundle_builder_blob_sha",
            EXPECTED_BUNDLE_BUILDER_BLOB,
        ),
        "scripts/build_phase_a_kaggle_bundle.py": (
            "phase_builder_blob_sha",
            EXPECTED_PHASE_BUILDER_BLOB,
        ),
        "configs/baseline_b021_robust.yaml": (
            "b21_config_blob_sha",
            EXPECTED_B21_CONFIG_BLOB,
        ),
    }
    for path, (manifest_key, expected) in checks.items():
        if blobs.get(path) != expected or request[manifest_key] != expected:
            raise SystemExit(f"science blob provenance mismatch: {path}")


def _build_runner(science_root: pathlib.Path, output_root: pathlib.Path) -> pathlib.Path:
    builder = science_root / "scripts/build_rank_transfer_kaggle_bundle.py"
    runner_a = output_root / "runner-a.py"
    runner_b = output_root / "runner-b.py"
    command = [
        sys.executable,
        str(builder),
        "--repository-root",
        str(science_root),
        "--source-commit",
        SCIENCE_SOURCE_COMMIT,
    ]
    subprocess.run([*command, "--output", str(runner_a)], check=True)
    subprocess.run([*command, "--output", str(runner_b)], check=True)
    if runner_a.read_bytes() != runner_b.read_bytes():
        raise SystemExit("native runner is not deterministic")

    text = runner_a.read_text(encoding="utf-8")
    placeholder = 'REQUEST_ID = "__REQUEST_ID__"'
    replacement = f'REQUEST_ID = "{REQUEST_ID}"'
    if text.count(placeholder) != 1:
        raise SystemExit("runner request-id placeholder mismatch")
    text = text.replace(placeholder, replacement, 1)
    compile(text, str(runner_a), "exec")
    runner_a.write_text(text, encoding="utf-8")
    py_compile.compile(str(runner_a), doraise=True)

    required = (
        '"kernel_stage": "phase_a_rank_transfer_task11_task12"',
        "cmi_flu/rank_transfer.py",
        "CMI_FLU_RANK_TRANSFER_COMPLETE",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"generated runner missing contract tokens: {missing}")
    subprocess.run([sys.executable, str(runner_a), "--self-test"], check=True)

    package = re.search(r'^PACKAGE_ZIP_SHA256 = "([0-9a-f]{64})"$', text, re.M)
    if package is None:
        raise SystemExit("generated runner package SHA missing")
    print(
        "RANK_TRANSFER_RUNNER "
        f"script_sha256={hashlib.sha256(runner_a.read_bytes()).hexdigest()} "
        f"package_sha256={package.group(1)}"
    )
    return runner_a


def main() -> int:
    args = parse_args()
    bridge_ref = args.bridge_ref.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", bridge_ref):
        raise SystemExit("bridge ref must be an exact lowercase commit SHA")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    bridge_root = output_root / "bridge"
    science_root = output_root / "science"
    bridge_root.mkdir(exist_ok=True)
    science_root.mkdir(exist_ok=True)

    token = os.environ.get("GITHUB_READ_TOKEN", "")
    if not token:
        raise SystemExit("GITHUB_READ_TOKEN is required")
    request_data = _bridge_contents(REQUEST_PATH, bridge_ref, token=token)
    lock_data = _bridge_contents(LOCK_PATH, bridge_ref, token=token)
    request_path = bridge_root / REQUEST_PATH
    lock_path = bridge_root / LOCK_PATH
    request_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_bytes(request_data)
    lock_path.write_bytes(lock_data)

    request = json.loads(request_data)
    if not isinstance(request, dict):
        raise SystemExit("request manifest must be a JSON object")
    _validate_request(request)
    blobs = _materialize_science(science_root)
    _verify_science(request, blobs)
    runner = _build_runner(science_root, output_root)
    print(f"CMI_FLU_RANK_TRANSFER_MATERIALIZE PASS runner={runner.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
