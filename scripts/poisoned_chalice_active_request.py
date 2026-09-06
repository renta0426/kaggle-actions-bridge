#!/usr/bin/env python3
"""Stage and validate the one active Poisoned Chalice kernel-run request.

The GitHub Actions workflow is intentionally stable. New experiments update the
active request and a request-specific launcher instead of adding another YAML
workflow. This controller is credential-free and only reads exact public files
at immutable Git commits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import ssl
import time
import urllib.request

BRIDGE_REPOSITORY = "renta0426/kaggle-actions-bridge"
RESEARCH_REPOSITORY = "renta0426/The-Poisoned-Chalice-of-LLM-Evaluation"
REQUEST_PATH = "requests/poisoned-chalice-kernel-run-active.json"
LOCK_PATH = "requirements/kaggle-2.2.4.lock"
LAUNCHER_PREFIX = "runners/poisoned_chalice/"
USER_AGENT = "kaggle-actions-bridge/1"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REQUEST_ID_RE = re.compile(r"^20[0-9]{6}-poisoned-chalice-[a-z0-9-]+-[0-9]{3}$")
TARGET_RE = re.compile(r"^renta0426/[a-z0-9][a-z0-9-]{2,80}$")
LOCK_LINE_RE = re.compile(
    r"^[a-z0-9][a-z0-9._-]*==[A-Za-z0-9][A-Za-z0-9._+-]* --hash=sha256:[0-9a-f]{64}$"
)

TOP_LEVEL_KEYS = {
    "schema_version",
    "request_id",
    "competition",
    "operation",
    "target",
    "launcher_path",
    "launcher_blob_sha",
    "research_repository",
    "research_commit",
    "research_files",
    "resource",
    "api_budget",
    "side_effects",
    "persistent_outputs",
    "automatic_compute_retries",
    "enable_internet",
    "competition_submission",
    "select_as_final",
    "clean_room",
    "scientific_contract",
}
RESOURCE_KEYS = {
    "accelerator",
    "machine_shape",
    "expected_visible_gpu_count",
    "expected_runtime_minutes",
    "hard_timeout_minutes",
    "max_active_runs",
    "min_remaining_quota_hours",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _safe_relative(path: str, *, prefix: str | None = None) -> str:
    if not isinstance(path, str) or not path or "\\" in path:
        raise ValueError("invalid relative path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError(f"unsafe relative path: {path}")
    normalized = str(pure)
    if prefix is not None and not normalized.startswith(prefix):
        raise ValueError(f"path outside allowed prefix: {path}")
    return normalized


def _fetch(url: str, maximum: int, attempts: int = 3) -> bytes:
    if maximum < 1 or maximum > 1_000_000:
        raise ValueError("invalid byte budget")
    context = ssl.create_default_context()
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=30, context=context) as response:
                data = response.read(maximum + 1)
            if not data or len(data) > maximum:
                raise RuntimeError(f"file outside byte budget: {len(data)} > {maximum}")
            return data
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(attempt + 1)
    raise RuntimeError(f"bounded read failed: {last}")


def _raw(repository: str, sha: str, path: str, maximum: int) -> bytes:
    if not SHA_RE.fullmatch(sha):
        raise ValueError("immutable 40-hex source SHA required")
    safe = _safe_relative(path)
    url = f"https://raw.githubusercontent.com/{repository}/{sha}/{safe}"
    return _fetch(url, maximum)


def validate_request(request: dict[str, object]) -> None:
    if set(request) != TOP_LEVEL_KEYS:
        raise ValueError(f"active request fields changed: {sorted(set(request) ^ TOP_LEVEL_KEYS)}")
    if request.get("schema_version") != 1:
        raise ValueError("unsupported request schema")
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not REQUEST_ID_RE.fullmatch(request_id):
        raise ValueError("invalid request_id")
    if request.get("competition") != "poisoned-chalice-icse27":
        raise ValueError("competition changed")
    if request.get("operation") != "kernel_run":
        raise ValueError("only kernel_run is supported")
    target = request.get("target")
    if not isinstance(target, str) or not TARGET_RE.fullmatch(target):
        raise ValueError("invalid Kaggle target")

    launcher_path = _safe_relative(str(request.get("launcher_path") or ""), prefix=LAUNCHER_PREFIX)
    if not launcher_path.endswith(".py"):
        raise ValueError("launcher must be Python")
    launcher_blob = request.get("launcher_blob_sha")
    if not isinstance(launcher_blob, str) or not SHA_RE.fullmatch(launcher_blob):
        raise ValueError("invalid launcher blob SHA")

    if request.get("research_repository") != RESEARCH_REPOSITORY:
        raise ValueError("research repository is not allowlisted")
    research_commit = request.get("research_commit")
    if not isinstance(research_commit, str) or not SHA_RE.fullmatch(research_commit):
        raise ValueError("invalid research commit")
    files = request.get("research_files")
    if not isinstance(files, list) or not 1 <= len(files) <= 12:
        raise ValueError("invalid research_files count")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "git_blob_sha", "max_bytes"}:
            raise ValueError("invalid research file descriptor")
        path = _safe_relative(str(item["path"]))
        if not path.startswith(("src/", "scripts/", "experiments/", "configs/")):
            raise ValueError(f"research path not allowlisted: {path}")
        if path in seen:
            raise ValueError("duplicate research path")
        seen.add(path)
        blob = item["git_blob_sha"]
        maximum = item["max_bytes"]
        if not isinstance(blob, str) or not SHA_RE.fullmatch(blob):
            raise ValueError("invalid research blob SHA")
        if not isinstance(maximum, int) or not 1 <= maximum <= 262_144:
            raise ValueError("invalid research byte budget")

    resource = request.get("resource")
    if not isinstance(resource, dict) or set(resource) != RESOURCE_KEYS:
        raise ValueError("resource contract changed")
    if resource.get("accelerator") != "gpu" or resource.get("machine_shape") != "NvidiaTeslaT4":
        raise ValueError("this stable runner currently supports only Kaggle T4 GPU runs")
    if resource.get("expected_visible_gpu_count") != 2:
        raise ValueError("unexpected Kaggle T4 visible GPU contract")
    for key, low, high in (
        ("expected_runtime_minutes", 1, 120),
        ("hard_timeout_minutes", 1, 180),
        ("max_active_runs", 1, 2),
    ):
        value = resource.get(key)
        if not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"invalid {key}")
    quota = resource.get("min_remaining_quota_hours")
    if not isinstance(quota, (int, float)) or not 0.25 <= float(quota) <= 12.0:
        raise ValueError("invalid min_remaining_quota_hours")

    api_budget = request.get("api_budget")
    if not isinstance(api_budget, dict) or set(api_budget) != {"max_calls"}:
        raise ValueError("api_budget contract changed")
    if not isinstance(api_budget.get("max_calls"), int) or not 5 <= int(api_budget["max_calls"]) <= 100:
        raise ValueError("invalid API call budget")
    if request.get("side_effects") != ["create one private notebook version"]:
        raise ValueError("side effect allowlist changed")
    if request.get("automatic_compute_retries") != 0:
        raise ValueError("automatic compute retry is forbidden")
    if request.get("competition_submission") is not False or request.get("select_as_final") is not False:
        raise ValueError("submission/final-selection is outside this runner")
    if not isinstance(request.get("enable_internet"), bool):
        raise ValueError("enable_internet must be explicit")

    outputs = request.get("persistent_outputs")
    if not isinstance(outputs, list) or not 1 <= len(outputs) <= 32:
        raise ValueError("persistent output allowlist required")
    if len(set(outputs)) != len(outputs):
        raise ValueError("duplicate persistent output")
    for output in outputs:
        _safe_relative(str(output))
    if not isinstance(request.get("clean_room"), dict) or not isinstance(request.get("scientific_contract"), dict):
        raise ValueError("clean_room and scientific_contract must be objects")


def validate_lock(data: bytes) -> None:
    lines = [line.strip() for line in data.decode("utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) != 33 or not all(LOCK_LINE_RE.fullmatch(line) for line in lines):
        raise ValueError("Kaggle CLI lock validation failed")


def stage(source_sha: str, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    request_bytes = _raw(BRIDGE_REPOSITORY, source_sha, REQUEST_PATH, 65_536)
    request = json.loads(request_bytes)
    validate_request(request)

    launcher_path = str(request["launcher_path"])
    launcher = _raw(BRIDGE_REPOSITORY, source_sha, launcher_path, 262_144)
    if git_blob_sha(launcher) != request["launcher_blob_sha"]:
        raise ValueError("launcher blob mismatch")
    compile(launcher, launcher_path, "exec")

    lock = _raw(BRIDGE_REPOSITORY, source_sha, LOCK_PATH, 65_536)
    validate_lock(lock)

    (output / "request.json").write_bytes(request_bytes)
    (output / "launcher.py").write_bytes(launcher)
    (output / "kaggle-2.2.4.lock").write_bytes(lock)
    research_root = output / "research"
    research_root.mkdir(parents=True, exist_ok=True)
    for item in request["research_files"]:
        data = _raw(
            RESEARCH_REPOSITORY,
            str(request["research_commit"]),
            str(item["path"]),
            int(item["max_bytes"]),
        )
        if git_blob_sha(data) != item["git_blob_sha"]:
            raise ValueError(f"research blob mismatch: {item['path']}")
        destination = research_root / str(item["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    safe = {
        "request_id": request["request_id"],
        "target": request["target"],
        "research_commit": request["research_commit"],
        "research_files": len(request["research_files"]),
        "accelerator": request["resource"]["accelerator"],
        "machine_shape": request["resource"]["machine_shape"],
        "automatic_compute_retries": request["automatic_compute_retries"],
        "competition_submission": request["competition_submission"],
    }
    print("POISONED_CHALICE_ACTIVE_REQUEST_STAGE PASS " + json.dumps(safe, sort_keys=True))
    return safe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    stage(args.source_sha, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
