"""Fetch the exact private research sources for shadow TPU pilot v1.

This process accepts only RESEARCH_REPO_READ_TOKEN, removes it from the process
environment immediately, downloads four allowlisted files from one pinned private
commit through the GitHub Contents API, verifies Git blob SHA and byte budgets,
and writes them only to the runner-local work directory. It never executes
research code and has no Kaggle capability.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import ssl
import urllib.request
from urllib.parse import quote, urlparse

TOKEN_ENV = "RESEARCH_REPO_READ_TOKEN"
REQUEST_ID = "20260904-poisoned-chalice-shadow-tpu-pilot-v1-001"
RESEARCH_REPOSITORY = "renta0426/The-Poisoned-Chalice-of-LLM-Evaluation"
RESEARCH_COMMIT = "4e221829e37af88553e780488b27730cf167d6f8"
EXPECTED_FILES = {
    "src/poisoned_chalice/shadow_protocol.py": (
        "1c47b80696050aa2e5e7c62384617df61ecb80da",
        131_072,
    ),
    "src/poisoned_chalice/shadow_training.py": (
        "2fe3cf2079659ae10e1ebe0d5973f3ce96c26a03",
        131_072,
    ),
    "scripts/train_shadow_model.py": (
        "8afc9d4126c8d39ce8ba6f7ab11a5c95b0d07d2b",
        65_536,
    ),
    "configs/shadow_tpu_pilot_v1.json": (
        "6c95af7488e506642e07a89301cb2db08571ba94",
        32_768,
    ),
}
MAX_API_RESPONSE_BYTES = 1_000_000


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        RejectRedirects(),
    )


def _validate_request(request: dict) -> None:
    exact = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": "poisoned-chalice-icse27",
        "operation": "kernel_run",
        "target": "renta0426/shadow-tpu-pilot-v1",
        "research_repository": RESEARCH_REPOSITORY,
        "research_commit": RESEARCH_COMMIT,
        "automatic_compute_retries": 0,
        "enable_internet": False,
        "competition_submission": False,
        "select_as_final": False,
        "runner_local_private_material_retention_days": 0,
    }
    for key, value in exact.items():
        if request.get(key) != value:
            raise RuntimeError(f"shadow TPU request contract changed: {key}")
    observed = {
        str(item.get("path")): (
            str(item.get("git_blob_sha")),
            int(item.get("max_bytes")),
        )
        for item in request.get("research_files", [])
    }
    if observed != EXPECTED_FILES:
        raise RuntimeError("shadow TPU private research file allowlist changed")
    for name in ("private_materializer_blob_sha", "launcher_blob_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(request.get(name) or "")):
            raise RuntimeError(f"shadow TPU bridge blob pin malformed: {name}")


def _download(token: str, path: str, expected_blob: str, max_bytes: int) -> bytes:
    owner, repository = RESEARCH_REPOSITORY.split("/", 1)
    api_url = (
        f"https://api.github.com/repos/{owner}/{repository}/contents/"
        f"{quote(path, safe='/')}?ref={RESEARCH_COMMIT}"
    )
    parsed = urlparse(api_url)
    if parsed.scheme != "https" or parsed.hostname != "api.github.com":
        raise RuntimeError("unexpected private GitHub API origin")
    request = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "kaggle-actions-bridge/1",
        },
    )
    with _opener().open(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected private source response status: {response.status}")
        payload_bytes = response.read(MAX_API_RESPONSE_BYTES + 1)
    if len(payload_bytes) > MAX_API_RESPONSE_BYTES:
        raise RuntimeError("private source API response exceeds byte budget")
    payload = json.loads(payload_bytes)
    if payload.get("type") != "file" or payload.get("encoding") != "base64":
        raise RuntimeError("private source Contents payload has unexpected type")
    if payload.get("sha") != expected_blob:
        raise RuntimeError(f"private source Git blob mismatch: {path}")
    encoded = re.sub(r"\s+", "", str(payload.get("content") or ""))
    data = base64.b64decode(encoded, validate=True)
    if int(payload.get("size", -1)) != len(data):
        raise RuntimeError(f"private source size mismatch: {path}")
    if len(data) > max_bytes:
        raise RuntimeError(f"private source exceeds byte budget: {path}")
    return data


def materialize(request_path: Path, work_root: Path) -> dict[str, object]:
    token = os.environ.pop(TOKEN_ENV, "")
    if not token or len(token) < 20:
        raise RuntimeError("missing scoped private-repository read token")
    if os.environ.get(TOKEN_ENV):
        raise RuntimeError("private-repository token remained in process environment")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    _validate_request(request)
    if work_root.exists() and any(work_root.iterdir()):
        raise RuntimeError("private research work root already exists and is not empty")
    research_root = work_root / "research"
    for path, (blob, maximum) in EXPECTED_FILES.items():
        data = _download(token, path, blob, maximum)
        destination = research_root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    token = ""
    return {
        "request_id": REQUEST_ID,
        "research_commit": RESEARCH_COMMIT,
        "source_files": len(EXPECTED_FILES),
        "research_root": str(research_root),
        "research_code_executed": False,
        "kaggle_operation_performed": False,
        "repository_token_in_child_environment": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize(args.request, args.work_root)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
