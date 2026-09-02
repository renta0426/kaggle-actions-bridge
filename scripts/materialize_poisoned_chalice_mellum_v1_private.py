"""Materialize the frozen Mellum kernel from one private research commit.

This wrapper is invoked only inside the protected Environment. It removes the
read-only repository token from the process environment before importing or
executing any research code, fetches exactly four allowlisted files through the
GitHub Contents API, and delegates byte/blob/notebook audits to the public
materializer.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import re
import ssl
import urllib.request
from urllib.parse import quote, unquote, urlparse


TOKEN_ENV = "RESEARCH_REPO_READ_TOKEN"
EXPECTED_REQUEST_ID = "20260903-poisoned-chalice-mellum-transfer-v1-001"
EXPECTED_TARGET = "renta0426/mellum-transfer-v1"
EXPECTED_REPOSITORY = "renta0426/The-Poisoned-Chalice-of-LLM-Evaluation"
EXPECTED_COMMIT = "9f6be68c27dc1f3326d68bed4e2abf80db893748"
EXPECTED_MATERIALIZER_PATH = "scripts/materialize_poisoned_chalice_mellum_v1.py"
EXPECTED_MATERIALIZER_BLOB = "fd9c645c9ff9a128b8f3a8dfdadd7c937e6d62a1"
EXPECTED_LOADER_PATH = "scripts/materialize_poisoned_chalice_mellum_v1_private.py"
EXPECTED_FILES = {
    "scripts/build_mellum_transfer_notebook.py": (
        "095725f10279431a0982e7ebb38faa5b03a7754e",
        131_072,
    ),
    "src/poisoned_chalice/stage2.py": (
        "9c2086f3c73ec5998ac3b50d7a4e166f6b1b4443",
        131_072,
    ),
    "configs/mellum_transfer_v1.json": (
        "2fa4b6cbe346283c9047b9228f6764bb52cecdd7",
        32_768,
    ),
    "experiments/pseudo-stage2-transfer-v1/transfer_sample_manifest.parquet": (
        "dedfd34d43e53c158398ae3cc99ed508cbe37f66",
        2_097_152,
    ),
}
EXPECTED_PATHS = set(EXPECTED_FILES)
MAX_API_RESPONSE_BYTES = 4_000_000


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        RejectRedirects(),
    )


def _load_materializer(path: Path):  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("mellum_public_materializer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load public Mellum materializer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_request_manifest(request: dict) -> None:
    exact = {
        "schema_version": 1,
        "request_id": EXPECTED_REQUEST_ID,
        "competition": "poisoned-chalice-icse27",
        "operation": "kernel_run",
        "target": EXPECTED_TARGET,
        "prerequisite_kernel": "renta0426/stage1-raw-fim-submission-v1",
        "materializer_path": EXPECTED_MATERIALIZER_PATH,
        "materializer_blob_sha": EXPECTED_MATERIALIZER_BLOB,
        "private_source_loader_path": EXPECTED_LOADER_PATH,
        "automatic_compute_retries": 0,
        "enable_internet": True,
        "competition_submission": False,
        "runner_local_private_material_retention_days": 0,
    }
    for key, value in exact.items():
        if request.get(key) != value:
            raise RuntimeError(f"private-source request contract changed: {key}")
    loader_blob = str(request.get("private_source_loader_blob_sha") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", loader_blob):
        raise RuntimeError("private-source loader Git blob pin is malformed")
    if request.get("resource") != {
        "accelerator": "gpu",
        "machine_shape": "NvidiaTeslaT4",
        "expected_runtime_minutes": 60,
        "hard_timeout_minutes": 180,
        "max_active_runs": 1,
        "min_remaining_quota_hours": 4.0,
    }:
        raise RuntimeError("private-source resource contract changed")
    if request.get("api_budget") != {
        "max_calls": 20,
        "poll_interval_seconds": 300,
        "max_pages": 2,
    }:
        raise RuntimeError("private-source API budget changed")
    if request.get("side_effects") != [
        "read four files from one private research commit",
        "create one private notebook version and start one T4 GPU run",
    ]:
        raise RuntimeError("private-source side-effect allowlist changed")
    source = request.get("research_source") or {}
    if source.get("repository") != EXPECTED_REPOSITORY:
        raise RuntimeError("request research repository changed")
    if source.get("commit") != EXPECTED_COMMIT:
        raise RuntimeError("request research commit changed")
    observed = {
        str(item.get("path")): (
            str(item.get("git_blob_sha")),
            int(item.get("max_bytes")),
        )
        for item in source.get("files", [])
    }
    if observed != EXPECTED_FILES:
        raise RuntimeError("request research file allowlist or pin changed")


def _raw_identity(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "raw.githubusercontent.com"
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("unexpected private research source URL")
    parts = parsed.path.lstrip("/").split("/", 3)
    if len(parts) != 4:
        raise RuntimeError("malformed private research source URL")
    owner, repository, commit, encoded_path = parts
    full_repository = f"{owner}/{repository}"
    path = unquote(encoded_path)
    if full_repository != EXPECTED_REPOSITORY or commit != EXPECTED_COMMIT:
        raise RuntimeError("private research source repository or commit changed")
    if path not in EXPECTED_PATHS:
        raise RuntimeError("private research source path is not allowlisted")
    return full_repository, commit, path


def _authenticated_downloader(token: str, request_manifest: dict):
    _validate_request_manifest(request_manifest)
    by_path = {
        str(item["path"]): item
        for item in request_manifest["research_source"]["files"]
    }

    def download(url: str, maximum_bytes: int) -> bytes:
        repository, commit, path = _raw_identity(url)
        entry = by_path[path]
        expected_blob = str(entry["git_blob_sha"])
        declared_maximum = int(entry["max_bytes"])
        if maximum_bytes != declared_maximum:
            raise RuntimeError("private source byte budget differs from request")

        owner, repository_name = repository.split("/", 1)
        api_url = (
            f"https://api.github.com/repos/{owner}/{repository_name}/contents/"
            f"{quote(path, safe='/')}?ref={commit}"
        )
        parsed = urlparse(api_url)
        if parsed.scheme != "https" or parsed.hostname != "api.github.com":
            raise RuntimeError("unexpected GitHub Contents API origin")
        api_request = urllib.request.Request(
            api_url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "kaggle-actions-bridge/1",
            },
        )
        with _opener().open(api_request, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"unexpected private source response status: {response.status}"
                )
            response_data = response.read(MAX_API_RESPONSE_BYTES + 1)
        if len(response_data) > MAX_API_RESPONSE_BYTES:
            raise RuntimeError("private source API response exceeds byte budget")
        payload = json.loads(response_data)
        if payload.get("type") != "file" or payload.get("encoding") != "base64":
            raise RuntimeError("private source Contents API payload has unexpected type")
        if payload.get("sha") != expected_blob:
            raise RuntimeError("private source Contents API Git blob mismatch")
        encoded = re.sub(r"\s+", "", str(payload.get("content") or ""))
        data = base64.b64decode(encoded, validate=True)
        if int(payload.get("size", -1)) != len(data):
            raise RuntimeError("private source Contents API size mismatch")
        if len(data) > maximum_bytes:
            raise RuntimeError("private source exceeds fixed byte budget")
        return data

    return download


def materialize(
    *,
    request_path: Path,
    materializer_path: Path,
    work_root: Path,
    bundle_root: Path,
) -> dict:
    token = os.environ.pop(TOKEN_ENV, "")
    if not token or len(token) < 20:
        raise RuntimeError("missing scoped private-repository read token")
    if os.environ.get(TOKEN_ENV):
        raise RuntimeError("private-repository token remained in process environment")

    request_manifest = json.loads(request_path.read_text(encoding="utf-8"))
    _validate_request_manifest(request_manifest)
    module = _load_materializer(materializer_path)
    module._validate_request = _validate_request_manifest
    module._download_raw = _authenticated_downloader(token, request_manifest)
    try:
        result = module.materialize(request_path, work_root, bundle_root)
    finally:
        token = ""
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize(
        request_path=args.request,
        materializer_path=args.materializer,
        work_root=args.work_root,
        bundle_root=args.bundle_root,
    )
    print(
        json.dumps(
            {
                "status": "materialized",
                "request_id": manifest["request_id"],
                "research_commit": manifest["research_commit"],
                "source_files": len(manifest["verified_source_blobs"]),
                "embedded_rows": manifest["audit"]["embedded_rows"],
                "target_labels_embedded": False,
                "repository_token_in_child_environment": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
