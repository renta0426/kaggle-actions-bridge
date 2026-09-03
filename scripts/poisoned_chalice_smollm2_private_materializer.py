"""Materialize the frozen SmolLM2 transfer Notebook from one private research commit.

The only credential accepted here is a fine-grained read token scoped to the private
research repository.  The token is removed from the process environment before any
private research code is executed.  Exactly five allowlisted files are fetched via
the GitHub Contents API, verified by Git blob SHA and byte budget, and used only on
the GitHub-hosted runner.  Private source is never emitted to logs or artifacts.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import sys
import urllib.request
from urllib.parse import quote, urlparse

TOKEN_ENV = "RESEARCH_REPO_READ_TOKEN"
REQUEST_ID = "20260903-poisoned-chalice-smollm2-transfer-v1-001"
TARGET = "renta0426/smollm2-transfer-v1"
RESEARCH_REPOSITORY = "renta0426/The-Poisoned-Chalice-of-LLM-Evaluation"
RESEARCH_COMMIT = "253806607ddec32364c89c39dd6f946599085868"
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"
MODEL_REVISION = "effd688a12921b4cc83e3312b6feb579f70f9c71"
EXPECTED_FILES = {
    "scripts/build_smollm2_transfer_notebook.py": (
        "1dae05433081721d660dc2203273c66797878678",
        131_072,
    ),
    "src/poisoned_chalice/stage2.py": (
        "9c2086f3c73ec5998ac3b50d7a4e166f6b1b4443",
        131_072,
    ),
    "src/poisoned_chalice/stage2_v2.py": (
        "9121a34d0f9ea84531c405bf127f8d8846d4274d",
        65_536,
    ),
    "configs/smollm2_transfer_v1.json": (
        "f58e31ed3d64259bc57bdaa40b8cc75b43e8d5a1",
        32_768,
    ),
    "experiments/pseudo-stage2-transfer-v1/transfer_sample_manifest.parquet": (
        "dedfd34d43e53c158398ae3cc99ed508cbe37f66",
        1_048_576,
    ),
}
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


def _validate_request(request: dict) -> None:
    exact = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": "poisoned-chalice-icse27",
        "operation": "kernel_run",
        "target": TARGET,
        "research_repository": RESEARCH_REPOSITORY,
        "research_commit": RESEARCH_COMMIT,
        "target_model_id": MODEL_ID,
        "target_model_revision": MODEL_REVISION,
        "automatic_compute_retries": 0,
        "enable_internet": True,
        "competition_submission": False,
        "select_as_final": False,
        "runner_local_private_material_retention_days": 0,
    }
    for key, value in exact.items():
        if request.get(key) != value:
            raise RuntimeError(f"SmolLM2 request contract changed: {key}")
    if request.get("resource") != {
        "accelerator": "gpu",
        "machine_shape": "NvidiaTeslaT4",
        "expected_runtime_minutes": 90,
        "hard_timeout_minutes": 180,
        "max_active_runs": 1,
        "min_remaining_quota_hours": 3.0,
    }:
        raise RuntimeError("SmolLM2 resource contract changed")
    if request.get("api_budget") != {
        "max_calls": 60,
        "max_recent_kernels_inspected": 25,
        "max_pages": 2,
    }:
        raise RuntimeError("SmolLM2 API budget changed")
    if request.get("side_effects") != [
        "read five files from one private research commit",
        "create one private SmolLM2 Notebook version and start one T4 GPU run",
    ]:
        raise RuntimeError("SmolLM2 side-effect allowlist changed")
    observed = {
        str(item.get("path")): (
            str(item.get("git_blob_sha")),
            int(item.get("max_bytes")),
        )
        for item in request.get("research_files", [])
    }
    if observed != EXPECTED_FILES:
        raise RuntimeError("SmolLM2 private research file allowlist changed")
    clean = request.get("clean_room") or {}
    expected_clean = {
        "target_labels_embedded": False,
        "target_labels_used_for_training_or_normalization": False,
        "previous_model_scores_used": False,
        "hidden_stage1_validation_labels_used": False,
        "public_leaderboard_tuning_used": False,
        "competition_submission_created": False,
    }
    if clean != expected_clean:
        raise RuntimeError("SmolLM2 clean-room contract changed")
    for name in ("private_materializer_blob_sha", "launcher_blob_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(request.get(name) or "")):
            raise RuntimeError(f"SmolLM2 bridge blob pin malformed: {name}")


def _download_private_file(token: str, path: str, expected_blob: str, max_bytes: int) -> bytes:
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


def _prediction_records(notebook: dict) -> list[dict]:
    records = None
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        value = cell.get("source", "")
        source = "".join(value) if isinstance(value, list) else str(value)
        if "PREDICTION_MANIFEST = pd.DataFrame(" not in source:
            continue
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "PREDICTION_MANIFEST"
                for target in node.targets
            ):
                continue
            if not isinstance(node.value, ast.Call) or not node.value.args:
                raise RuntimeError("SmolLM2 prediction manifest is not a literal DataFrame")
            records = ast.literal_eval(node.value.args[0])
    if not isinstance(records, list):
        raise RuntimeError("SmolLM2 prediction manifest literal missing")
    return records


def _audit_bundle(kernel_dir: Path) -> dict[str, object]:
    notebook_path = kernel_dir / "smollm2-transfer-v1.ipynb"
    metadata_path = kernel_dir / "kernel-metadata.json"
    if not notebook_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("SmolLM2 builder output is incomplete")
    raw = notebook_path.read_bytes()
    if len(raw) > 2_000_000:
        raise RuntimeError("SmolLM2 Notebook exceeds fixed byte budget")
    notebook = json.loads(raw)
    if notebook.get("nbformat") != 4 or len(notebook.get("cells", [])) != 8:
        raise RuntimeError("SmolLM2 Notebook structure changed")
    if [cell.get("id") for cell in notebook["cells"]] != [
        f"pc-smollm2-v1-{index:02d}" for index in range(8)
    ]:
        raise RuntimeError("SmolLM2 stable cell IDs changed")
    records = _prediction_records(notebook)
    allowed = {"sample_id", "content_sha256", "language", "sample_index"}
    if len(records) != 2000 or any(set(row) != allowed for row in records):
        raise RuntimeError("target label or unexpected field crossed prediction boundary")
    if [int(row["sample_index"]) for row in records] != list(range(2000)):
        raise RuntimeError("SmolLM2 embedded sample ordering changed")
    joined = "\n".join(
        "".join(cell.get("source", ""))
        if isinstance(cell.get("source", ""), list)
        else str(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for marker in (
        MODEL_ID,
        MODEL_REVISION,
        "fuse_membership_scores_v2",
        "minkpp_length_residual",
        "logrank_length_residual",
        "membership_score_v2",
        '"weights_searched": False',
        '"target_labels_used_for_training_or_normalization": False',
        '"previous_model_scores_used": False',
        '"submission_created": False',
    ):
        if marker not in joined:
            raise RuntimeError(f"SmolLM2 scientific marker missing: {marker}")
    if "KAGGLE_API_TOKEN" in joined or "competitions submit" in joined:
        raise RuntimeError("SmolLM2 Notebook gained Kaggle submission capability")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "id": TARGET,
        "code_file": notebook_path.name,
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"SmolLM2 metadata changed: {key}")
    return {
        "cells": 8,
        "rows": 2000,
        "notebook_bytes": len(raw),
        "target_labels_embedded": False,
        "competition_submission": False,
    }


def materialize(request_path: Path, work_root: Path, bundle_root: Path) -> dict[str, object]:
    token = os.environ.pop(TOKEN_ENV, "")
    if not token or len(token) < 20:
        raise RuntimeError("missing scoped private-repository read token")
    if os.environ.get(TOKEN_ENV):
        raise RuntimeError("private-repository token remained in process environment")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    _validate_request(request)
    research_root = work_root / "research"
    for path, (blob, maximum) in EXPECTED_FILES.items():
        data = _download_private_file(token, path, blob, maximum)
        destination = research_root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    token = ""

    builder = research_root / "scripts/build_smollm2_transfer_notebook.py"
    result = subprocess.run(
        [sys.executable, str(builder)],
        cwd=research_root,
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("private SmolLM2 Notebook materialization failed")
    generated = research_root / "notebooks/experiments/smollm2-transfer-v1"
    audit = _audit_bundle(generated)
    bundle_root.mkdir(parents=True, exist_ok=False)
    for name in ("smollm2-transfer-v1.ipynb", "kernel-metadata.json"):
        shutil.copy2(generated / name, bundle_root / name)
    return {
        "request_id": REQUEST_ID,
        "research_commit": RESEARCH_COMMIT,
        "source_files": len(EXPECTED_FILES),
        "audit": audit,
        "repository_token_in_child_environment": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize(args.request, args.work_root, args.bundle_root)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
