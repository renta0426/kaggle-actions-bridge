"""Materialize and audit the frozen Poisoned Chalice Mellum transfer kernel.

This script is executed only in a credential-free GitHub Actions job.  It downloads
four allowlisted files from one immutable research commit, verifies their Git blob
identities, runs the pinned notebook builder, normalizes cell IDs, and proves that
the embedded prediction manifest contains no target labels or prior-model scores.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import ssl
import subprocess
import sys
import urllib.request
from urllib.parse import quote, urlparse


EXPECTED_REQUEST_ID = "20260903-poisoned-chalice-mellum-transfer-v1-001"
EXPECTED_TARGET = "renta0426/mellum-transfer-v1"
EXPECTED_RESEARCH_REPOSITORY = "renta0426/The-Poisoned-Chalice-of-LLM-Evaluation"
EXPECTED_RESEARCH_COMMIT = "9f6be68c27dc1f3326d68bed4e2abf80db893748"
EXPECTED_MODEL_ID = "JetBrains/Mellum-4b-base"
EXPECTED_MODEL_REVISION = "83cce2605fbdf6a3868627e9b0a5924e0072b94d"
EXPECTED_UPSTREAM_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
EXPECTED_UPSTREAM_SHA256 = "6e8559279e8ebd94ec8c25561c91d38c454a4e109ea0813b07864ff6a66d4068"
EXPECTED_ROWS = 2_000
EXPECTED_LANGUAGE_COUNTS = {
    "Go": 400,
    "Java": 400,
    "Python": 400,
    "Ruby": 400,
    "Rust": 400,
}
ALLOWED_RECORD_KEYS = {"sample_id", "content_sha256", "language", "sample_index"}
FORBIDDEN_RECORD_KEYS = {"label", "membership", "is_member", "lumia_score"}


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        RejectRedirects(),
    )


def _download_raw(url: str, maximum_bytes: int) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "raw.githubusercontent.com":
        raise RuntimeError("research source must use raw.githubusercontent.com over HTTPS")
    request = urllib.request.Request(url, headers={"User-Agent": "kaggle-actions-bridge/1"})
    with _opener().open(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected source response status: {response.status}")
        data = response.read(maximum_bytes + 1)
    if len(data) > maximum_bytes:
        raise RuntimeError(f"research source exceeds fixed byte budget: {url}")
    return data


def _git_blob_sha(data: bytes) -> str:
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def _validate_request(request: dict) -> None:
    if request.get("schema_version") != 1:
        raise RuntimeError("unexpected request schema")
    if request.get("request_id") != EXPECTED_REQUEST_ID:
        raise RuntimeError("unexpected request identity")
    if request.get("competition") != "poisoned-chalice-icse27":
        raise RuntimeError("unexpected competition")
    if request.get("operation") != "kernel_run" or request.get("target") != EXPECTED_TARGET:
        raise RuntimeError("unexpected operation or target")
    if request.get("prerequisite_kernel") != "renta0426/stage1-raw-fim-submission-v1":
        raise RuntimeError("unexpected prerequisite kernel")
    source = request.get("research_source") or {}
    if source.get("repository") != EXPECTED_RESEARCH_REPOSITORY:
        raise RuntimeError("unexpected research repository")
    if source.get("commit") != EXPECTED_RESEARCH_COMMIT:
        raise RuntimeError("unexpected research commit")
    resource = request.get("resource") or {}
    if resource != {
        "accelerator": "gpu",
        "machine_shape": "NvidiaTeslaT4",
        "expected_runtime_minutes": 60,
        "hard_timeout_minutes": 180,
        "max_active_runs": 1,
        "min_remaining_quota_hours": 4.0,
    }:
        raise RuntimeError("resource contract changed")
    if request.get("api_budget") != {
        "max_calls": 20,
        "poll_interval_seconds": 300,
        "max_pages": 2,
    }:
        raise RuntimeError("API budget changed")
    if request.get("side_effects") != [
        "create one private notebook version and start one T4 GPU run"
    ]:
        raise RuntimeError("side-effect allowlist changed")
    if request.get("automatic_compute_retries") != 0:
        raise RuntimeError("automatic compute retries must remain zero")
    if request.get("enable_internet") is not True:
        raise RuntimeError("Mellum source/model retrieval requires Internet")
    if request.get("competition_submission") is not False:
        raise RuntimeError("Mellum transfer must not submit to a competition")
    if request.get("artifact_retention_days") != 1:
        raise RuntimeError("artifact retention changed")


def _materialize_sources(request: dict, research_root: Path) -> dict[str, str]:
    source = request["research_source"]
    files = source.get("files")
    if not isinstance(files, list) or len(files) != 4:
        raise RuntimeError("research source file allowlist must contain exactly four files")
    expected_paths = {
        "scripts/build_mellum_transfer_notebook.py",
        "src/poisoned_chalice/stage2.py",
        "configs/mellum_transfer_v1.json",
        "experiments/pseudo-stage2-transfer-v1/transfer_sample_manifest.parquet",
    }
    actual_paths = {str(item.get("path")) for item in files}
    if actual_paths != expected_paths:
        raise RuntimeError("research source path allowlist changed")

    verified: dict[str, str] = {}
    for item in files:
        path = str(item["path"])
        expected_blob = str(item["git_blob_sha"])
        maximum_bytes = int(item["max_bytes"])
        if not re.fullmatch(r"[0-9a-f]{40}", expected_blob):
            raise RuntimeError(f"invalid Git blob pin: {path}")
        if maximum_bytes < 1 or maximum_bytes > 2_097_152:
            raise RuntimeError(f"invalid source byte budget: {path}")
        url = (
            "https://raw.githubusercontent.com/"
            f"{source['repository']}/{source['commit']}/{quote(path, safe='/')}"
        )
        data = _download_raw(url, maximum_bytes)
        actual_blob = _git_blob_sha(data)
        if actual_blob != expected_blob:
            raise RuntimeError(f"research Git blob mismatch: {path}")
        destination = research_root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        verified[path] = actual_blob
    return verified


def _prediction_records(notebook: dict) -> list[dict]:
    matches: list[list[dict]] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source_value = cell.get("source", "")
        source = "".join(source_value) if isinstance(source_value, list) else str(source_value)
        if "PREDICTION_MANIFEST = pd.DataFrame(" not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "PREDICTION_MANIFEST" for target in node.targets):
                continue
            if not isinstance(node.value, ast.Call) or not node.value.args:
                raise RuntimeError("prediction manifest assignment is not a DataFrame literal")
            value = ast.literal_eval(node.value.args[0])
            if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
                raise RuntimeError("prediction manifest literal has an unexpected shape")
            matches.append(value)
    if len(matches) != 1:
        raise RuntimeError(f"expected one prediction manifest literal, got {len(matches)}")
    return matches[0]


def _validate_and_normalize_notebook(
    notebook_path: Path,
    metadata_path: Path,
    stage2_source: bytes,
) -> dict:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells") or []
    if notebook.get("nbformat") != 4 or len(cells) != 7:
        raise RuntimeError("unexpected Mellum notebook structure")
    for index, cell in enumerate(cells):
        cell["id"] = f"pc-mellum-v1-{index:02d}"
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook["cells"] = cells

    code_sources: list[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source_value = cell.get("source", "")
        code_sources.append(
            "".join(source_value) if isinstance(source_value, list) else str(source_value)
        )
    expected_stage2 = stage2_source.decode("utf-8")
    if sum(source == expected_stage2 for source in code_sources) != 1:
        raise RuntimeError("generic Stage 2 source was not embedded exactly once")

    records = _prediction_records(notebook)
    if len(records) != EXPECTED_ROWS:
        raise RuntimeError("embedded Mellum cohort row count changed")
    if any(set(record) != ALLOWED_RECORD_KEYS for record in records):
        raise RuntimeError("embedded prediction record schema changed")
    if any(FORBIDDEN_RECORD_KEYS.intersection(record) for record in records):
        raise RuntimeError("target label or prior score crossed the notebook boundary")
    sample_ids = [str(record["sample_id"]) for record in records]
    hashes = [str(record["content_sha256"]) for record in records]
    indices = [int(record["sample_index"]) for record in records]
    languages = [str(record["language"]) for record in records]
    if len(set(sample_ids)) != EXPECTED_ROWS or len(set(hashes)) != EXPECTED_ROWS:
        raise RuntimeError("embedded Mellum IDs/hashes are not unique")
    if indices != list(range(EXPECTED_ROWS)):
        raise RuntimeError("embedded Mellum sample_index is not contiguous")
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
        raise RuntimeError("embedded content hash is malformed")
    language_counts = {language: languages.count(language) for language in set(languages)}
    if language_counts != EXPECTED_LANGUAGE_COUNTS:
        raise RuntimeError("embedded Mellum language counts changed")

    combined = "\n".join(code_sources)
    for marker in (
        f"MODEL_ID = {EXPECTED_MODEL_ID!r}",
        f"MODEL_REVISION = {EXPECTED_MODEL_REVISION!r}",
        f"UPSTREAM_COMMIT = {EXPECTED_UPSTREAM_COMMIT!r}",
        f"UPSTREAM_SHA256 = {EXPECTED_UPSTREAM_SHA256!r}",
        "scores, scored = fuse_membership_scores(raw_features, RUNTIME_CONFIG)",
        '"target_labels_embedded_in_gpu_notebook": False',
        '"target_labels_used_for_training_or_normalization": False',
        '"previous_model_scores_used": False',
        '"submission_created": False',
    ):
        if marker not in combined:
            raise RuntimeError(f"Mellum notebook contract marker missing: {marker}")
    for forbidden in (
        "kaggle competitions submit",
        "competitions submit",
        "kernels push",
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
    ):
        if forbidden in combined:
            raise RuntimeError(f"forbidden notebook operation present: {forbidden}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "id": EXPECTED_TARGET,
        "title": "Mellum Transfer V1",
        "code_file": "mellum-transfer-v1.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu", "membership-inference", "transfer", "mellum"],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    if metadata != expected_metadata:
        raise RuntimeError("generated Mellum kernel metadata changed")

    notebook_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "embedded_rows": len(records),
        "embedded_record_keys": sorted(ALLOWED_RECORD_KEYS),
        "embedded_label_fields": [],
        "language_counts": language_counts,
        "cell_ids": [cell["id"] for cell in cells],
    }


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("pandas", "pyarrow", "nbformat"):
        result[package] = importlib.metadata.version(package)
    return result


def materialize(request_path: Path, work_root: Path, bundle_root: Path) -> dict:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    _validate_request(request)
    research_root = work_root / "research"
    verified = _materialize_sources(request, research_root)

    builder = research_root / "scripts/build_mellum_transfer_notebook.py"
    subprocess.run(
        [sys.executable, str(builder)],
        cwd=research_root,
        check=True,
        timeout=180,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
        },
    )

    generated = research_root / "notebooks/experiments/mellum-transfer-v1"
    notebook_path = generated / "mellum-transfer-v1.ipynb"
    metadata_path = generated / "kernel-metadata.json"
    if not notebook_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("Mellum builder did not create the expected kernel files")
    stage2_path = research_root / "src/poisoned_chalice/stage2.py"
    notebook_audit = _validate_and_normalize_notebook(
        notebook_path,
        metadata_path,
        stage2_path.read_bytes(),
    )

    kernel_root = bundle_root / "kernel"
    kernel_root.mkdir(parents=True, exist_ok=True)
    destination_notebook = kernel_root / notebook_path.name
    destination_metadata = kernel_root / metadata_path.name
    shutil.copy2(notebook_path, destination_notebook)
    shutil.copy2(metadata_path, destination_metadata)

    notebook_sha256 = hashlib.sha256(destination_notebook.read_bytes()).hexdigest()
    metadata_sha256 = hashlib.sha256(destination_metadata.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "request_id": request["request_id"],
        "target": request["target"],
        "research_repository": request["research_source"]["repository"],
        "research_commit": request["research_source"]["commit"],
        "verified_source_blobs": verified,
        "kernel_files": {
            destination_notebook.name: {
                "sha256": notebook_sha256,
                "bytes": destination_notebook.stat().st_size,
            },
            destination_metadata.name: {
                "sha256": metadata_sha256,
                "bytes": destination_metadata.stat().st_size,
            },
        },
        "build_environment": {
            "python": platform.python_version(),
            "packages": _package_versions(),
        },
        "audit": notebook_audit,
        "target_labels_embedded": False,
        "competition_submission": False,
    }
    bundle_root.mkdir(parents=True, exist_ok=True)
    (bundle_root / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize(args.request, args.work_root, args.bundle_root)
    print(
        json.dumps(
            {
                "status": "materialized",
                "request_id": manifest["request_id"],
                "research_commit": manifest["research_commit"],
                "notebook_sha256": manifest["kernel_files"]["mellum-transfer-v1.ipynb"]["sha256"],
                "embedded_rows": manifest["audit"]["embedded_rows"],
                "target_labels_embedded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
