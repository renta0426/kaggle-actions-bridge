#!/usr/bin/env python3
"""Fail-closed first-create launcher for STAGE2-EXTERNAL-TRANSFER-V1.

Read-only preflight is completed before exactly one Kaggle kernel push. The new
GPU run is not polled to completion here and no competition submission exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from kaggle_exact_identity import exact_metadata, exact_metadata_eventually, status_name

REQUEST_ID = "STAGE2-EXTERNAL-TRANSFER-V1-LAUNCH-001"
TASK_ID = "STAGE2-EXTERNAL-TRANSFER-V1"
EXPECTED_SCIENCE_SHA = "98d25516ba4558e4b5fa2b0b2baacb28be71db25"
TARGET = "renta0426/poisoned-chalice-stage2-external-transfer-v1"
TITLE = "Poisoned Chalice Stage2 External Transfer V1"
SOURCE = "renta0426/poisoned-chalice-stage2-deployment-validation-v1"
EXPECTED_SOURCE_SCRIPT_VERSION_ID = 349274389
EXPECTED_SOURCE_TASK = "STAGE2-DEPLOYMENT-VALIDATION-V1"
EXPECTED_SOURCE_MODEL = "bigcode/starcoder2-3b"
EXPECTED_SOURCE_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
ACCELERATOR = "NvidiaTeslaT4"
EXPECTED_TARGET_VERSION = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--science-dir", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def http_status(exc: BaseException) -> int | None:
    for value in (getattr(exc, "status", None), getattr(exc, "status_code", None), getattr(getattr(exc, "response", None), "status_code", None)):
        if type(value) is int and 100 <= value <= 599:
            return value
    return None


def load_request(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, value in {
        "request_id": REQUEST_ID,
        "task_id": TASK_ID,
        "execution_policy": "kaggle_native_capacity_v2",
        "automatic_compute_retries": 0,
        "operation": "create_private_kernel_once",
        "protected_environment": "kaggle-readonry",
    }.items():
        if data.get(key) != value:
            raise RuntimeError(f"request contract mismatch:{key}")
    science, target, source, side = data["science"], data["target"], data["source"], data["side_effect_budget"]
    if science.get("commit") != EXPECTED_SCIENCE_SHA:
        raise RuntimeError("science commit mismatch")
    expected_target = {
        "kernel_ref": TARGET, "title": TITLE, "expected_first_version": 1, "private": True,
        "accelerator": ACCELERATOR, "gpu": True, "internet": True,
    }
    if any(target.get(k) != v for k, v in expected_target.items()):
        raise RuntimeError("target contract mismatch")
    if source.get("kernel_ref") != SOURCE or source.get("expected_successful_script_version_id") != EXPECTED_SOURCE_SCRIPT_VERSION_ID:
        raise RuntimeError("source contract mismatch")
    if source.get("required_current_status") != "COMPLETE" or source.get("current_version_number_must_be_resolved_before_write") is not True:
        raise RuntimeError("source current-only guard missing")
    if source.get("task_id") != EXPECTED_SOURCE_TASK or source.get("model_id") != EXPECTED_SOURCE_MODEL or source.get("model_revision") != EXPECTED_SOURCE_REVISION:
        raise RuntimeError("source scientific identity mismatch")
    if side != {"kaggle_kernel_writes": 1, "competition_submissions": 0, "final_submission_selections": 0, "automatic_write_retries": 0}:
        raise RuntimeError("side-effect budget mismatch")
    return data


def verify_science_checkout(science_dir: Path) -> None:
    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=science_dir, text=True).strip()
    if observed != EXPECTED_SCIENCE_SHA:
        raise RuntimeError("checked-out Science SHA mismatch")
    for rel in (
        "scripts/build_stage2_external_transfer_v1_notebook.py",
        "configs/stage2_external_transfer_v1_execution_20260913.json",
        "scripts/run_stage2_external_transfer_v1.py",
    ):
        if not (science_dir / rel).is_file():
            raise RuntimeError(f"Science payload missing:{rel}")


def build_payload(science_dir: Path) -> tuple[Path, dict[str, Any]]:
    builder = science_dir / "scripts/build_stage2_external_transfer_v1_notebook.py"
    completed = subprocess.run([sys.executable, str(builder)], cwd=science_dir, capture_output=True, text=True, timeout=120, check=False)
    if completed.returncode != 0:
        raise RuntimeError("Science notebook builder failed")
    root = science_dir / "notebooks/experiments/poisoned-chalice-stage2-external-transfer-v1"
    meta_path = root / "kernel-metadata.json"
    nb_path = root / "poisoned-chalice-stage2-external-transfer-v1.ipynb"
    if not meta_path.is_file() or not nb_path.is_file():
        raise RuntimeError("built Kaggle payload missing")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "id": TARGET, "title": TITLE, "is_private": True, "enable_gpu": True, "enable_internet": True,
        "kernel_sources": [SOURCE], "competition_sources": [], "dataset_sources": [],
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(f"built metadata mismatch:{key}")
    low = nb_path.read_text(encoding="utf-8").casefold()
    if any(token in low for token in ("competition_submit(", "competitions submit")):
        raise RuntimeError("built notebook contains forbidden submission path")
    return root, {"science_sha": EXPECTED_SCIENCE_SHA, "notebook_sha256": sha256_file(nb_path), "metadata_sha256": sha256_file(meta_path)}


def authenticate():
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_"):
        raise RuntimeError("KAGGLE_API_TOKEN contract failed")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate(); return api


def source_preflight(api) -> dict[str, Any]:
    metadata = exact_metadata(api, SOURCE)
    if getattr(metadata, "ref", None) != SOURCE or getattr(metadata, "is_private", None) is not True:
        raise RuntimeError("source exact private identity mismatch")
    version = getattr(metadata, "current_version_number", None)
    if isinstance(version, bool) or version is None or int(version) < 1:
        raise RuntimeError("source current version unresolved")
    state = status_name(getattr(api.kernels_status(SOURCE), "status", None))
    if state != "COMPLETE":
        raise RuntimeError("source current version is not COMPLETE")
    observed_script_id = None
    for attr in ("current_version_id", "current_version_script_id", "script_version_id"):
        value = getattr(metadata, attr, None)
        if type(value) is int:
            observed_script_id = int(value); break
    if observed_script_id is not None and observed_script_id != EXPECTED_SOURCE_SCRIPT_VERSION_ID:
        raise RuntimeError("source current scriptVersionId mismatch")
    return {
        "kernel_ref": SOURCE, "current_version_number": int(version), "status": state,
        "expected_successful_script_version_id": EXPECTED_SOURCE_SCRIPT_VERSION_ID,
        "observed_script_version_id_if_exposed": observed_script_id,
    }


def verify_source_bundle_root(root: Path) -> dict[str, Any]:
    manifest_path, seal_path = root / "bundle_manifest.json", root / "SEALED.sha256"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise RuntimeError("source bundle manifest/seal missing")
    fields = seal_path.read_text(encoding="utf-8").strip().split()
    if not fields or fields[0] != sha256_file(manifest_path):
        raise RuntimeError("source bundle manifest seal mismatch")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "hr_metadata.json": m.get("hr", {}).get("metadata_sha256"),
        "hr_parameters.npz": m.get("hr", {}).get("parameters_sha256"),
        "gr_metadata.json": m.get("gr", {}).get("metadata_sha256"),
        "gr_parameters.npz": m.get("gr", {}).get("parameters_sha256"),
        "c1_source_model.joblib": m.get("c1", {}).get("sha256"),
        "hr_training_audit.json": m.get("hr", {}).get("training_audit_sha256"),
    }
    if any(not isinstance(v, str) or len(v) != 64 for v in expected.values()):
        raise RuntimeError("source bundle asset hashes incomplete")
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"source bundle asset mismatch:{name}")
    if m.get("task_id") != EXPECTED_SOURCE_TASK or m.get("model_id") != EXPECTED_SOURCE_MODEL or m.get("model_revision") != EXPECTED_SOURCE_REVISION:
        raise RuntimeError("source sealed bundle scientific identity mismatch")
    return {
        "task_id": m["task_id"], "model_id": m["model_id"], "model_revision": m["model_revision"],
        "bundle_manifest_sha256": sha256_file(manifest_path), "verified_asset_count": len(expected),
    }


def verify_current_source_bundle() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pc-source-output-") as tmp:
        out = Path(tmp)
        completed = subprocess.run(["kaggle", "kernels", "output", SOURCE, "-p", str(out), "-q", "-o"], capture_output=True, text=True, timeout=300, check=False)
        if completed.returncode != 0:
            raise RuntimeError("source current output read failed")
        manifests = list(out.rglob("bundle_manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError("source bundle manifest cardinality mismatch")
        return verify_source_bundle_root(manifests[0].parent)


def require_target_absent(api) -> None:
    try:
        exact_metadata(api, TARGET)
    except Exception as exc:
        if http_status(exc) != 404:
            raise RuntimeError("target absence not proven by exact endpoint") from exc
    else:
        raise RuntimeError("target already exists; first-create refused")
    recent = api.kernels_list(user="renta0426", sort_by="dateRun", page_size=100) or []
    if any(str(getattr(x, "ref", "")) == TARGET or str(getattr(x, "title", "")) == TITLE for x in recent):
        raise RuntimeError("target ref/title collision discovered")


def push_once(kernel_dir: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(kernel_dir), "--accelerator", ACCELERATOR, "--timeout", "180"],
        capture_output=True, text=True, timeout=210, check=False,
    )
    receipt = {
        "write_attempted": True, "write_count": 1, "automatic_write_retries": 0,
        "cli_return_code": int(completed.returncode),
        "stdout_bytes": len(completed.stdout.encode("utf-8", errors="replace")),
        "stderr_bytes": len(completed.stderr.encode("utf-8", errors="replace")),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8", errors="replace")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8", errors="replace")).hexdigest(),
    }
    if completed.returncode != 0:
        raise RuntimeError("Kaggle first-create push failed; retry prohibited")
    return receipt


def reconcile_new_target(api) -> dict[str, Any]:
    metadata = exact_metadata_eventually(api, TARGET, attempts=8, delay_seconds=3.0)
    if getattr(metadata, "ref", None) != TARGET or getattr(metadata, "is_private", None) is not True:
        raise RuntimeError("new target identity/private mismatch")
    if int(getattr(metadata, "current_version_number", -1)) != EXPECTED_TARGET_VERSION:
        raise RuntimeError("new target version mismatch")
    if getattr(metadata, "enable_gpu", None) is not True or getattr(metadata, "enable_internet", None) is not True:
        raise RuntimeError("new target runtime flags mismatch")
    state = status_name(getattr(api.kernels_status(TARGET), "status", None))
    return {"kernel_ref": TARGET, "version": 1, "status": state, "accelerator": ACCELERATOR, "private": True, "internet": True}


def main() -> int:
    a = parse_args(); request = load_request(a.request.resolve()); science_dir = a.science_dir.resolve()
    verify_science_checkout(science_dir); kernel_dir, payload = build_payload(science_dir)
    if a.self_test:
        print(json.dumps({"self_test": "PASS", "request_id": REQUEST_ID, "target": TARGET, "accelerator": ACCELERATOR, **payload}, sort_keys=True)); return 0
    api = authenticate()
    source_identity = source_preflight(api)
    source_bundle = verify_current_source_bundle()
    require_target_absent(api)
    print("STAGE2_EXT_PREWRITE_PASS " + json.dumps({"request_id": REQUEST_ID, "source": source_identity, "source_bundle": source_bundle, "target": TARGET, "accelerator": ACCELERATOR, **payload}, sort_keys=True))
    write = push_once(kernel_dir)
    target = reconcile_new_target(api)
    receipt = {
        "schema_version": 1, "request_id": REQUEST_ID, "task_id": TASK_ID,
        "execution_policy": request["execution_policy"], "source": source_identity, "source_bundle": source_bundle,
        "payload": payload, "write": write, "target": target,
        "competition_submission": False, "final_submission_selection": False, "continuous_monitoring": False,
    }
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("STAGE2_EXT_LAUNCH_CONFIRMED " + json.dumps(receipt, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
