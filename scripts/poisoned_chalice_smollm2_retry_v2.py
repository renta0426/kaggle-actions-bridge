"""Operational retry coordinator for the frozen SmolLM2 Stage2-v2 experiment.

Request 001 failed before Kaggle preflight because the protected environment did not
provide the scoped private-repository read token.  This coordinator preserves the
exact request-001 scientific payload and bridge implementations, adds a new immutable
request identity, and delegates materialization/launch only after translating the
retry metadata to the already-audited request-001 contract in runner-local memory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

REQUEST_ID = "20260904-poisoned-chalice-smollm2-transfer-v1-002"
PARENT_REQUEST_ID = "20260903-poisoned-chalice-smollm2-transfer-v1-001"
PRIOR_BRIDGE_RUN_ID = 33758422614
RETRY_REASON = "missing_scoped_private_repository_read_token_before_kaggle_preflight"
ORIGINAL_REQUEST_ID = "20260903-poisoned-chalice-smollm2-transfer-v1-001"
V1_LAUNCHER_BLOB_SHA = "9102feab89f1543c85546d69cd45eb54490cba63"
V1_MATERIALIZER_BLOB_SHA = "4c561d2cd852cdd8879a35c0fd9dfbb457e6760d"
RETRY_ONLY_FIELDS = {
    "parent_request_id",
    "prior_bridge_run_id",
    "retry_reason",
    "scientific_protocol_changed",
    "retry_coordinator_path",
    "retry_coordinator_blob_sha",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_request(request: dict) -> dict:
    normalized = {
        key: value for key, value in request.items() if key not in RETRY_ONLY_FIELDS
    }
    normalized["request_id"] = ORIGINAL_REQUEST_ID
    return normalized


def validate_request(
    request: dict,
    coordinator_path: Path,
    launcher_path: Path,
    materializer_path: Path,
) -> dict:
    if request.get("request_id") != REQUEST_ID:
        raise RuntimeError("SmolLM2 retry request ID changed")
    if request.get("parent_request_id") != PARENT_REQUEST_ID:
        raise RuntimeError("SmolLM2 retry parent request changed")
    if int(request.get("prior_bridge_run_id", -1)) != PRIOR_BRIDGE_RUN_ID:
        raise RuntimeError("SmolLM2 retry prior run changed")
    if request.get("retry_reason") != RETRY_REASON:
        raise RuntimeError("SmolLM2 retry reason changed")
    if request.get("scientific_protocol_changed") is not False:
        raise RuntimeError("SmolLM2 retry must not change scientific protocol")
    if request.get("retry_coordinator_path") != "scripts/poisoned_chalice_smollm2_retry_v2.py":
        raise RuntimeError("SmolLM2 retry coordinator path changed")
    expected_coordinator_blob = str(request.get("retry_coordinator_blob_sha") or "")
    if git_blob_sha(coordinator_path.read_bytes()) != expected_coordinator_blob:
        raise RuntimeError("SmolLM2 retry coordinator Git blob changed")
    if git_blob_sha(launcher_path.read_bytes()) != V1_LAUNCHER_BLOB_SHA:
        raise RuntimeError("SmolLM2 audited v1 launcher changed")
    if git_blob_sha(materializer_path.read_bytes()) != V1_MATERIALIZER_BLOB_SHA:
        raise RuntimeError("SmolLM2 audited v1 materializer changed")

    normalized = normalized_request(request)
    launcher = load_module(launcher_path, "pc_smollm2_launcher_v1")
    launcher.validate_request(normalized, launcher_path, materializer_path)
    launcher.validate_static(launcher_path)
    materializer = load_module(materializer_path, "pc_smollm2_materializer_v1")
    materializer._validate_request(normalized)
    return normalized


def write_normalized_request(normalized: dict, directory: Path) -> Path:
    path = directory / "request-v1-compatible.json"
    path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def materialize(
    request: dict,
    coordinator_path: Path,
    launcher_path: Path,
    materializer_path: Path,
    work_root: Path,
    bundle_root: Path,
) -> None:
    normalized = validate_request(
        request, coordinator_path, launcher_path, materializer_path
    )
    if not os.environ.get("RESEARCH_REPO_READ_TOKEN"):
        raise RuntimeError(
            "RESEARCH_REPO_READ_TOKEN is unavailable; add the scoped read-only secret "
            "before approving this protected retry"
        )
    materializer = load_module(materializer_path, "pc_smollm2_materializer_v1_exec")
    request_dir = work_root / "retry-request"
    request_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = write_normalized_request(normalized, request_dir)
    manifest = materializer.materialize(normalized_path, work_root / "private", bundle_root)
    if manifest.get("research_commit") != normalized["research_commit"]:
        raise RuntimeError("SmolLM2 materializer research commit mismatch")
    if int(manifest.get("source_files", -1)) != 5:
        raise RuntimeError("SmolLM2 materializer source-file count mismatch")
    print(
        "SMOLLM2_RETRY_MATERIALIZE PASS request=002 private_files=5 "
        "scientific_protocol_changed=false"
    )


def validate_bundle(
    request: dict,
    coordinator_path: Path,
    launcher_path: Path,
    materializer_path: Path,
    bundle_root: Path,
) -> None:
    validate_request(request, coordinator_path, launcher_path, materializer_path)
    launcher = load_module(launcher_path, "pc_smollm2_launcher_v1_bundle")
    launcher.validate_bundle(bundle_root)
    print(
        "SMOLLM2_RETRY_BUNDLE PASS request=002 target_labels=false "
        "submission=false"
    )


def execute(
    request: dict,
    coordinator_path: Path,
    launcher_path: Path,
    materializer_path: Path,
    bundle_root: Path,
    kaggle_bin: Path,
) -> None:
    validate_request(request, coordinator_path, launcher_path, materializer_path)
    launcher = load_module(launcher_path, "pc_smollm2_launcher_v1_exec")
    launcher.validate_bundle(bundle_root)
    launcher.execute(kaggle_bin, bundle_root)
    print(
        "SMOLLM2_RETRY_EXECUTION PASS request=002 target=renta0426/smollm2-transfer-v1 "
        "accelerator=gpu automatic_compute_retries=0 competition_submission=false"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--validate-bundle", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--coordinator", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--kaggle-bin", type=Path)
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    if args.static:
        validate_request(
            request, args.coordinator, args.launcher, args.materializer
        )
        print(
            "SMOLLM2_RETRY_STATIC PASS request=002 resource_class=gpu "
            "scientific_protocol_changed=false write_calls=1 automatic_retries=0 submissions=0"
        )
        return
    if args.bundle_root is None:
        raise SystemExit("selected mode requires --bundle-root")
    if args.materialize:
        if args.work_root is None:
            raise SystemExit("--materialize requires --work-root")
        materialize(
            request,
            args.coordinator,
            args.launcher,
            args.materializer,
            args.work_root,
            args.bundle_root,
        )
        return
    if args.validate_bundle:
        validate_bundle(
            request,
            args.coordinator,
            args.launcher,
            args.materializer,
            args.bundle_root,
        )
        return
    if args.kaggle_bin is None:
        raise SystemExit("--execute requires --kaggle-bin")
    execute(
        request,
        args.coordinator,
        args.launcher,
        args.materializer,
        args.bundle_root,
        args.kaggle_bin,
    )


if __name__ == "__main__":
    main()
