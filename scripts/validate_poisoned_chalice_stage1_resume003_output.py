"""Verify Stage 1 cached-continuation request 003 output.

The established output verifier predates the two bridge-only retry requests.  The
continuation algorithm and output contract did not change; only the immutable
request ID and observed source-kernel version changed.  This adapter verifies
those two current provenance fields explicitly, then delegates every feature,
OOF, shard, submission, and clean-room check to the pinned established verifier.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from typing import Any


ACTUAL_REQUEST_ID = "20260903-poisoned-chalice-stage1-resume-003"
ACTUAL_SOURCE_KERNEL_VERSION = 3
LEGACY_REQUEST_ID = "20260903-poisoned-chalice-stage1-resume-001"
LEGACY_SOURCE_KERNEL_VERSION = 2
BASE_VERIFIER_BLOB = "50307d5cc4ce305cda95f69d4dbf354cc9d743a6"
EXPECTED_OUTPUT_FILES = {
    "run_manifest.json",
    "submission.csv",
    "feature_schema.json",
    "source_shards.json",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def load_base_verifier(path: Path, expected_blob: str):  # noqa: ANN202
    data = path.read_bytes()
    actual_blob = git_blob_sha(data)
    if expected_blob != BASE_VERIFIER_BLOB or actual_blob != BASE_VERIFIER_BLOB:
        raise RuntimeError("base output verifier Git blob mismatch")
    spec = importlib.util.spec_from_file_location("stage1_resume_base_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load base output verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "REQUEST_ID", None) != LEGACY_REQUEST_ID:
        raise RuntimeError("base verifier request provenance changed")
    return module


def validate_current_provenance(manifest: dict[str, Any]) -> None:
    exact = {
        "request_id": ACTUAL_REQUEST_ID,
        "source_kernel_version": ACTUAL_SOURCE_KERNEL_VERSION,
        "source_kernel": "renta0426/stage1-raw-fim-submission-v1",
        "source_dataset": "renta0426/stage1-raw-fim-submission-v1-output",
        "source_dataset_version": 1,
        "source_kernel_status": "error_after_complete_feature_extraction",
        "source_failure": "Expected 113 base features, got 111",
        "source_extraction_reused": True,
        "gpu_forward_passes": 0,
        "accelerator": "cpu",
    }
    for key, value in exact.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"request-003 provenance mismatch: {key}")


def validate_output(
    root: Path,
    *,
    base_verifier_path: Path,
    expected_base_verifier_blob: str,
    expected_sha256: str | None = None,
    github_output: Path | None = None,
) -> dict[str, Any]:
    base = load_base_verifier(base_verifier_path, expected_base_verifier_blob)

    manifest_path = base.one_file(root, "run_manifest.json")
    submission_path = base.one_file(root, "submission.csv")
    schema_path = base.one_file(root, "feature_schema.json")
    shards_path = base.one_file(root, "source_shards.json")
    allowed = {
        manifest_path.resolve(),
        submission_path.resolve(),
        schema_path.resolve(),
        shards_path.resolve(),
    }
    unexpected = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() not in allowed
    ]
    if unexpected:
        raise RuntimeError(
            "downloaded output contains non-allowlisted files: "
            f"{sorted(path.name for path in unexpected)}"
        )
    if {path.name for path in allowed} != EXPECTED_OUTPUT_FILES:
        raise RuntimeError("output file allowlist changed")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_current_provenance(manifest)

    # Request 002 and request 003 changed provenance only.  Normalize precisely
    # those two audited fields in-memory so the established verifier checks the
    # unchanged scientific and submission contract.  The original manifest is
    # retained for schema and digest identity checks and is never rewritten.
    normalized = copy.deepcopy(manifest)
    normalized["request_id"] = LEGACY_REQUEST_ID
    normalized["source_kernel_version"] = LEGACY_SOURCE_KERNEL_VERSION
    metrics = base.validate_manifest(normalized)
    base.validate_schema(schema_path, manifest)
    base.validate_source_shards(shards_path)
    digest = base.validate_submission(submission_path, expected_sha256)
    if digest != manifest.get("submission_sha256"):
        raise RuntimeError("submission SHA-256 differs from request-003 manifest")

    oof_auc = float(metrics["oof_auc"])
    if not math.isfinite(oof_auc):
        raise RuntimeError("OOF AUC is non-finite")
    result = {
        "status": "verified",
        "request_id": ACTUAL_REQUEST_ID,
        "source_kernel_version": ACTUAL_SOURCE_KERNEL_VERSION,
        "submission_path": str(submission_path.resolve()),
        "submission_sha256": digest,
        "submission_rows": 5_000,
        "feature_counts": manifest["feature_counts"],
        **metrics,
        "hidden_validation_labels_used": False,
        "public_leaderboard_tuning_used": False,
        "provenance_fields_normalized_for_base_verifier": [
            "request_id",
            "source_kernel_version",
        ],
    }
    if github_output is not None:
        with github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"submission_sha256={digest}\n")
            handle.write(f"submission_path={submission_path.resolve()}\n")
            handle.write(f"oof_auc={oof_auc:.12f}\n")
            handle.write(f"oof_tpr_1pct={float(metrics['oof_tpr_1pct']):.12f}\n")
            handle.write(f"visible_auc={float(metrics['visible_auc']):.12f}\n")
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-verifier", type=Path, required=True)
    parser.add_argument(
        "--expected-base-verifier-blob",
        default=BASE_VERIFIER_BLOB,
    )
    parser.add_argument("--expected-sha256")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    expected = args.expected_sha256
    if expected is not None and not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise SystemExit("expected submission SHA-256 is malformed")
    validate_output(
        args.output_dir.resolve(),
        base_verifier_path=args.base_verifier.resolve(),
        expected_base_verifier_blob=args.expected_base_verifier_blob,
        expected_sha256=expected,
        github_output=args.github_output,
    )


if __name__ == "__main__":
    main()
