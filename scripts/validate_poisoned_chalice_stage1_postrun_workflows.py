"""Credential-free validation for resumed Stage 1 post-run workflows."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


VERIFIER_BLOB = "50307d5cc4ce305cda95f69d4dbf354cc9d743a6"
STATUS_REQUEST = Path("requests/poisoned-chalice-stage1-resume-status-v1.json")
SUBMIT_REQUEST = Path("requests/poisoned-chalice-stage1-resume-submit-v1.json")


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def require(source: str, markers: tuple[str, ...], label: str) -> None:
    for marker in markers:
        if marker not in source:
            raise RuntimeError(f"{label} missing invariant: {marker}")


def validate_status_request(path: Path, verifier_blob: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "request_id": "20260903-poisoned-chalice-stage1-resume-status-001",
        "parent_request_id": "20260903-poisoned-chalice-stage1-resume-001",
        "competition": "poisoned-chalice-icse27",
        "operation": "kernel_status_output",
        "target": "renta0426/stage1-raw-fim-resume-v1",
        "expected_kernel_version": 1,
        "output_allowlist": [
            "run_manifest.json",
            "submission.csv",
            "feature_schema.json",
            "source_shards.json",
        ],
        "verifier_path": "scripts/validate_poisoned_chalice_stage1_resume_output.py",
        "verifier_blob_sha": verifier_blob,
        "api_budget": {"max_calls": 8, "max_pages": 1},
        "side_effects": [],
    }
    if payload != expected:
        raise RuntimeError("status request differs from exact contract")


def validate_submit_request(path: Path, verifier_blob: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fixed = {
        "schema_version": 1,
        "request_id": "20260903-poisoned-chalice-stage1-resume-submit-001",
        "parent_status_request_id": "20260903-poisoned-chalice-stage1-resume-status-001",
        "parent_run_request_id": "20260903-poisoned-chalice-stage1-resume-001",
        "competition": "poisoned-chalice-icse27",
        "operation": "competition_submit",
        "target_kernel": "renta0426/stage1-raw-fim-resume-v1",
        "expected_kernel_version": 1,
        "submission_file": "submission.csv",
        "submission_message": (
            "Stage1 raw+structure+FIM cached continuation; OOF AUC 0.6645; "
            "no Public-LB tuning"
        ),
        "verifier_path": "scripts/validate_poisoned_chalice_stage1_resume_output.py",
        "verifier_blob_sha": verifier_blob,
        "submission_call_limit": 1,
        "side_effects": ["create one competition submission"],
        "automatic_submission_retries": 0,
        "select_as_final": False,
        "public_leaderboard_tuning_used": False,
    }
    for key, value in fixed.items():
        if payload.get(key) != value:
            raise RuntimeError(f"submission request mismatch: {key}")
    if set(payload) != set(fixed) | {"submission_sha256"}:
        raise RuntimeError("submission request contains unknown fields")
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("submission_sha256") or "")):
        raise RuntimeError("submission request SHA-256 is malformed")


def validate(root: Path) -> dict:
    status_path = root / ".github/workflows/86-poisoned-chalice-stage1-resume-status.yml"
    submit_path = root / ".github/workflows/87-poisoned-chalice-stage1-resume-submit.yml"
    verifier_path = root / "scripts/validate_poisoned_chalice_stage1_resume_output.py"
    status = status_path.read_text(encoding="utf-8")
    submit = submit_path.read_text(encoding="utf-8")
    verifier_data = verifier_path.read_bytes()
    verifier = verifier_data.decode("utf-8")
    ast.parse(verifier, filename=str(verifier_path))
    verifier_blob = git_blob_sha(verifier_data)
    if verifier_blob != VERIFIER_BLOB:
        raise RuntimeError("output verifier Git blob changed")

    require(
        verifier,
        (
            '"hidden_validation_labels_used": False',
            '"public_leaderboard_tuning_used": False',
            '"validation_labels_used_for_fit_or_feature_selection": False',
            '"source_extraction_reused": True',
            '"gpu_forward_passes": 0',
            '"base": 113',
            '"structure_raw": 50',
            '"fim": 11',
            '"total": 174',
            "EXPECTED_OOF_AUC = 0.664524",
            "OOF_AUC_TOLERANCE = 0.002",
            "submission differs from approved SHA-256",
        ),
        "output verifier",
    )

    common = (
        "permissions: {}",
        "group: kaggle-api-global",
        "cancel-in-progress: false",
        "environment: kaggle-readonry",
        "TARGET_KERNEL: renta0426/stage1-raw-fim-resume-v1",
        'EXPECTED_KERNEL_VERSION: "1"',
        VERIFIER_BLOB,
        "requirements/kaggle-2.2.4.lock",
        "run_manifest\\.json|submission\\.csv|feature_schema\\.json|source_shards\\.json",
    )
    for label, source in (("status workflow", status), ("submission workflow", submit)):
        require(source, common, label)
        if source.count("environment: kaggle-readonry") != 1:
            raise RuntimeError(f"{label} protected Environment count changed")
        for forbidden in (
            "workflow_dispatch",
            "actions/checkout",
            "actions/upload-artifact",
            "actions/download-artifact",
            "actions/cache",
            "continue-on-error: true",
            "kernels push",
            "kernels delete",
            "kernels cancel",
            "datasets create",
            "datasets version",
        ):
            if forbidden in source:
                raise RuntimeError(f"{label} gained forbidden operation: {forbidden}")
        if re.search(r"^\s*sleep\s+", source, flags=re.MULTILINE):
            raise RuntimeError(f"{label} must not poll")

    secret_marker = "$" + "{{ secrets.KAGGLE_API_TOKEN }}"
    if status.count(secret_marker) != 2:
        raise RuntimeError("status Kaggle-secret step count changed")
    if submit.count(secret_marker) != 3:
        raise RuntimeError("submission Kaggle-secret step count changed")
    if status.count('"${WORKDIR}/venv/bin/kaggle" kernels output') != 1:
        raise RuntimeError("status output call count changed")
    if submit.count('"${WORKDIR}/venv/bin/kaggle" kernels output') != 1:
        raise RuntimeError("submission output call count changed")
    if "competitions submit" in status:
        raise RuntimeError("read-only status workflow gained submission capability")
    if submit.count('"${WORKDIR}/venv/bin/kaggle" competitions submit') != 1:
        raise RuntimeError("submission workflow must contain exactly one submission call")

    require(
        status,
        (
            "STAGE1_RESUME_STATUS_REQUEST PASS side_effects=none",
            "STAGE1_RESUME_KERNEL_COMPLETE PASS version=1 cpu=true private=true",
            "POISONED_CHALICE_STAGE1_RESUME_OUTPUT_VERIFIED",
            "submission_sha256=",
            "STAGE1_RESUME_STATUS_LOCAL_OUTPUT_REMOVED",
        ),
        "status workflow",
    )
    require(
        submit,
        (
            "STAGE1_RESUME_SUBMISSION_REQUEST PASS calls=1 retries=0",
            "STAGE1_RESUME_SHA_BOUND_OUTPUT PASS",
            "STAGE1_RESUME_COMPETITION_SUBMISSION PASS",
            "submit_calls=1 automatic_retries=0 final_selection=false",
            '"submission_call_limit": 1',
            '"automatic_submission_retries": 0',
            '"select_as_final": False',
            "STAGE1_RESUME_SUBMISSION_LOCAL_MATERIAL_REMOVED",
        ),
        "submission workflow",
    )

    status_request_path = root / STATUS_REQUEST
    if status_request_path.exists():
        validate_status_request(status_request_path, verifier_blob)
    submit_request_path = root / SUBMIT_REQUEST
    if submit_request_path.exists():
        validate_submit_request(submit_request_path, verifier_blob)

    result = {
        "status": "pass",
        "verifier_blob": verifier_blob,
        "status_side_effects": 0,
        "submission_calls": 1,
        "sha_bound": True,
        "automatic_submission_retries": 0,
        "final_selection": False,
        "status_request_present": status_request_path.exists(),
        "submission_request_present": submit_request_path.exists(),
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    validate(args.root.resolve())


if __name__ == "__main__":
    main()
