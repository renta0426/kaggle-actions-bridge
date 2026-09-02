"""Credential-free validation for Stage 1 resume003 post-run workflows."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re


STATUS_REQUEST_PATH = Path(
    "requests/poisoned-chalice-stage1-resume003-status-v1.json"
)
BASE_VERIFIER_PATH = Path(
    "scripts/validate_poisoned_chalice_stage1_resume_output.py"
)
PROVENANCE_VERIFIER_PATH = Path(
    "scripts/validate_poisoned_chalice_stage1_resume003_output.py"
)
STATUS_WORKFLOW_PATH = Path(
    ".github/workflows/118-poisoned-chalice-stage1-resume003-status.yml"
)
SUBMIT_WORKFLOW_PATH = Path(
    ".github/workflows/119-poisoned-chalice-stage1-resume003-submit.yml"
)
OPTIONAL_SUBMIT_REQUEST_PATH = Path(
    "requests/poisoned-chalice-stage1-resume003-submit-v1.json"
)
BASE_VERIFIER_BLOB = "50307d5cc4ce305cda95f69d4dbf354cc9d743a6"
PROVENANCE_VERIFIER_BLOB = "1f5f3790e346e270249dbce2aae4494274979731"
STATUS_WORKFLOW_BLOB = "06f52e124a3461666628cf07515c84bf2869112e"
SUBMIT_WORKFLOW_BLOB = "37a08b4af142d3e35474fddd144859c53c909790"
SECRET_NAMES = (
    "KAGGLE_API_TOKEN",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "GH_TOKEN",
    "GITHUB_PAT",
    "SSH_PRIVATE_KEY",
)


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def require_markers(source: str, markers: tuple[str, ...], label: str) -> None:
    for marker in markers:
        if marker not in source:
            raise RuntimeError(f"{label} invariant missing: {marker}")


def literal_constants(tree: ast.Module) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        try:
            literal = ast.literal_eval(value)
        except (TypeError, ValueError):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                values[target.id] = literal
    return values


def validate_status_request(payload: dict) -> None:
    expected = {
        "schema_version": 1,
        "request_id": "20260903-poisoned-chalice-stage1-resume003-status-001",
        "parent_request_id": "20260903-poisoned-chalice-stage1-resume-003",
        "competition": "poisoned-chalice-icse27",
        "operation": "kernel_status_output",
        "target": "renta0426/stage1-raw-fim-resume-v1",
        "expected_kernel_version": 1,
        "source_kernel_version": 3,
        "output_allowlist": [
            "run_manifest.json",
            "submission.csv",
            "feature_schema.json",
            "source_shards.json",
        ],
        "base_verifier_path": str(BASE_VERIFIER_PATH),
        "base_verifier_blob_sha": BASE_VERIFIER_BLOB,
        "provenance_verifier_path": str(PROVENANCE_VERIFIER_PATH),
        "provenance_verifier_blob_sha": PROVENANCE_VERIFIER_BLOB,
        "api_budget": {"max_calls": 8, "max_pages": 1},
        "side_effects": [],
    }
    if payload != expected:
        raise RuntimeError("resume003 status request differs from exact contract")


def validate_provenance_verifier(data: bytes) -> None:
    if git_blob_sha(data) != PROVENANCE_VERIFIER_BLOB:
        raise RuntimeError("resume003 provenance verifier Git blob changed")
    source = data.decode("utf-8")
    tree = ast.parse(source, filename=str(PROVENANCE_VERIFIER_PATH))
    values = literal_constants(tree)
    expected = {
        "ACTUAL_REQUEST_ID": "20260903-poisoned-chalice-stage1-resume-003",
        "ACTUAL_SOURCE_KERNEL_VERSION": 3,
        "LEGACY_REQUEST_ID": "20260903-poisoned-chalice-stage1-resume-001",
        "LEGACY_SOURCE_KERNEL_VERSION": 2,
        "BASE_VERIFIER_BLOB": BASE_VERIFIER_BLOB,
        "EXPECTED_OUTPUT_FILES": {
            "run_manifest.json",
            "submission.csv",
            "feature_schema.json",
            "source_shards.json",
        },
    }
    for key, value in expected.items():
        if values.get(key) != value:
            raise RuntimeError(f"resume003 provenance verifier changed: {key}")
    require_markers(
        source,
        (
            'normalized["request_id"] = LEGACY_REQUEST_ID',
            'normalized["source_kernel_version"] = LEGACY_SOURCE_KERNEL_VERSION',
            "metrics = base.validate_manifest(normalized)",
            "base.validate_schema(schema_path, manifest)",
            "base.validate_source_shards(shards_path)",
            "digest = base.validate_submission(submission_path, expected_sha256)",
            '"provenance_fields_normalized_for_base_verifier"',
            '"request_id",',
            '"source_kernel_version",',
        ),
        "resume003 provenance verifier",
    )
    if "subprocess" in source or "urllib" in source or "requests" in source:
        raise RuntimeError("resume003 provenance verifier gained external execution or network access")
    if any(name in source for name in SECRET_NAMES):
        raise RuntimeError("resume003 provenance verifier references a credential")


def validate_workflow_blobs(root: Path) -> tuple[str, str]:
    base_data = (root / BASE_VERIFIER_PATH).read_bytes()
    provenance_data = (root / PROVENANCE_VERIFIER_PATH).read_bytes()
    status_data = (root / STATUS_WORKFLOW_PATH).read_bytes()
    submit_data = (root / SUBMIT_WORKFLOW_PATH).read_bytes()
    if git_blob_sha(base_data) != BASE_VERIFIER_BLOB:
        raise RuntimeError("base output verifier Git blob changed")
    validate_provenance_verifier(provenance_data)
    if git_blob_sha(status_data) != STATUS_WORKFLOW_BLOB:
        raise RuntimeError("resume003 status workflow Git blob changed")
    if git_blob_sha(submit_data) != SUBMIT_WORKFLOW_BLOB:
        raise RuntimeError("resume003 submission workflow Git blob changed")
    return status_data.decode("utf-8"), submit_data.decode("utf-8")


def validate_status_workflow(source: str) -> None:
    require_markers(
        source,
        (
            "permissions: {}",
            "group: kaggle-api-global",
            "cancel-in-progress: false",
            "environment: kaggle-readonry",
            str(STATUS_REQUEST_PATH),
            "20260903-poisoned-chalice-stage1-resume003-status-001",
            "20260903-poisoned-chalice-stage1-resume-003",
            "TARGET_KERNEL: renta0426/stage1-raw-fim-resume-v1",
            'EXPECTED_KERNEL_VERSION: "1"',
            BASE_VERIFIER_BLOB,
            PROVENANCE_VERIFIER_BLOB,
            "STAGE1_RESUME003_KERNEL_COMPLETE PASS version=1 cpu=true private=true internet=false",
            "POISONED_CHALICE_STAGE1_RESUME003_OUTPUT_VERIFIED side_effects=none",
            "submission_sha256=",
            "oof_auc=",
            "oof_tpr_1pct=",
            "visible_auc=",
            "STAGE1_RESUME003_STATUS_LOCAL_OUTPUT_REMOVED",
        ),
        "resume003 status workflow",
    )
    if source.count("${{ secrets.KAGGLE_API_TOKEN }}") != 2:
        raise RuntimeError("resume003 status Kaggle-token step count changed")
    if source.count("environment: kaggle-readonry") != 1:
        raise RuntimeError("resume003 status protected Environment count changed")
    if source.count('"${WORKDIR}/venv/bin/kaggle" kernels output') != 1:
        raise RuntimeError("resume003 status output call count changed")
    if "competitions submit" in source:
        raise RuntimeError("read-only resume003 status workflow gained submission capability")


def validate_submit_workflow(source: str) -> None:
    require_markers(
        source,
        (
            "permissions: {}",
            "group: kaggle-api-global",
            "cancel-in-progress: false",
            "environment: kaggle-readonry",
            str(OPTIONAL_SUBMIT_REQUEST_PATH),
            "20260903-poisoned-chalice-stage1-resume003-submit-001",
            "20260903-poisoned-chalice-stage1-resume003-status-001",
            "20260903-poisoned-chalice-stage1-resume-003",
            "TARGET_KERNEL: renta0426/stage1-raw-fim-resume-v1",
            'EXPECTED_KERNEL_VERSION: "1"',
            BASE_VERIFIER_BLOB,
            PROVENANCE_VERIFIER_BLOB,
            '"submission_call_limit": 1',
            '"automatic_submission_retries": 0',
            '"select_as_final": False',
            "STAGE1_RESUME003_SHA_BOUND_OUTPUT PASS",
            "STAGE1_RESUME003_COMPETITION_SUBMISSION PASS",
            "submit_calls=1 automatic_retries=0 final_selection=false",
            "STAGE1_RESUME003_SUBMISSION_LOCAL_MATERIAL_REMOVED",
        ),
        "resume003 submission workflow",
    )
    if source.count("${{ secrets.KAGGLE_API_TOKEN }}") != 3:
        raise RuntimeError("resume003 submission Kaggle-token step count changed")
    if source.count("environment: kaggle-readonry") != 1:
        raise RuntimeError("resume003 submission protected Environment count changed")
    if source.count('"${WORKDIR}/venv/bin/kaggle" kernels output') != 1:
        raise RuntimeError("resume003 submission output call count changed")
    if source.count('"${WORKDIR}/venv/bin/kaggle" competitions submit') != 1:
        raise RuntimeError("resume003 submission call count must equal one")


def validate_optional_submit_request(path: Path) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    fixed = {
        "schema_version": 1,
        "request_id": "20260903-poisoned-chalice-stage1-resume003-submit-001",
        "parent_status_request_id": "20260903-poisoned-chalice-stage1-resume003-status-001",
        "parent_run_request_id": "20260903-poisoned-chalice-stage1-resume-003",
        "competition": "poisoned-chalice-icse27",
        "operation": "competition_submit",
        "target_kernel": "renta0426/stage1-raw-fim-resume-v1",
        "expected_kernel_version": 1,
        "source_kernel_version": 3,
        "submission_file": "submission.csv",
        "submission_message": "Stage1 raw+structure+FIM cached continuation; OOF AUC 0.6645; no Public-LB tuning",
        "base_verifier_path": str(BASE_VERIFIER_PATH),
        "base_verifier_blob_sha": BASE_VERIFIER_BLOB,
        "provenance_verifier_path": str(PROVENANCE_VERIFIER_PATH),
        "provenance_verifier_blob_sha": PROVENANCE_VERIFIER_BLOB,
        "submission_call_limit": 1,
        "side_effects": ["create one competition submission"],
        "automatic_submission_retries": 0,
        "select_as_final": False,
        "public_leaderboard_tuning_used": False,
    }
    for key, value in fixed.items():
        if payload.get(key) != value:
            raise RuntimeError(f"optional resume003 submission request mismatch: {key}")
    if set(payload) != set(fixed) | {"submission_sha256"}:
        raise RuntimeError("optional resume003 submission request contains unknown fields")
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("submission_sha256") or "")):
        raise RuntimeError("optional resume003 submission request SHA-256 is malformed")


def validate_common_workflow_guards(status: str, submit: str) -> None:
    for label, source in (("status", status), ("submission", submit)):
        for forbidden in (
            "workflow_dispatch",
            "pull_request_target",
            "repository_dispatch",
            "issue_comment",
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
            "set -x",
            "eval ",
        ):
            if forbidden in source:
                raise RuntimeError(f"{label} workflow gained forbidden operation: {forbidden}")
        if re.search(r"\bwhile\s+true\b|\bfor\s+\(\(\s*;\s*;", source):
            raise RuntimeError(f"{label} workflow contains an unbounded loop")
        if re.search(r"^\s*sleep\s+", source, flags=re.MULTILINE):
            raise RuntimeError(f"{label} workflow must not poll")


def validate(root: Path) -> dict:
    for name in SECRET_NAMES:
        if os.environ.get(name):
            raise RuntimeError(f"static validation received unexpected credential: {name}")
    request = json.loads((root / STATUS_REQUEST_PATH).read_text(encoding="utf-8"))
    validate_status_request(request)
    status, submit = validate_workflow_blobs(root)
    validate_status_workflow(status)
    validate_submit_workflow(submit)
    validate_common_workflow_guards(status, submit)
    validate_optional_submit_request(root / OPTIONAL_SUBMIT_REQUEST_PATH)
    result = {
        "status": "pass",
        "parent_run_request_id": "20260903-poisoned-chalice-stage1-resume-003",
        "status_request_id": "20260903-poisoned-chalice-stage1-resume003-status-001",
        "source_kernel_version": 3,
        "target_kernel_version": 1,
        "status_side_effects": 0,
        "submission_calls": 1,
        "automatic_submission_retries": 0,
        "public_leaderboard_tuning_used": False,
        "final_selection": False,
        "base_verifier_blob": BASE_VERIFIER_BLOB,
        "provenance_verifier_blob": PROVENANCE_VERIFIER_BLOB,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    validate(args.root.resolve())


if __name__ == "__main__":
    main()
