"""Credential-free integration validation for the frozen Mellum transfer launch."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


EXPECTED_REQUEST_ID = "20260903-poisoned-chalice-mellum-transfer-v1-001"
EXPECTED_MATERIALIZER_BLOB = "fd9c645c9ff9a128b8f3a8dfdadd7c937e6d62a1"
EXPECTED_RESEARCH_COMMIT = "9f6be68c27dc1f3326d68bed4e2abf80db893748"
EXPECTED_SOURCE_FILES = {
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
EXPECTED_KERNEL_METADATA = {
    "id": "renta0426/mellum-transfer-v1",
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
ALLOWED_RECORD_KEYS = {"sample_id", "content_sha256", "language", "sample_index"}
FORBIDDEN_RECORD_KEYS = {"label", "membership", "is_member", "lumia_score"}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def source_text(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def prediction_records(notebook: dict) -> list[dict]:
    matches: list[list[dict]] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = source_text(cell)
        if "PREDICTION_MANIFEST = pd.DataFrame(" not in source:
            continue
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name)
                and target.id == "PREDICTION_MANIFEST"
                for target in node.targets
            ):
                continue
            if not isinstance(node.value, ast.Call) or not node.value.args:
                raise RuntimeError("prediction manifest is not a literal DataFrame")
            value = ast.literal_eval(node.value.args[0])
            if not isinstance(value, list) or not all(
                isinstance(row, dict) for row in value
            ):
                raise RuntimeError("prediction manifest literal has an invalid shape")
            matches.append(value)
    if len(matches) != 1:
        raise RuntimeError(f"expected one prediction manifest, got {len(matches)}")
    return matches[0]


def validate_request(request: dict, materializer: bytes) -> None:
    expected = {
        "schema_version": 1,
        "request_id": EXPECTED_REQUEST_ID,
        "competition": "poisoned-chalice-icse27",
        "operation": "kernel_run",
        "target": "renta0426/mellum-transfer-v1",
        "prerequisite_kernel": "renta0426/stage1-raw-fim-submission-v1",
        "materializer_path": "scripts/materialize_poisoned_chalice_mellum_v1.py",
        "materializer_blob_sha": EXPECTED_MATERIALIZER_BLOB,
        "automatic_compute_retries": 0,
        "enable_internet": True,
        "competition_submission": False,
        "artifact_retention_days": 1,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise RuntimeError(f"request contract mismatch: {key}")
    if git_blob_sha(materializer) != EXPECTED_MATERIALIZER_BLOB:
        raise RuntimeError("materializer content differs from request pin")
    compile(materializer, request["materializer_path"], "exec")

    source = request.get("research_source") or {}
    if source.get("repository") != (
        "renta0426/The-Poisoned-Chalice-of-LLM-Evaluation"
    ):
        raise RuntimeError("research repository changed")
    if source.get("commit") != EXPECTED_RESEARCH_COMMIT:
        raise RuntimeError("research commit changed")
    observed = {
        str(item.get("path")): (
            str(item.get("git_blob_sha")),
            int(item.get("max_bytes")),
        )
        for item in source.get("files", [])
    }
    if observed != EXPECTED_SOURCE_FILES:
        raise RuntimeError("research file allowlist or pin changed")

    if request.get("resource") != {
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


def validate_launch(launch: str) -> None:
    required = (
        "permissions: {}",
        "group: kaggle-resource-global",
        "cancel-in-progress: false",
        "environment: kaggle-readonry",
        "REQUESTED_ACCELERATOR: gpu",
        "TARGET_KERNEL: renta0426/mellum-transfer-v1",
        "PREREQUISITE_KERNEL: renta0426/stage1-raw-fim-submission-v1",
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
        "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093",
        "retention-days: 1",
        "MELLUM_KERNEL_BUNDLE PASS",
        "MELLUM_BUNDLE_REVALIDATED PASS",
        "MELLUM_LIVE_PREFLIGHT PASS",
        "MELLUM_LAUNCH_DEFERRED",
        "automatic_retries=0 submission=false",
        "target_labels_embedded=false",
    )
    for marker in required:
        if marker not in launch:
            raise RuntimeError(f"protected launch marker missing: {marker}")

    if launch.count('"${WORKDIR}/venv/bin/kaggle" kernels push') != 1:
        raise RuntimeError("protected launch must contain one executable kernels push")
    for forbidden in (
        '"${WORKDIR}/venv/bin/kaggle" competitions submit',
        '"${WORKDIR}/venv/bin/kaggle" kernels delete',
        '"${WORKDIR}/venv/bin/kaggle" kernels cancel',
        '"${WORKDIR}/venv/bin/kaggle" datasets create',
        '"${WORKDIR}/venv/bin/kaggle" models',
        "actions/cache",
        "workflow_dispatch",
        "continue-on-error: true",
    ):
        if forbidden in launch:
            raise RuntimeError(f"forbidden launch capability present: {forbidden}")
    if re.search(r"\bwhile\s+true\b|\bfor\s+\(\(\s*;\s*;", launch):
        raise RuntimeError("unbounded loop found in protected launch")
    if re.search(r"^\s*sleep\s+", launch, flags=re.MULTILINE):
        raise RuntimeError("protected launch must not poll")

    launch_marker = "\n  launch:\n"
    if launch_marker not in launch:
        raise RuntimeError("protected launch job is missing")
    prelaunch, protected = launch.split(launch_marker, 1)
    if "${{ secrets." in prelaunch:
        raise RuntimeError("credential reference appears before protected launch job")
    if "environment: kaggle-readonry" in prelaunch:
        raise RuntimeError("protected Environment appears outside launch job")
    if protected.count("environment: kaggle-readonry") != 1:
        raise RuntimeError("protected Environment count changed")
    if protected.count("${{ secrets.KAGGLE_API_TOKEN }}") != 2:
        raise RuntimeError("Kaggle credential use must remain in two bounded launch steps")


def validate_bundle(bundle: Path) -> dict:
    files = sorted(
        str(path.relative_to(bundle)) for path in bundle.rglob("*") if path.is_file()
    )
    if files != [
        "artifact_manifest.json",
        "kernel/kernel-metadata.json",
        "kernel/mellum-transfer-v1.ipynb",
    ]:
        raise RuntimeError(f"unexpected bundle file allowlist: {files}")

    manifest = json.loads(
        (bundle / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("request_id") != EXPECTED_REQUEST_ID:
        raise RuntimeError("bundle request identity mismatch")
    if manifest.get("research_commit") != EXPECTED_RESEARCH_COMMIT:
        raise RuntimeError("bundle research commit mismatch")
    if manifest.get("verified_source_blobs") != {
        path: pin for path, (pin, _) in EXPECTED_SOURCE_FILES.items()
    }:
        raise RuntimeError("bundle source verification record mismatch")
    if manifest.get("target_labels_embedded") is not False:
        raise RuntimeError("target labels crossed the build boundary")
    if manifest.get("competition_submission") is not False:
        raise RuntimeError("bundle gained competition-submission capability")
    build = manifest.get("build_environment") or {}
    if (build.get("packages") or {}) != {
        "pandas": "2.3.2",
        "pyarrow": "21.0.0",
        "nbformat": "5.10.4",
    }:
        raise RuntimeError("credential-free build package versions changed")

    notebook_path = bundle / "kernel/mellum-transfer-v1.ipynb"
    metadata_path = bundle / "kernel/kernel-metadata.json"
    for name, path in (
        ("mellum-transfer-v1.ipynb", notebook_path),
        ("kernel-metadata.json", metadata_path),
    ):
        declared = manifest["kernel_files"][name]
        if hashlib.sha256(path.read_bytes()).hexdigest() != declared["sha256"]:
            raise RuntimeError(f"bundle digest mismatch: {name}")
        if path.stat().st_size != declared["bytes"]:
            raise RuntimeError(f"bundle byte count mismatch: {name}")
    if notebook_path.stat().st_size > 4_000_000:
        raise RuntimeError("generated notebook exceeds fixed size budget")
    if metadata_path.stat().st_size > 32_768:
        raise RuntimeError("generated metadata exceeds fixed size budget")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata != EXPECTED_KERNEL_METADATA:
        raise RuntimeError("kernel metadata changed")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells") or []
    if notebook.get("nbformat") != 4 or len(cells) != 7:
        raise RuntimeError("generated notebook structure changed")
    if not all(
        cell.get("id") == f"pc-mellum-v1-{index:02d}"
        for index, cell in enumerate(cells)
    ):
        raise RuntimeError("stable notebook cell IDs are missing")
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = source_text(cell)
        if not source.lstrip().startswith("%"):
            ast.parse(source)

    records = prediction_records(notebook)
    if len(records) != 2_000:
        raise RuntimeError("embedded cohort row count changed")
    if any(set(record) != ALLOWED_RECORD_KEYS for record in records):
        raise RuntimeError("embedded prediction record schema changed")
    if any(FORBIDDEN_RECORD_KEYS.intersection(record) for record in records):
        raise RuntimeError("target label or previous score is embedded")
    if [int(record["sample_index"]) for record in records] != list(range(2_000)):
        raise RuntimeError("embedded cohort ordering changed")
    if len({str(record["sample_id"]) for record in records}) != 2_000:
        raise RuntimeError("embedded sample IDs are not unique")
    hashes = [str(record["content_sha256"]) for record in records]
    if len(set(hashes)) != 2_000 or any(
        not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes
    ):
        raise RuntimeError("embedded content hashes are invalid")
    languages = [str(record["language"]) for record in records]
    if {language: languages.count(language) for language in set(languages)} != {
        "Go": 400,
        "Java": 400,
        "Python": 400,
        "Ruby": 400,
        "Rust": 400,
    }:
        raise RuntimeError("embedded language counts changed")

    combined = "\n".join(source_text(cell) for cell in cells)
    for marker in (
        "MODEL_ID = 'JetBrains/Mellum-4b-base'",
        "MODEL_REVISION = '83cce2605fbdf6a3868627e9b0a5924e0072b94d'",
        "UPSTREAM_COMMIT = '413f56040e5b4805bcf15ed794dec56bc4e16b41'",
        "UPSTREAM_SHA256 = '6e8559279e8ebd94ec8c25561c91d38c454a4e109ea0813b07864ff6a66d4068'",
        "scores, scored = fuse_membership_scores(raw_features, RUNTIME_CONFIG)",
        '"target_labels_embedded_in_gpu_notebook": False',
        '"target_labels_used_for_training_or_normalization": False',
        '"previous_model_scores_used": False',
        '"submission_created": False',
    ):
        if marker not in combined:
            raise RuntimeError(f"generated notebook marker missing: {marker}")
    for forbidden in (
        "kaggle competitions submit",
        "competitions submit",
        "kernels push",
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
    ):
        if forbidden in combined:
            raise RuntimeError(f"forbidden notebook operation present: {forbidden}")

    audit = manifest.get("audit") or {}
    if audit.get("embedded_rows") != 2_000:
        raise RuntimeError("bundle audit row count mismatch")
    if audit.get("embedded_record_keys") != sorted(ALLOWED_RECORD_KEYS):
        raise RuntimeError("bundle audit record keys mismatch")
    if audit.get("embedded_label_fields") != []:
        raise RuntimeError("bundle audit reports label fields")
    return manifest


def validate(root: Path) -> dict:
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        if os.environ.get(name):
            raise RuntimeError(f"static validation received unexpected credential: {name}")

    request_path = root / "requests/poisoned-chalice-mellum-transfer-v1-launch.json"
    materializer_path = root / "scripts/materialize_poisoned_chalice_mellum_v1.py"
    launch_path = root / ".github/workflows/110-poisoned-chalice-mellum-transfer-v1-launch.yml"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    materializer = materializer_path.read_bytes()
    validate_request(request, materializer)
    validate_launch(launch_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="mellum-static-") as temporary:
        temporary_root = Path(temporary)
        bundle = temporary_root / "bundle"
        subprocess.run(
            [
                sys.executable,
                str(materializer_path),
                "--request",
                str(request_path),
                "--work-root",
                str(temporary_root / "materialize"),
                "--bundle-root",
                str(bundle),
            ],
            cwd=root,
            check=True,
            timeout=240,
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
        manifest = validate_bundle(bundle)

    result = {
        "status": "pass",
        "request_id": EXPECTED_REQUEST_ID,
        "research_commit": EXPECTED_RESEARCH_COMMIT,
        "source_files": len(EXPECTED_SOURCE_FILES),
        "kernel_files": sorted(manifest["kernel_files"]),
        "notebook_sha256": manifest["kernel_files"]["mellum-transfer-v1.ipynb"]["sha256"],
        "embedded_rows": manifest["audit"]["embedded_rows"],
        "target_labels_embedded": False,
        "kaggle_push_calls": 1,
        "competition_submission": False,
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
