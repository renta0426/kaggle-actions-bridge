#!/usr/bin/env python3
"""Execute exactly one protected Strategy-v3 V3-02 private validation-bank Notebook."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

from kaggle.api.kaggle_api_extended import KaggleApi

import cmi_flu_v3_batch1_execute as transport
from kaggle_current_output_read import read_current_output

REQUEST_PATH = "requests/cmi-flu-v3-v02-validation-bank-001.json"
REQUEST_ID = "20260912-cmi-flu-strategy-v3-v02-validation-bank-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET = "renta0426/cmi-flu-v3-v02-validation-bank-20260912-001"
TITLE = "cmi-flu-v3-v02-validation-bank-20260912-001"
EXPECTED_VERSION = 1
SCIENCE_COMMIT = "c47a9b1e5a6fdb889d9962e00cca7e418e366d23"
SCIENCE_BLOB = "e8e87bda474814ecf64b6b9530b46b5d81d1d27b"
RUNTIME_BYTES = 402983
RUNTIME_SHA256 = "9fb900680dfe0278c9a655d369bf113a292978d76ee2960a398f4ab4e7845390"
SOURCE_B = "renta0426/cmi-flu-e12c-manual-submission-20260911-002"
SOURCE_B_VERSION = 1
SOURCE_B_BYTES = 5926
SOURCE_B_SHA256 = "0f9df53c3aa8c6e4ac693f6a42dbd2633b4b61d1462df798a9bd767c220be3a5"
PRIVATE_NAMES = ["v3_v02_oof_bank.csv", "v3_v02_challenge_bank.csv"]
SAFE_NAMES = ["v3_v02_summary.json", "v3_v02_bank_manifest.json"]
ALLOWED = {
    "v3_v02_oof_bank.csv": 32 * 1024 * 1024,
    "v3_v02_challenge_bank.csv": 4 * 1024 * 1024,
    "v3_v02_summary.json": 4 * 1024 * 1024,
    "v3_v02_bank_manifest.json": 131072,
}
SOURCE_ALLOW = {"submission.csv": 16384, "manual-submission-manifest.json": 131072}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_transport() -> None:
    transport.REQUEST_ID = REQUEST_ID
    transport.TARGET = TARGET
    transport.TITLE = TITLE
    transport.EXPECTED_VERSION = EXPECTED_VERSION
    transport.SOURCE_B = SOURCE_B
    transport.SOURCE_B_VERSION = SOURCE_B_VERSION
    transport.SOURCE_B_BYTES = SOURCE_B_BYTES
    transport.SOURCE_B_SHA256 = SOURCE_B_SHA256
    transport.SOURCE_ALLOW = SOURCE_ALLOW


def validate_request(root: Path) -> dict:
    request = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    exact = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "execution_policy": "kaggle_native_capacity_v2",
        "competition": COMPETITION,
        "operation": "save_kernel_once",
        "target": TARGET,
        "title": TITLE,
        "expected_version": EXPECTED_VERSION,
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": "connector_verified_exact_blob_relay",
        "strategy_v3_v02_blob_sha": SCIENCE_BLOB,
        "automatic_compute_retries": 0,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "competition_submission_authorized": False,
        "manual_operator_submission_only": True,
        "final_submission_selection_authorized": False,
    }
    for key, value in exact.items():
        if request.get(key) != value:
            raise RuntimeError(f"V3-02 request mismatch:{key}")
    if request.get("approved_runtime") != {"bytes": RUNTIME_BYTES, "sha256": RUNTIME_SHA256}:
        raise RuntimeError("V3-02 approved runtime contract mismatch")
    source = request.get("source_B") or {}
    if (source.get("source_kernel"), source.get("expected_current_version"), source.get("bytes"), source.get("sha256")) != (SOURCE_B, 1, SOURCE_B_BYTES, SOURCE_B_SHA256):
        raise RuntimeError("V3-02 source B request mismatch")
    science = request.get("scientific_contract") or {}
    expected_science = {
        "stage": "V3-02", "new_candidate_conditions": 0, "fit_limit": 256,
        "pseudo40_repetitions": 200, "pseudo40_public_n": 12, "pseudo40_private_n": 28,
        "public_leaderboard_used": False, "complete_local_panel_selection_uses_target_values": False,
        "task12_supervised_raw_scale_branch_executed": False, "task14_supervised_model_executed": False,
    }
    if science != expected_science:
        raise RuntimeError("V3-02 scientific request contract mismatch")
    output = request.get("output_contract") or {}
    if output.get("private_row_level_files") != PRIVATE_NAMES or output.get("aggregate_safe_files") != SAFE_NAMES:
        raise RuntimeError("V3-02 output file contract mismatch")
    if output.get("public_artifact_upload") is not False or output.get("row_level_contents_may_exist_only_in_private_kaggle_output_and_ephemeral_runner_validation") is not True:
        raise RuntimeError("V3-02 privacy output contract mismatch")
    return request


def validate_title_target_identity() -> None:
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or TITLE != slug:
        raise RuntimeError("V3-02 title/target identity mismatch")
    normalized = re.sub(r"[^a-z0-9]+", "-", TITLE.casefold()).strip("-")
    if normalized != slug:
        raise RuntimeError("V3-02 title normalization mismatch")


def materialize(runtime: Path, root: Path) -> Path:
    validate_title_target_identity()
    raw = runtime.read_bytes()
    if len(raw) != RUNTIME_BYTES or hashlib.sha256(raw).hexdigest() != RUNTIME_SHA256:
        raise RuntimeError("V3-02 frozen runtime identity mismatch")
    text = raw.decode("utf-8")
    required = (REQUEST_ID, TARGET, SCIENCE_BLOB, SOURCE_B_SHA256, "CMI_FLU_V3_V02_RUNTIME_PASS", "new_conditions=0", "competition_submit=false", "public_used=false")
    if any(token not in text for token in required):
        raise RuntimeError("V3-02 runtime semantic identity changed")
    kernel = root / "kernel"; kernel.mkdir(parents=True)
    shutil.copyfile(runtime, kernel / "script.py")
    metadata = {
        "id": TARGET, "title": TITLE, "code_file": "script.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False, "competition_sources": [COMPETITION], "kernel_sources": [SOURCE_B],
        "dataset_sources": [], "model_sources": [],
    }
    (kernel / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return kernel


def validate_private_output(private_dir: Path) -> tuple[dict, dict]:
    actual = sorted(p.name for p in private_dir.iterdir() if p.is_file())
    expected = sorted(PRIVATE_NAMES + SAFE_NAMES)
    if actual != expected:
        raise RuntimeError("V3-02 private output allowlist mismatch")
    manifest = json.loads((private_dir / "v3_v02_bank_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("experiment") != "strategy_v3_v02_paired_validation_and_private_prediction_store":
        raise RuntimeError("V3-02 bank manifest identity mismatch")
    if manifest.get("row_level_contents_must_not_be_publicly_emitted") is not True or manifest.get("competition_submission_attempted") is not False:
        raise RuntimeError("V3-02 bank manifest privacy/submission boundary changed")
    files = manifest.get("files") or []
    if [x.get("filename") for x in files] != PRIVATE_NAMES or any(x.get("private_row_level") is not True for x in files):
        raise RuntimeError("V3-02 private bank manifest file contract changed")
    for item in files:
        path = private_dir / str(item["filename"])
        if int(item.get("bytes", -1)) != path.stat().st_size or item.get("sha256") != sha256_path(path):
            raise RuntimeError("V3-02 private bank persisted identity mismatch")
        if int(item.get("rows", 0)) <= 0:
            raise RuntimeError("V3-02 private bank row count invalid")
    summary = json.loads((private_dir / "v3_v02_summary.json").read_text(encoding="utf-8"))
    exact = {
        "schema_version": 1, "experiment": "strategy_v3_v02_paired_validation_and_private_prediction_store",
        "stage": "V3-02", "new_candidate_conditions": 0, "fit_limit": 256,
        "public_leaderboard_used": False, "competition_submission_attempted": False,
        "private_bank_contains_row_level_values": True, "aggregate_contains_row_level_values": False,
    }
    for key, value in exact.items():
        if summary.get(key) != value:
            raise RuntimeError(f"V3-02 summary contract mismatch:{key}")
    if not 0 < int(summary.get("fit_count", 0)) <= 256:
        raise RuntimeError("V3-02 fit count outside approved finite budget")
    runtime = summary.get("runtime") or {}
    if runtime.get("request_id") != REQUEST_ID or runtime.get("science_blob") != SCIENCE_BLOB or runtime.get("source_B_sha256") != SOURCE_B_SHA256 or runtime.get("runtime_terminal_marker") != "CMI_FLU_V3_V02_RUNTIME_PASS":
        raise RuntimeError("V3-02 runtime receipt mismatch")
    bank = summary.get("bank") or {}
    if int(bank.get("oof_rows", 0)) <= 0 or int(bank.get("challenge_rows", 0)) <= 0:
        raise RuntimeError("V3-02 aggregate bank counts invalid")
    if (summary.get("schema_caveat") or {}).get("use_explicit_fields") is not True:
        raise RuntimeError("V3-02 HAI metric-subject caveat missing")
    for name in SAFE_NAMES:
        if not 0 < (private_dir / name).stat().st_size <= ALLOWED[name]:
            raise RuntimeError("V3-02 safe output size contract violated")
    return summary, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approved-runtime-sha256", required=True)
    parser.add_argument("--path-self-test", action="store_true")
    args = parser.parse_args()
    root = args.repository_root.resolve(); validate_request(root); _configure_transport()
    if args.approved_runtime_sha256 != RUNTIME_SHA256:
        raise SystemExit("V3-02 command approved runtime digest mismatch")
    if hashlib.sha256(args.runtime.read_bytes()).hexdigest() != RUNTIME_SHA256:
        raise SystemExit("V3-02 runtime digest mismatch before any Kaggle call")
    transport.ensure_kaggle_cli_path()
    if args.path_self_test:
        validate_title_target_identity()
        with tempfile.TemporaryDirectory(prefix="v302-path-test-") as tmp:
            fresh = transport.fresh_source_output_dir(Path(tmp))
            if fresh.exists(): raise RuntimeError("V3-02 fresh output path regression")
        print("CMI_FLU_V3_V02_EXECUTOR_SELF_TEST PASS auth=false write=false compute=false submission=false")
        return 0
    if args.output_dir.exists():
        raise SystemExit("V3-02 safe output directory must be fresh")

    api = KaggleApi(); api.authenticate()
    with tempfile.TemporaryDirectory(prefix="v302-source-precheck-") as tmp:
        transport.verify_source_B(api, transport.fresh_source_output_dir(Path(tmp)))
    transport.require_fresh_target(api)

    with tempfile.TemporaryDirectory(prefix="v302-run-") as tmp:
        tmp_root = Path(tmp); kernel = materialize(args.runtime.resolve(), tmp_root)
        transport.verify_source_B(api); transport.require_fresh_target(api)
        response = None; failure: BaseException | None = None
        try:
            response = api.kernels_push(str(kernel))
        except Exception as exc:  # exactly one write call
            failure = exc
        transport.reconcile_fresh_write(api, response, failure)
        terminal = transport.wait_terminal(api)
        if terminal != "COMPLETE":
            raise RuntimeError(f"V3-02 Notebook did not complete successfully:{terminal}")
        transport.verify_source_B(api)
        private_dir = tmp_root / "private-output"
        read_current_output(kernel=TARGET, expected_version=EXPECTED_VERSION, allow=ALLOWED, output_dir=private_dir)
        summary, manifest = validate_private_output(private_dir)
        args.output_dir.mkdir(parents=True)
        for name in SAFE_NAMES:
            shutil.copyfile(private_dir / name, args.output_dir / name)
        bank_files = ",".join(f"{item['filename']}:{item['rows']}:{item['bytes']}:{item['sha256']}" for item in manifest["files"])
        print(f"CMI_FLU_V3_V02_PRIVATE_BANK PASS files={bank_files} contents_exposed=false")
        print(f"CMI_FLU_V3_V02_AGGREGATE PASS fit_count={summary['fit_count']} oof_rows={summary['bank']['oof_rows']} challenge_rows={summary['bank']['challenge_rows']} public_used=false competition_submit=false")

    print("CMI_FLU_V3_V02_EXECUTE PASS version=1 output_read=true aggregate_recovery_only=true private_bank_on_kaggle=true retries=0 competition_submit=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
