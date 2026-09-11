#!/usr/bin/env python3
"""Run exactly one protected CMI-Flu V3 batch-1 private Kaggle Notebook."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

from kaggle.api.kaggle_api_extended import KaggleApi

from kaggle_current_output_read import read_current_output
from kaggle_exact_identity import exact_metadata, exact_metadata_eventually, safe_exception, status_name, validate_metadata

REQUEST_ID = "20260911-cmi-flu-strategy-v3-batch1-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260911-001"
TITLE = "CMI Flu Strategy V3 Batch1 Audit Diagnostics 20260911 001"
EXPECTED_VERSION = 1
SOURCE_B = "renta0426/cmi-flu-e12c-manual-submission-20260911-002"
SOURCE_B_VERSION = 1
SOURCE_B_BYTES = 5926
SOURCE_B_SHA256 = "0f9df53c3aa8c6e4ac693f6a42dbd2633b4b61d1462df798a9bd767c220be3a5"
SCIENCE_COMMIT = "270cd6fc5e33579add8177303505e5f744ee5dbc"
SCIENCE_BLOB = "cedd8e2538a06c8b696e74631f0bb2fce427984b"
CSV_NAMES = [
    "v3_public_singleton_Task1_1.csv", "v3_public_singleton_Task1_2.csv",
    "v3_public_singleton_Task1_3.csv", "v3_public_singleton_Task1_4.csv",
    "v3_public_singleton_Task2_1.csv", "v3_public_singleton_Task2_2.csv",
]
SAFE_NAMES = [
    "teacher_ledger.json", "measurement_contracts.json", "split_support.json",
    "auxiliary_label_coverage.json", "source_alignment_audit.json",
    "diagnostic_manifest.json", "runtime_receipt.json",
]
ALLOWED = {name: 262144 for name in CSV_NAMES}
ALLOWED.update({name: 2097152 for name in SAFE_NAMES})
SOURCE_ALLOW = {"submission.csv": 16384, "manual-submission-manifest.json": 131072}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_kaggle_cli_path() -> str:
    import os
    python_bin = Path(sys.executable).parent
    existing = os.environ.get("PATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if str(python_bin) not in parts:
        os.environ["PATH"] = str(python_bin) + (os.pathsep + existing if existing else "")
    found = shutil.which("kaggle")
    if found is None or Path(found).parent != python_bin:
        raise RuntimeError("V3 batch1 locked Kaggle CLI unavailable in executing venv")
    return found


def verify_source_B(api: KaggleApi, output_dir: Path | None = None) -> None:
    meta = exact_metadata(api, SOURCE_B)
    validate_metadata(meta, SOURCE_B, SOURCE_B_VERSION)
    state = status_name(getattr(api.kernels_status(SOURCE_B), "status", None))
    if state != "COMPLETE":
        raise RuntimeError("V3 batch1 source B is not COMPLETE")
    if output_dir is not None:
        read_current_output(kernel=SOURCE_B, expected_version=SOURCE_B_VERSION, allow=SOURCE_ALLOW, output_dir=output_dir)
        source = output_dir / "submission.csv"
        if source.stat().st_size != SOURCE_B_BYTES or sha256_path(source) != SOURCE_B_SHA256:
            raise RuntimeError("V3 batch1 source B current output hash/size mismatch")
    print(f"CMI_FLU_V3_BATCH1_SOURCE_B_CURRENT PASS version=1 state=COMPLETE exact_hash={'true' if output_dir is not None else 'not_rechecked'} write=false submission=false")


def require_fresh_target(api: KaggleApi) -> None:
    try:
        exact_metadata(api, TARGET)
    except Exception as exc:
        status = safe_exception(exc).get("http_status")
        if status not in (403, 404):
            raise RuntimeError("V3 batch1 exact fresh-target check failed") from exc
    else:
        raise RuntimeError("V3 batch1 target already exists; write refused")
    print("CMI_FLU_V3_BATCH1_TARGET_ABSENT PASS write=false compute=false submission=false")


def materialize(runtime: Path, root: Path) -> Path:
    raw = runtime.read_bytes()
    if not raw or len(raw) >= 1100000:
        raise RuntimeError("V3 batch1 runtime source budget failed")
    text = raw.decode("utf-8")
    required = (REQUEST_ID, TARGET, SCIENCE_COMMIT, SCIENCE_BLOB, SOURCE_B, SOURCE_B_SHA256, "CMI_FLU_V3_BATCH1_RUNTIME_PASS", "model_fit_count=0", "competition_submit=false")
    if any(token not in text for token in required):
        raise RuntimeError("V3 batch1 runtime identity/output contract changed")
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


def reconcile_fresh_write(api: KaggleApi, response: object | None, failure: BaseException | None) -> str:
    failure_kind = "exception" if failure is not None else ("response_error" if str(getattr(response, "error", "") or "") else None)
    try:
        meta = exact_metadata_eventually(api, TARGET, attempts=8, delay_seconds=2.0)
    except Exception as exc:
        info = safe_exception(exc)
        print(f"CMI_FLU_V3_BATCH1_WRITE_UNRESOLVED client_failure={failure_kind or 'unknown'} reconcile_http={info.get('http_status')} reconcile_error_sha256={info.get('error_sha256')} write_calls=1 retries=0")
        raise RuntimeError("V3 batch1 ambiguous_write reconciliation required") from exc
    validate_metadata(meta, TARGET, EXPECTED_VERSION, cpu=True)
    state = status_name(getattr(api.kernels_status(TARGET), "status", None))
    print(f"CMI_FLU_V3_BATCH1_SAVEKERNEL OBSERVED version=1 state={state} write_calls=1 retries=0 client_failure={failure_kind or 'none'}")
    return state


def wait_terminal(api: KaggleApi) -> str:
    for index in range(51):
        state = status_name(getattr(api.kernels_status(TARGET), "status", None))
        if state in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}:
            return state
        if index < 50:
            time.sleep(60)
    raise RuntimeError("V3 batch1 status polling exhausted")


def validate_private_output(private_dir: Path) -> dict:
    actual = sorted(path.name for path in private_dir.iterdir() if path.is_file())
    expected = sorted(CSV_NAMES + SAFE_NAMES)
    if actual != expected:
        raise RuntimeError(f"V3 batch1 output allowlist mismatch:{actual}")
    manifest = json.loads((private_dir / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("diagnostic_not_final") is not True or manifest.get("all_generated_before_scoring") is not True:
        raise RuntimeError("V3 batch1 diagnostic manifest boundary changed")
    files = manifest.get("files") or []
    if len(files) != 6 or [item.get("filename") for item in files] != CSV_NAMES:
        raise RuntimeError("V3 batch1 diagnostic manifest file set changed")
    for item in files:
        path = private_dir / str(item["filename"])
        if int(item.get("bytes", -1)) != path.stat().st_size or item.get("sha256") != sha256_path(path):
            raise RuntimeError(f"V3 batch1 diagnostic persisted hash mismatch:{path.name}")
        if item.get("source_csv_sha256") != SOURCE_B_SHA256:
            raise RuntimeError("V3 batch1 diagnostic source identity changed")
    receipt = json.loads((private_dir / "runtime_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("runtime_terminal_marker") != "CMI_FLU_V3_BATCH1_RUNTIME_PASS":
        raise RuntimeError("V3 batch1 runtime terminal marker missing")
    if receipt.get("model_fit_count") != 0 or receipt.get("competition_submission_attempted") is not False:
        raise RuntimeError("V3 batch1 runtime no-fit/submission boundary failed")
    for name in SAFE_NAMES:
        if (private_dir / name).stat().st_size > ALLOWED[name]:
            raise RuntimeError(f"V3 batch1 aggregate output exceeded limit:{name}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--path-self-test", action="store_true")
    args = parser.parse_args()
    cli = ensure_kaggle_cli_path()
    if args.path_self_test:
        print(f"CMI_FLU_V3_BATCH1_PATH_SELF_TEST PASS python_bin={Path(sys.executable).parent} cli_name={Path(cli).name} auth=false write=false compute=false")
        return 0
    if args.runtime is None or args.output_dir is None:
        raise SystemExit("--runtime and --output-dir are required")
    if args.output_dir.exists():
        raise SystemExit("V3 batch1 aggregate output directory must be fresh")

    api = KaggleApi(); api.authenticate()
    with tempfile.TemporaryDirectory(prefix="cmi-v3-batch1-precheck-") as tmp:
        verify_source_B(api, Path(tmp))
    require_fresh_target(api)

    with tempfile.TemporaryDirectory(prefix="cmi-v3-batch1-run-") as tmp:
        tmp_root = Path(tmp)
        kernel = materialize(args.runtime.resolve(), tmp_root)
        verify_source_B(api)
        require_fresh_target(api)
        response = None; failure: BaseException | None = None
        try:
            response = api.kernels_push(str(kernel))
        except Exception as exc:  # exactly one write call
            failure = exc
        reconcile_fresh_write(api, response, failure)
        terminal = wait_terminal(api)
        if terminal != "COMPLETE":
            raise RuntimeError(f"V3 batch1 Notebook did not complete successfully:{terminal}")
        verify_source_B(api)
        private_dir = tmp_root / "private-output"
        read_current_output(kernel=TARGET, expected_version=EXPECTED_VERSION, allow=ALLOWED, output_dir=private_dir)
        manifest = validate_private_output(private_dir)
        args.output_dir.mkdir(parents=True)
        for name in SAFE_NAMES:
            shutil.copyfile(private_dir / name, args.output_dir / name)
        safe_files = ",".join(f"{item['filename']}:{item['bytes']}:{item['sha256']}" for item in manifest["files"])
        print(f"CMI_FLU_V3_BATCH1_PRIVATE_DIAGNOSTICS PASS files={safe_files} contents_exposed=false")

    print("CMI_FLU_V3_BATCH1_EXECUTE PASS version=1 output_read=true aggregate_recovery_only=true private_csvs_on_kaggle=true model_fit_count=0 retries=0 competition_submit=false manual_operator_submission_only=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
