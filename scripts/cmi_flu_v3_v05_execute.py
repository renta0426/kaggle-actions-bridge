#!/usr/bin/env python3
"""Run exactly one fresh private V3-05 Task1.3 scale-head Notebook."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

from kaggle.api.kaggle_api_extended import KaggleApi

from kaggle_current_output_read import read_current_output
from kaggle_exact_identity import (
    exact_metadata, exact_metadata_eventually, safe_exception, status_name, validate_metadata,
)

REQUEST_ID = "20260912-cmi-flu-strategy-v3-v05-task13-scale-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET = "renta0426/cmi-flu-v3-v05-task13-scale-20260912-001"
TITLE = "cmi-flu-v3-v05-task13-scale-20260912-001"
EXPECTED_VERSION = 1
SCIENCE_COMMIT = "2b8dad2edddfb4df168ecce5f55ad2576d22dc1e"
SCIENCE_BLOB = "136d92de6a5d4f23fb4bb66d4c339aa8f49e79f0"
PRIVATE_NAMES = ["v3_v05_task13_oof_bank.csv", "v3_v05_task13_challenge_bank.csv"]
SAFE_NAMES = ["v3_v05_task13_summary.json", "v3_v05_task13_bank_manifest.json"]
ALLOWED = {
    "v3_v05_task13_oof_bank.csv": 1048576,
    "v3_v05_task13_challenge_bank.csv": 1048576,
    "v3_v05_task13_summary.json": 2097152,
    "v3_v05_task13_bank_manifest.json": 262144,
}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_title_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def validate_title_target_identity() -> str:
    owner, target_slug = TARGET.split("/", 1)
    if owner != "renta0426" or normalize_title_slug(TITLE) != target_slug:
        raise RuntimeError("V3-05 title/target identity mismatch")
    return target_slug


def ensure_kaggle_cli_path() -> str:
    import os
    python_bin = Path(sys.executable).parent
    existing = os.environ.get("PATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if str(python_bin) not in parts:
        os.environ["PATH"] = str(python_bin) + (os.pathsep + existing if existing else "")
    found = shutil.which("kaggle")
    if found is None or Path(found).parent != python_bin:
        raise RuntimeError("V3-05 locked Kaggle CLI unavailable in executing venv")
    return found


def require_fresh_target(api: KaggleApi) -> None:
    try:
        exact_metadata(api, TARGET)
    except Exception as exc:
        status = safe_exception(exc).get("http_status")
        if status not in (403, 404):
            raise RuntimeError("V3-05 exact fresh-target check failed") from exc
    else:
        raise RuntimeError("V3-05 target already exists; write refused")
    print("CMI_FLU_V3_V05_TARGET_ABSENT PASS write=false compute=false submission=false")


def materialize(runtime: Path, root: Path) -> Path:
    validate_title_target_identity()
    raw = runtime.read_bytes()
    if not raw or len(raw) >= 1100000:
        raise RuntimeError("V3-05 runtime source budget failed")
    text = raw.decode("utf-8")
    required = (
        REQUEST_ID, TARGET, SCIENCE_COMMIT, SCIENCE_BLOB,
        "CMI_FLU_V3_V05_RUNTIME_PASS", "conditions=4", "competition_submit=false",
    )
    if any(token not in text for token in required):
        raise RuntimeError("V3-05 runtime identity/output contract changed")
    kernel = root / "kernel"
    kernel.mkdir(parents=True)
    shutil.copyfile(runtime, kernel / "script.py")
    metadata = {
        "id": TARGET,
        "title": TITLE,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "dataset_sources": [],
        "model_sources": [],
    }
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return kernel


def reconcile_fresh_write(api: KaggleApi, response: object | None, failure: BaseException | None) -> str:
    failure_kind = "exception" if failure is not None else (
        "response_error" if str(getattr(response, "error", "") or "") else None
    )
    try:
        meta = exact_metadata_eventually(api, TARGET, attempts=8, delay_seconds=2.0)
    except Exception as exc:
        info = safe_exception(exc)
        print(
            f"CMI_FLU_V3_V05_WRITE_UNRESOLVED client_failure={failure_kind or 'unknown'} "
            f"reconcile_http={info.get('http_status')} reconcile_error_sha256={info.get('error_sha256')} "
            "write_calls=1 retries=0"
        )
        raise RuntimeError("V3-05 ambiguous_write reconciliation required") from exc
    validate_metadata(meta, TARGET, EXPECTED_VERSION, cpu=True)
    state = status_name(getattr(api.kernels_status(TARGET), "status", None))
    print(
        f"CMI_FLU_V3_V05_SAVEKERNEL OBSERVED version=1 state={state} "
        f"write_calls=1 retries=0 client_failure={failure_kind or 'none'}"
    )
    return state


def wait_terminal(api: KaggleApi) -> str:
    for index in range(51):
        state = status_name(getattr(api.kernels_status(TARGET), "status", None))
        if state in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}:
            return state
        if index < 50:
            time.sleep(60)
    raise RuntimeError("V3-05 status polling exhausted")


def validate_private_output(private_dir: Path) -> tuple[dict, dict]:
    actual = sorted(path.name for path in private_dir.iterdir() if path.is_file())
    expected = sorted(PRIVATE_NAMES + SAFE_NAMES)
    if actual != expected:
        raise RuntimeError("V3-05 output allowlist mismatch")
    summary = json.loads((private_dir / "v3_v05_task13_summary.json").read_text())
    manifest = json.loads((private_dir / "v3_v05_task13_bank_manifest.json").read_text())
    runtime = summary.get("runtime") or {}
    if runtime.get("runtime_terminal_marker") != "CMI_FLU_V3_V05_RUNTIME_PASS":
        raise RuntimeError("V3-05 runtime marker missing")
    if runtime.get("request_id") != REQUEST_ID or runtime.get("science_blob") != SCIENCE_BLOB:
        raise RuntimeError("V3-05 runtime identity changed")
    if summary.get("new_candidate_conditions") != ["S1", "S2", "S3", "S4"]:
        raise RuntimeError("V3-05 candidate set changed")
    if int(summary.get("new_candidate_condition_count", -1)) != 4:
        raise RuntimeError("V3-05 candidate count changed")
    if int(summary.get("fit_count", 999999)) > 96:
        raise RuntimeError("V3-05 fit limit violated")
    if (summary.get("Task1.2") or {}).get("state") != "not_executed_data_limited":
        raise RuntimeError("V3-05 Task1.2 boundary changed")
    if summary.get("competition_submission_attempted") is not False or summary.get("public_leaderboard_used") is not False:
        raise RuntimeError("V3-05 public/submission boundary changed")
    if summary.get("raw_unit_conversion_performed") is not False or summary.get("pseudocount_added") is not False:
        raise RuntimeError("V3-05 measurement boundary changed")
    files = manifest.get("files") or []
    if len(files) != 2 or [item.get("filename") for item in files] != PRIVATE_NAMES:
        raise RuntimeError("V3-05 private bank manifest file set changed")
    for item in files:
        path = private_dir / str(item["filename"])
        if item.get("private_row_level") is not True:
            raise RuntimeError("V3-05 private-bank privacy flag changed")
        if int(item.get("bytes", -1)) != path.stat().st_size or item.get("sha256") != sha256_path(path):
            raise RuntimeError("V3-05 private-bank persisted hash mismatch")
    return summary, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--approved-runtime-sha256")
    parser.add_argument("--path-self-test", action="store_true")
    args = parser.parse_args()
    cli = ensure_kaggle_cli_path()
    if args.path_self_test:
        slug = validate_title_target_identity()
        print(
            f"CMI_FLU_V3_V05_EXECUTOR_SELF_TEST PASS python_bin={Path(sys.executable).parent} "
            f"cli_name={Path(cli).name} title_target_slug={slug} auth=false write=false compute=false submission=false"
        )
        return 0
    if args.runtime is None or args.output_dir is None or args.approved_runtime_sha256 is None:
        raise SystemExit("--runtime --output-dir and --approved-runtime-sha256 are required")
    if not re.fullmatch(r"[0-9a-f]{64}", args.approved_runtime_sha256):
        raise SystemExit("V3-05 approved runtime digest missing")
    if hashlib.sha256(args.runtime.read_bytes()).hexdigest() != args.approved_runtime_sha256:
        raise SystemExit("V3-05 approved runtime digest mismatch")
    if args.output_dir.exists():
        raise SystemExit("V3-05 aggregate output directory must be fresh")

    api = KaggleApi()
    api.authenticate()
    require_fresh_target(api)
    with tempfile.TemporaryDirectory(prefix="cmi-v305-run-") as tmp:
        tmp_root = Path(tmp)
        kernel = materialize(args.runtime.resolve(), tmp_root)
        require_fresh_target(api)
        response = None
        failure: BaseException | None = None
        try:
            response = api.kernels_push(str(kernel))
        except Exception as exc:
            failure = exc
        reconcile_fresh_write(api, response, failure)
        terminal = wait_terminal(api)
        if terminal != "COMPLETE":
            raise RuntimeError(f"V3-05 Notebook did not complete successfully:{terminal}")
        private_dir = tmp_root / "private-output"
        read_current_output(
            kernel=TARGET,
            expected_version=EXPECTED_VERSION,
            allow=ALLOWED,
            output_dir=private_dir,
        )
        _, manifest = validate_private_output(private_dir)
        args.output_dir.mkdir(parents=True)
        for name in SAFE_NAMES:
            shutil.copyfile(private_dir / name, args.output_dir / name)
        safe_files = ",".join(
            f"{item['filename']}:{item['rows']}:{item['bytes']}:{item['sha256']}"
            for item in manifest["files"]
        )
        print(f"CMI_FLU_V3_V05_PRIVATE_BANK PASS files={safe_files} contents_exposed=false")

    print(
        "CMI_FLU_V3_V05_EXECUTE PASS version=1 output_read=true aggregate_recovery_only=true "
        "private_csvs_on_kaggle=true retries=0 competition_submit=false manual_operator_submission_only=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
