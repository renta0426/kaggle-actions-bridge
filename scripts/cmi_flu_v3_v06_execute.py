#!/usr/bin/env python3
"""Execute exactly one fresh private V3-06 actual-ET calibration Notebook."""
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
    exact_metadata,
    exact_metadata_eventually,
    safe_exception,
    status_name,
    validate_metadata,
)

REQUEST_ID = "20260912-cmi-flu-strategy-v3-v06-actual-et-calibration-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET = "renta0426/cmi-flu-v3-v06-actual-et-calibration-20260912-001"
TITLE = "cmi-flu-v3-v06-actual-et-calibration-20260912-001"
EXPECTED_VERSION = 1
SCIENCE_COMMIT = "0d395e55e7d8d82a1167c5347cc65d0c051ad59f"
SCIENCE_BLOB = "5f75baa73750abed53b476ef9cf525e5dbf37a50"
PRIVATE_NAMES = ["v3_v06_oof_bank.csv", "v3_v06_challenge_bank.csv"]
SAFE_NAMES = ["v3_v06_summary.json", "v3_v06_bank_manifest.json"]
ALLOWED = {
    "v3_v06_oof_bank.csv": 8 * 1024 * 1024,
    "v3_v06_challenge_bank.csv": 2 * 1024 * 1024,
    "v3_v06_summary.json": 4 * 1024 * 1024,
    "v3_v06_bank_manifest.json": 256 * 1024,
}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_title_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def validate_title_target_identity() -> str:
    owner, target_slug = TARGET.split("/", 1)
    if owner != "renta0426" or normalize_title_slug(TITLE) != target_slug:
        raise RuntimeError("V3-06 title/target identity mismatch")
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
        raise RuntimeError("V3-06 locked Kaggle CLI unavailable in executing venv")
    return found


def require_fresh_target(api: KaggleApi) -> None:
    try:
        exact_metadata(api, TARGET)
    except Exception as exc:
        status = safe_exception(exc).get("http_status")
        if status not in (403, 404):
            raise RuntimeError("V3-06 exact fresh-target check failed") from exc
    else:
        raise RuntimeError("V3-06 target already exists; write refused")
    print("CMI_FLU_V3_V06_TARGET_ABSENT PASS write=false compute=false submission=false")


def materialize(runtime: Path, root: Path) -> Path:
    validate_title_target_identity()
    raw = runtime.read_bytes()
    if not raw or len(raw) >= 1_100_000:
        raise RuntimeError("V3-06 runtime source budget failed")
    text = raw.decode("utf-8")
    required = (
        REQUEST_ID,
        TARGET,
        SCIENCE_COMMIT,
        SCIENCE_BLOB,
        "CMI_FLU_V3_V06_RUNTIME_PASS",
        "task21_et_log_affine",
        "task22_et_log_affine",
        "competition_submit=false",
    )
    if any(token not in text for token in required):
        raise RuntimeError("V3-06 runtime identity/science contract changed")
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
            f"CMI_FLU_V3_V06_WRITE_UNRESOLVED client_failure={failure_kind or 'unknown'} "
            f"reconcile_http={info.get('http_status')} reconcile_error_sha256={info.get('error_sha256')} "
            "write_calls=1 retries=0"
        )
        raise RuntimeError("V3-06 ambiguous_write reconciliation required") from exc
    validate_metadata(meta, TARGET, EXPECTED_VERSION, cpu=True)
    state = status_name(getattr(api.kernels_status(TARGET), "status", None))
    print(
        f"CMI_FLU_V3_V06_SAVEKERNEL OBSERVED version=1 state={state} "
        f"write_calls=1 retries=0 client_failure={failure_kind or 'none'}"
    )
    return state


def wait_terminal(api: KaggleApi) -> str:
    # Fast startup checks catch immediate runtime failures; then bounded 60s polling.
    schedule = [5, 10, 20, 30, 60]
    for delay in schedule:
        state = status_name(getattr(api.kernels_status(TARGET), "status", None))
        if state in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}:
            return state
        time.sleep(delay)
    for _ in range(85):
        state = status_name(getattr(api.kernels_status(TARGET), "status", None))
        if state in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}:
            return state
        time.sleep(60)
    raise RuntimeError("V3-06 status polling exhausted")


def validate_private_output(private_dir: Path) -> tuple[dict, dict]:
    actual = sorted(path.name for path in private_dir.iterdir() if path.is_file())
    if actual != sorted(PRIVATE_NAMES + SAFE_NAMES):
        raise RuntimeError("V3-06 output allowlist mismatch")
    summary = json.loads((private_dir / SAFE_NAMES[0]).read_text())
    manifest = json.loads((private_dir / SAFE_NAMES[1]).read_text())
    runtime = summary.get("runtime") or {}
    if runtime.get("runtime_terminal_marker") != "CMI_FLU_V3_V06_RUNTIME_PASS":
        raise RuntimeError("V3-06 runtime marker missing")
    if runtime.get("request_id") != REQUEST_ID:
        raise RuntimeError("V3-06 runtime request identity changed")
    if runtime.get("science_commit") != SCIENCE_COMMIT or runtime.get("science_blob") != SCIENCE_BLOB:
        raise RuntimeError("V3-06 runtime science provenance changed")
    if summary.get("stage") != "V3-06" or summary.get("new_candidate_conditions") != [
        "task21_et_log_affine", "task22_et_log_affine"
    ]:
        raise RuntimeError("V3-06 stage/condition set changed")
    if int(summary.get("new_candidate_condition_count", -1)) != 2:
        raise RuntimeError("V3-06 condition count changed")
    if int(summary.get("fit_count", 999999)) > 256:
        raise RuntimeError("V3-06 fit cap violated")
    contract = summary.get("calibration_contract") or {}
    if contract.get("kind") != "log2_affine" or contract.get("inner_study_folds") != 3:
        raise RuntimeError("V3-06 calibration contract changed")
    if contract.get("affine_or_power_grid_reopened") is not False:
        raise RuntimeError("V3-06 forbidden calibration grid reopened")
    if summary.get("public_leaderboard_used") is not False or summary.get("competition_submission_attempted") is not False:
        raise RuntimeError("V3-06 Public/submission boundary changed")
    tasks = summary.get("tasks") or {}
    if sorted(tasks) != ["Task2.1", "Task2.2"]:
        raise RuntimeError("V3-06 task set changed")
    expected_models = {"Task2.1": "et_subtype_d3_l5", "Task2.2": "et_subtype_d5_l10"}
    for task, expected in expected_models.items():
        if ((tasks.get(task) or {}).get("model") or {}).get("name") != expected:
            raise RuntimeError(f"V3-06 incumbent changed:{task}")
        rank = (((tasks.get(task) or {}).get("challenge") or {}).get("rank_contract") or {})
        if rank.get("same_rank_vector") is not True or rank.get("same_tie_equivalence") is not True:
            raise RuntimeError(f"V3-06 Challenge monotone-rank contract failed:{task}")
    files = manifest.get("files") or []
    if [item.get("filename") for item in files] != PRIVATE_NAMES:
        raise RuntimeError("V3-06 private bank manifest changed")
    for item in files:
        path = private_dir / str(item["filename"])
        if item.get("private_row_level") is not True:
            raise RuntimeError("V3-06 private-bank privacy flag changed")
        if int(item.get("bytes", -1)) != path.stat().st_size or item.get("sha256") != sha256_path(path):
            raise RuntimeError("V3-06 private-bank persisted hash mismatch")
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
            f"CMI_FLU_V3_V06_EXECUTOR_SELF_TEST PASS python_bin={Path(sys.executable).parent} "
            f"cli_name={Path(cli).name} title_target_slug={slug} auth=false write=false compute=false submission=false"
        )
        return 0
    if args.runtime is None or args.output_dir is None or args.approved_runtime_sha256 is None:
        raise SystemExit("--runtime --output-dir and --approved-runtime-sha256 are required")
    if not re.fullmatch(r"[0-9a-f]{64}", args.approved_runtime_sha256):
        raise SystemExit("V3-06 approved runtime digest missing")
    if sha256_path(args.runtime) != args.approved_runtime_sha256:
        raise SystemExit("V3-06 approved runtime digest mismatch")
    if args.output_dir.exists():
        raise SystemExit("V3-06 aggregate output directory must be fresh")

    api = KaggleApi()
    api.authenticate()
    require_fresh_target(api)
    with tempfile.TemporaryDirectory(prefix="cmi-v306-run-") as tmp:
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
            raise RuntimeError(f"V3-06 Notebook did not complete successfully:{terminal}")
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
        print(f"CMI_FLU_V3_V06_PRIVATE_BANK PASS files={safe_files} contents_exposed=false")

    print(
        "CMI_FLU_V3_V06_EXECUTE PASS version=1 output_read=true aggregate_recovery_only=true "
        "private_csvs_on_kaggle=true retries=0 competition_submit=false manual_operator_submission_only=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
