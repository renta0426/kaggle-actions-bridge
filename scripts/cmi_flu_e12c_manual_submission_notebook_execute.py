#!/usr/bin/env python3
"""Create one private E12c Kaggle Notebook and recover its manual-submission outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

from kaggle.api.kaggle_api_extended import KaggleApi

from kaggle_current_output_read import read_current_output
from kaggle_exact_identity import exact_metadata, exact_metadata_eventually, safe_exception, status_name, validate_metadata

REQUEST_ID = "20260911-cmi-flu-e12c-manual-submission-notebook-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET = "renta0426/cmi-flu-e12c-manual-submission-20260911-001"
TITLE = "CMI Flu E12c Manual Submission 20260911 001"
EXPECTED_VERSION = 1
HISTORICAL_SOURCE = "renta0426/cmi-flu-manual-probe-files-20260906-001"
HISTORICAL_VERSION = 1
SCIENCE_COMMIT = "bb6f41ed81b0bbd4ecdc397c6abb9df671c6ebf8"
E12C_BLOB = "a495584a7e461478bd1a41d01d2536c4435a8f7f"
FINAL_CSV_SHA256 = "983aaf097d04477c4ccf7bf817fdf66e552937e69bceaa48cfb260cb84413f1b"
ALLOWED = {
    "submission.csv": 262144,
    "manual-submission-manifest.json": 65536,
}


def ensure_kaggle_cli_path() -> str:
    """Expose the locked Kaggle CLI installed beside the executing venv Python."""
    python_bin = Path(sys.executable).parent
    existing = __import__("os").environ.get("PATH", "")
    parts = [part for part in existing.split(__import__("os").pathsep) if part]
    if str(python_bin) not in parts:
        __import__("os").environ["PATH"] = str(python_bin) + (
            __import__("os").pathsep + existing if existing else ""
        )
    found = shutil.which("kaggle")
    if found is None:
        raise RuntimeError("E12c locked Kaggle CLI unavailable after venv-bin PATH repair")
    if Path(found).parent != python_bin:
        raise RuntimeError("E12c Kaggle CLI resolved outside executing venv bin")
    return found


def verify_historical_source(api: KaggleApi) -> None:
    meta = exact_metadata(api, HISTORICAL_SOURCE)
    validate_metadata(meta, HISTORICAL_SOURCE, HISTORICAL_VERSION)
    state = status_name(getattr(api.kernels_status(HISTORICAL_SOURCE), "status", None))
    if state != "COMPLETE":
        raise RuntimeError("E12c historical source is not COMPLETE")
    print(
        "CMI_FLU_E12C_SOURCE_CURRENT PASS "
        f"version={HISTORICAL_VERSION} state=COMPLETE write=false submission=false"
    )


def require_fresh_target(api: KaggleApi) -> None:
    try:
        exact_metadata(api, TARGET)
    except Exception as exc:
        status = safe_exception(exc).get("http_status")
        if status not in (403, 404):
            raise RuntimeError("E12c exact fresh-target check failed") from exc
    else:
        raise RuntimeError("E12c target already exists; write refused")
    print("CMI_FLU_E12C_TARGET_ABSENT PASS write=false compute=false submission=false")


def materialize(runtime: Path, root: Path) -> Path:
    raw = runtime.read_bytes()
    if len(raw) >= 900000:
        raise RuntimeError("E12c runtime exceeds source budget")
    text = raw.decode("utf-8")
    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'TARGET_KERNEL = "{TARGET}"',
        f'SCIENCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'E12C_BLOB = "{E12C_BLOB}"',
        f'FINAL_CSV_SHA256 = "{FINAL_CSV_SHA256}"',
        'submission_path = output_dir / "submission.csv"',
        "CMI_FLU_E12C_MANUAL_SUBMISSION_READY",
    )
    if any(token not in text for token in required):
        raise RuntimeError("E12c runtime identity/output contract changed")
    lowered = text.casefold()
    if "competition_submit(" in lowered or "kaggle competitions submit" in lowered:
        raise RuntimeError("E12c runtime contains Competition submit path")

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
        "kernel_sources": [HISTORICAL_SOURCE],
        "dataset_sources": [],
        "model_sources": [],
    }
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return kernel


def reconcile_fresh_write(api: KaggleApi, response: object | None, failure: BaseException | None) -> str:
    failure_kind = None
    if failure is not None:
        failure_kind = "exception"
    elif str(getattr(response, "error", "") or ""):
        failure_kind = "response_error"

    try:
        meta = exact_metadata_eventually(api, TARGET, attempts=8, delay_seconds=2.0)
    except Exception as exc:
        info = safe_exception(exc)
        if failure_kind is not None:
            print(
                "CMI_FLU_E12C_WRITE_UNRESOLVED "
                f"client_failure={failure_kind} reconcile_http={info.get('http_status')} "
                f"reconcile_error_sha256={info.get('error_sha256')} write_calls=1 retries=0"
            )
        raise RuntimeError("E12c ambiguous_write reconciliation required") from exc

    validate_metadata(meta, TARGET, EXPECTED_VERSION, cpu=True)
    state = status_name(getattr(api.kernels_status(TARGET), "status", None))
    print(
        "CMI_FLU_E12C_SAVEKERNEL OBSERVED "
        f"version={EXPECTED_VERSION} state={state} write_calls=1 retries=0 "
        f"client_failure={failure_kind or 'none'}"
    )
    return state


def wait_terminal(api: KaggleApi) -> str:
    for index in range(41):
        state = status_name(getattr(api.kernels_status(TARGET), "status", None))
        if state in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}:
            return state
        if index < 40:
            time.sleep(90)
    raise RuntimeError("E12c status polling exhausted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--path-self-test", action="store_true")
    args = parser.parse_args()

    cli = ensure_kaggle_cli_path()
    if args.path_self_test:
        print(
            "CMI_FLU_E12C_OUTPUT_READER_PATH_SELF_TEST PASS "
            f"python_bin={Path(sys.executable).parent} cli_name={Path(cli).name} "
            "auth=false write=false compute=false"
        )
        return 0
    if args.runtime is None or args.output_dir is None:
        raise SystemExit("--runtime and --output-dir are required")
    if args.output_dir.exists():
        raise SystemExit("E12c output directory must be fresh")

    api = KaggleApi()
    api.authenticate()
    verify_historical_source(api)
    require_fresh_target(api)

    with tempfile.TemporaryDirectory(prefix="cmi-e12c-manual-run-") as tmp:
        kernel = materialize(args.runtime.resolve(), Path(tmp))
        # Identity can change between initial checks and the write. Recheck only
        # the exact scientific input and exact target; no remote-capacity heuristic.
        verify_historical_source(api)
        require_fresh_target(api)

        response = None
        failure: BaseException | None = None
        try:
            response = api.kernels_push(str(kernel))
        except Exception as exc:  # exactly one write call
            failure = exc
        reconcile_fresh_write(api, response, failure)

        terminal = wait_terminal(api)
        if terminal != "COMPLETE":
            raise RuntimeError(f"E12c Notebook did not complete successfully:{terminal}")

        read_current_output(
            kernel=TARGET,
            expected_version=EXPECTED_VERSION,
            allow=ALLOWED,
            output_dir=args.output_dir.resolve(),
        )

    print(
        "CMI_FLU_E12C_EXECUTE PASS "
        "version=1 output_read=true submission_api=false manual_operator_submission_only=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
