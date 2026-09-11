#!/usr/bin/env python3
"""Create one private E12c-v2 Notebook and recover the generated manual submission."""
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
from kaggle_exact_identity import (
    exact_metadata,
    exact_metadata_eventually,
    safe_exception,
    status_name,
    validate_metadata,
)

REQUEST_ID = "20260911-cmi-flu-e12c-v2-historical-backbone-002"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET = "renta0426/cmi-flu-e12c-manual-submission-20260911-002"
TITLE = "CMI Flu E12c Manual Submission 20260911 002"
EXPECTED_VERSION = 1
HISTORICAL_SOURCE = "renta0426/cmi-flu-manual-probe-files-20260906-001"
HISTORICAL_VERSION = 1
SCIENCE_COMMIT = "9916c04a3d5510eead8f361960d73b5d94201892"
E12C_V2_BLOB = "238d3ad67984fdd8c375bb0aa263bb4717029027"
HISTORICAL_SHA256 = "365607d59cd530656b929a1c1c57412cc6d375265a8d1ba10d304c64e012f387"
TASK13_SHA256 = "de8bc3b6bbd3e3aad4099b83eb63a1ebbd81c4f4eeb60f77808858f4674be5da"
ALLOWED = {
    "submission.csv": 262144,
    "manual-submission-manifest.json": 65536,
}


def ensure_kaggle_cli_path() -> str:
    python_bin = Path(sys.executable).parent
    import os

    existing = os.environ.get("PATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if str(python_bin) not in parts:
        os.environ["PATH"] = str(python_bin) + (os.pathsep + existing if existing else "")
    found = shutil.which("kaggle")
    if found is None:
        raise RuntimeError("E12c-v2 locked Kaggle CLI unavailable after PATH repair")
    if Path(found).parent != python_bin:
        raise RuntimeError("E12c-v2 Kaggle CLI resolved outside executing venv bin")
    return found


def verify_historical_source(api: KaggleApi) -> None:
    meta = exact_metadata(api, HISTORICAL_SOURCE)
    validate_metadata(meta, HISTORICAL_SOURCE, HISTORICAL_VERSION)
    state = status_name(getattr(api.kernels_status(HISTORICAL_SOURCE), "status", None))
    if state != "COMPLETE":
        raise RuntimeError("E12c-v2 historical source is not COMPLETE")
    print(
        "CMI_FLU_E12C_V2_SOURCE_CURRENT PASS "
        f"version={HISTORICAL_VERSION} state=COMPLETE write=false submission=false"
    )


def require_fresh_target(api: KaggleApi) -> None:
    try:
        exact_metadata(api, TARGET)
    except Exception as exc:
        status = safe_exception(exc).get("http_status")
        if status not in (403, 404):
            raise RuntimeError("E12c-v2 exact fresh-target check failed") from exc
    else:
        raise RuntimeError("E12c-v2 target already exists; write refused")
    print("CMI_FLU_E12C_V2_TARGET_ABSENT PASS write=false compute=false submission=false")


def materialize(runtime: Path, root: Path) -> Path:
    raw = runtime.read_bytes()
    if len(raw) >= 950000:
        raise RuntimeError("E12c-v2 runtime exceeds source budget")
    text = raw.decode("utf-8")
    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'TARGET_KERNEL = "{TARGET}"',
        f'SCIENCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'E12C_V2_BLOB = "{E12C_V2_BLOB}"',
        f'HISTORICAL_BACKBONE_SHA256 = "{HISTORICAL_SHA256}"',
        f'EXPECTED_TASK13_SHA256 = "{TASK13_SHA256}"',
        "CMI_FLU_E12C_V2_MANUAL_SUBMISSION_READY",
        "refit_unchanged=false",
    )
    if any(token not in text for token in required):
        raise RuntimeError("E12c-v2 runtime identity/output contract changed")
    lowered = text.casefold()
    if "competition_submit(" in lowered or "kaggle competitions submit" in lowered:
        raise RuntimeError("E12c-v2 runtime contains Competition submit path")

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


def reconcile_fresh_write(
    api: KaggleApi,
    response: object | None,
    failure: BaseException | None,
) -> str:
    failure_kind = None
    if failure is not None:
        failure_kind = "exception"
    elif str(getattr(response, "error", "") or ""):
        failure_kind = "response_error"

    try:
        meta = exact_metadata_eventually(api, TARGET, attempts=8, delay_seconds=2.0)
    except Exception as exc:
        info = safe_exception(exc)
        print(
            "CMI_FLU_E12C_V2_WRITE_UNRESOLVED "
            f"client_failure={failure_kind or 'unknown'} reconcile_http={info.get('http_status')} "
            f"reconcile_error_sha256={info.get('error_sha256')} write_calls=1 retries=0"
        )
        raise RuntimeError("E12c-v2 ambiguous_write reconciliation required") from exc

    validate_metadata(meta, TARGET, EXPECTED_VERSION, cpu=True)
    state = status_name(getattr(api.kernels_status(TARGET), "status", None))
    print(
        "CMI_FLU_E12C_V2_SAVEKERNEL OBSERVED "
        f"version={EXPECTED_VERSION} state={state} write_calls=1 retries=0 "
        f"client_failure={failure_kind or 'none'}"
    )
    return state


def wait_terminal(api: KaggleApi) -> str:
    for index in range(31):
        state = status_name(getattr(api.kernels_status(TARGET), "status", None))
        if state in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}:
            return state
        if index < 30:
            time.sleep(60)
    raise RuntimeError("E12c-v2 status polling exhausted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--path-self-test", action="store_true")
    args = parser.parse_args()

    cli = ensure_kaggle_cli_path()
    if args.path_self_test:
        print(
            "CMI_FLU_E12C_V2_OUTPUT_READER_PATH_SELF_TEST PASS "
            f"python_bin={Path(sys.executable).parent} cli_name={Path(cli).name} "
            "auth=false write=false compute=false"
        )
        return 0
    if args.runtime is None or args.output_dir is None:
        raise SystemExit("--runtime and --output-dir are required")
    if args.output_dir.exists():
        raise SystemExit("E12c-v2 output directory must be fresh")

    api = KaggleApi()
    api.authenticate()
    verify_historical_source(api)
    require_fresh_target(api)

    with tempfile.TemporaryDirectory(prefix="cmi-e12c-v2-run-") as tmp:
        kernel = materialize(args.runtime.resolve(), Path(tmp))
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
            raise RuntimeError(f"E12c-v2 Notebook did not complete successfully:{terminal}")

        read_current_output(
            kernel=TARGET,
            expected_version=EXPECTED_VERSION,
            allow=ALLOWED,
            output_dir=args.output_dir.resolve(),
        )

    print(
        "CMI_FLU_E12C_V2_EXECUTE PASS "
        "version=1 output_read=true refit_unchanged=false submission_api=false "
        "manual_operator_submission_only=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
