#!/usr/bin/env python3
"""Read only the sanitized V3 runtime-failure marker from one ERROR Notebook."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

TARGET = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260912-002"
EXPECTED_VERSION = 1
EXPECTED_STATE = "ERROR"
MAX_LOG_FILES = 8
MAX_TOTAL_LOG_BYTES = 5 * 1024 * 1024
MARKER_RE = re.compile(
    r"CMI_FLU_V3_BATCH1_RUNTIME_FAIL stage=([A-Za-z0-9_]+) "
    r"type=([A-Za-z0-9_.]+) code=([0-9a-f]{20})"
)


def extract_marker(text: str) -> tuple[str, str, str]:
    matches = MARKER_RE.findall(text)
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(f"runtime_fail_marker_count={len(unique)}")
    return unique[0]


def self_test() -> int:
    sample = (
        "ignored private text\n"
        "CMI_FLU_V3_BATCH1_RUNTIME_FAIL stage=run_no_fit_audit "
        "type=KeyError code=0123456789abcdef0123\n"
        "ignored traceback\n"
    )
    marker = extract_marker(sample)
    if marker != ("run_no_fit_audit", "KeyError", "0123456789abcdef0123"):
        raise RuntimeError("marker_parser_mismatch")
    try:
        extract_marker("no marker")
    except RuntimeError:
        pass
    else:
        raise RuntimeError("missing_marker_not_rejected")
    print("CMI_FLU_V3_BATCH1_ERROR_SALVAGE_SELF_TEST PASS auth=false write=false compute=false submission=false")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    from kaggle.api.kaggle_api_extended import KaggleApi
    from kaggle_exact_identity import exact_metadata_eventually, status_name, validate_metadata

    api = KaggleApi(); api.authenticate()
    meta = exact_metadata_eventually(api, TARGET, attempts=6, delay_seconds=2.0)
    validate_metadata(meta, TARGET, EXPECTED_VERSION, cpu=True)
    state = status_name(getattr(api.kernels_status(TARGET), "status", None))
    if state != EXPECTED_STATE:
        raise RuntimeError(f"target_state_mismatch:{state}")

    kaggle_cli = shutil.which("kaggle")
    if not kaggle_cli:
        raise RuntimeError("official_kaggle_cli_missing")

    with tempfile.TemporaryDirectory(prefix="cmi-v3-error-salvage-") as tmp:
        root = Path(tmp) / "download"; root.mkdir()
        completed = subprocess.run(
            [kaggle_cli, "kernels", "output", TARGET, "-p", str(root)],
            check=False, capture_output=True, text=True, timeout=180, env=os.environ.copy(),
        )
        if completed.returncode != 0:
            digest = hashlib.sha256((completed.stdout + completed.stderr).encode()).hexdigest()
            raise RuntimeError(f"failed_log_output_cli rc={completed.returncode} diagnostic_sha256={digest}")
        validate_metadata(
            exact_metadata_eventually(api, TARGET, attempts=6, delay_seconds=2.0),
            TARGET,
            EXPECTED_VERSION,
            cpu=True,
        )
        state_after = status_name(getattr(api.kernels_status(TARGET), "status", None))
        if state_after != EXPECTED_STATE:
            raise RuntimeError(f"target_state_changed:{state_after}")

        paths = list(root.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise RuntimeError("symlink_in_failed_output")
        logs = [path for path in paths if path.is_file() and path.suffix == ".log"]
        if not 1 <= len(logs) <= MAX_LOG_FILES:
            raise RuntimeError(f"failed_log_file_count={len(logs)}")
        total = sum(path.stat().st_size for path in logs)
        if not 0 < total <= MAX_TOTAL_LOG_BYTES:
            raise RuntimeError("failed_log_byte_contract_violated")
        markers = []
        for path in logs:
            markers.append(extract_marker(path.read_text(encoding="utf-8", errors="replace")))
        unique = sorted(set(markers))
        if len(unique) != 1:
            raise RuntimeError(f"cross_log_marker_count={len(unique)}")
        stage, error_type, code = unique[0]
        log_fingerprint = hashlib.sha256(
            "\n".join(sorted(hashlib.sha256(path.read_bytes()).hexdigest() for path in logs)).encode()
        ).hexdigest()
        print(
            "CMI_FLU_V3_BATCH1_ERROR_SALVAGE PASS "
            f"version={EXPECTED_VERSION} state={EXPECTED_STATE} stage={stage} type={error_type} code={code} "
            f"log_files={len(logs)} log_bytes={total} log_fingerprint={log_fingerprint} "
            "write=false compute=false submission=false content_exposed=false"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
