#!/usr/bin/env python3
"""Read only safe V3-07 failure markers from the two resolved Kaggle kernels."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess

from kaggle.api.kaggle_api_extended import KaggleApi

REQUEST_ID = "20260913-cmi-flu-v3-v07-error-log-readout-001"
TARGETS = {
    "h1": "renta0426/cmi-flu-v3-07-h1-panel-mean-20260912-001",
    "h2": "renta0426/cmi-flu-v3-07-h2-retention-20260912-001",
}
FAIL_RE = re.compile(
    r"CMI_FLU_V307_RUNTIME_FAIL\s+stage=([A-Za-z0-9_]+)\s+type=([A-Za-z0-9_]+)\s+code=([0-9a-f]{20})"
)
FIT_BUDGET_MESSAGE = "run_v307:DataContractError:V3-07 fit budget exceeded: 257>256"
FIT_BUDGET_CODE = hashlib.sha256(FIT_BUDGET_MESSAGE.encode("utf-8")).hexdigest()[:20]
KNOWN_H1_CODE = "445d164432e5ff8e94cd"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def safe_status(api: KaggleApi, ref: str) -> tuple[str, bool]:
    try:
        raw = str(getattr(api.kernels_status(ref), "status", "") or "").upper()
        return raw, "ERROR" in raw
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}:{digest(str(exc))[:16]}", False


def read_one(api: KaggleApi, kaggle_cli: str, key: str, ref: str) -> dict:
    status, is_error = safe_status(api, ref)
    completed = subprocess.run(
        [kaggle_cli, "kernels", "logs", ref],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=os.environ.copy(),
    )
    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    markers = FAIL_RE.findall(combined)
    marker = None
    if markers:
        stage, error_type, code = markers[-1]
        marker = {"stage": stage, "type": error_type, "code": code}
    code = None if marker is None else marker["code"]
    category = "fit_budget_exceeded_257_gt_256" if code == FIT_BUDGET_CODE else (
        "runtime_failure_marker_other" if marker is not None else "runtime_failure_marker_unavailable"
    )
    return {
        "condition": key,
        "kernel": ref,
        "latest_status": status,
        "latest_status_is_error": is_error,
        "log_cli_return_code": int(completed.returncode),
        "log_bytes": len(combined.encode("utf-8", errors="replace")),
        "log_sha256": digest(combined),
        "failure_marker": marker,
        "safe_category": category,
        "matches_known_fit_budget_code": bool(code == FIT_BUDGET_CODE),
        "matches_prior_h1_code": bool(code == KNOWN_H1_CODE),
    }


def main() -> int:
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_"):
        raise SystemExit("KAGGLE_API_TOKEN contract failed")
    if FIT_BUDGET_CODE != KNOWN_H1_CODE:
        raise SystemExit("known H1 failure-code contract changed")
    kaggle_cli = shutil.which("kaggle")
    if not kaggle_cli:
        raise SystemExit("locked Kaggle CLI missing")
    api = KaggleApi(); api.authenticate()
    result = {key: read_one(api, kaggle_cli, key, ref) for key, ref in TARGETS.items()}
    payload = {
        "request_id": REQUEST_ID,
        "side_effects": [],
        "competition_submission_attempted": False,
        "final_submission_selection_attempted": False,
        "known_fit_budget_code": FIT_BUDGET_CODE,
        "targets": result,
    }
    print("CMI_FLU_V307_ERROR_LOG_READOUT " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
