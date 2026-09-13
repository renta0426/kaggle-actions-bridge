#!/usr/bin/env python3
"""Read only safe V3-07 failure details from the two consumed -002 kernels."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess

from kaggle.api.kaggle_api_extended import KaggleApi

REQUEST_ID = "20260914-cmi-flu-v3-v07-error-log-readout-002"
TARGETS = {
    "h1": {
        "ref": "renta0426/cmi-flu-v3-07-h1-panel-mean-20260913-002",
        "expected": {"stage": "run_v307", "type": "DataContractError", "code": "f26a5256a19c01b6f0a6"},
    },
    "h2": {
        "ref": "renta0426/cmi-flu-v3-07-h2-retention-20260913-002",
        "expected": {"stage": "run_v307", "type": "KeyError", "code": "eb03498a8ceb92036869"},
    },
}
FAIL_RE = re.compile(
    r"CMI_FLU_V307_RUNTIME_FAIL\s+stage=([A-Za-z0-9_]+)\s+type=([A-Za-z0-9_]+)\s+code=([0-9a-f]{20})"
)
DATA_ERROR_RE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_.]*\.)?DataContractError:\s+(V3-07[^\r\n]{0,500})")
KEY_ERROR_SIMPLE_RE = re.compile(r"KeyError:\s+['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]")
KEY_ERROR_INDEX_RE = re.compile(r"KeyError:\s+\"\[([^\r\n]{1,300})\] not in index\"")
SAFE_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def safe_status(api: KaggleApi, ref: str) -> tuple[str, bool]:
    try:
        raw = str(getattr(api.kernels_status(ref), "status", "") or "").upper()
        return raw, "ERROR" in raw
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}:{digest(str(exc))[:16]}", False


def sanitize_exception_detail(combined: str, error_type: str) -> dict[str, object] | None:
    if error_type == "DataContractError":
        matches = DATA_ERROR_RE.findall(combined)
        if not matches:
            return None
        detail = matches[-1].strip()
        if not detail.startswith("V3-07") or len(detail) > 500:
            return None
        if any(token in detail.casefold() for token in ("participant_id", "subject_group=", "prediction", "/kaggle/")):
            return None
        return {"kind": "data_contract", "message": detail}
    if error_type == "KeyError":
        simple = KEY_ERROR_SIMPLE_RE.findall(combined)
        if simple:
            key = simple[-1]
            if SAFE_FIELD_RE.fullmatch(key):
                return {"kind": "missing_field", "field": key}
        indexed = KEY_ERROR_INDEX_RE.findall(combined)
        if indexed:
            fields = re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", indexed[-1])
            fields = [field for field in fields if SAFE_FIELD_RE.fullmatch(field)]
            if fields and len(fields) <= 8:
                return {"kind": "missing_fields", "fields": fields}
    return None


def read_one(api: KaggleApi, kaggle_cli: str, key: str, target: dict[str, object]) -> dict[str, object]:
    ref = str(target["ref"])
    expected = dict(target["expected"])
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
    marker_matches = marker == expected
    detail = sanitize_exception_detail(combined, str(expected["type"])) if marker_matches else None
    return {
        "condition": key,
        "kernel": ref,
        "latest_status": status,
        "latest_status_is_error": is_error,
        "log_cli_return_code": int(completed.returncode),
        "log_bytes": len(combined.encode("utf-8", errors="replace")),
        "log_sha256": digest(combined),
        "failure_marker": marker,
        "failure_marker_matches_expected": marker_matches,
        "sanitized_exception_detail": detail,
    }


def main() -> int:
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_"):
        raise SystemExit("KAGGLE_API_TOKEN contract failed")
    kaggle_cli = shutil.which("kaggle")
    if not kaggle_cli:
        raise SystemExit("locked Kaggle CLI missing")
    api = KaggleApi(); api.authenticate()
    result = {key: read_one(api, kaggle_cli, key, target) for key, target in TARGETS.items()}
    payload = {
        "request_id": REQUEST_ID,
        "side_effects": [],
        "competition_submission_attempted": False,
        "final_submission_selection_attempted": False,
        "targets": result,
    }
    print("CMI_FLU_V307_ERROR_LOG_READOUT " + json.dumps(payload, sort_keys=True))
    if any(not item["failure_marker_matches_expected"] for item in result.values()):
        return 2
    if any(item["sanitized_exception_detail"] is None for item in result.values()):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
