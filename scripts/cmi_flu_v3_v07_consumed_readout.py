#!/usr/bin/env python3
"""Read exact V3-07 version-1 session outputs without GetKernel metadata.

Read-only by construction: no kernel push, version creation, Competition submission,
or selection mutation is present. Row-level banks may exist only in the runner-local
output directory so the existing sanitizer can validate them; this script never
prints their contents or signed output URLs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlparse

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

TARGETS = {
    "h1": {
        "condition": "task22_panel_mean",
        "owner": "renta0426",
        "slug": "cmi-flu-v3-v07-h1-panel-mean-20260912-001",
        "kernel": "renta0426/cmi-flu-v3-v07-h1-panel-mean-20260912-001",
        "version": 1,
        "prefix": "v3_v07_h1",
    },
    "h2": {
        "condition": "task23_retention",
        "owner": "renta0426",
        "slug": "cmi-flu-v3-v07-h2-retention-20260912-001",
        "kernel": "renta0426/cmi-flu-v3-v07-h2-retention-20260912-001",
        "version": 1,
        "prefix": "v3_v07_h2",
    },
}
FAIL_RE = re.compile(
    r"CMI_FLU_V307_RUNTIME_FAIL\s+stage=([A-Za-z0-9_]+)\s+type=([A-Za-z0-9_]+)\s+code=([0-9a-f]{20})"
)
MAX_FILE_BYTES = 16 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def expected_files(key: str) -> tuple[str, ...]:
    prefix = TARGETS[key]["prefix"]
    return (
        f"{prefix}_oof_bank.csv",
        f"{prefix}_challenge_bank.csv",
        f"{prefix}_summary.json",
        f"{prefix}_manifest.json",
    )


def exact_session_output(api: KaggleApi, key: str):
    """Read only the explicitly requested immutable version label.

    GetKernel is deliberately not used: the protected KGAT credential can push the
    private target but Kaggle currently returns HTTP 403 for GetKernel. There is no
    latest/current fallback here; version_label is fixed to the frozen request's v1.
    """
    target = TARGETS[key]
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = target["owner"]
        request.kernel_slug = target["slug"]
        request.version_label = str(target["version"])
        request.page_size = 1000
        return client.kernels.kernels_api_client.list_kernel_session_output(request)


def inspect(api: KaggleApi, key: str) -> tuple[dict, object]:
    target = TARGETS[key]
    output = exact_session_output(api, key)
    log = str(getattr(output, "log", "") or "")
    failures = FAIL_RE.findall(log)
    pass_marker = "CMI_FLU_V307_RUNTIME_PASS" in log and "CMI_FLU_V307_COMPLETE" in log
    fail_marker = "CMI_FLU_V307_FAILED" in log or bool(failures)
    marker_state = "COMPLETE" if pass_marker and not fail_marker else ("ERROR" if fail_marker else "UNKNOWN")
    compact = {
        "condition": target["condition"],
        "kernel": target["kernel"],
        "version": int(target["version"]),
        "version_read_method": "exact_session_output_version_label",
        "marker_state": marker_state,
        "runtime_pass": bool(pass_marker and not fail_marker),
        "runtime_fail": bool(fail_marker),
        "failure": None,
    }
    if failures:
        stage, error_type, failure_code = failures[-1]
        compact["failure"] = {"stage": stage, "type": error_type, "code": failure_code}
    return compact, output


def _signed_output_host_ok(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    return parsed.scheme == "https" and (
        host == "storage.googleapis.com"
        or host.endswith(".storage.googleapis.com")
        or host == "googleusercontent.com"
        or host.endswith(".googleusercontent.com")
        or host == "kaggleusercontent.com"
        or host.endswith(".kaggleusercontent.com")
    )


def recover_declared(output: object, key: str, output_dir: Path) -> None:
    wanted = expected_files(key)
    candidates: dict[str, list[str]] = {name: [] for name in wanted}
    for item in list(getattr(output, "files", None) or []):
        raw_name = str(getattr(item, "file_name", "") or "").replace("\\", "/")
        name = PurePosixPath(raw_name).name
        if name in candidates:
            url = str(getattr(item, "url", "") or "")
            candidates[name].append(url)

    chosen: dict[str, str] = {}
    for name in wanted:
        urls = [url for url in candidates[name] if url]
        if len(urls) != 1:
            raise RuntimeError(f"{key} declared output missing or ambiguous:{name}")
        if not _signed_output_host_ok(urls[0]):
            raise RuntimeError(f"{key} unexpected signed output host:{name}")
        chosen[name] = urls[0]

    output_dir.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.trust_env = False
    for name, url in chosen.items():
        response = session.get(url, timeout=60, allow_redirects=False)
        if response.status_code != 200:
            raise RuntimeError(f"{key} signed output download failed:{name}:http{response.status_code}")
        if len(response.content) > MAX_FILE_BYTES:
            raise RuntimeError(f"{key} declared output exceeds byte budget:{name}")
        (output_dir / name).write_bytes(response.content)

    found = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    if found != sorted(wanted):
        raise RuntimeError(f"{key} recovered file set mismatch:{found}")


def main() -> int:
    args = parse_args()
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_"):
        raise SystemExit("KAGGLE_API_TOKEN contract failed")
    api = KaggleApi()
    api.authenticate()

    root = args.output_dir.resolve()
    h1, h1_output = inspect(api, "h1")
    h2, h2_output = inspect(api, "h2")
    recovered = {"h1": False, "h2": False}
    for key, compact, output in (("h1", h1, h1_output), ("h2", h2, h2_output)):
        if compact["runtime_pass"] and not compact["runtime_fail"]:
            recover_declared(output, key, root / key)
            recovered[key] = True

    result = {
        "request_id": "20260913-cmi-flu-v3-v07-consumed-readout-003",
        "predecessor_request_id": "20260913-cmi-flu-v3-v07-consumed-readout-002",
        "failure_classification_repaired": "bridge_runtime_compatibility_getkernel_forbidden",
        "side_effects": [],
        "competition_submission_attempted": False,
        "final_submission_selection_attempted": False,
        "h1": h1,
        "h2": h2,
        "declared_outputs_recovered": recovered,
    }
    print("CMI_FLU_V307_CONSUMED_READOUT " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
