#!/usr/bin/env python3
"""Read terminal V3-07 H1/H2 kernel state and recover only declared H2 outputs.

Read-only by construction: no kernel push, version creation, Competition submission,
or selection mutation is present. Row-level H2 banks may exist only in the
runner-local output directory so the existing sanitizer can validate them; this
script never prints their contents.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelRequest,
    ApiListKernelSessionOutputRequest,
)

TARGETS = {
    "h1": {
        "condition": "task22_panel_mean",
        "owner": "renta0426",
        "slug": "cmi-flu-v3-v07-h1-panel-mean-20260912-001",
        "kernel": "renta0426/cmi-flu-v3-v07-h1-panel-mean-20260912-001",
        "version": 1,
    },
    "h2": {
        "condition": "task23_retention",
        "owner": "renta0426",
        "slug": "cmi-flu-v3-v07-h2-retention-20260912-001",
        "kernel": "renta0426/cmi-flu-v3-v07-h2-retention-20260912-001",
        "version": 1,
    },
}
FAIL_RE = re.compile(
    r"CMI_FLU_V307_RUNTIME_FAIL\s+stage=([A-Za-z0-9_]+)\s+type=([A-Za-z0-9_]+)\s+code=([0-9a-f]{20})"
)
H2_FILES = (
    "v3_v07_h2_oof_bank.csv",
    "v3_v07_h2_challenge_bank.csv",
    "v3_v07_h2_summary.json",
    "v3_v07_h2_manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def inspect(api: KaggleApi, key: str) -> tuple[dict, str]:
    target = TARGETS[key]
    # Private-kernel text search is not an identity primitive and can return zero
    # even when an exact private ref exists. Resolve the immutable owner/slug
    # directly, then validate the returned ref, privacy bit, and version.
    with api.build_kaggle_client() as client:
        meta_req = ApiGetKernelRequest()
        meta_req.user_name = target["owner"]
        meta_req.kernel_slug = target["slug"]
        metadata = client.kernels.kernels_api_client.get_kernel(meta_req).metadata
        if str(metadata.ref) != target["kernel"] or not bool(metadata.is_private):
            raise RuntimeError(f"{key} kernel identity/private mismatch")
        if int(metadata.current_version_number or 0) != int(target["version"]):
            raise RuntimeError(f"{key} current version changed")

        out_req = ApiListKernelSessionOutputRequest()
        out_req.user_name = target["owner"]
        out_req.kernel_slug = target["slug"]
        out_req.version_label = str(target["version"])
        out_req.page_size = 1000
        output = client.kernels.kernels_api_client.list_kernel_session_output(out_req)

    status = str(getattr(api.kernels_status(target["kernel"]), "status", "")).upper()
    if any(token in status for token in ("RUNNING", "QUEUED", "PENDING")):
        raise RuntimeError(f"{key} kernel is not terminal:{status}")

    log = str(getattr(output, "log", "") or "")
    failures = FAIL_RE.findall(log)
    pass_marker = "CMI_FLU_V307_RUNTIME_PASS" in log and "CMI_FLU_V307_COMPLETE" in log
    fail_marker = "CMI_FLU_V307_FAILED" in log or bool(failures)
    compact = {
        "condition": target["condition"],
        "kernel": target["kernel"],
        "version": int(target["version"]),
        "status": status,
        "runtime_pass": bool(pass_marker),
        "runtime_fail": bool(fail_marker),
        "failure": None,
    }
    if failures:
        stage, error_type, failure_code = failures[-1]
        compact["failure"] = {"stage": stage, "type": error_type, "code": failure_code}
    return compact, log


def recover_h2(api: KaggleApi, output_dir: Path) -> None:
    target = TARGETS["h2"]
    output_dir.mkdir(parents=True, exist_ok=False)
    pattern = r"(?:^|/)(?:v3_v07_h2_oof_bank\.csv|v3_v07_h2_challenge_bank\.csv|v3_v07_h2_summary\.json|v3_v07_h2_manifest\.json)$"
    _files, token = api.kernels_output(
        target["kernel"],
        str(output_dir),
        file_pattern=pattern,
        force=True,
        quiet=True,
        page_size=1000,
    )
    if token:
        raise RuntimeError("H2 output read returned unexpected pagination token")
    found: dict[str, list[Path]] = {}
    for path in output_dir.rglob("*"):
        if path.is_file() and path.name in H2_FILES:
            found.setdefault(path.name, []).append(path)
    for name in H2_FILES:
        paths = found.get(name, [])
        if len(paths) != 1:
            raise RuntimeError(f"H2 declared output missing or ambiguous:{name}")
        src = paths[0]
        dst = output_dir / name
        if src != dst:
            dst.write_bytes(src.read_bytes())
    for path in list(output_dir.rglob("*")):
        if path.is_file() and path.parent != output_dir:
            path.unlink()
    for path in sorted(output_dir.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def main() -> int:
    args = parse_args()
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_"):
        raise SystemExit("KAGGLE_API_TOKEN contract failed")
    api = KaggleApi()
    api.authenticate()
    h1, _h1_log = inspect(api, "h1")
    h2, _h2_log = inspect(api, "h2")
    recovered = False
    if h2["runtime_pass"] and not h2["runtime_fail"]:
        recover_h2(api, args.output_dir.resolve())
        recovered = True
    result = {
        "request_id": "20260913-cmi-flu-v3-v07-consumed-readout-002",
        "predecessor_request_id": "20260913-cmi-flu-v3-v07-consumed-readout-001",
        "failure_classification_repaired": "bridge_runtime_compatibility_private_kernel_search_not_identity_primitive",
        "side_effects": [],
        "competition_submission_attempted": False,
        "final_submission_selection_attempted": False,
        "h1": h1,
        "h2": h2,
        "h2_declared_outputs_recovered": recovered,
    }
    print("CMI_FLU_V307_CONSUMED_READOUT " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
