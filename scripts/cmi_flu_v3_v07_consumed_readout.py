#!/usr/bin/env python3
"""Recover consumed V3-07 v1 outputs after resolving Kaggle's actual kernel refs.

The original writes were acknowledged by `kaggle kernels push`, but later reads using
our requested slugs returned HTTP 403. Kaggle upstream documents that an incorrect
kernel slug may surface as 401/403 rather than 404. This read-only successor therefore
lists only the authenticated owner's recent kernels once, resolves each frozen exact
title uniquely, and then addresses version 1 of that resolved ref directly.

No kernel write, compute retry, Competition submission, Final Submission selection,
public artifact, or row-level log emission exists here. Row-level banks, when
recoverable, remain runner-local for the existing aggregate sanitizer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlparse

import requests
from requests.exceptions import HTTPError
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
from kagglesdk.security.types.oauth_service import IntrospectTokenRequest

OWNER = "renta0426"
TARGETS = {
    "h1": {
        "condition": "task22_panel_mean",
        "title": "CMI Flu V3-07 H1 Panel Mean 20260912 001",
        "planned_kernel": "renta0426/cmi-flu-v3-v07-h1-panel-mean-20260912-001",
        "version": 1,
        "prefix": "v3_v07_h1",
    },
    "h2": {
        "condition": "task23_retention",
        "title": "CMI Flu V3-07 H2 Retention 20260912 001",
        "planned_kernel": "renta0426/cmi-flu-v3-v07-h2-retention-20260912-001",
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


def resolve_exact_titles(api: KaggleApi) -> dict[str, str]:
    """Resolve actual refs from a single owner-scoped recent-kernel listing.

    No slug search is used. Exact frozen titles must each appear exactly once among
    the owner's 100 most recently run kernels. Unrelated kernel identities are never
    emitted.
    """
    items = api.kernels_list(user=OWNER, page_size=100, sort_by="dateRun") or []
    resolved: dict[str, str] = {}
    for key, target in TARGETS.items():
        matches: list[str] = []
        for item in items:
            if str(getattr(item, "title", "") or "") != target["title"]:
                continue
            ref = str(getattr(item, "ref", "") or "")
            if ref.startswith(f"{OWNER}/") and ref.count("/") == 1:
                matches.append(ref)
        matches = sorted(set(matches))
        if len(matches) != 1:
            raise RuntimeError(f"{key} exact-title resolution mismatch count={len(matches)}")
        resolved[key] = matches[0]
    return resolved


def token_scope_observation(api: KaggleApi, token: str) -> dict:
    """Emit only boolean scope hints; never emit token/scope strings."""
    try:
        with api.build_kaggle_client() as client:
            request = IntrospectTokenRequest()
            request.token = token
            response = client.security.oauth_client.introspect_token(request)
        values: list[str] = []
        for name in ("scope", "scopes", "permissions", "roles"):
            value = getattr(response, name, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                values.extend(str(x) for x in value)
            else:
                values.append(str(value))
        text = " ".join(values).casefold()
        return {
            "available": bool(values),
            "active": bool(getattr(response, "active", False)),
            "username_matches_owner": str(getattr(response, "username", "") or "") == OWNER,
            "kernels_get_visible": "kernels.get" in text,
            "kernels_viewer_visible": "kernels.viewer" in text,
            "kernels_editor_visible": "kernels.editor" in text,
            "resources_admin_visible": "resources.admin" in text,
        }
    except Exception as exc:
        return {"available": False, "error_type": type(exc).__name__}


def exact_session_output(api: KaggleApi, key: str, resolved_ref: str):
    owner, slug = resolved_ref.split("/", 1)
    if owner != OWNER:
        raise RuntimeError(f"{key} resolved owner mismatch")
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner
        request.kernel_slug = slug
        request.version_label = str(TARGETS[key]["version"])
        request.page_size = 1000
        return client.kernels.kernels_api_client.list_kernel_session_output(request)


def diagnostic_latest_state(api: KaggleApi, key: str, resolved_ref: str) -> dict:
    """Observe distinct read endpoints only after exact v1 session read is denied.

    Current/latest metadata is diagnostic only and is never accepted as a substitute
    for the frozen v1 scientific output.
    """
    result: dict[str, object] = {}
    try:
        status = str(getattr(api.kernels_status(resolved_ref), "status", "") or "").upper()
        result["latest_status"] = status or "UNKNOWN"
    except Exception as exc:
        result["latest_status_error_type"] = type(exc).__name__
        if isinstance(exc, HTTPError) and exc.response is not None:
            result["latest_status_http"] = int(exc.response.status_code)
    try:
        listing = api.kernels_list_files(resolved_ref, page_size=100)
        files = list(getattr(listing, "files", None) or getattr(listing, "dataset_files", None) or [])
        names: set[str] = set()
        for item in files:
            raw = str(getattr(item, "name", "") or getattr(item, "file_name", "") or "").replace("\\", "/")
            if raw:
                names.add(PurePosixPath(raw).name)
        expected = set(expected_files(key))
        result["latest_file_listing_accessible"] = True
        result["latest_expected_files_present"] = sorted(expected.intersection(names))
        result["latest_expected_file_count"] = len(expected.intersection(names))
        result["latest_total_file_count"] = len(names)
    except Exception as exc:
        result["latest_file_listing_accessible"] = False
        result["latest_file_listing_error_type"] = type(exc).__name__
        if isinstance(exc, HTTPError) and exc.response is not None:
            result["latest_file_listing_http"] = int(exc.response.status_code)
    return result


def inspect(api: KaggleApi, key: str, resolved_ref: str) -> tuple[dict, object | None]:
    target = TARGETS[key]
    compact: dict[str, object] = {
        "condition": target["condition"],
        "planned_kernel": target["planned_kernel"],
        "resolved_kernel": resolved_ref,
        "planned_ref_matches_resolved": resolved_ref == target["planned_kernel"],
        "version": int(target["version"]),
        "version_read_method": "exact_title_resolution_then_session_output_version_label",
        "marker_state": "UNREAD",
        "runtime_pass": False,
        "runtime_fail": False,
        "failure": None,
    }
    try:
        output = exact_session_output(api, key, resolved_ref)
    except Exception as exc:
        compact["exact_v1_read_error_type"] = type(exc).__name__
        if isinstance(exc, HTTPError) and exc.response is not None:
            compact["exact_v1_read_http"] = int(exc.response.status_code)
        compact["diagnostic_latest_only_not_scientific_output"] = diagnostic_latest_state(api, key, resolved_ref)
        return compact, None

    log = str(getattr(output, "log", "") or "")
    failures = FAIL_RE.findall(log)
    pass_marker = "CMI_FLU_V307_RUNTIME_PASS" in log and "CMI_FLU_V307_COMPLETE" in log
    fail_marker = "CMI_FLU_V307_FAILED" in log or bool(failures)
    compact["marker_state"] = "COMPLETE" if pass_marker and not fail_marker else ("ERROR" if fail_marker else "UNKNOWN")
    compact["runtime_pass"] = bool(pass_marker and not fail_marker)
    compact["runtime_fail"] = bool(fail_marker)
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

    resolved = resolve_exact_titles(api)
    auth = token_scope_observation(api, token)
    root = args.output_dir.resolve()
    states: dict[str, dict] = {}
    recovered = {"h1": False, "h2": False}
    for key in ("h1", "h2"):
        compact, output = inspect(api, key, resolved[key])
        states[key] = compact
        if output is not None and compact["runtime_pass"] and not compact["runtime_fail"]:
            recover_declared(output, key, root / key)
            recovered[key] = True

    result = {
        "request_id": "20260913-cmi-flu-v3-v07-consumed-readout-004",
        "predecessor_request_id": "20260913-cmi-flu-v3-v07-consumed-readout-003",
        "repair_hypothesis": "kaggle_403_can_mask_wrong_kernel_slug_resolve_by_frozen_exact_title",
        "auth_scope_observation": auth,
        "side_effects": [],
        "competition_submission_attempted": False,
        "final_submission_selection_attempted": False,
        "h1": states["h1"],
        "h2": states["h2"],
        "declared_outputs_recovered": recovered,
    }
    print("CMI_FLU_V307_CONSUMED_READOUT " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
