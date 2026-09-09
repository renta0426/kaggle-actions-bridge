#!/usr/bin/env python3
"""Recover the exact historical Starter++ token-stat shards for P1-03.

The request must pin both current Kaggle kernel versions before this script may
list or download outputs.  Search/latest substitution, write operations, and
compute retries are intentionally absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kaggle_exact_identity import IdentityError, exact_metadata, safe_exception, status_name


OPERATION = "recover_exact_sersem_token_cache"
SCIENCE_SHA = "44e602d1e61c5b5c603283c7d68a480ba626de70"
OUTPUT_SUFFIX = "starter_plus_10k/parts/token_statistics.part{shard:03d}.parquet"


def _validate_request(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema_version") != 1 or data.get("operation") != OPERATION:
        raise IdentityError("invalid_recovery_request")
    if data.get("science_sha") != SCIENCE_SHA:
        raise IdentityError("science_sha_mismatch")
    if data.get("environment") != "kaggle-readonry":
        raise IdentityError("invalid_recovery_environment")
    if data.get("output_suffix_template") != OUTPUT_SUFFIX:
        raise IdentityError("invalid_output_suffix_contract")
    if data.get("expected_total_shards") != 40:
        raise IdentityError("invalid_total_shard_contract")
    if data.get("version_substitution_allowed") is not False:
        raise IdentityError("version_substitution_forbidden")
    if data.get("automatic_retries") != 0 or data.get("kaggle_write_calls") != 0:
        raise IdentityError("invalid_recovery_side_effect_contract")
    if data.get("kaggle_compute_started") is not False:
        raise IdentityError("compute_must_remain_disabled")
    if data.get("membership_sources_allowed") != ["public_train_only_after_credential_removal"]:
        raise IdentityError("invalid_membership_source_contract")
    if data.get("performance_evaluation_allowed") is not True:
        raise IdentityError("performance_evaluation_contract_missing")

    budget = data.get("api_budget")
    if budget != {
        "get_kernel": 2,
        "status": 2,
        "max_list_pages_per_target": 3,
        "downloads": 40,
    }:
        raise IdentityError("invalid_recovery_api_budget")
    if data.get("max_file_bytes") != 67_108_864:
        raise IdentityError("invalid_file_byte_budget")
    if data.get("max_total_bytes") != 2_147_483_648:
        raise IdentityError("invalid_total_byte_budget")

    targets = data.get("targets")
    if not isinstance(targets, dict) or set(targets) != {"primary", "continuation"}:
        raise IdentityError("invalid_recovery_targets")
    primary = targets["primary"]
    if primary != {
        "kernel": "renta0426/starter-plus-10k-v2",
        "expected_current_version": 1,
        "expected_status": "ERROR",
        "shard_start": 0,
        "shard_end": 23,
    }:
        raise IdentityError("invalid_primary_recovery_contract")
    continuation = targets["continuation"]
    if not isinstance(continuation, dict):
        raise IdentityError("invalid_continuation_recovery_contract")
    if continuation.get("kernel") != "renta0426/starter-plus-10k-v2-continuation":
        raise IdentityError("invalid_continuation_kernel")
    version = continuation.get("expected_current_version")
    if isinstance(version, bool) or type(version) is not int or version < 1:
        raise IdentityError("continuation_version_not_frozen")
    if continuation.get("expected_status") != "COMPLETE":
        raise IdentityError("continuation_status_not_frozen")
    if continuation.get("shard_start") != 24 or continuation.get("shard_end") != 39:
        raise IdentityError("invalid_continuation_shards")
    return data


def _load_request(path: Path) -> dict[str, Any]:
    return _validate_request(json.loads(path.read_text(encoding="utf-8")))


def _expected_suffixes(target: dict[str, Any]) -> list[tuple[int, str]]:
    return [
        (shard, OUTPUT_SUFFIX.format(shard=shard))
        for shard in range(target["shard_start"], target["shard_end"] + 1)
    ]


def _select_exact_outputs(
    records: list[dict[str, Any]],
    target: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for shard, suffix in _expected_suffixes(target):
        matches = [record for record in records if record["name"].endswith(suffix)]
        if len(matches) != 1:
            raise IdentityError("required_token_shard_missing_or_ambiguous")
        selected.append((shard, matches[0]))
    return selected


def _observe_and_list(api: Any, target: dict[str, Any], max_pages: int) -> list[dict[str, Any]]:
    metadata = exact_metadata(api, target["kernel"])
    if getattr(metadata, "ref", None) != target["kernel"]:
        raise IdentityError("kernel_ref_mismatch")
    if getattr(metadata, "is_private", None) is not True:
        raise IdentityError("private_flag_not_proven")
    observed = getattr(metadata, "current_version_number", None)
    if isinstance(observed, bool) or observed is None or int(observed) != target["expected_current_version"]:
        raise IdentityError("current_version_mismatch")
    state = status_name(getattr(api.kernels_status(target["kernel"]), "status", None))
    if state != target["expected_status"]:
        raise IdentityError("kernel_status_mismatch")

    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    owner, slug = target["kernel"].split("/", 1)
    files: list[Any] = []
    page_token = ""
    with api.build_kaggle_client() as client:
        for page in range(max_pages):
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            request.page_size = 100
            request.page_token = page_token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            files.extend(list(getattr(response, "files", None) or []))
            page_token = str(getattr(response, "next_page_token", "") or "")
            if not page_token:
                break
        else:
            if page_token:
                raise IdentityError("output_inventory_exceeds_page_budget")

    records: list[dict[str, Any]] = []
    for item in files:
        name = str(getattr(item, "file_name", "") or "").replace("\\", "/")
        url = str(getattr(item, "url", "") or "")
        size = int(getattr(item, "total_bytes", 0) or getattr(item, "size", 0) or 0)
        if name and url:
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.hostname:
                raise IdentityError("non_https_output_url")
            records.append({"name": name, "url": url, "bytes": size})
    return records


def _download_selected(
    selected: list[tuple[int, dict[str, Any]]],
    output_dir: Path,
    *,
    max_file_bytes: int,
    remaining_total_bytes: int,
) -> tuple[list[dict[str, Any]], int]:
    import requests

    manifest: list[dict[str, Any]] = []
    total = 0
    for shard, record in selected:
        destination = output_dir / f"token_statistics.part{shard:03d}.parquet"
        digest = hashlib.sha256()
        written = 0
        with requests.get(record["url"], stream=True, timeout=(30, 120), allow_redirects=True) as response:
            response.raise_for_status()
            final = urlparse(str(response.url))
            if final.scheme != "https" or not final.hostname:
                raise IdentityError("non_https_output_redirect")
            with destination.open("wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if not block:
                        continue
                    written += len(block)
                    if written > max_file_bytes:
                        raise IdentityError("token_shard_exceeds_file_byte_budget")
                    if total + written > remaining_total_bytes:
                        raise IdentityError("token_cache_exceeds_total_byte_budget")
                    digest.update(block)
                    handle.write(block)
        if written <= 0:
            raise IdentityError("empty_token_shard")
        total += written
        manifest.append(
            {
                "shard": shard,
                "local_name": destination.name,
                "bytes": written,
                "sha256": digest.hexdigest(),
            }
        )
    return manifest, total


def _self_test() -> None:
    request = {
        "schema_version": 1,
        "operation": OPERATION,
        "science_sha": SCIENCE_SHA,
        "environment": "kaggle-readonry",
        "output_suffix_template": OUTPUT_SUFFIX,
        "expected_total_shards": 40,
        "version_substitution_allowed": False,
        "automatic_retries": 0,
        "kaggle_write_calls": 0,
        "kaggle_compute_started": False,
        "membership_sources_allowed": ["public_train_only_after_credential_removal"],
        "performance_evaluation_allowed": True,
        "api_budget": {
            "get_kernel": 2,
            "status": 2,
            "max_list_pages_per_target": 3,
            "downloads": 40,
        },
        "max_file_bytes": 67_108_864,
        "max_total_bytes": 2_147_483_648,
        "targets": {
            "primary": {
                "kernel": "renta0426/starter-plus-10k-v2",
                "expected_current_version": 1,
                "expected_status": "ERROR",
                "shard_start": 0,
                "shard_end": 23,
            },
            "continuation": {
                "kernel": "renta0426/starter-plus-10k-v2-continuation",
                "expected_current_version": 3,
                "expected_status": "COMPLETE",
                "shard_start": 24,
                "shard_end": 39,
            },
        },
    }
    _validate_request(request)
    records = [
        {"name": f"x/{OUTPUT_SUFFIX.format(shard=i)}", "url": "https://example.invalid/x", "bytes": 1}
        for i in range(40)
    ]
    first = _select_exact_outputs(records, request["targets"]["primary"])
    second = _select_exact_outputs(records, request["targets"]["continuation"])
    assert [x[0] for x in first] == list(range(24))
    assert [x[0] for x in second] == list(range(24, 40))
    print("SERSEM_CACHE_RECOVERY_SELF_TEST PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if args.request is None or args.output_dir is None:
        raise SystemExit("--request and --output-dir are required")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        request = _load_request(args.request)
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        all_manifest: list[dict[str, Any]] = []
        total_bytes = 0
        for name in ("primary", "continuation"):
            target = request["targets"][name]
            records = _observe_and_list(
                api,
                target,
                request["api_budget"]["max_list_pages_per_target"],
            )
            selected = _select_exact_outputs(records, target)
            downloaded, used = _download_selected(
                selected,
                args.output_dir,
                max_file_bytes=request["max_file_bytes"],
                remaining_total_bytes=request["max_total_bytes"] - total_bytes,
            )
            total_bytes += used
            all_manifest.extend({"source": name, **item} for item in downloaded)

        if len(all_manifest) != request["expected_total_shards"]:
            raise IdentityError("token_shard_count_mismatch")
        if sorted(item["shard"] for item in all_manifest) != list(range(40)):
            raise IdentityError("token_shard_index_mismatch")
        manifest = {
            "request_id": request["request_id"],
            "science_sha": request["science_sha"],
            "shards": all_manifest,
            "total_bytes": total_bytes,
            "outputs_downloaded": len(all_manifest),
            "kaggle_writes": 0,
            "compute_started": False,
        }
        (args.output_dir / "recovery_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print("SERSEM_CACHE_RECOVERY_FAILURE " + json.dumps(safe_exception(exc), sort_keys=True))
        return 1

    print(
        "SERSEM_CACHE_RECOVERY_RESULT "
        + json.dumps(
            {
                "request_id": request["request_id"],
                "shards": len(all_manifest),
                "total_bytes": total_bytes,
                "kaggle_writes": 0,
                "compute_started": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
