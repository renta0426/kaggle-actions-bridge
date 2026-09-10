#!/usr/bin/env python3
"""Read-only Kaggle pre/postflight for the single P1-03 post-5k repair launch.

This file never writes to Kaggle. The only write remains the frozen launcher's one
`kaggle kernels push`, after this preflight succeeds.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REQUEST_ID = "20260910-poisoned-chalice-p1-03-post-5k-development-audit-v1-001"
TARGET = "renta0426/poisoned-chalice-p1-03-post-5k-development-audit-v1"
SCIENCE_COMMIT = "feff26fa21184744efc5f4e91a64d22fa1de4b4e"
STAGE1_DATASET = "renta0426/stage1-raw-fim-submission-v1-output"
HIDDEN_CACHE = "renta0426/poisoned-chalice-lumia-hidden-state-cache-5000-v1"
HIDDEN_EVAL = "renta0426/poisoned-chalice-lumia-hidden-state-eval-5000-v1"


def load_request(path: Path) -> dict:
    request = json.loads(path.read_text(encoding="utf-8"))
    if request.get("request_id") != REQUEST_ID:
        raise RuntimeError("request id changed")
    if request.get("target") != TARGET or request.get("operation") != "save_kernel_once":
        raise RuntimeError("target/operation changed")
    if request.get("research_commit") != SCIENCE_COMMIT:
        raise RuntimeError("science commit changed")
    if request.get("automatic_compute_retries") != 0 or request.get("competition_submission") is not False:
        raise RuntimeError("retry/submission contract changed")
    resource = request.get("resource") or {}
    if resource.get("accelerator") != "cpu" or int(resource.get("max_active_runs", -1)) != 1:
        raise RuntimeError("CPU/concurrency contract changed")
    dataset = (request.get("inputs") or {}).get("dataset") or {}
    if dataset != {"ref": STAGE1_DATASET, "version": 1}:
        raise RuntimeError("Stage1 dataset contract changed")
    kernels = (request.get("inputs") or {}).get("kernels") or []
    expected_kernels = [
        {"ref": HIDDEN_CACHE, "version": 1},
        {"ref": HIDDEN_EVAL, "version": 1},
    ]
    if kernels != expected_kernels:
        raise RuntimeError("kernel input contract changed")
    clean = request.get("clean_room") or {}
    required_false = (
        "promotion_allowed",
        "fresh_holdout_consumed",
        "new_target_model_forward",
        "validation_rows_used",
        "hidden_stage1_validation_labels_used",
        "public_leaderboard_feedback_used",
    )
    if clean.get("development_only") is not True or any(clean.get(key) is not False for key in required_false):
        raise RuntimeError("clean-room contract changed")
    return request


def live_preflight(request: dict) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kaggle_exact_identity import exact_metadata, safe_exception, verify_current

    api = KaggleApi()
    api.authenticate()

    pages = api.competition_list_pages(request["competition"]) or []
    content: dict[str, str] = {}
    for page in pages:
        data = page.to_dict() if hasattr(page, "to_dict") else dict(page)
        name = str(data.get("name") or "").strip().lower()
        if name:
            content[name] = str(data.get("content") or "")
    if "rules" not in content or "evaluation" not in content or not any("data" in key for key in content):
        raise SystemExit("live competition contract pages unavailable")
    plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", content["evaluation"].replace(r"\%", "%"))).casefold()
    if not all(token in plain for token in ("auc", "novelty", "1%", "false-positive")):
        raise SystemExit("live evaluation contract changed")

    for source in request["inputs"]["kernels"]:
        state = verify_current(api, source["ref"], int(source["version"]), allow_failed=False, cpu=False)
        if state != "COMPLETE":
            raise SystemExit(f"kernel input not complete: {source['ref']} state={state}")

    # kaggle==2.2.4 exposes dataset_status(..., format='json(current_version_number)').
    # It internally calls ApiGetDatasetRequest/get_dataset for this projection.
    dataset_payload = json.loads(
        api.dataset_status(
            request["inputs"]["dataset"]["ref"],
            format="json(current_version_number)",
        )
    )
    observed = dataset_payload.get("current_version_number")
    expected = int(request["inputs"]["dataset"]["version"])
    if observed is None or int(observed) != expected:
        raise SystemExit(f"dataset current version mismatch: observed={observed!r} expected={expected}")

    try:
        exact_metadata(api, request["target"])
    except Exception as exc:
        info = safe_exception(exc)
        if info.get("http_status") != 404:
            raise SystemExit(f"target exact-absence check failed: class={info.get('class')} http={info.get('http_status')}")
    else:
        raise SystemExit("approved target already exists; refusing overwrite")

    recent = (api.kernels_list(user="renta0426", sort_by="dateRun", page_size=25) or [])[:25]
    active = 0
    unknown = 0
    for item in recent:
        ref = str(getattr(item, "ref", "") or "")
        if not ref:
            continue
        try:
            status = str(getattr(api.kernels_status(ref), "status", "")).upper()
        except Exception:
            unknown += 1
            continue
        if any(token in status for token in ("RUNNING", "QUEUED", "PENDING")):
            active += 1
    if unknown:
        raise SystemExit(f"conservative admission refused: unknown_statuses={unknown}")
    if active >= int(request["resource"]["max_active_runs"]):
        raise SystemExit(f"concurrency admission refused: active={active} max_active={request['resource']['max_active_runs']}")

    print(
        "P1_03_POST_5K_REPAIR_PREFLIGHT PASS "
        f"cpu=1 active={active} kernel_inputs=2 dataset_version={observed} target_absent=1 write_calls=0"
    )


def postwrite(request: dict) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kaggle_exact_identity import exact_metadata_eventually

    api = KaggleApi()
    api.authenticate()
    meta = exact_metadata_eventually(api, request["target"], attempts=6, delay_seconds=2.0)
    if getattr(meta, "ref", None) != request["target"]:
        raise SystemExit("post-write target ref mismatch")
    if getattr(meta, "is_private", None) is not True:
        raise SystemExit("post-write target is not private")
    if int(getattr(meta, "current_version_number", 0)) != 1:
        raise SystemExit("post-write version is not exactly 1")
    if getattr(meta, "enable_gpu", None) is not False or getattr(meta, "enable_tpu", None) is not False:
        raise SystemExit("post-write accelerator contract mismatch")
    if getattr(meta, "enable_internet", None) is not True:
        raise SystemExit("post-write internet contract mismatch")
    print("P1_03_POST_5K_REPAIR_POSTWRITE PASS target=exact version=1 cpu=1 private=1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--postwrite", action="store_true")
    args = parser.parse_args()
    request = load_request(args.request)
    if args.static:
        print("P1_03_POST_5K_REPAIR_STATIC PASS write_calls=0 retries=0 submissions=0")
        return 0
    if args.preflight:
        live_preflight(request)
        return 0
    postwrite(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
