#!/usr/bin/env python3
"""Wait for, verify, submit, and score one immutable CMI-Flu B2 run.

This script is designed for a protected GitHub Actions job. It transfers only
allowlisted aggregate metadata plus the exact submission file into runner-local
temporary storage. It never publishes Competition Data or row-level OOF output.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import re
import time
from collections import defaultdict
from typing import Any, Mapping
from urllib.parse import urlparse

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelRequest,
    ApiListKernelSessionOutputRequest,
)

COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_OWNER = "renta0426"
TARGET_SLUG = "cmi-flu-b2-taskwise-compact-20260903-001"
TARGET_KERNEL = f"{TARGET_OWNER}/{TARGET_SLUG}"
LAUNCH_REQUEST_ID = "20260903-cmi-flu-b2-001"
FINALIZE_REQUEST_ID = "20260903-cmi-flu-b2-finalize-001"
EXPECTED_KERNEL_VERSION = 1
EXPECTED_SOURCE_COMMIT = "1596ba5b7ab2369c0b70c9da178aec1d71c80658"
EXPECTED_PACKAGE_SHA256 = "390a84bfc4393960ad1dfeab6a92a56cc8db57b1ae7c8063d8c2bf8b072b8ce0"
TASK_COLUMNS = (
    "Task1.1",
    "Task1.2",
    "Task1.3",
    "Task1.4",
    "Task2.1",
    "Task2.2",
    "Task2.3",
)
EXPECTED_HEADER = ("participant_id", *TASK_COLUMNS)
COMPLETE_MARKER = "CMI_FLU_B2_COMPLETE"
FAILED_MARKER = "CMI_FLU_B2_FAILED"
OUTPUT_LIMITS = {
    "bridge-result.json": 262_144,
    "bridge-failure.json": 32_768,
    "b2-metrics-safe.json": 524_288,
    "submission.csv": 131_072,
}
COMPLETE_OUTPUTS = {
    "bridge-result.json",
    "b2-metrics-safe.json",
    "submission.csv",
}
ALLOWED_OUTPUTS = COMPLETE_OUTPUTS | {"bridge-failure.json"}
EXPECTED_REQUEST_KEYS = {
    "schema_version",
    "request_id",
    "parent_request_id",
    "competition",
    "operation",
    "target_kernel",
    "kernel_version",
    "source_commit",
    "package_zip_sha256",
    "kernel_wait",
    "score_wait",
    "api_budget",
    "side_effects",
    "automatic_compute_retries",
    "automatic_submission_retries",
    "public_score_tuning",
    "select_final_submission",
    "rules_checked_at_utc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    wait = subparsers.add_parser("wait-verify")
    wait.add_argument("--request", type=pathlib.Path, required=True)
    wait.add_argument("--output-dir", type=pathlib.Path, required=True)
    wait.add_argument("--state", type=pathlib.Path, required=True)

    preflight = subparsers.add_parser("preflight-submit")
    preflight.add_argument("--request", type=pathlib.Path, required=True)
    preflight.add_argument("--state", type=pathlib.Path, required=True)
    preflight.add_argument("--decision", type=pathlib.Path, required=True)

    score = subparsers.add_parser("wait-score")
    score.add_argument("--request", type=pathlib.Path, required=True)
    score.add_argument("--state", type=pathlib.Path, required=True)
    score.add_argument("--decision", type=pathlib.Path, required=True)

    return parser.parse_args()


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_request(path: pathlib.Path) -> dict[str, Any]:
    request = load_json(path)
    if not isinstance(request, dict) or set(request) != EXPECTED_REQUEST_KEYS:
        raise SystemExit("B2 finalization request schema mismatch")
    expected = {
        "schema_version": 1,
        "request_id": FINALIZE_REQUEST_ID,
        "parent_request_id": LAUNCH_REQUEST_ID,
        "competition": COMPETITION,
        "operation": "wait_verify_submit_score",
        "target_kernel": TARGET_KERNEL,
        "kernel_version": EXPECTED_KERNEL_VERSION,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "package_zip_sha256": EXPECTED_PACKAGE_SHA256,
        "kernel_wait": {"interval_seconds": 60, "max_attempts": 180},
        "score_wait": {"interval_seconds": 20, "max_attempts": 120},
        "api_budget": {"max_calls": 320, "max_pages": 1},
        "side_effects": [
            "create at most one Competition submission after exact output verification"
        ],
        "automatic_compute_retries": 0,
        "automatic_submission_retries": 0,
        "public_score_tuning": False,
        "select_final_submission": False,
        "rules_checked_at_utc": "2026-09-03T02:50:00Z",
    }
    if request != expected:
        differing = sorted(
            key for key in EXPECTED_REQUEST_KEYS if request.get(key) != expected.get(key)
        )
        raise SystemExit(f"B2 finalization request differs from exact allowlist: {differing}")
    return request


def normalize_page(page: Any) -> dict[str, Any]:
    if hasattr(page, "to_dict"):
        value = page.to_dict()
    elif isinstance(page, Mapping):
        value = dict(page)
    else:
        value = {}
    return value if isinstance(value, dict) else {}


def plain_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).casefold()


def validate_live_competition(api: KaggleApi) -> None:
    pages = api.competition_list_pages(COMPETITION) or []
    content: dict[str, str] = {}
    for page in pages:
        data = normalize_page(page)
        name = str(data.get("name") or "").strip().lower()
        if name:
            content[name] = str(data.get("content") or "")
    if (
        "rules" not in content
        or "evaluation" not in content
        or not any("data" in name for name in content)
    ):
        raise SystemExit("live Competition policy pages unavailable")
    rules = plain_text(content["rules"])
    evaluation = plain_text(content["evaluation"])
    data_text = " ".join(
        plain_text(content[name]) for name in content if "data" in name
    )
    required_rules = (
        "maximum of five (5) submissions per day",
        "maximum team size is five (5)",
        "data security",
        "external data and tools",
    )
    if not all(term in rules for term in required_rules):
        raise SystemExit("live Competition rules differ from approved guardrails")
    if not all(
        term in evaluation
        for term in ("mean spearman correlation", "40 donors", "-99")
    ):
        raise SystemExit("live Competition evaluation differs from approved guardrails")
    if (
        "sample_submission_part1.csv" not in data_text
        or "investigations_260821.tsv" not in data_text
    ):
        raise SystemExit("live Competition data description differs from expected inputs")
    print(f"CMI_FLU_B2_FINALIZE_LIVE_RULES PASS pages={len(content)}")


def get_kernel_metadata(api: KaggleApi) -> Any:
    discovered = api.kernels_list(
        user=TARGET_OWNER,
        search=TARGET_SLUG,
        page_size=5,
    ) or []
    refs = [str(getattr(item, "ref", "")) for item in discovered]
    if refs.count(TARGET_KERNEL) != 1:
        raise SystemExit("exact private B2 kernel was not discoverable once")
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name = TARGET_OWNER
        request.kernel_slug = TARGET_SLUG
        kernel = client.kernels.kernels_api_client.get_kernel(request)
    metadata = kernel.metadata
    if str(metadata.ref) != TARGET_KERNEL or not bool(metadata.is_private):
        raise SystemExit("private B2 kernel metadata identity mismatch")
    if bool(getattr(metadata, "enable_gpu", False)) or bool(
        getattr(metadata, "enable_tpu", False)
    ):
        raise SystemExit("B2 kernel used an unexpected accelerator")
    if bool(getattr(metadata, "enable_internet", False)):
        raise SystemExit("B2 kernel unexpectedly enabled Internet")
    version = int(metadata.current_version_number or 0)
    if version != EXPECTED_KERNEL_VERSION:
        raise SystemExit(
            f"B2 kernel version mismatch: expected={EXPECTED_KERNEL_VERSION} actual={version}"
        )
    return metadata


def read_kernel_output(api: KaggleApi) -> Any:
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = TARGET_OWNER
        request.kernel_slug = TARGET_SLUG
        request.version_label = str(EXPECTED_KERNEL_VERSION)
        request.page_size = 1000
        return client.kernels.kernels_api_client.list_kernel_session_output(request)


def select_allowlisted_outputs(response: Any) -> dict[str, str]:
    candidates: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item in list(response.files or []):
        raw_name = str(item.file_name or "").replace("\\", "/")
        base_name = pathlib.PurePosixPath(raw_name).name
        if base_name in ALLOWED_OUTPUTS:
            candidates[base_name].append((raw_name, str(item.url or "")))
    chosen: dict[str, str] = {}
    for base_name, items in candidates.items():
        preferred = [
            item
            for item in items
            if item[0] == base_name
            or item[0] == f"cmi-flu-b2/{base_name}"
            or item[0].endswith(f"/cmi-flu-b2/{base_name}")
        ]
        pool = preferred or items
        if len(pool) != 1:
            raise SystemExit(f"ambiguous allowlisted B2 output: {base_name}")
        chosen[base_name] = pool[0][1]
    return chosen


def download_outputs(chosen: Mapping[str, str], output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    for name, url in chosen.items():
        parsed = urlparse(url)
        host = (parsed.hostname or "").casefold()
        host_ok = (
            host == "storage.googleapis.com"
            or host.endswith(".storage.googleapis.com")
            or host.endswith(".googleusercontent.com")
            or host.endswith(".kaggleusercontent.com")
        )
        if parsed.scheme != "https" or not host_ok:
            raise SystemExit(f"unexpected signed-output host for {name}")
        response = session.get(
            url,
            stream=True,
            timeout=60,
            allow_redirects=False,
        )
        if response.status_code != 200:
            raise SystemExit(
                f"signed-output download failed for {name}: status={response.status_code}"
            )
        data = response.content
        if len(data) > OUTPUT_LIMITS[name]:
            raise SystemExit(f"allowlisted B2 output exceeds byte budget: {name}")
        (output_dir / name).write_bytes(data)


def finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"{label} is not numeric") from error
    if not math.isfinite(number):
        raise SystemExit(f"{label} is not finite")
    return number


def verify_complete_output(
    output_dir: pathlib.Path,
    *,
    log: str,
) -> dict[str, Any]:
    files = {path.name: path for path in output_dir.iterdir() if path.is_file()}
    if set(files) != COMPLETE_OUTPUTS:
        raise SystemExit(f"complete B2 output set mismatch: {sorted(files)}")
    if COMPLETE_MARKER not in log or FAILED_MARKER in log:
        raise SystemExit("B2 output lacks a unique successful terminal marker")

    result = load_json(files["bridge-result.json"])
    if result.get("schema_version") != 1:
        raise SystemExit("B2 result schema mismatch")
    if result.get("request_id") != LAUNCH_REQUEST_ID:
        raise SystemExit("B2 result request identity mismatch")
    if result.get("competition") != COMPETITION:
        raise SystemExit("B2 result competition mismatch")
    if result.get("kernel_stage") != "b2_taskwise_compact":
        raise SystemExit("B2 result stage mismatch")
    if result.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise SystemExit("B2 result source commit mismatch")
    if result.get("package_zip_sha256") != EXPECTED_PACKAGE_SHA256:
        raise SystemExit("B2 result package ZIP provenance mismatch")
    if int(result.get("submission_rows", 0)) != 40:
        raise SystemExit("B2 submission row count mismatch")
    if result.get("derived_panel_sizes") != {"vaccine": 3, "challenge": 12}:
        raise SystemExit("B2 HAI panel-size contract mismatch")

    expected_tasks = set(TASK_COLUMNS)
    selected = result.get("selected_models", {})
    task_cv = result.get("task_cv", {})
    if set(selected) != expected_tasks or set(task_cv) != expected_tasks:
        raise SystemExit("B2 seven-task metric/model contract mismatch")

    proxy = finite_number(
        result.get("supervised_public_proxy_macro_spearman"),
        "B2 supervised proxy macro",
    )
    finite_tasks = int(result.get("supervised_public_proxy_finite_tasks", 0))
    if finite_tasks < 4:
        raise SystemExit("B2 supervised proxy has fewer than four finite tasks")

    checksum = result.get("checksum", {})
    verified = checksum.get("verified", []) if isinstance(checksum, dict) else []
    skipped = checksum.get("skipped", []) if isinstance(checksum, dict) else []
    if len(verified) != 28 or skipped != ["md5sum"]:
        raise SystemExit("B2 Competition MD5 summary mismatch")

    stress = result.get("hai_stress_tests", {})
    if set(stress) != {"Task2.1", "Task2.2", "Task2.3"}:
        raise SystemExit("B2 HAI stress-test task set mismatch")

    submission_bytes = files["submission.csv"].read_bytes()
    metrics_bytes = files["b2-metrics-safe.json"].read_bytes()
    submission_digest = hashlib.sha256(submission_bytes).hexdigest()
    metrics_digest = hashlib.sha256(metrics_bytes).hexdigest()
    if submission_digest != result.get("submission_sha256"):
        raise SystemExit("B2 submission SHA-256 mismatch")
    if metrics_digest != result.get("metrics_sha256"):
        raise SystemExit("B2 metrics SHA-256 mismatch")

    metrics = load_json(files["b2-metrics-safe.json"])
    if not isinstance(metrics, dict):
        raise SystemExit("B2 safe metrics are not a JSON object")
    if set(metrics.get("tasks", {})) != expected_tasks:
        raise SystemExit("B2 safe metrics task set mismatch")
    if metrics.get("selected_models") != selected:
        raise SystemExit("B2 selected-model records disagree")
    if int(metrics.get("submission_validation", {}).get("rows", 0)) != 40:
        raise SystemExit("B2 safe metrics submission validation mismatch")

    with files["submission.csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 41 or tuple(rows[0]) != EXPECTED_HEADER:
        raise SystemExit("B2 submission shape/header mismatch")
    for column_index, task in enumerate(TASK_COLUMNS, start=1):
        values = [finite_number(row[column_index], f"{task} submission value") for row in rows[1:]]
        if any(value == -99 for value in values):
            raise SystemExit(f"B2 unexpected -99 values: {task}")
        if len(set(values)) < 2:
            raise SystemExit(f"B2 constant submission task: {task}")

    message = f"CMI-Flu B2 taskwise compact sha={submission_digest[:12]}"
    state = {
        "schema_version": 1,
        "finalize_request_id": FINALIZE_REQUEST_ID,
        "launch_request_id": LAUNCH_REQUEST_ID,
        "competition": COMPETITION,
        "kernel": TARGET_KERNEL,
        "kernel_version": EXPECTED_KERNEL_VERSION,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "package_zip_sha256": EXPECTED_PACKAGE_SHA256,
        "submission_path": str(files["submission.csv"].resolve()),
        "submission_sha256": submission_digest,
        "metrics_sha256": metrics_digest,
        "submission_message": message,
        "supervised_public_proxy_macro_spearman": proxy,
        "supervised_public_proxy_finite_tasks": finite_tasks,
        "selected_models": selected,
        "task_cv": task_cv,
        "hai_stress_tests": stress,
    }

    print(
        "CMI_FLU_B2_OUTPUT_VERIFIED "
        f"version={EXPECTED_KERNEL_VERSION} rows=40 "
        f"submission_sha256={submission_digest} metrics_sha256={metrics_digest} "
        f"proxy_macro={proxy:.12g} finite_tasks={finite_tasks}"
    )
    for task in TASK_COLUMNS:
        item = task_cv[task]
        print(
            "CMI_FLU_B2_TASK_CV "
            f"task={task} model={selected[task]} spearman={item.get('spearman')} "
            f"rmse={item.get('rmse')} splits={item.get('split_count')} "
            f"rows={item.get('training_rows')} subjects={item.get('training_subjects')} "
            f"studies={item.get('training_studies')}"
        )
    for task in ("Task2.1", "Task2.2", "Task2.3"):
        for family in ("leave_one_vaccine_season_out", "leave_one_strain_out"):
            item = stress[task].get(family, {})
            panel = item.get("panel_proxy", {}) if isinstance(item, dict) else {}
            rho_node = panel.get("spearman", {}) if isinstance(panel, dict) else {}
            rho = rho_node.get("value") if isinstance(rho_node, dict) else None
            print(
                "CMI_FLU_B2_HAI_STRESS "
                f"task={task} family={family} status={item.get('status', 'complete')} "
                f"splits={item.get('split_count')} panel_spearman={rho}"
            )
    return state


def validate_failure_output(output_dir: pathlib.Path, *, log: str) -> None:
    files = {path.name: path for path in output_dir.iterdir() if path.is_file()}
    if set(files) != {"bridge-failure.json"}:
        raise SystemExit(f"failed B2 output set mismatch: {sorted(files)}")
    failure = load_json(files["bridge-failure.json"])
    expected_keys = {
        "schema_version",
        "request_id",
        "competition",
        "source_commit",
        "stage",
        "exception_type",
        "error_code",
    }
    if not isinstance(failure, dict) or set(failure) != expected_keys:
        raise SystemExit("B2 failure manifest schema mismatch")
    if failure.get("request_id") != LAUNCH_REQUEST_ID:
        raise SystemExit("B2 failure request identity mismatch")
    if failure.get("competition") != COMPETITION:
        raise SystemExit("B2 failure competition mismatch")
    if failure.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise SystemExit("B2 failure source commit mismatch")
    if FAILED_MARKER not in log:
        raise SystemExit("B2 failure output lacks failure log marker")
    print(
        "CMI_FLU_B2_FAILED "
        f"stage={failure['stage']} exception_type={failure['exception_type']} "
        f"error_code={failure['error_code']}"
    )
    raise SystemExit(2)


def wait_and_verify(
    request: Mapping[str, Any],
    output_dir: pathlib.Path,
    state_path: pathlib.Path,
) -> None:
    api = KaggleApi()
    api.authenticate()
    validate_live_competition(api)
    metadata = get_kernel_metadata(api)
    print(
        "CMI_FLU_B2_KERNEL_METADATA PASS "
        f"version={int(metadata.current_version_number)} private=true "
        "accelerator=cpu internet=false"
    )

    interval = int(request["kernel_wait"]["interval_seconds"])
    max_attempts = int(request["kernel_wait"]["max_attempts"])
    terminal_response = None
    terminal_log = ""
    for attempt in range(1, max_attempts + 1):
        response = read_kernel_output(api)
        log = str(response.log or "")
        log_bytes = log.encode("utf-8", errors="replace")
        if len(log_bytes) > 2_000_000:
            raise SystemExit("private B2 log exceeds fixed byte budget")
        complete = COMPLETE_MARKER in log
        failed = FAILED_MARKER in log
        if complete and failed:
            raise SystemExit("private B2 log contains conflicting terminal markers")
        chosen = select_allowlisted_outputs(response)
        complete_ready = complete and COMPLETE_OUTPUTS.issubset(chosen)
        failure_ready = failed and "bridge-failure.json" in chosen
        if complete_ready or failure_ready:
            terminal_response = response
            terminal_log = log
            break
        if attempt == 1 or attempt % 5 == 0:
            print(
                "CMI_FLU_B2_KERNEL_WAIT "
                f"attempt={attempt}/{max_attempts} log_bytes={len(log_bytes)} "
                f"log_sha256={hashlib.sha256(log_bytes).hexdigest()} "
                f"complete_marker={str(complete).lower()} "
                f"failed_marker={str(failed).lower()} "
                f"allowlisted={','.join(sorted(chosen)) or 'none'}"
            )
        if attempt < max_attempts:
            time.sleep(interval)
    if terminal_response is None:
        raise SystemExit(
            f"B2 kernel did not expose a terminal verified output after {max_attempts} attempts"
        )

    chosen = select_allowlisted_outputs(terminal_response)
    needed = COMPLETE_OUTPUTS if COMPLETE_MARKER in terminal_log else {"bridge-failure.json"}
    download_outputs({name: chosen[name] for name in sorted(needed)}, output_dir)
    if FAILED_MARKER in terminal_log:
        validate_failure_output(output_dir, log=terminal_log)
    state = verify_complete_output(output_dir, log=terminal_log)
    atomic_write_json(state_path, state)
    print(f"CMI_FLU_B2_STATE_READY path={state_path}")


def normalize_submission(item: Any) -> dict[str, Any]:
    data = normalize_page(item)
    if data:
        return data
    names = (
        "ref",
        "id",
        "fileName",
        "description",
        "message",
        "status",
        "date",
        "submittedAt",
        "publicScore",
        "public_score",
        "privateScore",
    )
    return {name: getattr(item, name, None) for name in names}


def submission_message(row: Mapping[str, Any]) -> str:
    return str(row.get("description") or row.get("message") or "")


def submission_reference(row: Mapping[str, Any]) -> str:
    return str(row.get("ref") or row.get("id") or row.get("fileName") or "unknown")


def preflight_submit(
    request: Mapping[str, Any],
    state: Mapping[str, Any],
    decision_path: pathlib.Path,
) -> None:
    if state.get("finalize_request_id") != FINALIZE_REQUEST_ID:
        raise SystemExit("B2 state finalization identity mismatch")
    if state.get("kernel") != TARGET_KERNEL or int(state.get("kernel_version", 0)) != 1:
        raise SystemExit("B2 state kernel identity mismatch")
    digest = str(state.get("submission_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SystemExit("B2 state submission SHA-256 is malformed")
    expected_message = f"CMI-Flu B2 taskwise compact sha={digest[:12]}"
    if state.get("submission_message") != expected_message:
        raise SystemExit("B2 state submission message is not SHA-bound")
    submission_path = pathlib.Path(str(state.get("submission_path") or ""))
    if not submission_path.is_file():
        raise SystemExit("verified B2 submission file is absent")
    if hashlib.sha256(submission_path.read_bytes()).hexdigest() != digest:
        raise SystemExit("verified B2 submission file changed before preflight")

    api = KaggleApi()
    api.authenticate()
    validate_live_competition(api)
    history = [normalize_submission(item) for item in (api.competition_submissions(COMPETITION) or [])]
    matches = [row for row in history if submission_message(row) == expected_message]
    if len(matches) > 1:
        raise SystemExit(f"multiple existing B2 submissions match the SHA-bound message: {len(matches)}")

    limits = api.competition_get_submission_limits(COMPETITION)
    allowed = int(getattr(limits, "num_allowed_now", 0) or 0)
    today = int(getattr(limits, "num_today", 0) or 0)
    total = int(getattr(limits, "num_total", 0) or 0)

    if matches:
        decision = {
            "schema_version": 1,
            "should_submit": False,
            "message": expected_message,
            "submission_sha256": digest,
            "existing_reference": submission_reference(matches[0]),
            "reason": "matching SHA-bound submission already exists",
        }
        print(
            "CMI_FLU_B2_SUBMISSION_IDEMPOTENT "
            f"reference={decision['existing_reference']} sha256={digest}"
        )
    else:
        if allowed < 1 or today >= 5:
            raise SystemExit(
                f"Competition submission allowance unavailable: allowed_now={allowed} today={today}"
            )
        decision = {
            "schema_version": 1,
            "should_submit": True,
            "message": expected_message,
            "submission_sha256": digest,
            "existing_reference": None,
            "reason": "verified output has no matching existing submission",
        }
        print(
            "CMI_FLU_B2_SUBMISSION_PREFLIGHT PASS "
            f"allowed_now={allowed} submissions_today={today} lifetime={total} "
            f"sha256={digest}"
        )
    atomic_write_json(decision_path, decision)


def wait_for_score(
    request: Mapping[str, Any],
    state: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    digest = str(state.get("submission_sha256") or "")
    message = str(decision.get("message") or "")
    if message != f"CMI-Flu B2 taskwise compact sha={digest[:12]}":
        raise SystemExit("B2 score decision message mismatch")

    api = KaggleApi()
    api.authenticate()
    validate_live_competition(api)
    interval = int(request["score_wait"]["interval_seconds"])
    max_attempts = int(request["score_wait"]["max_attempts"])

    for attempt in range(1, max_attempts + 1):
        history = [
            normalize_submission(item)
            for item in (api.competition_submissions(COMPETITION) or [])
        ]
        matches = [row for row in history if submission_message(row) == message]
        if len(matches) > 1:
            raise SystemExit(f"multiple B2 submission records match the message: {len(matches)}")
        if matches:
            row = matches[0]
            reference = submission_reference(row)
            status = str(row.get("status") or "unknown")
            submitted = str(row.get("date") or row.get("submittedAt") or "unknown")
            raw_score = row.get("publicScore")
            if raw_score in (None, ""):
                raw_score = row.get("public_score")
            if raw_score not in (None, ""):
                score = finite_number(raw_score, "B2 Public score")
                print(
                    "CMI_FLU_B2_FINALIZED "
                    f"reference={reference} status={status} submitted_at={submitted} "
                    f"publicScore={score:.12g} submission_sha256={digest} "
                    f"proxy_macro={float(state['supervised_public_proxy_macro_spearman']):.12g} "
                    "automatic_compute_retries=0 automatic_submission_retries=0 "
                    "public_score_tuning=false select_final_submission=false"
                )
                return
            lowered = status.casefold()
            if any(token in lowered for token in ("error", "failed", "invalid")):
                raise SystemExit(
                    f"B2 submission entered terminal failure: reference={reference} status={status}"
                )
            if attempt == 1 or attempt % 5 == 0:
                print(
                    "CMI_FLU_B2_SCORE_WAIT "
                    f"attempt={attempt}/{max_attempts} reference={reference} "
                    f"status={status} submitted_at={submitted}"
                )
        elif attempt == 1 or attempt % 5 == 0:
            print(
                "CMI_FLU_B2_SCORE_WAIT "
                f"attempt={attempt}/{max_attempts} reference=not-yet-listed status=unknown"
            )
        if attempt < max_attempts:
            time.sleep(interval)
    raise SystemExit(
        f"B2 Public score was not available after {max_attempts} attempts"
    )


def main() -> int:
    args = parse_args()
    request = validate_request(args.request)
    if args.command == "wait-verify":
        wait_and_verify(request, args.output_dir, args.state)
        return 0
    state = load_json(args.state)
    if not isinstance(state, dict):
        raise SystemExit("B2 state is not a JSON object")
    if args.command == "preflight-submit":
        preflight_submit(request, state, args.decision)
        return 0
    decision = load_json(args.decision)
    if not isinstance(decision, dict):
        raise SystemExit("B2 submission decision is not a JSON object")
    wait_for_score(request, state, decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
