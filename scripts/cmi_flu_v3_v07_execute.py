#!/usr/bin/env python3
"""Execute exactly one approved V3-07 private CPU Notebook and recover its declared outputs."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cmi_flu_strategy_e05_execute as base

CONTRACTS = {
    "task22_panel_mean": {
        "request_id": "20260912-cmi-flu-strategy-v3-v07-h1-panel-mean-001",
        "target": "renta0426/cmi-flu-v3-v07-h1-panel-mean-20260912-001",
        "title": "CMI Flu V3-07 H1 Panel Mean 20260912 001",
        "prefix": "v3_v07_h1",
    },
    "task23_retention": {
        "request_id": "20260912-cmi-flu-strategy-v3-v07-h2-retention-001",
        "target": "renta0426/cmi-flu-v3-v07-h2-retention-20260912-001",
        "title": "CMI Flu V3-07 H2 Retention 20260912 001",
        "prefix": "v3_v07_h2",
    },
}
POLL_SECONDS = 120
MAX_POLLS = 65
EXPECTED_VERSION = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=sorted(CONTRACTS), required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--approved-runtime-sha256", required=True)
    p.add_argument("--bridge-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--path-self-test", action="store_true")
    return p.parse_args()


def configure(condition: str) -> dict[str, str]:
    c = CONTRACTS[condition]
    base.REQUEST_ID = c["request_id"]
    base.TARGET = c["target"]
    base.TARGET_SLUG = c["target"].split("/", 1)[1]
    base.TITLE = c["title"]
    base.EXPECTED_VERSION = EXPECTED_VERSION
    base.POLL_SECONDS = POLL_SECONDS
    base.MAX_POLLS = MAX_POLLS
    return c


def prewrite_guard(api, c: dict[str, str]) -> None:
    owner, slug = c["target"].split("/", 1)
    if owner != "renta0426" or slug != base.TARGET_SLUG or base.TITLE != c["title"]:
        raise RuntimeError("V3-07 target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=slug, page_size=20) or []
    except Exception as exc:
        raise RuntimeError("V3-07 duplicate sentinel unavailable") from exc
    if any(str(getattr(item, "ref", "")) == c["target"] for item in discovered):
        raise RuntimeError("V3-07 duplicate sentinel found exact target; write refused")


def wait(api, target: str) -> str:
    for _ in range(MAX_POLLS):
        direct = base.kernel_meta(api, target)
        if int(getattr(direct, "current_version_number", 0) or 0) != EXPECTED_VERSION:
            raise RuntimeError("V3-07 version changed during execution")
        status = str(getattr(api.kernels_status(target), "status", "")).upper()
        if "COMPLETE" in status:
            return status
        if any(token in status for token in ("ERROR", "CANCEL", "FAIL")):
            raise RuntimeError("V3-07 remote failed")
        if not any(token in status for token in ("RUNNING", "QUEUED", "PENDING")):
            raise RuntimeError(f"unknown V3-07 remote status:{status}")
        time.sleep(POLL_SECONDS)
    raise RuntimeError("V3-07 watcher expired; no write retry permitted; current-version read-only recovery required")


def validate_runtime(path: Path, expected_sha: str, condition: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha or len(expected_sha) != 64:
        raise RuntimeError(f"V3-07 approved runtime digest mismatch:{digest}")
    source = path.read_text(encoding="utf-8")
    if f"V307_CONDITION = {condition!r}" not in source:
        raise RuntimeError("V3-07 runtime condition mismatch")
    if "competition_submit(" in source.casefold() or "kaggle competitions submit" in source.casefold():
        raise RuntimeError("V3-07 runtime contains submission path")


def output_allowlist(prefix: str) -> list[tuple[str, int]]:
    return [
        (f"{prefix}_oof_bank.csv", 32 * 1024 * 1024),
        (f"{prefix}_challenge_bank.csv", 4 * 1024 * 1024),
        (f"{prefix}_summary.json", 8 * 1024 * 1024),
        (f"{prefix}_manifest.json", 1024 * 1024),
    ]


def main() -> int:
    args = parse_args()
    c = configure(args.condition)
    runtime = args.runtime.resolve()
    validate_runtime(runtime, args.approved_runtime_sha256, args.condition)
    if args.path_self_test:
        print(f"CMI_FLU_V307_EXECUTOR_SELF_TEST PASS condition={args.condition} target={c['target']} version=1 write_limit=1 retry=0 submission=false capacity_authority=kaggle")
        return 0

    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_"):
        raise SystemExit("KAGGLE_API_TOKEN contract failed")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    # Kaggle is the capacity/quota authority under execution policy v2. Do not
    # enumerate active sessions or impose a bridge-local concurrency admission gate.
    base.live_rules(api)
    prewrite_guard(api, c)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("V3-07 recovery output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    work = output_dir.parent / f"v307-execution-{args.condition}"
    work.mkdir(parents=True, exist_ok=False)
    try:
        # Exactly one write occurs here. No caller retries this operation.
        base.push(api, runtime, args.reference_dir.resolve(), work)
        status = wait(api, c["target"])
        direct = base.kernel_meta(api, c["target"])
        if int(getattr(direct, "current_version_number", 0) or 0) != EXPECTED_VERSION:
            raise RuntimeError("V3-07 version changed before output read")
        reader = args.bridge_root.resolve() / "scripts" / "kaggle_current_output_read.py"
        command = [sys.executable, str(reader), "--kernel", c["target"], "--expected-version", "1"]
        for name, limit in output_allowlist(c["prefix"]):
            command += ["--allow-file", f"{name}:{limit}"]
        command += ["--output-dir", str(output_dir)]
        completed = subprocess.run(command, timeout=300, check=False)
        if completed.returncode:
            raise RuntimeError("V3-07 declared-output read failed")
        direct = base.kernel_meta(api, c["target"])
        if int(getattr(direct, "current_version_number", 0) or 0) != EXPECTED_VERSION:
            raise RuntimeError("V3-07 version changed after output read")
        print(f"CMI_FLU_V307_REMOTE PASS condition={args.condition} status={status} version=1 write_count=1 retry=0 submission=false")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
