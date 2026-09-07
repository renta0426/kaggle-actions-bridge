#!/usr/bin/env python3
"""Run one private E05 005 diagnostic kernel and recover only its safe contract-site marker."""
from __future__ import annotations

import hashlib
import html
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import cmi_flu_strategy_e05_execute_v4 as v4

REQUEST_ID = "20260907-cmi-flu-e05-contract-site-005"
TARGET = "renta0426/cmi-flu-e05-contract-site-20260907-005"
TARGET_SLUG = TARGET.split("/", 1)[1]
TITLE = "CMI Flu E05 Contract Site Diagnostic 20260907 005"
EXPECTED_ERROR_CODE = "74bf25be40ca17429601"
EXPECTED_VERSION = 1
POLL_SECONDS = 30
MAX_POLLS = 80
MARKER_RE = re.compile(
    r"CMI_FLU_E05_CONTRACT_SITE stage=[A-Za-z0-9_]+ exception_type=[A-Za-z0-9_]+ "
    r"error_code=[0-9a-f]{20} sites=[A-Za-z0-9_./:<>,=-]+ sites_sha256=[0-9a-f]{20}"
)

base = v4.prior.base
if base.EXPECTED_VERSION != 1:
    raise SystemExit("E05 executor version contract changed")
if getattr(v4, "REQUEST_ID", None) != "20260907-cmi-flu-strategy-e05-hai-donor-strain-004":
    raise SystemExit("E05 v4 executor identity changed")

base.REQUEST_ID = REQUEST_ID
base.TARGET = TARGET
base.TARGET_SLUG = TARGET_SLUG
base.TITLE = TITLE


def prewrite_guard(api):
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != TARGET_SLUG or TITLE != "CMI Flu E05 Contract Site Diagnostic 20260907 005":
        raise RuntimeError("E05 contract-site target identity changed")
    try:
        discovered = api.kernels_list(user=owner, search=TARGET_SLUG, page_size=20) or []
    except Exception as error:
        raise RuntimeError("E05 contract-site duplicate sentinel unavailable") from error
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E05 contract-site exact target already exists; write refused")


base.prewrite_guard = prewrite_guard


def verify_exact(api):
    direct = base.kernel_meta(api, TARGET)
    if str(getattr(direct, "ref", "")) != TARGET:
        raise RuntimeError("diagnostic exact ref mismatch")
    if int(getattr(direct, "current_version_number", 0) or 0) != EXPECTED_VERSION:
        raise RuntimeError("diagnostic exact version mismatch")
    if not bool(getattr(direct, "is_private", False)):
        raise RuntimeError("diagnostic privacy mismatch")
    if bool(getattr(direct, "enable_gpu", False)) or bool(getattr(direct, "enable_tpu", False)) or bool(getattr(direct, "enable_internet", False)):
        raise RuntimeError("diagnostic resource/network mismatch")
    return direct


def wait_terminal(api) -> str:
    for _ in range(MAX_POLLS):
        verify_exact(api)
        status = str(getattr(api.kernels_status(TARGET), "status", "")).upper()
        if "COMPLETE" in status:
            return status
        if any(token in status for token in ("ERROR", "CANCEL", "FAIL")):
            return status
        if not any(token in status for token in ("RUNNING", "QUEUED", "PENDING")):
            raise RuntimeError(f"unknown diagnostic status:{status}")
        time.sleep(POLL_SECONDS)
    raise RuntimeError("diagnostic polling bound exceeded")


def recover_marker(api) -> str:
    verify_exact(api)
    kaggle_cli = shutil.which("kaggle")
    if not kaggle_cli:
        raise RuntimeError("official Kaggle CLI missing")
    with tempfile.TemporaryDirectory(prefix="e05-contract-site-output-") as tmp:
        root = Path(tmp)
        completed = subprocess.run(
            [kaggle_cli, "kernels", "output", TARGET, "-p", str(root)],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            digest = hashlib.sha256((completed.stdout + completed.stderr).encode()).hexdigest()
            raise RuntimeError(f"diagnostic output read failed rc={completed.returncode} sha256={digest}")
        verify_exact(api)
        files = [p for p in root.rglob("*") if p.is_file() and not p.is_symlink()]
        total = sum(p.stat().st_size for p in files)
        if total <= 0 or total > 32 * 1024 * 1024:
            raise RuntimeError("diagnostic output byte contract violated")
        markers: list[str] = []
        for path in files:
            if path.suffix.lower() not in {".log", ".html", ".txt"}:
                continue
            if path.stat().st_size > 8 * 1024 * 1024:
                continue
            text = html.unescape(path.read_text(encoding="utf-8", errors="replace"))
            markers.extend(MARKER_RE.findall(text))
        markers = list(dict.fromkeys(markers))
        expected = [marker for marker in markers if f"error_code={EXPECTED_ERROR_CODE}" in marker]
        if len(expected) != 1:
            safe = hashlib.sha256("\n".join(markers).encode()).hexdigest()
            raise RuntimeError(f"diagnostic marker missing_or_ambiguous count={len(expected)} marker_set_sha256={safe}")
        return expected[0]


def main() -> int:
    parsed = base.args()
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_"):
        raise SystemExit("KAGGLE_API_TOKEN contract failed")
    api = base.KaggleApi()
    api.authenticate()
    base.live_rules(api)
    prewrite_guard(api)
    active = base.active_counts(api)
    if active["unknown"] or active["cpu"] >= 1:
        raise SystemExit("CPU admission closed or resource classification unknown")

    work = parsed.output_dir.parent / "diagnostic-execution"
    work.mkdir(parents=True, exist_ok=False)
    try:
        base.push(api, parsed.runtime.resolve(), work)
        verify_exact(api)
        status = wait_terminal(api)
        if "COMPLETE" in status:
            print(f"CMI_FLU_E05_CONTRACT_SITE_UNEXPECTED_COMPLETE status={status} version=1")
            return 0
        marker = recover_marker(api)
        print(marker)
        print(f"CMI_FLU_E05_CONTRACT_SITE_RECOVERY PASS status={status} version=1 aggregate_only=true submission=false")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
