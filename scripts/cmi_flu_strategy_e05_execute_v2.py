#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle_exact_identity import exact_metadata_eventually

REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-002"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-002"
TARGET_SLUG = TARGET.split("/", 1)[1]
TITLE = "CMI Flu E05 HAI Donor Strain 20260907 002"
EXPECTED_VERSION = 1
POLL_SECONDS = 60
MAX_POLLS = 60


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def plain(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).casefold()


def kernel_meta(api, ref):
    # Exact GetKernel remains the identity authority.  The only reconciliation
    # is bounded repetition of that same read-only endpoint for the observed
    # transient post-push 403/404 class.  No search result can satisfy identity.
    return exact_metadata_eventually(api, ref, attempts=8, delay_seconds=3.0)


def prewrite_guard(api):
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != TARGET_SLUG:
        raise RuntimeError("E05 v2 target identity contract changed")
    if TITLE != "CMI Flu E05 HAI Donor Strain 20260907 002":
        raise RuntimeError("E05 v2 title contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=TARGET_SLUG, page_size=20) or []
    except Exception as error:
        raise RuntimeError("E05 v2 duplicate sentinel unavailable") from error
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E05 v2 duplicate sentinel found exact target; write refused")


def live_rules(api):
    pages = api.competition_list_pages(COMPETITION) or []
    content = {}
    for page in pages:
        data = page.to_dict() if hasattr(page, "to_dict") else dict(page)
        name = str(data.get("name") or "").strip().lower()
        content[name] = str(data.get("content") or "")
    if "rules" not in content or "evaluation" not in content or not any("data" in key for key in content):
        raise RuntimeError("live Competition pages unavailable")
    if "mean spearman correlation" not in plain(content["evaluation"]):
        raise RuntimeError("evaluation guard changed")


def active_counts(api):
    active = {"cpu": 0, "gpu": 0, "tpu": 0, "unknown": 0}
    for item in (api.kernels_list(user="renta0426", sort_by="dateRun", page_size=10) or [])[:10]:
        ref = str(getattr(item, "ref", ""))
        if not ref:
            continue
        try:
            status = str(getattr(api.kernels_status(ref), "status", "")).upper()
        except Exception:
            active["unknown"] += 1
            continue
        if not any(token in status for token in ("RUNNING", "QUEUED", "PENDING")):
            continue
        try:
            metadata = kernel_meta(api, ref)
            kind = "tpu" if bool(getattr(metadata, "enable_tpu", False)) else ("gpu" if bool(getattr(metadata, "enable_gpu", False)) else "cpu")
        except Exception:
            kind = "unknown"
        active[kind] += 1
    return active


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _write_receipt(*, phase: str, runtime: Path, attempted: bool, return_code=None, stdout="", stderr="", identity_confirmed=False):
    payload = {
        "request_id": REQUEST_ID,
        "phase": phase,
        "target_sha256": hashlib.sha256(TARGET.encode("utf-8")).hexdigest(),
        "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "write_attempted": bool(attempted),
        "cli_return_code": return_code,
        "stdout_bytes": len(stdout.encode("utf-8", errors="replace")),
        "stdout_sha256": _digest_text(stdout),
        "stderr_bytes": len(stderr.encode("utf-8", errors="replace")),
        "stderr_sha256": _digest_text(stderr),
        "exact_identity_confirmed": bool(identity_confirmed),
    }
    print("CMI_FLU_E05_V2_WRITE_RECEIPT " + json.dumps(payload, sort_keys=True))


def push(api, runtime: Path, work: Path):
    kernel_dir = work / "kernel"
    kernel_dir.mkdir(parents=True)
    shutil.copyfile(runtime, kernel_dir / "script.py")
    metadata = {
        "id": TARGET,
        "title": TITLE,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": [COMPETITION],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if set(p.name for p in kernel_dir.iterdir()) != {"script.py", "kernel-metadata.json"}:
        raise RuntimeError("E05 v2 kernel payload file set changed")

    _write_receipt(phase="before_write", runtime=runtime, attempted=False)
    completed = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    _write_receipt(
        phase="after_cli_write", runtime=runtime, attempted=True,
        return_code=int(completed.returncode), stdout=completed.stdout, stderr=completed.stderr,
    )
    if completed.returncode:
        raise RuntimeError(f"kaggle push failed rc={completed.returncode}")

    try:
        direct = kernel_meta(api, TARGET)
    except Exception as error:
        safe = hashlib.sha256(f"{type(error).__name__}:{error}".encode("utf-8", errors="replace")).hexdigest()
        print(f"CMI_FLU_E05_V2_WRITE_AMBIGUOUS exception_type={type(error).__name__} error_sha256={safe}")
        raise RuntimeError("E05 v2 push acknowledged but direct metadata unconfirmed") from error
    if (
        str(getattr(direct, "ref", "")) != TARGET
        or not bool(getattr(direct, "is_private", False))
        or bool(getattr(direct, "enable_gpu", False))
        or bool(getattr(direct, "enable_tpu", False))
        or bool(getattr(direct, "enable_internet", False))
    ):
        raise RuntimeError("pushed E05 v2 direct metadata contract mismatch")
    if int(getattr(direct, "current_version_number", 0) or 0) != EXPECTED_VERSION:
        raise RuntimeError("E05 v2 direct version mismatch")
    _write_receipt(
        phase="exact_identity_confirmed", runtime=runtime, attempted=True,
        return_code=int(completed.returncode), stdout=completed.stdout, stderr=completed.stderr,
        identity_confirmed=True,
    )


def wait(api):
    for _ in range(MAX_POLLS):
        direct = kernel_meta(api, TARGET)
        if int(getattr(direct, "current_version_number", 0) or 0) != EXPECTED_VERSION:
            raise RuntimeError("E05 v2 version changed during execution")
        status = str(getattr(api.kernels_status(TARGET), "status", "")).upper()
        if "COMPLETE" in status:
            return status
        if any(token in status for token in ("ERROR", "CANCEL", "FAIL")):
            raise RuntimeError("E05 v2 remote failed")
        if not any(token in status for token in ("RUNNING", "QUEUED", "PENDING")):
            raise RuntimeError(f"unknown E05 v2 status:{status}")
        time.sleep(POLL_SECONDS)
    raise RuntimeError("E05 v2 polling bound exceeded")


def main():
    parsed = args()
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_"):
        raise SystemExit("KAGGLE_API_TOKEN contract failed")
    api = KaggleApi(); api.authenticate(); live_rules(api); prewrite_guard(api)
    active = active_counts(api)
    if active["unknown"] or active["cpu"] >= 1:
        raise SystemExit("CPU admission closed or resource classification unknown")

    work = parsed.output_dir.parent / "execution"
    work.mkdir(parents=True, exist_ok=False)
    try:
        push(api, parsed.runtime.resolve(), work)
        status = wait(api)
        direct = kernel_meta(api, TARGET)
        if int(getattr(direct, "current_version_number", 0) or 0) != EXPECTED_VERSION:
            raise RuntimeError("E05 v2 version changed before output read")
        reader = parsed.bridge_root.resolve() / "scripts" / "kaggle_current_output_read.py"
        completed = subprocess.run(
            [
                sys.executable, str(reader), "--kernel", TARGET, "--expected-version", "1",
                "--allow-file", "bridge-result.json:1048576",
                "--allow-file", "metrics.json:12582912",
                "--allow-file", "summary.md:1048576",
                "--output-dir", str(parsed.output_dir.resolve()),
            ],
            timeout=240, check=False,
        )
        if completed.returncode:
            raise RuntimeError("aggregate output read failed")
        direct = kernel_meta(api, TARGET)
        if int(getattr(direct, "current_version_number", 0) or 0) != EXPECTED_VERSION:
            raise RuntimeError("E05 v2 version changed after output read")
        print(f"CMI_FLU_E05_V2_REMOTE PASS status={status} version=1")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
