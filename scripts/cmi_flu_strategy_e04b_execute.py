#!/usr/bin/env python3
"""One-shot private CPU executor for CMI-Flu E04b Task1.3."""
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
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

REQUEST_ID = "20260909-cmi-flu-strategy-e04b-task13-material-bridge-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET = "renta0426/cmi-flu-e04b-task13-material-bridge-20260909-001"
TARGET_SLUG = TARGET.split("/", 1)[1]
TITLE = "CMI Flu E04b Task13 Material Bridge 20260909 001"
EXPECTED_VERSION = 1
POLL_SECONDS = 60
MAX_POLLS = 65


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--bridge-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).casefold()


def kernel_meta(api: KaggleApi, ref: str):
    owner, slug = ref.split("/", 1)
    with api.build_kaggle_client() as client:
        q = ApiGetKernelRequest()
        q.user_name = owner
        q.kernel_slug = slug
        return client.kernels.kernels_api_client.get_kernel(q).metadata


def prewrite_guard(api: KaggleApi) -> None:
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != TARGET_SLUG or TITLE != "CMI Flu E04b Task13 Material Bridge 20260909 001":
        raise RuntimeError("E04b target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=TARGET_SLUG, page_size=20) or []
    except Exception as exc:
        raise RuntimeError("E04b duplicate sentinel unavailable") from exc
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E04b duplicate sentinel found exact target; write refused")


def live_rules(api: KaggleApi) -> None:
    pages = api.competition_list_pages(COMPETITION) or []
    content: dict[str, str] = {}
    for page in pages:
        data = page.to_dict() if hasattr(page, "to_dict") else dict(page)
        content[str(data.get("name") or "").strip().lower()] = str(data.get("content") or "")
    if "evaluation" not in content or "mean spearman correlation" not in plain(content["evaluation"]):
        raise RuntimeError("E04b live evaluation guard changed")


def active_counts(api: KaggleApi) -> dict[str, int]:
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
            meta = kernel_meta(api, ref)
            kind = "tpu" if bool(getattr(meta, "enable_tpu", False)) else ("gpu" if bool(getattr(meta, "enable_gpu", False)) else "cpu")
        except Exception:
            kind = "unknown"
        active[kind] += 1
    return active


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def receipt(phase: str, runtime: Path, attempted: bool, rc: int | None = None, stdout: str = "", stderr: str = "", confirmed: bool = False) -> None:
    payload = {
        "request_id": REQUEST_ID,
        "phase": phase,
        "target_sha256": hashlib.sha256(TARGET.encode()).hexdigest(),
        "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "write_attempted": bool(attempted),
        "cli_return_code": rc,
        "stdout_bytes": len(stdout.encode(errors="replace")),
        "stdout_sha256": digest_text(stdout),
        "stderr_bytes": len(stderr.encode(errors="replace")),
        "stderr_sha256": digest_text(stderr),
        "exact_identity_confirmed": bool(confirmed),
    }
    print("CMI_FLU_E04B_WRITE_RECEIPT " + json.dumps(payload, sort_keys=True))


def push(api: KaggleApi, runtime: Path, work: Path) -> None:
    kernel = work / "kernel"
    kernel.mkdir(parents=True)
    shutil.copyfile(runtime, kernel / "script.py")
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
    (kernel / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    receipt("before_write", runtime, False)
    completed = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(kernel)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    receipt("after_cli_write", runtime, True, int(completed.returncode), completed.stdout, completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"kaggle push failed rc={completed.returncode}")
    try:
        meta = kernel_meta(api, TARGET)
    except Exception as exc:
        code = hashlib.sha256(f"{type(exc).__name__}:{exc}".encode(errors="replace")).hexdigest()[:20]
        print(f"CMI_FLU_E04B_WRITE_AMBIGUOUS exception_type={type(exc).__name__} error_code={code}")
        raise RuntimeError("E04b push acknowledged but exact metadata unconfirmed") from exc
    if (
        str(getattr(meta, "ref", "")) != TARGET
        or not bool(getattr(meta, "is_private", False))
        or bool(getattr(meta, "enable_gpu", False))
        or bool(getattr(meta, "enable_tpu", False))
        or bool(getattr(meta, "enable_internet", False))
    ):
        raise RuntimeError("pushed E04b metadata contract mismatch")
    if int(getattr(meta, "current_version_number", 0) or 0) != EXPECTED_VERSION:
        raise RuntimeError("E04b direct version mismatch")
    receipt("exact_identity_confirmed", runtime, True, int(completed.returncode), completed.stdout, completed.stderr, True)


def wait(api: KaggleApi) -> str:
    for _ in range(MAX_POLLS):
        meta = kernel_meta(api, TARGET)
        if int(getattr(meta, "current_version_number", 0) or 0) != EXPECTED_VERSION:
            raise RuntimeError("E04b version changed during execution")
        status = str(getattr(api.kernels_status(TARGET), "status", "")).upper()
        if "COMPLETE" in status:
            return status
        if any(token in status for token in ("ERROR", "CANCEL", "FAIL")):
            raise RuntimeError("E04b remote failed")
        if not any(token in status for token in ("RUNNING", "QUEUED", "PENDING")):
            raise RuntimeError(f"unknown E04b status:{status}")
        time.sleep(POLL_SECONDS)
    raise RuntimeError("E04b polling bound exceeded")


def main() -> int:
    a = args()
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_"):
        raise SystemExit("KAGGLE_API_TOKEN contract failed")
    api = KaggleApi()
    api.authenticate()
    live_rules(api)
    prewrite_guard(api)
    active = active_counts(api)
    if active["unknown"] or active["cpu"] >= 1:
        raise SystemExit("CPU admission closed or resource classification unknown")

    work = a.output_dir.parent / "execution"
    work.mkdir(parents=True, exist_ok=False)
    try:
        push(api, a.runtime.resolve(), work)
        status = wait(api)
        meta = kernel_meta(api, TARGET)
        if int(getattr(meta, "current_version_number", 0) or 0) != EXPECTED_VERSION:
            raise RuntimeError("E04b version changed before output read")
        reader = a.bridge_root.resolve() / "scripts" / "kaggle_current_output_read.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(reader),
                "--kernel",
                TARGET,
                "--expected-version",
                str(EXPECTED_VERSION),
                "--allow-file",
                "bridge-result.json:1048576",
                "--allow-file",
                "metrics.json:16777216",
                "--allow-file",
                "summary.md:1048576",
                "--output-dir",
                str(a.output_dir.resolve()),
            ],
            timeout=240,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError("E04b aggregate output read failed")
        if int(getattr(kernel_meta(api, TARGET), "current_version_number", 0) or 0) != EXPECTED_VERSION:
            raise RuntimeError("E04b version changed after output read")
        print(f"CMI_FLU_E04B_REMOTE PASS status={status} version={EXPECTED_VERSION}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
