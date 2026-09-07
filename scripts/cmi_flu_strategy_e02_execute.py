#!/usr/bin/env python3
"""Execute exactly one approved private CPU E02 Kaggle run and read aggregate outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

REQUEST_ID = "20260907-cmi-flu-strategy-e02-task11-structured-logfc-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET = "renta0426/cmi-flu-e02-task11-structured-logfc-20260907-001"
TARGET_SLUG = "cmi-flu-e02-task11-structured-logfc-20260907-001"
TARGET_TITLE = "CMI Flu E02 Task11 Structured LogFC 20260907 001"
EXPECTED_VERSION = 1
POLL_SECONDS = 300
MAX_POLLS = 12


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
        request = ApiGetKernelRequest()
        request.user_name = owner
        request.kernel_slug = slug
        return client.kernels.kernels_api_client.get_kernel(request).metadata


def live_rules(api: KaggleApi) -> None:
    pages = api.competition_list_pages(COMPETITION) or []
    content = {}
    for page in pages:
        data = page.to_dict() if hasattr(page, "to_dict") else dict(page)
        name = str(data.get("name") or "").strip().lower()
        if name:
            content[name] = str(data.get("content") or "")
    if "rules" not in content or "evaluation" not in content or not any("data" in key for key in content):
        raise RuntimeError("live Competition pages unavailable")
    rules = plain(content["rules"])
    evaluation = plain(content["evaluation"])
    data_text = " ".join(plain(content[key]) for key in content if "data" in key)
    if not all(token in rules for token in ("data security", "external data and tools")):
        raise RuntimeError("live Competition rules differ from approved guardrails")
    if not all(token in evaluation for token in ("mean spearman correlation", "40 donors", "-99")):
        raise RuntimeError("live Competition evaluation differs from approved guardrails")
    if "sample_submission_part1.csv" not in data_text or "investigations_260821.tsv" not in data_text:
        raise RuntimeError("live Competition data description differs from approved guardrails")


def active_resource_counts(api: KaggleApi) -> dict[str, int]:
    active = {"cpu": 0, "gpu": 0, "tpu": 0, "unknown": 0}
    recent = (api.kernels_list(user="renta0426", sort_by="dateRun", page_size=10) or [])[:10]
    for item in recent:
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


def refuse_duplicate(api: KaggleApi) -> None:
    existing = api.kernels_list(user="renta0426", search=TARGET_SLUG, page_size=10) or []
    refs = {str(getattr(item, "ref", "")) for item in existing}
    if TARGET in refs:
        raise RuntimeError("approved E02 target already exists; duplicate write refused")


def push_one(api: KaggleApi, runtime: Path, work: Path) -> None:
    if not runtime.is_file():
        raise RuntimeError("generated runtime missing")
    kernel_dir = work / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(runtime, kernel_dir / "script.py")
    metadata = {
        "id": TARGET,
        "title": TARGET_TITLE,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": [COMPETITION],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    completed = subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], capture_output=True, text=True, timeout=180, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"kaggle kernels push failed rc={completed.returncode}")
    discovered = api.kernels_list(user="renta0426", search=TARGET_SLUG, page_size=10) or []
    refs = [str(getattr(item, "ref", "")) for item in discovered]
    if refs.count(TARGET) != 1:
        raise RuntimeError("pushed E02 kernel exact discoverability mismatch")
    meta = kernel_meta(api, TARGET)
    if str(getattr(meta, "ref", "")) != TARGET or not bool(getattr(meta, "is_private", False)):
        raise RuntimeError("pushed E02 kernel identity/privacy mismatch")
    if bool(getattr(meta, "enable_gpu", False)) or bool(getattr(meta, "enable_tpu", False)) or bool(getattr(meta, "enable_internet", False)):
        raise RuntimeError("pushed E02 kernel resource/network contract mismatch")
    if int(getattr(meta, "current_version_number", 0) or 0) != EXPECTED_VERSION:
        raise RuntimeError("pushed E02 kernel version mismatch")


def wait_terminal(api: KaggleApi) -> str:
    for _ in range(MAX_POLLS):
        status = str(getattr(api.kernels_status(TARGET), "status", "")).upper()
        if "COMPLETE" in status:
            return status
        if any(token in status for token in ("ERROR", "CANCEL", "FAIL")):
            raise RuntimeError("E02 remote kernel terminated unsuccessfully")
        if not any(token in status for token in ("RUNNING", "QUEUED", "PENDING")):
            raise RuntimeError("E02 remote kernel returned unknown nonterminal status")
        time.sleep(POLL_SECONDS)
    raise RuntimeError("E02 remote kernel exceeded bounded polling contract")


def main() -> int:
    ns = args()
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_"):
        raise SystemExit("KAGGLE_API_TOKEN contract failed")
    api = KaggleApi()
    api.authenticate()
    live_rules(api)
    refuse_duplicate(api)
    active = active_resource_counts(api)
    if active["unknown"] or active["cpu"] >= 1:
        raise SystemExit("CPU admission closed or active resource classification unknown")
    print(f"CMI_FLU_E02_PREFLIGHT PASS request_id={REQUEST_ID} cpu_active={active['cpu']} gpu_active={active['gpu']} tpu_active={active['tpu']}")

    work = ns.output_dir.parent / "execution"
    if work.exists():
        raise SystemExit("execution work directory already exists")
    work.mkdir(parents=True)
    try:
        push_one(api, ns.runtime.resolve(), work)
        print(f"CMI_FLU_E02_PUSH PASS request_id={REQUEST_ID} target_hash_only={__import__('hashlib').sha256(TARGET.encode()).hexdigest()[:20]} version={EXPECTED_VERSION}")
        status = wait_terminal(api)
        print(f"CMI_FLU_E02_REMOTE PASS status={status} version={EXPECTED_VERSION}")
        reader = ns.bridge_root.resolve() / "scripts" / "kaggle_current_output_read.py"
        completed = subprocess.run([
            sys.executable, str(reader),
            "--kernel", TARGET,
            "--expected-version", str(EXPECTED_VERSION),
            "--allow-file", "bridge-result.json:1048576",
            "--allow-file", "metrics.json:8388608",
            "--allow-file", "summary.md:1048576",
            "--output-dir", str(ns.output_dir.resolve()),
        ], check=False, timeout=240)
        if completed.returncode != 0:
            raise RuntimeError("aggregate current-output read failed")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
