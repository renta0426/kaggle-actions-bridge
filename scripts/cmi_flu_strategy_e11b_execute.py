#!/usr/bin/env python3
"""One-shot private CPU executor for CMI-Flu E11b TabPFN-3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

import cmi_flu_strategy_e11a_execute as prior

REQUEST_ID = "20260910-cmi-flu-strategy-e11b-tabpfn3-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-001"
TARGET_SLUG = TARGET.split("/", 1)[1]
TITLE = "CMI Flu E11b TabPFN3 20260910 001"
EXPECTED_VERSION = 1
MODEL_SOURCE = "prior-labsai/tabpfn-3/pytorch/default/1"
CHECKPOINT_FILENAME = "tabpfn-v3-regressor-v3_default.ckpt"
CHECKPOINT_BYTES = 233_289_807
POLL_SECONDS = 120
MAX_POLLS = 35

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260910-cmi-flu-strategy-e11a-task11-pairwise-001",
    "TARGET": "renta0426/cmi-flu-e11a-task11-pairwise-20260910-001",
    "TITLE": "CMI Flu E11a Task11 Pairwise 20260910 001",
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(prior, key, None) != value:
        raise SystemExit(f"E11b executor ancestry changed:{key}")

base = prior.base
if base.EXPECTED_VERSION != 1:
    raise SystemExit("E11b executor version contract changed")
if "exact_metadata_eventually" not in base.kernel_meta.__code__.co_names:
    raise SystemExit("E11b executor lacks bounded exact metadata reconciliation")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--bridge-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


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
    print("CMI_FLU_E11B_WRITE_RECEIPT " + json.dumps(payload, sort_keys=True))


def model_access_preflight() -> None:
    """Use only a read-only model-version file listing before any kernel write."""
    completed = subprocess.run(
        [
            "kaggle",
            "models",
            "instances",
            "versions",
            "files",
            MODEL_SOURCE,
            "--json",
            "--page-size",
            "20",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    stdout = completed.stdout
    stderr = completed.stderr
    marker = {
        "request_id": REQUEST_ID,
        "operation": "model_instance_version_files",
        "model_source_sha256": hashlib.sha256(MODEL_SOURCE.encode()).hexdigest(),
        "return_code": int(completed.returncode),
        "stdout_bytes": len(stdout.encode("utf-8", errors="replace")),
        "stdout_sha256": _digest_text(stdout),
        "stderr_bytes": len(stderr.encode("utf-8", errors="replace")),
        "stderr_sha256": _digest_text(stderr),
        "write_attempted": False,
    }
    print("CMI_FLU_E11B_MODEL_ACCESS_RECEIPT " + json.dumps(marker, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("E11b model access preflight failed; no kernel write attempted")
    try:
        payload = json.loads(stdout)
    except Exception as exc:
        raise RuntimeError("E11b model access preflight returned non-JSON") from exc
    serialized = json.dumps(payload, sort_keys=True)
    if CHECKPOINT_FILENAME not in serialized:
        raise RuntimeError("E11b model access preflight checkpoint not visible")
    # The listing should expose size metadata; require the expected byte identity when present.
    numeric_size_tokens = {str(CHECKPOINT_BYTES), f"{CHECKPOINT_BYTES}.0"}
    if not any(token in serialized for token in numeric_size_tokens):
        raise RuntimeError("E11b model access preflight checkpoint byte identity changed")
    print(
        "CMI_FLU_E11B_MODEL_ACCESS_PASS "
        f"model_source_sha256={hashlib.sha256(MODEL_SOURCE.encode()).hexdigest()} "
        f"checkpoint={CHECKPOINT_FILENAME} bytes={CHECKPOINT_BYTES} write=false compute=false"
    )


def prewrite_guard(api: KaggleApi) -> None:
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != TARGET_SLUG or TITLE != "CMI Flu E11b TabPFN3 20260910 001":
        raise RuntimeError("E11b target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=TARGET_SLUG, page_size=20) or []
    except Exception as exc:
        raise RuntimeError("E11b duplicate sentinel unavailable") from exc
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E11b duplicate sentinel found exact target; write refused")


def active_counts(api: KaggleApi) -> dict[str, int]:
    return base.active_counts(api)


def push(api: KaggleApi, runtime: Path, work: Path) -> None:
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
        "model_sources": [MODEL_SOURCE],
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if set(path.name for path in kernel_dir.iterdir()) != {"script.py", "kernel-metadata.json"}:
        raise RuntimeError("E11b kernel payload file set changed")
    materialized = json.loads((kernel_dir / "kernel-metadata.json").read_text())
    if materialized.get("model_sources") != [MODEL_SOURCE]:
        raise RuntimeError("E11b kernel model source contract changed")

    _write_receipt(phase="before_write", runtime=runtime, attempted=False)
    completed = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    _write_receipt(
        phase="after_cli_write",
        runtime=runtime,
        attempted=True,
        return_code=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if completed.returncode:
        raise RuntimeError(f"E11b kaggle push failed rc={completed.returncode}")

    try:
        direct = base.kernel_meta(api, TARGET)
    except Exception as exc:
        safe = hashlib.sha256(f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")).hexdigest()
        print(
            f"CMI_FLU_E11B_WRITE_AMBIGUOUS exception_type={type(exc).__name__} error_sha256={safe}"
        )
        raise RuntimeError("E11b push acknowledged but direct metadata unconfirmed") from exc
    if (
        str(getattr(direct, "ref", "")) != TARGET
        or not bool(getattr(direct, "is_private", False))
        or bool(getattr(direct, "enable_gpu", False))
        or bool(getattr(direct, "enable_tpu", False))
        or bool(getattr(direct, "enable_internet", False))
    ):
        raise RuntimeError("E11b pushed direct metadata contract mismatch")
    if int(getattr(direct, "current_version_number", 0) or 0) != EXPECTED_VERSION:
        raise RuntimeError("E11b direct version mismatch")
    _write_receipt(
        phase="exact_identity_confirmed",
        runtime=runtime,
        attempted=True,
        return_code=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        identity_confirmed=True,
    )


def wait(api: KaggleApi) -> str:
    for _ in range(MAX_POLLS):
        direct = base.kernel_meta(api, TARGET)
        if int(getattr(direct, "current_version_number", 0) or 0) != EXPECTED_VERSION:
            raise RuntimeError("E11b version changed during execution")
        status = str(getattr(api.kernels_status(TARGET), "status", "")).upper()
        if "COMPLETE" in status:
            return status
        if any(token in status for token in ("ERROR", "CANCEL", "FAIL")):
            raise RuntimeError("E11b remote failed")
        if not any(token in status for token in ("RUNNING", "QUEUED", "PENDING")):
            raise RuntimeError(f"unknown E11b remote status:{status}")
        time.sleep(POLL_SECONDS)
    raise RuntimeError(
        "E11b watcher expired while remote state remained unresolved; no write retry is permitted"
    )


def main() -> int:
    parsed = parse_args()
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_"):
        raise SystemExit("KAGGLE_API_TOKEN contract failed")
    api = KaggleApi()
    api.authenticate()
    base.live_rules(api)
    # Crucially, model access is checked before duplicate/admission/write and has no side effect.
    model_access_preflight()
    prewrite_guard(api)
    active = active_counts(api)
    if active["unknown"] or active["cpu"] >= 1:
        raise SystemExit("E11b CPU admission closed or resource classification unknown")

    work = parsed.output_dir.parent / "execution"
    work.mkdir(parents=True, exist_ok=False)
    try:
        push(api, parsed.runtime.resolve(), work)
        status = wait(api)
        direct = base.kernel_meta(api, TARGET)
        if int(getattr(direct, "current_version_number", 0) or 0) != EXPECTED_VERSION:
            raise RuntimeError("E11b version changed before output read")
        reader = parsed.bridge_root.resolve() / "scripts" / "kaggle_current_output_read.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(reader),
                "--kernel",
                TARGET,
                "--expected-version",
                "1",
                "--allow-file",
                "bridge-result.json:1048576",
                "--allow-file",
                "metrics.json:12582912",
                "--allow-file",
                "summary.md:1048576",
                "--output-dir",
                str(parsed.output_dir.resolve()),
            ],
            timeout=240,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError("E11b aggregate output read failed")
        direct = base.kernel_meta(api, TARGET)
        if int(getattr(direct, "current_version_number", 0) or 0) != EXPECTED_VERSION:
            raise RuntimeError("E11b version changed after output read")
        print(f"CMI_FLU_E11B_REMOTE PASS status={status} version=1 model_source_attached=true")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
