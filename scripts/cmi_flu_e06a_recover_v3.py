#!/usr/bin/env python3
"""Read-only recovery of E06a 003 after post-output completion-marker failure.

No Kaggle write, compute, submission, or version mutation is allowed. The
current version must be exactly 1 and may be in ERROR because the failure
occurred after scientific validation while serializing/completing outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from kaggle_exact_identity import exact_metadata, validate_metadata, verify_current, safe_exception

REQUEST = "20260908-cmi-flu-e06a-003-readonly-recovery-001"
ORIGINAL_REQUEST = "20260908-cmi-flu-strategy-e06a-task23-calibration-003"
TARGET = "renta0426/cmi-flu-e06a-task23-calibration-20260908-003"
VERSION = 1
RUNTIME_SHA256 = "c43767288c4994a743700831de5d1a24283b2f6a93f57844a555c19db96709b8"
SCIENCE_COMMIT = "c968d00eb209e55e53b88643d99382ad6915c87e"
LIMITS = {"metrics.json": 12_582_912, "summary.md": 1_048_576, "bridge-result.json": 1_048_576}
SAFE_RE = re.compile(
    r"CMI_FLU_E06A_SAFE_FAILURE_SITE stage=([a-z0-9_]{1,64}) "
    r"exception_type=([A-Za-z][A-Za-z0-9_]{0,63}) error_code=([0-9a-f]{20})"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kaggle_cli() -> str:
    found = shutil.which("kaggle")
    if found:
        return found
    adjacent = Path(sys.executable).with_name("kaggle")
    if adjacent.is_file():
        return str(adjacent)
    raise RuntimeError("kaggle_cli_missing")


def cli_read(verb: str, folder: Path) -> None:
    if verb not in {"pull", "output"}:
        raise ValueError("read_verb_not_allowed")
    completed = subprocess.run(
        [kaggle_cli(), "kernels", verb, TARGET, "-p", str(folder)],
        capture_output=True,
        check=False,
        timeout=180,
        env=os.environ.copy(),
    )
    if completed.returncode:
        digest = hashlib.sha256(completed.stdout + completed.stderr).hexdigest()
        raise RuntimeError(f"read_cli_failed:{verb}:{completed.returncode}:{digest}")


def recover(output_dir: Path, sanitizer: Path) -> None:
    if output_dir.exists():
        raise FileExistsError("recovery_output_exists")
    from kaggle.api.kaggle_api_extended import KaggleApi
    from cmi_flu_strategy_e05_execute import live_rules

    api = KaggleApi()
    api.authenticate()
    live_rules(api)
    state = verify_current(api, TARGET, VERSION, allow_failed=True, cpu=True)
    receipt = {
        "request_id": REQUEST,
        "original_request_id": ORIGINAL_REQUEST,
        "target": TARGET,
        "version": VERSION,
        "remote_status": state,
        "new_write_attempted": False,
        "new_compute_requested": False,
        "competition_submission_attempted": False,
        "science_commit": SCIENCE_COMMIT,
        "runtime_sha256": RUNTIME_SHA256,
    }

    with tempfile.TemporaryDirectory(prefix="cmi-e06a-v3-recovery-") as tmp:
        root = Path(tmp)
        code = root / "code"
        code.mkdir()
        cli_read("pull", code)
        code_files = [p for p in code.rglob("*") if p.is_file()]
        if len(code_files) != 1 or any(p.is_symlink() for p in code.rglob("*")):
            raise RuntimeError("current_code_file_contract")
        if sha256(code_files[0]) != RUNTIME_SHA256:
            raise RuntimeError("current_code_hash_mismatch")
        validate_metadata(exact_metadata(api, TARGET), TARGET, VERSION, cpu=True)

        downloaded = root / "outputs"
        downloaded.mkdir()
        cli_read("output", downloaded)
        validate_metadata(exact_metadata(api, TARGET), TARGET, VERSION, cpu=True)
        paths = list(downloaded.rglob("*"))
        if any(p.is_symlink() for p in paths):
            raise RuntimeError("output_symlink")
        files = [p for p in paths if p.is_file()]
        if not files or sum(p.stat().st_size for p in files) > 33_554_432:
            raise RuntimeError("recovery_download_size")

        logs = [p for p in files if p.suffix == ".log"]
        if len(logs) > 1:
            raise RuntimeError("transport_log_count")
        selected: dict[str, Path] = {}
        for p in files:
            if p in logs:
                continue
            if p.name not in LIMITS or p.name in selected or not 0 < p.stat().st_size <= LIMITS[p.name]:
                raise RuntimeError("unexpected_or_oversize_output")
            selected[p.name] = p
        if set(selected) != set(LIMITS):
            raise RuntimeError("declared_output_set_incomplete")

        safe_dir = root / "safe"
        safe_dir.mkdir()
        for name, source in selected.items():
            shutil.copyfile(source, safe_dir / name)

        completed = subprocess.run(
            [sys.executable, str(sanitizer), "--input-dir", str(safe_dir)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if completed.returncode:
            digest = hashlib.sha256((completed.stdout + completed.stderr).encode()).hexdigest()
            raise RuntimeError(f"sanitizer_failed:{digest}")

        signatures = []
        if logs:
            text = logs[0].read_text(encoding="utf-8", errors="replace")
            for stage, kind, code_value in SAFE_RE.findall(text):
                signatures.append({"stage": stage, "exception_type": kind, "error_code": code_value})
        receipt.update(
            {
                "output_names": sorted(selected),
                "scientific_outputs_validated": True,
                "failure_signatures": signatures[:2],
                "metrics_sha256": sha256(safe_dir / "metrics.json"),
                "summary_sha256": sha256(safe_dir / "summary.md"),
            }
        )
        output_dir.mkdir(parents=True)
        (output_dir / "recovery.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("E06A_V3_RECOVERY_VERIFIED " + json.dumps(receipt, sort_keys=True))
        print("E06A_V3_AGGREGATES_BEGIN")
        print(completed.stdout.strip())
        print("E06A_V3_AGGREGATES_END")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sanitizer", type=Path, required=True)
    args = p.parse_args()
    try:
        recover(args.output_dir.resolve(), args.sanitizer.resolve())
    except Exception as exc:
        print("E06A_V3_RECOVERY_FAILED " + json.dumps({"request_id": REQUEST, "new_write_attempted": False, **safe_exception(exc)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
