#!/usr/bin/env python3
"""Launch one repaired V3-07 -002 kernel and recover only declared outputs.

This executor deliberately does not use GetKernel. The -001 incident established
that Kaggle canonicalizes the title into the kernel slug and may return 403 for
a wrong ref. -002 therefore uses a pre-canonicalized expected ref, resolves the
fresh write by exact frozen title, and requires the resolved ref to equal it.
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
import time

CONTRACTS = {
    "task22_panel_mean": {
        "request_id": "20260913-cmi-flu-strategy-v3-v07-h1-panel-mean-002",
        "target": "renta0426/cmi-flu-v3-07-h1-panel-mean-20260913-002",
        "title": "CMI Flu V3 07 H1 Panel Mean 20260913 002",
        "prefix": "v3_v07_h1",
    },
    "task23_retention": {
        "request_id": "20260913-cmi-flu-strategy-v3-v07-h2-retention-002",
        "target": "renta0426/cmi-flu-v3-07-h2-retention-20260913-002",
        "title": "CMI Flu V3 07 H2 Retention 20260913 002",
        "prefix": "v3_v07_h2",
    },
}
COMPETITION = "cmi-flu-first-prediction-challenge"
EXPECTED_VERSION = 1
POLL_SECONDS = 60
MAX_POLLS = 125
REF_SHA = {
    "strain_sequences.csv": "63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887",
    "vaccine_strains_per_season.txt": "8f6c7116f37d29df0bb21d6049d82fa28b4e42b2d10ed9394a1ae6f926bd9f35",
}
FAIL_RE = re.compile(
    r"CMI_FLU_V307_RUNTIME_FAIL\s+stage=([A-Za-z0-9_]+)\s+type=([A-Za-z0-9_]+)\s+code=([0-9a-f]{20})"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=sorted(CONTRACTS), required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--approved-runtime-sha256", required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--path-self-test", action="store_true")
    return p.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def validate_runtime(path: Path, expected_sha: str, condition: str) -> None:
    if len(expected_sha) != 64 or _sha256(path) != expected_sha:
        raise RuntimeError("V3-07 repair approved runtime digest mismatch")
    source = path.read_text(encoding="utf-8")
    c = CONTRACTS[condition]
    required = (
        f"V307_CONDITION = {condition!r}",
        f"V307_REQUEST_ID = {c['request_id']!r}",
        f"V307_TARGET_KERNEL = {c['target']!r}",
        "V307_REPAIR_VERSION = \"strategy_v3_v07_v2_execution_contract_20260913\"",
    )
    if any(token not in source for token in required):
        raise RuntimeError("V3-07 repair runtime identity mismatch")
    low = source.casefold()
    forbidden = ("competition_" + "submit(", "kaggle competitions " + "submit", "competitions " + "submit")
    if any(token in low for token in forbidden):
        raise RuntimeError("V3-07 repair runtime contains submission path")


def output_names(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}_oof_bank.csv",
        f"{prefix}_challenge_bank.csv",
        f"{prefix}_summary.json",
        f"{prefix}_manifest.json",
    )


def _list_recent(api):
    return api.kernels_list(user="renta0426", sort_by="dateRun", page_size=100) or []


def prewrite_guard(api, c: dict[str, str]) -> None:
    refs = [str(getattr(x, "ref", "")) for x in _list_recent(api)]
    titles = [str(getattr(x, "title", "")) for x in _list_recent(api)]
    if c["target"] in refs or c["title"] in titles:
        raise RuntimeError("V3-07 repair target/title already exists; write refused")


def resolve_written_ref(api, c: dict[str, str]) -> str:
    for attempt in range(8):
        matches = []
        for item in _list_recent(api):
            if str(getattr(item, "title", "")) == c["title"]:
                matches.append(str(getattr(item, "ref", "")))
        matches = sorted(set(x for x in matches if x))
        if len(matches) == 1:
            if matches[0] != c["target"]:
                raise RuntimeError("V3-07 repair Kaggle canonical ref differs from approved target")
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError("V3-07 repair title resolution is ambiguous")
        if attempt < 7:
            time.sleep(5)
    raise RuntimeError("V3-07 repair write acknowledged but exact title was not resolvable")


def prepare_kernel(runtime: Path, references: Path, work: Path, c: dict[str, str]) -> Path:
    kernel_dir = work / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(runtime, kernel_dir / "script.py")
    for name, digest in REF_SHA.items():
        src = references / name
        if not src.is_file() or _sha256(src) != digest:
            raise RuntimeError(f"V3-07 repair locked reference mismatch:{name}")
        shutil.copyfile(src, kernel_dir / name)
    metadata = {
        "id": c["target"],
        "title": c["title"],
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": [COMPETITION],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return kernel_dir


def push_once(kernel_dir: Path, runtime: Path, c: dict[str, str]) -> None:
    before = {
        "request_id": c["request_id"], "phase": "before_write", "write_attempted": False,
        "runtime_sha256": _sha256(runtime), "target_sha256": hashlib.sha256(c["target"].encode()).hexdigest(),
    }
    print("CMI_FLU_V307_REPAIR_WRITE_RECEIPT " + json.dumps(before, sort_keys=True))
    completed = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
        capture_output=True, text=True, timeout=180, check=False,
    )
    after = {
        "request_id": c["request_id"], "phase": "after_cli_write", "write_attempted": True,
        "cli_return_code": int(completed.returncode), "runtime_sha256": _sha256(runtime),
        "target_sha256": hashlib.sha256(c["target"].encode()).hexdigest(),
        "stdout_bytes": len(completed.stdout.encode("utf-8", errors="replace")),
        "stdout_sha256": _digest_text(completed.stdout),
        "stderr_bytes": len(completed.stderr.encode("utf-8", errors="replace")),
        "stderr_sha256": _digest_text(completed.stderr),
    }
    print("CMI_FLU_V307_REPAIR_WRITE_RECEIPT " + json.dumps(after, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError(f"V3-07 repair Kaggle push failed rc={completed.returncode}")


def wait_terminal(api, ref: str) -> str:
    for i in range(MAX_POLLS):
        status = str(getattr(api.kernels_status(ref), "status", "")).upper()
        if "COMPLETE" in status:
            return "COMPLETE"
        if any(token in status for token in ("ERROR", "CANCEL", "FAIL")):
            return "ERROR"
        if not any(token in status for token in ("RUNNING", "QUEUED", "PENDING")):
            raise RuntimeError(f"V3-07 repair unknown remote status:{status}")
        if i + 1 < MAX_POLLS:
            time.sleep(POLL_SECONDS)
    raise RuntimeError("V3-07 repair watcher expired; no write retry permitted")


def safe_failure_marker(ref: str, condition: str) -> dict[str, object]:
    completed = subprocess.run(
        ["kaggle", "kernels", "logs", ref], capture_output=True, text=True, timeout=180, check=False
    )
    combined = completed.stdout + "\n" + completed.stderr
    matches = FAIL_RE.findall(combined)
    failure = None
    if matches:
        stage, kind, code = matches[-1]
        failure = {"stage": stage, "type": kind, "code": code}
    return {
        "condition": condition,
        "log_cli_return_code": int(completed.returncode),
        "raw_log_bytes": len(combined.encode("utf-8", errors="replace")),
        "raw_log_sha256": _digest_text(combined),
        "failure": failure,
    }


def recover_outputs(ref: str, c: dict[str, str], output_dir: Path) -> None:
    if output_dir.exists():
        raise RuntimeError("V3-07 repair output directory already exists")
    output_dir.mkdir(parents=True)
    names = output_names(c["prefix"])
    pattern = "^(?:" + "|".join(re.escape(name) for name in names) + ")$"
    completed = subprocess.run(
        ["kaggle", "kernels", "output", ref, "-p", str(output_dir), "-q", "-o", "--page-size", "100", "--file-pattern", pattern],
        capture_output=True, text=True, timeout=300, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"V3-07 repair output read failed rc={completed.returncode}")
    files = [p for p in output_dir.rglob("*") if p.is_file()]
    found = sorted(p.name for p in files)
    if found != sorted(names):
        raise RuntimeError(f"V3-07 repair declared output file set mismatch:{found}")
    for path in files:
        if path.stat().st_size <= 0 or path.stat().st_size > 32 * 1024 * 1024:
            raise RuntimeError(f"V3-07 repair output size invalid:{path.name}")
        if path.parent != output_dir:
            shutil.copyfile(path, output_dir / path.name)
    for path in list(output_dir.rglob("*")):
        if path.is_file() and path.parent != output_dir:
            path.unlink()
    for path in sorted(output_dir.rglob("*"), reverse=True):
        if path.is_dir():
            try: path.rmdir()
            except OSError: pass


def main() -> int:
    args = parse_args(); c = CONTRACTS[args.condition]
    runtime = args.runtime.expanduser().resolve()
    validate_runtime(runtime, args.approved_runtime_sha256, args.condition)
    if args.path_self_test:
        print(f"CMI_FLU_V307_REPAIR_EXECUTOR_SELF_TEST PASS condition={args.condition} target={c['target']} version=1 write_limit=1 retry=0 submission=false")
        return 0
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.startswith("KGAT_"):
        raise SystemExit("KAGGLE_API_TOKEN contract failed")
    from kaggle.api.kaggle_api_extended import KaggleApi
    import cmi_flu_strategy_e05_execute as base
    api = KaggleApi(); api.authenticate()
    base.live_rules(api)
    prewrite_guard(api, c)
    output_dir = args.output_dir.expanduser().resolve()
    work = output_dir.parent / f"v307-repair-execution-{args.condition}"
    work.mkdir(parents=True, exist_ok=False)
    try:
        kernel_dir = prepare_kernel(runtime, args.reference_dir.expanduser().resolve(), work, c)
        push_once(kernel_dir, runtime, c)
        ref = resolve_written_ref(api, c)
        print(f"CMI_FLU_V307_REPAIR_WRITE_CONFIRMED condition={args.condition} ref={ref} version=1")
        status = wait_terminal(api, ref)
        if status != "COMPLETE":
            diagnostic = safe_failure_marker(ref, args.condition)
            print("CMI_FLU_V307_REPAIR_REMOTE_FAIL " + json.dumps(diagnostic, sort_keys=True))
            return 2
        recover_outputs(ref, c, output_dir)
        print(f"CMI_FLU_V307_REPAIR_REMOTE_PASS condition={args.condition} ref={ref} version=1 write_count=1 retry=0 submission=false")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
