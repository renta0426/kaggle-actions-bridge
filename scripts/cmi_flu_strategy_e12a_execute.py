#!/usr/bin/env python3
"""One-shot private CPU executor for Strategy-v2 E12a reproduction audit."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from kaggle_current_output_read import read_current_output
from kaggle_exact_identity import ACTIVE, exact_metadata, exact_metadata_eventually, safe_exception, status_name, validate_metadata

REQUEST_ID = "20260910-cmi-flu-strategy-e12a-final-reproduction-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET = "renta0426/cmi-flu-e12a-final-reproduction-20260910-001"
TITLE = "CMI Flu E12a Final Reproduction 20260910 001"
EXPECTED_VERSION = 1
ALLOWED = {"bridge-result.json": 65536, "metrics.json": 262144, "summary.md": 65536}


def live_rules_preflight(api: KaggleApi) -> None:
    pages = api.competition_list_pages(COMPETITION) or []
    content = {}
    for page in pages:
        data = page.to_dict() if hasattr(page, "to_dict") else dict(page)
        name = str(data.get("name") or "").strip().lower()
        if name:
            content[name] = str(data.get("content") or "")
    if "rules" not in content or "evaluation" not in content or not any("data" in k for k in content):
        raise RuntimeError("E12a live Competition contract unavailable")
    plain = lambda text: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).casefold()
    rules = plain(content["rules"])
    evaluation = plain(content["evaluation"])
    data_text = " ".join(plain(content[k]) for k in content if "data" in k)
    if not all(token in rules for token in ("data security", "external data and tools")):
        raise RuntimeError("E12a live Rules guard changed")
    if not all(token in evaluation for token in ("mean spearman correlation", "40 donors", "-99")):
        raise RuntimeError("E12a live evaluation guard changed")
    if "sample_submission_part1.csv" not in data_text or "investigations_260821.tsv" not in data_text:
        raise RuntimeError("E12a live Data guard changed")
    print("CMI_FLU_E12A_RULES_PREFLIGHT PASS write=false compute=false submission=false")


def require_fresh_target(api: KaggleApi) -> None:
    owner, slug = TARGET.split("/", 1)
    exact_absent = False
    try:
        exact_metadata(api, TARGET)
    except Exception as exc:
        status = safe_exception(exc).get("http_status")
        if status not in (403, 404):
            raise RuntimeError("E12a exact fresh-target check failed") from exc
        exact_absent = True
    else:
        raise RuntimeError("E12a target already exists; write refused")
    visible = api.kernels_list(user=owner, search=slug, page_size=100) or []
    if len(visible) >= 100:
        raise RuntimeError("E12a target discovery saturated")
    if any(str(getattr(item, "ref", "") or "") == TARGET for item in visible):
        raise RuntimeError("E12a target already exists by exact-ref discovery; write refused")
    if not exact_absent:
        raise RuntimeError("E12a target absence not proven")
    print("CMI_FLU_E12A_TARGET_ABSENT PASS write=false compute=false")


def cpu_admission(api: KaggleApi) -> None:
    recent = (api.kernels_list(user="renta0426", sort_by="dateRun", page_size=25) or [])[:25]
    cpu_active = 0
    unknown = 0
    active_total = 0
    for item in recent:
        ref = str(getattr(item, "ref", "") or "")
        if not ref:
            continue
        try:
            state = status_name(getattr(api.kernels_status(ref), "status", None))
        except Exception:
            unknown += 1
            continue
        if state not in ACTIVE:
            continue
        active_total += 1
        try:
            meta = exact_metadata(api, ref)
            gpu = getattr(meta, "enable_gpu", None)
            tpu = getattr(meta, "enable_tpu", None)
            if type(gpu) is not bool or type(tpu) is not bool:
                raise RuntimeError("unknown accelerator")
        except Exception:
            unknown += 1
            continue
        if gpu is False and tpu is False:
            cpu_active += 1
    if unknown:
        raise RuntimeError("E12a CPU admission refused: unknown active resource classification")
    if cpu_active >= 1:
        raise RuntimeError("E12a CPU admission refused: active CPU Notebook exists")
    print(f"CMI_FLU_E12A_CPU_ADMISSION PASS cpu_active={cpu_active} active_total={active_total} write=false")


def materialize(runtime: Path, root: Path) -> Path:
    raw = runtime.read_bytes()
    if len(raw) >= 900000:
        raise RuntimeError("E12a runtime exceeds source budget")
    text = raw.decode("utf-8")
    for anchor in (
        'REQUEST_ID = "20260910-cmi-flu-strategy-e12a-final-reproduction-001"',
        'TARGET_KERNEL = "renta0426/cmi-flu-e12a-final-reproduction-20260910-001"',
        'SCIENCE_COMMIT = "f227694aee240a05de1b4e318a8c17acc2db5651"',
        'E12A_BLOB = "84c9368d520b2b7f88f189d9d9a92a3306e54b0e"',
    ):
        if anchor not in text:
            raise RuntimeError("E12a runtime identity changed")
    kernel = root / "kernel"
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
        "enable_tpu": False,
        "enable_internet": False,
        "competition_sources": [COMPETITION],
    }
    (kernel / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return kernel


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise SystemExit("E12a output directory must be fresh")
    api = KaggleApi()
    api.authenticate()
    live_rules_preflight(api)
    require_fresh_target(api)
    cpu_admission(api)
    with tempfile.TemporaryDirectory(prefix="cmi-e12a-run-") as tmp:
        kernel = materialize(args.runtime.resolve(), Path(tmp))
        # Write-immediate rechecks. No retry loop surrounds the write.
        require_fresh_target(api)
        cpu_admission(api)
        try:
            response = api.kernels_push(str(kernel))
        except Exception as exc:
            info = safe_exception(exc)
            print("CMI_FLU_E12A_SAVEKERNEL_FAILURE " + json.dumps({"write_attempted":True, **info}, sort_keys=True))
            raise RuntimeError("E12a SaveKernel failed; automatic retry forbidden") from exc
        error = str(getattr(response, "error", "") or "")
        if error:
            raise RuntimeError("E12a SaveKernel response contained error; automatic retry forbidden")
        meta = exact_metadata_eventually(api, TARGET, attempts=6, delay_seconds=2.0)
        validate_metadata(meta, TARGET, EXPECTED_VERSION, cpu=True)
        print("CMI_FLU_E12A_SAVEKERNEL PASS version=1 cpu=true private=true internet=false write_calls=1 retries=0")

        terminal = None
        for _ in range(35):
            state = status_name(getattr(api.kernels_status(TARGET), "status", None))
            if state in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}:
                terminal = state
                break
            time.sleep(120)
        if terminal != "COMPLETE":
            raise RuntimeError(f"E12a kernel did not complete successfully:{terminal or 'poll_exhausted'}")

        read_current_output(kernel=TARGET, expected_version=EXPECTED_VERSION, allow=ALLOWED, output_dir=args.output_dir.resolve())
    print("CMI_FLU_E12A_EXECUTE PASS version=1 aggregate_output_read=true submission=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
