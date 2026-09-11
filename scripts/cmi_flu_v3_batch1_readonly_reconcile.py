#!/usr/bin/env python3
"""Read-only reconciliation for the ambiguous CMI-Flu V3 batch-1 Kaggle write."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

from kaggle.api.kaggle_api_extended import KaggleApi

from kaggle_current_output_read import read_current_output
from kaggle_exact_identity import exact_metadata, safe_exception, status_name

REQUEST_ID = "20260912-cmi-flu-v3-batch1-readonly-reconcile-001"
WRITER_RUN = 34603833258
WRITER_JOB = 103277431613
INTENDED = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260911-001"
TITLE = "CMI Flu Strategy V3 Batch1 Audit Diagnostics 20260911 001"
TITLE_DERIVED = "renta0426/cmi-flu-strategy-v3-batch1-audit-diagnostics-20260911-001"
CANDIDATES = (INTENDED, TITLE_DERIVED)
EXPECTED_VERSION = 1
SOURCE_B = "renta0426/cmi-flu-e12c-manual-submission-20260911-002"
SOURCE_B_VERSION = 1
SOURCE_B_SHA256 = "0f9df53c3aa8c6e4ac693f6a42dbd2633b4b61d1462df798a9bd767c220be3a5"
CSV_NAMES = [
    "v3_public_singleton_Task1_1.csv", "v3_public_singleton_Task1_2.csv",
    "v3_public_singleton_Task1_3.csv", "v3_public_singleton_Task1_4.csv",
    "v3_public_singleton_Task2_1.csv", "v3_public_singleton_Task2_2.csv",
]
SAFE_NAMES = [
    "teacher_ledger.json", "measurement_contracts.json", "split_support.json",
    "auxiliary_label_coverage.json", "source_alignment_audit.json",
    "diagnostic_manifest.json", "runtime_receipt.json",
]
ALLOWED = {name: 262144 for name in CSV_NAMES}
ALLOWED.update({name: 2097152 for name in SAFE_NAMES})


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_kaggle_cli_path() -> str:
    import os
    python_bin = Path(sys.executable).parent
    existing = os.environ.get("PATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if str(python_bin) not in parts:
        os.environ["PATH"] = str(python_bin) + (os.pathsep + existing if existing else "")
    found = shutil.which("kaggle")
    if found is None or Path(found).parent != python_bin:
        raise RuntimeError("V3 reconciliation locked Kaggle CLI unavailable in executing venv")
    return found


def pure_title_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def probe(api: KaggleApi, kernel: str) -> dict:
    try:
        metadata = exact_metadata(api, kernel)
    except Exception as exc:
        info = safe_exception(exc)
        if info.get("http_status") in (403, 404):
            return {"ref": kernel, "exists": False, "http_status": info.get("http_status")}
        raise RuntimeError("V3 reconciliation exact metadata read failed") from exc
    if getattr(metadata, "ref", None) != kernel:
        raise RuntimeError("V3 reconciliation kernel ref mismatch")
    if getattr(metadata, "is_private", None) is not True:
        raise RuntimeError("V3 reconciliation private flag not proven")
    observed = getattr(metadata, "current_version_number", None)
    if isinstance(observed, bool) or observed is None:
        raise RuntimeError("V3 reconciliation current version missing")
    version = int(observed)
    if version < 1:
        raise RuntimeError("V3 reconciliation invalid current version")
    state = status_name(getattr(api.kernels_status(kernel), "status", None))
    cpu_offline = all(getattr(metadata, field, None) is False for field in ("enable_gpu", "enable_tpu", "enable_internet"))
    return {"ref": kernel, "exists": True, "version": version, "state": state, "cpu_offline": cpu_offline}


def validate_private_output(private_dir: Path, output_dir: Path) -> tuple[dict, str]:
    actual = sorted(path.name for path in private_dir.iterdir() if path.is_file())
    expected = sorted(CSV_NAMES + SAFE_NAMES)
    if actual != expected:
        raise RuntimeError(f"V3 reconciliation output allowlist mismatch:{actual}")
    manifest = json.loads((private_dir / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("diagnostic_not_final") is not True or manifest.get("all_generated_before_scoring") is not True:
        raise RuntimeError("V3 reconciliation diagnostic manifest boundary changed")
    files = manifest.get("files") or []
    if len(files) != 6 or [item.get("filename") for item in files] != CSV_NAMES:
        raise RuntimeError("V3 reconciliation diagnostic file set changed")
    summaries = []
    for item in files:
        path = private_dir / str(item.get("filename"))
        observed_bytes = path.stat().st_size
        observed_hash = sha256_path(path)
        if int(item.get("bytes", -1)) != observed_bytes or item.get("sha256") != observed_hash:
            raise RuntimeError(f"V3 reconciliation diagnostic persisted hash mismatch:{path.name}")
        if item.get("source_csv_sha256") != SOURCE_B_SHA256:
            raise RuntimeError("V3 reconciliation diagnostic source identity changed")
        summaries.append(f"{path.name}:{observed_bytes}:{observed_hash}")
    receipt = json.loads((private_dir / "runtime_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("runtime_terminal_marker") != "CMI_FLU_V3_BATCH1_RUNTIME_PASS":
        raise RuntimeError("V3 reconciliation runtime terminal marker missing")
    if receipt.get("model_fit_count") != 0 or receipt.get("competition_submission_attempted") is not False:
        raise RuntimeError("V3 reconciliation no-fit/no-submit boundary failed")
    for name in SAFE_NAMES:
        if (private_dir / name).stat().st_size > ALLOWED[name]:
            raise RuntimeError(f"V3 reconciliation aggregate output exceeded limit:{name}")
    output_dir.mkdir(parents=True)
    for name in SAFE_NAMES:
        shutil.copyfile(private_dir / name, output_dir / name)
    return manifest, ",".join(summaries)


def self_test() -> None:
    owner = INTENDED.split("/", 1)[0]
    expected_title_ref = f"{owner}/{pure_title_slug(TITLE)}"
    if expected_title_ref != TITLE_DERIVED:
        raise RuntimeError("V3 reconciliation title-derived candidate contract changed")
    if INTENDED == TITLE_DERIVED or "strategy" in INTENDED.split("/", 1)[1]:
        raise RuntimeError("V3 reconciliation candidate distinction missing")
    if len(set(CANDIDATES)) != 2 or len(CSV_NAMES) != 6 or len(SAFE_NAMES) != 7:
        raise RuntimeError("V3 reconciliation static cardinality failed")
    print("CMI_FLU_V3_BATCH1_RECONCILE_SELF_TEST PASS candidates=2 read_only=true writes=0 competition_submit=false")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if args.output_dir is None:
        raise SystemExit("--output-dir is required")
    if args.output_dir.exists():
        raise SystemExit("V3 reconciliation aggregate output directory must be fresh")
    ensure_kaggle_cli_path()
    api = KaggleApi(); api.authenticate()

    source = probe(api, SOURCE_B)
    if not source.get("exists") or source.get("version") != SOURCE_B_VERSION or source.get("state") != "COMPLETE":
        raise RuntimeError("V3 reconciliation source B identity/status changed")
    print("CMI_FLU_V3_BATCH1_RECONCILE_SOURCE_B PASS version=1 state=COMPLETE write=false")

    probes = [probe(api, ref) for ref in CANDIDATES]
    for item in probes:
        if item["exists"]:
            print(f"CMI_FLU_V3_BATCH1_RECONCILE_PROBE ref={item['ref']} exists=true version={item['version']} state={item['state']} cpu_offline={str(item['cpu_offline']).lower()} write=false")
        else:
            print(f"CMI_FLU_V3_BATCH1_RECONCILE_PROBE ref={item['ref']} exists=false http_status={item['http_status']} write=false")
    found = [item for item in probes if item["exists"]]
    if not found:
        print("CMI_FLU_V3_BATCH1_RECONCILE PASS outcome=none_found recovered=false writes=0 retries=0 competition_submit=false")
        return 0
    if len(found) != 1:
        print("CMI_FLU_V3_BATCH1_RECONCILE PASS outcome=collision recovered=false writes=0 retries=0 competition_submit=false")
        return 0
    chosen = found[0]
    if chosen["version"] != EXPECTED_VERSION:
        print(f"CMI_FLU_V3_BATCH1_RECONCILE PASS outcome=version_conflict ref={chosen['ref']} version={chosen['version']} recovered=false writes=0 retries=0 competition_submit=false")
        return 0
    if chosen["state"] != "COMPLETE":
        print(f"CMI_FLU_V3_BATCH1_RECONCILE PASS outcome=found_noncomplete ref={chosen['ref']} version=1 state={chosen['state']} recovered=false writes=0 retries=0 competition_submit=false")
        return 0
    if chosen.get("cpu_offline") is not True:
        raise RuntimeError("V3 reconciliation expected CPU/offline contract not proven")

    with tempfile.TemporaryDirectory(prefix="cmi-v3-batch1-reconcile-") as tmp:
        private_dir = Path(tmp) / "private-output"
        read_current_output(kernel=chosen["ref"], expected_version=EXPECTED_VERSION, allow=ALLOWED, output_dir=private_dir)
        _manifest, file_summary = validate_private_output(private_dir, args.output_dir)
    print(f"CMI_FLU_V3_BATCH1_RECONCILE_PRIVATE_DIAGNOSTICS PASS files={file_summary} contents_exposed=false")
    print(f"CMI_FLU_V3_BATCH1_RECONCILE PASS outcome=one_complete ref={chosen['ref']} version=1 state=COMPLETE recovered=true writes=0 retries=0 competition_submit=false model_fit_count=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
