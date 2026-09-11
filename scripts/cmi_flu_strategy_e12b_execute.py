#!/usr/bin/env python3
"""Retarget the audited E12a executor to E12b and repair current-output CLI PATH."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

import cmi_flu_strategy_e12a_execute as base

REQUEST_ID = "20260911-cmi-flu-strategy-e12b-challenge-freeze-001"
TARGET = "renta0426/cmi-flu-e12b-challenge-freeze-20260911-001"
TITLE = "CMI Flu E12b Challenge Freeze 20260911 001"
SCIENCE_COMMIT = "a411bf85a4a79fec2e9a2b9c2bc8cbe186adee5f"
E12B_BLOB = "af5df92ca81abc34a7f7046dba5bc284c98302d4"
E12B_CONTRACT_BLOB = "7e844b2d56bd6739793888bf6306946a0028d966"
BASE_EXECUTOR_BLOB = "46bfb2c29002775c57abcef024f239276470fd9e"
OLD_REQUEST_ID = "20260910-cmi-flu-strategy-e12a-final-reproduction-001"
OLD_TARGET = "renta0426/cmi-flu-e12a-final-reproduction-20260910-001"
OLD_TITLE = "CMI Flu E12a Final Reproduction 20260910 001"


def _verify_base_contract() -> None:
    if base.REQUEST_ID != OLD_REQUEST_ID:
        raise RuntimeError("E12b base executor request contract changed")
    if base.TARGET != OLD_TARGET or base.TITLE != OLD_TITLE:
        raise RuntimeError("E12b base executor target contract changed")
    if base.EXPECTED_VERSION != 1:
        raise RuntimeError("E12b base executor version contract changed")
    if base.ALLOWED != {
        "bridge-result.json": 65536,
        "metrics.json": 262144,
        "summary.md": 65536,
    }:
        raise RuntimeError("E12b base executor output contract changed")


def ensure_kaggle_cli_path() -> str:
    """Expose the locked CLI installed beside the protected-job Python executable."""
    python_bin = Path(sys.executable).resolve().parent
    existing = os.environ.get("PATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if str(python_bin) not in parts:
        os.environ["PATH"] = str(python_bin) + (os.pathsep + existing if existing else "")
    found = shutil.which("kaggle")
    if found is None:
        raise RuntimeError("E12b locked Kaggle CLI is not on PATH after exact venv-bin repair")
    found_path = Path(found).resolve()
    if found_path.parent != python_bin:
        raise RuntimeError("E12b Kaggle CLI resolved outside protected-job Python bin directory")
    return str(found_path)


def materialize(runtime: Path, root: Path) -> Path:
    raw = runtime.read_bytes()
    if len(raw) >= 900000:
        raise RuntimeError("E12b runtime exceeds source budget")
    text = raw.decode("utf-8")
    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'TARGET_KERNEL = "{TARGET}"',
        f'SCIENCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'E12B_BLOB = "{E12B_BLOB}"',
        f'E12B_CONTRACT_BLOB = "{E12B_CONTRACT_BLOB}"',
        '"Task1.3": "strict_asc_anchor"',
        '"row_level_candidate_persisted": False',
        "def load_e12b_module(",
        "CMI_FLU_E12B_COMPLETE",
    )
    if any(token not in text for token in required):
        raise RuntimeError("E12b runtime identity/freeze contract changed")
    if OLD_TARGET in text or OLD_REQUEST_ID in text:
        raise RuntimeError("E12b runtime references consumed parent identity")
    lowered = text.casefold()
    if "kaggle competitions submit" in lowered or "competition_submit(" in lowered:
        raise RuntimeError("E12b runtime contains submission path")

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
        "competition_sources": [base.COMPETITION],
    }
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return kernel


def main() -> int:
    _verify_base_contract()
    cli = ensure_kaggle_cli_path()
    if sys.argv[1:] == ["--path-self-test"]:
        print(
            "CMI_FLU_E12B_OUTPUT_READER_PATH_SELF_TEST PASS "
            f"python_bin={Path(sys.executable).resolve().parent} cli_name={Path(cli).name} "
            "auth=false write=false compute=false"
        )
        return 0
    base.REQUEST_ID = REQUEST_ID
    base.TARGET = TARGET
    base.TITLE = TITLE
    base.materialize = materialize
    if base.TARGET == OLD_TARGET:
        raise RuntimeError("E12b refused consumed parent target")
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
