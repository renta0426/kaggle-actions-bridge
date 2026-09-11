#!/usr/bin/env python3
"""Retarget the audited E12a one-shot executor to the fresh E12a-v2 reconciliation run."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import cmi_flu_strategy_e12a_execute as base

REQUEST_ID = "20260911-cmi-flu-strategy-e12a-v2-portfolio-reconciliation-001"
TARGET = "renta0426/cmi-flu-e12a-v2-portfolio-reconcile-20260911-001"
TITLE = "CMI Flu E12a v2 Portfolio Reconcile 20260911 001"
SCIENCE_COMMIT = "e6e05e578f172ce710bc0f1da55acdf290e83dd9"
E12A_V2_BLOB = "0c4c970c8bacfed61bfbb9587e0a0bfdec7903d9"
OLD_REQUEST_ID = "20260910-cmi-flu-strategy-e12a-final-reproduction-001"
OLD_TARGET = "renta0426/cmi-flu-e12a-final-reproduction-20260910-001"
OLD_TITLE = "CMI Flu E12a Final Reproduction 20260910 001"


def _verify_base_contract() -> None:
    if base.REQUEST_ID != OLD_REQUEST_ID:
        raise RuntimeError("E12a-v2 base executor request contract changed")
    if base.TARGET != OLD_TARGET or base.TITLE != OLD_TITLE:
        raise RuntimeError("E12a-v2 base executor target contract changed")
    if base.EXPECTED_VERSION != 1:
        raise RuntimeError("E12a-v2 base executor version contract changed")
    if base.ALLOWED != {
        "bridge-result.json": 65536,
        "metrics.json": 262144,
        "summary.md": 65536,
    }:
        raise RuntimeError("E12a-v2 base executor output contract changed")


def materialize(runtime: Path, root: Path) -> Path:
    raw = runtime.read_bytes()
    if len(raw) >= 900000:
        raise RuntimeError("E12a-v2 runtime exceeds source budget")
    text = raw.decode("utf-8")
    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'TARGET_KERNEL = "{TARGET}"',
        f'SCIENCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'E12A_V2_BLOB = "{E12A_V2_BLOB}"',
        '"Task1.3": "strict_asc_anchor"',
        '"final_portfolio_member": False',
        "def terminal_success_line(",
        "CMI_FLU_E12A_V2_COMPLETE",
    )
    if any(token not in text for token in required):
        raise RuntimeError("E12a-v2 runtime identity or terminal contract changed")
    if OLD_TARGET in text or OLD_REQUEST_ID in text:
        raise RuntimeError("E12a-v2 runtime references consumed E12a-v1 identity")
    if "len(result['tasks'])" in text:
        raise RuntimeError("E12a-v2 inherited terminal schema defect detected")
    if "kaggle competitions submit" in text.casefold() or "competition_submit" in text.casefold():
        raise RuntimeError("E12a-v2 runtime contains submission path")

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
    base.REQUEST_ID = REQUEST_ID
    base.TARGET = TARGET
    base.TITLE = TITLE
    base.materialize = materialize
    if base.TARGET == OLD_TARGET:
        raise RuntimeError("E12a-v2 refused consumed parent target")
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
