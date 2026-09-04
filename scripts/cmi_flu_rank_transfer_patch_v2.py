#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

TARGET_REQUEST_ID = "20260904-cmi-flu-rank-transfer-002"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-rank-transfer-20260904-002"


def _load_v1() -> ModuleType:
    path = Path(__file__).with_name("cmi_flu_rank_transfer_patch.py")
    spec = importlib.util.spec_from_file_location("cmi_flu_rank_transfer_patch_v1", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to load rank-transfer v1 patcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    v1 = _load_v1()
    v1.REQUEST_ID = TARGET_REQUEST_ID
    v1.TARGET_KERNEL = TARGET_KERNEL
    return int(v1.main())


if __name__ == "__main__":
    raise SystemExit(main())
