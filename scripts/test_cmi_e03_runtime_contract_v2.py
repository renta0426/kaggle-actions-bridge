#!/usr/bin/env python3
"""Run the frozen E03 runtime contract with repair request 002 identity."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

BASE = Path(__file__).with_name("test_cmi_e03_runtime_contract.py")
EXPECTED_REQUEST = "20260908-cmi-flu-strategy-e03-task11-optional-view-fusion-002"
EXPECTED_TARGET = "renta0426/cmi-flu-e03-task11-optview-20260908-002"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_cmi_e03_runtime_contract_v2.py RUNTIME")
    runtime = Path(sys.argv[1]).expanduser().resolve()
    base = load(BASE, "cmi_e03_runtime_contract_base")
    base.EXPECTED_REQUEST = EXPECTED_REQUEST
    rc = int(base.main())
    generated = load(runtime, "cmi_e03_runtime_repair_identity")
    if generated.REQUEST_ID != EXPECTED_REQUEST:
        raise SystemExit("E03 repair runtime request identity mismatch")
    if generated.TARGET_KERNEL != EXPECTED_TARGET:
        raise SystemExit("E03 repair runtime target identity mismatch")
    print(
        "CMI_FLU_E03_RUNTIME_REPAIR_IDENTITY PASS "
        f"request_id={EXPECTED_REQUEST} target={EXPECTED_TARGET}"
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
