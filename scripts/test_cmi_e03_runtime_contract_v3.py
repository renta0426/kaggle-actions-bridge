#!/usr/bin/env python3
"""Run frozen E03 runtime checks with request 003 identity and MD5 staging repair."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

BASE = Path(__file__).with_name("test_cmi_e03_runtime_contract.py")
EXPECTED_REQUEST = "20260908-cmi-flu-strategy-e03-task11-optional-view-fusion-003"
EXPECTED_TARGET = "renta0426/cmi-flu-e03-task11-optview-20260908-003"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_cmi_e03_runtime_contract_v3.py RUNTIME")
    runtime = Path(sys.argv[1]).expanduser().resolve()
    text = runtime.read_text(encoding="utf-8")
    base = load(BASE, "cmi_e03_runtime_contract_base_v3")
    base.EXPECTED_REQUEST = EXPECTED_REQUEST
    rc = int(base.main())
    generated = load(runtime, "cmi_e03_runtime_md5_stage_identity")
    if generated.REQUEST_ID != EXPECTED_REQUEST:
        raise SystemExit("E03 request 003 runtime identity mismatch")
    if generated.TARGET_KERNEL != EXPECTED_TARGET:
        raise SystemExit("E03 request 003 target identity mismatch")
    required = (
        'data_dir.symlink_to(input_dir, target_is_directory=True)',
        'md5_manifest_entry_not_staged',
        'md5_manifest_has_no_data_entries',
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"E03 request 003 MD5 staging repair missing: {missing}")
    print(
        "CMI_FLU_E03_RUNTIME_MD5_STAGE_REPAIR PASS "
        f"request_id={EXPECTED_REQUEST} target={EXPECTED_TARGET} full_input_symlink=true"
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
