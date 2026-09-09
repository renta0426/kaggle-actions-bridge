#!/usr/bin/env python3
"""Bridge regression checks for generated E06c runtime."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

E06C_BLOB = "90c6e3e2791cc2089d8666f4c8672a47953f5e12"
REQUEST_ID = "20260909-cmi-flu-strategy-e06c-d28-calibration-001"
TARGET = "renta0426/cmi-flu-e06c-d28-calibration-20260909-001"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    source = p.parse_args().runtime.read_text(encoding="utf-8")
    tree = ast.parse(source)
    compile(source, "generated_e06c_contract_test.py", "exec")
    funcs = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for name in ("load_e06_module", "validate_result", "render_summary", "locate_locked_reference"):
        if funcs.count(name) != 1:
            raise SystemExit(f"E06c runtime function count mismatch:{name}:{funcs.count(name)}")
    required = (
        REQUEST_ID,
        TARGET,
        E06C_BLOB,
        'strategy_e06c_blob_sha',
        'strategy_v2_e06c_d28_rank_preserving_calibration',
        'stage = "run_e06c"',
        'stage = "validate_e06c"',
        "tasks={len((result.get('tasks') or {}))}",
        '/tmp',
        'cmi-flu-e05-locked-references',
        'Task2.1',
        'Task2.2',
        'public_probe_authorized',
    )
    for token in required:
        if token not in source:
            raise SystemExit(f"E06c runtime required token missing:{token}")
    forbidden = (
        "/kaggle/working/.e05-locked-references",
        "conditions={len((result.get('conditions') or {}))}",
        "kaggle competitions submit",
        "competition_submit(",
    )
    for token in forbidden:
        if token in source:
            raise SystemExit(f"E06c runtime forbidden token present:{token}")
    print("CMI_FLU_E06C_RUNTIME_CONTRACT_PASS version=1 staging=/tmp submission=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
