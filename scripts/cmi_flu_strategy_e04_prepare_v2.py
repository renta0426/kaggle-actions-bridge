#!/usr/bin/env python3
"""E04 builder v2: preserve escaped newlines across the nested runtime template."""
from __future__ import annotations

import argparse
import builtins
import importlib.util
import subprocess
import sys
from pathlib import Path

BASE = "scripts/cmi_flu_strategy_e04_prepare.py"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load(root: Path):
    path = root / BASE
    spec = importlib.util.spec_from_file_location("cmi_flu_e04_prepare_v1", path)
    if spec is None or spec.loader is None:
        raise SystemExit("E04 v1 builder unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    a = args()
    root = a.repository_root.expanduser().resolve()
    output = a.output.expanduser().resolve()
    base = load(root)
    base.validate_request(root)
    e04, contract, synthetic, task13 = base.load_exact_e04(root)
    frozen = base.load_base(root)

    # v1's nested replacement source is correct Python text except that a
    # literal "\\n" in two generated helper functions is consumed once too
    # early.  Obtain that exact runtime while bypassing only its final compile,
    # restore escaped newlines, then perform the real compile here.
    original_compile = builtins.compile
    def compile_guard(source, filename, mode, *positional, **keywords):
        if filename == "generated_e04_runtime.py":
            return original_compile("pass\n", filename, mode, *positional, **keywords)
        return original_compile(source, filename, mode, *positional, **keywords)
    builtins.compile = compile_guard
    try:
        runtime, package_sha = base.build_runtime(frozen, root, e04, contract, synthetic, task13)
    finally:
        builtins.compile = original_compile
    runtime = runtime.replace('"\n"', '"\\n"')
    original_compile(runtime, "generated_e04_runtime.py", "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_E04_PREPARE_V2_PASS "
        f"science_commit={base.SCIENCE_COMMIT} e04_blob={base.E04_BLOB} contract_blob={base.CONTRACT_BLOB} "
        f"task13_blob={base.TASK13_BLOB} package_sha256={package_sha} runtime_sha256={base.sha256(runtime.encode())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
