#!/usr/bin/env python3
"""E08 builder v2: repair nested runtime escapes after E08 replacements."""
from __future__ import annotations

import argparse
import builtins
import importlib.util
import subprocess
import sys
from pathlib import Path

BASE = "scripts/cmi_flu_strategy_e08_prepare.py"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load(root: Path):
    path = root / BASE
    spec = importlib.util.spec_from_file_location("cmi_flu_e08_prepare_v1", path)
    if spec is None or spec.loader is None:
        raise SystemExit("E08 v1 builder unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    a = args()
    root = a.repository_root.expanduser().resolve()
    output = a.output.expanduser().resolve()
    base = load(root)
    base.validate_request(root)
    e08 = base.require_blob(root / base.E08_PATH, base.E08_BLOB, "cmi_flu/strategy_e08.py")
    e08_v2 = base.require_blob(root / base.E08_V2_PATH, base.E08_V2_BLOB, "cmi_flu/strategy_e08_v2.py")
    required = (
        'EXPERIMENT = "strategy_v2_e08_task13_applicability_audit"',
        'ONTOLOGY_VERSION = "e08_task13_ontology_audit_v1"',
        'MIN_BRIDGE_SUBJECTS = 8',
        'PAIRING_IMPLEMENTATION = "participant_unique_gate_v2"',
    )
    joined = e08 + e08_v2
    if any(token not in joined for token in required):
        raise SystemExit("E08 exact source contract token missing")
    if "kaggle competitions submit" in joined or "competition_submit(" in joined:
        raise SystemExit("E08 relayed source contains submission path")

    real_compile = builtins.compile
    def compile_guard(source, filename, mode, *positional, **keywords):
        if filename == "generated_e08_runtime.py":
            return real_compile("pass\n", filename, mode, *positional, **keywords)
        return real_compile(source, filename, mode, *positional, **keywords)
    builtins.compile = compile_guard
    try:
        runtime, package_sha = base.build_runtime(root, e08, e08_v2)
    finally:
        builtins.compile = real_compile

    # E08 inserts its own generated-source blocks after the inherited E04 v2
    # compatibility repair, so repair literal quote-newline-quote sequences a
    # second time. This is the same proven nested-runtime escape contract used
    # by E04 v2, applied at the correct final generation layer.
    runtime = runtime.replace('"\n"', '"\\n"')
    real_compile(runtime, "generated_e08_runtime.py", "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_E08_PREPARE_V2_PASS "
        f"science_commit={base.SCIENCE_COMMIT} e08_blob={base.E08_BLOB} e08_v2_blob={base.E08_V2_BLOB} "
        f"package_sha256={package_sha} runtime_sha256={base.sha256(runtime.encode())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
