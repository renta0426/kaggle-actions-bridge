#!/usr/bin/env python3
"""E04b builder v2: adapt to final E04 v2 anchors and separate synthetic promotion validation."""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

BASE = "scripts/cmi_flu_strategy_e04b_prepare.py"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load_base(root: Path):
    path = root / BASE
    spec = importlib.util.spec_from_file_location("cmi_flu_e04b_prepare_v1", path)
    if spec is None or spec.loader is None:
        raise SystemExit("E04b v1 builder unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    a = args()
    root = a.repository_root.expanduser().resolve()
    output = a.output.expanduser().resolve()
    base = load_base(root)

    # E04 v2's final generated runtime no longer contains the intermediate
    # `metric_value` function that E04 v1 used as a patch boundary.  Preserve
    # the E04b v1 source for provenance, but redirect that one structural end
    # anchor to the actual final E04 v2 boundary: render_summary.
    original_replace_block = base.replace_block

    def compatible_replace_block(text: str, start: str, end: str, replacement: str) -> str:
        if start == "def validate_result(result: dict, *, synthetic: bool) -> None:\n" and end == "def metric_value(":
            end = "def render_summary(result: dict) -> str:\n"
        return original_replace_block(text, start, end, replacement)

    base.replace_block = compatible_replace_block
    base.validate_request(root)
    e04b = base.require_blob(root / base.E04B_PATH, base.E04B_BLOB, "cmi_flu/strategy_e04b.py")
    e04b_synth = base.require_blob(root / base.E04B_SYNTH_PATH, base.E04B_SYNTH_BLOB, "cmi_flu/strategy_e04b_synthetic.py")
    runtime, runtime_sha = base.build_runtime(root, e04b, e04b_synth)

    old = '''    if decision.get("public_probe_authorized") is not False or decision.get("competition_candidate") is not False or decision.get("decision") != "no_promotion":
        raise BridgeContractError("e04b_promotion_boundary_mismatch")
'''
    new = '''    if decision.get("public_probe_authorized") is not False:
        raise BridgeContractError("e04b_public_probe_boundary_mismatch")
    if not synthetic and (decision.get("competition_candidate") is not False or decision.get("decision") != "no_promotion"):
        raise BridgeContractError("e04b_production_promotion_boundary_mismatch")
'''
    if runtime.count(old) != 1:
        raise SystemExit("E04b v2 promotion validation anchor changed")
    runtime = runtime.replace(old, new, 1)
    compile(runtime, "generated_e04b_runtime_v2.py", "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_E04B_PREPARE_V2_PASS "
        f"runtime_sha256_before_v2={runtime_sha} final_structure=validate_to_render_summary "
        "synthetic_promotion_independent=true production_no_promotion_locked=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
