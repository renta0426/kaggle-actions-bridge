#!/usr/bin/env python3
"""Validate privacy-safe E05 production failure-site instrumentation."""
from __future__ import annotations

import argparse
import runpy
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    a = p.parse_args()
    text = a.runtime.read_text(encoding="utf-8")
    assert text.count("CMI_FLU_E05_SAFE_FAILURE_SITE") == 1
    assert "CMI_FLU_E05_CONTRACT_SITE" not in text
    ns = runpy.run_path(str(a.runtime), run_name="e05_v5_safe_site")
    assert callable(ns.get("_e05_safe_failure_site"))
    probe = {}
    exec(compile("def boom():\n    raise RuntimeError('sensitive-message-must-not-emit')\n", "cmi_flu/strategy_e05.py", "exec"), probe, probe)
    try:
        probe["boom"]()
    except RuntimeError as exc:
        sites, digest = ns["_e05_safe_failure_site"](exc)
    else:
        raise AssertionError("probe did not fail")
    assert "cmi_flu/strategy_e05.py:boom:2" in sites
    assert "sensitive-message" not in sites
    assert len(digest) == 20
    print(
        "CMI_FLU_E05_V5_SAFE_FAILURE_SITE_PASS "
        "module_function_line=true message=false locals=false row_data=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
