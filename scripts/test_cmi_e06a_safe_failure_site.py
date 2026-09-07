#!/usr/bin/env python3
"""Validate privacy-safe E06a production failure-site instrumentation."""
from __future__ import annotations

import argparse
import runpy
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    args = p.parse_args()
    text = args.runtime.read_text(encoding="utf-8")
    assert text.count("CMI_FLU_E06A_SAFE_FAILURE_SITE") == 2
    assert "CMI_FLU_E06A_FAILED stage=" not in text
    ns = runpy.run_path(str(args.runtime), run_name="e06a_safe_site")
    assert callable(ns.get("_e06a_safe_failure_site"))
    probe = {}
    exec(
        compile(
            "def boom():\n    raise RuntimeError('sensitive-message-must-not-emit')\n",
            "cmi_flu/strategy_e06.py",
            "exec",
        ),
        probe,
        probe,
    )
    try:
        probe["boom"]()
    except RuntimeError as exc:
        sites, digest = ns["_e06a_safe_failure_site"](exc)
    else:
        raise AssertionError("probe did not fail")
    assert "cmi_flu/strategy_e06.py:boom:2" in sites
    assert "sensitive-message" not in sites
    assert len(digest) == 20
    print(
        "CMI_FLU_E06A_SAFE_FAILURE_SITE_PASS execute_site=true locate_site=true "
        "module_function_line=true message=false locals=false row_data=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
