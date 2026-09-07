#!/usr/bin/env python3
"""Reuse the E01 aggregate sanitizer for repair request 002."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REQUEST_ID = "20260907-cmi-flu-strategy-e01-paired-evaluation-repair-002"
BASE = Path(__file__).with_name("cmi_flu_strategy_e01_sanitize.py")


def main() -> int:
    spec = importlib.util.spec_from_file_location("cmi_flu_e01_sanitize_base", BASE)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load E01 base sanitizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "REQUEST_ID", None) != "20260907-cmi-flu-strategy-e01-paired-evaluation-001":
        raise SystemExit("E01 base sanitizer contract changed")
    module.REQUEST_ID = REQUEST_ID
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
