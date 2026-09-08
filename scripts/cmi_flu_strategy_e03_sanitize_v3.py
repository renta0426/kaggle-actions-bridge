#!/usr/bin/env python3
"""Validate E03 request 003 outputs using the frozen aggregate-only sanitizer."""
from __future__ import annotations

import importlib.util
from pathlib import Path

BASE = Path(__file__).with_name("cmi_flu_strategy_e03_sanitize.py")
REQUEST_ID = "20260908-cmi-flu-strategy-e03-task11-optional-view-fusion-003"


def main() -> int:
    spec = importlib.util.spec_from_file_location("cmi_flu_strategy_e03_sanitize_base_v3", BASE)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to import frozen E03 sanitizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REQUEST_ID = REQUEST_ID
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
