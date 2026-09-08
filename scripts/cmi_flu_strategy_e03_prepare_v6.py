#!/usr/bin/env python3
"""Build E03 repair request 002 with the science/runtime frozen from request 001.

The only intended operational change is the fresh, shorter Kaggle kernel identity.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

BASE = Path(__file__).with_name("cmi_flu_strategy_e03_prepare_v5.py")
REQUEST_ID = "20260908-cmi-flu-strategy-e03-task11-optional-view-fusion-002"
TARGET_KERNEL = "renta0426/cmi-flu-e03-task11-optview-20260908-002"
REQUEST_PATH = "requests/cmi-flu-strategy-e03-task11-optional-view-fusion-002.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    v5 = load_module(BASE, "cmi_flu_strategy_e03_prepare_v5_for_v6")
    original_v5_load_base = v5.load_base

    def load_base_v6():
        v2 = original_v5_load_base()
        v2.REQUEST_ID = REQUEST_ID
        v2.TARGET_KERNEL = TARGET_KERNEL
        v2.REQUEST_PATH = REQUEST_PATH

        original_v2_load_module = v2.load_module

        def load_module_v6(path: Path, name: str):
            loaded = original_v2_load_module(path, name)
            if Path(path).name == "cmi_flu_strategy_e03_prepare.py":
                loaded.REQUEST_ID = REQUEST_ID
                loaded.TARGET_KERNEL = TARGET_KERNEL
                loaded.REQUEST_PATH = REQUEST_PATH
            return loaded

        v2.load_module = load_module_v6
        return v2

    v5.load_base = load_base_v6
    return int(v5.main())


if __name__ == "__main__":
    raise SystemExit(main())
