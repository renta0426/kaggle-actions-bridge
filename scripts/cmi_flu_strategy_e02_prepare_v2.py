#!/usr/bin/env python3
"""Use the exact single-blob E02 science relay with the audited E02 builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

BASE = Path(__file__).with_name("cmi_flu_strategy_e02_prepare.py")


def main() -> int:
    spec = importlib.util.spec_from_file_location("cmi_flu_strategy_e02_prepare_base", BASE)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load E02 base builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "E02_BLOB", None) != "e6aaba7450a527de1e75ad6e7c0d9ef7e0031d7e":
        raise SystemExit("E02 base builder contract changed")
    payload_root = getattr(module, "PAYLOAD_ROOT")
    module.E02_PARTS = (f"{payload_root}/strategy_e02.py",)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
