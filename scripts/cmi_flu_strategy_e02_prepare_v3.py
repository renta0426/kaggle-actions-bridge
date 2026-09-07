#!/usr/bin/env python3
"""Normalize the frozen E01 execution stanza, then build E02 from exact blobs."""
from __future__ import annotations

import importlib.util
from pathlib import Path

BASE = Path(__file__).with_name("cmi_flu_strategy_e02_prepare.py")


def main() -> int:
    spec = importlib.util.spec_from_file_location("cmi_flu_strategy_e02_prepare_base_v3", BASE)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load E02 base builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "E02_BLOB", None) != "e6aaba7450a527de1e75ad6e7c0d9ef7e0031d7e":
        raise SystemExit("E02 base builder contract changed")
    payload_root = getattr(module, "PAYLOAD_ROOT")
    module.E02_PARTS = (f"{payload_root}/strategy_e02.py",)

    original_patch_runtime = module.patch_runtime
    canonical = '''        stage = "load_e01"\n        run_e01 = load_e01_module()\n        stage = "run_e01"\n        result = json_safe(dict(run_e01(config, inputs)))\n        stage = "validate_e01"\n'''

    def normalized_patch_runtime(runtime: str, e02: str, e02_v2: str) -> str:
        token = "run_e01 = load_e01_module()"
        if runtime.count(token) != 1:
            raise SystemExit("E02 frozen E01 execution token changed")
        pos = runtime.index(token)
        start = runtime.rfind("\n        stage = ", 0, pos)
        if start < 0:
            raise SystemExit("E02 frozen E01 execution start not found")
        start += 1
        end_marker = '        stage = "validate_e01"\n'
        end = runtime.find(end_marker, pos)
        if end < 0:
            raise SystemExit("E02 frozen E01 validate-stage marker changed")
        end += len(end_marker)
        runtime = runtime[:start] + canonical + runtime[end:]
        return original_patch_runtime(runtime, e02, e02_v2)

    module.patch_runtime = normalized_patch_runtime
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
