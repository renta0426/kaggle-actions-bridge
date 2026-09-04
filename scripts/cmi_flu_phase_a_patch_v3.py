#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

TARGET_REQUEST_ID = "20260904-cmi-flu-phase-a-003"


def _load_v2() -> ModuleType:
    path = Path(__file__).with_name("cmi_flu_phase_a_patch_v2.py")
    spec = importlib.util.spec_from_file_location("cmi_flu_phase_a_patch_v2_for_v3", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to load Phase A v2 patcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    v2 = _load_v2()
    original_load_v1 = v2._load_v1

    def load_v1_with_json_safe() -> ModuleType:
        v1 = original_load_v1()
        original_execute_source = v1.phase_a_execute_source

        def execute_source_with_json_safe() -> str:
            text = original_execute_source()
            helper = r'''
def json_safe(value: Any) -> Any:
    """Match the native Phase A bundle's aggregate-output serializer."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        import numpy as np
        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

'''
            marker = "def execute(input_dir: Path, output_dir: Path) -> Mapping[str, Any]:\n"
            if text.count(marker) != 1:
                raise SystemExit(f"Phase A execute marker count={text.count(marker)}")
            return text.replace(marker, helper + marker, 1)

        v1.phase_a_execute_source = execute_source_with_json_safe
        return v1

    v2._load_v1 = load_v1_with_json_safe
    v2.TARGET_REQUEST_ID = TARGET_REQUEST_ID
    return int(v2.main())


if __name__ == "__main__":
    raise SystemExit(main())
