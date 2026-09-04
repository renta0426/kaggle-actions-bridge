#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

TARGET_REQUEST_ID = "20260904-cmi-flu-phase-a-002"


def _load_v1() -> ModuleType:
    path = Path(__file__).with_name("cmi_flu_phase_a_patch.py")
    spec = importlib.util.spec_from_file_location("cmi_flu_phase_a_patch_v1", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to load Phase A v1 patcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    v1 = _load_v1()
    original_execute_source = v1.phase_a_execute_source

    def phase_a_execute_source_v2() -> str:
        text = original_execute_source()
        old = '''        import cmi_flu.evaluation as evaluation\n        import cmi_flu.runner as runner\n\n        robust_compact = runner.run_compact_task\n'''
        new = '''        import cmi_flu.evaluation as evaluation\n        import cmi_flu.models as models\n        import cmi_flu.runner as runner\n\n        # Phase A was developed against the native post-B2.1 package, where\n        # summarize_metric_frame lives in cmi_flu.models.  The audited bridge\n        # base is the older B2 package plus the B2.1 runtime adapter; that\n        # helper is absent there.  Reuse the adapter's byte-equivalent metric\n        # summarizer before importing the exact Phase A source.\n        metric_summary = adapter_namespace.get("_metric_summary")\n        if not callable(metric_summary):\n            raise BundleContractError("B2.1 adapter lacks Phase A metric-summary compatibility")\n        models.summarize_metric_frame = metric_summary\n\n        robust_compact = runner.run_compact_task\n'''
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"Phase A compatibility anchor count={count}")
        return text.replace(old, new, 1)

    v1.PHASE_A_REQUEST_ID = TARGET_REQUEST_ID
    v1.phase_a_execute_source = phase_a_execute_source_v2
    return int(v1.main())


if __name__ == "__main__":
    raise SystemExit(main())
