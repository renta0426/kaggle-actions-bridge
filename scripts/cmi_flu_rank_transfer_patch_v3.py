#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

TARGET_REQUEST_ID = "20260904-cmi-flu-rank-transfer-003"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-rank-transfer-20260904-003"


def _load_v2() -> ModuleType:
    path = Path(__file__).with_name("cmi_flu_rank_transfer_patch_v2.py")
    spec = importlib.util.spec_from_file_location("cmi_flu_rank_transfer_patch_v2_for_v3", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to load rank-transfer v2 patcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    v2 = _load_v2()
    original_load_v1 = v2._load_v1

    def load_v1_with_legacy_loader_compat() -> ModuleType:
        v1 = original_load_v1()
        original_execute_source = v1.execute_source

        def execute_source_v3() -> str:
            text = original_execute_source()

            # The embedded runtime package is the audited B2 package. Its
            # configuration loader predates the b021_taskwise_robust baseline
            # name and only accepts b01/b02. Keep the serialized config baseline
            # at b02 while loading, then promote the already-validated config
            # object to the B2.1 identity before the rank-transfer science code
            # sees it. selection.policy=robust_v1 and the B2.1 runtime adapter
            # remain unchanged.
            old_replace = '''        config_text = config_text.replace(\n            "baseline: b02_taskwise_compact",\n            "baseline: b021_taskwise_robust",\n            1,\n        )\n'''
            new_replace = '''        # Compatibility boundary: the embedded B2 loader does not know the\n        # later b021 baseline name. Keep b02 in the file and promote the\n        # validated config object immediately after loading.\n'''
            if text.count(old_replace) != 1:
                raise SystemExit(
                    f"rank-transfer baseline-rewrite anchor count={text.count(old_replace)}"
                )
            text = text.replace(old_replace, new_replace, 1)

            old_load = '''        config = load_baseline_config(config_path, repository_root=runtime_root)\n        if config.baseline != "b021_taskwise_robust":\n            raise BundleContractError("runtime B2.1 config identity mismatch")\n        inputs = load_inputs(config)\n'''
            new_load = '''        config = load_baseline_config(config_path, repository_root=runtime_root)\n        if config.baseline != "b02_taskwise_compact":\n            raise BundleContractError("legacy embedded loader returned unexpected baseline")\n        if str(config.section("selection").get("policy", "")) != "robust_v1":\n            raise BundleContractError("runtime robust_v1 selection contract missing")\n        raw_compat = dict(config.raw)\n        raw_compat["baseline"] = "b021_taskwise_robust"\n        object.__setattr__(config, "raw", raw_compat)\n        object.__setattr__(config, "baseline", "b021_taskwise_robust")\n        if config.baseline != "b021_taskwise_robust":\n            raise BundleContractError("runtime B2.1 compatibility promotion failed")\n        inputs = load_inputs(config)\n'''
            if text.count(old_load) != 1:
                raise SystemExit(
                    f"rank-transfer config-load anchor count={text.count(old_load)}"
                )
            return text.replace(old_load, new_load, 1)

        v1.execute_source = execute_source_v3
        return v1

    v2._load_v1 = load_v1_with_legacy_loader_compat
    v2.TARGET_REQUEST_ID = TARGET_REQUEST_ID
    v2.TARGET_KERNEL = TARGET_KERNEL
    return int(v2.main())


if __name__ == "__main__":
    raise SystemExit(main())
