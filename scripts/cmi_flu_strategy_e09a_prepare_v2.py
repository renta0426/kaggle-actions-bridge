#!/usr/bin/env python3
"""E09a builder v2: keep production corrected-RNA schema strict, relax synthetic fixture only."""
from __future__ import annotations

import cmi_flu_strategy_e09a_prepare as v1

_ORIGINAL_PATCH = v1.patch_runtime


def patch_runtime(*args, **kwargs):
    runtime = _ORIGINAL_PATCH(*args, **kwargs)
    old = '''    if corrected.get("value_column") != "batch_corrected_expression" or int(corrected.get("numeric_parse_failures", -1)) != 0:
        raise BridgeContractError("e09a_corrected_rna_value_contract")'''
    new = '''    expected_corrected_column = "tpm" if synthetic else "batch_corrected_expression"
    if corrected.get("value_column") != expected_corrected_column or int(corrected.get("numeric_parse_failures", -1)) != 0:
        raise BridgeContractError("e09a_corrected_rna_value_contract")'''
    if runtime.count(old) != 1:
        raise SystemExit(f"E09a v2 corrected-RNA validator anchor count={runtime.count(old)}")
    runtime = runtime.replace(old, new, 1)
    compile(runtime, "generated_e09a_runtime_v2.py", "exec")
    return runtime


v1.patch_runtime = patch_runtime


def main() -> int:
    return v1.main()


if __name__ == "__main__":
    raise SystemExit(main())
