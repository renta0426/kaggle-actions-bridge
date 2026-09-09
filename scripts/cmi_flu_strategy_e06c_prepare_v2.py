#!/usr/bin/env python3
"""Apply the proven E06a completion-marker repair before E06c specialization."""
from __future__ import annotations

import cmi_flu_strategy_e06c_prepare as v1
from cmi_flu_e06a_completion_marker import patch_runtime as patch_completion_marker

_ORIGINAL_BUILD_PRIOR_RUNTIME = v1.build_prior_runtime


def build_prior_runtime(root, reference_dir):
    """Return E06a-v3 runtime after its known completion-schema repair.

    E06a-v3 itself predates the reusable completion-marker repair. E06b applies
    that repair before specializing its runtime. E06c must do the same so the
    v1 E06c builder can deterministically replace the intermediate conditions
    marker with its final task-count marker.
    """
    v3, runtime = _ORIGINAL_BUILD_PRIOR_RUNTIME(root, reference_dir)
    patched = patch_completion_marker(runtime)
    expected = "conditions={len((result.get('conditions') or {}))}"
    if patched.count(expected) != 1:
        raise SystemExit(f"E06c v2 intermediate completion marker count={patched.count(expected)}")
    return v3, patched


# Patch only the inherited-runtime construction hook. All exact source loading,
# result validation, provenance injection, no-submit checks and final compile
# remain owned by the audited v1 E06c builder.
v1.build_prior_runtime = build_prior_runtime


def main() -> int:
    return v1.main()


if __name__ == "__main__":
    raise SystemExit(main())
