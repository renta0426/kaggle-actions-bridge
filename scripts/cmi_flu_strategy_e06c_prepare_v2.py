#!/usr/bin/env python3
"""Apply inherited E06 runtime repairs before E06c specialization."""
from __future__ import annotations

import cmi_flu_strategy_e06c_prepare as v1
from cmi_flu_e06a_completion_marker import patch_runtime as patch_completion_marker

_ORIGINAL_BUILD_PRIOR_RUNTIME = v1.build_prior_runtime
_LEGACY_SELF_TEST = '''    if "/kaggle/working/.e05-locked-references" in globals().get("__file__", ""):
        raise BridgeContractError("e06c_locked_reference_output_path")
'''


def build_prior_runtime(root, reference_dir):
    """Return the repaired E06a-v3 runtime and a self-test-safe AST helper.

    E06a-v3 predates the reusable completion-marker repair; E06b applies that
    repair before specialization, so E06c must do the same.

    E06c-v1 also placed the legacy staging path literally inside the generated
    self-test and then (correctly) rejected that literal anywhere in the final
    runtime.  Strip only that self-referential assertion while the independent
    builder, runtime-regression and workflow checks continue to reject the
    legacy path and require /tmp staging.
    """
    v3, runtime = _ORIGINAL_BUILD_PRIOR_RUNTIME(root, reference_dir)
    patched = patch_completion_marker(runtime)
    expected = "conditions={len((result.get('conditions') or {}))}"
    if patched.count(expected) != 1:
        raise SystemExit(f"E06c v2 intermediate completion marker count={patched.count(expected)}")

    original_replace = v3.replace_top_level_function

    def replace_top_level_function(text: str, name: str, replacement: str) -> str:
        if name == "self_test":
            if replacement.count(_LEGACY_SELF_TEST) != 1:
                raise SystemExit(
                    f"E06c v2 self-referential staging assertion count={replacement.count(_LEGACY_SELF_TEST)}"
                )
            replacement = replacement.replace(_LEGACY_SELF_TEST, "", 1)
        return original_replace(text, name, replacement)

    v3.replace_top_level_function = replace_top_level_function
    return v3, patched


# Patch only inherited runtime construction/AST replacement. Exact science
# loading, E06c validation, provenance, no-submit checks and final compile remain
# owned by the audited v1 E06c builder.
v1.build_prior_runtime = build_prior_runtime


def main() -> int:
    return v1.main()


if __name__ == "__main__":
    raise SystemExit(main())
