#!/usr/bin/env python3
"""Regression for the E06a post-output completion marker inherited from E05."""
from __future__ import annotations

from cmi_flu_e06a_completion_marker import OLD, NEW, patch_runtime


def main() -> int:
    source = "print(f\"tasks={len(result['tasks'])}\")"
    fixed = patch_runtime(source)
    assert OLD not in fixed
    assert NEW in fixed
    assert "result['tasks']" not in fixed
    assert "result.get('conditions')" in fixed
    print("CMI_FLU_E06A_COMPLETION_MARKER_REGRESSION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
