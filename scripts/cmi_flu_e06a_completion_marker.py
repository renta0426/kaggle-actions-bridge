#!/usr/bin/env python3
"""Patch inherited E05 task-count completion marker for E06a descendants."""
from __future__ import annotations

import argparse
from pathlib import Path

OLD = "tasks={len(result['tasks'])}"
NEW = "conditions={len((result.get('conditions') or {}))}"


def patch_runtime(text: str) -> str:
    if text.count(OLD) != 1:
        raise ValueError(f"E06a completion-marker anchor count={text.count(OLD)}")
    patched = text.replace(OLD, NEW, 1)
    if OLD in patched or patched.count(NEW) != 1:
        raise ValueError("E06a completion-marker patch contract failed")
    compile(patched, "generated_e06a_completion_fixed.py", "exec")
    return patched


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = args.runtime.read_text(encoding="utf-8")
    patched = patch_runtime(source)
    args.output.write_text(patched, encoding="utf-8")
    print("CMI_FLU_E06A_COMPLETION_MARKER_PASS inherited_tasks_key=false conditions_count=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
