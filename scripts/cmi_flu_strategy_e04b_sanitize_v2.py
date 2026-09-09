#!/usr/bin/env python3
"""Repair E04b sanitizer: keep raw ontology row count separate from donor prediction rows."""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

BASE = "scripts/cmi_flu_strategy_e04b_sanitize.py"

OLD = '''    if ontology.get("changed_rows_by_split") != {"challenge": 40, "historical": 0}:
        raise SystemExit(f"E04b unexpected ontology row changes:{ontology.get('changed_rows_by_split')}")
    if ontology.get("changed_rows") != 40 or ontology.get("changed_rows_by_study") != {"2025LJI": 40}:
        raise SystemExit("E04b ontology real-data change count mismatch")
'''

NEW = '''    changed_rows = ontology.get("changed_rows")
    by_split = ontology.get("changed_rows_by_split") or {}
    by_study = ontology.get("changed_rows_by_study") or {}
    if not isinstance(changed_rows, int) or changed_rows < 40:
        raise SystemExit(f"E04b raw ontology change count invalid:{changed_rows!r}")
    if set(by_split) != {"challenge", "historical"} or by_split.get("historical") != 0 or by_split.get("challenge") != changed_rows:
        raise SystemExit(f"E04b ontology split accounting mismatch:{by_split}")
    if by_study != {"2025LJI": changed_rows}:
        raise SystemExit(f"E04b ontology study accounting mismatch:{by_study}")
'''


def patched_source(root: Path) -> str:
    source = (root / BASE).read_text(encoding="utf-8")
    if source.count(OLD) != 1:
        raise SystemExit("E04b sanitizer v1 row-count anchor changed")
    source = source.replace(OLD, NEW, 1)
    compile(source, "cmi_flu_strategy_e04b_sanitize_v2_runtime.py", "exec")
    return source


def self_test(root: Path) -> int:
    source = patched_source(root)
    if 'transfer.get("challenge_correction_rows") != [40, 40]' not in source:
        raise SystemExit("E04b sanitizer v2 lost the 40-row donor correction contract")
    if 'transfer.get("supervised_strict_source_subjects") != [23, 23]' not in source:
        raise SystemExit("E04b sanitizer v2 lost the 23-source contract")
    if 'decision.get("competition_candidate") is not False' not in source:
        raise SystemExit("E04b sanitizer v2 lost the no-promotion contract")
    print("CMI_FLU_E04B_SANITIZER_V2_SELF_TEST PASS raw_rows_separate_from_prediction_rows=true")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    if args.self_test:
        return self_test(root)
    if args.input_dir is None:
        raise SystemExit("--input-dir required")
    source = patched_source(root)
    module = types.ModuleType("cmi_flu_strategy_e04b_sanitize_v2_runtime")
    module.__file__ = str(root / BASE)
    exec(compile(source, module.__file__, "exec"), module.__dict__, module.__dict__)
    old_argv = sys.argv
    try:
        sys.argv = [module.__file__, "--input-dir", str(args.input_dir)]
        return int(module.main())
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
