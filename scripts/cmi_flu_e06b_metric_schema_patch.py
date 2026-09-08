#!/usr/bin/env python3
"""Patch E06b bridge readers to the canonical MetricValue ``value`` field."""
from __future__ import annotations

import argparse
from pathlib import Path

DOUBLE_OLD = '(challenge.get("rank_spearman") or {}).get("spearman")'
DOUBLE_NEW = '(challenge.get("rank_spearman") or {}).get("value")'
SINGLE_OLD = "(challenge.get('rank_spearman') or {}).get('spearman')"
SINGLE_NEW = "(challenge.get('rank_spearman') or {}).get('value')"


def patch_runtime(text: str) -> str:
    if text.count(DOUBLE_OLD) != 1:
        raise ValueError(f"E06b validator MetricValue anchor count={text.count(DOUBLE_OLD)}")
    if text.count(SINGLE_OLD) != 1:
        raise ValueError(f"E06b summary MetricValue anchor count={text.count(SINGLE_OLD)}")
    patched = text.replace(DOUBLE_OLD, DOUBLE_NEW, 1).replace(SINGLE_OLD, SINGLE_NEW, 1)
    if DOUBLE_OLD in patched or SINGLE_OLD in patched:
        raise ValueError("E06b stale rank metric key remained")
    if patched.count(DOUBLE_NEW) != 1 or patched.count(SINGLE_NEW) != 1:
        raise ValueError("E06b canonical rank metric key count changed")
    compile(patched, "generated_e06b_metric_schema_fixed.py", "exec")
    return patched


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    args = p.parse_args()
    path = args.runtime.expanduser().resolve()
    patched = patch_runtime(path.read_text(encoding="utf-8"))
    path.write_text(patched, encoding="utf-8")
    print("CMI_FLU_E06B_METRIC_SCHEMA_PASS rank_spearman_field=value stale_field=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
