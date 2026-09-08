#!/usr/bin/env python3
"""Run E06b sanitizer with the canonical MetricValue ``value`` rank field."""
from __future__ import annotations

from pathlib import Path

SOURCE = Path(__file__).with_name("cmi_flu_strategy_e06b_sanitize.py")
OLD = '(challenge.get("rank_spearman") or {}).get("spearman")'
NEW = '(challenge.get("rank_spearman") or {}).get("value")'


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    if source.count(OLD) != 1:
        raise SystemExit(f"E06b sanitizer MetricValue anchor count={source.count(OLD)}")
    patched = source.replace(OLD, NEW, 1)
    if OLD in patched or patched.count(NEW) != 1:
        raise SystemExit("E06b sanitizer MetricValue patch contract failed")
    namespace = {"__name__": "__main__", "__file__": str(SOURCE)}
    exec(compile(patched, str(SOURCE), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
