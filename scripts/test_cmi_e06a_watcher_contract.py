#!/usr/bin/env python3
"""Static regression for E06a watcher horizon after E05 007 exceeded the old 60-minute bound."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "cmi_flu_strategy_e06a_execute.py"


def _literal_assignment(tree: ast.Module, name: str):
    matches = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    assert len(matches) == 1, name
    return ast.literal_eval(matches[0].value)


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert _literal_assignment(tree, "REQUEST_ID") == "20260908-cmi-flu-strategy-e06a-task23-calibration-001"
    assert _literal_assignment(tree, "TARGET") == "renta0426/cmi-flu-e06a-task23-calibration-20260908-001"
    poll_seconds = int(_literal_assignment(tree, "POLL_SECONDS"))
    max_polls = int(_literal_assignment(tree, "MAX_POLLS"))
    horizon_seconds = poll_seconds * max_polls
    assert poll_seconds == 120
    assert max_polls == 65
    assert horizon_seconds == 130 * 60
    assert horizon_seconds >= 120 * 60
    assert "base.wait = wait" in text
    assert "base.prewrite_guard = prewrite_guard" in text
    assert "no write retry is permitted" in text
    assert "current-version recovery is required" in text
    assert "polling bound exceeded" not in text
    print(
        "CMI_FLU_E06A_WATCHER_CONTRACT_PASS horizon_minutes=130 "
        "hard_runtime_minutes=120 write_retry=false expiry_is_remote_failure=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
