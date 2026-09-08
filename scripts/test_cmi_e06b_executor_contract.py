#!/usr/bin/env python3
"""Static contract for the fresh-target one-shot E06b executor."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "cmi_flu_strategy_e06b_execute.py"


def assignment(tree: ast.Module, name: str):
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
    assert assignment(tree, "REQUEST_ID") == "20260908-cmi-flu-strategy-e06b-task23-two-head-001"
    assert assignment(tree, "TARGET") == "renta0426/cmi-flu-e06b-task23-two-head-20260908-001"
    assert assignment(tree, "TITLE") == "CMI Flu E06b Task23 Two Head 20260908 001"
    assert assignment(tree, "_EXPECTED_PRIOR") == {
        "REQUEST_ID": "20260908-cmi-flu-strategy-e06a-task23-calibration-003",
        "TARGET": "renta0426/cmi-flu-e06a-task23-calibration-20260908-003",
        "TITLE": "CMI Flu E06a Task23 Calibration 20260908 003",
    }
    assert "prior.prior.prior.POLL_SECONDS != 120 or prior.prior.prior.MAX_POLLS != 65" in text
    assert "base.wait = prior.prior.prior.wait" in text
    assert "base.prewrite_guard = prewrite_guard" in text
    assert "kernels\", \"push" not in text
    assert "kaggle competitions submit" not in text
    assert "competition_submit" not in text
    print(
        "CMI_FLU_E06B_EXECUTOR_CONTRACT_PASS fresh_target=true watcher_minutes=130 "
        "write_retry=false submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
