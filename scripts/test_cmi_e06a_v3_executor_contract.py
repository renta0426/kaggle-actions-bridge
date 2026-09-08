#!/usr/bin/env python3
"""Static contract for the fresh-target E06a 003 executor."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "cmi_flu_strategy_e06a_execute_v3.py"


def _assignment(tree: ast.Module, name: str):
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
    assert _assignment(tree, "REQUEST_ID") == "20260908-cmi-flu-strategy-e06a-task23-calibration-003"
    assert _assignment(tree, "TARGET") == "renta0426/cmi-flu-e06a-task23-calibration-20260908-003"
    assert _assignment(tree, "TITLE") == "CMI Flu E06a Task23 Calibration 20260908 003"
    assert _assignment(tree, "_EXPECTED_PRIOR") == {
        "REQUEST_ID": "20260908-cmi-flu-strategy-e06a-task23-calibration-002",
        "TARGET": "renta0426/cmi-flu-e06a-task23-calibration-20260908-002",
        "TITLE": "CMI Flu E06a Task23 Calibration 20260908 002",
    }
    assert "prior.prior.POLL_SECONDS != 120 or prior.prior.MAX_POLLS != 65" in text
    assert "base.wait = prior.prior.wait" in text
    assert "base.prewrite_guard = prewrite_guard" in text
    assert "kernels\", \"push" not in text
    assert "kaggle competitions submit" not in text
    assert "competition_submit" not in text
    print(
        "CMI_FLU_E06A_V3_EXECUTOR_CONTRACT_PASS fresh_target=true "
        "watcher_minutes=130 write_retry=false submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
