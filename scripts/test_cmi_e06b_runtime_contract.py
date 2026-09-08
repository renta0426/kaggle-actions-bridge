#!/usr/bin/env python3
"""Static production-runtime guard for E06b identity, leakage and completion schema."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

REQUEST_ID = "20260908-cmi-flu-strategy-e06b-task23-two-head-001"
TARGET = "renta0426/cmi-flu-e06b-task23-two-head-20260908-001"
COMPLETION = "conditions={len((result.get('conditions') or {}))}"


def has_result_tasks_subscript(text: str) -> bool:
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "result":
            if isinstance(node.slice, ast.Constant) and node.slice.value == "tasks":
                return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    args = p.parse_args()
    text = args.runtime.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert not has_result_tasks_subscript(text)
    assert text.count(COMPLETION) == 1
    assert "tasks={len(result['tasks'])}" not in text
    assert f'REQUEST_ID = "{REQUEST_ID}"' in text
    assert f'TARGET_KERNEL = "{TARGET}"' in text
    assert '"observed_d28_used_as_d365_feature": False' in text
    assert '"d28_role": "auxiliary_training_label_only"' in text
    assert 'stage = "run_e06b"' in text
    assert 'stage = "validate_e06b"' in text
    assert "run_e06b(config, inputs" in text
    assert "kaggle competitions submit" not in text
    assert "competition_submit" not in text
    funcs = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for name in ("execute", "validate_result", "render_summary", "load_e06_module"):
        assert funcs.count(name) == 1, name
    print(
        "CMI_FLU_E06B_RUNTIME_CONTRACT_PASS completion_conditions=true "
        "result_tasks=false d28_feature=false execute=true submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
