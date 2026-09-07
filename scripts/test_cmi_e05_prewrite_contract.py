#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one function {name}")
    return matches[0]


def call_names(node: ast.AST) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        target = child.func
        if isinstance(target, ast.Name):
            result.append((child.lineno, target.id))
        elif isinstance(target, ast.Attribute):
            result.append((child.lineno, target.attr))
    return sorted(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")
    tree = ast.parse(text)

    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "direct_target_absent" in names:
        raise SystemExit("obsolete pre-write exact metadata lookup restored")
    guard = function(tree, "prewrite_guard")
    push = function(tree, "push")

    guard_calls = call_names(guard)
    if not any(name == "kernels_list" for _, name in guard_calls):
        raise SystemExit("prewrite duplicate sentinel missing")
    if any(name == "kernel_meta" for _, name in guard_calls):
        raise SystemExit("prewrite guard must not require exact metadata for absent target")

    push_calls = call_names(push)
    run_lines = [line for line, name in push_calls if name == "run"]
    exact_lines = [line for line, name in push_calls if name == "kernel_meta"]
    if len(run_lines) != 1 or not exact_lines or min(exact_lines) <= run_lines[0]:
        raise SystemExit("exact metadata must be obtained only after the single CLI write")

    required = (
        "CMI_FLU_E05_WRITE_RECEIPT",
        'phase="before_write"',
        'phase="after_cli_write"',
        'phase="exact_identity_confirmed"',
        "E05 push acknowledged but direct metadata unconfirmed",
    )
    if any(token not in text for token in required):
        raise SystemExit("safe write receipt contract incomplete")
    if "competition_submit" in text or "kaggle competitions submit" in text:
        raise SystemExit("E05 executor contains competition submission path")

    print("CMI_FLU_E05_PREWRITE_CONTRACT PASS search_is_sentinel_only=true exact_metadata_postwrite=true receipts=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
