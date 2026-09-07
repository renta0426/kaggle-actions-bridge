#!/usr/bin/env python3
"""Repair only the E02 runtime output sink; scientific sources stay byte-exact.

Input is the SHA-256 locked runtime already pushed to Kaggle. Unlike the earlier
string-slicing chain, edits are confined to the AST's top-level execute function.
This builder never downloads code, authenticates, or launches a Notebook.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import runpy

ORIGINAL_SHA = "ad36802efe2af9d4678b5d216b4afbb904a74746ec971818e75fdb2692709f37"
SOURCE_KEYS = ("E01_SOURCE", "E01_V2_SOURCE", "E02_SOURCE", "E02_V2_SOURCE", "CONFIG_TEXT", "PACKAGE_B64", "B21_ADAPTER_SOURCE")


def replace_top_level_function(source: str, name: str, transform) -> str:
    nodes = [node for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(nodes) != 1:
        raise ValueError("top_level_function_identity_mismatch")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    original = "".join(lines[node.lineno - 1:node.end_lineno])
    replacement = transform(original)
    parsed = ast.parse(replacement)
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef) or parsed.body[0].name != name:
        raise ValueError("replacement_function_contract")
    result = "".join(lines[:node.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno:])
    compile(result, "repaired_e02.py", "exec")
    return result


def repair(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_SHA:
        raise ValueError("unapproved_original_runtime")
    def sink(body: str) -> str:
        replacements = {
            '    runtime_root = output_dir / "e01-runtime"': '    import tempfile\n    runtime_root = Path(tempfile.mkdtemp(prefix="cmi-e02-runtime-"))',
            '            "frozen_incumbent": result["frozen_incumbent"],': '            "task": result["task"],\n            "fixed_conditions": result["fixed_conditions"],\n            "strategy_e02_blob_sha": E02_BLOB,\n            "strategy_e02_v2_blob_sha": E02_V2_BLOB,',
            "tasks={len(result['tasks'])}": "task={result['task']} conditions={len(result['conditions'])}",
            '"CMI_FLU_E01_COMPLETE "': '"CMI_FLU_E02_COMPLETE "',
        }
        for old, new in replacements.items():
            if body.count(old) != 1:
                raise ValueError("frozen_output_sink_contract_changed")
            body = body.replace(old, new, 1)
        for stage in ("load", "run", "validate"):
            body = body.replace(f'stage = "{stage}_e01"', f'stage = "{stage}_e02"')
        return body
    result = replace_top_level_function(source, "execute", sink)
    # The embedded science/config/package string literals must remain identical.
    def constants(text):
        values = {}
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in SOURCE_KEYS:
                values[node.targets[0].id] = ast.literal_eval(node.value)
        return values
    if len(constants(source)) != len(SOURCE_KEYS) or constants(source) != constants(result):
        raise ValueError("scientific_source_changed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("release_output_already_exists")
    result = repair(args.original.read_text())
    args.output.write_text(result)
    namespace = runpy.run_path(str(args.output), run_name="e02_release_selftest")
    if namespace["self_test"]() != 0:
        raise RuntimeError("release_selftest_failed")
    print("E02_RELEASE_BUILD PASS science_unchanged=true sha256=" + hashlib.sha256(result.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
