#!/usr/bin/env python3
"""Outcome-independent E11b builder repair: active-AST stale-writer check."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

V1 = Path(__file__).with_name("cmi_flu_strategy_e11b_prepare.py")
V1_BLOB = "e79d38117faeed313eccd00b1380e6c63e895fa9"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def patched_source() -> str:
    data = V1.read_bytes()
    found = git_blob_sha(data)
    if found != V1_BLOB:
        raise SystemExit(f"E11b prepare-v2 ancestry changed:{found}")
    source = data.decode("utf-8")

    old_import = '''def install_tabpfn_wheel() -> dict:
    import base64 as _b64
    import importlib.metadata as _metadata
'''
    new_import = '''def install_tabpfn_wheel() -> dict:
    import base64 as _b64
    import importlib.metadata as _metadata
    import subprocess as _subprocess
'''
    if source.count(old_import) != 1:
        raise SystemExit("E11b prepare-v2 wheel-import anchor changed")
    source = source.replace(old_import, new_import, 1)
    if source.count("completed=subprocess.run(") != 1:
        raise SystemExit("E11b prepare-v2 wheel-subprocess anchor changed")
    source = source.replace("completed=subprocess.run(", "completed=_subprocess.run(", 1)

    old_check = '''    if "result['tasks']" in runtime:
        raise SystemExit("E11b generated runtime retains stale tasks writer access")
'''
    new_check = '''    tree = ast.parse(runtime)
    active_tasks = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != "result":
            continue
        sl = node.slice
        if isinstance(sl, ast.Constant) and sl.value == "tasks":
            active_tasks.append((getattr(node, "lineno", -1), getattr(node, "col_offset", -1)))
    if active_tasks:
        raise SystemExit(f"E11b generated runtime retains active stale tasks access:{active_tasks}")
'''
    if source.count(old_check) != 1:
        raise SystemExit("E11b prepare-v2 active-AST anchor changed")
    source = source.replace(old_check, new_check, 1)
    compile(source, str(V1), "exec")
    return source


def main() -> int:
    source = patched_source()
    namespace = {
        "__name__": "cmi_flu_strategy_e11b_prepare_v2_impl",
        "__file__": str(V1),
    }
    exec(compile(source, str(V1), "exec"), namespace, namespace)
    rc = int(namespace["main"]())
    print(
        "CMI_FLU_E11B_PREPARE_V2_PASS repair=active_ast_tasks_check "
        "wheel_subprocess_local_import=true science_change=false"
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
