#!/usr/bin/env python3
"""E11b full-validator regression with active-AST and frozen-package bootstrap."""
from __future__ import annotations

import hashlib
from pathlib import Path

V1 = Path(__file__).with_name("cmi_flu_strategy_e11b_validator_contract_test.py")
V1_BLOB = "caed05b707187924f75e311149561f55c89a9a84"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def patched_source() -> str:
    data = V1.read_bytes()
    found = git_blob_sha(data)
    if found != V1_BLOB:
        raise SystemExit(f"E11b validator-v2 ancestry changed:{found}")
    source = data.decode("utf-8")
    if source.count("import importlib.util\n") != 1:
        raise SystemExit("E11b validator-v2 import anchor changed")
    source = source.replace(
        "import importlib.util\n",
        "import importlib.util\nimport ast\nimport sys\nimport tempfile\n",
        1,
    )
    old = '''    if "result['tasks']" in text:
        raise SystemExit("E11b validator runtime retains stale tasks writer access")
'''
    new = '''    tree = ast.parse(text)
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
        raise SystemExit(f"E11b validator runtime retains active stale tasks writer access:{active_tasks}")
'''
    if source.count(old) != 1:
        raise SystemExit("E11b validator-v2 stale-check anchor changed")
    source = source.replace(old, new, 1)

    old_bootstrap = '''    runtime = load_runtime(runtime_path)
    run_e11b, run_synthetic, e11b = runtime.load_e11b_modules()
'''
    new_bootstrap = '''    runtime = load_runtime(runtime_path)
    package_root = Path(tempfile.mkdtemp(prefix="cmi-flu-e11b-validator-"))
    package_path = package_root / "cmi_flu_bundle.zip"
    package_path.write_bytes(runtime.package_bytes())
    sys.path.insert(0, str(package_path))
    run_e11b, run_synthetic, e11b = runtime.load_e11b_modules()
'''
    if source.count(old_bootstrap) != 1:
        raise SystemExit("E11b validator-v2 package-bootstrap anchor changed")
    source = source.replace(old_bootstrap, new_bootstrap, 1)
    compile(source, str(V1), "exec")
    return source


def main() -> int:
    source = patched_source()
    namespace = {
        "__name__": "cmi_flu_strategy_e11b_validator_v2_impl",
        "__file__": str(V1),
    }
    exec(compile(source, str(V1), "exec"), namespace, namespace)
    rc = int(namespace["main"]())
    print(
        "CMI_FLU_E11B_VALIDATOR_V2_PASS active_ast_tasks_check=true "
        "frozen_package_bootstrap=true"
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
