#!/usr/bin/env python3
"""Build E03 request 003 by repairing only Competition-input staging.

Request 002 proved the shorter Kaggle identity and reached real runtime execution,
but the E03 staging tree exposed only CORE_FILES while the frozen B2.1 loader
verifies every non-self entry in the delivered md5sum manifest.  Request 003
keeps the science/runtime/resource contract fixed and changes only data/raw to
point at the complete Competition input directory, matching the previously
successful Task1.1 prior-immunity runtime pattern.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

BASE = Path(__file__).with_name("cmi_flu_strategy_e03_prepare_v5.py")
REQUEST_ID = "20260908-cmi-flu-strategy-e03-task11-optional-view-fusion-003"
TARGET_KERNEL = "renta0426/cmi-flu-e03-task11-optview-20260908-003"
REQUEST_PATH = "requests/cmi-flu-strategy-e03-task11-optional-view-fusion-003.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    v5 = load_module(BASE, "cmi_flu_strategy_e03_prepare_v5_for_v7")
    original_v5_load_base = v5.load_base

    def load_base_v7():
        v2 = original_v5_load_base()
        v2.REQUEST_ID = REQUEST_ID
        v2.TARGET_KERNEL = TARGET_KERNEL
        v2.REQUEST_PATH = REQUEST_PATH

        original_v2_load_module = v2.load_module

        def load_module_v7(path: Path, name: str):
            loaded = original_v2_load_module(path, name)
            if Path(path).name == "cmi_flu_strategy_e03_prepare.py":
                loaded.REQUEST_ID = REQUEST_ID
                loaded.TARGET_KERNEL = TARGET_KERNEL
                loaded.REQUEST_PATH = REQUEST_PATH
            return loaded

        v2.load_module = load_module_v7
        original_patch_runtime = v2.patch_runtime

        def patch_runtime_v7(runtime: str, adapter: str) -> str:
            runtime = original_patch_runtime(runtime, adapter)
            old = '''    data_dir = work / "data" / "raw"\n    data_dir.mkdir(parents=True, exist_ok=True)\n    for name in CORE_FILES:\n        source = input_dir / name\n        target = data_dir / name\n        try:\n            target.symlink_to(source)\n        except OSError:\n            shutil.copy2(source, target)\n'''
            new = '''    data_parent = work / "data"\n    data_parent.mkdir(parents=True, exist_ok=True)\n    data_dir = data_parent / "raw"\n    data_dir.symlink_to(input_dir, target_is_directory=True)\n    manifest = data_dir / "md5sum"\n    if not manifest.is_file():\n        raise BridgeContractError("md5_manifest_not_staged")\n    manifest_entries = 0\n    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):\n        if not line.strip():\n            continue\n        parts = line.split(maxsplit=1)\n        if len(parts) != 2:\n            raise BridgeContractError(f"malformed_md5_manifest_line:{line_number}")\n        digest, name = parts\n        name = name.strip().lstrip("*")\n        rel = Path(name)\n        if len(digest) != 32 or rel.is_absolute() or ".." in rel.parts:\n            raise BridgeContractError(f"unsafe_md5_manifest_entry:{line_number}")\n        if name != "md5sum" and not (data_dir / rel).is_file():\n            raise BridgeContractError(f"md5_manifest_entry_not_staged:{name}")\n        if name != "md5sum":\n            manifest_entries += 1\n    if manifest_entries < 1:\n        raise BridgeContractError("md5_manifest_has_no_data_entries")\n'''
            runtime = replace_once(runtime, old, new, "runtime complete Competition input staging")
            if 'data_dir.symlink_to(input_dir, target_is_directory=True)' not in runtime:
                raise SystemExit("E03 full Competition input symlink repair missing")
            if 'md5_manifest_entry_not_staged' not in runtime:
                raise SystemExit("E03 MD5 staging preflight missing")
            compile(runtime, "generated_cmi_e03_runtime_v7.py", "exec")
            return runtime

        v2.patch_runtime = patch_runtime_v7
        return v2

    v5.load_base = load_base_v7
    return int(v5.main())


if __name__ == "__main__":
    raise SystemExit(main())
