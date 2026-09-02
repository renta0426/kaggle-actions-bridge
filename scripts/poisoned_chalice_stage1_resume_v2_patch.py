"""Patch only the immutable request/source-version provenance in the resume notebook."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


OLD_REQUEST_ID = "20260903-poisoned-chalice-stage1-resume-001"
NEW_REQUEST_ID = "20260903-poisoned-chalice-stage1-resume-002"
OLD_SOURCE_VERSION = 2
NEW_SOURCE_VERSION = 3
EXPECTED_CELL_ID = "pc-stage1-resume-01"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one provenance marker: {old!r}")
    return source.replace(old, new, 1)


def patch(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells") or []
    matches = [cell for cell in cells if cell.get("id") == EXPECTED_CELL_ID]
    if len(matches) != 1 or matches[0].get("cell_type") != "code":
        raise RuntimeError("resume runtime cell identity changed")
    cell = matches[0]
    value = cell.get("source", "")
    source = "".join(value) if isinstance(value, list) else str(value)
    source = replace_once(
        source,
        f"REQUEST_ID = {OLD_REQUEST_ID!r}",
        f"REQUEST_ID = {NEW_REQUEST_ID!r}",
    )
    source = replace_once(
        source,
        f"SOURCE_KERNEL_VERSION = {OLD_SOURCE_VERSION}",
        f"SOURCE_KERNEL_VERSION = {NEW_SOURCE_VERSION}",
    )
    if OLD_REQUEST_ID in source:
        raise RuntimeError("old resume request ID remains after patch")
    if f"SOURCE_KERNEL_VERSION = {OLD_SOURCE_VERSION}" in source:
        raise RuntimeError("old source-kernel version remains after patch")
    ast.parse(source, filename=path.name)
    cell["source"] = source.splitlines(keepends=True)
    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "patched",
        "request_id": NEW_REQUEST_ID,
        "source_kernel_version": NEW_SOURCE_VERSION,
        "algorithm_changed": False,
        "changed_constants": ["REQUEST_ID", "SOURCE_KERNEL_VERSION"],
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    args = parser.parse_args()
    patch(args.notebook)


if __name__ == "__main__":
    main()
