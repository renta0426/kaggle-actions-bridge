"""Patch the flattened Poisoned Chalice Stage 1 notebook after assembly.

The research implementation normally imports separate Python modules.  The bridge
builder flattens those module definition cells into one notebook namespace, so a
runner constant can accidentally overwrite a source-module global.  This patch is
request-specific and deliberately narrow: namespace every runner constant, verify
there is no remaining runner/source constant collision, and add stable nbformat
cell IDs.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re


RENAMES = {
    "METHOD": "STAGE1_METHOD",
    "CORE_COMMIT": "STAGE1_CORE_COMMIT",
    "SOURCE_COMMIT": "STAGE1_SOURCE_COMMIT",
    "SEED": "STAGE1_SEED",
    "TRAIN_PER_LANGUAGE": "STAGE1_TRAIN_PER_LANGUAGE",
    "SHARD_SIZE": "STAGE1_SHARD_SIZE",
    "EXPECTED_TRAIN_ROWS": "STAGE1_EXPECTED_TRAIN_ROWS",
    "EXPECTED_VALIDATION_ROWS": "STAGE1_EXPECTED_VALIDATION_ROWS",
    "EXPECTED_BASE_FEATURES": "STAGE1_EXPECTED_BASE_FEATURES",
    "EXPECTED_STRUCTURE_FEATURES": "STAGE1_EXPECTED_STRUCTURE_FEATURES",
    "EXPECTED_FIM_FEATURES": "STAGE1_EXPECTED_FIM_FEATURES",
    "EXPECTED_OOF_AUC": "STAGE1_EXPECTED_OOF_AUC",
    "OOF_AUC_TOLERANCE": "STAGE1_OOF_AUC_TOLERANCE",
}


def source_text(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def top_level_upper_assignments(source: str) -> set[str]:
    tree = ast.parse(source)
    result: set[str] = set()
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                result.add(target.id)
    return result


def patch(notebook_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells") or []
    runner_indices = [
        index
        for index, cell in enumerate(cells)
        if cell.get("cell_type") == "code"
        and "EXPECTED_TRAIN_ROWS = 10000" in source_text(cell)
        and "EXPECTED_OOF_AUC = 0.664524" in source_text(cell)
        and "manifest = main()" in source_text(cell)
    ]
    if len(runner_indices) != 1:
        raise SystemExit(f"expected exactly one Stage1 runner cell, got {len(runner_indices)}")
    runner_index = runner_indices[0]
    runner = source_text(cells[runner_index])

    # The failure on request 003 was caused by the first mapping here:
    # starter_plus.EXPECTED_TRAIN_ROWS is a per-language dict, while the flattened
    # runner rebound EXPECTED_TRAIN_ROWS to integer 10000 in the same namespace.
    for old, new in sorted(RENAMES.items(), key=lambda item: -len(item[0])):
        runner = re.sub(rf"\b{re.escape(old)}\b", new, runner)
    cells[runner_index]["source"] = runner.splitlines(keepends=True)

    if "EXPECTED_TRAIN_ROWS = 10000" in runner:
        raise SystemExit("known request-003 namespace collision was not removed")
    if "STAGE1_EXPECTED_TRAIN_ROWS = 10000" not in runner:
        raise SystemExit("namespaced Stage1 train-row constant missing")
    if "STAGE1_EXPECTED_OOF_AUC = 0.664524" not in runner:
        raise SystemExit("namespaced frozen OOF contract missing")

    runner_constants = top_level_upper_assignments(runner)
    source_constants: set[str] = set()
    for index, cell in enumerate(cells):
        if index == runner_index or cell.get("cell_type") != "code":
            continue
        source = source_text(cell)
        if source.lstrip().startswith("%"):
            continue
        source_constants |= top_level_upper_assignments(source)
    collisions = sorted(runner_constants & source_constants)
    if collisions:
        raise SystemExit(f"runner/source uppercase namespace collision remains: {collisions}")

    # Compile every ordinary Python cell now so syntax problems fail before a
    # Kaggle GPU version is created.  The install-magic cell is intentionally skipped.
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = source_text(cell)
        if source.lstrip().startswith("%"):
            continue
        ast.parse(source, filename=f"cell-{index}")

    for index, cell in enumerate(cells):
        cell["id"] = f"pc-stage1-{index:02d}"

    notebook["cells"] = cells
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "patched",
                "runner_cell": runner_index,
                "runner_constants": sorted(runner_constants),
                "source_constants_checked": len(source_constants),
                "cell_ids_added": len(cells),
                "namespace_collisions": collisions,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    patch(args.notebook)


if __name__ == "__main__":
    main()
