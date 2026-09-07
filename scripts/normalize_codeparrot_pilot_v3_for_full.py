#!/usr/bin/env python3
"""Normalize the sealed CodeParrot pilot-v3 manifest selection source shape.

This is a source-shape compatibility helper only. It does not change scoring,
rows, labels, metrics, model identity, or authorization. It finds the
``manifest`` dict semantically via Python AST, verifies the frozen pilot
selection value, and rewrites only that key/value span into the canonical
format expected by the already-audited 200->4000 transformer.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import tempfile

import nbformat

OLD_SELECTION = "first_200_sample_ids_from_label_free_4000_prediction_input"
CANONICAL = '"selection":\n                "' + OLD_SELECTION + '"'
REQUIRED_MARKERS = (
    "CODEPARROT_PRIVATE_SNAPSHOT_COMPAT_V3",
    "CODEPARROT_MEAN_LOG_RANK_REPAIR_V2",
    'summary["mean_log_rank"] = float(np.log1p(rank_integer).mean())',
    '"full_4000_scoring_authorized": False',
    '"performance_metrics_computed": False',
    '"membership_join_performed": False',
)


def _source(cell: nbformat.NotebookNode) -> str:
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def _offset(lines: list[str], lineno: int, col: int) -> int:
    return sum(len(line) for line in lines[: lineno - 1]) + col


def _manifest_selection_span(source: str) -> tuple[int, int]:
    tree = ast.parse(source)
    matches: list[tuple[ast.AST, ast.AST]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != "manifest" or not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "selection"
                and isinstance(value, ast.Constant)
                and value.value == OLD_SELECTION
            ):
                matches.append((key, value))
    if len(matches) != 1:
        raise RuntimeError(f"expected one manifest selection AST field, found {len(matches)}")
    key, value = matches[0]
    if any(getattr(node, attr, None) is None for node in (key, value) for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset")):
        raise RuntimeError("manifest selection AST positions unavailable")
    lines = source.splitlines(keepends=True)
    start = _offset(lines, key.lineno, key.col_offset)
    end = _offset(lines, value.end_lineno, value.end_col_offset)
    return start, end


def normalize_notebook(root: Path) -> Path:
    candidates = sorted(root.glob("*.ipynb"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one notebook, found {len(candidates)}")
    path = candidates[0]
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)

    combined = "\n".join(_source(cell) for cell in notebook.cells if cell.cell_type == "code")
    missing = [marker for marker in REQUIRED_MARKERS if marker not in combined]
    if missing:
        raise RuntimeError(f"sealed pilot-v3 markers missing before normalization: {missing}")

    indexes: list[int] = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source = _source(cell)
        try:
            _manifest_selection_span(source)
        except (SyntaxError, RuntimeError):
            continue
        indexes.append(index)
    if len(indexes) != 1:
        raise RuntimeError(f"expected one manifest-bearing code cell, found {len(indexes)}")

    index = indexes[0]
    source = _source(notebook.cells[index])
    start, end = _manifest_selection_span(source)
    normalized = source[:start] + CANONICAL + source[end:]
    compile(normalized, f"{path}:normalized_manifest", "exec")
    if CANONICAL not in normalized:
        raise RuntimeError("canonical manifest selection was not materialized")
    # Semantic re-parse is the authoritative check; formatting is secondary.
    _manifest_selection_span(normalized)
    notebook.cells[index]["source"] = normalized
    notebook.cells[index]["execution_count"] = None
    notebook.cells[index]["outputs"] = []
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    return path


def _fixture(root: Path, manifest_source: str) -> None:
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("# CODEPARROT_PRIVATE_SNAPSHOT_COMPAT_V3\n# CODEPARROT_MEAN_LOG_RANK_REPAIR_V2\nsummary = {}\nrank_integer=np.array([1])\nsummary[\"mean_log_rank\"] = float(np.log1p(rank_integer).mean())"),
        nbformat.v4.new_code_cell(
            manifest_source
            + '\nprint({"full_4000_scoring_authorized": False, "performance_metrics_computed": False, "membership_join_performed": False})\n'
        ),
    ])
    nbformat.write(notebook, root / "pilot.ipynb")


def self_test() -> None:
    variants = (
        'manifest={"selection":"' + OLD_SELECTION + '"}',
        'manifest = {\n    "selection": \n        "' + OLD_SELECTION + '",\n}',
        "manifest = {'selection': '" + OLD_SELECTION + "'}",
    )
    for variant in variants:
        with tempfile.TemporaryDirectory() as temp_value:
            root = Path(temp_value)
            _fixture(root, variant)
            path = normalize_notebook(root)
            source = "\n".join(_source(cell) for cell in nbformat.read(path, as_version=4).cells if cell.cell_type == "code")
            assert CANONICAL in source
            assert OLD_SELECTION in source
            assert 'summary["mean_log_rank"] = float(np.log1p(rank_integer).mean())' in source
    print("CODEPARROT_V3_MANIFEST_NORMALIZER_SELF_TEST PASS variants=3 semantic_ast=1 scoring_changes=0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.notebook_dir is None:
        parser.error("--notebook-dir is required")
    path = normalize_notebook(args.notebook_dir)
    print(f"CODEPARROT_V3_MANIFEST_NORMALIZER PASS notebook={path.name} semantic_ast=1 scoring_changes=0")


if __name__ == "__main__":
    main()
