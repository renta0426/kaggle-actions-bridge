"""Minimal nbformat-v4 writer used only by the allowlisted bridge builder."""

from __future__ import annotations

import json
from pathlib import Path


class NotebookNode(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


class v4:
    @staticmethod
    def new_markdown_cell(source=""):
        return NotebookNode(
            {
                "cell_type": "markdown",
                "metadata": NotebookNode(),
                "source": source,
            }
        )

    @staticmethod
    def new_code_cell(source=""):
        return NotebookNode(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": NotebookNode(),
                "outputs": [],
                "source": source,
            }
        )

    @staticmethod
    def new_notebook():
        return NotebookNode(
            {
                "cells": [],
                "metadata": NotebookNode(),
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )


def validate(notebook):
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("invalid notebook v4 root")
    seen = set()
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") not in {"code", "markdown", "raw"}:
            raise ValueError(f"invalid cell type at {index}")
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in seen:
            raise ValueError(f"invalid or duplicate cell id at {index}")
        seen.add(cell_id)
        if "source" not in cell:
            raise ValueError(f"cell source missing at {index}")
        if cell["cell_type"] == "code":
            if "execution_count" not in cell or not isinstance(cell.get("outputs"), list):
                raise ValueError(f"invalid code cell at {index}")


def write(notebook, path):
    validate(notebook)
    Path(path).write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
