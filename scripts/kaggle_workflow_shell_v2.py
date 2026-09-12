#!/usr/bin/env python3
"""Credential-free shell-state checks for GitHub Actions workflows.

GitHub's ``$GITHUB_ENV`` publishes values to later *steps*. It does not update
the environment of the shell process that appends to the file. A workflow that
writes ``NAME=...`` to ``$GITHUB_ENV`` and then expands ``$NAME`` later in the
same ``run: |`` block therefore has a deterministic pre-write failure unless
NAME was also assigned/exported in that shell.

This linter is intentionally independent of Kaggle credentials and resource
capacity. It is a static bridge correctness check.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable


class ShellPolicyError(RuntimeError):
    pass


# Actions steps normally use ``- run: |`` while mapping-style reusable blocks
# can use ``run: |``. The indentation captured here is the indentation of the
# list marker / mapping key, which is also where the next step begins.
_RUN_BLOCK = re.compile(
    r"^(?P<indent>[ ]*)(?:-[ ]+)?run:[ ]*[|>][+-]?[0-9]*[ ]*$"
)
_DIRECT_ASSIGN = re.compile(r"^[ ]*(?:export[ ]+)?(?P<name>[A-Z][A-Z0-9_]*)=")
_GITHUB_ENV_NAME = re.compile(r"\b(?P<name>[A-Z][A-Z0-9_]*)=")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def run_blocks(text: str) -> list[tuple[int, list[str]]]:
    """Extract literal/folded ``run`` blocks without depending on PyYAML."""
    lines = text.splitlines()
    blocks: list[tuple[int, list[str]]] = []
    index = 0
    while index < len(lines):
        match = _RUN_BLOCK.match(lines[index])
        if not match:
            index += 1
            continue
        base_indent = len(match.group("indent"))
        start_line = index + 1
        index += 1
        block: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.strip() and _indent(line) <= base_indent:
                break
            block.append(line)
            index += 1
        blocks.append((start_line, block))
    return blocks


def _plain_reference(line: str, name: str) -> bool:
    # Parameter expansions with a fallback such as ${NAME-} are safe under
    # `set -u` and intentionally do not claim that NAME exists in this shell.
    return bool(
        re.search(r"\$\{" + re.escape(name) + r"\}", line)
        or re.search(r"\$" + re.escape(name) + r"(?![A-Za-z0-9_])", line)
    )


def validate_workflow(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for start_line, block in run_blocks(text):
        shell_assigned: set[str] = set()
        published: dict[str, int] = {}
        for offset, line in enumerate(block, 1):
            stripped = line.lstrip()
            assignment = _DIRECT_ASSIGN.match(stripped)
            if assignment and not stripped.startswith(("echo ", "printf ")):
                shell_assigned.add(assignment.group("name"))

            if "GITHUB_ENV" in line and ">>" in line:
                before_redirect = line.split(">>", 1)[0]
                for match in _GITHUB_ENV_NAME.finditer(before_redirect):
                    name = match.group("name")
                    if name == "GITHUB_ENV":
                        continue
                    published.setdefault(name, start_line + offset)
                continue

            for name, publish_line in published.items():
                if name in shell_assigned:
                    continue
                if _plain_reference(line, name):
                    use_line = start_line + offset
                    raise ShellPolicyError(
                        "same_step_github_env_visibility_forbidden:"
                        f"{path}:{name}:published_line={publish_line}:used_line={use_line}"
                    )


def _all_workflows(root: Path) -> dict[str, Path]:
    directory = root / ".github" / "workflows"
    if not directory.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".yml", ".yaml"}
    }


def validate_changed(base: Path, head: Path) -> list[str]:
    base_files = _all_workflows(base)
    head_files = _all_workflows(head)
    checked: list[str] = []
    for rel, path in sorted(head_files.items()):
        old = base_files.get(rel)
        if old is not None and old.read_bytes() == path.read_bytes():
            continue
        validate_workflow(path)
        checked.append(rel)
    return checked


def _pass(items: Iterable[str]) -> None:
    values = list(items)
    print(f"KAGGLE_WORKFLOW_SHELL_V2 PASS checked={len(values)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("workflow")
    one.add_argument("path", type=Path)
    changed = sub.add_parser("changed")
    changed.add_argument("base", type=Path)
    changed.add_argument("head", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "workflow":
            validate_workflow(args.path)
            _pass([str(args.path)])
        else:
            _pass(validate_changed(args.base, args.head))
    except ShellPolicyError as exc:
        print(f"KAGGLE_WORKFLOW_SHELL_V2 FAIL {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
