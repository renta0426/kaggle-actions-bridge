#!/usr/bin/env python3
"""Map a sanitized E05 NameError hash to a code identifier without Competition data."""
from __future__ import annotations
import argparse, ast, hashlib, io, runpy, zipfile
from pathlib import Path


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def names_from_source(source: str) -> set[str]:
    tree = ast.parse(source)
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--error-code", required=True)
    a = p.parse_args()
    ns = runpy.run_path(str(a.runtime), run_name="e05_nameerror_hash_diagnostic")
    names: set[str] = set()
    for key in (
        "E05_SOURCE",
        "HAI_TRANSFER_SOURCE",
        "E01_SOURCE",
        "E01_V2_SOURCE",
        "B21_ADAPTER_SOURCE",
    ):
        source = ns.get(key)
        if isinstance(source, str):
            names.update(names_from_source(source))
    with zipfile.ZipFile(io.BytesIO(ns["package_bytes"]())) as zf:
        for member in zf.namelist():
            if member.endswith(".py"):
                names.update(names_from_source(zf.read(member).decode("utf-8")))

    matches: list[tuple[str, str]] = []
    for name in sorted(names):
        messages = (
            f"NameError:name '{name}' is not defined",
            f"NameError:cannot access free variable '{name}' where it is not associated with a value in enclosing scope",
        )
        for message in messages:
            if short_hash(message) == a.error_code:
                matches.append((name, message))
    if len(matches) != 1:
        raise SystemExit(
            f"E05_NAMEERROR_HASH_UNRESOLVED matches={matches} candidate_names={len(names)}"
        )
    name, message = matches[0]
    print(
        "CMI_FLU_E05_NAMEERROR_HASH_MATCH "
        f"identifier={name} error_code={a.error_code} "
        f"full_sha256={hashlib.sha256(message.encode()).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
