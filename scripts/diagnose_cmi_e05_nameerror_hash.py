#!/usr/bin/env python3
"""Map a sanitized E05 NameError hash to a code token without Competition data."""
from __future__ import annotations
import argparse, ast, hashlib, io, re, runpy, zipfile
from pathlib import Path

TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def tokens_from_source(source: str) -> set[str]:
    tree = ast.parse(source)
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return names | set(TOKEN_RE.findall(source))


def possible_messages(name: str) -> tuple[str, ...]:
    return (
        f"NameError:name '{name}' is not defined",
        f"NameError:free variable '{name}' referenced before assignment in enclosing scope",
        f"NameError:cannot access free variable '{name}' where it is not associated with a value in enclosing scope",
        f"NameError:cannot access local variable '{name}' where it is not associated with a value",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--error-code", required=True)
    a = p.parse_args()
    runtime_text = a.runtime.read_text(encoding="utf-8")
    ns = runpy.run_path(str(a.runtime), run_name="e05_nameerror_hash_diagnostic")
    tokens: set[str] = set(TOKEN_RE.findall(runtime_text))
    for key in (
        "E05_SOURCE", "HAI_TRANSFER_SOURCE", "E01_SOURCE", "E01_V2_SOURCE", "B21_ADAPTER_SOURCE",
    ):
        source = ns.get(key)
        if isinstance(source, str):
            tokens.update(tokens_from_source(source))
    with zipfile.ZipFile(io.BytesIO(ns["package_bytes"]())) as zf:
        for member in zf.namelist():
            if member.endswith(".py"):
                tokens.update(tokens_from_source(zf.read(member).decode("utf-8")))

    matches: list[tuple[str, str]] = []
    for token in sorted(tokens):
        for message in possible_messages(token):
            if short_hash(message) == a.error_code:
                matches.append((token, message))
    if len(matches) != 1:
        raise SystemExit(f"E05_NAMEERROR_HASH_UNRESOLVED matches={matches} candidate_tokens={len(tokens)}")
    token, message = matches[0]
    definition_count = len(re.findall(rf"(?m)^def\s+{re.escape(token)}\s*\(", runtime_text))
    bare_load_count = len(re.findall(rf"\b{re.escape(token)}\s*\(", runtime_text))
    namespace_present = token in ns
    print(
        "CMI_FLU_E05_NAMEERROR_HASH_MATCH "
        f"identifier={token} error_code={a.error_code} "
        f"definition_count={definition_count} call_token_count={bare_load_count} "
        f"runtime_namespace_present={str(namespace_present).lower()} "
        f"full_sha256={hashlib.sha256(message.encode()).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
