#!/usr/bin/env python3
"""Diagnose a sanitized E05 DataContractError hash from the exact generated runtime.

Secret-free: this inspects source strings and the frozen package only. It never
contacts Kaggle and never reads Competition data.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import runpy
import zipfile
from pathlib import Path


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def raise_messages(source: str, label: str):
    tree = ast.parse(source)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        exc = node.exc
        if not isinstance(exc, ast.Call):
            continue
        func = exc.func
        name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
        if name != "DataContractError" or not exc.args:
            continue
        arg = exc.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.append(("static", arg.value, label, getattr(node, "lineno", -1)))
        elif isinstance(arg, ast.JoinedStr):
            pieces = []
            for part in arg.values:
                if isinstance(part, ast.Constant):
                    pieces.append(str(part.value))
                elif isinstance(part, ast.FormattedValue):
                    try:
                        expr = ast.unparse(part.value)
                    except Exception:
                        expr = "?"
                    pieces.append("{" + expr + "}")
            out.append(("fstring", "".join(pieces), label, getattr(node, "lineno", -1)))
        else:
            try:
                text = ast.unparse(arg)
            except Exception:
                text = type(arg).__name__
            out.append(("expr", text, label, getattr(node, "lineno", -1)))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--error-code", required=True)
    a = p.parse_args()
    ns = runpy.run_path(str(a.runtime), run_name="e05_datacontract_hash_diagnostic")
    sources = []
    for key in ("E05_SOURCE", "HAI_TRANSFER_SOURCE", "E01_SOURCE", "E01_V2_SOURCE", "B21_ADAPTER_SOURCE"):
        value = ns.get(key)
        if isinstance(value, str):
            sources.append((key, value))
    with zipfile.ZipFile(io.BytesIO(ns["package_bytes"]())) as zf:
        for member in zf.namelist():
            if member.endswith(".py"):
                sources.append((member, zf.read(member).decode("utf-8")))

    candidates = []
    for label, source in sources:
        candidates.extend(raise_messages(source, label))

    static_matches = []
    for kind, message, label, line in candidates:
        if kind != "static":
            continue
        full = "DataContractError:" + message
        if short_hash(full) == a.error_code:
            static_matches.append((message, label, line, hashlib.sha256(full.encode()).hexdigest()))

    print(f"E05_DATACONTRACT_DIAG sources={len(sources)} raises={len(candidates)} static_matches={static_matches}")
    print("E05_DATACONTRACT_DYNAMIC_TEMPLATES_BEGIN")
    for kind, message, label, line in candidates:
        if kind != "static":
            print(f"{label}:{line}:{kind}:{message}")
    print("E05_DATACONTRACT_DYNAMIC_TEMPLATES_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
