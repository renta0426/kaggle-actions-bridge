#!/usr/bin/env python3
"""Final V3-06 builder: bind validated runtime to exact science main commit."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile

import cmi_flu_v3_v06_prepare_v3 as parent

OLD_SCIENCE_COMMIT = "dc35aa83fe32c725394e0bc354345a4535e9df45"
SCIENCE_COMMIT = "0d395e55e7d8d82a1167c5347cc65d0c051ad59f"
SCIENCE_BLOB = "5f75baa73750abed53b476ef9cf525e5dbf37a50"


def build_runtime(root: Path, output: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="v306-v4-") as td:
        raw = parent.build_runtime(root, Path(td) / "parent.py")
    count = raw.count(OLD_SCIENCE_COMMIT)
    if count < 1:
        raise ValueError("V3-06 final science commit binding anchor missing")
    raw = raw.replace(OLD_SCIENCE_COMMIT, SCIENCE_COMMIT)
    if OLD_SCIENCE_COMMIT in raw or SCIENCE_COMMIT not in raw:
        raise ValueError("V3-06 final science commit binding failed")
    if f'V306_SCIENCE_BLOB = {SCIENCE_BLOB!r}' not in raw:
        raise ValueError("V3-06 final science blob binding changed")
    compile(raw, "v306-runtime-v4.py", "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(raw, encoding="utf-8")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = build_runtime(args.repository_root.resolve(), args.output.resolve()).encode()
    print(
        f"V306_PREPARE_V4 PASS bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()} "
        f"science_commit={SCIENCE_COMMIT} science_blob={SCIENCE_BLOB} exact_main=true submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
