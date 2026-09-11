#!/usr/bin/env python3
"""Apply the one-line E12a-v2 terminal-test repair, then build the exact runtime."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

BASE = "scripts/cmi_flu_strategy_e12a_v2_prepare.py"
BASE_BLOB = "e9909b877ec9d2d46cc287174e21a5a865a69463"
OLD = '    if "tasks=" in terminal:\n'
NEW = '    if " tasks=" in terminal:\n'


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    base_path = root / BASE
    raw = base_path.read_bytes()
    if git_blob(raw) != BASE_BLOB:
        raise SystemExit("E12a-v2 base builder changed before terminal-test repair")
    source = raw.decode("utf-8")
    if source.count(OLD) != 1 or source.count(NEW) != 0:
        raise SystemExit("E12a-v2 terminal-test repair anchor changed")
    patched = source.replace(OLD, NEW, 1)
    if patched.count(NEW) != 1 or patched.count(OLD) != 0:
        raise SystemExit("E12a-v2 terminal-test repair incomplete")
    with tempfile.TemporaryDirectory(prefix="cmi-e12a-v2-prepare-repair-") as tmp:
        patched_path = Path(tmp) / "prepare.py"
        patched_path.write_text(patched, encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(patched_path),
                "--repository-root",
                str(root),
                "--output",
                str(args.output.expanduser().resolve()),
            ],
            check=True,
        )
    print(
        "CMI_FLU_E12A_V2_PREPARE_V2 PASS "
        f"base_blob={BASE_BLOB} repair=terminal_exact_token_boundary_only "
        "science_changed=false runtime_execution_logic_changed=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
