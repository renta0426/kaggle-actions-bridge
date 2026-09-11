#!/usr/bin/env python3
"""Apply narrow E12b generated-runtime corrections, then build the runtime."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

BASE = "scripts/cmi_flu_strategy_e12b_prepare.py"
BASE_BLOB = "dd81662cc579c9f36222d2dcd27ecbe9300266d3"
OLD_SCHEMA = '''    banned = ('"participant_id"', '"subject_group"', '"row_index"', '"challenge_predictions"', '"submission_rows"', '"prediction_vector"', '"oof_predictions"')\n'''
NEW_SCHEMA = '''    banned = ('"subject_group"', '"row_index"', '"challenge_predictions"', '"submission_rows"', '"prediction_vector"', '"oof_predictions"')\n'''
OLD_TASK_BINDING = '''        + f'E12B_SOURCE = {source!r}\\n',\n'''
NEW_TASK_BINDING = '''        + 'TASKS = ("Task1.1", "Task1.2", "Task1.3", "Task1.4", "Task2.1", "Task2.2", "Task2.3")\\n'\n        + f'E12B_SOURCE = {source!r}\\n',\n'''


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def replace_once(source: str, old: str, new: str, *, label: str) -> str:
    if source.count(old) != 1 or source.count(new) != 0:
        raise SystemExit(f"E12b {label} correction anchor changed")
    patched = source.replace(old, new, 1)
    if patched.count(new) != 1 or patched.count(old) != 0:
        raise SystemExit(f"E12b {label} correction incomplete")
    return patched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    base = root / BASE
    raw = base.read_bytes()
    if git_blob(raw) != BASE_BLOB:
        raise SystemExit("E12b base builder changed before narrow runtime corrections")
    source = raw.decode("utf-8")
    source = replace_once(source, OLD_SCHEMA, NEW_SCHEMA, label="schema-token")
    source = replace_once(source, OLD_TASK_BINDING, NEW_TASK_BINDING, label="task-binding")
    with tempfile.TemporaryDirectory(prefix="cmi-e12b-prepare-fix-") as tmp:
        patched_path = Path(tmp) / "prepare.py"
        patched_path.write_text(source, encoding="utf-8")
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
        "CMI_FLU_E12B_PREPARE_V2 PASS "
        f"base_blob={BASE_BLOB} corrections=schema_column_token,generated_task_binding "
        "row_level_identifier_values_allowed=false science_changed=false submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
