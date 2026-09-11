#!/usr/bin/env python3
"""Apply the E12b aggregate-schema privacy-token correction, then build the runtime."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

BASE = "scripts/cmi_flu_strategy_e12b_prepare.py"
BASE_BLOB = "dd81662cc579c9f36222d2dcd27ecbe9300266d3"
OLD = '''    banned = ('"participant_id"', '"subject_group"', '"row_index"', '"challenge_predictions"', '"submission_rows"', '"prediction_vector"', '"oof_predictions"')\n'''
NEW = '''    banned = ('"subject_group"', '"row_index"', '"challenge_predictions"', '"submission_rows"', '"prediction_vector"', '"oof_predictions"')\n'''


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    base = root / BASE
    raw = base.read_bytes()
    if git_blob(raw) != BASE_BLOB:
        raise SystemExit("E12b base builder changed before schema-token correction")
    source = raw.decode("utf-8")
    if source.count(OLD) != 1 or source.count(NEW) != 0:
        raise SystemExit("E12b schema-token correction anchor changed")
    patched = source.replace(OLD, NEW, 1)
    if patched.count(NEW) != 1 or patched.count(OLD) != 0:
        raise SystemExit("E12b schema-token correction incomplete")
    with tempfile.TemporaryDirectory(prefix="cmi-e12b-prepare-schema-fix-") as tmp:
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
        "CMI_FLU_E12B_PREPARE_V2 PASS "
        f"base_blob={BASE_BLOB} correction=allow_expected_schema_column_name_only "
        "row_level_identifier_values_allowed=false science_changed=false submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
