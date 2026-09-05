#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import pathlib
import py_compile

B64_PLACEHOLDER = "__CMI_FLU_BACKBONE_B64__"
SHA_PLACEHOLDER = "__CMI_FLU_BACKBONE_SHA256__"
EXPECTED_COLUMNS = (
    "participant_id",
    "Task1.1",
    "Task1.2",
    "Task1.3",
    "Task1.4",
    "Task2.1",
    "Task2.2",
    "Task2.3",
)


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one placeholder, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=pathlib.Path, required=True)
    parser.add_argument("--backbone", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    template = args.template.expanduser().resolve().read_text(encoding="utf-8")
    data = args.backbone.expanduser().resolve().read_bytes()
    if not data or len(data) > 262144:
        raise SystemExit("B2.1 backbone CSV outside byte budget")
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SystemExit("B2.1 backbone CSV is not UTF-8") from error
    rows = list(csv.reader(io.StringIO(decoded)))
    if not rows or tuple(rows[0]) != EXPECTED_COLUMNS:
        raise SystemExit("B2.1 backbone CSV header mismatch")
    if len(rows) != 41:
        raise SystemExit(f"B2.1 backbone CSV expected 40 data rows, found {len(rows) - 1}")
    if any(len(row) != len(EXPECTED_COLUMNS) for row in rows):
        raise SystemExit("B2.1 backbone CSV has malformed row width")
    ids = [row[0] for row in rows[1:]]
    if any(not value for value in ids) or len(set(ids)) != 40:
        raise SystemExit("B2.1 backbone participant IDs are missing or duplicated")

    digest = hashlib.sha256(data).hexdigest()
    encoded = base64.b64encode(data).decode("ascii")
    bound = replace_once(template, B64_PLACEHOLDER, encoded, label="backbone base64")
    bound = replace_once(bound, SHA_PLACEHOLDER, digest, label="backbone SHA-256")
    if B64_PLACEHOLDER in bound or SHA_PLACEHOLDER in bound:
        raise SystemExit("controlled-probe runtime retains an unbound backbone placeholder")
    if "competition_submit" in bound or "kaggle competitions submit" in bound:
        raise SystemExit("bound controlled-probe runtime contains a submission path")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    compile(bound, str(output), "exec")
    output.write_text(bound, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)
    print(
        "CMI_FLU_PUBLIC_PROBES_BACKBONE_BIND PASS "
        f"rows=40 bytes={len(data)} sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
