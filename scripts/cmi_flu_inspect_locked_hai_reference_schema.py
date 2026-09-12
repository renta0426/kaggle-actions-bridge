#!/usr/bin/env python3
"""Validate only the schema of hash-verified locked HAI organizer references.

This is intentionally row-free and credential-free. It exists to prevent model
runtimes from assuming a reference-file column contract that CI never checked.
The validator is stdlib-only so it can run before any modeling environment is
activated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

EXPECTED_SHA256 = {
    "strain_sequences.csv": "63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887",
    "vaccine_strains_per_season.txt": "8f6c7116f37f29df0bb21d6049d82fa28b4e42b2d10ed9394a1ae6f926bd9f35",
}
EXPECTED_STRAIN_SEQUENCE_COLUMNS = ("Virus", "Sequence", "Status_of_sequence")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return tuple(next(reader))
        except StopIteration as exc:
            raise SystemExit("strain sequence reference is empty") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.reference_dir.expanduser().resolve()
    for name, expected in EXPECTED_SHA256.items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"locked reference integrity mismatch:{name}")
    columns = read_csv_header(root / "strain_sequences.csv")
    if columns != EXPECTED_STRAIN_SEQUENCE_COLUMNS:
        raise SystemExit(
            "strain sequence reference header changed: "
            f"expected={EXPECTED_STRAIN_SEQUENCE_COLUMNS}, actual={columns}"
        )
    print("CMI_FLU_HAI_REFERENCE_SCHEMA name=strain_sequences.csv columns=" + ",".join(columns))
    print("CMI_FLU_HAI_REFERENCE_SCHEMA_PASS row_data_emitted=false hash_verified=true exact_header=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
