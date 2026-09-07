#!/usr/bin/env python3
"""Print only the schema of hash-verified locked HAI organizer references.

This is intentionally row-free and credential-free. It exists to prevent model
runtimes from assuming a reference-file column contract that CI never checked.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

EXPECTED_SHA256 = {
    "strain_sequences.csv": "63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887",
    "vaccine_strains_per_season.txt": "8f6c7116f37f29df0bb21d6049d82fa28b4e42b2d10ed9394a1ae6f926bd9f35",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.reference_dir.expanduser().resolve()
    for name, expected in EXPECTED_SHA256.items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"locked reference integrity mismatch:{name}")
    columns = pd.read_csv(root / "strain_sequences.csv", nrows=0).columns.tolist()
    if not columns or any(not str(column).strip() for column in columns):
        raise SystemExit("strain sequence reference has invalid header")
    print("CMI_FLU_HAI_REFERENCE_SCHEMA name=strain_sequences.csv columns=" + ",".join(map(str, columns)))
    print("CMI_FLU_HAI_REFERENCE_SCHEMA_PASS row_data_emitted=false hash_verified=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
