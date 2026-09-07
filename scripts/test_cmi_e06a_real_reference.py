#!/usr/bin/env python3
"""Verify E06a uses the exact organizer sequence reference through its native-schema adapter."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import pandas as pd

EXPECTED_SHA256 = "63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887"
EXPECTED_COLUMNS = ("Virus", "Sequence", "Status_of_sequence")


def load_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("e06a_real_reference_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    args = p.parse_args()
    sequence_path = args.reference_dir.resolve() / "strain_sequences.csv"
    assert sequence_path.is_file()
    assert hashlib.sha256(sequence_path.read_bytes()).hexdigest() == EXPECTED_SHA256
    reference = pd.read_csv(sequence_path)
    assert tuple(map(str, reference.columns)) == EXPECTED_COLUMNS

    runtime = load_runtime(args.runtime.resolve())
    with tempfile.TemporaryDirectory(prefix="e06a-real-reference-") as tmp:
        package_path = Path(tmp) / "cmi_flu_bundle.zip"
        package_path.write_bytes(runtime.package_bytes())
        sys.path.insert(0, str(package_path))
        try:
            run, hai, e06 = runtime.load_e06_module()
            assert callable(run) and callable(e06.run_strategy_e06a)
            adapter = sys.modules["cmi_flu.hai_transfer_v2"]
            normalized = adapter.normalize_sequence_reference_schema(reference)
            assert tuple(normalized.columns) == (
                "virus_strain",
                "sequence",
                "sequence_status",
            )
            assert hai.build_sequence_lookup is adapter.build_sequence_lookup
            lookup = hai.build_sequence_lookup(reference)
            assert isinstance(lookup, dict) and len(lookup) == 352
            assert all(entry.get("sequence") for entry in lookup.values())
        finally:
            sys.path.remove(str(package_path))
    print(
        "CMI_FLU_E06A_REAL_REFERENCE_PASS hash_verified=true native_header=true "
        "lookup_entries=352 row_data_emitted=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
