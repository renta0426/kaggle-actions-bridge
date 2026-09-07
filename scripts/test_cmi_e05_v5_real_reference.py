#!/usr/bin/env python3
"""Execute the E05 006 sequence adapter against the exact locked organizer file."""
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
    spec = importlib.util.spec_from_file_location("e05_v5_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    a = p.parse_args()
    sequence_path = a.reference_dir.resolve() / "strain_sequences.csv"
    assert sequence_path.is_file()
    assert sha256(sequence_path) == EXPECTED_SHA256
    reference = pd.read_csv(sequence_path)
    assert tuple(map(str, reference.columns)) == EXPECTED_COLUMNS

    runtime = load_runtime(a.runtime.resolve())
    with tempfile.TemporaryDirectory(prefix="e05-v5-real-ref-") as tmp:
        package = Path(tmp) / "cmi_flu_bundle.zip"
        package.write_bytes(runtime.package_bytes())
        sys.path.insert(0, str(package))
        try:
            run, hai = runtime.load_e05_module()
            assert callable(run)
            adapter = sys.modules["cmi_flu.hai_transfer_v2"]
            e05v3 = sys.modules["cmi_flu.strategy_e05_v3"]
            assert hai.build_sequence_lookup is adapter.build_sequence_lookup
            assert callable(e05v3.run_strategy_e05)
            normalized = adapter.normalize_sequence_reference_schema(reference)
            assert tuple(normalized.columns) == ("virus_strain", "sequence", "sequence_status")
            lookup = hai.build_sequence_lookup(reference)
            assert isinstance(lookup, dict) and lookup
            assert all(entry.get("sequence") for entry in lookup.values())
            assert all("complete" in entry and "sequence_status" in entry for entry in lookup.values())
        finally:
            sys.path.remove(str(package))
    print(
        "CMI_FLU_E05_V5_REAL_REFERENCE_PASS "
        f"hash_verified=true native_header=true lookup_entries={len(lookup)} "
        "row_data_emitted=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
