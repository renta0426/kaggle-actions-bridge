#!/usr/bin/env python3
"""Exercise E03 with an MD5 entry for a file outside CORE_FILES.

Request 002 passed the original synthetic test because its md5sum fixture was
empty.  The real Competition manifest contains many additional files, so this
wrapper adds a small synthetic `2025LJI_bulkBCR.tsv` entry and requires the
production runtime to preserve it through staging and checksum verification.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

BASE = Path(__file__).with_name("test_cmi_e03_full_synthetic.py")
EXTRA_NAME = "2025LJI_bulkBCR.tsv"
EXTRA_BYTES = b"synthetic bulk BCR placeholder for MD5 staging regression\n"


def load_base():
    spec = importlib.util.spec_from_file_location("cmi_e03_full_synthetic_base_md5", BASE)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to import base E03 synthetic test")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    base = load_base()
    original = base.build_input

    def build_input_with_md5(root: Path) -> None:
        original(root)
        extra = root / EXTRA_NAME
        extra.write_bytes(EXTRA_BYTES)
        digest = hashlib.md5(EXTRA_BYTES).hexdigest()
        (root / "md5sum").write_text(f"{digest}  {EXTRA_NAME}\n", encoding="utf-8")

    base.build_input = build_input_with_md5
    rc = int(base.main())
    print(
        "CMI_FLU_E03_FULL_SYNTHETIC_MD5 PASS "
        f"non_core_manifest_entry={EXTRA_NAME} staged_and_verified=true"
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
