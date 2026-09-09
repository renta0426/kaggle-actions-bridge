#!/usr/bin/env python3
"""Sanitize the already-completed E10-003 output without another Kaggle compute run."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cmi_flu_strategy_e10_repair_sanitize as prior

REQUEST_ID = "20260910-cmi-flu-strategy-e10-pooled-domain-correction-003"
TARGET = "renta0426/cmi-flu-e10-pooled-domain-correction-20260910-003"

if prior.REQUEST_ID != "20260909-cmi-flu-strategy-e10-pooled-domain-correction-002":
    raise SystemExit("E10 output-recovery sanitizer request ancestry changed")
if prior.TARGET != "renta0426/cmi-flu-e10-pooled-domain-correction-20260909-002":
    raise SystemExit("E10 output-recovery sanitizer target ancestry changed")
if prior.SCIENCE_COMMIT != "3a8728879179115487f5ce0ba4a4a0467e1910bb":
    raise SystemExit("E10 output-recovery sanitizer science ancestry changed")
if prior.E10_BLOB != "638b09860f85689714cb0b354f4f377a405deeac":
    raise SystemExit("E10 output-recovery sanitizer E10 blob ancestry changed")
if prior.E10_SYNTH_BLOB != "f1f33f3909e64e69fb859d98b4163cdb4860d64d":
    raise SystemExit("E10 output-recovery sanitizer synthetic blob ancestry changed")
if prior.MIN_SOURCE_SUBJECTS != 5:
    raise SystemExit("E10 output-recovery sanitizer source-size ancestry changed")

prior.REQUEST_ID = REQUEST_ID
prior.TARGET = TARGET


def normalize_bridge_payload(payload: dict) -> dict:
    """Treat omitted synthetic as the canonical real-run representation.

    The E10 real-run bridge writer predates the synthetic-mode field and omits it,
    while synthetic output explicitly emits synthetic=true. Therefore only an
    absent key or explicit false is accepted for recovery; true is rejected.
    """
    normalized = dict(payload)
    if "synthetic" in normalized and normalized["synthetic"] is not False:
        raise SystemExit("E10 output-recovery refuses synthetic output")
    normalized["synthetic"] = False
    return normalized


def self_test() -> int:
    base = {"request_id": REQUEST_ID}
    assert normalize_bridge_payload(base)["synthetic"] is False
    assert normalize_bridge_payload({**base, "synthetic": False})["synthetic"] is False
    try:
        normalize_bridge_payload({**base, "synthetic": True})
    except SystemExit:
        pass
    else:
        raise SystemExit("E10 output-recovery synthetic=true rejection failed")
    print("CMI_FLU_E10_OUTPUT_RECOVERY_SANITIZER_SELF_TEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.input_dir is None:
        raise SystemExit("--input-dir is required")

    source = args.input_dir.expanduser().resolve()
    files = sorted(path.name for path in source.iterdir() if path.is_file())
    if files != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit(f"E10 output-recovery file set mismatch:{files}")

    bridge = json.loads((source / "bridge-result.json").read_text(encoding="utf-8"))
    # Confirm the observed production representation before compatibility
    # normalization. An unexpected explicit truthy synthetic marker is fatal.
    normalized = normalize_bridge_payload(bridge)

    with tempfile.TemporaryDirectory(prefix="cmi-flu-e10-output-recovery-") as tmp:
        root = Path(tmp)
        shutil.copy2(source / "metrics.json", root / "metrics.json")
        shutil.copy2(source / "summary.md", root / "summary.md")
        (root / "bridge-result.json").write_text(
            json.dumps(normalized, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        old_argv = sys.argv[:]
        try:
            sys.argv = ["cmi_flu_strategy_e10_repair_sanitize.py", "--input-dir", str(root)]
            rc = prior.main()
        finally:
            sys.argv = old_argv
    if rc != 0:
        raise SystemExit(rc)
    print("CMI_FLU_E10_OUTPUT_RECOVERY_SANITIZER_PASS source_bridge_synthetic_missing_or_false=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
