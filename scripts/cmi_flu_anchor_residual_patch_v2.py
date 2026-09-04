#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

OLD_STAGE = 'TARGET_STAGE = "phase_a_rank_transfer_task11_task12"'
NEW_STAGE = 'TARGET_STAGE = "phase_a_anchor_residual_task11_task12"'
REQUEST_ID = 'REQUEST_ID = "20260904-cmi-flu-anchor-residual-001"'
SOURCE_COMMIT = 'SOURCE_COMMIT = "23ab4ff53d65eeb8b8e5582f5442081f245f03b3"'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = args.source.read_text(encoding="utf-8")
    if text.count(OLD_STAGE) != 1:
        raise SystemExit(f"anchor-residual stage anchor count={text.count(OLD_STAGE)}")
    if text.count(REQUEST_ID) != 1 or text.count(SOURCE_COMMIT) != 1:
        raise SystemExit("anchor-residual identity was not established before stage fix")
    text = text.replace(OLD_STAGE, NEW_STAGE, 1)
    compile(text, str(args.output), "exec")
    args.output.write_text(text, encoding="utf-8")
    print("CMI_FLU_ANCHOR_RESIDUAL_PATCH_V2 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
