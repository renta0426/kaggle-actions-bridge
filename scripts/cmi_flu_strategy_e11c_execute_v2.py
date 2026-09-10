#!/usr/bin/env python3
"""Manual E11c-002 executor; retargets the audited E11c-001 transport only.

E11c-001 stopped at the CPU admission guard before SaveKernel.  This module
changes only request/target/title identity to a fresh 002 target.  The frozen
science runtime, model transport, source-attachment validator, live Rules
preflight, CPU admission, one-shot SaveKernel, watcher, and aggregate reader
remain the E11c-001 implementation.
"""
from __future__ import annotations

import cmi_flu_strategy_e11c_execute as base

REQUEST_ID = "20260910-cmi-flu-strategy-e11c-robust-tabpfn-002"
TARGET = "renta0426/cmi-flu-e11c-robust-tabpfn-20260910-002"
TARGET_SLUG = TARGET.split("/", 1)[1]
TITLE = "CMI Flu E11c Robust TabPFN 20260910 002"

PARENT_REQUEST_ID = "20260910-cmi-flu-strategy-e11c-robust-tabpfn-001"
PARENT_TARGET = "renta0426/cmi-flu-e11c-robust-tabpfn-20260910-001"
PARENT_ACTIONS_RUN = 34474957373
PARENT_ACTIONS_JOB = 102863653370

if base.REQUEST_ID != PARENT_REQUEST_ID:
    raise SystemExit("E11c-002 executor parent request changed")
if base.TARGET != PARENT_TARGET:
    raise SystemExit("E11c-002 executor parent target changed")
if base.TITLE != "CMI Flu E11c Robust TabPFN 20260910 001":
    raise SystemExit("E11c-002 executor parent title changed")
if base.CONSUMED_E11B_TARGET != "renta0426/cmi-flu-e11b-tabpfn3-20260910-005":
    raise SystemExit("E11c-002 consumed E11b target changed")
if base.EXPECTED_VERSION != 1:
    raise SystemExit("E11c-002 expected version changed")

# Retarget the module globals resolved by the already-reviewed E11c functions.
base.REQUEST_ID = REQUEST_ID
base.TARGET = TARGET
base.TARGET_SLUG = TARGET_SLUG
base.TITLE = TITLE

base.v4.REQUEST_ID = REQUEST_ID
base.v4.TARGET = TARGET
base.v4.TARGET_SLUG = TARGET_SLUG
base.v4.TITLE = TITLE

base.prior.REQUEST_ID = REQUEST_ID
base.prior.TARGET = TARGET
base.prior.TARGET_SLUG = TARGET_SLUG
base.prior.TITLE = TITLE

# Do not replace any callable.  In particular, the existing prewrite guard
# still performs one live fresh-target check and the inherited main still
# refuses CPU admission when another CPU Notebook is active/unknown.


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
