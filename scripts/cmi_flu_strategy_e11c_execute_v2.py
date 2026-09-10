#!/usr/bin/env python3
"""Manual E11c-002 executor; retarget the audited E11c-001 transport only.

E11c-001 stopped at the CPU admission guard before SaveKernel. This module
changes request/target/title identity to a fresh 002 target and rebinds only
the identity guard whose 001 title was intentionally literal. Frozen science,
model transport, source-attachment validation, live Rules preflight, CPU
admission, one-shot SaveKernel, watcher, and aggregate reader are unchanged.
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


def prewrite_guard(api) -> None:
    """Same E11c guard as 001, with only the literal title/target retargeted."""
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != TARGET_SLUG:
        raise RuntimeError("E11c-002 target identity contract changed")
    if TITLE != "CMI Flu E11c Robust TabPFN 20260910 002":
        raise RuntimeError("E11c-002 title identity contract changed")
    if TARGET == base.CONSUMED_E11B_TARGET or TARGET_SLUG.endswith("tabpfn3-20260910-005"):
        raise RuntimeError("E11c-002 refused consumed E11b-005 identity")
    base._require_fresh_target_by_list(api)


# E11c-001 wired its guard into the inherited E11b orchestration module.
# Rebind that exact hook and nothing else. CPU admission remains in prior.main.
base.prewrite_guard = prewrite_guard
base.prior.prewrite_guard = prewrite_guard


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
