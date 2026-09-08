#!/usr/bin/env python3
"""One-shot repaired E06a 002 executor on a fresh target/version."""
from __future__ import annotations

import cmi_flu_strategy_e06a_execute as prior

REQUEST_ID = "20260908-cmi-flu-strategy-e06a-task23-calibration-002"
TARGET = "renta0426/cmi-flu-e06a-task23-calibration-20260908-002"
TITLE = "CMI Flu E06a Task23 Calibration 20260908 002"

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260908-cmi-flu-strategy-e06a-task23-calibration-001",
    "TARGET": "renta0426/cmi-flu-e06a-task23-calibration-20260908-001",
    "TITLE": "CMI Flu E06a Task23 Calibration 20260908 001",
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(prior, key, None) != value:
        raise SystemExit(f"E06a v1 executor contract changed:{key}")
if prior.POLL_SECONDS != 120 or prior.MAX_POLLS != 65:
    raise SystemExit("E06a watcher horizon contract changed")
base = prior.base
if base.EXPECTED_VERSION != 1:
    raise SystemExit("E06a v2 executor version contract changed")
if "exact_metadata_eventually" not in base.kernel_meta.__code__.co_names:
    raise SystemExit("E06a v2 executor lacks bounded exact metadata reconciliation")

# The inherited watcher reads these globals from the prior module at call time.
prior.REQUEST_ID = REQUEST_ID
prior.TARGET = TARGET
prior.TITLE = TITLE
base.REQUEST_ID = REQUEST_ID
base.TARGET = TARGET
base.TARGET_SLUG = TARGET.split("/", 1)[1]
base.TITLE = TITLE


def prewrite_guard(api):
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != base.TARGET_SLUG or TITLE != "CMI Flu E06a Task23 Calibration 20260908 002":
        raise RuntimeError("E06a v2 target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=base.TARGET_SLUG, page_size=20) or []
    except Exception as error:
        raise RuntimeError("E06a v2 duplicate sentinel unavailable") from error
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E06a v2 duplicate sentinel found exact target; write refused")


base.prewrite_guard = prewrite_guard
base.wait = prior.wait


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
