#!/usr/bin/env python3
"""One-shot private CPU executor for fresh E06b Task2.3 two-head target."""
from __future__ import annotations

import cmi_flu_strategy_e06a_execute_v3 as prior

REQUEST_ID = "20260908-cmi-flu-strategy-e06b-task23-two-head-001"
TARGET = "renta0426/cmi-flu-e06b-task23-two-head-20260908-001"
TITLE = "CMI Flu E06b Task23 Two Head 20260908 001"

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260908-cmi-flu-strategy-e06a-task23-calibration-003",
    "TARGET": "renta0426/cmi-flu-e06a-task23-calibration-20260908-003",
    "TITLE": "CMI Flu E06a Task23 Calibration 20260908 003",
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(prior, key, None) != value:
        raise SystemExit(f"E06a v3 executor contract changed:{key}")
if prior.prior.prior.POLL_SECONDS != 120 or prior.prior.prior.MAX_POLLS != 65:
    raise SystemExit("E06b watcher horizon contract changed")
base = prior.base
if base.EXPECTED_VERSION != 1:
    raise SystemExit("E06b executor version contract changed")
if "exact_metadata_eventually" not in base.kernel_meta.__code__.co_names:
    raise SystemExit("E06b executor lacks bounded exact metadata reconciliation")

# The inherited watcher lives in the E06a v1 module and resolves its globals at
# call time. Align every layer to the fresh E06b identity.
prior.REQUEST_ID = REQUEST_ID
prior.TARGET = TARGET
prior.TITLE = TITLE
prior.prior.REQUEST_ID = REQUEST_ID
prior.prior.TARGET = TARGET
prior.prior.TITLE = TITLE
prior.prior.prior.REQUEST_ID = REQUEST_ID
prior.prior.prior.TARGET = TARGET
prior.prior.prior.TITLE = TITLE
base.REQUEST_ID = REQUEST_ID
base.TARGET = TARGET
base.TARGET_SLUG = TARGET.split("/", 1)[1]
base.TITLE = TITLE


def prewrite_guard(api):
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != base.TARGET_SLUG or TITLE != "CMI Flu E06b Task23 Two Head 20260908 001":
        raise RuntimeError("E06b target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=base.TARGET_SLUG, page_size=20) or []
    except Exception as error:
        raise RuntimeError("E06b duplicate sentinel unavailable") from error
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E06b duplicate sentinel found exact target; write refused")


base.prewrite_guard = prewrite_guard
base.wait = prior.prior.prior.wait


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
