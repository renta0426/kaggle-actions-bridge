#!/usr/bin/env python3
"""One-shot E06a 003 executor on a fresh target after validator repair."""
from __future__ import annotations

import cmi_flu_strategy_e06a_execute_v2 as prior

REQUEST_ID = "20260908-cmi-flu-strategy-e06a-task23-calibration-003"
TARGET = "renta0426/cmi-flu-e06a-task23-calibration-20260908-003"
TITLE = "CMI Flu E06a Task23 Calibration 20260908 003"

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260908-cmi-flu-strategy-e06a-task23-calibration-002",
    "TARGET": "renta0426/cmi-flu-e06a-task23-calibration-20260908-002",
    "TITLE": "CMI Flu E06a Task23 Calibration 20260908 002",
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(prior, key, None) != value:
        raise SystemExit(f"E06a v2 executor contract changed:{key}")
if prior.prior.POLL_SECONDS != 120 or prior.prior.MAX_POLLS != 65:
    raise SystemExit("E06a watcher horizon contract changed")
base = prior.base
if base.EXPECTED_VERSION != 1:
    raise SystemExit("E06a v3 executor version contract changed")
if "exact_metadata_eventually" not in base.kernel_meta.__code__.co_names:
    raise SystemExit("E06a v3 executor lacks bounded exact metadata reconciliation")

# The watcher function is defined in the v1 E06a executor and resolves its
# module globals at call time. Keep every inherited identity aligned to 003.
prior.REQUEST_ID = REQUEST_ID
prior.TARGET = TARGET
prior.TITLE = TITLE
prior.prior.REQUEST_ID = REQUEST_ID
prior.prior.TARGET = TARGET
prior.prior.TITLE = TITLE
base.REQUEST_ID = REQUEST_ID
base.TARGET = TARGET
base.TARGET_SLUG = TARGET.split("/", 1)[1]
base.TITLE = TITLE


def prewrite_guard(api):
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != base.TARGET_SLUG or TITLE != "CMI Flu E06a Task23 Calibration 20260908 003":
        raise RuntimeError("E06a v3 target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=base.TARGET_SLUG, page_size=20) or []
    except Exception as error:
        raise RuntimeError("E06a v3 duplicate sentinel unavailable") from error
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E06a v3 duplicate sentinel found exact target; write refused")


base.prewrite_guard = prewrite_guard
base.wait = prior.prior.wait


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
