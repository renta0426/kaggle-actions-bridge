#!/usr/bin/env python3
"""One-shot E05 003 executor; reuses the audited 002 transport executor with fresh identity."""
from __future__ import annotations

import cmi_flu_strategy_e05_execute_v2 as base

REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-003"
TARGET = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-003"
TITLE = "CMI Flu E05 HAI Donor Strain 20260907 003"

_EXPECTED_BASE = {
    "REQUEST_ID": "20260907-cmi-flu-strategy-e05-hai-donor-strain-002",
    "TARGET": "renta0426/cmi-flu-e05-hai-donor-strain-20260907-002",
    "TITLE": "CMI Flu E05 HAI Donor Strain 20260907 002",
    "EXPECTED_VERSION": 1,
}
for key, value in _EXPECTED_BASE.items():
    if getattr(base, key, None) != value:
        raise SystemExit(f"E05 v2 executor contract changed:{key}")

base.REQUEST_ID = REQUEST_ID
base.TARGET = TARGET
base.TARGET_SLUG = TARGET.split("/", 1)[1]
base.TITLE = TITLE


def prewrite_guard(api):
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != base.TARGET_SLUG or TITLE != "CMI Flu E05 HAI Donor Strain 20260907 003":
        raise RuntimeError("E05 v3 target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=base.TARGET_SLUG, page_size=20) or []
    except Exception as error:
        raise RuntimeError("E05 v3 duplicate sentinel unavailable") from error
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E05 v3 duplicate sentinel found exact target; write refused")


base.prewrite_guard = prewrite_guard


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
