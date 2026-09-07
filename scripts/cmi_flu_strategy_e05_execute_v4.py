#!/usr/bin/env python3
"""One-shot E05 004 executor with fresh target identity."""
from __future__ import annotations

import cmi_flu_strategy_e05_execute_v3 as prior

REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-004"
TARGET = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-004"
TITLE = "CMI Flu E05 HAI Donor Strain 20260907 004"

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260907-cmi-flu-strategy-e05-hai-donor-strain-003",
    "TARGET": "renta0426/cmi-flu-e05-hai-donor-strain-20260907-003",
    "TITLE": "CMI Flu E05 HAI Donor Strain 20260907 003",
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(prior, key, None) != value:
        raise SystemExit(f"E05 v3 executor contract changed:{key}")
if prior.base.EXPECTED_VERSION != 1:
    raise SystemExit("E05 v3 executor version contract changed")

prior.base.REQUEST_ID = REQUEST_ID
prior.base.TARGET = TARGET
prior.base.TARGET_SLUG = TARGET.split("/", 1)[1]
prior.base.TITLE = TITLE


def prewrite_guard(api):
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != prior.base.TARGET_SLUG or TITLE != "CMI Flu E05 HAI Donor Strain 20260907 004":
        raise RuntimeError("E05 v4 target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=prior.base.TARGET_SLUG, page_size=20) or []
    except Exception as error:
        raise RuntimeError("E05 v4 duplicate sentinel unavailable") from error
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E05 v4 duplicate sentinel found exact target; write refused")


prior.base.prewrite_guard = prewrite_guard


def main() -> int:
    return prior.base.main()


if __name__ == "__main__":
    raise SystemExit(main())
