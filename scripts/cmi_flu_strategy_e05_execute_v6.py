#!/usr/bin/env python3
"""One-shot E05 007 executor on a fresh target/version."""
from __future__ import annotations

import cmi_flu_strategy_e05_execute_v5 as prior

REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-007"
TARGET = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-007"
TITLE = "CMI Flu E05 HAI Donor Strain 20260907 007"

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260907-cmi-flu-strategy-e05-hai-donor-strain-006",
    "TARGET": "renta0426/cmi-flu-e05-hai-donor-strain-20260907-006",
    "TITLE": "CMI Flu E05 HAI Donor Strain 20260907 006",
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(prior, key, None) != value:
        raise SystemExit(f"E05 v5 executor contract changed:{key}")
base = prior.base
if base.EXPECTED_VERSION != 1:
    raise SystemExit("E05 v5 executor version contract changed")
if "exact_metadata_eventually" not in base.kernel_meta.__code__.co_names:
    raise SystemExit("E05 reusable executor lacks bounded exact metadata reconciliation")

base.REQUEST_ID = REQUEST_ID
base.TARGET = TARGET
base.TARGET_SLUG = TARGET.split("/", 1)[1]
base.TITLE = TITLE


def prewrite_guard(api):
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != base.TARGET_SLUG or TITLE != "CMI Flu E05 HAI Donor Strain 20260907 007":
        raise RuntimeError("E05 v6 target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=base.TARGET_SLUG, page_size=20) or []
    except Exception as error:
        raise RuntimeError("E05 v6 duplicate sentinel unavailable") from error
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E05 v6 duplicate sentinel found exact target; write refused")


base.prewrite_guard = prewrite_guard


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
