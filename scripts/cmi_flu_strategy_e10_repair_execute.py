#!/usr/bin/env python3
"""One-shot private CPU executor for repaired CMI-Flu E10 request 002."""
from __future__ import annotations

import cmi_flu_strategy_e10_execute as prior

REQUEST_ID = "20260909-cmi-flu-strategy-e10-pooled-domain-correction-002"
TARGET = "renta0426/cmi-flu-e10-pooled-domain-correction-20260909-002"
TITLE = "CMI Flu E10 Pooled Domain Correction 20260909 002"

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260909-cmi-flu-strategy-e10-pooled-domain-correction-001",
    "TARGET": "renta0426/cmi-flu-e10-pooled-domain-correction-20260909-001",
    "TITLE": "CMI Flu E10 Pooled Domain Correction 20260909 001",
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(prior, key, None) != value:
        raise SystemExit(f"E10 repair executor ancestry changed:{key}")

base = prior.base
watcher = prior.prior.prior.prior
if base.EXPECTED_VERSION != 1:
    raise SystemExit("E10 repair executor version contract changed")
if "exact_metadata_eventually" not in base.kernel_meta.__code__.co_names:
    raise SystemExit("E10 repair executor lacks bounded exact metadata reconciliation")
if base.wait is not watcher.wait:
    raise SystemExit("E10 repair inherited watcher identity changed")

# E10-001 already installed the proven v1 E06a watcher into `base`. Retarget
# every module whose globals that watcher can resolve; do not replace `wait`.
prior.REQUEST_ID = REQUEST_ID
prior.TARGET = TARGET
prior.TITLE = TITLE
prior.prior.REQUEST_ID = REQUEST_ID
prior.prior.TARGET = TARGET
prior.prior.TITLE = TITLE
prior.prior.prior.REQUEST_ID = REQUEST_ID
prior.prior.prior.TARGET = TARGET
prior.prior.prior.TITLE = TITLE
watcher.REQUEST_ID = REQUEST_ID
watcher.TARGET = TARGET
watcher.TITLE = TITLE
base.REQUEST_ID = REQUEST_ID
base.TARGET = TARGET
base.TARGET_SLUG = TARGET.split("/", 1)[1]
base.TITLE = TITLE


def prewrite_guard(api):
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != base.TARGET_SLUG or TITLE != "CMI Flu E10 Pooled Domain Correction 20260909 002":
        raise RuntimeError("E10 repair target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=base.TARGET_SLUG, page_size=20) or []
    except Exception as error:
        raise RuntimeError("E10 repair duplicate sentinel unavailable") from error
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E10 repair duplicate sentinel found exact target; write refused")


base.prewrite_guard = prewrite_guard


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
