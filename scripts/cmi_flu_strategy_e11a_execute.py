#!/usr/bin/env python3
"""One-shot private CPU executor for CMI-Flu E11a Task1.1 pairwise ranker."""
from __future__ import annotations

import cmi_flu_strategy_e10_validator_repair_execute as prior

REQUEST_ID = "20260910-cmi-flu-strategy-e11a-task11-pairwise-001"
TARGET = "renta0426/cmi-flu-e11a-task11-pairwise-20260910-001"
TITLE = "CMI Flu E11a Task11 Pairwise 20260910 001"

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260910-cmi-flu-strategy-e10-pooled-domain-correction-003",
    "TARGET": "renta0426/cmi-flu-e10-pooled-domain-correction-20260910-003",
    "TITLE": "CMI Flu E10 Pooled Domain Correction 20260910 003",
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(prior, key, None) != value:
        raise SystemExit(f"E11a executor ancestry changed:{key}")

base = prior.base
watcher = prior.watcher
if base.EXPECTED_VERSION != 1:
    raise SystemExit("E11a executor version contract changed")
if "exact_metadata_eventually" not in base.kernel_meta.__code__.co_names:
    raise SystemExit("E11a executor lacks bounded exact metadata reconciliation")
if base.wait is not watcher.wait:
    raise SystemExit("E11a inherited watcher identity changed")

modules = [
    prior,
    prior.prior,
    prior.prior.prior,
    prior.prior.prior.prior,
    prior.prior.prior.prior.prior,
    prior.prior.prior.prior.prior.prior,
    watcher,
]
seen = set()
for module in modules:
    if id(module) in seen:
        continue
    seen.add(id(module))
    if hasattr(module, "REQUEST_ID"):
        module.REQUEST_ID = REQUEST_ID
    if hasattr(module, "TARGET"):
        module.TARGET = TARGET
    if hasattr(module, "TITLE"):
        module.TITLE = TITLE

base.REQUEST_ID = REQUEST_ID
base.TARGET = TARGET
base.TARGET_SLUG = TARGET.split("/", 1)[1]
base.TITLE = TITLE


def prewrite_guard(api):
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != base.TARGET_SLUG or TITLE != "CMI Flu E11a Task11 Pairwise 20260910 001":
        raise RuntimeError("E11a target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=base.TARGET_SLUG, page_size=20) or []
    except Exception as error:
        raise RuntimeError("E11a duplicate sentinel unavailable") from error
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E11a duplicate sentinel found exact target; write refused")


base.prewrite_guard = prewrite_guard


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
