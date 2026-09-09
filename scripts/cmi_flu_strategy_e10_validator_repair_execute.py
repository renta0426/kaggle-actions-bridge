#!/usr/bin/env python3
"""One-shot private CPU executor for fresh E10 validator-runtime repair 003."""
from __future__ import annotations

import cmi_flu_strategy_e10_repair_execute as prior

REQUEST_ID = "20260910-cmi-flu-strategy-e10-pooled-domain-correction-003"
TARGET = "renta0426/cmi-flu-e10-pooled-domain-correction-20260910-003"
TITLE = "CMI Flu E10 Pooled Domain Correction 20260910 003"

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260909-cmi-flu-strategy-e10-pooled-domain-correction-002",
    "TARGET": "renta0426/cmi-flu-e10-pooled-domain-correction-20260909-002",
    "TITLE": "CMI Flu E10 Pooled Domain Correction 20260909 002",
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(prior, key, None) != value:
        raise SystemExit(f"E10 validator-repair executor ancestry changed:{key}")

base = prior.base
watcher = prior.watcher
if base.EXPECTED_VERSION != 1:
    raise SystemExit("E10 validator-repair executor version contract changed")
if "exact_metadata_eventually" not in base.kernel_meta.__code__.co_names:
    raise SystemExit("E10 validator-repair lacks bounded exact metadata reconciliation")
if base.wait is not watcher.wait:
    raise SystemExit("E10 validator-repair inherited watcher identity changed")

# Retarget every inherited module whose globals the proven watcher/write path can
# resolve. Keep the watcher implementation itself unchanged.
modules = [
    prior,
    prior.prior,
    prior.prior.prior,
    prior.prior.prior.prior,
    prior.prior.prior.prior.prior,
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
    if owner != "renta0426" or slug != base.TARGET_SLUG or TITLE != "CMI Flu E10 Pooled Domain Correction 20260910 003":
        raise RuntimeError("E10 validator-repair target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=base.TARGET_SLUG, page_size=20) or []
    except Exception as error:
        raise RuntimeError("E10 validator-repair duplicate sentinel unavailable") from error
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E10 validator-repair duplicate sentinel found exact target; write refused")


base.prewrite_guard = prewrite_guard


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
