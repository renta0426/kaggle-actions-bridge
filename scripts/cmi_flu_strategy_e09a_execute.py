#!/usr/bin/env python3
"""One-shot private CPU executor for E09a module/teacher applicability audit."""
from __future__ import annotations

import time

import cmi_flu_strategy_e06c_execute as prior

REQUEST_ID = "20260909-cmi-flu-strategy-e09a-module-teacher-audit-001"
TARGET = "renta0426/cmi-flu-e09a-module-teacher-audit-20260909-001"
TITLE = "CMI Flu E09a Module Teacher Audit 20260909 001"
POLL_SECONDS = 120
MAX_POLLS = 80

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260909-cmi-flu-strategy-e06c-d28-calibration-001",
    "TARGET": "renta0426/cmi-flu-e06c-d28-calibration-20260909-001",
    "TITLE": "CMI Flu E06c D28 Calibration 20260909 001",
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(prior, key, None) != value:
        raise SystemExit(f"E06c executor contract changed:{key}")
base = prior.base
if base.EXPECTED_VERSION != 1:
    raise SystemExit("E09a executor version contract changed")
if "exact_metadata_eventually" not in base.kernel_meta.__code__.co_names:
    raise SystemExit("E09a executor lacks bounded exact metadata reconciliation")

# E06c already aligns its E06a ancestry during import. Re-align every inherited
# identity to the fresh E09a target; no compute retry path is introduced.
prior.REQUEST_ID = REQUEST_ID
prior.TARGET = TARGET
prior.TITLE = TITLE
prior.prior.REQUEST_ID = REQUEST_ID
prior.prior.TARGET = TARGET
prior.prior.TITLE = TITLE
prior.prior.prior.REQUEST_ID = REQUEST_ID
prior.prior.prior.TARGET = TARGET
prior.prior.prior.TITLE = TITLE
prior.prior.prior.prior.REQUEST_ID = REQUEST_ID
prior.prior.prior.prior.TARGET = TARGET
prior.prior.prior.prior.TITLE = TITLE
base.REQUEST_ID = REQUEST_ID
base.TARGET = TARGET
base.TARGET_SLUG = TARGET.split("/", 1)[1]
base.TITLE = TITLE


def prewrite_guard(api):
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != base.TARGET_SLUG or TITLE != "CMI Flu E09a Module Teacher Audit 20260909 001":
        raise RuntimeError("E09a target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=base.TARGET_SLUG, page_size=20) or []
    except Exception as error:
        raise RuntimeError("E09a duplicate sentinel unavailable") from error
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E09a duplicate sentinel found exact target; write refused")


def wait(api):
    """Watch the one approved run only; never retry the write."""
    for _ in range(MAX_POLLS):
        direct = base.kernel_meta(api, TARGET)
        if int(getattr(direct, "current_version_number", 0) or 0) != base.EXPECTED_VERSION:
            raise RuntimeError("E09a version changed during execution")
        status = str(getattr(api.kernels_status(TARGET), "status", "")).upper()
        if "COMPLETE" in status:
            return status
        if any(token in status for token in ("ERROR", "CANCEL", "FAIL")):
            raise RuntimeError("E09a remote failed")
        if not any(token in status for token in ("RUNNING", "QUEUED", "PENDING")):
            raise RuntimeError(f"unknown E09a remote status:{status}")
        time.sleep(POLL_SECONDS)
    raise RuntimeError(
        "E09a watcher expired while remote state remained unresolved; "
        "no write retry is permitted and current-version recovery is required"
    )


base.prewrite_guard = prewrite_guard
base.wait = wait


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
