#!/usr/bin/env python3
"""One-shot E06a CPU executor with polling horizon aligned to the request timeout."""
from __future__ import annotations

import time

import cmi_flu_strategy_e05_execute_v6 as prior

REQUEST_ID = "20260908-cmi-flu-strategy-e06a-task23-calibration-001"
TARGET = "renta0426/cmi-flu-e06a-task23-calibration-20260908-001"
TITLE = "CMI Flu E06a Task23 Calibration 20260908 001"
POLL_SECONDS = 120
MAX_POLLS = 65  # 130 minutes; request hard runtime is 120 minutes.

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260907-cmi-flu-strategy-e05-hai-donor-strain-007",
    "TARGET": "renta0426/cmi-flu-e05-hai-donor-strain-20260907-007",
    "TITLE": "CMI Flu E05 HAI Donor Strain 20260907 007",
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(prior, key, None) != value:
        raise SystemExit(f"E05 v6 executor contract changed:{key}")
base = prior.base
if base.EXPECTED_VERSION != 1:
    raise SystemExit("E06a executor version contract changed")
if "exact_metadata_eventually" not in base.kernel_meta.__code__.co_names:
    raise SystemExit("E06a executor lacks bounded exact metadata reconciliation")

base.REQUEST_ID = REQUEST_ID
base.TARGET = TARGET
base.TARGET_SLUG = TARGET.split("/", 1)[1]
base.TITLE = TITLE


def prewrite_guard(api):
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != base.TARGET_SLUG or TITLE != "CMI Flu E06a Task23 Calibration 20260908 001":
        raise RuntimeError("E06a target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=base.TARGET_SLUG, page_size=20) or []
    except Exception as error:
        raise RuntimeError("E06a duplicate sentinel unavailable") from error
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E06a duplicate sentinel found exact target; write refused")


def wait(api):
    """Watch only; never retry the write. Polling exhaustion is not remote failure."""
    for _ in range(MAX_POLLS):
        direct = base.kernel_meta(api, TARGET)
        if int(getattr(direct, "current_version_number", 0) or 0) != base.EXPECTED_VERSION:
            raise RuntimeError("E06a version changed during execution")
        status = str(getattr(api.kernels_status(TARGET), "status", "")).upper()
        if "COMPLETE" in status:
            return status
        if any(token in status for token in ("ERROR", "CANCEL", "FAIL")):
            raise RuntimeError("E06a remote failed")
        if not any(token in status for token in ("RUNNING", "QUEUED", "PENDING")):
            raise RuntimeError(f"unknown E06a remote status:{status}")
        time.sleep(POLL_SECONDS)
    raise RuntimeError(
        "E06a watcher expired while remote state remained unresolved; "
        "no write retry is permitted and current-version recovery is required"
    )


base.prewrite_guard = prewrite_guard
base.wait = wait


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
