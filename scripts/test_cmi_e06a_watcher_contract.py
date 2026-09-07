#!/usr/bin/env python3
"""Regression for E06a watcher horizon after E05 007 outlived the old 60-minute poll bound."""
from __future__ import annotations

import cmi_flu_strategy_e06a_execute as executor


def main() -> int:
    assert executor.REQUEST_ID == "20260908-cmi-flu-strategy-e06a-task23-calibration-001"
    assert executor.TARGET == "renta0426/cmi-flu-e06a-task23-calibration-20260908-001"
    assert executor.POLL_SECONDS == 120
    assert executor.MAX_POLLS == 65
    horizon_seconds = executor.POLL_SECONDS * executor.MAX_POLLS
    assert horizon_seconds >= 120 * 60
    assert horizon_seconds == 130 * 60
    assert executor.base.wait is executor.wait
    assert executor.base.prewrite_guard is executor.prewrite_guard

    constants = " ".join(
        str(item)
        for item in executor.wait.__code__.co_consts
        if isinstance(item, str)
    )
    assert "no write retry is permitted" in constants
    assert "current-version recovery is required" in constants
    assert "polling bound exceeded" not in constants
    print(
        "CMI_FLU_E06A_WATCHER_CONTRACT_PASS horizon_minutes=130 "
        "hard_runtime_minutes=120 write_retry=false expiry_is_remote_failure=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
