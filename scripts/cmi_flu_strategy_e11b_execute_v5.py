#!/usr/bin/env python3
"""E11b 005 executor: preserve 004 transport/science; replace only the prewrite absence guard."""
from __future__ import annotations

from kaggle.api.kaggle_api_extended import KaggleApi

import cmi_flu_strategy_e11b_execute_v4 as v4

REQUEST_ID = "20260910-cmi-flu-strategy-e11b-tabpfn3-005"
TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-005"
TARGET_SLUG = TARGET.split("/", 1)[1]
TITLE = "CMI Flu E11b TabPFN3 20260910 005"
FAILED_003_TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-003"
FAILED_004_TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-004"

if v4.REQUEST_ID != "20260910-cmi-flu-strategy-e11b-tabpfn3-004":
    raise SystemExit("E11b executor-v5 request ancestry changed")
if v4.TARGET != "renta0426/cmi-flu-e11b-tabpfn3-20260910-004":
    raise SystemExit("E11b executor-v5 target ancestry changed")
if v4.TITLE != "CMI Flu E11b TabPFN3 20260910 004":
    raise SystemExit("E11b executor-v5 title ancestry changed")
if v4.MAX_RUNTIME_BYTES != 900_000:
    raise SystemExit("E11b executor-v5 runtime budget changed")

# Retarget the reviewed 004 transport functions. Their global lookups remain dynamic.
v4.REQUEST_ID = REQUEST_ID
v4.TARGET = TARGET
v4.TARGET_SLUG = TARGET_SLUG
v4.TITLE = TITLE
v4.prior.REQUEST_ID = REQUEST_ID
v4.prior.TARGET = TARGET
v4.prior.TARGET_SLUG = TARGET_SLUG
v4.prior.TITLE = TITLE


def _require_exact_absent_by_list(api: KaggleApi, target: str, label: str) -> None:
    owner, slug = target.split("/", 1)
    discovered = api.kernels_list(user=owner, search=slug, page_size=20) or []
    exact = [item for item in discovered if str(getattr(item, "ref", "")) == target]
    if exact:
        raise RuntimeError(f"E11b 005 exact target exists; write refused:{label}")
    print(
        "CMI_FLU_E11B_EXACT_ABSENT "
        f"label={label} confirmed=true method=kernels_list_exact_ref"
    )


def prewrite_guard(api: KaggleApi) -> None:
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != TARGET_SLUG or TITLE != "CMI Flu E11b TabPFN3 20260910 005":
        raise RuntimeError("E11b 005 target identity contract changed")
    _require_exact_absent_by_list(api, FAILED_003_TARGET, "failed_003")
    _require_exact_absent_by_list(api, FAILED_004_TARGET, "failed_004")
    _require_exact_absent_by_list(api, TARGET, "fresh_005")


# Preserve model-access preflight and SaveKernel implementation from 004; replace only guard.
v4.prior.model_access_preflight = v4.model_access_preflight
v4.prior.prewrite_guard = prewrite_guard
v4.prior.push = v4.push


def main() -> int:
    return v4.prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
