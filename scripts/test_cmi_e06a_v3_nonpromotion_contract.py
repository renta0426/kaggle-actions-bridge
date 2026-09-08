#!/usr/bin/env python3
"""Regression: candidate rank failure must serialize as non-promotion, not abort E06a."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path


def load_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("e06a_v3_nonpromotion_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    args = p.parse_args()
    runtime = load_runtime(args.runtime.resolve())
    original = json.loads(args.metrics.read_text(encoding="utf-8"))

    result = copy.deepcopy(original)
    condition = result["conditions"]["b21_reference"]
    candidate = condition["historical_nested_study_out"]["calibrated"]["positive_affine"]
    candidate["rank_preservation"]["preserved_within_tolerance"] = False
    candidate["rank_preservation"]["max_absolute_percentile_difference"] = 0.02564102564102564
    decision = condition["promotion"]["positive_affine"]
    decision["checks"]["held_study_rank_preserved"] = False
    decision["passed"] = False
    if result.get("selected_promoted_condition") == "b21_reference__positive_affine":
        result["selected_promoted_condition"] = None

    # The observed production class is a valid aggregate negative candidate.
    runtime.validate_result(result)

    inconsistent = copy.deepcopy(result)
    inconsistent["conditions"]["b21_reference"]["promotion"]["positive_affine"]["passed"] = True
    try:
        runtime.validate_result(inconsistent)
    except Exception:
        pass
    else:
        raise AssertionError("E06a v3 accepted promotion=true with held-rank preservation=false")

    print(
        "CMI_FLU_E06A_V3_NONPROMOTION_CONTRACT_PASS "
        "held_rank_false_valid=true promotion_false_required=true inconsistent_promotion_rejected=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
