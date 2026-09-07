#!/usr/bin/env python3
"""Validate E02 current outputs and print only aggregate-safe diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260907-cmi-flu-strategy-e02-task11-structured-logfc-001"
SCIENCE_COMMIT = "0d399f89f35145e99fafd10d3061d9295cd749bb"
E02_BLOB = "e6aaba7450a527de1e75ad6e7c0d9ef7e0031d7e"
E02_V2_BLOB = "afdfe373a64444ef10dffcc6f7d9d65a8dd7b27c"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
BANNED = ('"participant_id"', '"subject_group"', '"row_index"', '"oof_predictions"', '"challenge_predictions"', '"predictions"')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    return p.parse_args()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_or_none(value) -> bool:
    if value is None:
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> int:
    args = parse_args()
    root = args.input_dir.expanduser().resolve()
    files = sorted(p.name for p in root.iterdir() if p.is_file())
    if files != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E02 safe-output file set mismatch")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E02 aggregate privacy contract failed: {name}")
    bridge = json.loads(texts["bridge-result.json"])
    metrics = json.loads(texts["metrics.json"])
    for key, value in {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "config_blob_sha": CONFIG_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }.items():
        if bridge.get(key) != value:
            raise SystemExit(f"E02 bridge contract mismatch: {key}")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json") or bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E02 output hash mismatch")
    if metrics.get("experiment") != "strategy_v2_e02_task11_structured_logfc":
        raise SystemExit("E02 experiment identity mismatch")
    if metrics.get("comparison_contract") != "paired_subject_purged_v2" or metrics.get("task") != "Task1.1":
        raise SystemExit("E02 evaluation contract mismatch")
    cohort = metrics.get("cohort") or {}
    if cohort.get("train_rows") != 127 or cohort.get("train_studies") != 4 or cohort.get("challenge_rows") != 40:
        raise SystemExit("E02 cohort contract mismatch")
    fixed = metrics.get("fixed_conditions") or {}
    if fixed.get("b21_model") != "pls_2" or fixed.get("anchor_residual_model") != "pls_1" or fixed.get("anchor_residual_lambda") != 0.25 or fixed.get("structured_ridge_alpha") != 10.0:
        raise SystemExit("E02 fixed-condition mismatch")
    if metrics.get("competition_submission_attempted") is not False or metrics.get("leaderboard_used_for_selection") is not False:
        raise SystemExit("E02 selection/submission contract mismatch")
    if metrics.get("contains_participant_identifiers") is not False or metrics.get("contains_row_level_predictions") is not False:
        raise SystemExit("E02 privacy flag mismatch")

    conditions = metrics.get("conditions") or {}
    for key in ("anchor", "b21", "anchor_residual", "structured_ridge"):
        payload = conditions.get(key) or {}
        m = payload.get("metrics") or {}
        if m.get("rows") != 127:
            raise SystemExit(f"E02 condition rows mismatch: {key}")
        strict = m.get("study_equal_weight_spearman_mean_strict")
        pooled = (m.get("pooled_within_study_rank_spearman") or {}).get("value")
        if not finite_or_none(strict) or not finite_or_none(pooled):
            raise SystemExit(f"E02 non-finite aggregate metric: {key}")
        print(f"Task1.1 condition={key} strict={strict} pooled_within_study_rank={pooled} undefined={m.get('undefined_fold_count')} constant={m.get('constant_fold_count')}")

    comparisons = metrics.get("comparisons") or {}
    for key in ("structured_ridge_vs_anchor", "structured_ridge_vs_b21", "anchor_residual_vs_anchor", "b21_vs_anchor"):
        comp = comparisons.get(key) or {}
        delta = comp.get("study_mean_delta_strict")
        if not finite_or_none(delta):
            raise SystemExit(f"E02 non-finite paired delta: {key}")
        print(f"Task1.1 comparison={key} delta={delta} candidate_heuristic={comp.get('passes_candidate_delta_heuristic')} all_folds_valid={comp.get('all_folds_valid')} large_study_review={len(comp.get('large_studies_requiring_review') or [])}")

    controls = metrics.get("negative_controls") or {}
    availability = (controls.get("availability_only") or {}).get("metrics") or {}
    shuffle = (controls.get("structured_subject_block_label_shuffle") or {}).get("metrics") or {}
    print(f"Task1.1 availability_strict={availability.get('study_equal_weight_spearman_mean_strict')} shuffle_strict={shuffle.get('study_equal_weight_spearman_mean_strict')}")
    repeat = metrics.get("repeat_baseline_extension") or {}
    print(f"Task1.1 repeat_status={(repeat.get('audit') or {}).get('status')} repeat_executed={repeat.get('condition_executed')} repeat_candidate={repeat.get('candidate')}")
    print(f"CMI_FLU_E02_RESULT PASS aggregate_only=true submission=false science_commit={SCIENCE_COMMIT} e02_blob={E02_BLOB} e02_v2_blob={E02_V2_BLOB} md5_verified={bridge.get('md5_verified_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
