#!/usr/bin/env python3
"""Validate E01 current outputs and print only aggregate-safe diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260907-cmi-flu-strategy-e01-paired-evaluation-001"
SCIENCE_COMMIT = "0b2ecb47eaa09f22450424c9c06dc88cf44bc1fb"
E01_BLOB = "dd27aea0cf97d41bad3cec64819c4c4269d94cbd"
E01_V2_BLOB = "8cc64dc5ab9483d5957cfada18d445188566c56c"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
EXPECTED_INCUMBENT = {
    "Task1.1": "b21_pls_2",
    "Task1.2": "task12_anchor_residual_et_d5_l5_sqrt_lambda0.5",
    "Task1.3": "b21_pls_1",
    "Task2.1": "b21_et_subtype_d3_l5",
    "Task2.2": "b21_et_subtype_d5_l10",
    "Task2.3": "b21_ridge_exact_a100",
}
BANNED = ('"participant_id"', '"subject_group"', '"row_index"', '"oof_predictions"', '"challenge_predictions"')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    return p.parse_args()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> int:
    args = parse_args()
    root = args.input_dir.expanduser().resolve()
    files = sorted(p.name for p in root.iterdir() if p.is_file())
    if files != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E01 safe-output file set mismatch")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E01 aggregate privacy contract failed: {name}")
    bridge = json.loads(texts["bridge-result.json"])
    metrics = json.loads(texts["metrics.json"])
    expected_bridge = {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e01_blob_sha": E01_BLOB,
        "strategy_e01_v2_blob_sha": E01_V2_BLOB,
        "config_blob_sha": CONFIG_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
    for key, value in expected_bridge.items():
        if bridge.get(key) != value:
            raise SystemExit(f"E01 bridge contract mismatch: {key}")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json"):
        raise SystemExit("E01 metrics hash mismatch")
    if bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E01 summary hash mismatch")
    if metrics.get("experiment") != "strategy_v2_e01_paired_evaluation":
        raise SystemExit("E01 experiment identity mismatch")
    if metrics.get("comparison_contract") != "paired_subject_purged_v2":
        raise SystemExit("E01 comparison contract mismatch")
    if metrics.get("frozen_incumbent") != EXPECTED_INCUMBENT:
        raise SystemExit("E01 frozen incumbent mismatch")
    if set((metrics.get("tasks") or {})) != set(EXPECTED_INCUMBENT):
        raise SystemExit("E01 task set mismatch")
    if metrics.get("competition_submission_attempted") is not False or metrics.get("leaderboard_used_for_selection") is not False:
        raise SystemExit("E01 selection/submission contract mismatch")
    if metrics.get("contains_participant_identifiers") is not False or metrics.get("contains_row_level_predictions") is not False:
        raise SystemExit("E01 privacy flag mismatch")
    if metrics.get("subset_sensitivity") != {"n": 28, "repetitions": 200, "without_replacement": True}:
        raise SystemExit("E01 sensitivity contract mismatch")

    print(
        "CMI_FLU_E01_RESULT PASS aggregate_only=true submission=false "
        f"science_commit={SCIENCE_COMMIT} md5_verified={bridge.get('md5_verified_count')}"
    )
    for task in EXPECTED_INCUMBENT:
        payload = metrics["tasks"][task]
        incumbent = payload.get("incumbent") or {}
        if incumbent.get("name") != EXPECTED_INCUMBENT[task]:
            raise SystemExit(f"E01 incumbent mismatch: {task}")
        im = incumbent.get("metrics") or {}
        strict = im.get("study_equal_weight_spearman_mean_strict")
        diagnostic = im.get("study_equal_weight_spearman_mean_finite_diagnostic")
        pooled = (im.get("pooled_within_study_rank_spearman") or {}).get("value")
        if strict is not None and not finite(strict):
            raise SystemExit(f"E01 non-finite strict score: {task}")
        if diagnostic is not None and not finite(diagnostic):
            raise SystemExit(f"E01 non-finite diagnostic score: {task}")
        if pooled is not None and not finite(pooled):
            raise SystemExit(f"E01 non-finite pooled score: {task}")
        raw_rmse = (im.get("raw_scale_rmse") or {}).get("value") if isinstance(im.get("raw_scale_rmse"), dict) else None
        print(
            f"{task} incumbent={incumbent['name']} strict={strict} finite={diagnostic} "
            f"pooled_within_study_rank={pooled} raw_rmse={raw_rmse} "
            f"undefined={im.get('undefined_fold_count')} constant={im.get('constant_fold_count')}"
        )
        for name, comp in sorted((payload.get("comparisons") or {}).items()):
            delta = comp.get("study_mean_delta_strict")
            if delta is not None and not finite(delta):
                raise SystemExit(f"E01 non-finite paired delta: {task}/{name}")
            print(
                f"{task} comparison={name} delta={delta} "
                f"candidate_heuristic={comp.get('passes_candidate_delta_heuristic')} "
                f"all_folds_valid={comp.get('all_folds_valid')} "
                f"large_study_review={len(comp.get('large_studies_requiring_review') or [])}"
            )
        shuffle = (payload.get("subject_block_label_shuffle_negative_control") or {}).get("metrics") or {}
        print(
            f"{task} shuffle_strict={shuffle.get('study_equal_weight_spearman_mean_strict')} "
            f"shuffle_finite={shuffle.get('study_equal_weight_spearman_mean_finite_diagnostic')}"
        )
        sensitivity = payload.get("sensitivity_28") or {}
        for comparison_name, evidence in sorted(sensitivity.items()):
            for study in evidence.get("studies") or []:
                if study.get("status") == "ok":
                    print(
                        f"{task} sensitivity={comparison_name} study={study.get('study')} n={study.get('n')} "
                        f"delta_mean={study.get('spearman_delta_mean')} q10={study.get('spearman_delta_q10')} "
                        f"q90={study.get('spearman_delta_q90')} better_fraction={study.get('candidate_better_fraction')}"
                    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
