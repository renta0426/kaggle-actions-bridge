#!/usr/bin/env python3
"""Authoritative entry point and regression suite for the public CodeParrot evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_codeparrot_frozen_full_public as core


def self_test() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.95, 0.85, 0.4, 0.3], dtype=float)
    boundary = core.conservative_boundary(labels, scores, 0.25)
    assert boundary["allowed_false_positives"] == 1
    assert boundary["achieved_false_positives"] == 1
    assert boundary["cutoff_score"] == 0.8
    distinct = core.low_fpr(labels, scores, 0.25)
    assert distinct["conservative_empirical"]["true_positives"] == 2
    assert distinct["conservative_empirical"]["member_count"] == 4
    assert distinct["conservative_empirical"]["tpr"] == 0.5

    tied = np.array([0.9, 0.9, 0.2, 0.1, 0.95, 0.85, 0.3, 0.0], dtype=float)
    tie_boundary = core.conservative_boundary(labels, tied, 0.25)
    assert tie_boundary["allowed_false_positives"] == 1
    assert tie_boundary["achieved_false_positives"] == 0
    assert tie_boundary["cutoff_score"] == 0.9
    assert tie_boundary["excluded_tie_block_size"] == 2
    tie_report = core.low_fpr(labels, tied, 0.25)
    assert tie_report["conservative_empirical"]["true_positives"] == 1

    hashes = [f"{index:064x}" for index in range(4)]
    raw = pd.DataFrame({
        "sample_id": [f"cp-{value}" for value in hashes],
        "content_sha256": hashes,
        "language": ["Python"] * 4,
        "token_count": [10, 20, 30, 40],
        "window_count": [1, 1, 2, 3],
        "loss_multiwindow": [1.0, 2.0, 4.0, 3.0],
        "legacy_minkpp10": [2.0, 1.0, 3.0, 4.0],
        "paper_minkpp10": [4.0, 3.0, 2.0, 1.0],
        "local_64": [1.0, 2.0, 3.0, 4.0],
        "neg_mean_log_rank": [1.0, 2.0, 3.0, 4.0],
    })
    prepared, digest = core.seal_label_free_predictors(raw)
    assert prepared.short_half_local_or_paper_v1.tolist() == [1.0, 0.75, 0.75, 1.0]
    expected_stage2 = np.array([
        (0.25 + 0.50 + 0.25) / 3.0,
        (0.50 + 0.25 + 0.50) / 3.0,
        (1.00 + 0.75 + 0.75) / 3.0,
        (0.75 + 1.00 + 1.00) / 3.0,
    ])
    assert np.allclose(prepared.stage2_v1_rank_fusion.to_numpy(float), expected_stage2)
    assert len(digest) == 64

    assert core.copies_one(1) and core.copies_one("1")
    assert not core.copies_one(True) and not core.copies_one(2)
    assert core.length_bucket(512) == 9
    assert core.length_bucket(32768) == 14
    assert core.length_bucket(1_000_000) == 14
    print(
        "CODEPARROT_PUBLIC_EVALUATOR_V2_SELF_TEST PASS "
        "distinct_fp_budget=1 distinct_tp=2 tie_block_unsplit=1 "
        "aux_rank_fixture=1 stage2_rank_fixture=1"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.prediction_dir is None or args.output_dir is None:
        parser.error("--prediction-dir and --output-dir are required")
    result = core.run(args.prediction_dir, args.output_dir)
    metrics = result["predictor_metrics"]
    boot = result["primary_paired_repository_group_bootstrap"]["conservative_empirical_tpr"]
    print(
        "CODEPARROT_PUBLIC_FROZEN_EVAL_V2 PASS "
        + json.dumps(
            {
                "rows": result["rows"],
                "local_64_auc": metrics["local_64"]["auc_roc"],
                "local_64_tpr_01": metrics["local_64"]["low_fpr"]["0.010"]["conservative_empirical"]["tpr"],
                "aux_auc": metrics["short_half_local_or_paper_v1"]["auc_roc"],
                "aux_tpr_01": metrics["short_half_local_or_paper_v1"]["low_fpr"]["0.010"]["conservative_empirical"]["tpr"],
                "stage2_v1_auc": metrics["stage2_v1_rank_fusion"]["auc_roc"],
                "bootstrap_tpr_delta_mean": boot["mean"],
                "bootstrap_tpr_delta_lower95": boot["lower_95"],
                "bootstrap_tpr_delta_upper95": boot["upper_95"],
                "promote_aux": result["promote_short_half_local_or_paper_v1"],
                "label_free_predictor_sha256": result["label_free_predictor_sha256"],
                "evaluation_manifest_sha256": result["evaluation_manifest_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
