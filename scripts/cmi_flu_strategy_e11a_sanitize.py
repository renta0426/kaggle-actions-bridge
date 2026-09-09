#!/usr/bin/env python3
"""Validate and print aggregate-only E11a Task1.1 pairwise outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260910-cmi-flu-strategy-e11a-task11-pairwise-001"
TARGET = "renta0426/cmi-flu-e11a-task11-pairwise-20260910-001"
SCIENCE_COMMIT = "4d8c568e3b26ad6ba48c076bcb28fda64a0dc72f"
E11_BLOB = "ec7e709ac83f9792b6cc4d63ad398a8a68e16e7b"
E11_SYNTH_BLOB = "ff5c7cbc5aea7ebdbb9e16853d26ec8179cf2d6c"
PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 10,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 20260910,
}
BANNED = (
    '"participant_id"', '"subject_group"', '"row_index"',
    '"oof_predictions"', '"challenge_predictions"', 'ROW_', 'SUB_',
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SystemExit(f"E11a nonfinite numeric value:{value!r}")
    return number


def validate_fit(audit: dict, *, expected_studies: int, final: bool) -> dict:
    if audit.get("pairs_within_study_only") is not True:
        raise SystemExit("E11a pair policy mismatch")
    if audit.get("cross_study_pair_labels_used") is not False:
        raise SystemExit("E11a cross-study pair-label boundary failed")
    directed = int(audit.get("directed_pairs", 0))
    positive = int(audit.get("positive_pairs", -1))
    negative = int(audit.get("negative_pairs", -2))
    if directed < 100 or positive != negative or positive + negative != directed:
        raise SystemExit("E11a pair-count/balance contract failed")
    if int(audit.get("training_studies", -1)) != expected_studies:
        raise SystemExit("E11a source-study count mismatch")
    transformed = int(audit.get("transformed_features", 9999))
    if transformed > 512:
        raise SystemExit("E11a transformed-feature guard failed")
    if audit.get("reference_aggregation") != "study_equal":
        raise SystemExit("E11a reference aggregation mismatch")
    if audit.get("held_outcomes_used_for_fit") is not False:
        raise SystemExit("E11a held-outcome fit boundary failed")
    if audit.get("classifier") != "HistGradientBoostingClassifier" or audit.get("classifier_params") != PARAMS:
        raise SystemExit("E11a classifier contract mismatch")
    fit_seconds = finite(audit.get("fit_seconds"))
    if final and int(audit.get("training_rows", -1)) != 127:
        raise SystemExit("E11a final training-row count mismatch")
    return {
        "directed_pairs": directed,
        "unordered_pairs": int(audit.get("unordered_pairs", -1)),
        "transformed_features": transformed,
        "fit_seconds": fit_seconds,
        "training_rows": int(audit.get("training_rows", -1)),
        "training_studies": int(audit.get("training_studies", -1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    root = parser.parse_args().input_dir.resolve()
    files = sorted(path.name for path in root.iterdir() if path.is_file())
    if files != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit(f"E11a safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E11a aggregate privacy contract failed:{name}")

    bridge = json.loads(texts["bridge-result.json"])
    metrics = json.loads(texts["metrics.json"])
    expected_bridge = {
        "request_id": REQUEST_ID,
        "target_kernel": TARGET,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e11_blob_sha": E11_BLOB,
        "strategy_e11_synthetic_blob_sha": E11_SYNTH_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
    for key, expected in expected_bridge.items():
        if bridge.get(key) != expected:
            raise SystemExit(f"E11a bridge contract mismatch:{key}:{bridge.get(key)!r}")
    if bridge.get("synthetic") is True:
        raise SystemExit("E11a real output marked synthetic")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json"):
        raise SystemExit("E11a metrics hash mismatch")
    if bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E11a summary hash mismatch")

    if int(metrics.get("schema_version", -1)) != 1 or metrics.get("experiment") != "strategy_v2_e11a_task11_pairwise_ranker":
        raise SystemExit("E11a experiment identity mismatch")
    if metrics.get("comparison_contract") != "paired_subject_purged_v2":
        raise SystemExit("E11a comparison contract mismatch")
    for key in (
        "cross_study_pair_labels_used", "held_outcomes_used_for_fit",
        "held_outcomes_used_for_scoring", "leaderboard_used_for_selection",
        "competition_submission_attempted", "incumbent_changed", "public_probe_authorized",
    ):
        if metrics.get(key) is not False:
            raise SystemExit(f"E11a execution/leakage boundary mismatch:{key}")
    if int(metrics.get("automatic_compute_retries", -1)) != 0:
        raise SystemExit("E11a retry boundary mismatch")

    frozen = metrics.get("frozen_conditions") or {}
    expected_frozen = {
        "task": "Task1.1",
        "base_model": "pls_2",
        "pair_policy": "within_source_study_only_symmetric",
        "pair_tie_tolerance": 1e-12,
        "max_rows_per_study": 80,
        "max_transformed_features": 512,
        "minimum_directed_pairs": 100,
        "reference_aggregation": "study_equal",
        "classifier": "HistGradientBoostingClassifier",
        "classifier_params": PARAMS,
        "promotion_mean_delta": 0.02,
        "promotion_minimum_study_delta": -0.10,
        "promotion_requires_strict_majority_wins": True,
    }
    if frozen != expected_frozen:
        raise SystemExit("E11a frozen-condition mismatch")

    task = metrics.get("task") or {}
    if task.get("task") != "Task1.1":
        raise SystemExit("E11a task mismatch")
    if task.get("base_contract") != {"kind": "b21", "model": "pls_2", "current_competition_incumbent": True}:
        raise SystemExit("E11a base contract mismatch")
    folds = task.get("folds") or []
    if len(folds) != 4 or {str(fold.get("held_study")) for fold in folds} != {"SDY180", "SDY515", "SDY519", "SDY56"}:
        raise SystemExit("E11a held-study contract mismatch")

    aggregate_folds = []
    deltas = []
    for fold in folds:
        base = finite(fold.get("base_spearman"))
        pairwise = finite(fold.get("pairwise_spearman"))
        delta = finite(fold.get("pairwise_delta"))
        if abs((pairwise - base) - delta) > 1e-12:
            raise SystemExit("E11a fold delta arithmetic mismatch")
        if fold.get("held_outcomes_used_for_fit") is not False or fold.get("held_outcomes_used_for_scoring") is not False:
            raise SystemExit("E11a held-outcome fold boundary failed")
        fit = validate_fit(fold.get("fit_audit") or {}, expected_studies=3, final=False)
        aggregate_folds.append({
            "held_study": str(fold.get("held_study")),
            "n": int(fold.get("n", 0)),
            "base_spearman": base,
            "pairwise_spearman": pairwise,
            "delta": delta,
            "fit": fit,
        })
        deltas.append(delta)

    wins = sum(value > 0 for value in deltas)
    required = len(deltas) // 2 + 1
    mean_delta = sum(deltas) / len(deltas)
    minimum_delta = min(deltas)
    expected_pass = bool(mean_delta >= 0.02 and minimum_delta >= -0.10 and wins >= required)
    candidate = task.get("candidate") or {}
    if candidate.get("name") != "pairwise_hgb":
        raise SystemExit("E11a candidate identity mismatch")
    promotion = candidate.get("promotion") or {}
    if promotion.get("passed") is not expected_pass:
        raise SystemExit("E11a promotion boolean mismatch")
    if abs(finite(promotion.get("mean_delta")) - mean_delta) > 1e-12 or abs(finite(promotion.get("minimum_delta")) - minimum_delta) > 1e-12:
        raise SystemExit("E11a promotion metric mismatch")
    if int(promotion.get("wins", -1)) != wins or int(promotion.get("required_wins", -1)) != required:
        raise SystemExit("E11a promotion win mismatch")

    agreement = candidate.get("challenge_agreement_vs_base") or {}
    challenge_rank = finite((agreement.get("rank_spearman") or {}).get("value"))
    changed = int(agreement.get("changed_rank_count", -1))
    if not 0 <= changed <= 40:
        raise SystemExit("E11a Challenge changed-rank count mismatch")
    final_fit = validate_fit(candidate.get("fit") or {}, expected_studies=4, final=True)

    feasibility = task.get("feasibility") or {}
    if feasibility.get("study_counts") != {"SDY180": 34, "SDY515": 16, "SDY519": 17, "SDY56": 60}:
        raise SystemExit("E11a feasibility study counts mismatch")
    outer = feasibility.get("outer_splits") or []
    if len(outer) != 4 or any(int(split.get("subject_overlap", -1)) != 0 for split in outer):
        raise SystemExit("E11a feasibility subject-purge mismatch")
    if feasibility.get("ready_for_one_cpu_pairwise_comparison") is not True:
        raise SystemExit("E11a feasibility gate not ready")

    selected = task.get("selected_local_candidate")
    if (selected == "pairwise_hgb") is not expected_pass:
        raise SystemExit("E11a local selection mismatch")
    if task.get("competition_candidate") is not False or task.get("public_probe_authorized") is not False or int(task.get("challenge_rows", -1)) != 40:
        raise SystemExit("E11a competition boundary mismatch")

    aggregate = {
        "folds": aggregate_folds,
        "promotion": {
            "passed": expected_pass,
            "mean_delta": mean_delta,
            "minimum_delta": minimum_delta,
            "wins": wins,
            "required_wins": required,
        },
        "challenge_rank_agreement": challenge_rank,
        "challenge_changed_ranks": changed,
        "final_fit": final_fit,
    }
    print(
        "CMI_FLU_E11A_RESULT PASS aggregate_only=true submission=false public_probe=false "
        f"result={json.dumps(aggregate, sort_keys=True, separators=(',', ':'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
