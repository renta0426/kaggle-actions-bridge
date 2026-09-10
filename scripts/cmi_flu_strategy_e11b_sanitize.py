#!/usr/bin/env python3
"""Validate and print aggregate-only E11b Task1.1 TabPFN-3 outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260910-cmi-flu-strategy-e11b-tabpfn3-001"
TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-001"
SCIENCE_COMMIT = "95c692bf2e27f6d07ea31a45fd788293666ea94a"
E11B_BLOB = "8a291ae4952786bf16ee667caa5557599310e3fd"
E11B_SYNTH_BLOB = "f25d04d75db584db78d26a232ab84dc5ab1bbf7d"
PACKAGE_VERSION = "8.5.0"
SOURCE_COMMIT = "9ed44abd5882140b88c9f2816c5791987ce059b9"
WHEEL_FILENAME = "tabpfn-8.5.0-py3-none-any.whl"
WHEEL_SHA256 = "4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0"
MODEL_SOURCE = "prior-labsai/tabpfn-3/pytorch/default/1"
CHECKPOINT_FILENAME = "tabpfn-v3-regressor-v3_default.ckpt"
CHECKPOINT_BYTES = 233_289_807
CHECKPOINT_SHA256 = "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301"
PARAMS = {
    "n_estimators": 8,
    "device": "cpu",
    "fit_mode": "fit_preprocessors",
    "memory_saving_mode": "auto",
    "random_state": 20260910,
    "n_preprocessing_jobs": 1,
    "show_progress_bar": False,
    "ignore_pretraining_limits": False,
}
BANNED = (
    '"participant_id"',
    '"subject_group"',
    '"row_index"',
    '"oof_predictions"',
    '"challenge_predictions"',
    "ROW_",
    "SUB_",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SystemExit(f"E11b nonfinite numeric value:{value!r}")
    return number


def validate_fit(audit: dict, *, expected_studies: int, final: bool) -> dict:
    if audit.get("backend") != "tabpfn":
        raise SystemExit("E11b backend identity mismatch")
    if audit.get("package_verified") is not True:
        raise SystemExit("E11b package verification missing")
    if audit.get("package_version_expected") != PACKAGE_VERSION:
        raise SystemExit("E11b package version mismatch")
    if audit.get("package_source_commit") != SOURCE_COMMIT:
        raise SystemExit("E11b source commit mismatch")
    if audit.get("wheel_filename") != WHEEL_FILENAME or audit.get("wheel_sha256") != WHEEL_SHA256:
        raise SystemExit("E11b wheel identity mismatch")
    if audit.get("model_source") != MODEL_SOURCE:
        raise SystemExit("E11b model source mismatch")
    checkpoint = audit.get("checkpoint") or {}
    if (
        checkpoint.get("verified") is not True
        or checkpoint.get("filename") != CHECKPOINT_FILENAME
        or int(checkpoint.get("bytes", 0)) != CHECKPOINT_BYTES
        or checkpoint.get("sha256") != CHECKPOINT_SHA256
        or checkpoint.get("kaggle_model_source") != MODEL_SOURCE
    ):
        raise SystemExit("E11b checkpoint verification mismatch")
    if int(audit.get("training_studies", -1)) != expected_studies:
        raise SystemExit("E11b training-study count mismatch")
    raw_features = int(audit.get("raw_features", 9999))
    if not 0 < raw_features <= 200:
        raise SystemExit("E11b raw-feature guard failed")
    if audit.get("target_transform") != "log":
        raise SystemExit("E11b target-transform mismatch")
    if audit.get("tabpfn_params") != PARAMS:
        raise SystemExit("E11b TabPFN parameter mismatch")
    if audit.get("held_outcomes_used_for_fit") is not False:
        raise SystemExit("E11b held-outcome fit boundary failed")
    fit_seconds = finite(audit.get("fit_seconds"))
    rows = int(audit.get("training_rows", -1))
    if final and rows != 127:
        raise SystemExit("E11b final training-row count mismatch")
    return {
        "raw_features": raw_features,
        "numeric_features": int(audit.get("numeric_features", -1)),
        "categorical_features": int(audit.get("categorical_features", -1)),
        "fit_seconds": fit_seconds,
        "training_rows": rows,
        "training_studies": int(audit.get("training_studies", -1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    root = parser.parse_args().input_dir.resolve()
    files = sorted(path.name for path in root.iterdir() if path.is_file())
    if files != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit(f"E11b safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E11b aggregate privacy contract failed:{name}")

    bridge = json.loads(texts["bridge-result.json"])
    metrics = json.loads(texts["metrics.json"])
    expected_bridge = {
        "request_id": REQUEST_ID,
        "target_kernel": TARGET,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e11b_blob_sha": E11B_BLOB,
        "strategy_e11b_synthetic_blob_sha": E11B_SYNTH_BLOB,
        "tabpfn_package_version": PACKAGE_VERSION,
        "tabpfn_wheel_sha256": WHEEL_SHA256,
        "tabpfn_model_source": MODEL_SOURCE,
        "tabpfn_checkpoint_sha256": CHECKPOINT_SHA256,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
    for key, expected in expected_bridge.items():
        if bridge.get(key) != expected:
            raise SystemExit(f"E11b bridge contract mismatch:{key}:{bridge.get(key)!r}")
    if bridge.get("synthetic") is True:
        raise SystemExit("E11b real output marked synthetic")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json"):
        raise SystemExit("E11b metrics hash mismatch")
    if bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E11b summary hash mismatch")

    if int(metrics.get("schema_version", -1)) != 1 or metrics.get("experiment") != "strategy_v2_e11b_task11_tabpfn3":
        raise SystemExit("E11b experiment identity mismatch")
    if metrics.get("comparison_contract") != "paired_subject_purged_v2":
        raise SystemExit("E11b comparison contract mismatch")
    for key in (
        "cross_study_target_scale_redefined",
        "held_outcomes_used_for_fit",
        "held_outcomes_used_for_scoring",
        "leaderboard_used_for_selection",
        "public_probe_authorized",
        "competition_submission_attempted",
        "incumbent_changed",
    ):
        if metrics.get(key) is not False:
            raise SystemExit(f"E11b execution/leakage boundary mismatch:{key}")
    if int(metrics.get("automatic_compute_retries", -1)) != 0:
        raise SystemExit("E11b retry boundary mismatch")

    expected_frozen = {
        "task": "Task1.1",
        "base_model": "pls_2",
        "target_transform": "log",
        "feature_space": "unchanged_compact_b21_task11",
        "max_raw_features": 200,
        "tabpfn_package_version": PACKAGE_VERSION,
        "tabpfn_source_commit": SOURCE_COMMIT,
        "tabpfn_wheel_filename": WHEEL_FILENAME,
        "tabpfn_wheel_sha256": WHEEL_SHA256,
        "kaggle_model_source": MODEL_SOURCE,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "tabpfn_params": PARAMS,
        "promotion_mean_delta": 0.02,
        "promotion_minimum_study_delta": -0.10,
        "promotion_requires_strict_majority_wins": True,
    }
    if (metrics.get("frozen_conditions") or {}) != expected_frozen:
        raise SystemExit("E11b frozen-condition mismatch")
    license_info = metrics.get("external_model_license") or {}
    if (
        license_info.get("model_license") != "tabpfn-3-license-v1.0"
        or license_info.get("weight_redistribution_by_project") is not False
        or license_info.get("official_kaggle_model_input") is not True
    ):
        raise SystemExit("E11b external-model license boundary mismatch")

    task = metrics.get("task") or {}
    if task.get("task") != "Task1.1":
        raise SystemExit("E11b task mismatch")
    if task.get("base_contract") != {
        "kind": "b21",
        "model": "pls_2",
        "target_transform": "log",
        "current_competition_incumbent": True,
    }:
        raise SystemExit("E11b base contract mismatch")
    folds = task.get("folds") or []
    if len(folds) != 4 or {str(fold.get("held_study")) for fold in folds} != {
        "SDY180",
        "SDY515",
        "SDY519",
        "SDY56",
    }:
        raise SystemExit("E11b held-study contract mismatch")

    aggregate_folds = []
    deltas = []
    for fold in folds:
        base_score = finite(fold.get("base_spearman"))
        tabpfn_score = finite(fold.get("tabpfn_spearman"))
        delta = finite(fold.get("tabpfn_delta"))
        if abs((tabpfn_score - base_score) - delta) > 1e-12:
            raise SystemExit("E11b fold delta arithmetic mismatch")
        if fold.get("held_outcomes_used_for_fit") is not False or fold.get("held_outcomes_used_for_scoring") is not False:
            raise SystemExit("E11b held-outcome fold boundary failed")
        fit = validate_fit(fold.get("fit_audit") or {}, expected_studies=3, final=False)
        aggregate_folds.append(
            {
                "held_study": str(fold.get("held_study")),
                "n": int(fold.get("n", 0)),
                "base_spearman": base_score,
                "tabpfn_spearman": tabpfn_score,
                "delta": delta,
                "fit": fit,
            }
        )
        deltas.append(delta)

    wins = sum(value > 0 for value in deltas)
    required = len(deltas) // 2 + 1
    mean_delta = sum(deltas) / len(deltas)
    minimum_delta = min(deltas)
    expected_pass = bool(mean_delta >= 0.02 and minimum_delta >= -0.10 and wins >= required)
    candidate = task.get("candidate") or {}
    if candidate.get("name") != "tabpfn3_default_cpu":
        raise SystemExit("E11b candidate identity mismatch")
    promotion = candidate.get("promotion") or {}
    if promotion.get("passed") is not expected_pass:
        raise SystemExit("E11b promotion boolean mismatch")
    if (
        abs(finite(promotion.get("mean_delta")) - mean_delta) > 1e-12
        or abs(finite(promotion.get("minimum_delta")) - minimum_delta) > 1e-12
    ):
        raise SystemExit("E11b promotion metric mismatch")
    if int(promotion.get("wins", -1)) != wins or int(promotion.get("required_wins", -1)) != required:
        raise SystemExit("E11b promotion win mismatch")
    if candidate.get("competition_candidate") is not expected_pass or candidate.get("public_probe_authorized") is not False:
        raise SystemExit("E11b candidate promotion boundary mismatch")

    agreement = candidate.get("challenge_agreement_vs_base") or {}
    challenge_rank = finite((agreement.get("rank_spearman") or {}).get("value"))
    changed = int(agreement.get("changed_rank_count", -1))
    if not 0 <= changed <= 40:
        raise SystemExit("E11b Challenge changed-rank count mismatch")
    final_fit = validate_fit(task.get("final_fit_audit") or {}, expected_studies=4, final=True)

    feasibility = task.get("feasibility") or {}
    if (
        feasibility.get("ready_for_one_cpu_tabpfn_comparison") is not True
        or int(feasibility.get("train_rows", -1)) != 127
        or int(feasibility.get("train_studies", -1)) != 4
        or int(feasibility.get("challenge_rows", -1)) != 40
        or feasibility.get("target_transform") != "log"
        or feasibility.get("package_version") != PACKAGE_VERSION
        or feasibility.get("source_commit") != SOURCE_COMMIT
        or feasibility.get("wheel_sha256") != WHEEL_SHA256
        or feasibility.get("kaggle_model_source") != MODEL_SOURCE
        or feasibility.get("checkpoint_filename") != CHECKPOINT_FILENAME
        or int(feasibility.get("checkpoint_bytes", 0)) != CHECKPOINT_BYTES
        or feasibility.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or feasibility.get("tabpfn_params") != PARAMS
    ):
        raise SystemExit("E11b feasibility identity mismatch")
    outer = feasibility.get("outer_splits") or []
    if len(outer) != 4 or any(int(split.get("subject_overlap", -1)) != 0 for split in outer):
        raise SystemExit("E11b feasibility subject-purge mismatch")
    if int(task.get("challenge_rows", -1)) != 40:
        raise SystemExit("E11b challenge-row count mismatch")

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
        "external_model": {
            "package_version": PACKAGE_VERSION,
            "model_source": MODEL_SOURCE,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
    }
    print(
        "CMI_FLU_E11B_RESULT PASS aggregate_only=true submission=false public_probe=false "
        f"result={json.dumps(aggregate, sort_keys=True, separators=(',', ':'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
