"""No-fit V3-02/V3-05 saved-bank reconciliation; never emits row-level data.

The two runs are immutable. This module checks artifact identities before parsing,
compares paired saved predictions, and diagnoses ECDF support without fitting or
changing any predictor. It cannot establish historical split identity when the
old bank did not persist split assignments.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

EXPERIMENT = "strategy_v3_saved_bank_reconciliation_v1"
TARGET = "strict_ASC_D7_target"
BASELINE = "strict_ASC_raw_baseline"
ALIAS_HASH = "6626de8ebc839dbdac44faa19e8c3280348325d2a214fe940984a9d209fe2985"
ARTIFACTS = {
    "v02": {
        "v3_v02_oof_bank.csv": (1636096, "a0c4b4a02eceae7ce31806ecbca8cf45d084c45d2041fe82101750f5f46502c6"),
        "v3_v02_challenge_bank.csv": (125889, "44fe10216e588423cbfe00a997c2fc94951b86b46c67593eac56d1c22f29a137"),
        "v3_v02_summary.json": (125930, "1ed0f4998e6d9c90c8a9caa7abbca84844524fb92131aabee2088dbea5908c41"),
        "v3_v02_bank_manifest.json": (661, "accf5b9090074dba0af8ac2988fab417c5f3309b4a08e5eedaecf45d9c7cc532"),
    },
    "v05": {
        "v3_v05_task13_oof_bank.csv": (13975, "36699e669464f1408163888248af1e750502e2455c05048cfb8b821d368fd28f"),
        "v3_v05_task13_challenge_bank.csv": (6846, "da782f693a98b145302fcf91f757bda796c2c654752b0ca05ddb041e2216b016"),
        "v3_v05_task13_summary.json": (19568, "f479fb84173ed52bf1e89862ee2493000cbfe7ded24087592eec2a687aaaeb12"),
        "v3_v05_task13_bank_manifest.json": (649, "e3056e6d8237845957e69bad2a568add6aa3b67a74d476109f0e5b4654c9af50"),
    },
}


class AuditError(ValueError):
    """Bounded structural error; callers must not log arbitrary data values."""


def check(condition: bool, code: str) -> None:
    if not condition:
        raise AuditError(code)


def finite(frame: pd.DataFrame, columns: list[str]) -> None:
    check(set(columns).issubset(frame.columns), "required_numeric_columns_missing")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    check(bool(np.isfinite(values).all()), "nonfinite_numeric_values")


def unique(frame: pd.DataFrame, keys: list[str]) -> None:
    check(set(keys).issubset(frame.columns), "identity_columns_missing")
    check(not frame[keys].isna().any().any(), "missing_identity")
    check(not frame.duplicated(keys).any(), "duplicate_identity")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    status = "constant_target" if np.unique(y).size < 2 else (
        "constant_prediction" if np.unique(pred).size < 2 else "ok")
    return {"n": len(y), "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
            "spearman": float(spearmanr(y, pred).statistic) if status == "ok" else None,
            "spearman_status": status}


def paired(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    check(a.shape == b.shape and a.ndim == 1, "paired_shape_mismatch")
    return {"n": len(a), "max_absolute_difference": float(np.max(np.abs(a - b))),
            "exact_equal": bool(np.array_equal(a, b)),
            "equal_at_1e_12": bool(np.allclose(a, b, rtol=0, atol=1e-12)),
            "same_weak_order": bool(np.array_equal(rankdata(a), rankdata(b))),
            "changed_rank_count": int(np.sum(rankdata(a) != rankdata(b)))}


def audit_frames(old: pd.DataFrame, old_challenge: pd.DataFrame,
                 new: pd.DataFrame, challenge: pd.DataFrame,
                 summary: dict[str, Any]) -> dict[str, Any]:
    """Analyze already-saved numeric vectors; no model/data loader is invoked."""
    check(set(old.columns) == {"task", "candidate", "prediction_space", "target_unit", "provenance", "participant_id", "subject_group", "study_group", "target", "prediction"}, "v02_oof_schema_changed")
    check(set(old_challenge.columns) == {"task", "candidate", "prediction_space", "target_unit", "provenance", "participant_id", "prediction"}, "v02_challenge_schema_changed")
    check(set(new.columns) == {"task", "row_index", "repeat", "split", TARGET, "raw_baseline", "S1", "S2", "S3", "S4", "b21_compatible_pls1", "participant_id", "subject_group", "study_group"}, "v05_oof_schema_changed")
    check(set(challenge.columns) == {"participant_id", "subject_group", "study_group", BASELINE, "raw_baseline", "b21_compatible_pls1", "S1", "S2", "S3", "S4"}, "v05_challenge_schema_changed")
    old = old.loc[old.task.eq("Task1.3") & old.candidate.eq("b21_pls_1")].copy()
    oc = old_challenge.loc[old_challenge.task.eq("Task1.3") & old_challenge.candidate.eq("b21_pls_1_fresh")].copy()
    check(len(old) == 23 and len(oc) == 40 and len(new) == 69 and len(challenge) == 40, "frozen_support_changed")
    check(new.task.eq("Task1.3").all() and old.prediction_space.eq("raw_target").all() and oc.prediction_space.eq("raw_target").all(), "task_or_prediction_space_changed")
    for f in (old, oc, challenge):
        unique(f, ["participant_id"])
    unique(new, ["participant_id", "repeat"])
    unique(new, ["row_index", "repeat"])
    check(set(new.repeat) == {0, 1, 2}, "repeat_set_changed")
    check(new.groupby("participant_id").size().eq(3).all(), "incomplete_repeats")
    for c in ("row_index", "subject_group", "study_group", TARGET, "raw_baseline"):
        check(new.groupby("participant_id")[c].nunique(dropna=False).eq(1).all(), "repeated_identity_or_teacher_changed")
    check(new.groupby("row_index").participant_id.nunique().eq(1).all(), "row_index_rebound")
    check(old.subject_group.nunique() == 23 and new.subject_group.nunique() == 23 and challenge.subject_group.nunique() == 40, "independent_subject_count_changed")
    check(set(old.study_group) <= {"2024UGA", "2024_UGA"} and set(new.study_group) <= {"2024UGA", "2024_UGA"} and set(challenge.study_group) == {"2025LJI"}, "unexpected_study_alias")
    finite(old, ["target", "prediction"]); finite(oc, ["prediction"])
    finite(new, [TARGET, "raw_baseline", "S1", "S2", "S3", "S4", "b21_compatible_pls1"])
    finite(challenge, [BASELINE, "raw_baseline", "S1", "S2", "S3", "S4", "b21_compatible_pls1"])
    check(set(old.participant_id) == set(new.participant_id) and set(oc.participant_id) == set(challenge.participant_id), "cross_run_cohort_changed")
    agg = new.groupby("participant_id", sort=True).agg({TARGET: "first", "raw_baseline": "first", "subject_group": "first", "b21_compatible_pls1": "mean"})
    old = old.set_index("participant_id").loc[agg.index]
    check(old.subject_group.eq(agg.subject_group).all(), "cross_run_subject_mapping_changed")
    target_cmp = paired(old.target.to_numpy(float), agg[TARGET].to_numpy(float))
    check(target_cmp["equal_at_1e_12"], "cross_run_teacher_changed")
    cb = challenge.set_index("participant_id").sort_index()
    oc = oc.set_index("participant_id").loc[cb.index]
    x = agg.raw_baseline.to_numpy(float); cx = cb.raw_baseline.to_numpy(float)
    check(np.array_equal(cx, cb[BASELINE].to_numpy(float)), "challenge_baseline_alias_values_changed")
    check(bool((x >= 0).all() and (cx >= 0).all()), "negative_raw_flow")
    score = np.array([(np.sum(x < v) + .5 * np.sum(x == v)) / len(x) for v in cx])
    p = summary["Task1.3"]["full_fit_parameters"]
    expected_s3 = float(p["S3"]["a"]) + float(p["S3"]["b"]) * score
    s3_cmp = paired(expected_s3, cb.S3.to_numpy(float))
    check(s3_cmp["equal_at_1e_12"], "S3_saved_formula_mismatch")
    check(np.allclose(float(p["S2"]["c"]) * cx, cb.S2, rtol=0, atol=1e-12), "S2_saved_formula_mismatch")
    check(cb.S1.nunique() == 1, "full_fit_S1_not_constant")
    check(summary["runtime"]["repair_study_alias_sha256"] == ALIAS_HASH, "study_alias_hash_changed")
    by_repeat = []
    for repeat, group in new.groupby("repeat", sort=True):
        folds = group.split.astype(str)
        check(folds.str.fullmatch(rf"repeat={int(repeat)}/fold=[0-4]").all() and folds.nunique() == 5, "split_labels_changed")
        by_repeat.append({"repeat": int(repeat), "rows": len(group), "fold_sizes": sorted(int(n) for n in group.groupby("split").size())})
    old_pred = old.prediction.to_numpy(float); new_pred = agg.b21_compatible_pls1.to_numpy(float)
    result = {
        "schema_version": 1, "experiment": EXPERIMENT, "model_fit_count": 0,
        "kaggle_write_count": 0, "kaggle_compute_launch_count": 0, "competition_submission_count": 0,
        "individual_identifiers_or_prediction_vectors_emitted": False,
        "paired_teacher": target_cmp, "source_unique_subjects": 23, "challenge_unique_subjects": 40,
        "reference_oof": {"comparison": paired(old_pred, new_pred),
            "v02_metrics": metrics(agg[TARGET].to_numpy(float), old_pred),
            "v05_metrics": metrics(agg[TARGET].to_numpy(float), new_pred),
            "historical_split_identity": "unresolved_V3_02_bank_omits_assignments",
            "difference_cause": "not_attributed_to_precision_without_source_feature_split_evidence"},
        "reference_challenge": paired(oc.prediction.to_numpy(float), cb.b21_compatible_pls1.to_numpy(float)),
        "S3_support": {"source_min": float(x.min()), "source_max": float(x.max()),
            "challenge_min": float(cx.min()), "challenge_max": float(cx.max()),
            "below_source_min": int(np.sum(cx < x.min())), "above_source_max": int(np.sum(cx > x.max())),
            "within_source_range": int(np.sum((cx >= x.min()) & (cx <= x.max()))),
            "empirical_score_unique": int(np.unique(score).size), "saved_prediction_unique": int(cb.S3.nunique()),
            "slope_positive": bool(p["S3"]["b"] > 0), "saved_formula": s3_cmp,
            "ECDF_collapsed": bool(np.unique(score).size == 1),
            "supports_correction_or_unit_conversion": False},
        "S2_actual_rank_contract": paired(cx, cb.S2.to_numpy(float)),
        "S4_actual_rank_contract": paired(cx, cb.S4.to_numpy(float)),
        "repeat_coverage": by_repeat,
        "new_candidates_created": 0, "historical_results_rewritten": False,
    }
    # A schema-built result contains only aggregate numbers, booleans, fixed text;
    # never copy arbitrary input metadata into an exportable dictionary.
    json.dumps(result, allow_nan=False)
    return result


def verify_artifacts(root: Path, artifacts: dict[str, tuple[int, str]]) -> None:
    check(root.is_dir() and not root.is_symlink(), "invalid_artifact_directory")
    check({p.name for p in root.iterdir()} == set(artifacts), "artifact_allowlist_changed")
    for name, (size, digest) in artifacts.items():
        p = root / name
        check(p.is_file() and not p.is_symlink() and p.stat().st_size == size, "artifact_size_or_type_changed")
        check(hashlib.sha256(p.read_bytes()).hexdigest() == digest, "artifact_hash_changed")


def audit_paths(v02: Path, v05: Path) -> dict[str, Any]:
    for name, path in (("v02", v02), ("v05", v05)):
        verify_artifacts(path, ARTIFACTS[name])
    def read(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, dtype={"participant_id": str, "subject_group": str, "study_group": str}, float_precision="round_trip")
    result = audit_frames(read(v02 / "v3_v02_oof_bank.csv"), read(v02 / "v3_v02_challenge_bank.csv"), read(v05 / "v3_v05_task13_oof_bank.csv"), read(v05 / "v3_v05_task13_challenge_bank.csv"), json.loads((v05 / "v3_v05_task13_summary.json").read_text()))
    result["input_artifact_sha256"] = {name: digest for items in ARTIFACTS.values() for name, (_, digest) in items.items()}
    return result
