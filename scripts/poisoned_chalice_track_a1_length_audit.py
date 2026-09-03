"""CPU-only Stage 1 score-track audit using the existing 10k raw+structure+FIM cache.

The input is the user's private Kaggle Dataset generated from the completed feature
extraction.  This script never contacts Kaggle and never reads validation shards.
It emits aggregate metrics only; row-level predictions and private feature values
remain runner-local.

Candidate set is intentionally small and frozen before execution:

* length_only: log1p(token_count), log1p(window_count) diagnostic;
* raw_fim_current: the frozen 174-feature raw+structure+FIM baseline;
* raw_fim_no_length: baseline without token_count/window_count;
* raw_fim_log_replace: no-length baseline plus log-count terms;
* raw_fim_log_add: current baseline plus log-count terms.

No feature×length interaction search is performed in v1.  Public-LB and visible
validation scores are not inputs to selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPECTED_ROWS = 10_000
EXPECTED_SHARDS = 40
EXPECTED_ROWS_PER_SHARD = 250
EXPECTED_LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
EXPECTED_BASE = 113
EXPECTED_STRUCTURE = 50
EXPECTED_FIM = 11
EXPECTED_TOTAL = 174
EXPECTED_BASELINE_AUC = 0.664524
BASELINE_AUC_TOLERANCE = 0.002
SEED = 2027
REPEATED_SEEDS = tuple(range(2027, 2037))
CDF_SOURCES = ("score_loss_mean__max", "min_k_10__max", "min_kpp_10__max")
EXCLUDED_COLUMNS = {
    "sample_id", "language", "membership", "label", "split", "content",
    "validation_order",
}
VARIANT_ORDER = (
    "length_only",
    "raw_fim_current",
    "raw_fim_no_length",
    "raw_fim_log_replace",
    "raw_fim_log_add",
)


def discover_train_shards(root: Path) -> list[Path]:
    all_parts = sorted(root.rglob("features.part*.parquet"))
    train = [
        path for path in all_parts
        if "/train_10k/parts/" in path.as_posix()
        or path.as_posix().endswith(
            tuple(f"train_10k/parts/features.part{i:03d}.parquet" for i in range(EXPECTED_SHARDS))
        )
    ]
    # The suffix tuple above is defensive for absolute roots.  Enforce exact names
    # after discovery rather than trusting directory enumeration.
    by_name = {path.name: path for path in train}
    expected_names = {f"features.part{i:03d}.parquet" for i in range(EXPECTED_SHARDS)}
    if set(by_name) != expected_names or len(train) != EXPECTED_SHARDS:
        raise RuntimeError(
            f"train shard inventory mismatch: count={len(train)} names={len(by_name)}"
        )
    for path in train:
        parquet = pq.ParquetFile(path)
        if int(parquet.metadata.num_rows) != EXPECTED_ROWS_PER_SHARD:
            raise RuntimeError("train shard row-count contract changed")
        columns = set(parquet.schema_arrow.names)
        if not {"sample_id", "language", "membership", "label"}.issubset(columns):
            raise RuntimeError("train shard identity/label columns missing")
        if "validation_order" in columns:
            raise RuntimeError("validation shard crossed train-only audit boundary")
    return [by_name[f"features.part{i:03d}.parquet"] for i in range(EXPECTED_SHARDS)]


def read_train(root: Path) -> pd.DataFrame:
    frame = pd.concat(
        [pd.read_parquet(path) for path in discover_train_shards(root)],
        ignore_index=True,
        sort=False,
    )
    if len(frame) != EXPECTED_ROWS or not frame.sample_id.is_unique:
        raise RuntimeError("10k train cache coverage changed")
    if tuple(sorted(frame.language.dropna().unique())) != EXPECTED_LANGUAGES:
        raise RuntimeError("language coverage changed")
    if frame.label.isna().any():
        raise RuntimeError("source training labels are missing")
    membership = frame.membership.astype("string").str.lower().map(
        {"member": 1, "non-member": 0, "non_member": 0}
    )
    if membership.isna().any() or not np.array_equal(
        membership.to_numpy(int), frame.label.to_numpy(int)
    ):
        raise RuntimeError("membership/label mismatch")
    balance = frame.groupby(["language", "label"]).size()
    if len(balance) != 10 or not balance.eq(1000).all():
        raise RuntimeError("language/label balance changed")
    frame = frame.sort_values("sample_id").reset_index(drop=True).copy()
    frame["label"] = frame.label.astype(int)
    return frame


def numeric_features(frame: pd.DataFrame) -> list[str]:
    return [
        name
        for name in frame.select_dtypes(include=[np.number]).columns
        if name not in EXCLUDED_COLUMNS and not name.startswith("cdf_")
    ]


def feature_families(frame: pd.DataFrame) -> dict[str, list[str]]:
    numeric = numeric_features(frame)
    families = {
        "base": [name for name in numeric if not name.startswith(("ast_", "fim_"))],
        "structure_raw": [name for name in numeric if name.startswith("ast_")],
        "fim": [name for name in numeric if name.startswith("fim_")],
    }
    counts = {key: len(value) for key, value in families.items()}
    expected = {"base": EXPECTED_BASE, "structure_raw": EXPECTED_STRUCTURE, "fim": EXPECTED_FIM}
    if counts != expected:
        raise RuntimeError(f"feature-family contract changed: {counts}")
    ordered = families["base"] + families["structure_raw"] + families["fim"]
    if len(ordered) != EXPECTED_TOTAL or len(set(ordered)) != EXPECTED_TOTAL:
        raise RuntimeError("ordered feature contract changed")
    for required in ("token_count", "window_count", *CDF_SOURCES):
        if required not in families["base"]:
            raise RuntimeError(f"required base feature missing: {required}")
    all_nan = [column for column in ordered if frame[column].isna().all()]
    if all_nan:
        raise RuntimeError(f"all-NaN source features: {all_nan[:5]}")
    return families


def prepare_variants(frame: pd.DataFrame) -> dict[str, list[str]]:
    families = feature_families(frame)
    current = families["base"] + families["structure_raw"] + families["fim"]
    no_length = [name for name in current if name not in {"token_count", "window_count"}]
    work = frame
    work["log_token_count"] = np.log1p(work.token_count.to_numpy(float))
    work["log_window_count"] = np.log1p(work.window_count.to_numpy(float))
    variants = {
        "length_only": ["log_token_count", "log_window_count"],
        "raw_fim_current": current,
        "raw_fim_no_length": no_length,
        "raw_fim_log_replace": no_length + ["log_token_count", "log_window_count"],
        "raw_fim_log_add": current + ["log_token_count", "log_window_count"],
    }
    if tuple(variants) != VARIANT_ORDER:
        raise RuntimeError("variant order changed")
    if len(variants["raw_fim_current"]) != 174:
        raise RuntimeError("current feature count changed")
    if len(variants["raw_fim_no_length"]) != 172:
        raise RuntimeError("no-length feature count changed")
    if len(variants["raw_fim_log_replace"]) != 174:
        raise RuntimeError("log-replace feature count changed")
    if len(variants["raw_fim_log_add"]) != 176:
        raise RuntimeError("log-add feature count changed")
    return variants


def add_fold_cdf(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    columns: Sequence[str] = CDF_SOURCES,
    min_language_rows: int = 20,
) -> pd.DataFrame:
    output = target.copy()
    nonmember = reference.loc[reference.label == 0]
    if nonmember.empty:
        raise RuntimeError("CDF reference has no non-members")
    for column in columns:
        global_values = np.sort(nonmember[column].dropna().to_numpy(float))
        if not len(global_values):
            raise RuntimeError(f"CDF source empty: {column}")
        per_language = {
            language: np.sort(group[column].dropna().to_numpy(float))
            for language, group in nonmember.groupby("language")
        }
        calibrated = []
        for language, value in zip(output.language, output[column]):
            values = per_language.get(language, np.array([], dtype=float))
            if len(values) < min_language_rows:
                values = global_values
            calibrated.append(np.searchsorted(values, value, side="right") / len(values))
        output[f"cdf_{column}"] = calibrated
    return output


def pipeline(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.2, max_iter=2_000, random_state=seed),
    )


def stratified_splits(frame: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    strata = frame.language.astype(str) + "_" + frame.label.astype(int).astype(str)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(splitter.split(frame, strata))


def leave_language_out_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    language = frame.language.to_numpy(str)
    index = np.arange(len(frame))
    return [
        (index[language != value], index[language == value])
        for value in EXPECTED_LANGUAGES
    ]


def length_holdout_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    bucket = pd.qcut(frame.token_count, 5, labels=False, duplicates="drop")
    if bucket.isna().any() or bucket.nunique() != 5:
        raise RuntimeError("token_count does not support five strict holdouts")
    values = bucket.astype(int).to_numpy()
    index = np.arange(len(frame))
    return [
        (index[values != value], index[values == value])
        for value in sorted(set(values))
    ]


def cross_fit(
    frame: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    features: Sequence[str],
    seed: int,
) -> np.ndarray:
    cdf_sources = [column for column in CDF_SOURCES if column in features]
    model_features = list(features) + [f"cdf_{column}" for column in cdf_sources]
    prediction = np.full(len(frame), np.nan, dtype=float)
    seen = np.zeros(len(frame), dtype=int)
    labels = frame.label.to_numpy(int)
    for fold, (fit_index, hold_index) in enumerate(splits):
        fit = frame.iloc[fit_index]
        hold = frame.iloc[hold_index]
        if cdf_sources:
            fit = add_fold_cdf(fit, fit, cdf_sources)
            hold = add_fold_cdf(frame.iloc[fit_index], hold, cdf_sources)
        model = pipeline(seed + fold)
        model.fit(fit[model_features], labels[fit_index])
        prediction[hold_index] = model.predict_proba(hold[model_features])[:, 1]
        seen[hold_index] += 1
    if not np.isfinite(prediction).all() or not np.all(seen == 1):
        raise RuntimeError("OOF coverage failure")
    return prediction


def low_fpr_metrics(y_true: Sequence[int], score: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    values = np.asarray(score, dtype=float)
    fpr, tpr, _ = roc_curve(y, values)
    result = {
        "auc": float(roc_auc_score(y, values)),
        "pauc_01": float(roc_auc_score(y, values, max_fpr=0.01)),
    }
    for rate in (0.001, 0.005, 0.01, 0.02):
        result[f"tpr_at_{rate:g}_fpr"] = float(tpr[fpr <= rate].max())
    return result


def evaluate_once(
    frame: pd.DataFrame,
    variants: dict[str, list[str]],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    split_name: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for variant in VARIANT_ORDER:
        score = cross_fit(frame, splits, variants[variant], seed)
        predictions[variant] = score
        rows.append({"split": split_name, "variant": variant} | low_fpr_metrics(frame.label, score))
    return pd.DataFrame(rows), predictions


def language_metrics(frame: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for variant in VARIANT_ORDER:
        score = predictions[variant]
        for language, indices in frame.groupby("language", sort=True).groups.items():
            index = np.asarray(list(indices), dtype=int)
            rows.append(
                {"variant": variant, "language": language, "rows": len(index)}
                | low_fpr_metrics(frame.label.iloc[index], score[index])
            )
    return pd.DataFrame(rows)


def repeated_cv(frame: pd.DataFrame, variants: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for seed in REPEATED_SEEDS:
        splits = stratified_splits(frame, seed)
        for variant in VARIANT_ORDER:
            score = cross_fit(frame, splits, variants[variant], seed)
            rows.append(
                {"seed": seed, "variant": variant}
                | low_fpr_metrics(frame.label, score)
            )
    return pd.DataFrame(rows)


def repeated_summary(repeated: pd.DataFrame) -> pd.DataFrame:
    baseline = repeated.loc[repeated.variant == "raw_fim_current"].set_index("seed")
    rows = []
    for variant, group in repeated.groupby("variant", sort=False):
        group = group.set_index("seed").loc[list(REPEATED_SEEDS)]
        wins = int((group.auc > baseline.auc).sum()) if variant != "raw_fim_current" else 0
        rows.append(
            {
                "variant": variant,
                "seeds": len(group),
                "auc_mean": float(group.auc.mean()),
                "auc_std": float(group.auc.std(ddof=0)),
                "auc_min": float(group.auc.min()),
                "auc_max": float(group.auc.max()),
                "pauc_01_mean": float(group.pauc_01.mean()),
                "tpr_at_0.01_fpr_mean": float(group["tpr_at_0.01_fpr"].mean()),
                "auc_wins_vs_current": wins,
            }
        )
    return pd.DataFrame(rows)


def score_track_gates(
    metrics: pd.DataFrame,
    languages: pd.DataFrame,
    repeats: pd.DataFrame,
) -> pd.DataFrame:
    strat = metrics.loc[metrics.split == "stratified"].set_index("variant")
    length = metrics.loc[metrics.split == "length_holdout"].set_index("variant")
    repeated = repeats.set_index("variant")
    worst = languages.groupby("variant").auc.min().to_dict()
    baseline = strat.loc["raw_fim_current"]
    baseline_length = length.loc["raw_fim_current"]
    baseline_repeat = repeated.loc["raw_fim_current"]
    baseline_worst = worst["raw_fim_current"]

    candidates = ("raw_fim_no_length", "raw_fim_log_replace", "raw_fim_log_add")
    rows = []
    for candidate in candidates:
        row = strat.loc[candidate]
        strict = length.loc[candidate]
        repeat = repeated.loc[candidate]
        checks = {
            "auc_improved": bool(row.auc > baseline.auc),
            "low_fpr_non_degrading": bool(
                row["tpr_at_0.01_fpr"] >= baseline["tpr_at_0.01_fpr"]
                or row.pauc_01 >= baseline.pauc_01
            ),
            "worst_language_within_0.005": bool(
                worst[candidate] >= baseline_worst - 0.005
            ),
            "repeated_mean_auc_improved": bool(
                repeat.auc_mean > baseline_repeat.auc_mean
            ),
            "wins_at_least_8_of_10_seeds": bool(
                int(repeat.auc_wins_vs_current) >= 8
            ),
            "visible_validation_used": False,
            "public_leaderboard_used": False,
        }
        score_only_frozen = bool(all(value for key, value in checks.items() if key not in {"visible_validation_used", "public_leaderboard_used"}))
        stage2_eligible = bool(
            score_only_frozen
            and strict.auc >= baseline_length.auc
            and strict["tpr_at_0.01_fpr"] >= baseline_length["tpr_at_0.01_fpr"]
        )
        rows.append(
            {
                "variant": candidate,
                "score_only_frozen": score_only_frozen,
                "stage2_eligible": stage2_eligible,
                **checks,
                "auc": float(row.auc),
                "auc_delta": float(row.auc - baseline.auc),
                "pauc_01": float(row.pauc_01),
                "tpr_at_0.01_fpr": float(row["tpr_at_0.01_fpr"]),
                "length_holdout_auc": float(strict.auc),
                "length_holdout_auc_delta": float(strict.auc - baseline_length.auc),
                "worst_language_auc": float(worst[candidate]),
                "repeated_auc_mean": float(repeat.auc_mean),
                "repeated_auc_std": float(repeat.auc_std),
                "auc_wins_vs_current": int(repeat.auc_wins_vs_current),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = read_train(args.dataset_root)
    variants = prepare_variants(frame)

    strat_metrics, strat_predictions = evaluate_once(
        frame, variants, stratified_splits(frame, SEED), "stratified", SEED
    )
    baseline_auc = float(
        strat_metrics.loc[strat_metrics.variant == "raw_fim_current", "auc"].iloc[0]
    )
    if abs(baseline_auc - EXPECTED_BASELINE_AUC) > BASELINE_AUC_TOLERANCE:
        raise RuntimeError(
            f"raw+FIM baseline fidelity failed: {baseline_auc:.6f}"
        )
    loo_metrics, _ = evaluate_once(
        frame, variants, leave_language_out_splits(frame), "leave_one_language_out", SEED
    )
    length_metrics_frame, _ = evaluate_once(
        frame, variants, length_holdout_splits(frame), "length_holdout", SEED
    )
    metrics = pd.concat(
        [strat_metrics, loo_metrics, length_metrics_frame], ignore_index=True
    )
    languages = language_metrics(frame, strat_predictions)
    repeated = repeated_cv(frame, variants)
    repeats = repeated_summary(repeated)
    gates = score_track_gates(metrics, languages, repeats)

    length_diag = {
        "token_count": low_fpr_metrics(frame.label, frame.token_count),
        "window_count": low_fpr_metrics(frame.label, frame.window_count),
        "token_count_member_mean": float(frame.loc[frame.label == 1, "token_count"].mean()),
        "token_count_nonmember_mean": float(frame.loc[frame.label == 0, "token_count"].mean()),
        "window_count_member_mean": float(frame.loc[frame.label == 1, "window_count"].mean()),
        "window_count_nonmember_mean": float(frame.loc[frame.label == 0, "window_count"].mean()),
    }
    frozen = gates.loc[gates.score_only_frozen, "variant"].tolist()
    stage2 = gates.loc[gates.stage2_eligible, "variant"].tolist()
    summary = {
        "status": "complete",
        "experiment": "stage1-track-a1-length-audit-v1",
        "source_dataset": "renta0426/stage1-raw-fim-submission-v1-output",
        "source_dataset_version": 1,
        "train_rows": len(frame),
        "train_shards": EXPECTED_SHARDS,
        "feature_counts": {
            "base": EXPECTED_BASE,
            "structure_raw": EXPECTED_STRUCTURE,
            "fim": EXPECTED_FIM,
            "current_total": EXPECTED_TOTAL,
        },
        "baseline_auc": baseline_auc,
        "expected_baseline_auc": EXPECTED_BASELINE_AUC,
        "length_diagnostic": length_diag,
        "score_only_frozen_candidates": frozen,
        "stage2_eligible_candidates": stage2,
        "interaction_search_performed": False,
        "visible_validation_loaded": False,
        "visible_validation_used": False,
        "public_leaderboard_used": False,
        "competition_submission_created": False,
        "gpu_or_tpu_compute_used": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    languages.to_csv(args.output_dir / "language_metrics.csv", index=False)
    repeated.to_csv(args.output_dir / "repeated_cv.csv", index=False)
    repeats.to_csv(args.output_dir / "repeated_cv_summary.csv", index=False)
    gates.to_csv(args.output_dir / "candidate_gates.csv", index=False)
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Aggregate-only stdout.  Never print sample IDs, paths, feature values, or
    # row-level predictions because this script is intended for the public bridge.
    print("POISONED_CHALICE_TRACK_A1_METRICS")
    print(metrics.to_csv(index=False).strip())
    print("POISONED_CHALICE_TRACK_A1_REPEATED")
    print(repeats.to_csv(index=False).strip())
    print("POISONED_CHALICE_TRACK_A1_GATES")
    print(gates.to_csv(index=False).strip())
    print("POISONED_CHALICE_TRACK_A1_SUMMARY")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
