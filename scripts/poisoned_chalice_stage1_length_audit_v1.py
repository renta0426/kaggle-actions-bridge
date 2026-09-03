"""Self-contained CPU-only Stage 1 raw+structure+FIM length audit.

This is a public bridge copy of the predeclared research analysis. It reads only
an already-created private Stage1 feature dataset on the ephemeral GitHub runner.
It performs no model forward pass, Kaggle compute, submission, hidden-label use,
or Public-LB tuning.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

CDF_SOURCES = ("score_loss_mean__max", "min_k_10__max", "min_kpp_10__max")
EXCLUDED_COLUMNS = {"sample_id", "language", "membership", "label", "split", "content"}
SEEDS = tuple(range(2027, 2037))
EXPECTED_ROWS = 10_000
EXPECTED_SHARDS = 40
EXPECTED_ROWS_PER_SHARD = 250
EXPECTED_BASE = 113
EXPECTED_STRUCTURE = 50
EXPECTED_FIM = 11
EXPECTED_BASELINE_AUC = 0.664524
BASELINE_TOLERANCE = 0.002
BASELINE = "raw_plus_fim"
CANDIDATES = (
    BASELINE,
    "no_explicit_length",
    "replace_raw_length_with_log",
    "plus_log_length",
    "plus_length_interactions",
)
DIAGNOSTICS = ("length_only_raw", "length_only_log")


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in EXCLUDED_COLUMNS and not column.startswith("cdf_")
    ]


def _pipeline(seed: int):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.2, max_iter=2_000, random_state=seed),
    )


def add_fold_cdf(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    columns: Sequence[str] = CDF_SOURCES,
    min_language_rows: int = 20,
) -> pd.DataFrame:
    output = target.copy()
    nonmember = reference[reference.label == 0]
    if nonmember.empty:
        raise ValueError("CDF reference has no non-members")
    for column in columns:
        global_values = np.sort(nonmember[column].dropna().to_numpy(float))
        language_values = {
            language: np.sort(group[column].dropna().to_numpy(float))
            for language, group in nonmember.groupby("language")
        }
        calibrated = []
        for language, value in zip(output.language, output[column]):
            values = language_values.get(language, np.array([], dtype=float))
            if len(values) < min_language_rows:
                values = global_values
            calibrated.append(np.searchsorted(values, value, side="right") / len(values))
        output[f"cdf_{column}"] = calibrated
    return output


def cross_fit_meta(
    frame: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    base_features: Sequence[str],
    seed: int,
) -> np.ndarray:
    cdf_features = [f"cdf_{column}" for column in CDF_SOURCES if column in base_features]
    model_features = list(base_features) + cdf_features
    prediction = np.full(len(frame), np.nan, dtype=float)
    seen = np.zeros(len(frame), dtype=int)
    labels = frame.label.astype(int).to_numpy()
    for fold, (fit_index, hold_index) in enumerate(splits):
        fit = add_fold_cdf(frame.iloc[fit_index], frame.iloc[fit_index])
        hold = add_fold_cdf(frame.iloc[fit_index], frame.iloc[hold_index])
        model = _pipeline(seed + fold)
        model.fit(fit[model_features], labels[fit_index])
        prediction[hold_index] = model.predict_proba(hold[model_features])[:, 1]
        seen[hold_index] += 1
    if not np.isfinite(prediction).all() or not np.all(seen == 1):
        raise RuntimeError(
            f"OOF coverage invalid: missing={(seen == 0).sum()} repeated={(seen > 1).sum()}"
        )
    return prediction


def stratified_splits(frame: pd.DataFrame, n_splits: int, seed: int):
    from sklearn.model_selection import StratifiedKFold

    strata = frame.language.astype(str) + "_" + frame.label.astype(int).astype(str)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(frame, strata))


def leave_one_language_out_splits(frame: pd.DataFrame):
    indices = np.arange(len(frame))
    language = frame.language.to_numpy()
    return [
        (indices[language != name], indices[language == name])
        for name in sorted(frame.language.unique())
    ]


def length_holdout_splits(frame: pd.DataFrame, bins: int = 5):
    bucket = pd.qcut(frame.token_count, bins, labels=False, duplicates="drop")
    if bucket.isna().any() or bucket.nunique() < 2:
        raise ValueError("token_count does not support at least two length holdouts")
    indices = np.arange(len(frame))
    return [
        (indices[bucket.to_numpy() != value], indices[bucket.to_numpy() == value])
        for value in sorted(bucket.unique())
    ]


def low_fpr_metrics(y_true: Sequence[int], score: Sequence[float]) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score, roc_curve

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


def conservative_detection_set(
    sample_ids: Sequence[str],
    y_true: Sequence[int],
    score: Sequence[float],
    target_fpr: float = 0.01,
) -> set[str]:
    ids = np.asarray(sample_ids)
    y = np.asarray(y_true, dtype=int)
    values = np.asarray(score, dtype=float)
    negatives = np.sort(values[y == 0])[::-1]
    allowed = max(1, int(math.floor(target_fpr * len(negatives))))
    threshold = negatives[allowed - 1]
    return set(ids[(y == 1) & (values > threshold)])


def load_train_frame(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("train_10k/parts/features.part*.parquet"))
    if len(paths) != EXPECTED_SHARDS:
        raise RuntimeError(f"expected {EXPECTED_SHARDS} train shards, found {len(paths)}")
    parts = []
    for path in paths:
        frame = pd.read_parquet(path)
        if len(frame) != EXPECTED_ROWS_PER_SHARD:
            raise RuntimeError(f"unexpected shard rows: {path.name} -> {len(frame)}")
        parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    if len(frame) != EXPECTED_ROWS or not frame.sample_id.is_unique:
        raise RuntimeError("Stage1 train feature coverage mismatch")
    required = {"sample_id", "language", "label", "token_count", "window_count"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Stage1 train feature schema missing: {sorted(missing)}")
    frame = frame.sort_values("sample_id").reset_index(drop=True)
    frame["label"] = frame.label.astype(int)
    if set(frame.label.unique()) != {0, 1}:
        raise RuntimeError("Stage1 labels are not binary")
    counts = frame.groupby(["language", "label"]).size()
    if len(counts) != 10 or not counts.eq(1000).all():
        raise RuntimeError("unexpected language/label balance")
    return frame


def feature_schema(frame: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    numeric = numeric_feature_columns(frame)
    base = [column for column in numeric if not column.startswith(("ast_", "fim_"))]
    structure = [column for column in numeric if column.startswith("ast_")]
    fim = [column for column in numeric if column.startswith("fim_")]
    counts = {
        "base": len(base),
        "structure": len(structure),
        "fim": len(fim),
        "total": len(numeric),
    }
    expected = {
        "base": EXPECTED_BASE,
        "structure": EXPECTED_STRUCTURE,
        "fim": EXPECTED_FIM,
        "total": EXPECTED_BASE + EXPECTED_STRUCTURE + EXPECTED_FIM,
    }
    if counts != expected:
        raise RuntimeError(f"Stage1 feature schema changed: {counts}")
    return numeric, counts


def with_log_length(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["audit_log_token_count"] = np.log1p(output.token_count.to_numpy(float))
    output["audit_log_window_count"] = np.log1p(output.window_count.to_numpy(float))
    return output


def simple_features(all_features: Sequence[str], variant: str) -> list[str]:
    all_features = list(all_features)
    without_length = [
        column for column in all_features if column not in {"token_count", "window_count"}
    ]
    if variant == BASELINE:
        return all_features
    if variant == "no_explicit_length":
        return without_length
    if variant == "replace_raw_length_with_log":
        return without_length + ["audit_log_token_count", "audit_log_window_count"]
    if variant == "plus_log_length":
        return all_features + ["audit_log_token_count", "audit_log_window_count"]
    if variant == "length_only_raw":
        return ["token_count", "window_count"]
    if variant == "length_only_log":
        return ["audit_log_token_count", "audit_log_window_count"]
    raise KeyError(variant)


def cross_fit_interactions(
    frame: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    all_features: Sequence[str],
    seed: int,
) -> np.ndarray:
    labels = frame.label.to_numpy(int)
    prediction = np.full(len(frame), np.nan, dtype=float)
    seen = np.zeros(len(frame), dtype=int)
    all_features = list(all_features)
    sources = [column for column in all_features if column not in {"token_count", "window_count"}]
    cdf_features = [f"cdf_{column}" for column in CDF_SOURCES if column in all_features]
    base_model_features = all_features + cdf_features + [
        "audit_log_token_count",
        "audit_log_window_count",
    ]
    interaction_names = [f"audit_int_token__{column}" for column in sources] + [
        f"audit_int_window__{column}" for column in sources
    ]
    model_features = base_model_features + interaction_names

    for fold, (fit_index, hold_index) in enumerate(splits):
        fit = with_log_length(add_fold_cdf(frame.iloc[fit_index], frame.iloc[fit_index]))
        hold = with_log_length(add_fold_cdf(frame.iloc[fit_index], frame.iloc[hold_index]))
        token_center = float(fit.audit_log_token_count.mean())
        window_center = float(fit.audit_log_window_count.mean())
        fit_token = fit.audit_log_token_count.to_numpy(float) - token_center
        hold_token = hold.audit_log_token_count.to_numpy(float) - token_center
        fit_window = fit.audit_log_window_count.to_numpy(float) - window_center
        hold_window = hold.audit_log_window_count.to_numpy(float) - window_center

        fit_interactions = {
            f"audit_int_token__{column}": fit[column].to_numpy(float) * fit_token
            for column in sources
        }
        fit_interactions.update(
            {
                f"audit_int_window__{column}": fit[column].to_numpy(float) * fit_window
                for column in sources
            }
        )
        hold_interactions = {
            f"audit_int_token__{column}": hold[column].to_numpy(float) * hold_token
            for column in sources
        }
        hold_interactions.update(
            {
                f"audit_int_window__{column}": hold[column].to_numpy(float) * hold_window
                for column in sources
            }
        )
        fit = pd.concat([fit, pd.DataFrame(fit_interactions, index=fit.index)], axis=1)
        hold = pd.concat([hold, pd.DataFrame(hold_interactions, index=hold.index)], axis=1)

        model = _pipeline(seed + fold)
        model.fit(fit[model_features], labels[fit_index])
        prediction[hold_index] = model.predict_proba(hold[model_features])[:, 1]
        seen[hold_index] += 1

    if not np.isfinite(prediction).all() or not np.all(seen == 1):
        raise RuntimeError("interaction OOF coverage invalid")
    return prediction


def predict(
    frame: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    all_features: Sequence[str],
    variant: str,
    seed: int,
) -> np.ndarray:
    if variant == "plus_length_interactions":
        return cross_fit_interactions(frame, splits, all_features, seed)
    work = (
        with_log_length(frame)
        if variant in {"replace_raw_length_with_log", "plus_log_length", "length_only_log"}
        else frame
    )
    return cross_fit_meta(work, splits, simple_features(all_features, variant), seed)


def per_language_metrics(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    rows = []
    work = frame[["language", "label"]].copy()
    work["score"] = score
    for language, group in work.groupby("language", sort=True):
        rows.append({"language": language} | low_fpr_metrics(group.label, group.score))
    return pd.DataFrame(rows)


def evaluate_variant(
    frame: pd.DataFrame,
    all_features: Sequence[str],
    variant: str,
    seed: int,
) -> tuple[list[dict[str, object]], pd.DataFrame, set[str]]:
    split_specs = {
        "stratified": stratified_splits(frame, 5, seed),
        "leave_one_language_out": leave_one_language_out_splits(frame),
        "length_holdout": length_holdout_splits(frame, 5),
    }
    rows = []
    language_frame = None
    detection: set[str] = set()
    for split_name, splits in split_specs.items():
        score = predict(frame, splits, all_features, variant, seed)
        rows.append({"split": split_name, "variant": variant} | low_fpr_metrics(frame.label, score))
        if split_name == "stratified":
            language_frame = per_language_metrics(frame, score)
            detection = conservative_detection_set(frame.sample_id, frame.label, score, 0.01)
    assert language_frame is not None
    language_frame.insert(0, "variant", variant)
    return rows, language_frame, detection


def repeated_cv(
    frame: pd.DataFrame,
    all_features: Sequence[str],
    variant: str,
) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        score = predict(frame, stratified_splits(frame, 5, seed), all_features, variant, seed)
        rows.append({"variant": variant, "seed": seed} | low_fpr_metrics(frame.label, score))
    return pd.DataFrame(rows)


def summarize_repeated(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in frame.groupby("variant", sort=False):
        rows.append(
            {
                "variant": variant,
                "seeds": len(group),
                "auc_mean": float(group.auc.mean()),
                "auc_std": float(group.auc.std(ddof=0)),
                "auc_min": float(group.auc.min()),
                "auc_max": float(group.auc.max()),
                "tpr1_mean": float(group["tpr_at_0.01_fpr"].mean()),
                "tpr1_std": float(group["tpr_at_0.01_fpr"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def choose_score_candidate(
    split_metrics: pd.DataFrame,
    language_metrics: pd.DataFrame,
    repeated: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    canonical = split_metrics.loc[split_metrics.split == "stratified"].set_index("variant")
    length = split_metrics.loc[split_metrics.split == "length_holdout"].set_index("variant")
    repeat = repeated.set_index("variant")
    worst = language_metrics.groupby("variant").auc.min()
    base = canonical.loc[BASELINE]
    base_repeat = repeat.loc[BASELINE]
    base_worst = float(worst[BASELINE])
    rows = []
    for variant in CANDIDATES:
        current = canonical.loc[variant]
        current_repeat = repeat.loc[variant]
        checks = {
            "repeated_auc_gain_ge_0.001": (
                bool(current_repeat.auc_mean >= base_repeat.auc_mean + 0.001)
                if variant != BASELINE
                else True
            ),
            "worst_language_within_0.01": bool(float(worst[variant]) >= base_worst - 0.01),
            "tpr1_not_worse_by_more_than_0.005": bool(
                current["tpr_at_0.01_fpr"] >= base["tpr_at_0.01_fpr"] - 0.005
            ),
        }
        rows.append(
            {
                "variant": variant,
                "eligible_score_only": bool(all(checks.values())),
                **checks,
                "auc": float(current.auc),
                "tpr1": float(current["tpr_at_0.01_fpr"]),
                "repeated_auc_mean": float(current_repeat.auc_mean),
                "repeated_auc_std": float(current_repeat.auc_std),
                "worst_language_auc": float(worst[variant]),
                "length_holdout_auc": float(length.loc[variant].auc),
                "length_holdout_tpr1": float(length.loc[variant]["tpr_at_0.01_fpr"]),
                "length_robust_vs_baseline": bool(
                    length.loc[variant].auc >= length.loc[BASELINE].auc
                    and length.loc[variant]["tpr_at_0.01_fpr"]
                    >= length.loc[BASELINE]["tpr_at_0.01_fpr"]
                ),
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[(table.variant != BASELINE) & table.eligible_score_only]
    selected = BASELINE
    if not eligible.empty:
        selected = str(
            eligible.sort_values(
                ["repeated_auc_mean", "tpr1", "variant"],
                ascending=[False, False, True],
            ).iloc[0].variant
        )
    summary = {
        "selected": selected,
        "baseline": BASELINE,
        "selection_rule": (
            "highest repeated_auc_mean among non-baseline variants with >=0.001 mean AUC gain, "
            "worst-language AUC within 0.01, and canonical TPR@1% no more than 0.005 below baseline"
        ),
    }
    return summary, table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = load_train_frame(args.dataset_root.resolve())
    all_features, schema = feature_schema(frame)

    split_rows: list[dict[str, object]] = []
    language_rows = []
    detections: dict[str, set[str]] = {}
    for variant in CANDIDATES + DIAGNOSTICS:
        rows, language, detection = evaluate_variant(frame, all_features, variant, SEEDS[0])
        split_rows.extend(rows)
        language_rows.append(language)
        detections[variant] = detection

    split_metrics = pd.DataFrame(split_rows)
    language_metrics = pd.concat(language_rows, ignore_index=True)
    baseline_auc = float(
        split_metrics.loc[
            (split_metrics.split == "stratified") & (split_metrics.variant == BASELINE), "auc"
        ].iloc[0]
    )
    if abs(baseline_auc - EXPECTED_BASELINE_AUC) > BASELINE_TOLERANCE:
        raise RuntimeError(
            f"raw_plus_fim reproduction failed: {baseline_auc:.6f} vs "
            f"{EXPECTED_BASELINE_AUC:.6f}±{BASELINE_TOLERANCE:.6f}"
        )

    repeated = pd.concat(
        [repeated_cv(frame, all_features, variant) for variant in CANDIDATES],
        ignore_index=True,
    )
    repeated_summary = summarize_repeated(repeated)
    selection, selection_table = choose_score_candidate(
        split_metrics, language_metrics, repeated_summary
    )

    novelty_rows = []
    union_all = set().union(*detections.values())
    for variant, values in detections.items():
        others = set().union(*(value for name, value in detections.items() if name != variant))
        novelty_rows.append(
            {
                "variant": variant,
                "true_positives": len(values),
                "unique_true_positives": len(values - others),
                "union_true_positives": len(union_all),
            }
        )
    novelty = pd.DataFrame(novelty_rows)

    split_metrics.to_csv(output / "split_metrics.csv", index=False)
    language_metrics.to_csv(output / "language_metrics.csv", index=False)
    repeated.to_csv(output / "repeated_cv.csv", index=False)
    repeated_summary.to_csv(output / "repeated_cv_summary.csv", index=False)
    selection_table.to_csv(output / "candidate_selection.csv", index=False)
    novelty.to_csv(output / "unique_true_positives.csv", index=False)

    summary = {
        "status": "complete",
        "rows": len(frame),
        "feature_schema": schema,
        "expected_baseline_auc": EXPECTED_BASELINE_AUC,
        "baseline_auc": baseline_auc,
        "baseline_reproduced": True,
        "score_candidate": selection,
        "mellum_artifacts_read": False,
        "mellum_labels_used": False,
        "hidden_stage1_validation_labels_used": False,
        "sample_id_used_as_feature": False,
        "public_leaderboard_used_for_selection": False,
        "model_forward_pass": False,
        "candidate_set_predeclared": list(CANDIDATES),
        "diagnostics": list(DIAGNOSTICS),
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Only aggregate, non-row-level outputs are printed by the bridge workflow.
    print("STAGE1_LENGTH_AUDIT_METRICS")
    print(split_metrics.to_csv(index=False).strip())
    print("STAGE1_LENGTH_AUDIT_REPEATED")
    print(repeated_summary.to_csv(index=False).strip())
    print("STAGE1_LENGTH_AUDIT_SELECTION")
    print(selection_table.to_csv(index=False).strip())
    print("STAGE1_LENGTH_AUDIT_NOVELTY")
    print(novelty.to_csv(index=False).strip())
    print("STAGE1_LENGTH_AUDIT_SUMMARY")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
