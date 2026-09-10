"""CPU-only clean-room evaluation primitives for Starter++ feature caches."""

from __future__ import annotations

from collections import defaultdict
from hashlib import blake2b
from typing import Iterable, Sequence
import math
import re

import numpy as np
import pandas as pd


CDF_SOURCES = ("score_loss_mean__max", "min_k_10__max", "min_kpp_10__max")
EXCLUDED_COLUMNS = {"sample_id", "language", "membership", "label", "split", "content"}


def feature_families(columns: Iterable[str]) -> dict[str, list[str]]:
    families = {name: [] for name in (
        "length", "loss", "mink", "minkpp", "rank_confidence", "local_span", "surp"
    )}
    for column in columns:
        if column in ("token_count", "window_count"):
            families["length"].append(column)
        elif column.startswith("score_loss_"):
            families["loss"].append(column)
        elif column.startswith("min_kpp_"):
            families["minkpp"].append(column)
        elif column.startswith("min_k_"):
            families["mink"].append(column)
        elif column.startswith(("correct_z_", "mean_log_rank", "top1_", "top5_", "margin_", "entropy_")):
            families["rank_confidence"].append(column)
        elif column.startswith(("best_local_", "longest_top1_")):
            families["local_span"].append(column)
        elif column.startswith("surp_"):
            families["surp"].append(column)
    assigned = {c for values in families.values() for c in values}
    numeric = [c for c in columns if c not in EXCLUDED_COLUMNS and not c.startswith("cdf_")]
    unknown = set(numeric).difference(assigned)
    if unknown:
        raise ValueError(f"unassigned numeric features: {sorted(unknown)}")
    return families


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.select_dtypes(include=[np.number]).columns
            if c not in EXCLUDED_COLUMNS and not c.startswith("cdf_")]


def rank_percentile(values: Sequence[float]) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def attack_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    scores = {
        "loss_multiwindow": frame["score_loss_mean__max"].to_numpy(float),
        "loss_window_mean": frame["score_loss_mean__mean"].to_numpy(float),
        "mink_05": frame["min_k_05__max"].to_numpy(float),
        "mink_10": frame["min_k_10__max"].to_numpy(float),
        "mink_20": frame["min_k_20__max"].to_numpy(float),
        "minkpp_rawselected_10": frame["min_kpp_10__max"].to_numpy(float),
        "surp_10": frame["surp_10__max"].to_numpy(float),
        "local_64": frame["best_local_64__max"].to_numpy(float),
        "top1_rate": frame["top1_rate__max"].to_numpy(float),
        "neg_mean_log_rank": -frame["mean_log_rank__mean"].to_numpy(float),
    }
    if "min_kpp_zselect_10__max" in frame:
        scores["minkpp_zselected_10"] = frame["min_kpp_zselect_10__max"].to_numpy(float)
    return scores


def add_fold_cdf(reference: pd.DataFrame, target: pd.DataFrame,
                 columns: Sequence[str] = CDF_SOURCES, min_language_rows: int = 20) -> pd.DataFrame:
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


def cross_fit_meta(frame: pd.DataFrame, splits: Sequence[tuple[np.ndarray, np.ndarray]],
                   base_features: Sequence[str], seed: int) -> np.ndarray:
    cdf_features = [f"cdf_{c}" for c in CDF_SOURCES if c in base_features]
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
        raise RuntimeError(f"OOF coverage invalid: missing={(seen == 0).sum()}, repeated={(seen > 1).sum()}")
    return prediction


def exact_final_ensemble(frame: pd.DataFrame, meta_prediction: Sequence[float]) -> np.ndarray:
    return (
        0.55 * rank_percentile(meta_prediction)
        + 0.20 * rank_percentile(frame["score_loss_mean__max"])
        + 0.20 * rank_percentile(frame["min_k_10__max"])
        + 0.05 * rank_percentile(frame["min_kpp_10__max"])
    )


def stratified_splits(frame: pd.DataFrame, n_splits: int, seed: int):
    from sklearn.model_selection import StratifiedKFold
    strata = frame.language.astype(str) + "_" + frame.label.astype(int).astype(str)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(frame, strata))


def grouped_splits(frame: pd.DataFrame, groups: Sequence[object], n_splits: int, seed: int):
    from sklearn.model_selection import StratifiedGroupKFold
    strata = frame.language.astype(str) + "_" + frame.label.astype(int).astype(str)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(frame, strata, groups=np.asarray(groups)))


def leave_one_language_out_splits(frame: pd.DataFrame):
    indices = np.arange(len(frame))
    return [(indices[frame.language.to_numpy() != language], indices[frame.language.to_numpy() == language])
            for language in sorted(frame.language.unique())]


def length_holdout_splits(frame: pd.DataFrame, bins: int = 5):
    bucket = pd.qcut(frame.token_count, bins, labels=False, duplicates="drop")
    if bucket.isna().any() or bucket.nunique() < 2:
        raise ValueError("token_count does not support at least two length holdouts")
    indices = np.arange(len(frame))
    return [(indices[bucket.to_numpy() != value], indices[bucket.to_numpy() == value])
            for value in sorted(bucket.unique())], bucket.astype(int).to_numpy()


def low_fpr_metrics(y_true: Sequence[int], score: Sequence[float]) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score, roc_curve
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    fpr, tpr, _ = roc_curve(y, score)
    result = {"auc": float(roc_auc_score(y, score)),
              "pauc_01": float(roc_auc_score(y, score, max_fpr=0.01))}
    for rate in (0.001, 0.005, 0.01, 0.02):
        result[f"tpr_at_{rate:g}_fpr"] = float(tpr[fpr <= rate].max())
    return result


def conservative_detection_set(sample_ids: Sequence[str], y_true: Sequence[int],
                               score: Sequence[float], target_fpr: float = 0.01) -> set[str]:
    ids = np.asarray(sample_ids)
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    negatives = np.sort(score[y == 0])[::-1]
    allowed = max(1, int(math.floor(target_fpr * len(negatives))))
    threshold = negatives[allowed - 1]
    return set(ids[(y == 1) & (score > threshold)])


def overlap_tables(sample_ids: Sequence[str], y_true: Sequence[int],
                   scores: dict[str, Sequence[float]], target_fpr: float = 0.01):
    sets = {name: conservative_detection_set(sample_ids, y_true, score, target_fpr)
            for name, score in scores.items()}
    jaccard_rows = []
    unique_rows = []
    for left, left_set in sets.items():
        others = set().union(*(value for name, value in sets.items() if name != left))
        unique_rows.append({"attack": left, "true_positives": len(left_set),
                            "unique_true_positives": len(left_set - others)})
        for right, right_set in sets.items():
            union = left_set | right_set
            jaccard_rows.append({"left": left, "right": right,
                                  "jaccard": len(left_set & right_set) / len(union) if union else 1.0})
    return pd.DataFrame(jaccard_rows), pd.DataFrame(unique_rows), sets


def stratified_bootstrap_metrics(frame: pd.DataFrame, score: Sequence[float],
                                 replicates: int, seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    work = frame[["language", "label"]].copy()
    work["score"] = np.asarray(score, dtype=float)
    strata = [group.index.to_numpy() for _, group in work.groupby(["language", "label"])]
    rows = []
    for _ in range(replicates):
        sampled = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in strata])
        rows.append(low_fpr_metrics(work.label.to_numpy()[sampled], work.score.to_numpy()[sampled]))
    result = {}
    for metric in rows[0]:
        values = np.array([row[metric] for row in rows])
        result[metric] = {"mean": float(values.mean()), "lower_95": float(np.quantile(values, 0.025)),
                          "upper_95": float(np.quantile(values, 0.975))}
    return result


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\s]", re.UNICODE)


def token_shingles(content: str, size: int = 5) -> set[str]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(content)]
    if len(tokens) < size:
        return {"\x1f".join(tokens)} if tokens else {""}
    return {"\x1f".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


def _minhash(shingles: set[str], num_perm: int) -> np.ndarray:
    prime = np.uint64(18_446_744_073_709_551_557)
    hashed = []
    for shingle in shingles:
        digest = blake2b(shingle.encode("utf-8"), digest_size=16).digest()
        hashed.append((int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:], "little") | 1))
    signature = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    permutation = np.arange(num_perm, dtype=np.uint64)
    for first, second in hashed:
        values = (np.uint64(first) + permutation * np.uint64(second)) % prime
        signature = np.minimum(signature, values)
    return signature


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value
    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def minhash_groups(contents: Sequence[str], num_perm: int = 64, bands: int = 16,
                   shingle_size: int = 5, jaccard_threshold: float = 0.6) -> tuple[np.ndarray, dict[str, int]]:
    if num_perm % bands:
        raise ValueError("num_perm must be divisible by bands")
    shingle_sets = [token_shingles(content, shingle_size) for content in contents]
    signatures = [_minhash(shingles, num_perm) for shingles in shingle_sets]
    rows_per_band = num_perm // bands
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    for index, signature in enumerate(signatures):
        for band in range(bands):
            block = signature[band * rows_per_band:(band + 1) * rows_per_band]
            buckets[(band, block.tobytes())].append(index)
    candidates: set[tuple[int, int]] = set()
    for values in buckets.values():
        if len(values) > 1:
            candidates.update((values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values)))
    union = _UnionFind(len(contents))
    accepted = 0
    for left, right in candidates:
        a, b = shingle_sets[left], shingle_sets[right]
        similarity = len(a & b) / len(a | b)
        if similarity >= jaccard_threshold:
            union.union(left, right)
            accepted += 1
    roots = np.array([union.find(index) for index in range(len(contents))])
    root_to_group = {root: group for group, root in enumerate(sorted(set(roots)))}
    groups = np.array([root_to_group[root] for root in roots])
    sizes = pd.Series(groups).value_counts()
    diagnostics = {"samples": len(contents), "candidate_pairs": len(candidates),
                   "accepted_pairs": accepted, "groups": int(sizes.size),
                   "non_singleton_groups": int((sizes > 1).sum()),
                   "largest_group": int(sizes.max())}
    return groups, diagnostics
