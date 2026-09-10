#!/usr/bin/env python3
"""P1-03 post-5k development audit: fusion, content controls, portable hidden scalars.

This analysis is development-only. It performs no target-model forward pass and no
competition submission. The current 5k hidden-state cohort has already been seen;
therefore fusion weights and portable-scalar results produced here are exploratory
until frozen and confirmed on a fresh disjoint cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

HIDDEN_OOF_SHA256 = "c6d35c24087eac4809678bb5cae3d02fef630d07eb6f4033c50b8e285ce56fb4"
CACHE_MANIFEST_SHA256 = "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa"
DATASET_ID = "Poisoned-Chalice/ICSE-2027-public"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
EXPECTED_PUBLIC_ROWS = {"Go": 10000, "Java": 10000, "Python": 10000, "Ruby": 10000, "Rust": 9040}
EXPECTED_HIDDEN_ROWS = 5000
EXPECTED_STAGE1_ROWS = 10000
SEED = 20260909
STAGE1_SEED = 2027
MEAN_TOP5 = "mean_only_top5_ensemble"
CALLER_TOP5 = "lumia_caller_literal_top5_ensemble"
HELPER_TOP5 = "lumia_helper_language_aware_top5_ensemble"
POOLING_VARIANTS = ("mean", "caller_weighted", "helper_weighted")
CDF_SOURCES = ("score_loss_mean__max", "min_k_10__max", "min_kpp_10__max")
EXCLUDED_COLUMNS = {"sample_id", "language", "membership", "label", "split", "content"}
FUSION_HIDDEN_WEIGHTS = (0.25, 0.50, 0.75)
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260910


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def low_fpr_metrics(y_true: Sequence[int], score: Sequence[float]) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score, roc_curve
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    fpr, tpr, _ = roc_curve(y, s)
    out = {"auc": float(roc_auc_score(y, s)), "pauc_01": float(roc_auc_score(y, s, max_fpr=0.01))}
    for rate in (0.001, 0.005, 0.01, 0.02):
        out[f"tpr_at_{rate:g}_fpr"] = float(tpr[fpr <= rate].max())
    return out


def conservative_detection_set(ids: Sequence[str], y_true: Sequence[int], score: Sequence[float], fpr: float = 0.01) -> set[str]:
    ids = np.asarray(ids, dtype=object)
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    negatives = np.sort(s[y == 0])[::-1]
    allowed = max(1, int(math.floor(fpr * len(negatives))))
    threshold = negatives[allowed - 1]
    return set(ids[(y == 1) & (s > threshold)])


def stratified_indices(frame: pd.DataFrame, seed: int = SEED):
    from sklearn.model_selection import StratifiedKFold
    strata = frame.language.astype(str) + "_" + frame.label.astype(int).astype(str)
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(frame, strata))


def paired_bootstrap(frame: pd.DataFrame, left: np.ndarray, right: np.ndarray, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> dict:
    rng = np.random.default_rng(seed)
    cells = [g.index.to_numpy() for _, g in frame.groupby(["language", "label"], sort=True)]
    point_left = low_fpr_metrics(frame.label, left)
    point_right = low_fpr_metrics(frame.label, right)
    keys = ("auc", "pauc_01", "tpr_at_0.001_fpr", "tpr_at_0.005_fpr", "tpr_at_0.01_fpr", "tpr_at_0.02_fpr")
    samples = {key: [] for key in keys}
    y_all = frame.label.to_numpy(int)
    for _ in range(replicates):
        draw = np.concatenate([rng.choice(cell, size=len(cell), replace=True) for cell in cells])
        lm = low_fpr_metrics(y_all[draw], left[draw])
        rm = low_fpr_metrics(y_all[draw], right[draw])
        for key in keys:
            samples[key].append(lm[key] - rm[key])
    out = {"replicates": replicates, "seed": seed, "left": point_left, "right": point_right, "delta": {}}
    for key in keys:
        values = np.asarray(samples[key], dtype=float)
        out["delta"][key] = {"point": float(point_left[key] - point_right[key]), "lower_95": float(np.quantile(values, 0.025)), "upper_95": float(np.quantile(values, 0.975))}
    return out


def per_language(frame: pd.DataFrame, score: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for language, group in frame.assign(_score=score).groupby("language", sort=True):
        out[str(language)] = low_fpr_metrics(group.label, group._score)
    return out


def locate_exact(root: Path, filename: str, sha256: str) -> Path:
    matches = [path for path in root.rglob(filename) if path.is_file() and sha256_file(path) == sha256]
    if len(matches) != 1:
        raise RuntimeError(f"exact {filename} resolution failed: {len(matches)} matches")
    return matches[0]


def load_hidden_oof(path: Path) -> pd.DataFrame:
    if sha256_file(path) != HIDDEN_OOF_SHA256:
        raise RuntimeError("hidden OOF SHA-256 changed")
    frame = pd.read_csv(path)
    required = {"sample_id", "language", "label", MEAN_TOP5, CALLER_TOP5, HELPER_TOP5}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"hidden OOF columns changed: missing={sorted(required - set(frame.columns))}")
    if len(frame) != EXPECTED_HIDDEN_ROWS or frame.sample_id.duplicated().any():
        raise RuntimeError("hidden OOF coverage changed")
    frame["label"] = frame.label.astype(int)
    counts = frame.groupby(["language", "label"]).size().to_dict()
    expected = {(language, label): 500 for language in LANGUAGES for label in (0, 1)}
    if counts != expected:
        raise RuntimeError(f"hidden OOF balance changed: {counts}")
    return frame.reset_index(drop=True)


# Frozen Stage1 plus_length_interactions reproduction.
def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.select_dtypes(include=[np.number]).columns if c not in EXCLUDED_COLUMNS and not c.startswith("cdf_")]


def load_stage1_train(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("train_10k/parts/features.part*.parquet"))
    if len(paths) != 40:
        raise RuntimeError(f"expected 40 Stage1 train shards, found {len(paths)}")
    parts = []
    for path in paths:
        part = pd.read_parquet(path)
        if len(part) != 250:
            raise RuntimeError(f"unexpected Stage1 shard size {path}: {len(part)}")
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    if len(frame) != EXPECTED_STAGE1_ROWS or not frame.sample_id.is_unique:
        raise RuntimeError("Stage1 train coverage changed")
    frame["label"] = frame.label.astype(int)
    counts = frame.groupby(["language", "label"]).size()
    if len(counts) != 10 or not counts.eq(1000).all():
        raise RuntimeError("Stage1 language/label balance changed")
    return frame


def add_fold_cdf(reference: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    nonmember = reference[reference.label == 0]
    for column in CDF_SOURCES:
        global_values = np.sort(nonmember[column].dropna().to_numpy(float))
        language_values = {language: np.sort(group[column].dropna().to_numpy(float)) for language, group in nonmember.groupby("language")}
        values = []
        for language, value in zip(out.language, out[column]):
            ref = language_values.get(language, np.array([], dtype=float))
            if len(ref) < 20:
                ref = global_values
            values.append(np.searchsorted(ref, value, side="right") / len(ref))
        out[f"cdf_{column}"] = values
    return out


def stage1_pipeline(seed: int):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.2, max_iter=2000, random_state=seed))


def stage1_interaction_oof(frame: pd.DataFrame) -> np.ndarray:
    all_features = numeric_feature_columns(frame)
    base = [c for c in all_features if not c.startswith(("ast_", "fim_"))]
    structure = [c for c in all_features if c.startswith("ast_")]
    fim = [c for c in all_features if c.startswith("fim_")]
    if (len(base), len(structure), len(fim), len(all_features)) != (113, 50, 11, 174):
        raise RuntimeError("Stage1 feature schema changed")
    sources = [c for c in all_features if c not in {"token_count", "window_count"}]
    cdf_features = [f"cdf_{c}" for c in CDF_SOURCES]
    base_features = all_features + cdf_features + ["audit_log_token_count", "audit_log_window_count"]
    int_names = [f"audit_int_token__{c}" for c in sources] + [f"audit_int_window__{c}" for c in sources]
    model_features = base_features + int_names
    labels = frame.label.to_numpy(int)
    pred = np.full(len(frame), np.nan)
    seen = np.zeros(len(frame), dtype=int)
    from sklearn.model_selection import StratifiedKFold
    strata = frame.language.astype(str) + "_" + frame.label.astype(str)
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=STAGE1_SEED).split(frame, strata)
    for fold, (fit_idx, hold_idx) in enumerate(splits):
        fit = add_fold_cdf(frame.iloc[fit_idx], frame.iloc[fit_idx])
        hold = add_fold_cdf(frame.iloc[fit_idx], frame.iloc[hold_idx])
        for work in (fit, hold):
            work["audit_log_token_count"] = np.log1p(work.token_count.to_numpy(float))
            work["audit_log_window_count"] = np.log1p(work.window_count.to_numpy(float))
        token_center = float(fit.audit_log_token_count.mean())
        window_center = float(fit.audit_log_window_count.mean())
        for work, token_delta, window_delta in ((fit, fit.audit_log_token_count.to_numpy(float) - token_center, fit.audit_log_window_count.to_numpy(float) - window_center), (hold, hold.audit_log_token_count.to_numpy(float) - token_center, hold.audit_log_window_count.to_numpy(float) - window_center)):
            ints = {f"audit_int_token__{c}": work[c].to_numpy(float) * token_delta for c in sources}
            ints.update({f"audit_int_window__{c}": work[c].to_numpy(float) * window_delta for c in sources})
            work[list(ints)] = pd.DataFrame(ints, index=work.index)
        model = stage1_pipeline(STAGE1_SEED + fold)
        model.fit(fit[model_features], labels[fit_idx])
        pred[hold_idx] = model.predict_proba(hold[model_features])[:, 1]
        seen[hold_idx] += 1
    if not np.isfinite(pred).all() or not np.all(seen == 1):
        raise RuntimeError("Stage1 OOF coverage invalid")
    auc = low_fpr_metrics(labels, pred)["auc"]
    if abs(auc - 0.68013844) > 0.002:
        raise RuntimeError(f"Stage1 plus_length_interactions reproduction gate failed: {auc}")
    return pred


def rank01(values: Sequence[float]) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy(float)


def fusion_audit(stage1: pd.DataFrame, hidden: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    merged = hidden.merge(stage1[["sample_id", "language", "label", "stage1_plus_length_interactions"]], on="sample_id", how="inner", suffixes=("", "_stage1"), validate="one_to_one")
    if merged.sample_id.duplicated().any():
        raise RuntimeError("fusion join produced duplicates")
    if not (merged.language == merged.language_stage1).all() or not (merged.label == merged.label_stage1).all():
        raise RuntimeError("fusion join language/label mismatch")
    merged = merged.drop(columns=["language_stage1", "label_stage1"]).reset_index(drop=True)
    if len(merged) < 200:
        raise RuntimeError(f"Stage1/hidden intersection too small for audit: {len(merged)}")
    score_names = [MEAN_TOP5, "stage1_plus_length_interactions"]
    hr = rank01(merged[MEAN_TOP5]); sr = rank01(merged.stage1_plus_length_interactions)
    for weight in FUSION_HIDDEN_WEIGHTS:
        name = f"rank_fusion_hidden_{weight:.2f}"
        merged[name] = weight * hr + (1.0 - weight) * sr
        score_names.append(name)
    metrics = {}
    for name in score_names:
        score = merged[name].to_numpy(float)
        metrics[name] = low_fpr_metrics(merged.label, score) | {"per_language": per_language(merged, score)}
    correlations = {"pearson": float(merged[[MEAN_TOP5, "stage1_plus_length_interactions"]].corr(method="pearson").iloc[0, 1]), "spearman": float(merged[[MEAN_TOP5, "stage1_plus_length_interactions"]].corr(method="spearman").iloc[0, 1])}
    detections = {name: conservative_detection_set(merged.sample_id, merged.label, merged[name], 0.01) for name in score_names}
    overlap = {}
    for left in score_names:
        overlap[left] = {}
        for right in score_names:
            union = detections[left] | detections[right]
            overlap[left][right] = {"jaccard": float(len(detections[left] & detections[right]) / len(union)) if union else 1.0, "left_unique": len(detections[left] - detections[right]), "right_unique": len(detections[right] - detections[left]), "intersection": len(detections[left] & detections[right]), "union": len(union)}
    bootstrap = {}
    for idx, name in enumerate(score_names):
        if name != MEAN_TOP5:
            bootstrap[f"{name}_minus_{MEAN_TOP5}"] = paired_bootstrap(merged, merged[name].to_numpy(float), merged[MEAN_TOP5].to_numpy(float), seed=BOOTSTRAP_SEED + idx)
    return {"intersection_rows": len(merged), "intersection_fraction_of_hidden": len(merged) / len(hidden), "language_label_counts": {f"{k[0]}|{k[1]}": int(v) for k, v in merged.groupby(["language", "label"]).size().items()}, "metrics": metrics, "correlations": correlations, "detection_overlap": overlap, "paired_bootstrap": bootstrap, "fusion_candidates_are_development_only": True, "cross_fitted_two_score_logistic_not_run": True, "reason": "candidate count intentionally limited before fresh confirmation"}, merged


# No-target content controls.
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|==|!=|<=|>=|=>|->|::|&&|\|\||\S")
KEYWORD_GROUPS = {"control": {"if", "else", "for", "while", "switch", "case", "match", "when", "do"}, "definition": {"def", "func", "function", "class", "struct", "interface", "trait", "impl", "fn"}, "module": {"import", "from", "package", "use", "require", "include", "module"}, "flow": {"return", "yield", "break", "continue", "async", "await", "go", "defer"}, "exception": {"try", "catch", "except", "finally", "throw", "raise", "panic"}, "literal": {"true", "false", "null", "nil", "none"}}


def static_features(text: str, language: str) -> dict[str, float]:
    text = str(text); n = max(1, len(text)); lines = text.splitlines() or [""]; tokens = TOKEN_RE.findall(text)
    ids = [t for t in tokens if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t)]; nums = [t for t in tokens if re.fullmatch(r"\d+(?:\.\d+)?", t)]
    line_lengths = np.asarray([len(x) for x in lines], dtype=float)
    counts = np.bincount(np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8), minlength=256).astype(float); probs = counts[counts > 0] / max(1.0, counts.sum()); entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0
    out = {"log_chars": math.log1p(len(text)), "log_lines": math.log1p(len(lines)), "log_tokens": math.log1p(len(tokens)), "avg_line_len": float(line_lengths.mean()), "std_line_len": float(line_lengths.std()), "max_line_len": float(line_lengths.max()), "whitespace_frac": sum(ch.isspace() for ch in text) / n, "digit_frac": sum(ch.isdigit() for ch in text) / n, "upper_frac": sum(ch.isupper() for ch in text) / n, "underscore_frac": text.count("_") / n, "tab_frac": text.count("\t") / n, "blank_line_frac": sum(not x.strip() for x in lines) / max(1, len(lines)), "comment_marker_rate": (text.count("//") + text.count("/*") + text.count("#")) / max(1, len(lines)), "quote_rate": (text.count("\"") + text.count("'") + text.count("`")) / n, "brace_rate": (text.count("{") + text.count("}")) / n, "paren_rate": (text.count("(") + text.count(")")) / n, "bracket_rate": (text.count("[") + text.count("]")) / n, "semicolon_rate": text.count(";") / n, "comma_rate": text.count(",") / n, "colon_rate": text.count(":") / n, "operator_rate": sum(text.count(op) for op in ("=", "+", "-", "*", "/", "%", "<", ">", "!", "&", "|")) / n, "identifier_rate": len(ids) / max(1, len(tokens)), "unique_identifier_rate": len(set(ids)) / max(1, len(ids)), "avg_identifier_len": float(np.mean([len(x) for x in ids])) if ids else 0.0, "max_identifier_len": float(max([len(x) for x in ids], default=0)), "numeric_literal_rate": len(nums) / max(1, len(tokens)), "byte_entropy": entropy}
    lowered = [t.lower() for t in ids]
    for name, group in KEYWORD_GROUPS.items(): out[f"kw_{name}_rate"] = sum(t in group for t in lowered) / max(1, len(tokens))
    for lang in LANGUAGES: out[f"language_{lang}"] = float(language == lang)
    return out


def static_control_oof(frame: pd.DataFrame, splits) -> tuple[np.ndarray, pd.DataFrame]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    features = pd.DataFrame([static_features(text, language) for text, language in zip(frame.content, frame.language)])
    y = frame.label.to_numpy(int); pred = np.full(len(frame), np.nan)
    for fold, (fit, hold) in enumerate(splits):
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.2, max_iter=2000, random_state=SEED + fold)); model.fit(features.iloc[fit], y[fit]); pred[hold] = model.predict_proba(features.iloc[hold])[:, 1]
    if not np.isfinite(pred).all(): raise RuntimeError("C0 static OOF incomplete")
    return pred, features


def ngram_control_oof(frame: pd.DataFrame, splits) -> np.ndarray:
    from sklearn.feature_extraction.text import FeatureUnion, HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    union = FeatureUnion([("char", HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=2**16, alternate_sign=True, norm="l2", lowercase=False)), ("token", HashingVectorizer(analyzer="word", ngram_range=(1, 2), n_features=2**16, alternate_sign=True, norm="l2", lowercase=False, token_pattern=r"(?u)\b\w+\b"))])
    X = union.fit_transform(frame.content.astype(str).tolist()); y = frame.label.to_numpy(int); pred = np.full(len(frame), np.nan)
    for fold, (fit, hold) in enumerate(splits):
        model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=SEED + 100 + fold); model.fit(X[fit], y[fit]); pred[hold] = model.predict_proba(X[hold])[:, 1]
    if not np.isfinite(pred).all(): raise RuntimeError("C1 ngram OOF incomplete")
    return pred


class DSU:
    def __init__(self, n: int): self.p = list(range(n)); self.sz = [1] * n
    def find(self, x: int) -> int:
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b: return
        if self.sz[a] < self.sz[b]: a, b = b, a
        self.p[b] = a; self.sz[a] += self.sz[b]


def simhash64(text: str) -> int:
    tokens = TOKEN_RE.findall(str(text))
    if len(tokens) < 5: shingles = [" ".join(tokens)]
    else:
        total = len(tokens) - 4; starts = range(total) if total <= 256 else np.linspace(0, total - 1, 256, dtype=int); shingles = [" ".join(tokens[int(i):int(i)+5]) for i in starts]
    digests = b"".join(hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=8).digest() for s in shingles)
    if not digests: return 0
    bits = np.unpackbits(np.frombuffer(digests, dtype=np.uint8).reshape(-1, 8), axis=1, bitorder="little"); majority = bits.sum(axis=0) * 2 >= bits.shape[0]
    value = 0
    for bit, flag in enumerate(majority.tolist()):
        if flag: value |= 1 << bit
    return value


def content_groups(frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    hashes = [simhash64(text) for text in frame.content]; buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, value in enumerate(hashes):
        for chunk in range(8): buckets[(chunk, (value >> (chunk * 8)) & 0xFF)].append(idx)
    dsu = DSU(len(frame)); checked = set()
    for items in buckets.values():
        if len(items) > 128: continue
        for pos, left in enumerate(items):
            for right in items[pos + 1:]:
                pair = (left, right) if left < right else (right, left)
                if pair in checked: continue
                checked.add(pair)
                if (hashes[left] ^ hashes[right]).bit_count() <= 6: dsu.union(left, right)
    groups = np.asarray([dsu.find(i) for i in range(len(frame))], dtype=int); sizes = pd.Series(groups).value_counts()
    return groups, {"method": "token_5gram_simhash64_hamming_le_6", "non_singleton_groups": int((sizes > 1).sum()), "rows_in_non_singleton_groups": int(sizes[sizes > 1].sum()) if (sizes > 1).any() else 0, "max_group_size": int(sizes.max()), "candidate_pairs_checked": len(checked)}


def group_splits(frame: pd.DataFrame, groups: np.ndarray):
    from sklearn.model_selection import StratifiedGroupKFold
    strata = frame.language.astype(str) + "_" + frame.label.astype(str)
    return list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(frame, strata, groups=groups))


def control_audit(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    splits = stratified_indices(frame, SEED); c0, static_table = static_control_oof(frame, splits); c1 = ngram_control_oof(frame, splits); hidden = frame[MEAN_TOP5].to_numpy(float)
    result = {"random_outer_oof": {"hidden_mean_top5": low_fpr_metrics(frame.label, hidden) | {"per_language": per_language(frame, hidden)}, "C0_static_logistic": low_fpr_metrics(frame.label, c0) | {"per_language": per_language(frame, c0)}, "C1_hashed_char_token_ngram_logistic": low_fpr_metrics(frame.label, c1) | {"per_language": per_language(frame, c1)}}, "paired_bootstrap": {"hidden_minus_C0": paired_bootstrap(frame, hidden, c0, seed=BOOTSTRAP_SEED + 200), "hidden_minus_C1": paired_bootstrap(frame, hidden, c1, seed=BOOTSTRAP_SEED + 201)}}
    groups, group_diag = content_groups(frame); strict = {"group_diagnostics": group_diag, "directly_paired_to_hidden": False}
    try:
        gs = group_splits(frame, groups); c0g, _ = static_control_oof(frame, gs); c1g = ngram_control_oof(frame, gs); strict["C0_static_group_oof"] = low_fpr_metrics(frame.label, c0g) | {"per_language": per_language(frame, c0g)}; strict["C1_ngram_group_oof"] = low_fpr_metrics(frame.label, c1g) | {"per_language": per_language(frame, c1g)}; strict["status"] = "complete"
    except Exception as exc:
        strict["status"] = "unavailable"; strict["error"] = f"{type(exc).__name__}: {exc}"
    result["strict_content_group_cv"] = strict
    score_table = frame[["sample_id", "language", "label"]].copy(); score_table["C0_static_oof"] = c0; score_table["C1_ngram_oof"] = c1
    return result, static_table, score_table


# Portable hidden-state scalars.
def profile_features(values: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(values, dtype=np.float64); n, depth = values.shape
    if depth < 2: raise RuntimeError(f"portable profile too short: {prefix} depth={depth}")
    x = np.linspace(0.0, 1.0, depth); grid = np.linspace(0.0, 1.0, 16); interp = np.vstack([np.interp(grid, x, row) for row in values]); q = np.quantile(values, [0.25, 0.50, 0.75], axis=1).T; xm = x - x.mean(); denom = float(np.dot(xm, xm)); slope = ((values - values.mean(axis=1, keepdims=True)) @ xm) / denom
    stats = np.column_stack([values.mean(axis=1), values.std(axis=1), values.min(axis=1), values.max(axis=1), q, slope]); names = [f"{prefix}_depth_{i:02d}" for i in range(16)] + [f"{prefix}_{name}" for name in ("mean", "std", "min", "max", "q25", "q50", "q75", "slope")]
    return np.column_stack([interp, stats]), names


def scalarize_hidden(array: np.ndarray) -> tuple[np.ndarray, list[str], float]:
    x = np.asarray(array, dtype=np.float64)
    if x.ndim != 3: raise RuntimeError("hidden shard must be [rows,layers,hidden]")
    eps = 1e-12; rms = np.sqrt(np.mean(x * x, axis=2)); l2_over_sqrt = np.linalg.norm(x, axis=2) / math.sqrt(x.shape[2]); rms_equivalence = float(np.max(np.abs(rms - l2_over_sqrt))); left, right = x[:, :-1, :], x[:, 1:, :]; left_norm = np.linalg.norm(left, axis=2); right_norm = np.linalg.norm(right, axis=2); adjacent_cos = np.sum(left * right, axis=2) / np.maximum(left_norm * right_norm, eps); delta = right - left; delta_norm = np.linalg.norm(delta, axis=2); rel_delta = delta_norm / np.maximum(right_norm, eps); dleft, dright = delta[:, :-1, :], delta[:, 1:, :]; curvature = np.sum(dleft * dright, axis=2) / np.maximum(np.linalg.norm(dleft, axis=2) * np.linalg.norm(dright, axis=2), eps)
    blocks, names = [], []
    for values, prefix in ((rms, "rms"), (adjacent_cos, "adjacent_cos"), (rel_delta, "relative_delta_norm"), (curvature, "delta_direction_cos")):
        block, block_names = profile_features(values, prefix); blocks.append(block); names.extend(block_names)
    return np.column_stack(blocks), names, rms_equivalence


def load_cache_metadata(cache_root: Path, manifest: dict) -> pd.DataFrame:
    info = manifest.get("metadata") or {}; path = cache_root / str(info.get("sample_metadata"))
    if sha256_file(path) != info.get("sample_metadata_sha256"): raise RuntimeError("cache sample metadata SHA changed")
    frame = pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    if len(frame) != EXPECTED_HIDDEN_ROWS or frame.sample_id.duplicated().any(): raise RuntimeError("cache metadata coverage changed")
    return frame


def extract_portable_scalars(cache_root: Path, hidden: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    manifest_path = cache_root / "cache_manifest.json"
    if sha256_file(manifest_path) != CACHE_MANIFEST_SHA256: raise RuntimeError("cache manifest SHA changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("rows") != 5000 or manifest.get("num_layers") != 30 or manifest.get("hidden_size") != 3072: raise RuntimeError("cache shape contract changed")
    metadata = load_cache_metadata(cache_root, manifest)
    if metadata.sample_id.tolist() != hidden.sample_id.tolist(): raise RuntimeError("hidden OOF order differs from sealed cache metadata")
    variant_parts = {name: [] for name in POOLING_VARIANTS}; feature_names = None; max_equivalence = 0.0; shards = manifest.get("shards") or []
    if len(shards) != 20: raise RuntimeError("cache shard count changed")
    for expected_idx, shard in enumerate(shards):
        path = cache_root / str(shard.get("path"))
        if int(shard.get("shard_index", -1)) != expected_idx or sha256_file(path) != shard.get("sha256"): raise RuntimeError(f"cache shard identity failed at {expected_idx}")
        with np.load(path, allow_pickle=False) as archive:
            for variant in POOLING_VARIANTS:
                block, names, equivalence = scalarize_hidden(np.asarray(archive[variant], dtype=np.float32)); feature_names = names if feature_names is None else feature_names
                if names != feature_names: raise RuntimeError("portable scalar schema drift")
                max_equivalence = max(max_equivalence, equivalence); variant_parts[variant].append(block.astype(np.float32))
    assert feature_names is not None
    table = hidden[["sample_id", "language", "label"]].copy()
    for variant in POOLING_VARIANTS:
        values = np.vstack(variant_parts[variant])
        if values.shape[0] != len(table): raise RuntimeError("portable scalar row coverage changed")
        block = pd.DataFrame(values, columns=[f"{variant}__{name}" for name in feature_names]); table = pd.concat([table, block], axis=1)
    return table, {"profile_grid_points": 16, "scalar_features_per_pooling_variant": len(feature_names), "pooling_variants": list(POOLING_VARIANTS), "l2_over_sqrt_d_equals_rms_max_abs_error": max_equivalence, "l2_over_sqrt_d_omitted_from_learner_as_exact_redundancy": True, "architecture_portability": "hidden-size-normalized norms plus normalized-relative-depth interpolation; raw hidden dimensions never enter the learner"}


def scalar_oof(table: pd.DataFrame, variant: str, splits) -> np.ndarray:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    columns = [c for c in table.columns if c.startswith(f"{variant}__")]; X = table[columns]; y = table.label.to_numpy(int); pred = np.full(len(table), np.nan)
    for fold, (fit, hold) in enumerate(splits):
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.2, max_iter=2000, random_state=SEED + 500 + fold)); model.fit(X.iloc[fit], y[fit]); pred[hold] = model.predict_proba(X.iloc[hold])[:, 1]
    if not np.isfinite(pred).all(): raise RuntimeError(f"portable scalar OOF incomplete: {variant}")
    return pred


def portable_audit(hidden: pd.DataFrame, scalar_table: pd.DataFrame, diag: dict) -> tuple[dict, pd.DataFrame]:
    splits = stratified_indices(hidden, SEED); scores = {}; metrics = {}
    for variant in POOLING_VARIANTS:
        pred = scalar_oof(scalar_table, variant, splits); scores[variant] = pred; metrics[variant] = low_fpr_metrics(hidden.label, pred) | {"per_language": per_language(hidden, pred)}
    primary = scores["mean"]; hidden_score = hidden[MEAN_TOP5].to_numpy(float)
    result = {"diagnostics": diag, "primary": "mean", "learner": "median_imputation + standard_scaling + logistic_regression(C=0.2)", "metrics": metrics, "paired_bootstrap_primary_vs_raw_hidden_probe": paired_bootstrap(hidden, primary, hidden_score, seed=BOOTSTRAP_SEED + 400), "caller_helper_are_ablation_only": True, "development_only": True}
    scores_table = hidden[["sample_id", "language", "label"]].copy()
    for variant, pred in scores.items(): scores_table[f"portable_{variant}_oof"] = pred
    return result, scores_table


def load_public_content(hidden: pd.DataFrame) -> pd.DataFrame:
    from datasets import load_dataset
    ids = set(hidden.sample_id.astype(str)); parts = []
    for language in LANGUAGES:
        frame = load_dataset(DATASET_ID, language, split="train", revision=DATASET_REVISION).to_pandas()
        if len(frame) != EXPECTED_PUBLIC_ROWS[language]: raise RuntimeError(f"public train row count changed for {language}")
        if not {"sample_id", "membership", "content"}.issubset(frame.columns): raise RuntimeError(f"public train columns missing for {language}")
        frame = frame[frame.sample_id.astype(str).isin(ids)][["sample_id", "membership", "content"]].copy(); frame["language"] = language; parts.append(frame)
    public = pd.concat(parts, ignore_index=True)
    if len(public) != len(hidden) or public.sample_id.duplicated().any(): raise RuntimeError("public content join coverage changed")
    normalized = public.membership.astype("string").str.strip().str.lower().str.replace("_", "-", regex=False); public["label_public"] = normalized.map({"member": 1, "non-member": 0, "nonmember": 0})
    merged = hidden.merge(public[["sample_id", "language", "content", "label_public"]], on="sample_id", how="left", suffixes=("", "_public"), validate="one_to_one", sort=False)
    if merged.content.isna().any() or not (merged.language == merged.language_public).all() or not (merged.label == merged.label_public).all(): raise RuntimeError("public content identity mismatch")
    return merged.drop(columns=["language_public", "label_public"]).reset_index(drop=True)


def json_dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args(); root = args.input_root.resolve(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    hidden_path = locate_exact(root, "oof_predictions.csv", HIDDEN_OOF_SHA256); cache_manifest = locate_exact(root, "cache_manifest.json", CACHE_MANIFEST_SHA256); hidden = load_hidden_oof(hidden_path)
    stage1_frame = load_stage1_train(root); stage1_score = stage1_interaction_oof(stage1_frame); stage1_scored = stage1_frame[["sample_id", "language", "label"]].copy(); stage1_scored["stage1_plus_length_interactions"] = stage1_score; stage1_reproduction = low_fpr_metrics(stage1_scored.label, stage1_score); fusion, common = fusion_audit(stage1_scored, hidden)
    content_frame = load_public_content(hidden); controls, static_table, control_scores = control_audit(content_frame)
    scalar_table, scalar_diag = extract_portable_scalars(cache_manifest.parent, hidden); portable, portable_scores = portable_audit(hidden, scalar_table, scalar_diag)
    json_dump(out / "fusion_audit.json", fusion); json_dump(out / "content_control_audit.json", controls); json_dump(out / "portable_scalar_audit.json", portable); common.to_parquet(out / "fusion_common_scores.parquet", index=False); control_scores.to_parquet(out / "content_control_oof_scores.parquet", index=False); scalar_table.to_parquet(out / "portable_scalar_table.parquet", index=False); portable_scores.to_parquet(out / "portable_scalar_oof_scores.parquet", index=False); static_table.to_parquet(out / "static_feature_table.parquet", index=False)
    summary = {"schema_version": 1, "task_id": "P1-03-post-5k-development-audit-v1", "status": "complete", "development_only": True, "no_new_target_model_forward": True, "competition_submission": False, "hidden_oof_sha256": HIDDEN_OOF_SHA256, "cache_manifest_sha256": CACHE_MANIFEST_SHA256, "stage1_source_rows": len(stage1_frame), "stage1_reproduction": stage1_reproduction, "fusion_intersection_rows": fusion["intersection_rows"], "content_control_rows": len(content_frame), "portable_scalar_rows": len(scalar_table), "fusion_candidate_hidden_rank_weights": list(FUSION_HIDDEN_WEIGHTS), "strong_content_control": "hashed char 3-5gram + token 1-2gram + fixed regularized logistic", "strict_content_group_method": controls["strict_content_group_cv"]["group_diagnostics"]["method"], "portable_primary": "mean pooled hidden -> normalized-depth scalar profile -> fixed logistic", "anti_posthoc": {"current_5k_used_only_as_development": True, "no_fusion_promoted_here": True, "no_public_lb_used": True, "no_validation_rows_used": True, "no_formula_added_after_results": True}, "persistent_outputs": sorted(p.name for p in out.iterdir()) + ["run_manifest.json"]}
    json_dump(out / "run_manifest.json", summary); print(json.dumps(summary, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
