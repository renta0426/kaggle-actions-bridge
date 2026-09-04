from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def low_fpr_metrics(y_true, score):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    fpr, tpr, _ = roc_curve(y, s)
    out = {
        "auc": float(roc_auc_score(y, s)),
        "pauc_01": float(roc_auc_score(y, s, max_fpr=0.01)),
    }
    for rate in (0.001, 0.005, 0.01, 0.02):
        out[f"tpr_at_{rate:g}_fpr"] = float(tpr[fpr <= rate].max())
    return out


def detection_set(ids, y_true, score, target_fpr=0.01):
    ids = np.asarray(ids)
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    neg = np.sort(s[y == 0])[::-1]
    allowed = max(1, int(math.floor(target_fpr * len(neg))))
    threshold = neg[allowed - 1]
    return set(ids[(y == 1) & (s > threshold)])


def bootstrap(frame, scores, replicates, seed):
    rng = np.random.default_rng(seed)
    strata = [g.index.to_numpy() for _, g in frame.groupby(["language", "label"], sort=True)]
    keep = ["membership_score_v1", "membership_score_v2", "loss_multiwindow", "standard_minkpp", "neg_mean_log_rank"]
    metric_names = ["auc", "tpr_at_0.01_fpr"]
    values = {name: {metric: [] for metric in metric_names} for name in keep}
    differences = {
        "v2_minus_v1_auc": [],
        "v2_minus_loss_auc": [],
        "v2_minus_v1_tpr_at_0.01_fpr": [],
        "v2_minus_loss_tpr_at_0.01_fpr": [],
    }
    y_all = frame.label.to_numpy(int)
    for _ in range(replicates):
        sampled = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in strata])
        y = y_all[sampled]
        row = {}
        for name in keep:
            m = low_fpr_metrics(y, scores[name][sampled])
            row[name] = m
            for metric in metric_names:
                values[name][metric].append(m[metric])
        differences["v2_minus_v1_auc"].append(row["membership_score_v2"]["auc"] - row["membership_score_v1"]["auc"])
        differences["v2_minus_loss_auc"].append(row["membership_score_v2"]["auc"] - row["loss_multiwindow"]["auc"])
        differences["v2_minus_v1_tpr_at_0.01_fpr"].append(row["membership_score_v2"]["tpr_at_0.01_fpr"] - row["membership_score_v1"]["tpr_at_0.01_fpr"])
        differences["v2_minus_loss_tpr_at_0.01_fpr"].append(row["membership_score_v2"]["tpr_at_0.01_fpr"] - row["loss_multiwindow"]["tpr_at_0.01_fpr"])

    def summarize(xs):
        arr = np.asarray(xs, dtype=float)
        return {
            "mean": float(arr.mean()),
            "lower_95": float(np.quantile(arr, 0.025)),
            "upper_95": float(np.quantile(arr, 0.975)),
            "prob_gt_zero": float((arr > 0).mean()),
        }

    return {
        "predictors": {
            name: {metric: summarize(xs) for metric, xs in metrics.items()}
            for name, metrics in values.items()
        },
        "paired_differences": {name: summarize(xs) for name, xs in differences.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--request", type=Path, required=True)
    args = p.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_manifest = {
        "status": "complete",
        "experiment": "smollm2-transfer-v1",
        "method": "stage2_v1_and_deconfounded_v2_frozen_public_reconstruction",
        "model_id": request["target_model_id"],
        "model_revision": request["target_model_revision"],
        "rows": request["expected_rows"],
        "source_commit": request["label_source"]["commit"],
        "source_eval_sha256": request["label_source"]["sha256"],
        "target_labels_embedded_in_gpu_notebook": False,
        "target_labels_used_for_training_or_normalization": False,
        "previous_model_scores_used": False,
        "hidden_stage1_validation_labels_used": False,
        "public_leaderboard_tuning_used": False,
        "submission_created": False,
        "private_repository_content_used": False,
        "cohort_reconstructed_from_pinned_public_sources": True,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise SystemExit(f"manifest mismatch: {key} observed={manifest.get(key)!r} expected={value!r}")
    fidelity = manifest.get("fidelity") or {}
    if fidelity.get("passed") is not True or fidelity.get("rank_exact_match") is not True:
        raise SystemExit("fidelity gate did not pass")
    if float(fidelity.get("max_logp_absolute_difference", 1)) != 0.0 or float(fidelity.get("max_z_absolute_difference", 1)) != 0.0:
        raise SystemExit("fidelity differences are nonzero")

    features = pd.read_parquet(args.features).sort_values("sample_index").reset_index(drop=True)
    required = {
        "sample_index", "sample_id", "language", "token_count", "window_count",
        "loss_multiwindow", "standard_minkpp", "best_local_span", "neg_mean_log_rank",
        "minkpp_length_residual", "logrank_length_residual",
        "membership_score_v1", "membership_score_v2",
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise SystemExit(f"feature output missing columns: {missing}")
    if len(features) != request["expected_rows"] or not features.sample_id.is_unique:
        raise SystemExit("feature output coverage mismatch")
    if features.sample_index.tolist() != list(range(request["expected_rows"])):
        raise SystemExit("feature sample order mismatch")
    forbidden = {"label", "membership", "is_member", "lumia_score", "sersem_score", "hidden_label"}
    leaked = sorted(forbidden.intersection(features.columns))
    if leaked:
        raise SystemExit(f"label/prior-score leakage in GPU output: {leaked}")

    labels = pd.read_parquet(args.labels)
    if hashlib.sha256(args.labels.read_bytes()).hexdigest() != request["label_source"]["sha256"]:
        raise SystemExit("label source SHA256 mismatch")
    if not {"content", "language", "is_member", "membership"}.issubset(labels.columns):
        raise SystemExit("label source schema mismatch")
    labels = labels.dropna(subset=["content"]).copy()
    labels["content_sha256"] = labels.content.map(lambda x: hashlib.sha256(str(x).encode("utf-8")).hexdigest())
    labels["label"] = labels.is_member.astype(int)
    expected_label = labels.membership.map({"member": 1, "non-member": 0})
    if expected_label.isna().any() or not np.array_equal(expected_label.to_numpy(int), labels.label.to_numpy(int)):
        raise SystemExit("label encoding mismatch")
    features["content_sha256"] = features.sample_id.str.rsplit("-", n=1).str[-1]
    evaluated = features.merge(
        labels[["content_sha256", "language", "label"]],
        on="content_sha256", how="left", validate="one_to_one", suffixes=("", "_label")
    )
    if evaluated.label.isna().any() or not (evaluated.language == evaluated.language_label).all():
        raise SystemExit("score/label join incomplete")
    evaluated.label = evaluated.label.astype(int)
    balance = evaluated.groupby(["language", "label"]).size().to_dict()
    expected_balance = {(lang, lab): 200 for lang in ("Go", "Java", "Python", "Ruby", "Rust") for lab in (0, 1)}
    if balance != expected_balance:
        raise SystemExit(f"cohort balance mismatch: {balance}")

    score_columns = request["evaluation_contract"]["overall_predictors"]
    scores = {}
    for name in score_columns:
        source = "best_local_span" if name == "local_64" else name
        scores[name] = evaluated[source].to_numpy(float)
    if not np.isfinite(np.column_stack(list(scores.values()))).all():
        raise SystemExit("non-finite evaluation score")

    overall = {name: low_fpr_metrics(evaluated.label, score) for name, score in scores.items()}
    language_metrics = {}
    for language, group in evaluated.groupby("language", sort=True):
        idx = group.index.to_numpy()
        language_metrics[language] = {name: low_fpr_metrics(evaluated.label.iloc[idx], score[idx]) for name, score in scores.items()}

    length_bin = pd.qcut(evaluated.token_count, 5, labels=False, duplicates="drop")
    if length_bin.isna().any() or length_bin.nunique() != 5:
        raise SystemExit("length quintile contract changed")
    evaluated["length_bin"] = length_bin.astype(int)
    length_metrics = {}
    for value, group in evaluated.groupby("length_bin", sort=True):
        idx = group.index.to_numpy()
        if evaluated.label.iloc[idx].nunique() != 2:
            raise SystemExit(f"length bin {value} lacks both labels")
        length_metrics[str(int(value))] = {
            "rows": int(len(idx)),
            "token_count_min": int(evaluated.token_count.iloc[idx].min()),
            "token_count_max": int(evaluated.token_count.iloc[idx].max()),
            "metrics": {name: low_fpr_metrics(evaluated.label.iloc[idx], score[idx]) for name, score in scores.items()},
        }

    score_frame = pd.DataFrame(scores)
    score_length = {
        name: {
            "token_count_spearman": float(pd.Series(score).corr(evaluated.token_count, method="spearman")),
            "window_count_spearman": float(pd.Series(score).corr(evaluated.window_count, method="spearman")),
        }
        for name, score in scores.items()
    }
    pair_corr = score_frame.corr(method="spearman")
    score_pair_spearman = {
        left: {right: float(pair_corr.loc[left, right]) for right in score_columns}
        for left in score_columns
    }

    sets = {name: detection_set(evaluated.sample_id, evaluated.label, score, 0.01) for name, score in scores.items()}
    detection = {}
    for left, left_set in sets.items():
        others = set().union(*(s for name, s in sets.items() if name != left))
        detection[left] = {
            "true_positives": len(left_set),
            "unique_true_positives": len(left_set - others),
            "jaccard": {
                right: (len(left_set & right_set) / len(left_set | right_set) if (left_set | right_set) else 1.0)
                for right, right_set in sets.items()
            },
        }

    bootstrap_summary = bootstrap(
        evaluated,
        scores,
        int(request["evaluation_contract"]["stratified_bootstrap_replicates"]),
        int(request["evaluation_contract"]["stratified_bootstrap_seed"]),
    )

    worst_language_auc = {
        name: min(language_metrics[lang][name]["auc"] for lang in language_metrics)
        for name in score_columns
    }
    worst_length_auc = {
        name: min(length_metrics[bin_]["metrics"][name]["auc"] for bin_ in length_metrics)
        for name in score_columns
    }
    result = {
        "status": "complete",
        "request_id": request["request_id"],
        "rows": len(evaluated),
        "balance": {f"{lang}/{lab}": int(n) for (lang, lab), n in balance.items()},
        "manifest": {
            "wall_seconds": float(manifest.get("wall_seconds")),
            "v1_score_min": float(manifest.get("v1_score_min")),
            "v1_score_max": float(manifest.get("v1_score_max")),
            "v2_score_min": float(manifest.get("v2_score_min")),
            "v2_score_max": float(manifest.get("v2_score_max")),
            "runtime": manifest.get("runtime"),
            "fidelity": fidelity,
        },
        "overall": overall,
        "language_metrics": language_metrics,
        "length_metrics": length_metrics,
        "score_length_spearman": score_length,
        "score_pair_spearman": score_pair_spearman,
        "detection_at_1pct_fpr": detection,
        "bootstrap": bootstrap_summary,
        "worst_language_auc": worst_language_auc,
        "worst_length_auc": worst_length_auc,
        "clean_room": request["clean_room"],
    }
    print("SMOLLM2_AGGREGATE_EVALUATION_JSON_BEGIN")
    print(json.dumps(result, sort_keys=True))
    print("SMOLLM2_AGGREGATE_EVALUATION_JSON_END")


if __name__ == "__main__":
    main()
