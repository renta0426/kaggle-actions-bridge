"""Build the private CPU notebook that evaluates frozen Mellum predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap


TARGET = "renta0426/mellum-transfer-evaluation-v1"
GPU_KERNEL = "renta0426/mellum-transfer-v1"
COHORT_KERNEL = "renta0426/mellum-transfer-cohort-v1"
UPSTREAM_REPOSITORY = "https://github.com/Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026"
UPSTREAM_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
UPSTREAM_EVAL_PATH = "data/7b_train_test/eval_results.parquet"
UPSTREAM_SHA256 = "6e8559279e8ebd94ec8c25561c91d38c454a4e109ea0813b07864ff6a66d4068"
EXPECTED_ROWS = 2_000


def _cell(cell_type: str, source: str, index: int) -> dict:
    payload = {
        "cell_type": cell_type,
        "id": f"pc-mellum-eval-{index:02d}",
        "metadata": {},
        "source": textwrap.dedent(source).lstrip("\n").splitlines(keepends=True),
    }
    if cell_type == "code":
        payload |= {"execution_count": None, "outputs": []}
    return payload


SETUP_SOURCE = f'''
from pathlib import Path
import hashlib
import json
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

EVAL_TARGET = {TARGET!r}
EVAL_GPU_KERNEL = {GPU_KERNEL!r}
EVAL_COHORT_KERNEL = {COHORT_KERNEL!r}
EVAL_UPSTREAM_REPOSITORY = {UPSTREAM_REPOSITORY!r}
EVAL_UPSTREAM_COMMIT = {UPSTREAM_COMMIT!r}
EVAL_UPSTREAM_EVAL_PATH = {UPSTREAM_EVAL_PATH!r}
EVAL_UPSTREAM_SHA256 = {UPSTREAM_SHA256!r}
EVAL_EXPECTED_ROWS = {EXPECTED_ROWS}
EVAL_OUTPUT = Path("/kaggle/working/mellum_transfer_evaluation_v1")
EVAL_OUTPUT.mkdir(parents=True, exist_ok=True)
EVAL_STARTED = time.perf_counter()
'''


PREDICTION_SOURCE = r'''
feature_candidates = [
    path for path in Path("/kaggle/input").rglob("sample_features.parquet")
    if "mellum-transfer-v1" in str(path)
    and "mellum-transfer-evaluation" not in str(path)
]
gpu_manifest_candidates = [
    path for path in Path("/kaggle/input").rglob("run_manifest.json")
    if "mellum-transfer-v1" in str(path)
    and "mellum-transfer-evaluation" not in str(path)
]
cohort_candidates = [
    path for path in Path("/kaggle/input").rglob("prediction_manifest.parquet")
    if "mellum-transfer-cohort-v1" in str(path)
]
cohort_manifest_candidates = [
    path for path in Path("/kaggle/input").rglob("run_manifest.json")
    if "mellum-transfer-cohort-v1" in str(path)
]
if not all(len(values) == 1 for values in (
    feature_candidates, gpu_manifest_candidates, cohort_candidates,
    cohort_manifest_candidates,
)):
    raise RuntimeError("expected one GPU feature/manifest and one cohort feature/manifest")

gpu_manifest = json.loads(gpu_manifest_candidates[0].read_text(encoding="utf-8"))
cohort_manifest = json.loads(cohort_manifest_candidates[0].read_text(encoding="utf-8"))
expected_gpu = {
    "status": "complete",
    "experiment": "mellum-transfer-v1",
    "method": "model_invariant_v1",
    "model_id": "JetBrains/Mellum-4b-base",
    "model_revision": "83cce2605fbdf6a3868627e9b0a5924e0072b94d",
    "rows": EVAL_EXPECTED_ROWS,
    "target_labels_embedded_in_gpu_notebook": False,
    "target_labels_used_for_training_or_normalization": False,
    "previous_model_scores_used": False,
    "hidden_stage1_validation_labels_used": False,
    "public_leaderboard_tuning_used": False,
    "submission_created": False,
}
for key, expected in expected_gpu.items():
    if gpu_manifest.get(key) != expected:
        raise RuntimeError(f"GPU manifest mismatch for {key}")
if (gpu_manifest.get("fidelity") or {}).get("passed") is not True:
    raise RuntimeError("GPU fidelity gate did not pass")
if cohort_manifest.get("prediction_manifest_sha256") != gpu_manifest.get(
    "cohort_prediction_manifest_sha256"
):
    raise RuntimeError("GPU run used a different frozen cohort")

features = pd.read_parquet(feature_candidates[0]).sort_values("sample_index").reset_index(drop=True)
cohort = pd.read_parquet(cohort_candidates[0]).sort_values("sample_index").reset_index(drop=True)
for name, frame in (("features", features), ("cohort", cohort)):
    if len(frame) != EVAL_EXPECTED_ROWS or frame.sample_index.tolist() != list(range(EVAL_EXPECTED_ROWS)):
        raise RuntimeError(f"{name} coverage/order mismatch")
if not features.sample_id.equals(cohort.sample_id):
    raise RuntimeError("GPU predictions and cohort IDs differ")
for forbidden in ("label", "membership", "is_member", "lumia_score"):
    if forbidden in features.columns:
        raise RuntimeError(f"label/prior score crossed GPU prediction boundary: {forbidden}")

# Reproduce the final score before any membership label is loaded.
for destination, source in (
    ("rank_loss", "loss_multiwindow"),
    ("rank_minkpp", "standard_minkpp"),
    ("rank_local_span", "best_local_span"),
):
    reproduced = features.groupby("language", sort=False)[source].rank(
        method="average", pct=True
    )
    if not np.allclose(
        reproduced.to_numpy(float), features[destination].to_numpy(float),
        rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError(f"rank reproduction mismatch: {destination}")
reproduced_score = features[["rank_loss", "rank_minkpp", "rank_local_span"]].mean(axis=1)
if not np.allclose(
    reproduced_score.to_numpy(float), features.membership_score.to_numpy(float),
    rtol=0.0, atol=1e-12,
):
    raise RuntimeError("fixed fusion reproduction mismatch")
prediction_only = features.copy()
print({"prediction_rows_verified_before_labels": len(prediction_only),
       "prediction_sha256": gpu_manifest["sample_features_sha256"]})
'''


LABEL_SOURCE = r'''
# Labels are materialized only after the frozen predictions and fusion above have
# been fully verified.
upstream_root = Path("/kaggle/working/sersem_upstream")
subprocess.run(
    [
        "git", "clone", "--filter=blob:none", "--no-checkout",
        EVAL_UPSTREAM_REPOSITORY + ".git", str(upstream_root),
    ],
    check=True,
)
subprocess.run(
    ["git", "-C", str(upstream_root), "sparse-checkout", "init", "--cone"],
    check=True,
)
subprocess.run(
    ["git", "-C", str(upstream_root), "sparse-checkout", "set", "data"],
    check=True,
)
subprocess.run(
    ["git", "-C", str(upstream_root), "checkout", EVAL_UPSTREAM_COMMIT],
    check=True,
)
actual_commit = subprocess.run(
    ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
if actual_commit != EVAL_UPSTREAM_COMMIT:
    raise RuntimeError(f"upstream commit mismatch: {actual_commit}")
source_path = upstream_root / EVAL_UPSTREAM_EVAL_PATH
actual_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
if actual_sha256 != EVAL_UPSTREAM_SHA256:
    raise RuntimeError(f"upstream artifact mismatch: {actual_sha256}")

labels = pd.read_parquet(
    source_path,
    columns=["content", "language", "membership", "is_member"],
).dropna(subset=["content"]).copy()
labels["content_sha256"] = labels.content.map(
    lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
)
labels = labels[labels.content_sha256.isin(set(cohort.content_sha256))].copy()
if len(labels) != EVAL_EXPECTED_ROWS or labels.content_sha256.duplicated().any():
    raise RuntimeError("target label coverage mismatch")
labels["label"] = labels.is_member.astype(int)
expected_label = labels.membership.map({"member": 1, "non-member": 0})
if expected_label.isna().any() or not np.array_equal(
    expected_label.to_numpy(int), labels.label.to_numpy(int)
):
    raise RuntimeError("target membership labels are inconsistent")
label_table = cohort[["sample_id", "content_sha256", "language"]].merge(
    labels[["content_sha256", "language", "membership", "label"]],
    on="content_sha256",
    validate="one_to_one",
    suffixes=("", "_label"),
)
if not (label_table.language == label_table.language_label).all():
    raise RuntimeError("prediction/label language mismatch")
evaluated = prediction_only.merge(
    label_table[["sample_id", "membership", "label"]],
    on="sample_id",
    how="left",
    validate="one_to_one",
).sort_values("sample_index").reset_index(drop=True)
if evaluated.label.isna().any():
    raise RuntimeError("prediction/label join is incomplete")
print({"labels_joined_after_prediction": True,
       "balance": evaluated.groupby(["language", "label"]).size().to_dict()})
'''


EVALUATION_SOURCE = r'''
def low_fpr_metrics(y_true, score):
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


def detection_set(frame, score, target_fpr=0.01):
    nonmember = np.asarray(score)[frame.label.to_numpy(int) == 0]
    threshold = float(np.quantile(nonmember, 1.0 - target_fpr, method="higher"))
    mask = (frame.label.to_numpy(int) == 1) & (np.asarray(score) >= threshold)
    return set(frame.loc[mask, "sample_id"])


scores = {
    "model_invariant_v1": evaluated.membership_score.to_numpy(float),
    "loss_multiwindow": evaluated.loss_multiwindow.to_numpy(float),
    "standard_minkpp": evaluated.standard_minkpp.to_numpy(float),
    "local_64": evaluated.best_local_span.to_numpy(float),
    "neg_mean_log_rank": evaluated.neg_mean_log_rank.to_numpy(float),
}
metrics_rows = []
language_rows = []
length_rows = []
evaluated["length_bin"] = pd.qcut(
    evaluated.token_count, 5, labels=False, duplicates="drop"
).astype(int)
for predictor, values in scores.items():
    metrics_rows.append({"predictor": predictor, "scope": "overall"} | low_fpr_metrics(
        evaluated.label, values
    ))
    for language, index in evaluated.groupby("language", sort=True).groups.items():
        indices = np.asarray(list(index), dtype=int)
        language_rows.append(
            {"predictor": predictor, "language": language, "rows": len(indices)}
            | low_fpr_metrics(evaluated.label.iloc[indices], values[indices])
        )
    for length_bin, index in evaluated.groupby("length_bin", sort=True).groups.items():
        indices = np.asarray(list(index), dtype=int)
        if evaluated.label.iloc[indices].nunique() != 2:
            raise RuntimeError(f"length bin {length_bin} lacks both labels")
        length_rows.append(
            {
                "predictor": predictor,
                "length_bin": int(length_bin),
                "rows": len(indices),
                "token_count_min": int(evaluated.token_count.iloc[indices].min()),
                "token_count_max": int(evaluated.token_count.iloc[indices].max()),
            }
            | low_fpr_metrics(evaluated.label.iloc[indices], values[indices])
        )
metrics = pd.DataFrame(metrics_rows)
language_metrics = pd.DataFrame(language_rows)
length_metrics = pd.DataFrame(length_rows)
detections = {name: detection_set(evaluated, values) for name, values in scores.items()}
unique_rows = []
jaccard_rows = []
for left, left_set in detections.items():
    others = set().union(*(values for name, values in detections.items() if name != left))
    unique_rows.append({"predictor": left, "detected_members": len(left_set),
                        "unique_true_positives": len(left_set - others)})
    for right, right_set in detections.items():
        union = left_set | right_set
        jaccard_rows.append({"left": left, "right": right,
                             "jaccard": len(left_set & right_set) / len(union) if union else 1.0})
unique = pd.DataFrame(unique_rows)
jaccard = pd.DataFrame(jaccard_rows)
correlation = pd.DataFrame(scores).corr(method="spearman")
correlation.index.name = "left"
correlation = correlation.reset_index().melt(
    id_vars="left", var_name="right", value_name="spearman"
)

metrics.to_csv(EVAL_OUTPUT / "metrics.csv", index=False)
language_metrics.to_csv(EVAL_OUTPUT / "language_metrics.csv", index=False)
length_metrics.to_csv(EVAL_OUTPUT / "length_metrics.csv", index=False)
unique.to_csv(EVAL_OUTPUT / "unique_true_positives.csv", index=False)
jaccard.to_csv(EVAL_OUTPUT / "detection_jaccard.csv", index=False)
correlation.to_csv(EVAL_OUTPUT / "score_spearman.csv", index=False)
evaluated.to_parquet(EVAL_OUTPUT / "predictions_with_labels.parquet", index=False)

index = metrics.set_index("predictor")
fusion = index.loc["model_invariant_v1"]
loss = index.loc["loss_multiwindow"]
worst_language = {
    predictor: float(language_metrics.loc[language_metrics.predictor == predictor, "auc"].min())
    for predictor in scores
}
worst_length = {
    predictor: float(length_metrics.loc[length_metrics.predictor == predictor, "auc"].min())
    for predictor in scores
}
promotion_checks = {
    "auc_above_loss": bool(fusion.auc > loss.auc),
    "tpr_at_1pct_above_loss": bool(
        fusion["tpr_at_0.01_fpr"] > loss["tpr_at_0.01_fpr"]
    ),
    "worst_language_within_0.01_of_loss": bool(
        worst_language["model_invariant_v1"] >= worst_language["loss_multiwindow"] - 0.01
    ),
    "fidelity_passed": True,
    "target_label_clean": True,
}
promoted = bool(all(promotion_checks.values()))
summary = {
    "status": "complete",
    "experiment": "mellum-transfer-v1",
    "rows": len(evaluated),
    "model_id": gpu_manifest["model_id"],
    "model_revision": gpu_manifest["model_revision"],
    "promoted": promoted,
    "promotion_checks": promotion_checks,
    "worst_language_auc": worst_language,
    "worst_length_auc": worst_length,
    "detected_members_at_1pct_fpr": {name: len(values) for name, values in detections.items()},
    "target_labels_used_for_training_or_normalization": False,
    "target_labels_joined_after_prediction": True,
    "previous_model_scores_used": False,
    "hidden_stage1_validation_labels_used": False,
    "public_leaderboard_tuning_used": False,
    "runtime_seconds": time.perf_counter() - EVAL_STARTED,
}
(EVAL_OUTPUT / "evaluation_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
rows = []
for predictor, row in index.iterrows():
    rows.append(
        f"| {predictor} | {row.auc:.6f} | {row['tpr_at_0.01_fpr']:.4f} | "
        f"{row.pauc_01:.6f} | {worst_language[predictor]:.6f} | "
        f"{worst_length[predictor]:.6f} |"
    )
report = f"""# Frozen Mellum-4B transfer evaluation

All target-model scores and the fixed fusion were verified before target labels
were loaded. No target label or previous-model score was used for fitting,
normalization, feature selection, or weighting.

| Predictor | AUC | TPR@1% | pAUC@1% | Worst-language AUC | Worst-length AUC |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Promoted: **{str(promoted).lower()}**.
"""
(EVAL_OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")
print(report)
print(json.dumps(summary, indent=2, sort_keys=True, default=str))
'''


def build(output_dir: Path) -> tuple[Path, Path]:
    cells = [
        _cell(
            "markdown",
            """
            # Frozen Mellum-4B transfer evaluation v1

            CPU-only post-prediction evaluation. The frozen GPU scores are fully
            reproduced before membership labels are loaded and joined.
            """,
            0,
        ),
        _cell("code", SETUP_SOURCE, 1),
        _cell("code", PREDICTION_SOURCE, 2),
        _cell("code", LABEL_SOURCE, 3),
        _cell("code", EVALUATION_SOURCE, 4),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": TARGET,
        "title": "Mellum Transfer Evaluation V1",
        "code_file": "mellum-transfer-evaluation-v1.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["membership-inference", "transfer", "evaluation", "cpu"],
        "dataset_sources": [],
        "kernel_sources": [GPU_KERNEL, COHORT_KERNEL],
        "competition_sources": [],
        "model_sources": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / metadata["code_file"]
    metadata_path = output_dir / "kernel-metadata.json"
    notebook_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return notebook_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    notebook_path, metadata_path = build(args.output_dir)
    print(json.dumps({"notebook": str(notebook_path), "metadata": str(metadata_path)}))


if __name__ == "__main__":
    main()
