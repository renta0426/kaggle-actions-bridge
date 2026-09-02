"""Build a private CPU notebook that resumes Stage 1 from cached feature shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap


TARGET = "renta0426/stage1-raw-fim-resume-v1"
SOURCE_DATASET = "renta0426/stage1-raw-fim-submission-v1-output"
REQUEST_ID = "20260903-poisoned-chalice-stage1-resume-001"


RUNTIME_SOURCE = r'''
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import math
import platform
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REQUEST_ID = __REQUEST_ID__
SOURCE_DATASET = __SOURCE_DATASET__
SOURCE_DATASET_VERSION = 1
SOURCE_KERNEL = "renta0426/stage1-raw-fim-submission-v1"
SOURCE_KERNEL_VERSION = 2
SOURCE_FAILURE = "Expected 113 base features, got 111"
BRIDGE_BUILDER_BLOB_SHA = __BUILDER_BLOB_SHA__
METHOD = "raw_plus_fim_resume_from_cached_shards"
SEED = 2027
EXPECTED_TRAIN_ROWS = 10000
EXPECTED_VALIDATION_ROWS = 5000
EXPECTED_TRAIN_SHARDS = 40
EXPECTED_VALIDATION_SHARDS = 20
EXPECTED_ROWS_PER_SHARD = 250
EXPECTED_BASE_FEATURES = 113
EXPECTED_STRUCTURE_FEATURES = 50
EXPECTED_FIM_FEATURES = 11
EXPECTED_OOF_AUC = 0.664524
OOF_AUC_TOLERANCE = 0.002
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
CDF_SOURCES = ("score_loss_mean__max", "min_k_10__max", "min_kpp_10__max")
EXCLUDED_COLUMNS = {
    "sample_id", "language", "membership", "label", "split", "content",
    "validation_order",
}
OUTPUT = Path("/kaggle/working")


def file_sha256(path, chunk_size=1 << 20):
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def locate_input_root():
    exact = Path("/kaggle/input/stage1-raw-fim-submission-v1-output")
    if exact.is_dir():
        return exact
    candidates = sorted(
        path for path in Path("/kaggle/input").iterdir()
        if path.is_dir() and "stage1-raw-fim-submission-v1-output" in path.name
    )
    if len(candidates) != 1:
        raise RuntimeError(f"unable to identify the one frozen source dataset: {candidates}")
    return candidates[0]


def discover_shards(root):
    paths = sorted(root.rglob("features.part*.parquet"))
    if len(paths) != EXPECTED_TRAIN_SHARDS + EXPECTED_VALIDATION_SHARDS:
        raise RuntimeError(f"expected 60 feature shards, found {len(paths)}")
    train_paths, validation_paths, manifest = [], [], []
    for path in paths:
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        columns = tuple(parquet.schema_arrow.names)
        if rows != EXPECTED_ROWS_PER_SHARD:
            raise RuntimeError(f"unexpected shard rows: {path} -> {rows}")
        if not {"sample_id", "language", "membership", "label"}.issubset(columns):
            raise RuntimeError(f"identity/label columns missing from {path}")
        split = "validation" if "validation_order" in columns else "train"
        (validation_paths if split == "validation" else train_paths).append(path)
        manifest.append({
            "relative_path": str(path.relative_to(root)),
            "split": split,
            "rows": rows,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "columns": len(columns),
        })
    if len(train_paths) != EXPECTED_TRAIN_SHARDS:
        raise RuntimeError(f"expected 40 train shards, found {len(train_paths)}")
    if len(validation_paths) != EXPECTED_VALIDATION_SHARDS:
        raise RuntimeError(f"expected 20 validation shards, found {len(validation_paths)}")
    return train_paths, validation_paths, manifest


def read_shards(paths):
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True, sort=False)


def membership_labels(frame):
    return frame.membership.astype("string").str.lower().map(
        {"member": 1, "non-member": 0, "non_member": 0}
    )


def numeric_feature_columns(frame):
    # token_count and window_count intentionally remain included. The failed
    # bridge-local evaluator removed exactly these two columns while still
    # asserting the original 113-column contract.
    return [
        name for name in frame.select_dtypes(include=[np.number]).columns
        if name not in EXCLUDED_COLUMNS and not name.startswith("cdf_")
    ]


def split_feature_columns(frame):
    numeric = numeric_feature_columns(frame)
    base = [name for name in numeric if not name.startswith(("ast_", "fim_"))]
    structure = [name for name in numeric if name.startswith("ast_")]
    fim = [name for name in numeric if name.startswith("fim_")]
    return {"base": base, "structure_raw": structure, "fim": fim}


def validate_frames(train, validation):
    if len(train) != EXPECTED_TRAIN_ROWS or len(validation) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError(f"row mismatch: train={len(train)} validation={len(validation)}")
    for name, frame in (("train", train), ("validation", validation)):
        if frame.columns.duplicated().any():
            raise RuntimeError(f"duplicate columns in {name}")
        if not frame.sample_id.is_unique:
            raise RuntimeError(f"duplicate sample_id in {name}")
        if set(frame.language.dropna().unique()) != set(LANGUAGES):
            raise RuntimeError(f"language coverage changed in {name}")
    if set(train.sample_id).intersection(validation.sample_id):
        raise RuntimeError("train/validation sample_id overlap")

    if train.label.isna().any():
        raise RuntimeError("training labels are missing")
    expected_train = membership_labels(train)
    if expected_train.isna().any() or not np.array_equal(
        expected_train.to_numpy(int), train.label.to_numpy(int)
    ):
        raise RuntimeError("training membership/label mismatch")
    balance = train.groupby(["language", "label"]).size()
    if len(balance) != 10 or not balance.eq(1000).all():
        raise RuntimeError(f"training language/label balance changed: {balance.to_dict()}")

    if "validation_order" not in validation:
        raise RuntimeError("validation_order missing")
    order = validation.validation_order.astype(int)
    if sorted(order.tolist()) != list(range(EXPECTED_VALIDATION_ROWS)):
        raise RuntimeError("validation_order is not a complete permutation")
    visible = validation[validation.label.notna()].copy()
    for language in LANGUAGES:
        counts = visible.loc[
            visible.language == language, "membership"
        ].value_counts().to_dict()
        if counts != {"member": 245, "non-member": 255}:
            raise RuntimeError(f"visible validation counts changed for {language}: {counts}")
    expected_visible = membership_labels(visible)
    if expected_visible.isna().any() or not np.array_equal(
        expected_visible.to_numpy(int), visible.label.to_numpy(int)
    ):
        raise RuntimeError("visible validation membership/label mismatch")

    families = split_feature_columns(train)
    counts = {name: len(values) for name, values in families.items()}
    expected = {"base": 113, "structure_raw": 50, "fim": 11}
    if counts != expected:
        raise RuntimeError(f"frozen feature counts changed: {counts} != {expected}")
    for name in ("token_count", "window_count", *CDF_SOURCES):
        if name not in families["base"]:
            raise RuntimeError(f"required base feature missing: {name}")

    features = families["base"] + families["structure_raw"] + families["fim"]
    entirely_missing = [name for name in features if train[name].isna().all()]
    if entirely_missing:
        raise RuntimeError(f"all-NaN training features: {entirely_missing}")
    validation = validation.copy()
    missing_validation = [name for name in features if name not in validation]
    for name in missing_validation:
        validation[name] = np.nan
    validation[features] = validation[features].apply(pd.to_numeric, errors="raise")
    return train.copy(), validation, families, missing_validation


def add_fold_cdf(reference, target, columns=CDF_SOURCES, min_language_rows=20):
    output = target.copy()
    nonmember = reference[reference.label == 0]
    if nonmember.empty:
        raise RuntimeError("CDF reference has no non-members")
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


def pipeline(seed):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.2, max_iter=2000, random_state=seed),
    )


def stratified_splits(frame, n_splits, seed):
    strata = frame.language.astype(str) + "_" + frame.label.astype(int).astype(str)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(frame, strata))


def cross_fit_meta(frame, splits, features, seed):
    cdf_features = [f"cdf_{name}" for name in CDF_SOURCES if name in features]
    model_features = list(features) + cdf_features
    prediction = np.full(len(frame), np.nan, dtype=float)
    seen = np.zeros(len(frame), dtype=int)
    labels = frame.label.astype(int).to_numpy()
    for fold, (fit_index, hold_index) in enumerate(splits):
        fit = add_fold_cdf(frame.iloc[fit_index], frame.iloc[fit_index])
        hold = add_fold_cdf(frame.iloc[fit_index], frame.iloc[hold_index])
        model = pipeline(seed + fold)
        model.fit(fit[model_features], labels[fit_index])
        prediction[hold_index] = model.predict_proba(hold[model_features])[:, 1]
        seen[hold_index] += 1
    if not np.isfinite(prediction).all() or not np.all(seen == 1):
        raise RuntimeError("OOF coverage invalid")
    return prediction


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


def main():
    started = time.perf_counter()
    input_root = locate_input_root()
    train_paths, validation_paths, source_manifest = discover_shards(input_root)
    train = read_shards(train_paths)
    validation = read_shards(validation_paths)
    train, validation, families, missing_validation = validate_frames(train, validation)
    feature_columns = families["base"] + families["structure_raw"] + families["fim"]
    feature_schema_sha256 = sha256(
        json.dumps(feature_columns, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    train = train.sort_values("sample_id").reset_index(drop=True)
    train["label"] = train.label.astype(int)
    splits = stratified_splits(train, 5, SEED)
    oof_score = cross_fit_meta(train, splits, feature_columns, SEED)
    oof_metrics = low_fpr_metrics(train.label, oof_score)
    oof_delta = float(oof_metrics["auc"] - EXPECTED_OOF_AUC)
    if abs(oof_delta) > OOF_AUC_TOLERANCE:
        raise RuntimeError(
            f"OOF reproduction failed: {oof_metrics['auc']:.6f} versus "
            f"{EXPECTED_OOF_AUC:.6f}+/-{OOF_AUC_TOLERANCE:.6f}"
        )

    cdf_features = [f"cdf_{name}" for name in CDF_SOURCES if name in feature_columns]
    model_features = feature_columns + cdf_features
    fit_frame = add_fold_cdf(train, train)
    target_frame = add_fold_cdf(train, validation)
    model = pipeline(SEED)
    model.fit(fit_frame[model_features], train.label.to_numpy(int))
    validation_score = model.predict_proba(target_frame[model_features])[:, 1]
    if not np.isfinite(validation_score).all():
        raise RuntimeError("non-finite validation scores")

    validation = validation.copy()
    validation["membership_score"] = validation_score
    visible_mask = validation.label.notna().to_numpy()
    visible_metrics = low_fpr_metrics(
        validation.loc[visible_mask, "label"].astype(int),
        validation.loc[visible_mask, "membership_score"],
    )
    submission = validation.sort_values("validation_order")[[
        "sample_id", "membership_score"
    ]].copy()
    if len(submission) != EXPECTED_VALIDATION_ROWS or not submission.sample_id.is_unique:
        raise RuntimeError("submission coverage mismatch")
    scores = submission.membership_score.to_numpy(float)
    if not np.isfinite(scores).all() or not ((0.0 <= scores) & (scores <= 1.0)).all():
        raise RuntimeError("submission scores are outside [0,1]")

    submission_path = OUTPUT / "submission.csv"
    submission.to_csv(submission_path, index=False)
    submission_sha256 = sha256(submission_path.read_bytes()).hexdigest()
    pd.DataFrame({
        "sample_id": train.sample_id,
        "language": train.language,
        "label": train.label,
        "membership_score": oof_score,
    }).to_parquet(OUTPUT / "oof_predictions.parquet", index=False)
    validation.sort_values("validation_order")[[
        "validation_order", "sample_id", "language", "label", "membership_score"
    ]].to_parquet(OUTPUT / "validation_scores.parquet", index=False)

    schema = {
        "families": families,
        "counts": {name: len(values) for name, values in families.items()},
        "total": len(feature_columns),
        "cdf_added_at_fit": cdf_features,
        "ordered_feature_sha256": feature_schema_sha256,
        "missing_validation_columns_filled_with_nan": missing_validation,
    }
    (OUTPUT / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUT / "source_shards.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "status": "complete",
        "request_id": REQUEST_ID,
        "method": METHOD,
        "seed": SEED,
        "bridge_builder_blob_sha": BRIDGE_BUILDER_BLOB_SHA,
        "source_dataset": SOURCE_DATASET,
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "source_kernel": SOURCE_KERNEL,
        "source_kernel_version": SOURCE_KERNEL_VERSION,
        "source_kernel_status": "error_after_complete_feature_extraction",
        "source_failure": SOURCE_FAILURE,
        "source_extraction_reused": True,
        "gpu_forward_passes": 0,
        "accelerator": "cpu",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "visible_validation_rows": int(visible_mask.sum()),
        "train_shards": len(train_paths),
        "validation_shards": len(validation_paths),
        "feature_counts": {
            "base": len(families["base"]),
            "structure_raw": len(families["structure_raw"]),
            "fim": len(families["fim"]),
            "total": len(feature_columns),
            "cdf_added_at_fit": len(cdf_features),
        },
        "feature_schema_sha256": feature_schema_sha256,
        "reproduced_oof": oof_metrics,
        "expected_oof_auc": EXPECTED_OOF_AUC,
        "oof_auc_delta": oof_delta,
        "visible_validation_diagnostic_only": visible_metrics,
        "submission_file": submission_path.name,
        "submission_rows": len(submission),
        "submission_sha256": submission_sha256,
        "bridge_regression": {
            "incorrectly_excluded_features": ["token_count", "window_count"],
            "observed_before_fix": 111,
            "expected_and_restored": 113,
        },
        "runtime_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "hidden_validation_labels_used": False,
        "public_leaderboard_tuning_used": False,
        "validation_labels_used_for_fit_or_feature_selection": False,
        "submission_created": True,
    }
    (OUTPUT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return manifest

manifest = main()
'''


def build(output_dir: Path, builder_blob_sha: str) -> dict:
    if len(builder_blob_sha) != 40 or any(c not in "0123456789abcdef" for c in builder_blob_sha):
        raise ValueError("builder_blob_sha must be a 40-character Git blob SHA")
    runtime = textwrap.dedent(RUNTIME_SOURCE)
    runtime = runtime.replace("__REQUEST_ID__", repr(REQUEST_ID))
    runtime = runtime.replace("__SOURCE_DATASET__", repr(SOURCE_DATASET))
    runtime = runtime.replace("__BUILDER_BLOB_SHA__", repr(builder_blob_sha))
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "pc-stage1-resume-00",
                "metadata": {},
                "source": [
                    "# Stage 1 raw+FIM cached continuation\n",
                    "\n",
                    "Reuses the completed 10k train and 5k validation feature shards. "
                    "No model is loaded and no GPU/TPU forward pass is performed.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "pc-stage1-resume-01",
                "metadata": {},
                "outputs": [],
                "source": runtime.splitlines(keepends=True),
            },
        ],
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
        "title": "Stage1 Raw FIM Resume V1",
        "code_file": "stage1-raw-fim-resume-v1.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [SOURCE_DATASET],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "keywords": ["cpu", "membership-inference", "resume", "cached-features"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / metadata["code_file"]
    metadata_path = output_dir / "kernel-metadata.json"
    notebook_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "target": TARGET,
        "source_dataset": SOURCE_DATASET,
        "notebook": str(notebook_path),
        "metadata": str(metadata_path),
        "cells": len(notebook["cells"]),
        "accelerator": "cpu",
        "internet": False,
        "submission_inside_notebook": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--builder-blob-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.builder_blob_sha), sort_keys=True))


if __name__ == "__main__":
    main()
