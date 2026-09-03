"""Build the frozen CPU-only Stage 1 length-interaction submission notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap


TARGET = "renta0426/stage1-length-interactions-v1"
SOURCE_DATASET = "renta0426/stage1-raw-fim-submission-v1-output"
REQUEST_ID = "20260903-poisoned-chalice-stage1-length-interactions-run-001"


RUNTIME_SOURCE = r'''
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
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
BRIDGE_BUILDER_BLOB_SHA = __BUILDER_BLOB_SHA__
METHOD = "plus_length_interactions_v1"
SEED = 2027
EXPECTED_TRAIN_ROWS = 10000
EXPECTED_VALIDATION_ROWS = 5000
EXPECTED_TRAIN_SHARDS = 40
EXPECTED_VALIDATION_SHARDS = 20
EXPECTED_ROWS_PER_SHARD = 250
EXPECTED_BASE_FEATURES = 113
EXPECTED_STRUCTURE_FEATURES = 50
EXPECTED_FIM_FEATURES = 11
EXPECTED_SOURCE_FEATURES = 174
EXPECTED_MODEL_FEATURES = 523
EXPECTED_OOF_AUC = 0.68013844
OOF_AUC_TOLERANCE = 0.002
EXPECTED_OOF_TPR1 = 0.0682
OOF_TPR1_TOLERANCE = 0.01
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
    root = Path("/kaggle/input")
    candidates = sorted(
        path for path in root.iterdir()
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
            raise RuntimeError(f"unexpected shard rows: {path.name} -> {rows}")
        if not {"sample_id", "language"}.issubset(columns):
            raise RuntimeError(f"identity columns missing from {path.name}")
        split = "validation" if "validation_order" in columns else "train"
        if split == "train" and "label" not in columns:
            raise RuntimeError(f"training label missing from {path.name}")
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


def numeric_feature_columns(frame):
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
    train = train.copy()
    train["label"] = train.label.astype(int)
    if set(train.label.unique()) != {0, 1}:
        raise RuntimeError("training labels are not binary")
    balance = train.groupby(["language", "label"]).size()
    if len(balance) != 10 or not balance.eq(1000).all():
        raise RuntimeError(f"training language/label balance changed: {balance.to_dict()}")

    if "validation_order" not in validation:
        raise RuntimeError("validation_order missing")
    order = validation.validation_order.astype(int)
    if sorted(order.tolist()) != list(range(EXPECTED_VALIDATION_ROWS)):
        raise RuntimeError("validation_order is not a complete permutation")

    families = split_feature_columns(train)
    counts = {name: len(values) for name, values in families.items()}
    expected = {"base": 113, "structure_raw": 50, "fim": 11}
    if counts != expected:
        raise RuntimeError(f"frozen feature counts changed: {counts} != {expected}")
    for name in ("token_count", "window_count", *CDF_SOURCES):
        if name not in families["base"]:
            raise RuntimeError(f"required base feature missing: {name}")

    features = families["base"] + families["structure_raw"] + families["fim"]
    if len(features) != EXPECTED_SOURCE_FEATURES:
        raise RuntimeError(f"expected {EXPECTED_SOURCE_FEATURES} source features, got {len(features)}")
    entirely_missing = [name for name in features if train[name].isna().all()]
    if entirely_missing:
        raise RuntimeError(f"all-NaN training features: {entirely_missing}")

    validation = validation.copy()
    missing_validation = [name for name in features if name not in validation]
    for name in missing_validation:
        validation[name] = np.nan
    validation[features] = validation[features].apply(pd.to_numeric, errors="raise")
    return train, validation, families, missing_validation


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


def with_log_length(frame):
    output = frame.copy()
    output["audit_log_token_count"] = np.log1p(output.token_count.to_numpy(float))
    output["audit_log_window_count"] = np.log1p(output.window_count.to_numpy(float))
    return output


def interaction_layout(feature_columns):
    sources = [name for name in feature_columns if name not in {"token_count", "window_count"}]
    cdf_features = [f"cdf_{name}" for name in CDF_SOURCES if name in feature_columns]
    interactions = [f"audit_int_token__{name}" for name in sources] + [
        f"audit_int_window__{name}" for name in sources
    ]
    model_features = list(feature_columns) + cdf_features + [
        "audit_log_token_count", "audit_log_window_count",
    ] + interactions
    if len(sources) != 172 or len(interactions) != 344 or len(model_features) != EXPECTED_MODEL_FEATURES:
        raise RuntimeError(
            f"interaction layout changed: sources={len(sources)} interactions={len(interactions)} "
            f"model_features={len(model_features)}"
        )
    return sources, cdf_features, interactions, model_features


def add_interactions(frame, sources, token_center, window_center):
    token_delta = frame.audit_log_token_count.to_numpy(float) - token_center
    window_delta = frame.audit_log_window_count.to_numpy(float) - window_center
    values = {
        f"audit_int_token__{name}": frame[name].to_numpy(float) * token_delta
        for name in sources
    }
    values.update({
        f"audit_int_window__{name}": frame[name].to_numpy(float) * window_delta
        for name in sources
    })
    return pd.concat([frame, pd.DataFrame(values, index=frame.index)], axis=1)


def cross_fit_interactions(frame, splits, feature_columns, seed):
    labels = frame.label.to_numpy(int)
    prediction = np.full(len(frame), np.nan, dtype=float)
    seen = np.zeros(len(frame), dtype=int)
    sources, _, _, model_features = interaction_layout(feature_columns)
    for fold, (fit_index, hold_index) in enumerate(splits):
        fit = with_log_length(add_fold_cdf(frame.iloc[fit_index], frame.iloc[fit_index]))
        hold = with_log_length(add_fold_cdf(frame.iloc[fit_index], frame.iloc[hold_index]))
        token_center = float(fit.audit_log_token_count.mean())
        window_center = float(fit.audit_log_window_count.mean())
        fit = add_interactions(fit, sources, token_center, window_center)
        hold = add_interactions(hold, sources, token_center, window_center)
        model = pipeline(seed + fold)
        model.fit(fit[model_features], labels[fit_index])
        prediction[hold_index] = model.predict_proba(hold[model_features])[:, 1]
        seen[hold_index] += 1
    if not np.isfinite(prediction).all() or not np.all(seen == 1):
        raise RuntimeError("interaction OOF coverage invalid")
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
    splits = stratified_splits(train, 5, SEED)
    oof_score = cross_fit_interactions(train, splits, feature_columns, SEED)
    oof_metrics = low_fpr_metrics(train.label, oof_score)
    if abs(oof_metrics["auc"] - EXPECTED_OOF_AUC) > OOF_AUC_TOLERANCE:
        raise RuntimeError(
            f"candidate OOF AUC reproduction failed: {oof_metrics['auc']:.6f} versus "
            f"{EXPECTED_OOF_AUC:.6f}+/-{OOF_AUC_TOLERANCE:.6f}"
        )
    if abs(oof_metrics["tpr_at_0.01_fpr"] - EXPECTED_OOF_TPR1) > OOF_TPR1_TOLERANCE:
        raise RuntimeError(
            f"candidate OOF TPR@1% reproduction failed: {oof_metrics['tpr_at_0.01_fpr']:.6f}"
        )

    sources, cdf_features, interaction_names, model_features = interaction_layout(feature_columns)
    fit_frame = with_log_length(add_fold_cdf(train, train))
    target_frame = with_log_length(add_fold_cdf(train, validation))
    token_center = float(fit_frame.audit_log_token_count.mean())
    window_center = float(fit_frame.audit_log_window_count.mean())
    fit_frame = add_interactions(fit_frame, sources, token_center, window_center)
    target_frame = add_interactions(target_frame, sources, token_center, window_center)

    model = pipeline(SEED)
    model.fit(fit_frame[model_features], train.label.to_numpy(int))
    validation_score = model.predict_proba(target_frame[model_features])[:, 1]
    if not np.isfinite(validation_score).all():
        raise RuntimeError("non-finite validation scores")

    submission = validation[["validation_order", "sample_id"]].copy()
    submission["membership_score"] = validation_score
    submission = submission.sort_values("validation_order")[["sample_id", "membership_score"]]
    if len(submission) != EXPECTED_VALIDATION_ROWS or not submission.sample_id.is_unique:
        raise RuntimeError("submission coverage mismatch")
    scores = submission.membership_score.to_numpy(float)
    if not np.isfinite(scores).all() or not ((0.0 <= scores) & (scores <= 1.0)).all():
        raise RuntimeError("submission scores are outside [0,1]")

    submission_path = OUTPUT / "submission.csv"
    submission.to_csv(submission_path, index=False)
    submission_sha256 = file_sha256(submission_path)

    schema = {
        "source_families": families,
        "source_counts": {name: len(values) for name, values in families.items()},
        "source_total": len(feature_columns),
        "cdf_features": cdf_features,
        "log_length_features": ["audit_log_token_count", "audit_log_window_count"],
        "interaction_sources": len(sources),
        "interaction_features": len(interaction_names),
        "model_total": len(model_features),
        "ordered_source_feature_sha256": feature_schema_sha256,
        "missing_validation_columns_filled_with_nan": missing_validation,
    }
    (OUTPUT / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUT / "source_shards.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = {
        "status": "complete",
        "request_id": REQUEST_ID,
        "method": METHOD,
        "seed": SEED,
        "bridge_builder_blob_sha": BRIDGE_BUILDER_BLOB_SHA,
        "source_dataset": SOURCE_DATASET,
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "source_extraction_reused": True,
        "model_forward_pass": False,
        "accelerator": "cpu",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_shards": len(train_paths),
        "validation_shards": len(validation_paths),
        "feature_counts": {
            "source": len(feature_columns),
            "cdf": len(cdf_features),
            "log_length": 2,
            "interaction_sources": len(sources),
            "interactions": len(interaction_names),
            "model_total": len(model_features),
        },
        "feature_schema_sha256": feature_schema_sha256,
        "reproduced_candidate_oof": oof_metrics,
        "expected_candidate_oof_auc": EXPECTED_OOF_AUC,
        "full_train_log_token_center": token_center,
        "full_train_log_window_center": window_center,
        "submission_file": submission_path.name,
        "submission_rows": len(submission),
        "submission_sha256": submission_sha256,
        "runtime_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "validation_labels_read": False,
        "validation_membership_read": False,
        "validation_labels_used_for_fit_or_feature_selection": False,
        "hidden_validation_labels_used": False,
        "public_leaderboard_tuning_used": False,
        "mellum_artifacts_read": False,
        "mellum_labels_used": False,
        "submission_created": True,
        "competition_submission_performed_inside_notebook": False,
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
                "id": "pc-stage1-length-int-00",
                "metadata": {},
                "source": [
                    "# Stage 1 frozen length-interaction submission candidate\n",
                    "\n",
                    "CPU-only continuation from the immutable raw+structure+FIM feature cache. "
                    "It reproduces the frozen OOF candidate before fitting all 10k training rows. "
                    "No StarCoder2 forward pass and no validation-label fitting are performed.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "pc-stage1-length-int-01",
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
        "title": "Stage1 Length Interactions V1",
        "code_file": "stage1-length-interactions-v1.ipynb",
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
        "keywords": ["cpu", "membership-inference", "stage1", "length-interactions"],
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
        "expected_model_features": 523,
        "expected_oof_auc": 0.68013844,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--builder-blob-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.builder_blob_sha), sort_keys=True))


if __name__ == "__main__":
    main()
