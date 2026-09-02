"""Recover the frozen Stage 1 submission from failed version-2 feature shards.

The source GPU run completed all 40 train and 20 validation feature shards, then
failed before fitting because the public bridge builder accidentally excluded
``token_count`` and ``window_count`` from its numeric feature inventory.  This
CPU-only script reuses those immutable private outputs.  It performs no model
loading, tokenization, parsing, feature extraction, or competition submission.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SOURCE_KERNEL = "renta0426/stage1-raw-fim-submission-v1"
SOURCE_KERNEL_VERSION = 2
SOURCE_EXPECTED_STATUS = "ERROR"
SOURCE_CACHE_DIRECTORY = "stage1_raw_fim_v1"
OUTPUT_ROOT = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")
METHOD = "raw_plus_fim"
RECOVERY_METHOD = "fit_only_from_failed_v2_feature_shards"
CORE_COMMIT = "bcf6d44db86fdda4ac9dd07b2d0172de471ab67a"
SOURCE_COMMIT = "28c34307c71db733dab2744ec9ec46549e54d424"
MODEL_ID = "bigcode/starcoder2-3b"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
DATASET_ID = "Poisoned-Chalice/ICSE-2027-public"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
SEED = 2027
EXPECTED_TRAIN_ROWS = 10_000
EXPECTED_VALIDATION_ROWS = 5_000
EXPECTED_TRAIN_SHARDS = 40
EXPECTED_VALIDATION_SHARDS = 20
EXPECTED_SHARD_ROWS = 250
EXPECTED_BASE_FEATURES = 113
EXPECTED_STRUCTURE_FEATURES = 50
EXPECTED_FIM_FEATURES = 11
EXPECTED_OOF_AUC = 0.664524
OOF_AUC_TOLERANCE = 0.002
CDF_SOURCES = ("score_loss_mean__max", "min_k_10__max", "min_kpp_10__max")
BROKEN_BUILDER_EXCLUSIONS = ("token_count", "window_count")
NON_FEATURE_COLUMNS = {
    "sample_id",
    "language",
    "membership",
    "label",
    "split",
    "content",
    "validation_order",
}
SHARD_PATTERN = re.compile(r"^features\.part(?P<index>[0-9]{3})\.parquet$")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _part_index(path: Path) -> int:
    match = SHARD_PATTERN.fullmatch(path.name)
    if match is None:
        raise RuntimeError(f"unexpected feature shard name: {path.name}")
    return int(match.group("index"))


def _locate_shards(split: str, expected_count: int) -> list[Path]:
    suffix = Path(SOURCE_CACHE_DIRECTORY) / split / "parts"
    candidates = [
        path
        for path in INPUT_ROOT.rglob("features.part*.parquet")
        if path.is_file() and suffix.as_posix() in path.parent.as_posix()
    ]
    candidates = sorted(candidates, key=_part_index)
    if len(candidates) != expected_count:
        observed = sorted(path.relative_to(INPUT_ROOT).as_posix() for path in candidates)
        raise RuntimeError(
            f"expected {expected_count} {split} shards from {SOURCE_KERNEL} v2, "
            f"found {len(candidates)}; observed={observed[:5]}"
        )
    indices = [_part_index(path) for path in candidates]
    if indices != list(range(expected_count)):
        raise RuntimeError(f"non-contiguous {split} shard indices: {indices}")
    return candidates


def _load_split(
    split: str,
    expected_count: int,
    expected_rows: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    paths = _locate_shards(split, expected_count)
    parts: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    expected_columns: list[str] | None = None
    for index, path in enumerate(paths):
        frame = pd.read_parquet(path)
        if len(frame) != EXPECTED_SHARD_ROWS:
            raise RuntimeError(
                f"{split} shard {index} has {len(frame)} rows, expected {EXPECTED_SHARD_ROWS}"
            )
        if "sample_id" not in frame or not frame.sample_id.is_unique:
            raise RuntimeError(f"{split} shard {index} has invalid sample IDs")
        columns = frame.columns.tolist()
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise RuntimeError(f"{split} shard {index} column order/schema changed")
        inventory.append(
            {
                "split": split,
                "part": index,
                "rows": len(frame),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    if len(result) != expected_rows or not result.sample_id.is_unique:
        raise RuntimeError(f"{split} feature coverage failure")
    return result, inventory


def _numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in NON_FEATURE_COLUMNS and not column.startswith("cdf_")
    ]


def _feature_inventory(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric = _numeric_feature_columns(frame)
    base = [column for column in numeric if not column.startswith(("ast_", "fim_"))]
    structure = [column for column in numeric if column.startswith("ast_")]
    fim = [column for column in numeric if column.startswith("fim_")]
    if len(base) != EXPECTED_BASE_FEATURES:
        raise RuntimeError(f"Expected {EXPECTED_BASE_FEATURES} base features, got {len(base)}")
    if len(structure) != EXPECTED_STRUCTURE_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_STRUCTURE_FEATURES} structure features, got {len(structure)}"
        )
    if len(fim) != EXPECTED_FIM_FEATURES:
        raise RuntimeError(f"Expected {EXPECTED_FIM_FEATURES} FIM features, got {len(fim)}")
    if not set(BROKEN_BUILDER_EXCLUSIONS).issubset(base):
        missing = sorted(set(BROKEN_BUILDER_EXCLUSIONS).difference(base))
        raise RuntimeError(f"recovery features still omit the two diagnosed columns: {missing}")
    return base, structure, fim


def _add_fold_cdf(
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
        if not len(global_values):
            raise RuntimeError(f"CDF source contains no finite non-member values: {column}")
        language_values = {
            language: np.sort(group[column].dropna().to_numpy(float))
            for language, group in nonmember.groupby("language", sort=False)
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


def _stratified_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold

    strata = frame.language.astype(str) + "_" + frame.label.astype(int).astype(str)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return list(splitter.split(frame, strata))


def _cross_fit_meta(
    frame: pd.DataFrame,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    feature_columns: Sequence[str],
) -> np.ndarray:
    cdf_features = [f"cdf_{column}" for column in CDF_SOURCES if column in feature_columns]
    model_features = list(feature_columns) + cdf_features
    prediction = np.full(len(frame), np.nan, dtype=float)
    seen = np.zeros(len(frame), dtype=np.int8)
    labels = frame.label.astype(int).to_numpy()
    for fold, (fit_index, hold_index) in enumerate(splits):
        fit = _add_fold_cdf(frame.iloc[fit_index], frame.iloc[fit_index])
        hold = _add_fold_cdf(frame.iloc[fit_index], frame.iloc[hold_index])
        model = _pipeline(SEED + fold)
        model.fit(fit[model_features], labels[fit_index])
        prediction[hold_index] = model.predict_proba(hold[model_features])[:, 1]
        seen[hold_index] += 1
    if not np.isfinite(prediction).all() or not np.all(seen == 1):
        raise RuntimeError("OOF coverage is incomplete or repeated")
    return prediction


def _low_fpr_metrics(y_true: Sequence[int], score: Sequence[float]) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.asarray(y_true, dtype=int)
    values = np.asarray(score, dtype=float)
    fpr, tpr, _ = roc_curve(y, values)
    metrics = {
        "auc": float(roc_auc_score(y, values)),
        "pauc_01": float(roc_auc_score(y, values, max_fpr=0.01)),
    }
    for rate in (0.001, 0.005, 0.01, 0.02):
        metrics[f"tpr_at_{rate:g}_fpr"] = float(tpr[fpr <= rate].max())
    return metrics


def _validate_source_frames(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    if set(train.sample_id).intersection(validation.sample_id):
        raise RuntimeError("train and validation sample IDs overlap")
    if train.label.isna().any():
        raise RuntimeError("train labels are incomplete")
    train["label"] = train.label.astype(int)
    expected_balance = train.groupby(["language", "label"], sort=True).size()
    if len(expected_balance) != 10 or not expected_balance.eq(1_000).all():
        raise RuntimeError(f"train language/label balance changed: {expected_balance.to_dict()}")
    if "validation_order" not in validation:
        raise RuntimeError("validation_order is missing from recovered shards")
    order = validation.validation_order.astype(int).sort_values().tolist()
    if order != list(range(EXPECTED_VALIDATION_ROWS)):
        raise RuntimeError("validation_order is not a complete 0..4999 permutation")


def _fit_and_score(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    base, structure, fim = _feature_inventory(train)
    feature_columns = base + structure + fim
    train = train.sort_values("sample_id").reset_index(drop=True).copy()
    oof = _cross_fit_meta(train, _stratified_splits(train), feature_columns)
    oof_metrics = _low_fpr_metrics(train.label, oof)
    delta = float(oof_metrics["auc"] - EXPECTED_OOF_AUC)
    if abs(delta) > OOF_AUC_TOLERANCE:
        raise RuntimeError(
            "Frozen raw_plus_fim reproduction gate failed: "
            f"AUC={oof_metrics['auc']:.6f}, "
            f"expected={EXPECTED_OOF_AUC:.6f}+/-{OOF_AUC_TOLERANCE:.6f}"
        )

    cdf_features = [f"cdf_{column}" for column in CDF_SOURCES if column in feature_columns]
    model_features = feature_columns + cdf_features
    fit_frame = _add_fold_cdf(train, train)
    target_frame = _add_fold_cdf(train, validation)
    model = _pipeline(SEED)
    model.fit(fit_frame[model_features], train.label.to_numpy(int))
    score = model.predict_proba(target_frame[model_features])[:, 1]
    if not np.isfinite(score).all() or not np.all((0.0 <= score) & (score <= 1.0)):
        raise RuntimeError("validation scores are non-finite or outside [0, 1]")
    diagnostics: dict[str, object] = {
        "feature_counts": {
            "base": len(base),
            "structure_raw": len(structure),
            "fim": len(fim),
            "total": len(feature_columns),
            "cdf_added_at_fit": len(cdf_features),
        },
        "reproduced_oof": oof_metrics,
        "expected_oof_auc": EXPECTED_OOF_AUC,
        "oof_auc_delta": delta,
        "recovered_columns": list(BROKEN_BUILDER_EXCLUSIONS),
    }
    return score, diagnostics


def main() -> dict[str, object]:
    started = time.perf_counter()
    if not INPUT_ROOT.is_dir():
        raise RuntimeError("/kaggle/input is unavailable")

    train, train_inventory = _load_split(
        "train_10k", EXPECTED_TRAIN_SHARDS, EXPECTED_TRAIN_ROWS
    )
    validation, validation_inventory = _load_split(
        "validation_5k", EXPECTED_VALIDATION_SHARDS, EXPECTED_VALIDATION_ROWS
    )
    _validate_source_frames(train, validation)
    score, diagnostics = _fit_and_score(train, validation)

    scored = validation.copy()
    scored["membership_score"] = score
    visible_mask = scored.label.notna().to_numpy()
    visible_metrics: dict[str, float] | None = None
    if visible_mask.any() and scored.loc[visible_mask, "label"].nunique() == 2:
        visible_metrics = _low_fpr_metrics(
            scored.loc[visible_mask, "label"].astype(int),
            scored.loc[visible_mask, "membership_score"],
        )

    submission = scored.sort_values("validation_order")[
        ["sample_id", "membership_score"]
    ].copy()
    if len(submission) != EXPECTED_VALIDATION_ROWS or not submission.sample_id.is_unique:
        raise RuntimeError("submission row or sample-ID coverage failure")
    if not np.isfinite(submission.membership_score.to_numpy(float)).all():
        raise RuntimeError("submission contains non-finite scores")

    submission_path = OUTPUT_ROOT / "submission.csv"
    submission.to_csv(submission_path, index=False)
    submission_digest = _sha256(submission_path)
    inventory = train_inventory + validation_inventory
    manifest: dict[str, object] = {
        "status": "complete",
        "method": METHOD,
        "recovery_method": RECOVERY_METHOD,
        "source_kernel": SOURCE_KERNEL,
        "source_kernel_version": SOURCE_KERNEL_VERSION,
        "source_kernel_expected_status": SOURCE_EXPECTED_STATUS,
        "root_cause": (
            "bridge builder excluded token_count and window_count from numeric features, "
            "reducing the frozen base inventory from 113 to 111"
        ),
        "core_commit": CORE_COMMIT,
        "source_commit": SOURCE_COMMIT,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "visible_validation_rows": int(visible_mask.sum()),
        **diagnostics,
        "visible_validation_diagnostic_only": visible_metrics,
        "reused_feature_shards": {
            "train": EXPECTED_TRAIN_SHARDS,
            "validation": EXPECTED_VALIDATION_SHARDS,
            "total": len(inventory),
        },
        "reused_feature_bytes": int(sum(int(item["bytes"]) for item in inventory)),
        "feature_shard_inventory": inventory,
        "submission_file": submission_path.name,
        "submission_rows": len(submission),
        "submission_sha256": submission_digest,
        "runtime_seconds": time.perf_counter() - started,
        "feature_extraction_repeated": False,
        "model_forward_passes": 0,
        "gpu_used": False,
        "hidden_validation_labels_used": False,
        "public_leaderboard_tuning_used": False,
        "validation_labels_used_for_fit_or_feature_selection": False,
        "competition_submission_created": False,
    }
    (OUTPUT_ROOT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "recovery_method": RECOVERY_METHOD,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "feature_counts": diagnostics["feature_counts"],
                "oof_auc": diagnostics["reproduced_oof"]["auc"],
                "submission_sha256": submission_digest,
                "feature_extraction_repeated": False,
                "gpu_used": False,
            },
            sort_keys=True,
        )
    )
    return manifest


if __name__ == "__main__":
    main()
