"""Strategy v2 E02: Task1.1 denominator-aware structured logFC transfer.

E02 keeps the canonical Task1.1 target (D1 CXCL10 / canonical Pre-vacc
CXCL10) unchanged.  It evaluates fixed references and one predeclared small
shared-slope Ridge that removes study-specific response offsets and uses only
one representation per baseline cytokine.

The module returns aggregate diagnostics only.  Participant-level OOF and
Challenge predictions are used transiently inside the process and are never
included in the returned payload.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .aliases import canonicalize_cytokine, canonicalize_timepoint
from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns, require_finite
from .cv import NamedSplit
from .datasets import TaskDataset
from .evaluation import default_splits_for_task, run_compact_task
from .metrics import percentile_rank, safe_spearman
from .models import ModelSpec, fit_final_model
from .runner import InputBundle, build_b02_datasets
from .strategy_e01 import (
    _anchor_frame,
    _compact_base_frame,
    _oriented_anchor,
    _rank_within,
)
from .strategy_e01_v2 import _metric_summary, _paired_comparison, _sensitivity_28


EXPERIMENT = "strategy_v2_e02_task11_structured_logfc"
RANDOM_SEED = 20260907
EXPECTED_TRAIN_ROWS = 127
EXPECTED_TRAIN_STUDIES = 4
EXPECTED_CHALLENGE_ROWS = 40
ANCHOR_COLUMN = "cytokine_rank__CXCL10"
RAW_BASELINE_COLUMN = "cytokine_raw__CXCL10"
LOG1P_BASELINE_COLUMN = "cytokine_log1p__CXCL10"
B21_MODEL = "pls_2"
ANCHOR_RESIDUAL_MODEL = "pls_1"
ANCHOR_RESIDUAL_LAMBDA = 0.25
RIDGE_ALPHA = 10.0
REPEAT_TIMEPOINTS = ("-14", "0")
REPEAT_MIN_PAIRED_PER_STUDY = 3
REPEAT_MIN_SOURCE_STUDIES_PER_FOLD = 2
REPEAT_COLUMNS = (
    "__e02_cxcl10_repeat_delta_log1p",
    "__e02_cxcl10_repeat_variance_log1p",
)
RESIDUAL_TARGET_COLUMN = "__e02_task11_anchor_residual"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def _find_spec(config: BaselineConfig, model_set: str, name: str) -> ModelSpec:
    matches = [spec for spec in config.model_specs(model_set) if spec.name == name]
    if len(matches) != 1:
        raise DataContractError(
            f"E02 expected exactly one {model_set}/{name} spec; found {len(matches)}"
        )
    return matches[0]


def _identity_spec(spec: ModelSpec) -> ModelSpec:
    return replace(spec, target_transform="identity", clip_min=None)


def _validate_task11_contract(dataset: TaskDataset) -> None:
    dataset.validate()
    if dataset.task != "Task1.1":
        raise DataContractError(f"E02 requires Task1.1, found {dataset.task}")
    if len(dataset.train) != EXPECTED_TRAIN_ROWS:
        raise DataContractError(
            f"E02 fixed Task1.1 cohort expected {EXPECTED_TRAIN_ROWS} rows; "
            f"found {len(dataset.train)}"
        )
    if int(dataset.train["study_group"].nunique()) != EXPECTED_TRAIN_STUDIES:
        raise DataContractError(
            f"E02 fixed Task1.1 cohort expected {EXPECTED_TRAIN_STUDIES} studies"
        )
    if len(dataset.challenge) != EXPECTED_CHALLENGE_ROWS:
        raise DataContractError(
            f"E02 fixed Challenge cohort expected {EXPECTED_CHALLENGE_ROWS} rows; "
            f"found {len(dataset.challenge)}"
        )
    require_columns(
        dataset.train,
        [ANCHOR_COLUMN, RAW_BASELINE_COLUMN, LOG1P_BASELINE_COLUMN],
        table_name="E02 Task1.1 train",
    )
    require_columns(
        dataset.challenge,
        [ANCHOR_COLUMN, RAW_BASELINE_COLUMN, LOG1P_BASELINE_COLUMN],
        table_name="E02 Task1.1 challenge",
    )


def _fixed_b21(
    dataset: TaskDataset,
    *,
    config: BaselineConfig,
    splits: Sequence[NamedSplit],
) -> tuple[pd.DataFrame, Any]:
    """Regenerate frozen B2.1 PLS2 under the exact E02 split contract."""

    spec = _find_spec(config, "task_11", B21_MODEL)
    result = run_compact_task(
        dataset,
        specs=[spec],
        splits=splits,
        random_state=config.random_state,
        selection_policy="robust_v1",
    )
    if result.selected_spec.name != B21_MODEL:
        raise DataContractError(
            f"E02 B2.1 identity mismatch: {result.selected_spec.name}"
        )
    return _compact_base_frame(dataset, result), result


def _fixed_anchor_residual(
    dataset: TaskDataset,
    *,
    config: BaselineConfig,
    splits: Sequence[NamedSplit],
) -> pd.DataFrame:
    """Regenerate the frozen Task1.1 PLS1/lambda=.25 anchor-residual reference."""

    spec = _identity_spec(_find_spec(config, "task_11", ANCHOR_RESIDUAL_MODEL))
    excluded = tuple(dict.fromkeys([*dataset.excluded_columns, dataset.target_column]))
    parts: list[pd.DataFrame] = []
    for split in splits:
        train = dataset.train.iloc[split.train_indices].copy()
        validation = dataset.train.iloc[split.validation_indices].copy()
        train_anchor = _oriented_anchor(
            "Task1.1",
            train,
            train,
            anchor_column=ANCHOR_COLUMN,
            target_column=dataset.target_column,
        )
        validation_anchor = _oriented_anchor(
            "Task1.1",
            train,
            validation,
            anchor_column=ANCHOR_COLUMN,
            target_column=dataset.target_column,
        )
        train[RESIDUAL_TARGET_COLUMN] = (
            _rank_within(train, dataset.target_column) - train_anchor
        )
        _, correction = fit_final_model(
            train,
            validation,
            target_column=RESIDUAL_TARGET_COLUMN,
            spec=spec,
            excluded_columns=excluded,
        )
        correction = np.asarray(correction, dtype=float)
        require_finite(correction, name="E02.Task1.1.anchor_residual_correction")
        score = validation_anchor + ANCHOR_RESIDUAL_LAMBDA * correction
        parts.append(
            pd.DataFrame(
                {
                    "row_index": split.validation_indices,
                    "participant_id": validation["participant_id"].astype(str).to_numpy(),
                    "subject_group": validation["subject_group"].astype(str).to_numpy(),
                    "study_group": validation["study_group"].astype(str).to_numpy(),
                    "target": pd.to_numeric(
                        validation[dataset.target_column], errors="coerce"
                    ).to_numpy(dtype=float),
                    "prediction": score,
                }
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    if frame["row_index"].duplicated().any():
        raise DataContractError("E02 anchor-residual OOF predicts a row more than once")
    if set(frame["row_index"].astype(int)) != set(range(len(dataset.train))):
        raise DataContractError("E02 anchor-residual OOF coverage mismatch")
    return frame.sort_values("row_index").reset_index(drop=True)


def _structured_other_columns(dataset: TaskDataset) -> list[str]:
    """Freeze one log1p representation for every non-CXCL10 common cytokine."""

    train = set(dataset.train.columns)
    challenge = set(dataset.challenge.columns)
    common = train & challenge
    columns = sorted(
        c
        for c in common
        if c.startswith("cytokine_log1p__") and c != LOG1P_BASELINE_COLUMN
    )
    if not columns:
        raise DataContractError("E02 found no common non-CXCL10 log1p cytokines")
    return columns


def _raw_structured_features(
    frame: pd.DataFrame,
    *,
    other_columns: Sequence[str],
    repeat_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, Mapping[str, int]]:
    require_columns(
        frame,
        ["study_group", RAW_BASELINE_COLUMN, *other_columns, *repeat_columns],
        table_name="E02 structured feature source",
    )
    raw = pd.to_numeric(frame[RAW_BASELINE_COLUMN], errors="coerce").to_numpy(dtype=float)
    valid_baseline = np.isfinite(raw) & (raw > 0.0)
    log_b = np.full(len(frame), np.nan, dtype=float)
    log_b[valid_baseline] = np.log(raw[valid_baseline])
    data: dict[str, np.ndarray] = {"log_baseline_CXCL10": log_b}
    for column in other_columns:
        name = "state__" + column.split("__", 1)[1]
        data[name] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    for column in repeat_columns:
        data[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    features = pd.DataFrame(data)
    return features, {
        "rows": int(len(frame)),
        "nonpositive_or_missing_logB": int((~valid_baseline).sum()),
        "raw_feature_missing_cells": int(features.isna().sum().sum()),
    }


def _study_local_robust_scale(
    frame: pd.DataFrame,
    raw_features: pd.DataFrame,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """X-only study-local median/IQR scaling.

    This deliberately self-normalizes each held study using held X only.  It is
    therefore a declared transductive preprocessing step; held y is never used.
    Missing values are mapped to zero after centering, equivalent to the local
    median.  IQR-zero features are fixed to zero.
    """

    if len(frame) != len(raw_features):
        raise DataContractError("E02 feature/frame row count mismatch")
    work = frame.reset_index(drop=True)
    raw = raw_features.reset_index(drop=True)
    scaled = np.zeros((len(raw), raw.shape[1]), dtype=float)
    zero_iqr = 0
    all_missing = 0
    for _, positions in work.groupby(
        "study_group", dropna=False, observed=True, sort=False
    ).indices.items():
        pos = np.asarray(positions, dtype=int)
        block = raw.iloc[pos].to_numpy(dtype=float)
        for j in range(block.shape[1]):
            values = block[:, j]
            finite = np.isfinite(values)
            if not finite.any():
                all_missing += 1
                continue
            observed = values[finite]
            median = float(np.median(observed))
            q25, q75 = np.quantile(observed, [0.25, 0.75])
            iqr = float(q75 - q25)
            if not np.isfinite(iqr) or iqr <= 1e-12:
                zero_iqr += 1
                continue
            transformed = np.zeros(len(values), dtype=float)
            transformed[finite] = (values[finite] - median) / iqr
            scaled[pos, j] = transformed
    require_finite(scaled, name="E02 study-local scaled features")
    return scaled, {
        "study_blocks": int(work["study_group"].nunique(dropna=False)),
        "feature_count": int(raw.shape[1]),
        "zero_iqr_study_features": int(zero_iqr),
        "all_missing_study_features": int(all_missing),
        "held_x_self_normalization": True,
        "missing_after_scaling": 0,
    }


def _center_logfc_target(
    frame: pd.DataFrame,
    *,
    target_column: str,
    values_override: np.ndarray | None = None,
) -> np.ndarray:
    require_columns(frame, ["study_group", target_column], table_name="E02 target frame")
    if values_override is None:
        values = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float)
    else:
        values = np.asarray(values_override, dtype=float)
        if values.shape != (len(frame),):
            raise DataContractError("E02 target override shape mismatch")
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise DataContractError("E02 Task1.1 logFC target must be finite and positive")
    y = np.log(values)
    centered = np.empty_like(y)
    work = frame.reset_index(drop=True)
    for _, positions in work.groupby(
        "study_group", dropna=False, observed=True, sort=False
    ).indices.items():
        pos = np.asarray(positions, dtype=int)
        centered[pos] = y[pos] - float(np.mean(y[pos]))
    require_finite(centered, name="E02 centered logFC target")
    return centered


def _shuffle_target_within_study(
    frame: pd.DataFrame,
    *,
    target_column: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    require_columns(
        frame,
        ["study_group", "subject_group", target_column],
        table_name="E02 shuffle source",
    )
    values = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float)
    require_finite(values, name="E02 shuffle target")
    shuffled = values.copy()
    moved = 0
    work = frame.reset_index(drop=True)
    for _, positions in work.groupby(
        "study_group", dropna=False, observed=True, sort=False
    ).indices.items():
        pos = np.asarray(positions, dtype=int)
        subjects = work.iloc[pos]["subject_group"].astype(str)
        if subjects.duplicated().any():
            raise DataContractError(
                "E02 compact shuffle expects one row per subject within study"
            )
        perm = rng.permutation(len(pos))
        shuffled[pos] = values[pos[perm]]
        moved += int(np.sum(perm != np.arange(len(pos))))
    return shuffled, moved


def _fit_structured_oof(
    dataset: TaskDataset,
    *,
    splits: Sequence[NamedSplit],
    other_columns: Sequence[str],
    repeat_columns: Sequence[str] = (),
    shuffle_labels: bool = False,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED)
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    moved_total = 0
    feature_names: list[str] | None = None
    for split in splits:
        train = dataset.train.iloc[split.train_indices].copy()
        validation = dataset.train.iloc[split.validation_indices].copy()
        train_raw, train_raw_diag = _raw_structured_features(
            train,
            other_columns=other_columns,
            repeat_columns=repeat_columns,
        )
        validation_raw, validation_raw_diag = _raw_structured_features(
            validation,
            other_columns=other_columns,
            repeat_columns=repeat_columns,
        )
        if feature_names is None:
            feature_names = list(train_raw.columns)
        if list(train_raw.columns) != feature_names or list(validation_raw.columns) != feature_names:
            raise DataContractError("E02 structured feature order changed across folds")
        x_train, train_scale_diag = _study_local_robust_scale(train, train_raw)
        x_validation, validation_scale_diag = _study_local_robust_scale(
            validation, validation_raw
        )
        if shuffle_labels:
            shuffled, moved = _shuffle_target_within_study(
                train,
                target_column=dataset.target_column,
                rng=rng,
            )
            moved_total += moved
            y_train = _center_logfc_target(
                train,
                target_column=dataset.target_column,
                values_override=shuffled,
            )
        else:
            y_train = _center_logfc_target(
                train,
                target_column=dataset.target_column,
            )
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        model.fit(x_train, y_train)
        prediction = np.asarray(model.predict(x_validation), dtype=float).reshape(-1)
        require_finite(prediction, name="E02 structured held-study prediction")
        coef = np.asarray(model.coef_, dtype=float).reshape(-1)
        log_b_index = feature_names.index("log_baseline_CXCL10")
        held = sorted(validation["study_group"].astype(str).unique())
        folds.append(
            {
                "held_studies": held,
                "validation_rows": int(len(validation)),
                "training_rows": int(len(train)),
                "training_studies": int(train["study_group"].nunique()),
                "feature_count": int(len(feature_names)),
                "logB_coefficient": float(coef[log_b_index]),
                "coefficient_l2_norm": float(np.linalg.norm(coef)),
                "train_raw_diagnostics": train_raw_diag,
                "validation_raw_diagnostics": validation_raw_diag,
                "train_scale_diagnostics": train_scale_diag,
                "validation_scale_diagnostics": validation_scale_diag,
            }
        )
        parts.append(
            pd.DataFrame(
                {
                    "row_index": split.validation_indices,
                    "participant_id": validation["participant_id"].astype(str).to_numpy(),
                    "subject_group": validation["subject_group"].astype(str).to_numpy(),
                    "study_group": validation["study_group"].astype(str).to_numpy(),
                    "target": pd.to_numeric(
                        validation[dataset.target_column], errors="coerce"
                    ).to_numpy(dtype=float),
                    "prediction": prediction,
                }
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    if frame["row_index"].duplicated().any():
        raise DataContractError("E02 structured OOF predicts a row more than once")
    if set(frame["row_index"].astype(int)) != set(range(len(dataset.train))):
        raise DataContractError("E02 structured OOF coverage mismatch")
    return frame.sort_values("row_index").reset_index(drop=True), {
        "model": "ridge",
        "alpha": RIDGE_ALPHA,
        "fit_intercept": False,
        "target": "log(Task1.1_fold_change) centered within each training study",
        "x_normalization": "study-local median/IQR; held X self-normalization",
        "held_y_used_for_preprocessing": False,
        "study_id_used_as_feature": False,
        "feature_names": feature_names or [],
        "fold_diagnostics": folds,
        "label_shuffle": bool(shuffle_labels),
        "moved_subject_assignments_across_folds": int(moved_total),
    }


def _fit_structured_challenge(
    dataset: TaskDataset,
    *,
    other_columns: Sequence[str],
    repeat_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    train_raw, _ = _raw_structured_features(
        dataset.train,
        other_columns=other_columns,
        repeat_columns=repeat_columns,
    )
    challenge_raw, challenge_raw_diag = _raw_structured_features(
        dataset.challenge,
        other_columns=other_columns,
        repeat_columns=repeat_columns,
    )
    if list(train_raw.columns) != list(challenge_raw.columns):
        raise DataContractError("E02 Challenge feature order mismatch")
    x_train, train_scale = _study_local_robust_scale(dataset.train, train_raw)
    x_challenge, challenge_scale = _study_local_robust_scale(
        dataset.challenge, challenge_raw
    )
    y_train = _center_logfc_target(dataset.train, target_column=dataset.target_column)
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
    model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_challenge), dtype=float).reshape(-1)
    require_finite(prediction, name="E02 structured Challenge score")
    result = dataset.challenge[["participant_id"]].copy()
    result["prediction"] = prediction
    coef = np.asarray(model.coef_, dtype=float).reshape(-1)
    names = list(train_raw.columns)
    return result, {
        "rows": int(len(result)),
        "prediction_unique": int(pd.Series(prediction).nunique()),
        "feature_count": int(len(names)),
        "logB_coefficient": float(coef[names.index("log_baseline_CXCL10")]),
        "coefficient_l2_norm": float(np.linalg.norm(coef)),
        "train_scale_diagnostics": train_scale,
        "challenge_scale_diagnostics": challenge_scale,
        "challenge_raw_diagnostics": challenge_raw_diag,
    }


def _fixed_anchor_residual_challenge(
    dataset: TaskDataset,
    *,
    config: BaselineConfig,
) -> pd.DataFrame:
    spec = _identity_spec(_find_spec(config, "task_11", ANCHOR_RESIDUAL_MODEL))
    direction_anchor = _oriented_anchor(
        "Task1.1",
        dataset.train,
        dataset.train,
        anchor_column=ANCHOR_COLUMN,
        target_column=dataset.target_column,
    )
    challenge_anchor = _oriented_anchor(
        "Task1.1",
        dataset.train,
        dataset.challenge,
        anchor_column=ANCHOR_COLUMN,
        target_column=dataset.target_column,
    )
    train = dataset.train.copy()
    train[RESIDUAL_TARGET_COLUMN] = (
        _rank_within(train, dataset.target_column) - direction_anchor
    )
    excluded = tuple(dict.fromkeys([*dataset.excluded_columns, dataset.target_column]))
    _, correction = fit_final_model(
        train,
        dataset.challenge,
        target_column=RESIDUAL_TARGET_COLUMN,
        spec=spec,
        excluded_columns=excluded,
    )
    score = challenge_anchor + ANCHOR_RESIDUAL_LAMBDA * np.asarray(
        correction, dtype=float
    )
    require_finite(score, name="E02 anchor-residual Challenge score")
    result = dataset.challenge[["participant_id"]].copy()
    result["prediction"] = score
    return result


def _challenge_anchor(dataset: TaskDataset) -> pd.DataFrame:
    score = _oriented_anchor(
        "Task1.1",
        dataset.train,
        dataset.challenge,
        anchor_column=ANCHOR_COLUMN,
        target_column=dataset.target_column,
    )
    result = dataset.challenge[["participant_id"]].copy()
    result["prediction"] = score
    return result


def _challenge_agreement(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
) -> Mapping[str, Any]:
    merged = candidate.merge(
        reference,
        on="participant_id",
        how="inner",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    if len(merged) != len(candidate) or len(merged) != len(reference):
        raise DataContractError("E02 Challenge agreement does not align all rows")
    c = pd.to_numeric(merged["prediction_candidate"], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(merged["prediction_reference"], errors="coerce").to_numpy(dtype=float)
    require_finite(c, name="E02 Challenge candidate score")
    require_finite(r, name="E02 Challenge reference score")
    c_rank = percentile_rank(c)
    r_rank = percentile_rank(r)
    return _json_safe(
        {
            "rows": int(len(merged)),
            "rank_spearman": safe_spearman(c_rank, r_rank).to_dict(),
            "mean_absolute_percentile_rank_difference": float(
                np.mean(np.abs(c_rank - r_rank))
            ),
            "max_absolute_percentile_rank_difference": float(
                np.max(np.abs(c_rank - r_rank))
            ),
        }
    )


def _repeat_feature_table(cytokine: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        cytokine,
        ["participant_id", "study_accession", "timepoint", "analyte", "value"],
        table_name="E02 repeat cytokine",
    )
    work = cytokine[
        ["participant_id", "study_accession", "timepoint", "analyte", "value"]
    ].copy()
    work["analyte"] = work["analyte"].map(canonicalize_cytokine)
    work["timepoint"] = work["timepoint"].map(canonicalize_timepoint)
    work = work.loc[
        work["analyte"].eq("CXCL10") & work["timepoint"].isin(REPEAT_TIMEPOINTS)
    ].copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work.loc[work["value"] < 0.0, "value"] = np.nan
    work = work.dropna(subset=["value"])
    if work.empty:
        return pd.DataFrame(
            columns=["participant_id", "study_group", *REPEAT_COLUMNS]
        )
    grouped = (
        work.groupby(
            ["participant_id", "study_accession", "timepoint"],
            dropna=False,
            observed=True,
        )["value"]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot(
        index=["participant_id", "study_accession"],
        columns="timepoint",
        values="value",
    )
    for timepoint in REPEAT_TIMEPOINTS:
        if timepoint not in pivot:
            pivot[timepoint] = np.nan
    paired = pivot.dropna(subset=list(REPEAT_TIMEPOINTS)).reset_index()
    if paired.empty:
        return pd.DataFrame(
            columns=["participant_id", "study_group", *REPEAT_COLUMNS]
        )
    left = np.log1p(paired[REPEAT_TIMEPOINTS[0]].to_numpy(dtype=float))
    right = np.log1p(paired[REPEAT_TIMEPOINTS[1]].to_numpy(dtype=float))
    paired[REPEAT_COLUMNS[0]] = right - left
    paired[REPEAT_COLUMNS[1]] = np.var(
        np.column_stack([left, right]), axis=1, ddof=0
    )
    return paired[
        ["participant_id", "study_accession", *REPEAT_COLUMNS]
    ].rename(columns={"study_accession": "study_group"})


def _repeat_support_audit(
    inputs: InputBundle,
    *,
    dataset: TaskDataset,
    splits: Sequence[NamedSplit],
) -> tuple[Mapping[str, Any], pd.DataFrame, pd.DataFrame]:
    train_repeat = _repeat_feature_table(inputs.tables["public_cytokine"])
    challenge_repeat = _repeat_feature_table(inputs.tables["challenge_cytokine"])
    counts_series = (
        train_repeat.groupby("study_group", observed=True)["participant_id"]
        .nunique()
        if not train_repeat.empty
        else pd.Series(dtype=int)
    )
    counts = {str(k): int(v) for k, v in counts_series.items()}
    supported = sorted(
        study
        for study, count in counts.items()
        if count >= REPEAT_MIN_PAIRED_PER_STUDY
    )
    fold_support = []
    full_gate = True
    for split in splits:
        train_studies = set(
            dataset.train.iloc[split.train_indices]["study_group"].astype(str)
        )
        source = sorted(train_studies.intersection(supported))
        ok = len(source) >= REPEAT_MIN_SOURCE_STUDIES_PER_FOLD
        full_gate = full_gate and ok
        fold_support.append(
            {
                "held_out_group": None
                if split.held_out_group is None
                else str(split.held_out_group),
                "eligible_source_studies": source,
                "eligible_source_study_count": int(len(source)),
                "gate_passed": bool(ok),
            }
        )
    challenge_paired = int(challenge_repeat["participant_id"].nunique()) if not challenge_repeat.empty else 0
    challenge_ok = challenge_paired >= REPEAT_MIN_PAIRED_PER_STUDY
    gate = bool(
        len(supported) >= 2
        and full_gate
        and challenge_ok
    )
    status = "ready" if gate else "data_limited"
    return (
        _json_safe(
            {
                "status": status,
                "required_timepoints": list(REPEAT_TIMEPOINTS),
                "feature_scope": "CXCL10-only log1p difference and two-point variance",
                "min_paired_subjects_per_study": REPEAT_MIN_PAIRED_PER_STUDY,
                "min_source_studies_per_outer_fold": REPEAT_MIN_SOURCE_STUDIES_PER_FOLD,
                "paired_subjects_by_historical_study": counts,
                "eligible_historical_studies": supported,
                "fold_support": fold_support,
                "challenge_paired_subjects": challenge_paired,
                "challenge_gate_passed": challenge_ok,
                "full_outer_contract_gate_passed": full_gate,
            }
        ),
        train_repeat,
        challenge_repeat,
    )


def _augment_with_repeat(
    dataset: TaskDataset,
    train_repeat: pd.DataFrame,
    challenge_repeat: pd.DataFrame,
) -> TaskDataset:
    train = dataset.train.merge(
        train_repeat,
        on=["participant_id", "study_group"],
        how="left",
        validate="one_to_one",
    )
    challenge = dataset.challenge.merge(
        challenge_repeat,
        on=["participant_id", "study_group"],
        how="left",
        validate="one_to_one",
    )
    augmented = TaskDataset(
        task=dataset.task,
        train=train,
        challenge=challenge,
        target_column=dataset.target_column,
        excluded_columns=dataset.excluded_columns,
        metadata={
            **dict(dataset.metadata),
            "e02_repeat_features": list(REPEAT_COLUMNS),
        },
    )
    augmented.validate()
    return augmented


def _availability_frame(
    dataset: TaskDataset,
    *,
    other_columns: Sequence[str],
    repeat_columns: Sequence[str] = (),
) -> pd.DataFrame:
    raw, _ = _raw_structured_features(
        dataset.train,
        other_columns=other_columns,
        repeat_columns=repeat_columns,
    )
    available = raw.notna().sum(axis=1).to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "participant_id": dataset.train["participant_id"].astype(str).to_numpy(),
            "subject_group": dataset.train["subject_group"].astype(str).to_numpy(),
            "study_group": dataset.train["study_group"].astype(str).to_numpy(),
            "target": pd.to_numeric(
                dataset.train[dataset.target_column], errors="coerce"
            ).to_numpy(dtype=float),
            "prediction": available,
        }
    )


def _metrics(frame: pd.DataFrame, *, scale: str) -> Mapping[str, Any]:
    return _metric_summary(frame, prediction_scale=scale)


def _condition_payload(
    *,
    name: str,
    frame: pd.DataFrame,
    scale: str,
    diagnostics: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    return _json_safe(
        {
            "name": name,
            "metrics": _metrics(frame, scale=scale),
            "diagnostics": diagnostics or {},
        }
    )


def _comparison_bundle(
    conditions: Mapping[str, pd.DataFrame],
    *,
    candidate_names: Sequence[str],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    comparisons: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    references = ("anchor", "b21")
    for candidate in candidate_names:
        for reference in references:
            key = f"{candidate}_vs_{reference}"
            comparisons[key] = _paired_comparison(
                conditions[candidate],
                conditions[reference],
                candidate_name=candidate,
                reference_name=reference,
            )
            sensitivity[key] = _sensitivity_28(
                conditions[candidate],
                conditions[reference],
                candidate_name=candidate,
                reference_name=reference,
            )
    comparisons["b21_vs_anchor"] = _paired_comparison(
        conditions["b21"],
        conditions["anchor"],
        candidate_name="b21",
        reference_name="anchor",
    )
    sensitivity["b21_vs_anchor"] = _sensitivity_28(
        conditions["b21"],
        conditions["anchor"],
        candidate_name="b21",
        reference_name="anchor",
    )
    return _json_safe(comparisons), _json_safe(sensitivity)


def run_strategy_e02(
    config: BaselineConfig,
    inputs: InputBundle,
) -> Mapping[str, Any]:
    """Run E02 fixed-condition Task1.1 structured logFC evaluation."""

    if config.baseline != "b021_taskwise_robust":
        raise DataContractError("E02 requires the B2.1 robust configuration")
    if str(config.section("selection").get("policy", "")) != "robust_v1":
        raise DataContractError("E02 requires selection.policy=robust_v1")
    datasets = build_b02_datasets(config, inputs)
    dataset = datasets["Task1.1"]
    if not isinstance(dataset, TaskDataset):
        raise DataContractError("E02 Task1.1 dataset type mismatch")
    _validate_task11_contract(dataset)
    splits = default_splits_for_task(dataset, random_state=config.random_state)

    anchor = _anchor_frame(dataset, splits)
    b21, b21_result = _fixed_b21(
        dataset,
        config=config,
        splits=splits,
    )
    anchor_residual = _fixed_anchor_residual(
        dataset,
        config=config,
        splits=splits,
    )
    other_columns = _structured_other_columns(dataset)
    structured, structured_diag = _fit_structured_oof(
        dataset,
        splits=splits,
        other_columns=other_columns,
    )
    structured_shuffle, shuffle_diag = _fit_structured_oof(
        dataset,
        splits=splits,
        other_columns=other_columns,
        shuffle_labels=True,
    )
    availability = _availability_frame(
        dataset,
        other_columns=other_columns,
    )

    conditions: dict[str, pd.DataFrame] = {
        "anchor": anchor,
        "b21": b21,
        "anchor_residual": anchor_residual,
        "structured_ridge": structured,
    }
    condition_payloads: dict[str, Any] = {
        "anchor": _condition_payload(
            name="same_readout_oriented_anchor",
            frame=anchor,
            scale="rank_score",
        ),
        "b21": _condition_payload(
            name="b21_pls_2",
            frame=b21,
            scale="raw_target",
            diagnostics={
                "selected_model": b21_result.selected_spec.name,
                "reproduced": b21_result.selected_spec.name == B21_MODEL,
            },
        ),
        "anchor_residual": _condition_payload(
            name="anchor_residual_pls_1_lambda0.25",
            frame=anchor_residual,
            scale="rank_score",
            diagnostics={
                "model": ANCHOR_RESIDUAL_MODEL,
                "lambda": ANCHOR_RESIDUAL_LAMBDA,
                "frozen_reference_source": "phase_a_anchor_residual_task11_task12",
            },
        ),
        "structured_ridge": _condition_payload(
            name="study_centered_shared_logfc_ridge_alpha10",
            frame=structured,
            scale="centered_logfc_score",
            diagnostics=structured_diag,
        ),
    }

    repeat_audit, train_repeat, challenge_repeat = _repeat_support_audit(
        inputs,
        dataset=dataset,
        splits=splits,
    )
    repeat_frame: pd.DataFrame | None = None
    repeat_diag: Mapping[str, Any] | None = None
    augmented: TaskDataset | None = None
    if repeat_audit["status"] == "ready":
        augmented = _augment_with_repeat(dataset, train_repeat, challenge_repeat)
        repeat_frame, repeat_diag = _fit_structured_oof(
            augmented,
            splits=splits,
            other_columns=other_columns,
            repeat_columns=REPEAT_COLUMNS,
        )
        conditions["structured_ridge_plus_repeat"] = repeat_frame
        condition_payloads["structured_ridge_plus_repeat"] = _condition_payload(
            name="study_centered_shared_logfc_ridge_alpha10_plus_repeat",
            frame=repeat_frame,
            scale="centered_logfc_score",
            diagnostics=repeat_diag,
        )

    candidate_names = ["anchor_residual", "structured_ridge"]
    if repeat_frame is not None:
        candidate_names.append("structured_ridge_plus_repeat")
    comparisons, sensitivity = _comparison_bundle(
        conditions,
        candidate_names=candidate_names,
    )
    if repeat_frame is not None:
        comparisons["structured_ridge_plus_repeat_vs_structured_ridge"] = _paired_comparison(
            repeat_frame,
            structured,
            candidate_name="structured_ridge_plus_repeat",
            reference_name="structured_ridge",
        )
        sensitivity["structured_ridge_plus_repeat_vs_structured_ridge"] = _sensitivity_28(
            repeat_frame,
            structured,
            candidate_name="structured_ridge_plus_repeat",
            reference_name="structured_ridge",
        )

    anchor_challenge = _challenge_anchor(dataset)
    b21_challenge = b21_result.challenge_predictions[
        ["participant_id", "prediction"]
    ].copy()
    ar_challenge = _fixed_anchor_residual_challenge(dataset, config=config)
    structured_challenge, structured_challenge_diag = _fit_structured_challenge(
        dataset,
        other_columns=other_columns,
    )
    challenge_agreement: dict[str, Any] = {
        "anchor_residual_vs_anchor": _challenge_agreement(
            ar_challenge, anchor_challenge
        ),
        "anchor_residual_vs_b21": _challenge_agreement(
            ar_challenge, b21_challenge
        ),
        "structured_ridge_vs_anchor": _challenge_agreement(
            structured_challenge, anchor_challenge
        ),
        "structured_ridge_vs_b21": _challenge_agreement(
            structured_challenge, b21_challenge
        ),
        "structured_ridge_diagnostics": structured_challenge_diag,
    }
    if repeat_frame is not None and augmented is not None:
        repeat_challenge, repeat_challenge_diag = _fit_structured_challenge(
            augmented,
            other_columns=other_columns,
            repeat_columns=REPEAT_COLUMNS,
        )
        challenge_agreement["structured_ridge_plus_repeat_vs_anchor"] = _challenge_agreement(
            repeat_challenge, anchor_challenge
        )
        challenge_agreement["structured_ridge_plus_repeat_vs_structured_ridge"] = _challenge_agreement(
            repeat_challenge, structured_challenge
        )
        challenge_agreement["structured_ridge_plus_repeat_diagnostics"] = repeat_challenge_diag

    structured_vs_anchor = comparisons["structured_ridge_vs_anchor"]
    structured_candidate = bool(
        structured_vs_anchor.get("passes_candidate_delta_heuristic")
        and not structured_vs_anchor.get("large_studies_requiring_review")
    )
    repeat_candidate = None
    if repeat_frame is not None:
        comp = comparisons["structured_ridge_plus_repeat_vs_anchor"]
        repeat_candidate = bool(
            comp.get("passes_candidate_delta_heuristic")
            and not comp.get("large_studies_requiring_review")
        )

    return _json_safe(
        {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "random_seed": RANDOM_SEED,
            "comparison_contract": "paired_subject_purged_v2",
            "task": "Task1.1",
            "cohort": {
                "train_rows": int(len(dataset.train)),
                "train_studies": int(dataset.train["study_group"].nunique()),
                "challenge_rows": int(len(dataset.challenge)),
                "split_count": int(len(splits)),
            },
            "fixed_conditions": {
                "b21_model": B21_MODEL,
                "anchor_residual_model": ANCHOR_RESIDUAL_MODEL,
                "anchor_residual_lambda": ANCHOR_RESIDUAL_LAMBDA,
                "structured_ridge_alpha": RIDGE_ALPHA,
                "structured_target": "logFC centered within each training study",
                "other_cytokine_representation": "one log1p column per non-CXCL10 common analyte",
                "explicit_baseline_term": "log(canonical Pre-vacc CXCL10)",
                "held_x_self_normalization": True,
                "held_y_used_for_preprocessing": False,
                "study_id_used_as_feature": False,
            },
            "conditions": condition_payloads,
            "comparisons": comparisons,
            "sensitivity_28": sensitivity,
            "negative_controls": {
                "availability_only": {
                    "metrics": _metrics(availability, scale="availability_count"),
                    "feature_count": int(
                        len(_raw_structured_features(
                            dataset.train, other_columns=other_columns
                        )[0].columns)
                    ),
                },
                "structured_subject_block_label_shuffle": {
                    "metrics": _metrics(
                        structured_shuffle,
                        scale="centered_logfc_score",
                    ),
                    "seed": RANDOM_SEED,
                    "scope": "within_training_study_subject_block",
                    "moved_subject_assignments_across_folds": int(
                        shuffle_diag["moved_subject_assignments_across_folds"]
                    ),
                },
            },
            "repeat_baseline_extension": {
                "audit": repeat_audit,
                "condition_executed": repeat_frame is not None,
                "candidate": repeat_candidate,
            },
            "challenge_agreement": challenge_agreement,
            "decision": {
                "structured_ridge_candidate_over_anchor": structured_candidate,
                "repeat_extension_candidate_over_anchor": repeat_candidate,
                "candidate_delta_threshold": 0.02,
                "large_study_decline_review_threshold": -0.10,
                "thresholds_are_decision_heuristics_not_significance_tests": True,
                "primary_reference_for_new_structure": "same_readout_oriented_anchor",
            },
            "contains_participant_identifiers": False,
            "contains_row_level_predictions": False,
            "leaderboard_used_for_selection": False,
            "competition_submission_attempted": False,
            "notes": [
                "The canonical Task1.1 fold-change target and denominator are unchanged.",
                "Held-study X self-normalization is declared transductive preprocessing and uses no held y.",
                "The structured Ridge does not receive study identity as a prediction feature.",
                "The repeat-baseline extension is skipped as data_limited unless every outer fold has at least two eligible source studies and the Challenge has paired -14/0 CXCL10 support.",
                "A single fixed label shuffle is a diagnostic, not an empirical null distribution.",
                "No model family is rejected solely because one fixed condition is negative.",
            ],
        }
    )
