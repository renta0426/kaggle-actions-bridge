"""Strategy-v2 E11b: finite TabPFN-3 Task1.1 model-family comparison.

E11b changes one modeling dimension relative to the frozen Task1.1 B2.1
incumbent: the regressor family. It preserves the compact Task1.1 feature
space, the B2.1 log-target contract, and the same subject-purged
leave-one-study-out evaluation.

The real TabPFN dependency is intentionally lazy. Secret-free science CI can
exercise the full data/leakage/scoring contract with an injected deterministic
stand-in, while the protected Kaggle runtime must provide the pinned official
TabPFN package and the exact official checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns, require_finite
from .datasets import TaskDataset
from .evaluation import default_splits_for_task
from .metrics import percentile_rank, safe_spearman
from .models import ModelSpec, fit_final_model, prepare_model_frame
from .runner import InputBundle, build_b02_datasets

EXPERIMENT = "strategy_v2_e11b_task11_tabpfn3"
COMPARISON_CONTRACT = "paired_subject_purged_v2"
TASK = "Task1.1"
BASE_MODEL_SET = "task_11"
BASE_MODEL = "pls_2"
TARGET_TRANSFORM = "log"

EXPECTED_TRAIN_ROWS = 127
EXPECTED_TRAIN_STUDIES = 4
EXPECTED_CHALLENGE_ROWS = 40
MAX_RAW_FEATURES = 200

TABPFN_PACKAGE_VERSION = "8.5.0"
TABPFN_SOURCE_COMMIT = "9ed44abd5882140b88c9f2816c5791987ce059b9"
TABPFN_WHEEL_FILENAME = "tabpfn-8.5.0-py3-none-any.whl"
TABPFN_WHEEL_SHA256 = "4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0"

TABPFN_KAGGLE_MODEL_SOURCE = "prior-labsai/tabpfn-3/pytorch/default/1"
TABPFN_CHECKPOINT_FILENAME = "tabpfn-v3-regressor-v3_default.ckpt"
TABPFN_CHECKPOINT_BYTES = 233_289_807
TABPFN_CHECKPOINT_SHA256 = "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301"

TABPFN_LICENSE = {
    "model_license": "tabpfn-3-license-v1.0",
    "permitted_scope": "non-commercial data-science competition and academic research",
    "weight_redistribution_by_project": False,
    "official_kaggle_model_input": True,
}

TABPFN_PARAMS: Mapping[str, Any] = {
    "n_estimators": 8,
    "device": "cpu",
    "fit_mode": "fit_preprocessors",
    "memory_saving_mode": "auto",
    "random_state": 20260910,
    "n_preprocessing_jobs": 1,
    "show_progress_bar": False,
    "ignore_pretraining_limits": False,
}

PROMOTION_MEAN_DELTA = 0.02
PROMOTION_MIN_STUDY_DELTA = -0.10

EstimatorFactory = Callable[..., Any]


@dataclass
class TabPFNFit:
    estimator: Any
    feature_columns: tuple[str, ...]
    categorical_indices: tuple[int, ...]
    fit_audit: Mapping[str, Any]


def _find_base_spec(config: BaselineConfig) -> ModelSpec:
    matches = [spec for spec in config.model_specs(BASE_MODEL_SET) if spec.name == BASE_MODEL]
    if len(matches) != 1:
        raise DataContractError(
            f"E11b frozen base {BASE_MODEL!r} resolved to {len(matches)} specs"
        )
    spec = matches[0]
    if spec.target_transform != TARGET_TRANSFORM:
        raise DataContractError(
            f"E11b expected frozen base target transform {TARGET_TRANSFORM!r}, "
            f"found {spec.target_transform!r}"
        )
    return spec


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tabpfn_checkpoint(path: Path) -> dict[str, Any]:
    """Verify exact official TabPFN-3 regression checkpoint identity."""
    path = Path(path)
    if path.name != TABPFN_CHECKPOINT_FILENAME:
        raise DataContractError(
            f"E11b checkpoint filename mismatch:{path.name!r}"
        )
    if not path.is_file():
        raise DataContractError("E11b checkpoint path is not a file")
    size = int(path.stat().st_size)
    if size != TABPFN_CHECKPOINT_BYTES:
        raise DataContractError(
            f"E11b checkpoint byte mismatch:{size}!={TABPFN_CHECKPOINT_BYTES}"
        )
    digest = _sha256_file(path)
    if digest != TABPFN_CHECKPOINT_SHA256:
        raise DataContractError("E11b checkpoint SHA-256 mismatch")
    return {
        "filename": TABPFN_CHECKPOINT_FILENAME,
        "bytes": size,
        "sha256": digest,
        "kaggle_model_source": TABPFN_KAGGLE_MODEL_SOURCE,
        "verified": True,
    }


def _tabpfn_frames(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    dataset: TaskDataset,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[str], list[str]]:
    """Build the unchanged compact feature frame without scaling or one-hot encoding."""
    excluded = tuple(
        dict.fromkeys(
            [
                *dataset.excluded_columns,
                dataset.target_column,
                "participant_id",
                "subject",
                "study_accession",
                "subject_context",
                "study_accession_context",
                "subject_group",
                "study_group",
                "row_index",
            ]
        )
    )
    train_x, numeric, categorical = prepare_model_frame(
        training,
        excluded_columns=excluded,
    )
    pred_x, pred_numeric, pred_categorical = prepare_model_frame(
        prediction,
        excluded_columns=excluded,
    )
    if list(train_x.columns) != list(pred_x.columns):
        raise DataContractError(
            "E11b train/prediction compact feature order differs"
        )
    if numeric != pred_numeric or categorical != pred_categorical:
        raise DataContractError("E11b train/prediction inferred feature types differ")
    if not 0 < train_x.shape[1] <= MAX_RAW_FEATURES:
        raise DataContractError(
            f"E11b raw feature count {train_x.shape[1]} outside 1..{MAX_RAW_FEATURES}"
        )
    forbidden = {
        "participant_id",
        "subject",
        "study_accession",
        "subject_group",
        "study_group",
        dataset.target_column,
    }
    leaked = forbidden.intersection(train_x.columns)
    if leaked:
        raise DataContractError(f"E11b forbidden features escaped exclusion:{sorted(leaked)}")
    categorical_indices = [int(train_x.columns.get_loc(column)) for column in categorical]
    return train_x, pred_x, categorical_indices, numeric, categorical


def _forward_target(values: Sequence[float]) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    require_finite(raw, name="E11b.raw_target")
    if (raw <= 0.0).any():
        raise DataContractError("E11b log target requires strictly positive values")
    transformed = np.log(raw)
    require_finite(transformed, name="E11b.log_target")
    return transformed


def _inverse_target(values: Sequence[float]) -> np.ndarray:
    transformed = np.asarray(values, dtype=float)
    require_finite(transformed, name="E11b.log_prediction")
    raw = np.exp(np.clip(transformed, -50.0, 50.0))
    require_finite(raw, name="E11b.raw_prediction")
    return np.maximum(raw, 0.0)


def _build_real_estimator(
    *,
    checkpoint_path: Path,
    categorical_indices: Sequence[int],
) -> Any:
    try:
        found_version = importlib.metadata.version("tabpfn")
    except importlib.metadata.PackageNotFoundError as exc:
        raise DataContractError("E11b tabpfn package is not installed") from exc
    if found_version != TABPFN_PACKAGE_VERSION:
        raise DataContractError(
            f"E11b tabpfn version mismatch:{found_version}!={TABPFN_PACKAGE_VERSION}"
        )
    verify_tabpfn_checkpoint(checkpoint_path)
    try:
        from tabpfn import TabPFNRegressor
    except Exception as exc:  # noqa: BLE001 - converted to explicit runtime contract failure
        raise DataContractError(f"E11b cannot import TabPFNRegressor:{type(exc).__name__}") from exc
    return TabPFNRegressor(
        model_path=str(checkpoint_path),
        categorical_features_indices=list(categorical_indices),
        **dict(TABPFN_PARAMS),
    )


def fit_tabpfn_regressor(
    training: pd.DataFrame,
    *,
    dataset: TaskDataset,
    checkpoint_path: Path | None,
    estimator_factory: EstimatorFactory | None = None,
) -> TabPFNFit:
    """Fit E11b from source-training rows only."""
    require_columns(
        training,
        [dataset.target_column, "subject_group", "study_group"],
        table_name="E11b TabPFN training",
    )
    train_x, _, categorical_indices, numeric, categorical = _tabpfn_frames(
        training,
        training.iloc[:1].drop(columns=[dataset.target_column], errors="ignore").copy(),
        dataset=dataset,
    )
    target = _forward_target(
        pd.to_numeric(training[dataset.target_column], errors="coerce").to_numpy(dtype=float)
    )

    checkpoint_audit: dict[str, Any]
    if estimator_factory is None:
        if checkpoint_path is None:
            raise DataContractError("E11b real TabPFN fit requires checkpoint_path")
        checkpoint_audit = verify_tabpfn_checkpoint(Path(checkpoint_path))
        estimator = _build_real_estimator(
            checkpoint_path=Path(checkpoint_path),
            categorical_indices=categorical_indices,
        )
        backend = "tabpfn"
        package_verified = True
    else:
        estimator = estimator_factory(
            model_path=checkpoint_path,
            categorical_features_indices=tuple(categorical_indices),
            params=dict(TABPFN_PARAMS),
        )
        checkpoint_audit = {
            "verified": False,
            "synthetic_injected_backend": True,
            "expected_filename": TABPFN_CHECKPOINT_FILENAME,
            "expected_bytes": TABPFN_CHECKPOINT_BYTES,
            "expected_sha256": TABPFN_CHECKPOINT_SHA256,
        }
        backend = "synthetic_injected"
        package_verified = False

    started = time.perf_counter()
    estimator.fit(train_x, target)
    fit_seconds = float(time.perf_counter() - started)
    return TabPFNFit(
        estimator=estimator,
        feature_columns=tuple(train_x.columns),
        categorical_indices=tuple(categorical_indices),
        fit_audit={
            "backend": backend,
            "package_version_expected": TABPFN_PACKAGE_VERSION,
            "package_source_commit": TABPFN_SOURCE_COMMIT,
            "package_verified": package_verified,
            "wheel_filename": TABPFN_WHEEL_FILENAME,
            "wheel_sha256": TABPFN_WHEEL_SHA256,
            "checkpoint": checkpoint_audit,
            "model_source": TABPFN_KAGGLE_MODEL_SOURCE,
            "training_rows": int(len(training)),
            "training_subjects": int(training["subject_group"].nunique()),
            "training_studies": int(training["study_group"].nunique()),
            "raw_features": int(train_x.shape[1]),
            "numeric_features": int(len(numeric)),
            "categorical_features": int(len(categorical)),
            "categorical_indices": list(categorical_indices),
            "target_transform": TARGET_TRANSFORM,
            "tabpfn_params": dict(TABPFN_PARAMS),
            "held_outcomes_used_for_fit": False,
            "fit_seconds": fit_seconds,
        },
    )


def score_tabpfn_regressor(
    model: TabPFNFit,
    prediction: pd.DataFrame,
    *,
    dataset: TaskDataset,
) -> np.ndarray:
    """Predict without accepting any outcome argument."""
    if dataset.target_column in prediction.columns:
        raise DataContractError(
            "E11b scoring frame must not contain the held/challenge outcome"
        )
    actual_x, _, actual_cat, _, _ = _tabpfn_frames(
        prediction,
        prediction,
        dataset=dataset,
    )
    if tuple(actual_x.columns) != model.feature_columns:
        raise DataContractError("E11b scoring feature order differs from fitted model")
    if tuple(actual_cat) != model.categorical_indices:
        raise DataContractError("E11b scoring categorical index contract changed")
    transformed = np.asarray(model.estimator.predict(actual_x), dtype=float).reshape(-1)
    return _inverse_target(transformed)


def _base_score(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    dataset: TaskDataset,
    spec: ModelSpec,
) -> np.ndarray:
    _, values = fit_final_model(
        training,
        prediction,
        target_column=dataset.target_column,
        spec=spec,
        excluded_columns=dataset.excluded_columns,
    )
    values = np.asarray(values, dtype=float).reshape(-1)
    require_finite(values, name="E11b.base_score")
    return values


def _spearman_value(target: Sequence[float], score: Sequence[float]) -> float | None:
    metric = safe_spearman(target, score)
    return float(metric.value) if metric.status == "ok" else None


def _promotion(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [
        float(fold["tabpfn_delta"])
        for fold in folds
        if fold.get("tabpfn_delta") is not None
        and np.isfinite(float(fold["tabpfn_delta"]))
    ]
    if not deltas:
        return {
            "passed": False,
            "reason": "no_usable_held_studies",
            "mean_delta": None,
            "minimum_delta": None,
            "wins": 0,
            "required_wins": 0,
        }
    wins = int(sum(value > 0 for value in deltas))
    required = len(deltas) // 2 + 1
    mean_delta = float(np.mean(deltas))
    minimum_delta = float(np.min(deltas))
    return {
        "passed": bool(
            mean_delta >= PROMOTION_MEAN_DELTA
            and minimum_delta >= PROMOTION_MIN_STUDY_DELTA
            and wins >= required
        ),
        "usable_held_studies": int(len(deltas)),
        "mean_delta": mean_delta,
        "median_delta": float(np.median(deltas)),
        "minimum_delta": minimum_delta,
        "maximum_delta": float(np.max(deltas)),
        "wins": wins,
        "required_wins": required,
        "mean_delta_threshold": PROMOTION_MEAN_DELTA,
        "minimum_study_delta_threshold": PROMOTION_MIN_STUDY_DELTA,
    }


def _challenge_agreement(base: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    base_rank = percentile_rank(base)
    candidate_rank = percentile_rank(candidate)
    return {
        "rank_spearman": safe_spearman(base_rank, candidate_rank).to_dict(),
        "changed_rank_count": int(np.sum(np.abs(base_rank - candidate_rank) > 1e-12)),
        "mean_absolute_percentile_shift": float(
            np.mean(np.abs(base_rank - candidate_rank))
        ),
        "max_absolute_percentile_shift": float(
            np.max(np.abs(base_rank - candidate_rank))
        ),
    }


def audit_e11b_feasibility(dataset: TaskDataset) -> dict[str, Any]:
    """Outcome-safe E11b structural/leakage gate."""
    dataset.validate()
    if dataset.task != TASK:
        raise DataContractError(f"E11b only supports {TASK}")
    if len(dataset.train) != EXPECTED_TRAIN_ROWS:
        raise DataContractError(
            f"E11b expected {EXPECTED_TRAIN_ROWS} train rows, found {len(dataset.train)}"
        )
    if int(dataset.train["study_group"].nunique()) != EXPECTED_TRAIN_STUDIES:
        raise DataContractError("E11b Task1.1 study count changed")
    if len(dataset.challenge) != EXPECTED_CHALLENGE_ROWS:
        raise DataContractError("E11b Challenge row count changed")

    train_x, challenge_x, categorical_indices, numeric, categorical = _tabpfn_frames(
        dataset.train,
        dataset.challenge,
        dataset=dataset,
    )
    if list(train_x.columns) != list(challenge_x.columns):
        raise DataContractError("E11b feature alignment changed")

    split_audit: list[dict[str, Any]] = []
    splits = default_splits_for_task(dataset)
    for split in splits:
        source = dataset.train.iloc[split.train_indices]
        held = dataset.train.iloc[split.validation_indices]
        source_subjects = set(source["subject_group"].astype(str))
        held_subjects = set(held["subject_group"].astype(str))
        overlap = source_subjects.intersection(held_subjects)
        if overlap:
            raise DataContractError("E11b subject purge failed")
        if int(source["study_group"].nunique()) != EXPECTED_TRAIN_STUDIES - 1:
            raise DataContractError("E11b source study count changed in outer fold")
        split_audit.append(
            {
                "split": str(split.name),
                "held_study": str(split.held_out_group),
                "training_rows": int(len(source)),
                "held_rows": int(len(held)),
                "training_studies": int(source["study_group"].nunique()),
                "subject_overlap": 0,
                "held_outcomes_used_for_fit": False,
                "held_outcomes_used_for_scoring": False,
            }
        )
    return {
        "ready_for_one_cpu_tabpfn_comparison": True,
        "train_rows": int(len(dataset.train)),
        "train_studies": int(dataset.train["study_group"].nunique()),
        "challenge_rows": int(len(dataset.challenge)),
        "raw_features": int(train_x.shape[1]),
        "numeric_features": int(len(numeric)),
        "categorical_features": int(len(categorical)),
        "categorical_indices": list(categorical_indices),
        "outer_splits": split_audit,
        "target_transform": TARGET_TRANSFORM,
        "package_version": TABPFN_PACKAGE_VERSION,
        "source_commit": TABPFN_SOURCE_COMMIT,
        "wheel_sha256": TABPFN_WHEEL_SHA256,
        "kaggle_model_source": TABPFN_KAGGLE_MODEL_SOURCE,
        "checkpoint_filename": TABPFN_CHECKPOINT_FILENAME,
        "checkpoint_bytes": TABPFN_CHECKPOINT_BYTES,
        "checkpoint_sha256": TABPFN_CHECKPOINT_SHA256,
        "tabpfn_params": dict(TABPFN_PARAMS),
    }


def evaluate_task_e11b(
    dataset: TaskDataset,
    *,
    config: BaselineConfig,
    checkpoint_path: Path | None,
    estimator_factory: EstimatorFactory | None = None,
) -> dict[str, Any]:
    feasibility = audit_e11b_feasibility(dataset)
    base_spec = _find_base_spec(config)
    folds: list[dict[str, Any]] = []

    for split in default_splits_for_task(dataset):
        source = dataset.train.iloc[split.train_indices].copy()
        held = dataset.train.iloc[split.validation_indices].copy()
        held_features = held.drop(columns=[dataset.target_column]).copy()
        target = pd.to_numeric(
            held[dataset.target_column], errors="coerce"
        ).to_numpy(dtype=float)
        require_finite(target, name="E11b.held_target")

        base = _base_score(source, held_features, dataset=dataset, spec=base_spec)
        model = fit_tabpfn_regressor(
            source,
            dataset=dataset,
            checkpoint_path=checkpoint_path,
            estimator_factory=estimator_factory,
        )
        candidate = score_tabpfn_regressor(
            model,
            held_features,
            dataset=dataset,
        )
        base_spearman = _spearman_value(target, base)
        tabpfn_spearman = _spearman_value(target, candidate)
        delta = (
            float(tabpfn_spearman - base_spearman)
            if base_spearman is not None and tabpfn_spearman is not None
            else None
        )
        folds.append(
            {
                "held_study": str(split.held_out_group),
                "n": int(len(held)),
                "base_spearman": base_spearman,
                "tabpfn_spearman": tabpfn_spearman,
                "tabpfn_delta": delta,
                "fit_audit": dict(model.fit_audit),
                "held_outcomes_used_for_fit": False,
                "held_outcomes_used_for_scoring": False,
            }
        )

    base_challenge = _base_score(
        dataset.train,
        dataset.challenge,
        dataset=dataset,
        spec=base_spec,
    )
    final_model = fit_tabpfn_regressor(
        dataset.train,
        dataset=dataset,
        checkpoint_path=checkpoint_path,
        estimator_factory=estimator_factory,
    )
    tabpfn_challenge = score_tabpfn_regressor(
        final_model,
        dataset.challenge,
        dataset=dataset,
    )
    promotion = _promotion(folds)
    return {
        "task": TASK,
        "base_contract": {
            "kind": "b21",
            "model": BASE_MODEL,
            "target_transform": TARGET_TRANSFORM,
            "current_competition_incumbent": True,
        },
        "folds": folds,
        "candidate": {
            "name": "tabpfn3_default_cpu",
            "promotion": promotion,
            "challenge_agreement_vs_base": _challenge_agreement(
                base_challenge, tabpfn_challenge
            ),
            "competition_candidate": bool(promotion["passed"]),
            "public_probe_authorized": False,
        },
        "final_fit_audit": dict(final_model.fit_audit),
        "challenge_rows": int(len(dataset.challenge)),
        "feasibility": feasibility,
    }


def run_e11b(
    config: BaselineConfig,
    inputs: InputBundle,
    *,
    checkpoint_path: Path | None,
    estimator_factory: EstimatorFactory | None = None,
) -> dict[str, Any]:
    if config.baseline != "b021_taskwise_robust":
        raise DataContractError("E11b requires the B2.1 robust config")
    if str(config.section("selection").get("policy", "")) != "robust_v1":
        raise DataContractError("E11b requires robust_v1 selection policy")
    datasets = build_b02_datasets(config, inputs)
    dataset = datasets.get(TASK)
    if not isinstance(dataset, TaskDataset):
        raise DataContractError("E11b missing Task1.1 compact dataset")
    task = evaluate_task_e11b(
        dataset,
        config=config,
        checkpoint_path=checkpoint_path,
        estimator_factory=estimator_factory,
    )
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "comparison_contract": COMPARISON_CONTRACT,
        "frozen_conditions": {
            "task": TASK,
            "base_model": BASE_MODEL,
            "target_transform": TARGET_TRANSFORM,
            "feature_space": "unchanged_compact_b21_task11",
            "max_raw_features": MAX_RAW_FEATURES,
            "tabpfn_package_version": TABPFN_PACKAGE_VERSION,
            "tabpfn_source_commit": TABPFN_SOURCE_COMMIT,
            "tabpfn_wheel_filename": TABPFN_WHEEL_FILENAME,
            "tabpfn_wheel_sha256": TABPFN_WHEEL_SHA256,
            "kaggle_model_source": TABPFN_KAGGLE_MODEL_SOURCE,
            "checkpoint_filename": TABPFN_CHECKPOINT_FILENAME,
            "checkpoint_bytes": TABPFN_CHECKPOINT_BYTES,
            "checkpoint_sha256": TABPFN_CHECKPOINT_SHA256,
            "tabpfn_params": dict(TABPFN_PARAMS),
            "promotion_mean_delta": PROMOTION_MEAN_DELTA,
            "promotion_minimum_study_delta": PROMOTION_MIN_STUDY_DELTA,
            "promotion_requires_strict_majority_wins": True,
        },
        "external_model_license": dict(TABPFN_LICENSE),
        "task": task,
        "cross_study_target_scale_redefined": False,
        "held_outcomes_used_for_fit": False,
        "held_outcomes_used_for_scoring": False,
        "automatic_compute_retries": 0,
        "leaderboard_used_for_selection": False,
        "public_probe_authorized": False,
        "competition_submission_attempted": False,
        "incumbent_changed": False,
    }


__all__ = [
    "EXPERIMENT",
    "TABPFN_CHECKPOINT_BYTES",
    "TABPFN_CHECKPOINT_FILENAME",
    "TABPFN_CHECKPOINT_SHA256",
    "TABPFN_KAGGLE_MODEL_SOURCE",
    "TABPFN_PACKAGE_VERSION",
    "TABPFN_PARAMS",
    "TABPFN_SOURCE_COMMIT",
    "TABPFN_WHEEL_FILENAME",
    "TABPFN_WHEEL_SHA256",
    "TARGET_TRANSFORM",
    "audit_e11b_feasibility",
    "evaluate_task_e11b",
    "fit_tabpfn_regressor",
    "run_e11b",
    "score_tabpfn_regressor",
    "verify_tabpfn_checkpoint",
]
