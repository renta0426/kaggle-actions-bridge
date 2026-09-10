"""Competition-Data-free synthetic contract for E11b TabPFN plumbing."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .configuration import BaselineConfig
from .datasets import TaskDataset
from .strategy_e11b import (
    MAX_RAW_FEATURES,
    TABPFN_CHECKPOINT_BYTES,
    TABPFN_CHECKPOINT_FILENAME,
    TABPFN_CHECKPOINT_SHA256,
    TABPFN_KAGGLE_MODEL_SOURCE,
    TABPFN_PACKAGE_VERSION,
    TABPFN_PARAMS,
    TABPFN_SOURCE_COMMIT,
    TABPFN_WHEEL_SHA256,
    TARGET_TRANSFORM,
    audit_e11b_feasibility,
    evaluate_task_e11b,
)

SEED = 20260910
STUDY_COUNTS = {"SDY180": 34, "SDY515": 16, "SDY519": 17, "SDY56": 60}
CPU_FEASIBILITY_SECONDS = 30.0


class SyntheticTabPFNStandIn:
    """Deterministic sklearn regressor with a TabPFN-compatible surface."""

    def __init__(
        self,
        *,
        categorical_features_indices: tuple[int, ...],
        params: dict,
    ) -> None:
        self.categorical_features_indices = tuple(categorical_features_indices)
        self.params = dict(params)
        self.columns_: list[str] | None = None
        self.model = Ridge(alpha=1.0)

    @staticmethod
    def _encode(frame: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.get_dummies(frame.copy(), dummy_na=True, dtype=float)
        return encoded.astype(float)

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "SyntheticTabPFNStandIn":
        encoded = self._encode(x)
        self.columns_ = list(encoded.columns)
        self.model.fit(encoded, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.columns_ is None:
            raise RuntimeError("synthetic stand-in is not fitted")
        encoded = self._encode(x).reindex(columns=self.columns_, fill_value=0.0)
        return np.asarray(self.model.predict(encoded), dtype=float)


def synthetic_estimator_factory(
    *,
    model_path,
    categorical_features_indices,
    params,
):
    if model_path is not None:
        raise RuntimeError("synthetic E11b must not consume a checkpoint")
    if dict(params) != dict(TABPFN_PARAMS):
        raise RuntimeError("synthetic E11b parameter contract changed")
    return SyntheticTabPFNStandIn(
        categorical_features_indices=tuple(categorical_features_indices),
        params=dict(params),
    )


def _rows(
    rng: np.random.Generator,
    study: str,
    n: int,
    *,
    include_target: bool,
    start: int,
) -> pd.DataFrame:
    x = rng.normal(size=(n, 6))
    sex = np.where(np.arange(n) % 2 == 0, "F", "M")
    race = np.where(np.arange(n) % 3 == 0, "A", "B")
    log_target = (
        1.8
        + 0.45 * x[:, 0]
        - 0.30 * x[:, 1]
        + 0.12 * x[:, 2]
        + 0.08 * (sex == "F").astype(float)
        + 0.05 * (race == "A").astype(float)
        + rng.normal(scale=0.04, size=n)
    )
    frame = pd.DataFrame(
        {
            "participant_id": [f"P{start+i:05d}" for i in range(n)],
            "subject_group": [f"S{start+i:05d}" for i in range(n)],
            "study_group": study,
            "cytokine_a": x[:, 0],
            "cytokine_b": x[:, 1],
            "cytokine_c": x[:, 2],
            "cytokine_d": x[:, 3],
            "age": 40.0 + 8.0 * x[:, 4],
            "age_missing": 0,
            "biological_sex": sex,
            "race": race,
        }
    )
    if include_target:
        frame["target_value"] = np.exp(log_target)
    return frame


def make_dataset() -> TaskDataset:
    rng = np.random.default_rng(SEED)
    parts: list[pd.DataFrame] = []
    start = 0
    for study, count in STUDY_COUNTS.items():
        parts.append(_rows(rng, study, count, include_target=True, start=start))
        start += count
    challenge = _rows(
        rng,
        "CHALLENGE",
        40,
        include_target=False,
        start=10_000,
    )
    return TaskDataset(
        task="Task1.1",
        train=pd.concat(parts, ignore_index=True),
        challenge=challenge,
        target_column="target_value",
    )


def make_config() -> BaselineConfig:
    root = Path(".").resolve()
    raw = {
        "schema_version": 1,
        "baseline": "b021_taskwise_robust",
        "competition_slug": "synthetic",
        "selection": {"policy": "robust_v1"},
        "model_sets": {
            "task_11": [
                {
                    "name": "pls_2",
                    "family": "pls",
                    "params": {"n_components": 2},
                    "target_transform": "log",
                    "clip_min": 0.0,
                }
            ]
        },
    }
    return BaselineConfig(
        source_path=root / "synthetic.yaml",
        repository_root=root,
        raw=raw,
        baseline="b021_taskwise_robust",
        competition_slug="synthetic",
        random_state=SEED,
        verify_md5=False,
        data_dir=root,
        external_dir=root,
        artifacts_dir=root,
    )


def run_synthetic() -> dict:
    dataset = make_dataset()
    feasibility = audit_e11b_feasibility(dataset)
    started = time.perf_counter()
    task = evaluate_task_e11b(
        dataset,
        config=make_config(),
        checkpoint_path=None,
        estimator_factory=synthetic_estimator_factory,
    )
    elapsed = float(time.perf_counter() - started)

    folds = task["folds"]
    result = {
        "schema_version": 1,
        "experiment": "synthetic_e11b_tabpfn_contract",
        "study_counts": dict(STUDY_COUNTS),
        "feasibility_ready": feasibility["ready_for_one_cpu_tabpfn_comparison"],
        "outer_split_count": len(feasibility["outer_splits"]),
        "subject_overlap_max": max(
            split["subject_overlap"] for split in feasibility["outer_splits"]
        ),
        "raw_features": feasibility["raw_features"],
        "categorical_features": feasibility["categorical_features"],
        "target_transform": feasibility["target_transform"],
        "package_version": TABPFN_PACKAGE_VERSION,
        "source_commit": TABPFN_SOURCE_COMMIT,
        "wheel_sha256": TABPFN_WHEEL_SHA256,
        "kaggle_model_source": TABPFN_KAGGLE_MODEL_SOURCE,
        "checkpoint_filename": TABPFN_CHECKPOINT_FILENAME,
        "checkpoint_bytes": TABPFN_CHECKPOINT_BYTES,
        "checkpoint_sha256": TABPFN_CHECKPOINT_SHA256,
        "tabpfn_params": dict(TABPFN_PARAMS),
        "fold_count": len(folds),
        "finite_fold_scores": all(
            np.isfinite(float(fold["tabpfn_spearman"])) for fold in folds
        ),
        "all_synthetic_backends": all(
            fold["fit_audit"]["backend"] == "synthetic_injected" for fold in folds
        ),
        "checkpoint_verified_in_synthetic": any(
            bool(fold["fit_audit"]["checkpoint"].get("verified")) for fold in folds
        ),
        "challenge_rank_agreement_status": task["candidate"][
            "challenge_agreement_vs_base"
        ]["rank_spearman"]["status"],
        "fit_and_score_seconds": elapsed,
        "cpu_feasibility_seconds": CPU_FEASIBILITY_SECONDS,
        "held_outcomes_used_for_fit": False,
        "held_outcomes_used_for_scoring": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
    }
    if result["subject_overlap_max"] != 0:
        raise RuntimeError("E11b synthetic subject-purge contract failed")
    if not result["finite_fold_scores"]:
        raise RuntimeError("E11b synthetic fold score is non-finite")
    if not result["all_synthetic_backends"] or result["checkpoint_verified_in_synthetic"]:
        raise RuntimeError("E11b synthetic backend boundary failed")
    if result["raw_features"] > MAX_RAW_FEATURES:
        raise RuntimeError("E11b synthetic raw-feature guard failed")
    if result["categorical_features"] < 1:
        raise RuntimeError("E11b synthetic categorical-index contract not exercised")
    if result["target_transform"] != TARGET_TRANSFORM:
        raise RuntimeError("E11b synthetic target-transform contract changed")
    if result["fit_and_score_seconds"] > CPU_FEASIBILITY_SECONDS:
        raise RuntimeError("E11b synthetic CPU feasibility budget exceeded")
    if result["challenge_rank_agreement_status"] != "ok":
        raise RuntimeError("E11b synthetic challenge-rank diagnostic failed")
    return result


__all__ = [
    "make_config",
    "make_dataset",
    "run_synthetic",
    "synthetic_estimator_factory",
]
