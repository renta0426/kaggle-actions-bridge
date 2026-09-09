"""Synthetic, Competition-Data-free contract for E11a pairwise ranking."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .datasets import TaskDataset
from .metrics import safe_spearman
from .strategy_e11 import (
    MAX_TRANSFORMED_FEATURES,
    PAIRWISE_PARAMS,
    audit_e11a_feasibility,
    build_within_study_pairs,
    fit_pairwise_ranker,
    score_pairwise_ranker,
)

SEED = 20260910
STUDY_COUNTS = {"SDY180": 34, "SDY515": 16, "SDY519": 17, "SDY56": 60}
CPU_FEASIBILITY_SECONDS = 30.0


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
    study_offset = {
        "SDY180": 2.0,
        "SDY515": -1.0,
        "SDY519": 0.5,
        "SDY56": 3.0,
        "CHALLENGE": -0.3,
    }[study]
    latent = (
        3.0 * x[:, 0]
        - 2.0 * x[:, 1]
        + 0.5 * x[:, 2]
        + 0.2 * x[:, 0] * x[:, 3]
    )
    frame = pd.DataFrame(
        {
            "participant_id": [f"P{start + i:04d}" for i in range(n)],
            "subject_group": [f"S{start + i:04d}" for i in range(n)],
            "study_group": study,
            "cytokine_a": x[:, 0],
            "cytokine_b": x[:, 1],
            "cytokine_c": x[:, 2],
            "cytokine_d": x[:, 3],
            "age": 35.0 + 5.0 * x[:, 4],
            "biological_sex": sex,
        }
    )
    if include_target:
        frame["target"] = latent + study_offset + rng.normal(scale=0.15, size=n)
    return frame


def make_dataset() -> TaskDataset:
    rng = np.random.default_rng(SEED)
    parts = []
    offset = 0
    for study, count in STUDY_COUNTS.items():
        parts.append(_rows(rng, study, count, include_target=True, start=offset))
        offset += count
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
        target_column="target",
    )


def run_synthetic() -> dict:
    dataset = make_dataset()
    feasibility = audit_e11a_feasibility(dataset)

    numeric = dataset.train[
        ["cytokine_a", "cytokine_b", "cytokine_c", "cytokine_d", "age"]
    ].to_numpy(dtype=float)
    pair_x, pair_y, pair_audit = build_within_study_pairs(
        numeric,
        dataset.train["target"],
        dataset.train["study_group"],
        dataset.train["subject_group"],
    )

    held_study = "SDY56"
    train = dataset.train.loc[dataset.train["study_group"].ne(held_study)].copy()
    held = dataset.train.loc[dataset.train["study_group"].eq(held_study)].copy()
    training_view = TaskDataset(
        task=dataset.task,
        train=train,
        challenge=held.drop(columns=["target"]).copy(),
        target_column=dataset.target_column,
    )
    started = time.perf_counter()
    model = fit_pairwise_ranker(train, dataset=training_view)
    score = score_pairwise_ranker(
        model,
        held.drop(columns=["target"]).copy(),
        dataset=training_view,
    )
    elapsed = float(time.perf_counter() - started)
    metric = safe_spearman(held["target"].to_numpy(dtype=float), score)
    if metric.status != "ok":
        raise RuntimeError(f"E11a synthetic Spearman failed:{metric.status}")

    result = {
        "schema_version": 1,
        "experiment": "synthetic_e11a_pairwise_contract",
        "study_counts": dict(STUDY_COUNTS),
        "feasibility_ready": feasibility["ready_for_one_cpu_pairwise_comparison"],
        "outer_split_count": len(feasibility["outer_splits"]),
        "subject_overlap_max": max(
            split["subject_overlap"] for split in feasibility["outer_splits"]
        ),
        "pair_policy": pair_audit["pairs_within_study_only"],
        "cross_study_pair_labels_used": pair_audit["cross_study_pair_labels_used"],
        "directed_pairs": pair_audit["directed_pairs"],
        "positive_pairs": pair_audit["positive_pairs"],
        "negative_pairs": pair_audit["negative_pairs"],
        "antisymmetric_design_max_error": float(
            np.max(np.abs(pair_x[0::2] + pair_x[1::2]))
        ),
        "held_study": held_study,
        "held_rows": int(len(held)),
        "held_spearman": float(metric.value),
        "fit_and_score_seconds": elapsed,
        "cpu_feasibility_seconds": CPU_FEASIBILITY_SECONDS,
        "transformed_features": int(model.fit_audit["transformed_features"]),
        "max_transformed_features": MAX_TRANSFORMED_FEATURES,
        "classifier_params": dict(PAIRWISE_PARAMS),
        "held_outcomes_used_for_fit": False,
        "held_outcomes_used_for_scoring": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
    }
    if result["positive_pairs"] != result["negative_pairs"]:
        raise RuntimeError("E11a synthetic directed pair balance failed")
    if result["antisymmetric_design_max_error"] > 1e-12:
        raise RuntimeError("E11a synthetic pair antisymmetry failed")
    if result["subject_overlap_max"] != 0:
        raise RuntimeError("E11a synthetic subject-purge contract failed")
    if result["held_spearman"] < 0.80:
        raise RuntimeError("E11a synthetic ranking signal too weak")
    if result["fit_and_score_seconds"] > CPU_FEASIBILITY_SECONDS:
        raise RuntimeError("E11a synthetic CPU feasibility budget exceeded")
    if result["transformed_features"] > MAX_TRANSFORMED_FEATURES:
        raise RuntimeError("E11a synthetic transformed feature guard failed")
    return result


__all__ = ["make_dataset", "run_synthetic"]
