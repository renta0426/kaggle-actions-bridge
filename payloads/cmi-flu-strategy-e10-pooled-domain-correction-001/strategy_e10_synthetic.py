"""Competition-data-free synthetic contract for E10 pooled domain correction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import TaskDataset
from .strategy_e10 import (
    _apply_correction,
    _promotion,
    _shrunk_deviation_predict,
    _weighted_shared_predict,
    xonly_source_weights,
)


def _frame(study: str, n: int, *, shift: float) -> pd.DataFrame:
    index = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "participant_id": [f"ROW_{study}_{i}" for i in range(n)],
            "subject_group": [f"SUB_{study}_{i}" for i in range(n)],
            "study_group": study,
            "age": 30.0 + shift + index,
            "biological_sex": np.where((index.astype(int) % 2) == 0, "F", "M"),
            "race": np.where((index.astype(int) % 3) == 0, "A", "B"),
            "flow_rank__Classical_monocytes": np.clip(
                (index + 1.0) / (n + 1.0) + 0.02 * shift, 0.0, 1.0
            ),
            "flow_aux": 0.25 * index + shift,
            "target": 0.1 * index + shift,
        }
    )


def run_synthetic() -> dict:
    training = pd.concat(
        [
            _frame("A", 12, shift=0.0),
            _frame("B", 12, shift=1.0),
            _frame("C", 12, shift=2.0),
        ],
        ignore_index=True,
    )
    target = _frame("T", 8, shift=0.8).drop(columns=["target"])
    dataset = TaskDataset(
        task="Task1.2",
        train=training,
        challenge=target,
        target_column="target",
    )
    residual = (
        0.04 * pd.to_numeric(training["flow_aux"]).to_numpy(dtype=float)
        - 0.01 * pd.to_numeric(training["age"]).to_numpy(dtype=float)
    )

    weights = xonly_source_weights(training, target, assay_prefix="flow_")
    row_weight = np.array(
        [weights["source_weights"][study] for study in training["study_group"].astype(str)],
        dtype=float,
    )
    weighted, weighted_fit = _weighted_shared_predict(
        training,
        target,
        dataset=dataset,
        residual=residual,
        sample_weight=row_weight,
    )
    hierarchical, hierarchical_fit = _shrunk_deviation_predict(
        training,
        target,
        dataset=dataset,
        residual=residual,
    )

    base = np.linspace(0.1, 0.9, len(target))
    candidate, movement = _apply_correction(base, weighted)
    passing_folds = [
        {"xonly_weighted_shared_delta": 0.03},
        {"xonly_weighted_shared_delta": 0.025},
        {"xonly_weighted_shared_delta": 0.01},
    ]
    promotion = _promotion(passing_folds, "xonly_weighted_shared")

    return {
        "schema_version": 1,
        "experiment": "synthetic_e10_contract",
        "source_weight_min": min(weights["source_weights"].values()),
        "source_weight_max": max(weights["source_weights"].values()),
        "row_weight_mean": weights["row_weight_mean"],
        "row_weight_ess_fraction": weights["row_weight_ess_fraction"],
        "held_outcomes_used": weights["held_outcomes_used"],
        "weighted_prediction_rows": int(len(weighted)),
        "hierarchical_prediction_rows": int(len(hierarchical)),
        "weighted_finite": bool(np.isfinite(weighted).all()),
        "hierarchical_finite": bool(np.isfinite(hierarchical).all()),
        "weighted_pooled_model_count": weighted_fit["pooled_model_count"],
        "weighted_isolated_source_models": weighted_fit["isolated_source_models_fit"],
        "hierarchical_pooled_model_count": hierarchical_fit["pooled_model_count"],
        "hierarchical_isolated_source_models": hierarchical_fit["isolated_source_models_fit"],
        "hierarchical_unseen_deviation": hierarchical_fit["unseen_target_deviation_columns"],
        "candidate_rows": int(len(candidate)),
        "candidate_finite": bool(np.isfinite(candidate).all()),
        "movement_cap": movement["max_absolute_rank_correction"],
        "promotion_passed": promotion["passed"],
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
    }


__all__ = ["run_synthetic"]
