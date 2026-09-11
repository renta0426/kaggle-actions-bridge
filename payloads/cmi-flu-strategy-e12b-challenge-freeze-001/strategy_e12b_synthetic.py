"""Competition-data-free regression fixtures for Strategy-v2 E12b."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import PARTICIPANT_ID_COLUMN, TASK_COLUMNS
from .strategy_e12b import summarize_candidate_frames


def synthetic_e12b_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = [f"SYN_{i:03d}" for i in range(40)]
    sample = pd.DataFrame({PARTICIPANT_ID_COLUMN: ids})
    for task in TASK_COLUMNS:
        sample[task] = -99.0

    base = pd.DataFrame({PARTICIPANT_ID_COLUMN: ids})
    for offset, task in enumerate(TASK_COLUMNS):
        base[task] = np.linspace(0.1 + offset, 4.0 + offset, 40)

    task12 = base.copy(deep=True)
    task12["Task1.2"] = np.linspace(10.0, 1.0, 40) + np.arange(40) * 0.001

    final = task12.copy(deep=True)
    strict = np.arange(40, dtype=float)
    strict[-4:] = strict[:4]  # 36 unique values, matching the frozen real-data contract.
    final["Task1.3"] = strict

    return sample, base, task12, final


def run_synthetic_e12b() -> dict:
    sample, base, task12, final = synthetic_e12b_frames()
    result = summarize_candidate_frames(sample, base, task12, final)
    return {
        "changed_from_b21": result["structural_controls"][
            "final_changed_tasks_vs_regenerated_b21"
        ],
        "changed_from_task12_only": result["structural_controls"][
            "final_changed_tasks_vs_regenerated_task12_only"
        ],
        "portfolio_task_count": result["portfolio_task_count"],
        "challenge_rows": result["challenge_rows"],
        "task13_unique": result["task_prediction_summaries"]["Task1.3"][
            "prediction_unique"
        ],
        "semantic_hash": result["fingerprint_contract"][
            "semantic_submission_sha256"
        ],
        "canonical_csv_hash": result["fingerprint_contract"][
            "canonical_csv_sha256"
        ],
        "next_step": result["next_step"],
        "row_level_candidate_persisted": result["row_level_candidate_persisted"],
        "competition_submission_attempted": result[
            "competition_submission_attempted"
        ],
    }
