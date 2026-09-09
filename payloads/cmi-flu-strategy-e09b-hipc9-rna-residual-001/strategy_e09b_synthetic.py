"""Competition-data-free synthetic fixture for E09b HIPC9 residual evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy_e09b import GENE_COLUMNS, QUALIFYING_STUDIES, evaluate_task_from_donor_frames


def _make_study(study: str, n: int, offset: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    u = np.linspace(0.02, 0.98, n)
    nuisance = np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
    target = u
    base = u + 0.22 * nuisance
    ids = [f"SYN_E09B_{study}_{offset + i:03d}" for i in range(n)]
    donor = pd.DataFrame(
        {
            "study_group": study,
            "participant_id": ids,
            "target": target,
            "prediction": base,
        }
    )
    rna = pd.DataFrame({"study_group": study, "participant_id": ids})
    for index, column in enumerate(GENE_COLUMNS):
        if index == 0:
            rna[column] = (nuisance + 1.0) / 2.0
        else:
            rna[column] = u
    rna["hipc7_score"] = u
    rna["gene_coverage"] = len(GENE_COLUMNS)
    rna["rna_eligible"] = True
    return donor, rna


def run_synthetic(task: str = "Task2.1") -> dict:
    donors = []
    rnas = []
    for i, study in enumerate(QUALIFYING_STUDIES):
        donor, rna = _make_study(study, 30, i * 100)
        donors.append(donor)
        rnas.append(rna)
    historical = pd.concat(donors, ignore_index=True)
    historical_rna = pd.concat(rnas, ignore_index=True)

    n = 20
    u = np.linspace(0.03, 0.97, n)
    nuisance = np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
    ids = [f"SYN_E09B_CHALLENGE_{i:03d}" for i in range(n)]
    challenge_base = pd.DataFrame({"participant_id": ids, "prediction": u + 0.22 * nuisance})
    challenge_rna = pd.DataFrame({"participant_id": ids, "study_group": "2025LJI"})
    for index, column in enumerate(GENE_COLUMNS):
        challenge_rna[column] = (nuisance + 1.0) / 2.0 if index == 0 else u
    challenge_rna["hipc7_score"] = u
    challenge_rna["gene_coverage"] = len(GENE_COLUMNS)
    challenge_rna["rna_eligible"] = True

    return evaluate_task_from_donor_frames(
        task=task,
        historical_base=historical,
        challenge_base=challenge_base,
        historical_rna=historical_rna,
        challenge_rna=challenge_rna,
    )


__all__ = ["run_synthetic"]
