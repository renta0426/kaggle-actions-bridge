"""Competition-Data-free fixture for E06c D28 calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import HAIModelDataset
from .models import ModelSpec


def make_e06c_dataset() -> HAIModelDataset:
    rows = []
    strains = ("H1", "H3", "B")
    for study_index, study in enumerate(("A", "B", "C", "D")):
        for donor_index in range(6):
            participant = f"SYN_E06C_{study}_{donor_index}"
            subject = f"SYN_E06C_SUBJECT_{study}_{donor_index}"
            age = 25.0 + 2.0 * donor_index + study_index
            for strain_index, strain in enumerate(strains):
                pre = 24.0 + 4.0 * donor_index + 3.0 * strain_index + study_index
                fold = 1.4 + 0.07 * donor_index + 0.025 * strain_index + 0.01 * study_index
                post = pre * fold
                rows.append(
                    {
                        "participant_id": participant,
                        "subject_group": subject,
                        "study_group": study,
                        "virus_strain": strain,
                        "log2_pre_hai": np.log2(pre),
                        "age": age,
                        "post_hai": post,
                        "target_log2_fold": np.log2(fold),
                    }
                )
    train = pd.DataFrame(rows)

    challenge_rows = []
    for donor_index in range(5):
        for strain_index, strain in enumerate(strains):
            pre = 26.0 + 3.0 * donor_index + 2.0 * strain_index
            challenge_rows.append(
                {
                    "participant_id": f"SYN_E06C_CHALLENGE_{donor_index}",
                    "subject_group": f"SYN_E06C_CHALLENGE_SUBJECT_{donor_index}",
                    "study_group": "CHALLENGE",
                    "virus_strain": strain,
                    "log2_pre_hai": np.log2(pre),
                    "age": 31.0 + donor_index,
                }
            )
    return HAIModelDataset(
        day=28,
        train=train,
        challenge=pd.DataFrame(challenge_rows),
        target_representation="residual",
        metadata={"challenge_panel_strains": list(strains)},
    )


def ridge_spec() -> ModelSpec:
    return ModelSpec(
        name="ridge_exact_a100",
        family="ridge",
        params={"alpha": 100.0},
        target_transform="identity",
    )


def run_synthetic() -> dict:
    from .strategy_e06c import run_e06c_on_datasets

    base = make_e06c_dataset()
    return dict(
        run_e06c_on_datasets(
            b21_dataset=base,
            target_domain_dataset=base,
            spec=ridge_spec(),
            task_panels={"Task2.1": ("H1", "H3"), "Task2.2": ("H1", "H3", "B")},
            expected_donors=5,
        )
    )
