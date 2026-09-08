"""Competition-Data-free fixture for E04 native and frozen-runtime tests only."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pandas as pd

STRICT_GATE = "CD19+ CD20lo CD27hi CD38hi CD56- CD71hi"
PROXY_GATE = "hCD3E-, hCD14-, hCD19+, hIGHD-, hMS4A1-, hCD27+, hCD38++"


def make_e04_tables() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(1404)
    people, investigations, public, challenge = [], [], [], []
    for study, n in (("2024 UGA", 15), ("SDY272", 18), ("2025LJI", 12)):
        arm = "SYN_ARM_" + study.replace(" ", "_")
        investigations.append({"study_accession": study, "arm_id": arm, "vaccine_season": "2024-25", "vaccine": "synthetic", "year": 2024})
        for i in range(n):
            pid = f"SYN_ONLY_{study.replace(' ', '_')}_{i:03d}"
            subject = "SUBJECT_" + pid
            people.append({"participant_id": pid, "subject": subject, "study_accession": study, "arm_id": arm, "age": 25+i, "biological_sex": "Female" if i % 2 else "Male", "race": "Unknown"})
            a, memory, naive = rng.uniform(0.01, 0.8, 3)
            response = 0.1 + 0.65*a + 0.25*memory + 0.06*rng.normal()
            populations = [("B_cells" if study == "SDY272" else "Antibody-secreting_cells_(ASC)", PROXY_GATE if study == "SDY272" else STRICT_GATE, a, max(response, 0.001)), ("Naive_B_cells", "CD19+ CD27-", naive, naive*1.1), ("Memory_B_cells", "CD19+ CD27+", memory, memory*1.1)]
            for name, gate, baseline, post in populations:
                for day in (("-14", "0", "Pre-vacc") if study == "2025LJI" else ("0", "Pre-vacc", "7")):
                    scale = 1000 if study == "SDY272" and name == "B_cells" else 1
                    row = {"participant_id": pid, "subject": subject, "study_accession": study, "timepoint": day, "name": name, "population_definition": gate, "parent_population": "PBMC", "unit": "cells" if scale == 1000 else "percentage", "material": "PBMC", "comments": "", "value": scale*(post if day == "7" else baseline)}
                    (challenge if study == "2025LJI" else public).append(row)
    return {"participants": pd.DataFrame(people), "investigations": pd.DataFrame(investigations), "public_flow": pd.DataFrame(public), "challenge_flow": pd.DataFrame(challenge)}


def run_synthetic(config):
    from .strategy_e04 import run_e04
    tables = make_e04_tables()
    result = run_e04(config, SimpleNamespace(tables=tables), expected_counts=None)
    return result, tables
