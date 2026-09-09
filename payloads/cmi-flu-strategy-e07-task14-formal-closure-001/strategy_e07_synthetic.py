"""Competition-Data-free fixture for E07 Task1.4 formal closure tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_e07_tables(*, n: int = 8) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(1407)
    aim_rows = []
    participants = []
    hla_rows = []
    vdj_rows = []
    stimulations = ("Conserved", "DMSO", "H1_pool", "H3_pool", "B_pool")
    for i in range(n):
        pid = f"SYN_ONLY_E07_{i:03d}"
        subject = f"SYN_SUBJECT_E07_{i:03d}"
        participants.append({"participant_id": pid, "subject": subject, "study_accession": "2025LJI"})
        conserved0 = 0.02 + 0.005 * i + rng.normal(scale=0.001)
        conserved14 = conserved0 + rng.normal(scale=0.004)
        for stim_i, stimulation in enumerate(stimulations):
            if stimulation == "Conserved":
                day14 = conserved14
                day0 = conserved0
            elif stimulation == "DMSO":
                day14 = 0.006 + 0.0007 * (n - i) + rng.normal(scale=0.0005)
                day0 = 0.005 + 0.0005 * i + rng.normal(scale=0.0005)
            else:
                base = 0.01 + 0.002 * stim_i + 0.001 * i
                day14 = base + rng.normal(scale=0.001)
                day0 = base + rng.normal(scale=0.001)
            prevacc = (day14 + day0) / 2.0
            for timepoint, value in (("-14", day14), ("0", day0), ("Pre-vacc", prevacc)):
                aim_rows.append(
                    {
                        "participant_id": pid,
                        "subject": subject,
                        "study_accession": "2025LJI",
                        "timepoint": timepoint,
                        "stimulation": stimulation,
                        "name": "AIM_positive_T_cells",
                        "value": max(float(value), 0.00001),
                        "unit": "percentage",
                        "parent_population": "CD3_positive",
                        "population_definition": "synthetic_CD4_or_CD8_AIM",
                        "material": "PBMCs",
                        "comments": "",
                    }
                )
        # Generic single-cell VDJ deliberately lacks antigen specificity and
        # cell-subset annotation.  It must not become a teacher merely because
        # clone/chain/depth data exist.
        if i < n - 1:
            for chain in ("TRA", "TRB"):
                for j in range(3):
                    vdj_rows.append(
                        {
                            "participant_id": pid,
                            "timepoint": "0",
                            "chain": chain,
                            "productive": True,
                            "umis": 5 + i + j,
                            "barcode": f"SYN_BARCODE_{i}_{chain}_{j}",
                            "contig_id": f"SYN_CONTIG_{i}_{chain}_{j}",
                            "cdr3": f"SYN_CDR3_{i}_{chain}_{j}",
                        }
                    )
        if i < n // 2:
            hla_rows.append({"subject": subject, "locus_name": "HLA-A", "allele_1": "A*01:01", "allele_2": "A*02:01", "study_accession": "2025LJI"})
    return {
        "challenge_aim": pd.DataFrame(aim_rows),
        "challenge_vdj": pd.DataFrame(vdj_rows),
        "participants": pd.DataFrame(participants),
        "participant_hla": pd.DataFrame(hla_rows),
    }


def run_synthetic() -> tuple[dict, dict[str, pd.DataFrame]]:
    from .strategy_e07 import run_e07

    tables = make_e07_tables()
    result = run_e07(
        challenge_aim=tables["challenge_aim"],
        challenge_vdj=tables["challenge_vdj"],
        participants=tables["participants"],
        participant_hla=tables["participant_hla"],
        expected_real_counts=False,
        epitope_hla_map_available=False,
    )
    return result, tables
