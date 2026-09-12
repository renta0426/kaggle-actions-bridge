"""Synthetic-only fixtures shared by science tests and exact bridge runtime CI.

Every identifier/value is invented here. This module is CI-only and must never
be a substitute for Competition input or for the immutable E12c-v2 CSV.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

TASKS = ("Task1.1", "Task1.2", "Task1.3", "Task1.4", "Task2.1", "Task2.2", "Task2.3")
STRAINS = ("H1N1 A/Victoria/4897/2022", "Vic B/Austria/1359417/2021", "H3N2 A/Tasmania/503/2020")
RNA = {
    "2019_UGA": "publicData_rnaseq_2019_UGA.tsv",
    "2020_UGA": "publicData_rnaseq_2020_UGA.tsv",
    "2024_UGA": "publicData_rnaseq_2024_UGA.tsv",
    "SDY224": "publicData_rnaseq_SDY224_Bcells.tsv",
    "SDY2867": "publicData_rnaseq_SDY2867.tsv",
    "SDY2941": "publicData_rnaseq_SDY2941.tsv",
}
ASC = "Antibody-secreting_cells_(ASC)"


def make_workspace(root: Path) -> tuple[Path, Path]:
    """Build a complete small input, including empty strict Task1.2 and RNA."""
    data = root / "competition"
    source = root / "source-b"
    data.mkdir(parents=True, exist_ok=False)
    source.mkdir(parents=True, exist_ok=False)
    sizes = {"SDY180": 8, "SDY515": 8, "SDY519": 8, "SDY56": 40,
             "2024UGA": 8, "SDY296": 8, "SDY301": 8, "SDY416": 1, "2025LJI": 40}
    people = []
    for study, count in sizes.items():
        for i in range(count):
            pid = f"SYN_{study}_{i:02d}"
            subject = "SYN_SHARED_SUBJECT" if i == 0 and study in {"SDY180", "SDY515"} else f"SYN_SUBJECT_{study}_{i:02d}"
            people.append({"participant_id": pid, "subject": subject, "study_accession": study,
                           "arm_id": f"SYN_ARM_{study}", "age": 20 + i % 40,
                           "biological_sex": "female" if i % 2 else "male", "race": "synthetic",
                           "geolocation": "synthetic", "source": "synthetic"})
    tables = {"participants.tsv": people,
              "investigations_260821.tsv": [{"study_accession": s, "arm_id": f"SYN_ARM_{s}",
                                             "vaccine_season": "2025-26", "vaccine": "IIV", "year": 2025}
                                            for s in sizes]}
    for name in ("publicData_cytokine.tsv", "publicData_ex_vivo_flow.tsv", "publicData_serology_260821.tsv",
                 "2025LJI_cytokine.tsv", "2025LJI_ex_vivo_flow.tsv", "2025LJI_serology.tsv", "2025LJI_aim.tsv"):
        tables[name] = []
    for j, person in enumerate(people):
        ids = {k: person[k] for k in ("participant_id", "subject", "study_accession")}
        study = person["study_accession"]
        challenge = study == "2025LJI"
        prefix = "2025LJI" if challenge else "publicData"
        if challenge or study in {"SDY180", "SDY515", "SDY519", "SDY56"}:
            for analyte, offset in (("IP10", 0.0), ("IL8", 1.0)):
                for day in (["Pre-vacc"] if challenge else ["Pre-vacc", "1"]):
                    tables[f"{prefix}_cytokine.tsv"].append({**ids, "timepoint": day, "analyte": analyte,
                        "value": (2 + j / 10 + offset) * (1.3 if day == "1" else 1),
                        "unit": "pg/ml", "measurementTechnique": "synthetic", "material": "plasma"})
        populations = [ASC, "Classical_monocytes"] if challenge else ([ASC] if study == "2024UGA" else (["Classical_monocytes"] if study in {"SDY296", "SDY301", "SDY416"} else []))
        for population in populations:
            points = ["Pre-vacc"] if challenge else (["Pre-vacc", "7"] if population == ASC else ["Pre-vacc", "1"])
            for point in points:
                if population == ASC and point == "7" and int(person["participant_id"][-2:]) >= 6:
                    continue
                incompatible = not challenge and population == "Classical_monocytes"
                tables[f"{prefix}_ex_vivo_flow.tsv"].append({**ids, "timepoint": point, "name": population,
                    "value": 0.1 + j / 1000 + (0.01 if point in {"1", "7"} else 0),
                    "unit": "cells/ul" if incompatible else "percentage",
                    "material": "whole blood" if incompatible else ("PBMCs" if challenge else "PBMC"),
                    "parent_population": "PBMC", "population_definition": "CD14+CD16-" if population != ASC else "CD19+CD20-CD27+CD38++",
                    "comments": ""})
        serology_file = "2025LJI_serology.tsv" if challenge else "publicData_serology_260821.tsv"
        for k, strain in enumerate(STRAINS):
            points = ["Pre-vacc"] if challenge else ["Pre-vacc", "28", "365"]
            for point in points:
                if point == "365" and k == 2:
                    continue
                tables[serology_file].append({**ids, "timepoint": point, "virus_strain": strain,
                    "assay": "HAI", "value": 5.0 * 2 ** ((j + k) % 5) * (1.2 if point == "28" else 1),
                    "virus_in_vaccine": int(k < 2)})
        if challenge:
            for point, offset in (("-14", 0), ("0", 0.01), ("Pre-vacc", 0.007)):
                tables["2025LJI_aim.tsv"].append({**ids, "timepoint": point, "stimulation": "Conserved",
                    "name": "CD4 AIM+", "value": 0.1 + j / 1000 + offset})
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(data / name, sep="\t", index=False)
    for study, name in {**RNA, "2025LJI": "2025LJI_rnaseq.tsv"}.items():
        ids = [p["participant_id"] for p in people if p["study_accession"] == ("2024UGA" if study == "2024_UGA" else study)]
        if not ids:
            ids = [f"SYN_RNA_{study}_{i}" for i in range(2)]
        pd.DataFrame([{"participant_id": pid, "timepoint": "0", "ensembl_gene_id": "SYN_GENE",
                       "tpm": 1.0, "material": "PBMC", "study_accession": study} for pid in ids]).to_csv(data / name, sep="\t", index=False)
    challenge_ids = [p["participant_id"] for p in people if p["study_accession"] == "2025LJI"]
    sample = pd.DataFrame({"participant_id": challenge_ids, **{t: [-99] * 40 for t in TASKS}})
    sample.to_csv(data / "sample_submission_part1.csv", index=False)
    prediction = sample.copy()
    for j, task in enumerate(TASKS):
        prediction[task] = np.arange(40, dtype=float) / 17 + j / 13
    prediction.loc[:2, "Task1.3"] = 0.12345678901234566
    prediction.to_csv(source / "submission.csv", index=False, float_format="%.17g", lineterminator="\n")
    manifest = "".join(f"{hashlib.md5(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(data.iterdir()) if p.is_file())
    (data / "md5sum").write_text(manifest, encoding="utf-8")
    return data, source / "submission.csv"


@contextmanager
def forbid_fit_and_submit():
    """Observe actual Python calls, not just importability or source substrings."""
    previous = sys.getprofile()
    forbidden = {"fit", "fit_transform", "fit_predict", "partial_fit", "competitions_submit",
                 "competition_submit", "kernels_push", "run_e04", "run_e04b"}
    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_name in forbidden:
            raise AssertionError("forbidden model-fit or Kaggle write call in no-fit integration test")
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(previous)


def assert_audit(result: dict) -> None:
    assert set(result["teacher_ledger"]) == set(TASKS)
    assert result["model_fit_count"] == 0
    assert result["competition_submission_attempted"] is False
    assert result["privacy"]["aggregate_only"] is True
    t12 = result["teacher_ledger"]["Task1.2"]
    assert t12["target_bearing_participant_years"] == 17
    assert t12["baseline_join_rows"] == 0
    assert t12["split_support"]["status"] == "data_limited"
    assert result["source_alignment_audit"]["Task1.2_strict_baseline_join"]["target_counts_match_reference"] is False
    assert result["reference_expectations_not_mutated_to_match_runtime"]["Task1.2_target_by_study"] == {"SDY296": 36, "SDY301": 40, "SDY416": 5}
    assert result["source_alignment_audit"]["Task1.3_RNA"]["strict_teacher_with_day0_rna"] == 6
    assert result["source_alignment_audit"]["Task1.3_RNA"]["strict_teacher_with_measurement_compatible_day0_rna"] == 6
    assert result["teacher_ledger"]["Task1.4"]["target_bearing_participant_years"] == 0
    folds = result["split_support"]["Task1.1"]["folds"]
    assert any(f["n_subjects_purged"] > 0 for f in folds)
    json.dumps(result, allow_nan=False)


def assert_diagnostics(source: Path, output: Path, manifest: dict) -> None:
    assert manifest["diagnostic_not_final"] is True
    assert len(manifest["files"]) == 6
    with source.open(newline="") as stream:
        original = list(csv.reader(stream))
    for item in manifest["files"]:
        path = output / item["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        with path.open(newline="") as stream:
            actual = list(csv.reader(stream))
        assert actual[0] == original[0]
        assert len(actual) == len(original) == 41
        index = original[0].index(item["task"])
        for old, new in zip(original[1:], actual[1:], strict=True):
            assert old[0] == new[0]
            assert old[index] == new[index]
            assert all(new[k] == "-99" for k in range(1, 8) if k != index)
