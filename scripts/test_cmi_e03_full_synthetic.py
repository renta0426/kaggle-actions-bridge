#!/usr/bin/env python3
"""Exercise the generated E03 production path on a Challenge-shaped synthetic snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd

PUBLIC_STUDIES = (
    ("SDY180", 34),
    ("SDY515", 16),
    ("SDY519", 17),
    ("SDY56", 60),
)
STRAINS = (
    "H1N1 A/Victoria/4897/2022",
    "H3N2 A/District Of Columbia/27/2023_MDCK",
    "Vic B/Austria/1359417/2021",
    "H1N1 A/California/7/2009",
    "H3N2 A/Massachusetts/18/2022_MDCK",
    "H3N2 A/Tasmania/503/2020",
    "H3N2 A/Washington/2/2019",
    "H3N2 A/Pennsylvania/525/2025",
    "H1N1 A/Brisbane/2/2018",
    "H3N2 A/Victoria/2570/2019",
    "H1N1 A/Michigan/45/2015",
    "Vic B/Brisbane/60/2008",
)
VACCINE_STRAINS = STRAINS[:3]
TASK_COLUMNS = (
    "Task1.1", "Task1.2", "Task1.3", "Task1.4",
    "Task2.1", "Task2.2", "Task2.3",
)
EXPECTED_OUTPUTS = {"bridge-result.json", "metrics.json", "summary.md"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_tsv(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def participant(pid: str, subject: str, study: str, arm: str, index: int) -> dict:
    return {
        "participant_id": pid,
        "subject": subject,
        "study_accession": study,
        "arm_id": arm,
        "age": 18 + index % 55,
        "biological_sex": "female" if index % 2 else "male",
        "race": "Asian" if index % 4 == 0 else "White",
        "geolocation": "US: California" if study == "2025LJI" else "US: Georgia",
        "source": "inhouse" if study == "2025LJI" else "public",
    }


def build_input(root: Path) -> None:
    root.mkdir(parents=True)
    public: list[dict] = []
    global_index = 0
    for study_index, (study, count) in enumerate(PUBLIC_STUDIES):
        for local_index in range(count):
            public.append(
                participant(
                    f"{study}_P{local_index:03d}",
                    f"{study}_S{local_index:03d}",
                    study,
                    f"ARM_{study}",
                    global_index,
                )
            )
            global_index += 1
    challenge = [
        participant(f"T{i:02d}", f"TS{i:02d}", "2025LJI", "ARMLJI2025", i)
        for i in range(40)
    ]
    write_tsv([*public, *challenge], root / "participants.tsv")

    investigations = []
    for idx, (study, _) in enumerate(PUBLIC_STUDIES):
        year = 2019 + idx
        investigations.append(
            {
                "study_accession": study,
                "arm_id": f"ARM_{study}",
                "vaccine_season": f"{year}-{str(year + 1)[-2:]}",
                "vaccine": "IIV",
                "year": year,
            }
        )
    investigations.append(
        {
            "study_accession": "2025LJI",
            "arm_id": "ARMLJI2025",
            "vaccine_season": "2025-26",
            "vaccine": "IIV",
            "year": 2025,
        }
    )
    write_tsv(investigations, root / "investigations_260821.tsv")

    public_cytokine: list[dict] = []
    public_flow: list[dict] = []
    public_serology: list[dict] = []
    offset = 0
    for study_index, (study, count) in enumerate(PUBLIC_STUDIES):
        people = public[offset:offset + count]
        offset += count
        for local_index, person in enumerate(people):
            base = 1.0 + 0.08 * local_index + 0.17 * study_index
            ids = {key: person[key] for key in ("participant_id", "subject", "study_accession")}
            for analyte, analyte_offset in (("IP10", 0.0), ("IL8", 1.0), ("TNFa", 0.5)):
                pre = base + analyte_offset
                response = 1.04 + 0.012 * (local_index % 11) + 0.018 * study_index
                public_cytokine.extend(
                    [
                        {
                            **ids,
                            "timepoint": "Pre-vacc",
                            "analyte": analyte,
                            "value": pre,
                            "unit": "pg/ml",
                            "measurementTechnique": "Luminex",
                            "material": "plasma",
                        },
                        {
                            **ids,
                            "timepoint": 1,
                            "analyte": analyte,
                            "value": pre * response,
                            "unit": "pg/ml",
                            "measurementTechnique": "Luminex",
                            "material": "plasma",
                        },
                    ]
                )
            flow_specs = (
                ("Classical_monocytes", "CD45+ CD14+", 1.0, 1),
                ("Antibody-secreting_cells_(ASC)", "CD19+ CD27+ CD38++", 0.08, 7),
                ("B_cells", "CD19+", 0.7, 7),
            )
            for population, gate, multiplier, day in flow_specs:
                pre = (base + 1.0) * multiplier
                public_flow.extend(
                    [
                        {
                            **ids,
                            "timepoint": "Pre-vacc",
                            "name": population,
                            "value": pre,
                            "unit": "percentage",
                            "material": "PBMC",
                            "parent_population": "PBMC",
                            "population_definition": gate,
                            "comments": "",
                        },
                        {
                            **ids,
                            "timepoint": day,
                            "name": population,
                            "value": pre * (1.5 + 0.02 * (local_index % 13) + 0.03 * study_index),
                            "unit": "percentage",
                            "material": "PBMC",
                            "parent_population": "PBMC",
                            "population_definition": gate,
                            "comments": "",
                        },
                    ]
                )
            for strain_index, strain in enumerate(STRAINS):
                pre = 5.0 * 2 ** ((local_index + strain_index + study_index) % 5)
                for day, value in (
                    ("Pre-vacc", pre),
                    (28, pre * (1.25 + 0.01 * (local_index % 9) + 0.02 * (strain_index % 3))),
                    (365, pre * (0.72 + 0.01 * (local_index % 7) + 0.015 * (strain_index % 4))),
                ):
                    public_serology.append(
                        {
                            **ids,
                            "timepoint": day,
                            "virus_strain": strain,
                            "assay": "HAI",
                            "value": value,
                            "virus_in_vaccine": int(strain in VACCINE_STRAINS),
                        }
                    )
    write_tsv(public_cytokine, root / "publicData_cytokine.tsv")
    write_tsv(public_flow, root / "publicData_ex_vivo_flow.tsv")
    write_tsv(public_serology, root / "publicData_serology_260821.tsv")

    challenge_cytokine: list[dict] = []
    challenge_flow: list[dict] = []
    challenge_aim: list[dict] = []
    challenge_serology: list[dict] = []
    for index, person in enumerate(challenge):
        base = 1.4 + 0.11 * index
        ids = {key: person[key] for key in ("participant_id", "subject", "study_accession")}
        for analyte, analyte_offset in (("IP10", 0.0), ("IL8", 1.0), ("TNFa", 0.5)):
            challenge_cytokine.append(
                {
                    **ids,
                    "timepoint": "Pre-vacc",
                    "analyte": analyte,
                    "value": base + analyte_offset,
                    "unit": "pg/ml",
                    "measurementTechnique": "Legendplex",
                    "material": "plasma",
                }
            )
        for population, gate, multiplier in (
            ("Classical_monocytes", "CD45+ CD14+", 1.0),
            ("Antibody-secreting_cells_(ASC)", "CD19+ CD27+ CD38++", 0.08),
            ("B_cells", "CD19+", 0.7),
        ):
            challenge_flow.append(
                {
                    **ids,
                    "timepoint": "Pre-vacc",
                    "name": population,
                    "value": (base + 1.0) * multiplier,
                    "unit": "percentage",
                    "material": "PBMC",
                    "parent_population": "PBMC",
                    "population_definition": gate,
                    "comments": "",
                }
            )
        challenge_aim.append(
            {
                **ids,
                "timepoint": "Pre-vacc",
                "stimulation": "Conserved",
                "name": "CD4 AIM+",
                "value": 0.01 + index * 0.002,
                "unit": "percentage",
                "parent_population": "CD4",
                "population_definition": "synthetic",
                "material": "PBMC",
                "comments": "",
            }
        )
        for strain_index, strain in enumerate(STRAINS):
            challenge_serology.append(
                {
                    **ids,
                    "timepoint": "Pre-vacc",
                    "virus_strain": strain,
                    "assay": "HAI",
                    "value": 5.0 * 2 ** ((index + strain_index) % 5),
                    "virus_in_vaccine": int(strain in VACCINE_STRAINS),
                }
            )
    write_tsv(challenge_cytokine, root / "2025LJI_cytokine.tsv")
    write_tsv(challenge_flow, root / "2025LJI_ex_vivo_flow.tsv")
    write_tsv(challenge_aim, root / "2025LJI_aim.tsv")
    write_tsv(challenge_serology, root / "2025LJI_serology.tsv")

    sample = pd.DataFrame({"participant_id": [person["participant_id"] for person in challenge]})
    for task in TASK_COLUMNS:
        sample[task] = -99.0
    sample.to_csv(root / "sample_submission_part1.csv", index=False)
    (root / "md5sum").write_text("", encoding="utf-8")


def run_checked(command: list[str], label: str) -> None:
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0:
        print(
            f"{label}_FAIL rc={completed.returncode} "
            f"stdout_bytes={len(completed.stdout)} stdout_sha256={sha256_bytes(completed.stdout)} "
            f"stderr_bytes={len(completed.stderr)} stderr_sha256={sha256_bytes(completed.stderr)}"
        )
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--sanitizer", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.expanduser().resolve()
    sanitizer = args.sanitizer.expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="cmi-e03-full-synthetic-") as temporary:
        root = Path(temporary)
        input_dir = root / "competition"
        output_dir = root / "output"
        build_input(input_dir)
        run_checked(
            [sys.executable, str(runtime), "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
            "E03_FULL_SYNTHETIC_RUNTIME",
        )
        if {path.name for path in output_dir.iterdir()} != EXPECTED_OUTPUTS:
            raise SystemExit("E03 full synthetic output allowlist mismatch")
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        if metrics.get("experiment") != "strategy_v2_e03_task11_optional_view_fusion":
            raise SystemExit("E03 full synthetic experiment identity mismatch")
        if len(metrics.get("folds") or []) != 4:
            raise SystemExit("E03 full synthetic did not execute four held-study folds")
        if int((metrics.get("challenge") or {}).get("rows", -1)) != 40:
            raise SystemExit("E03 full synthetic Challenge row contract mismatch")
        run_checked(
            [sys.executable, str(sanitizer), "--input-dir", str(output_dir)],
            "E03_FULL_SYNTHETIC_SANITIZER",
        )
    print("CMI_FLU_E03_FULL_SYNTHETIC PASS production_execute=true folds=4 challenge_rows=40 outputs=3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
