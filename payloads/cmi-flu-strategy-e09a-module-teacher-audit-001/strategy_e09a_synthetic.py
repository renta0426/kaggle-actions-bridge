"""Competition-data-free synthetic fixture for E09a applicability decisions."""
from __future__ import annotations

from .strategy_e09a import PanelPresence, RnaPrivateScan, summarize_e09a


def _scan(study: str, *, participants: int, genes: int, material: str = "PBMC", negative_values: int = 0) -> RnaPrivateScan:
    pids = {f"SYN_E09A_{study}_{index}" for index in range(participants)}
    gene_set = {f"ENSG_SYN_{index:05d}" for index in range(genes)}
    materials = {pid: {material} for pid in pids}
    aggregate = {
        "file": f"{study}.tsv",
        "study": study,
        "value_column": "tpm",
        "rows": participants * genes,
        "numeric_parse_failures": 0,
        "negative_values": negative_values,
        "zero_values": 0,
        "day0_rows": participants * genes,
        "negative_baseline_rows": max(participants - 1, 0) * genes,
        "all_participants": participants,
        "day0_participants": participants,
        "negative_baseline_participants": max(participants - 1, 0),
        "prevacc_union_participants": participants,
        "day0_and_negative_baseline_participants": max(participants - 1, 0),
        "day0_gene_count": genes,
        "day0_gene_set_sha256": "synthetic",
        "day0_material_categories": [material],
        "day0_material_row_counts": {material: participants * genes},
        "timepoint_participant_counts": {"0": participants},
    }
    return RnaPrivateScan(
        file=f"{study}.tsv",
        expected_study=study,
        value_column="tpm",
        all_participants={study: set(pids)},
        day0_participants={study: set(pids)},
        negative_baseline_participants={study: set(sorted(pids)[:-1])},
        prevacc_union_participants={study: set(pids)},
        day0_genes={study: gene_set},
        day0_materials_by_participant={study: materials},
        aggregate=aggregate,
    )


def run_synthetic() -> dict:
    a = _scan("A", participants=12, genes=1800)
    b = _scan("B", participants=12, genes=1700)
    c = _scan("C", participants=12, genes=1800, material="B_cell")
    scans = {"A": a, "B": b, "C": c}
    challenge = _scan("2025LJI", participants=5, genes=2000)
    corrected = _scan("2025LJI", participants=4, genes=2000, negative_values=25)

    vaccine = ("H1", "H3", "B")
    challenge_panel = tuple(f"V{index}" for index in range(12))
    participant_sets = {task: {} for task in ("Task2.1", "Task2.2", "Task2.3")}
    matched = {task: {} for task in ("Task2.1", "Task2.2", "Task2.3")}
    panels = {"Task2.1": vaccine, "Task2.2": challenge_panel, "Task2.3": challenge_panel}
    for task, panel in panels.items():
        for study, scan in scans.items():
            pids = set(scan.day0_participants[study])
            participant_sets[task][study] = pids
            matched[task][study] = {pid: set(panel) for pid in pids}
    serology = PanelPresence(participant_sets=participant_sets, matched_strains=matched)

    strict_keys = {("A", pid) for pid in a.day0_participants["A"]}
    cytokine = {"1": {"A": set(a.day0_participants["A"])}}
    flow = {"1": {"B": set(b.day0_participants["B"])}, "7": {"A": set(a.day0_participants["A"])}}
    public_bcr = {
        "A": set(a.day0_participants["A"]),
        "B": set(b.day0_participants["B"]),
    }
    challenge_bcr = {"2025LJI": set(challenge.day0_participants["2025LJI"])}

    return dict(
        summarize_e09a(
            scans=scans,
            challenge_raw=challenge,
            challenge_corrected=corrected,
            serology=serology,
            strict_task13_keys=strict_keys,
            cytokine_presence=cytokine,
            flow_presence=flow,
            public_bcr=public_bcr,
            challenge_bcr=challenge_bcr,
            vaccine_strains=vaccine,
            challenge_strains=challenge_panel,
            manifest_entries_present=True,
        )
    )


__all__ = ["run_synthetic"]
