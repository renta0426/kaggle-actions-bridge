"""Strategy-v2 E09a: outcome-independent module/teacher applicability audit.

E09a does not fit a predictor. It establishes whether baseline RNA/repertoire
views can be joined to directly supervised historical endpoints under the
frozen study/participant boundary before any high-dimensional or module model
is attempted. Only aggregate counts and non-identifying schema diagnostics are
exportable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .aliases import canonicalize_flow_population, canonicalize_strain, canonicalize_timepoint
from .contracts import DataContractError
from .strategy_e04_contract import audit_measurements
from .strategy_e04b import apply_material_plural_ontology

EXPERIMENT = "strategy_v2_e09a_module_teacher_applicability_audit"
DIRECT_TASKS = ("Task1.3", "Task2.1", "Task2.2", "Task2.3")
PUBLIC_RNA_FILES: Mapping[str, str] = {
    "2019_UGA": "publicData_rnaseq_2019_UGA.tsv",
    "2020_UGA": "publicData_rnaseq_2020_UGA.tsv",
    "2024_UGA": "publicData_rnaseq_2024_UGA.tsv",
    "SDY224": "publicData_rnaseq_SDY224_Bcells.tsv",
    "SDY2867": "publicData_rnaseq_SDY2867.tsv",
    "SDY2941": "publicData_rnaseq_SDY2941.tsv",
}
CHALLENGE_RAW_RNA = "2025LJI_rnaseq.tsv"
CHALLENGE_CORRECTED_RNA = "2025LJI_rnaseq-BATCH-CORRECTED.tsv"
PUBLIC_SEROLOGY = "publicData_serology_260821.tsv"
PUBLIC_CYTOKINE = "publicData_cytokine.tsv"
PUBLIC_FLOW = "publicData_ex_vivo_flow.tsv"
CHALLENGE_FLOW = "2025LJI_ex_vivo_flow.tsv"
PUBLIC_BCR = "publicData_bulkBCR.tsv"
CHALLENGE_BCR = "2025LJI_bulkBCR.tsv"
MD5_MANIFEST = "md5sum"

MIN_DIRECT_STUDIES = 2
MIN_SUBJECTS_PER_STUDY = 8
MIN_TOTAL_SUBJECTS = 24
MIN_COMMON_GENES = 1000
CHALLENGE_EXPECTED_DONORS = 40
CHUNK_ROWS = 500_000


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _material_category(value: Any) -> str:
    raw = re.sub(r"[^a-z0-9]+", "", _text(value).casefold())
    if not raw:
        return "not_reported"
    if "pbmc" in raw or "peripheralbloodmononuclear" in raw:
        return "PBMC"
    if "wholeblood" in raw:
        return "whole_blood"
    if "bcell" in raw:
        return "B_cell"
    if "blood" in raw:
        return "other_blood"
    return "other_reported"


def _sha_items(values: Sequence[str] | set[str]) -> str:
    normalized = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _require_files(root: Path, names: Sequence[str]) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise DataContractError(f"E09a competition files missing: {missing}")


def _manifest_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise DataContractError("E09a malformed md5 manifest")
        names.add(parts[1].strip().lstrip("*"))
    return names


@dataclass
class RnaPrivateScan:
    file: str
    expected_study: str
    value_column: str
    all_participants: dict[str, set[str]]
    day0_participants: dict[str, set[str]]
    negative_baseline_participants: dict[str, set[str]]
    prevacc_union_participants: dict[str, set[str]]
    day0_genes: dict[str, set[str]]
    day0_materials_by_participant: dict[str, dict[str, set[str]]]
    aggregate: dict[str, Any]


def _scan_rna(path: Path, *, expected_study: str, value_column: str) -> RnaPrivateScan:
    required = [
        "participant_id",
        "timepoint",
        "ensembl_gene_id",
        value_column,
        "material",
        "study_accession",
    ]
    all_participants: dict[str, set[str]] = {}
    day0: dict[str, set[str]] = {}
    negative: dict[str, set[str]] = {}
    union: dict[str, set[str]] = {}
    genes: dict[str, set[str]] = {}
    material_by_pid: dict[str, dict[str, set[str]]] = {}
    timepoint_pid: dict[str, dict[str, set[str]]] = {}
    value_rows = 0
    nonnumeric_rows = 0
    negative_values = 0
    zero_values = 0
    day0_rows = 0
    negative_baseline_rows = 0
    material_rows: dict[str, int] = {}
    studies_seen: set[str] = set()

    try:
        chunks = pd.read_csv(
            path,
            sep="\t",
            usecols=required,
            dtype={"participant_id": "string", "timepoint": "string", "ensembl_gene_id": "string", "material": "string", "study_accession": "string"},
            chunksize=CHUNK_ROWS,
            low_memory=False,
        )
    except ValueError as error:
        raise DataContractError(f"E09a RNA schema mismatch:{path.name}") from error

    for chunk in chunks:
        chunk["participant_id"] = chunk["participant_id"].astype("string").str.strip()
        chunk["study_accession"] = chunk["study_accession"].astype("string").str.strip()
        chunk["_time"] = chunk["timepoint"].map(canonicalize_timepoint)
        chunk["_numeric_time"] = pd.to_numeric(chunk["_time"], errors="coerce")
        chunk["_value"] = pd.to_numeric(chunk[value_column], errors="coerce")
        chunk["_material"] = chunk["material"].map(_material_category)
        studies_seen.update(str(value) for value in chunk["study_accession"].dropna().unique())
        value_rows += int(len(chunk))
        nonnumeric_rows += int(chunk["_value"].isna().sum())
        negative_values += int(chunk["_value"].lt(0).fillna(False).sum())
        zero_values += int(chunk["_value"].eq(0).fillna(False).sum())

        for study, group in chunk.groupby("study_accession", sort=False):
            study = str(study)
            pids = set(group["participant_id"].dropna().astype(str))
            all_participants.setdefault(study, set()).update(pids)
            timepoint_pid.setdefault(study, {})
            for timepoint, tgroup in group.groupby("_time", sort=False):
                timepoint_pid[study].setdefault(str(timepoint), set()).update(
                    tgroup["participant_id"].dropna().astype(str)
                )

            d0 = group["_time"].eq("0")
            neg = group["_numeric_time"].lt(0).fillna(False)
            pre = d0 | neg | group["_time"].eq("Pre-vacc")
            day0_rows += int(d0.sum())
            negative_baseline_rows += int(neg.sum())
            day0.setdefault(study, set()).update(group.loc[d0, "participant_id"].dropna().astype(str))
            negative.setdefault(study, set()).update(group.loc[neg, "participant_id"].dropna().astype(str))
            union.setdefault(study, set()).update(group.loc[pre, "participant_id"].dropna().astype(str))
            genes.setdefault(study, set()).update(group.loc[d0, "ensembl_gene_id"].dropna().astype(str))
            material_by_pid.setdefault(study, {})
            d0frame = group.loc[d0, ["participant_id", "_material"]].dropna(subset=["participant_id"])
            for pid, pgroup in d0frame.groupby("participant_id", sort=False):
                material_by_pid[study].setdefault(str(pid), set()).update(
                    str(value) for value in pgroup["_material"].dropna().unique()
                )
            for category, count in group.loc[d0, "_material"].value_counts().items():
                material_rows[str(category)] = material_rows.get(str(category), 0) + int(count)

    if expected_study != "2025LJI" and studies_seen != {expected_study}:
        raise DataContractError(
            f"E09a RNA study identity mismatch:{path.name}:expected={expected_study}:seen={sorted(studies_seen)}"
        )
    if expected_study == "2025LJI" and studies_seen != {"2025LJI"}:
        raise DataContractError(f"E09a Challenge RNA study identity mismatch:{sorted(studies_seen)}")

    study = expected_study
    d0p = day0.get(study, set())
    negp = negative.get(study, set())
    aggregate = {
        "file": path.name,
        "study": study,
        "value_column": value_column,
        "rows": value_rows,
        "numeric_parse_failures": nonnumeric_rows,
        "negative_values": negative_values,
        "zero_values": zero_values,
        "day0_rows": day0_rows,
        "negative_baseline_rows": negative_baseline_rows,
        "all_participants": len(all_participants.get(study, set())),
        "day0_participants": len(d0p),
        "negative_baseline_participants": len(negp),
        "prevacc_union_participants": len(union.get(study, set())),
        "day0_and_negative_baseline_participants": len(d0p & negp),
        "day0_gene_count": len(genes.get(study, set())),
        "day0_gene_set_sha256": _sha_items(genes.get(study, set())),
        "day0_material_categories": sorted(
            category for category in material_rows if category != "not_reported"
        ),
        "day0_material_row_counts": dict(sorted(material_rows.items())),
        "timepoint_participant_counts": {
            key: len(value) for key, value in sorted(timepoint_pid.get(study, {}).items())
        },
    }
    return RnaPrivateScan(
        file=path.name,
        expected_study=study,
        value_column=value_column,
        all_participants=all_participants,
        day0_participants=day0,
        negative_baseline_participants=negative,
        prevacc_union_participants=union,
        day0_genes=genes,
        day0_materials_by_participant=material_by_pid,
        aggregate=aggregate,
    )


@dataclass
class PanelPresence:
    participant_sets: dict[str, dict[str, set[str]]]
    matched_strains: dict[str, dict[str, dict[str, set[str]]]]


def _serology_presence(
    path: Path,
    *,
    vaccine_strains: Sequence[str],
    challenge_strains: Sequence[str],
) -> PanelPresence:
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=["participant_id", "timepoint", "virus_strain", "study_accession"],
        dtype="string",
        low_memory=False,
    )
    frame["_time"] = frame["timepoint"].map(canonicalize_timepoint)
    frame["_strain"] = frame["virus_strain"].map(canonicalize_strain)
    vaccine = {canonicalize_strain(value) for value in vaccine_strains}
    challenge = {canonicalize_strain(value) for value in challenge_strains}
    specs = {
        "Task2.1": ("28", vaccine),
        "Task2.2": ("28", challenge),
        "Task2.3": ("365", challenge),
    }
    sets: dict[str, dict[str, set[str]]] = {task: {} for task in specs}
    matched: dict[str, dict[str, dict[str, set[str]]]] = {task: {} for task in specs}
    for task, (timepoint, panel) in specs.items():
        subset = frame.loc[frame["_time"].eq(timepoint) & frame["_strain"].isin(panel)]
        for (study, pid), group in subset.groupby(["study_accession", "participant_id"], sort=False):
            study = str(study); pid = str(pid)
            sets[task].setdefault(study, set()).add(pid)
            matched[task].setdefault(study, {})[pid] = set(group["_strain"].dropna().astype(str))
    return PanelPresence(sets, matched)


def _presence_table(path: Path, *, timepoints: Sequence[str], extra_column: str | None = None) -> dict[str, dict[str, set[str]]]:
    usecols = ["participant_id", "timepoint", "study_accession"] + ([extra_column] if extra_column else [])
    frame = pd.read_csv(path, sep="\t", usecols=usecols, dtype="string", low_memory=False)
    frame["_time"] = frame["timepoint"].map(canonicalize_timepoint)
    out: dict[str, dict[str, set[str]]] = {str(point): {} for point in timepoints}
    for point in timepoints:
        subset = frame.loc[frame["_time"].eq(str(point))]
        for study, group in subset.groupby("study_accession", sort=False):
            out[str(point)].setdefault(str(study), set()).update(group["participant_id"].dropna().astype(str))
    return out


def _bcr_presence(path: Path) -> dict[str, set[str]]:
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=["participant_id", "timepoint", "study_accession"],
        dtype="string",
        low_memory=False,
    )
    frame["_time"] = frame["timepoint"].map(canonicalize_timepoint)
    out: dict[str, set[str]] = {}
    subset = frame.loc[frame["_time"].eq("0")]
    for study, group in subset.groupby("study_accession", sort=False):
        out[str(study)] = set(group["participant_id"].dropna().astype(str))
    return out


def _compatible_day0_subjects(scan: RnaPrivateScan, challenge_categories: set[str]) -> set[str]:
    study = scan.expected_study
    result = set()
    for pid, categories in scan.day0_materials_by_participant.get(study, {}).items():
        if categories & challenge_categories:
            result.add(pid)
    return result


def _teacher_task_summary(
    *,
    scans: Mapping[str, RnaPrivateScan],
    serology: PanelPresence,
    task: str,
    challenge_categories: set[str],
    challenge_genes: set[str],
    requested_panel_size: int,
) -> dict[str, Any]:
    studies = []
    total_compatible_overlap = 0
    ready_studies = 0
    for study, scan in scans.items():
        day0 = scan.day0_participants.get(study, set())
        label = serology.participant_sets[task].get(study, set())
        overlap = day0 & label
        material_compatible = _compatible_day0_subjects(scan, challenge_categories)
        compatible_overlap = overlap & material_compatible
        gene_overlap = scan.day0_genes.get(study, set()) & challenge_genes
        matched = serology.matched_strains[task].get(study, {})
        complete = sum(1 for pid in compatible_overlap if len(matched.get(pid, set())) == requested_panel_size)
        matched_counts = [len(matched.get(pid, set())) for pid in compatible_overlap]
        study_ready = (
            len(compatible_overlap) >= MIN_SUBJECTS_PER_STUDY
            and len(gene_overlap) >= MIN_COMMON_GENES
        )
        ready_studies += int(study_ready)
        total_compatible_overlap += len(compatible_overlap)
        studies.append(
            {
                "study": study,
                "rna_day0_subjects": len(day0),
                "task_panel_proxy_subjects": len(label),
                "paired_subjects": len(overlap),
                "material_compatible_paired_subjects": len(compatible_overlap),
                "complete_requested_panel_subjects": complete,
                "matched_panel_strain_min": min(matched_counts) if matched_counts else 0,
                "matched_panel_strain_max": max(matched_counts) if matched_counts else 0,
                "common_day0_genes_with_challenge": len(gene_overlap),
                "common_gene_set_sha256": _sha_items(gene_overlap),
                "meets_per_study_data_gate": bool(study_ready),
            }
        )
    ready = ready_studies >= MIN_DIRECT_STUDIES and total_compatible_overlap >= MIN_TOTAL_SUBJECTS
    return {
        "task": task,
        "requested_panel_size": requested_panel_size,
        "source_studies": studies,
        "studies_meeting_per_study_gate": ready_studies,
        "total_material_compatible_paired_subjects": total_compatible_overlap,
        "direct_rna_teacher_data_ready": bool(ready),
        "readiness_rule": {
            "minimum_studies": MIN_DIRECT_STUDIES,
            "minimum_subjects_per_study": MIN_SUBJECTS_PER_STUDY,
            "minimum_total_subjects": MIN_TOTAL_SUBJECTS,
            "minimum_common_genes": MIN_COMMON_GENES,
        },
        "official_complete_panel_cv_claimed": False,
    }


def _task13_summary(
    *,
    scans: Mapping[str, RnaPrivateScan],
    strict_keys: set[tuple[str, str]],
    challenge_categories: set[str],
    challenge_genes: set[str],
) -> dict[str, Any]:
    studies = []
    ready_studies = 0
    total = 0
    for study, scan in scans.items():
        strict = {pid for s, pid in strict_keys if s == study}
        day0 = scan.day0_participants.get(study, set())
        overlap = day0 & strict
        compatible = overlap & _compatible_day0_subjects(scan, challenge_categories)
        gene_overlap = scan.day0_genes.get(study, set()) & challenge_genes
        study_ready = len(compatible) >= MIN_SUBJECTS_PER_STUDY and len(gene_overlap) >= MIN_COMMON_GENES
        ready_studies += int(study_ready)
        total += len(compatible)
        studies.append(
            {
                "study": study,
                "rna_day0_subjects": len(day0),
                "strict_task13_teacher_subjects": len(strict),
                "paired_subjects": len(overlap),
                "material_compatible_paired_subjects": len(compatible),
                "common_day0_genes_with_challenge": len(gene_overlap),
                "meets_per_study_data_gate": bool(study_ready),
            }
        )
    ready = ready_studies >= MIN_DIRECT_STUDIES and total >= MIN_TOTAL_SUBJECTS
    return {
        "task": "Task1.3",
        "source_studies": studies,
        "studies_meeting_per_study_gate": ready_studies,
        "total_material_compatible_paired_subjects": total,
        "direct_rna_teacher_data_ready": bool(ready),
        "official_strict_gate_teacher_required": True,
        "proxy_teacher_promoted_to_strict": False,
    }


def _auxiliary_overlap(
    *,
    scans: Mapping[str, RnaPrivateScan],
    cytokine: dict[str, dict[str, set[str]]],
    flow: dict[str, dict[str, set[str]]],
    serology: PanelPresence,
) -> list[dict[str, Any]]:
    rows = []
    for study, scan in scans.items():
        base = scan.day0_participants.get(study, set())
        rows.append(
            {
                "study": study,
                "rna_day0_subjects": len(base),
                "paired_D1_cytokine_subjects": len(base & cytokine.get("1", {}).get(study, set())),
                "paired_D1_flow_subjects": len(base & flow.get("1", {}).get(study, set())),
                "paired_D7_flow_subjects": len(base & flow.get("7", {}).get(study, set())),
                "paired_D28_Task2.1_proxy_subjects": len(base & serology.participant_sets["Task2.1"].get(study, set())),
                "paired_D28_Task2.2_proxy_subjects": len(base & serology.participant_sets["Task2.2"].get(study, set())),
                "paired_D365_Task2.3_proxy_subjects": len(base & serology.participant_sets["Task2.3"].get(study, set())),
            }
        )
    return rows


def _repertoire_summary(
    *,
    public_bcr: Mapping[str, set[str]],
    challenge_bcr: Mapping[str, set[str]],
    serology: PanelPresence,
) -> dict[str, Any]:
    studies = []
    for study, base in sorted(public_bcr.items()):
        studies.append(
            {
                "study": study,
                "baseline_bulk_bcr_subjects": len(base),
                "paired_Task2.1_proxy_subjects": len(base & serology.participant_sets["Task2.1"].get(study, set())),
                "paired_Task2.2_proxy_subjects": len(base & serology.participant_sets["Task2.2"].get(study, set())),
                "paired_Task2.3_proxy_subjects": len(base & serology.participant_sets["Task2.3"].get(study, set())),
            }
        )
    robust = sum(
        1
        for row in studies
        if max(row["paired_Task2.1_proxy_subjects"], row["paired_Task2.2_proxy_subjects"], row["paired_Task2.3_proxy_subjects"])
        >= MIN_SUBJECTS_PER_STUDY
    )
    return {
        "public_bulk_bcr_studies": studies,
        "challenge_bulk_bcr_subjects": len(challenge_bcr.get("2025LJI", set())),
        "studies_with_at_least_8_paired_hai_proxy_subjects": robust,
        "cross_study_direct_repertoire_teacher_ready": bool(robust >= MIN_DIRECT_STUDIES),
        "single_cell_vdj_external_teacher_available": False,
        "generic_clonality_promoted_to_antigen_specific_teacher": False,
    }


def summarize_e09a(
    *,
    scans: Mapping[str, RnaPrivateScan],
    challenge_raw: RnaPrivateScan,
    challenge_corrected: RnaPrivateScan,
    serology: PanelPresence,
    strict_task13_keys: set[tuple[str, str]],
    cytokine_presence: dict[str, dict[str, set[str]]],
    flow_presence: dict[str, dict[str, set[str]]],
    public_bcr: Mapping[str, set[str]],
    challenge_bcr: Mapping[str, set[str]],
    vaccine_strains: Sequence[str],
    challenge_strains: Sequence[str],
    manifest_entries_present: bool,
) -> dict[str, Any]:
    challenge_study = "2025LJI"
    challenge_day0 = challenge_raw.day0_participants.get(challenge_study, set())
    challenge_negative = challenge_raw.negative_baseline_participants.get(challenge_study, set())
    challenge_genes = challenge_raw.day0_genes.get(challenge_study, set())
    challenge_material_by_pid = challenge_raw.day0_materials_by_participant.get(challenge_study, {})
    challenge_categories = set().union(*challenge_material_by_pid.values()) if challenge_material_by_pid else set()
    challenge_categories.discard("not_reported")
    corrected_day0 = challenge_corrected.day0_participants.get(challenge_study, set())

    task13 = _task13_summary(
        scans=scans,
        strict_keys=strict_task13_keys,
        challenge_categories=challenge_categories,
        challenge_genes=challenge_genes,
    )
    task21 = _teacher_task_summary(
        scans=scans,
        serology=serology,
        task="Task2.1",
        challenge_categories=challenge_categories,
        challenge_genes=challenge_genes,
        requested_panel_size=len({canonicalize_strain(x) for x in vaccine_strains}),
    )
    task22 = _teacher_task_summary(
        scans=scans,
        serology=serology,
        task="Task2.2",
        challenge_categories=challenge_categories,
        challenge_genes=challenge_genes,
        requested_panel_size=len({canonicalize_strain(x) for x in challenge_strains}),
    )
    task23 = _teacher_task_summary(
        scans=scans,
        serology=serology,
        task="Task2.3",
        challenge_categories=challenge_categories,
        challenge_genes=challenge_genes,
        requested_panel_size=len({canonicalize_strain(x) for x in challenge_strains}),
    )
    tasks = {item["task"]: item for item in (task13, task21, task22, task23)}
    hai_ready = any(tasks[task]["direct_rna_teacher_data_ready"] for task in ("Task2.1", "Task2.2", "Task2.3"))
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "audit_type": "outcome_independent_presence_schema_and_measurement_compatibility",
        "outcome_values_accessed": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_attempted": False,
        "manifest_entries_present_for_audited_files": bool(manifest_entries_present),
        "challenge_rna": {
            "raw": challenge_raw.aggregate,
            "batch_corrected": challenge_corrected.aggregate,
            "expected_donors": CHALLENGE_EXPECTED_DONORS,
            "raw_day0_donors": len(challenge_day0),
            "raw_negative_baseline_donors": len(challenge_negative),
            "raw_day0_or_negative_baseline_donors": len(challenge_day0 | challenge_negative),
            "raw_day0_and_negative_baseline_donors": len(challenge_day0 & challenge_negative),
            "raw_day0_missing_donors_require_frozen_base_fallback": CHALLENGE_EXPECTED_DONORS - len(challenge_day0),
            "batch_corrected_day0_donors": len(corrected_day0),
            "batch_corrected_has_negative_values": bool(challenge_corrected.aggregate["negative_values"] > 0),
            "public_batch_corrected_counterpart_available": False,
            "batch_corrected_form_admissible_for_cross_study_module_training": False,
            "challenge_day0_material_categories": sorted(challenge_categories),
        },
        "public_rna": {study: scan.aggregate for study, scan in scans.items()},
        "task_readiness": tasks,
        "auxiliary_teacher_overlap": _auxiliary_overlap(
            scans=scans,
            cytokine=cytokine_presence,
            flow=flow_presence,
            serology=serology,
        ),
        "repertoire_applicability": _repertoire_summary(
            public_bcr=public_bcr,
            challenge_bcr=challenge_bcr,
            serology=serology,
        ),
        "decision": {
            "e09b_hai_module_data_gate_ready": bool(hai_ready),
            "task13_rna_module_data_gate_ready": bool(task13["direct_rna_teacher_data_ready"]),
            "module_gene_sets_frozen": False,
            "model_fitting_authorized_by_e09a": False,
            "next_if_hai_ready": "freeze_small_external_gene_modules_and_symbol_to_ensembl_mapping_before_E09b",
            "next_if_hai_not_ready": "defer_RNA_direct_teacher_and_advance_to_E10_domain_correction",
            "batch_corrected_challenge_only_form_selected": False,
            "missing_view_policy": "fallback_to_frozen_base",
        },
        "interpretation_limits": [
            "teacher_presence_is_not_teacher_predictive_value",
            "historical_HAI_panels_are_incomplete_proxies_not_official_2025_complete_panel_CV",
            "material_category_matching_does_not_prove_processing_equivalence",
            "common_ensembl_ids_do_not_define_a_biological_module",
            "E09a_does_not_fit_or_rank_any_predictor",
        ],
    }


def run_e09a_from_paths(
    data_dir: str | Path,
    *,
    vaccine_strains: Sequence[str],
    challenge_strains: Sequence[str],
) -> dict[str, Any]:
    root = Path(data_dir).expanduser().resolve()
    names = [
        *PUBLIC_RNA_FILES.values(),
        CHALLENGE_RAW_RNA,
        CHALLENGE_CORRECTED_RNA,
        PUBLIC_SEROLOGY,
        PUBLIC_CYTOKINE,
        PUBLIC_FLOW,
        CHALLENGE_FLOW,
        PUBLIC_BCR,
        CHALLENGE_BCR,
        MD5_MANIFEST,
    ]
    _require_files(root, names)
    manifest = _manifest_names(root / MD5_MANIFEST)
    manifest_entries_present = all(name in manifest for name in names if name != MD5_MANIFEST)

    scans = {
        study: _scan_rna(root / filename, expected_study=study, value_column="tpm")
        for study, filename in PUBLIC_RNA_FILES.items()
    }
    challenge_raw = _scan_rna(
        root / CHALLENGE_RAW_RNA,
        expected_study="2025LJI",
        value_column="tpm",
    )
    challenge_corrected = _scan_rna(
        root / CHALLENGE_CORRECTED_RNA,
        expected_study="2025LJI",
        value_column="batch_corrected_expression",
    )
    serology = _serology_presence(
        root / PUBLIC_SEROLOGY,
        vaccine_strains=vaccine_strains,
        challenge_strains=challenge_strains,
    )
    cytokine_presence = _presence_table(root / PUBLIC_CYTOKINE, timepoints=("1",))
    flow_presence = _presence_table(root / PUBLIC_FLOW, timepoints=("1", "7"))
    public_bcr = _bcr_presence(root / PUBLIC_BCR)
    challenge_bcr = _bcr_presence(root / CHALLENGE_BCR)

    public_flow = pd.read_csv(root / PUBLIC_FLOW, sep="\t", low_memory=False)
    challenge_flow = pd.read_csv(root / CHALLENGE_FLOW, sep="\t", low_memory=False)
    transformed, ontology = apply_material_plural_ontology(
        {"public_flow": public_flow, "challenge_flow": challenge_flow}
    )
    measurement = audit_measurements(transformed["public_flow"], transformed["challenge_flow"])
    eligible = measurement.eligibility
    strict_mask = (
        eligible["study"].eq("2024UGA")
        & eligible["within_gate_compatible"].fillna(False).astype(bool)
        & eligible["challenge_gate_compatible"].fillna(False).astype(bool)
    )
    strict_keys = set(
        zip(
            eligible.loc[strict_mask, "study"].astype(str),
            eligible.loc[strict_mask, "participant_id"].astype(str),
        )
    )
    result = summarize_e09a(
        scans=scans,
        challenge_raw=challenge_raw,
        challenge_corrected=challenge_corrected,
        serology=serology,
        strict_task13_keys=strict_keys,
        cytokine_presence=cytokine_presence,
        flow_presence=flow_presence,
        public_bcr=public_bcr,
        challenge_bcr=challenge_bcr,
        vaccine_strains=vaccine_strains,
        challenge_strains=challenge_strains,
        manifest_entries_present=manifest_entries_present,
    )
    result["task13_measurement_bridge"] = {
        "ontology_version": ontology["ontology_version"],
        "material_alias_changed_rows": int(ontology["changed_rows"]),
        "strict_2024UGA_teacher_subjects": len(strict_keys),
        "strict_gate_reconstructed": False,
    }
    return result


__all__ = [
    "EXPERIMENT",
    "PUBLIC_RNA_FILES",
    "MIN_DIRECT_STUDIES",
    "MIN_SUBJECTS_PER_STUDY",
    "MIN_TOTAL_SUBJECTS",
    "MIN_COMMON_GENES",
    "RnaPrivateScan",
    "PanelPresence",
    "summarize_e09a",
    "run_e09a_from_paths",
]
