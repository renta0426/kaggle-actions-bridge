"""Strategy-v2 E07: Task1.4 challenge-only zero-shot applicability audit.

Task1.4 has no public AIM outcome labels.  E07 therefore never manufactures a
supervised target from another readout.  It audits the conserved-pool baseline
anchor, an optional negative-control contrast, and whether the delivered TCR/HLA
metadata can honestly instantiate the external antigen-specific T-cell evidence.
Only aggregate dictionaries may leave this module.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .aliases import canonicalize_aim_stimulation, canonicalize_timepoint
from .contracts import DataContractError, require_columns
from .targets import build_task_14_anchor

EXPERIMENT = "strategy_v2_e07_task14_formal_closure"
TASK = "Task1.4"
CONTRACT_VERSION = "e07_task14_zero_shot_audit_v1"
EXPECTED_CHALLENGE_SUBJECTS = 40
EXPECTED_AIM_ROWS = 600
EXPECTED_STIMULATIONS = 5
EXPECTED_BASELINE_TIMEPOINTS = ("-14", "0", "Pre-vacc")
PREVACC_TOLERANCE = 1e-8
MIN_EXTERNAL_TEACHER_COVERAGE = 28

NEGATIVE_STIMULATION_KEYS = {
    "dmso",
    "negative",
    "negativecontrol",
    "neg",
    "negcontrol",
    "unstimulated",
    "unstim",
    "media",
    "medium",
    "mock",
    "vehicle",
}

ANTIGEN_SPECIFICITY_TOKENS = (
    "antigen",
    "epitope",
    "specificity",
    "tetramer",
    "dextramer",
    "multimer",
)
CELL_SUBSET_TOKENS = (
    "celltype",
    "cell_type",
    "phenotype",
    "subset",
    "annotation",
    "cluster",
)


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _text(value).casefold())


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("E07 ranks require a nonempty finite vector")
    if len(values) == 1:
        return np.array([0.5])
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def _correlation(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("E07 paired correlation shape mismatch")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"n": int(len(a)), "status": "nonfinite", "value": None}
    if len(a) < 3:
        return {"n": int(len(a)), "status": "insufficient_n", "value": None}
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return {"n": int(len(a)), "status": "constant", "value": None}
    value = float(spearmanr(a, b).statistic)
    return {"n": int(len(a)), "status": "ok" if np.isfinite(value) else "undefined", "value": value if np.isfinite(value) else None}


def _movement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    if reference.shape != candidate.shape:
        raise ValueError("E07 movement alignment mismatch")
    a, b = _percentile(reference), _percentile(candidate)
    shift = np.abs(a - b)
    return {
        "n": int(len(reference)),
        "rank_spearman": _correlation(reference, candidate),
        "changed_rank_count": int(np.sum(shift > 1e-12)),
        "mean_absolute_percentile_shift": float(shift.mean()),
        "max_absolute_percentile_shift": float(shift.max()),
    }


def _prepare_aim(aim: pd.DataFrame) -> pd.DataFrame:
    required = {"participant_id", "timepoint", "stimulation", "name", "value", "unit", "parent_population", "population_definition", "material"}
    require_columns(aim, sorted(required), table_name="E07 challenge AIM")
    frame = aim.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["_time"] = frame["timepoint"].map(canonicalize_timepoint)
    frame["_stim"] = frame["stimulation"].map(canonicalize_aim_stimulation).map(_text)
    frame["_name"] = frame["name"].map(_text)
    frame["_value"] = pd.to_numeric(frame["value"], errors="coerce")
    if not np.isfinite(frame["_value"].to_numpy(float)).all():
        raise DataContractError("E07 AIM contains nonfinite values")
    metadata = ["unit", "parent_population", "population_definition", "material"]
    frame["_metadata_signature"] = [_sha([_text(row[column]).casefold() for column in metadata]) for _, row in frame.iterrows()]
    return frame


def _collapsed_aim(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, Any]] = []
    conflicts = 0
    for (pid, stim, name, time), group in frame.groupby(["participant_id", "_stim", "_name", "_time"], sort=True, observed=True):
        signatures = set(group["_metadata_signature"])
        if len(signatures) != 1:
            conflicts += 1
            continue
        rows.append({"participant_id": str(pid), "stimulation": str(stim), "name": str(name), "timepoint": str(time), "value": float(group["_value"].mean()), "source_rows": int(len(group))})
    collapsed = pd.DataFrame(rows)
    if collapsed.empty:
        raise DataContractError("E07 AIM collapse produced no valid rows")
    if collapsed.duplicated(["participant_id", "stimulation", "name", "timepoint"]).any():
        raise DataContractError("E07 AIM collapsed key is not unique")
    return collapsed, conflicts


def _participant_stimulus(collapsed: pd.DataFrame, *, timepoint: str) -> pd.DataFrame:
    selected = collapsed.loc[collapsed["timepoint"].eq(timepoint)].copy()
    if selected.empty:
        return pd.DataFrame(columns=["participant_id", "stimulation", "value", "name_count"])
    return selected.groupby(["participant_id", "stimulation"], observed=True).agg(value=("value", "mean"), name_count=("name", "nunique")).reset_index()


def _prevacc_arithmetic_audit(collapsed: pd.DataFrame) -> dict[str, Any]:
    pivot = collapsed.pivot_table(index=["participant_id", "stimulation", "name"], columns="timepoint", values="value", aggfunc="first")
    required = list(EXPECTED_BASELINE_TIMEPOINTS)
    complete = pivot.dropna(subset=required) if set(required).issubset(pivot.columns) else pivot.iloc[:0]
    if complete.empty:
        return {"complete_measure_keys": 0, "within_tolerance": 0, "all_within_tolerance": False, "max_absolute_error": None, "mean_absolute_error": None, "tolerance": PREVACC_TOLERANCE}
    expected = (complete["-14"].to_numpy(float) + complete["0"].to_numpy(float)) / 2.0
    observed = complete["Pre-vacc"].to_numpy(float)
    error = np.abs(observed - expected)
    ok = np.isclose(observed, expected, rtol=0.0, atol=PREVACC_TOLERANCE)
    return {"complete_measure_keys": int(len(complete)), "within_tolerance": int(ok.sum()), "all_within_tolerance": bool(ok.all()), "max_absolute_error": float(error.max()), "mean_absolute_error": float(error.mean()), "tolerance": PREVACC_TOLERANCE}


def _conserved_repeat_audit(collapsed: pd.DataFrame) -> dict[str, Any]:
    by_stim_minus14 = _participant_stimulus(collapsed, timepoint="-14")
    by_stim_day0 = _participant_stimulus(collapsed, timepoint="0")
    a = by_stim_minus14.loc[by_stim_minus14["stimulation"].eq("Conserved"), ["participant_id", "value"]]
    b = by_stim_day0.loc[by_stim_day0["stimulation"].eq("Conserved"), ["participant_id", "value"]]
    pair = a.merge(b, on="participant_id", suffixes=("_minus14", "_day0"), validate="one_to_one")
    if pair.empty:
        return {"paired_subjects": 0, "spearman": {"n": 0, "status": "insufficient_n", "value": None}}
    delta = np.abs(pair["value_minus14"].to_numpy(float) - pair["value_day0"].to_numpy(float))
    return {"paired_subjects": int(len(pair)), "spearman": _correlation(pair["value_minus14"].to_numpy(float), pair["value_day0"].to_numpy(float)), "mean_absolute_change": float(delta.mean()), "median_absolute_change": float(np.median(delta)), "max_absolute_change": float(delta.max())}


def _negative_control(collapsed: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, Any]:
    stimuli = sorted(set(collapsed["stimulation"].astype(str)))
    candidates = [stim for stim in stimuli if _key(stim) in NEGATIVE_STIMULATION_KEYS]
    result: dict[str, Any] = {"candidate_count": len(candidates), "candidate_names": candidates, "unique_negative_control": len(candidates) == 1, "background_contrast_target_semantics_established": False, "sensitivity_available": False, "paired_subjects": 0, "movement_vs_raw_anchor": None}
    if len(candidates) != 1:
        return result
    negative = candidates[0]
    pre = collapsed.loc[collapsed["timepoint"].eq("Pre-vacc")]
    conserved = pre.loc[pre["stimulation"].eq("Conserved"), ["participant_id", "name", "value"]]
    control = pre.loc[pre["stimulation"].eq(negative), ["participant_id", "name", "value"]]
    paired = conserved.merge(control, on=["participant_id", "name"], suffixes=("_conserved", "_negative"), validate="one_to_one")
    if paired.empty:
        return result
    contrast = paired.assign(contrast=paired["value_conserved"] - paired["value_negative"]).groupby("participant_id", observed=True)["contrast"].mean().rename("contrast").reset_index()
    anchor_pair = anchor[["participant_id", "anchor"]].merge(contrast, on="participant_id", validate="one_to_one")
    result.update(paired_subjects=int(len(anchor_pair)), paired_measure_rows=int(len(paired)), sensitivity_available=bool(len(anchor_pair) == len(anchor)), movement_vs_raw_anchor=_movement(anchor_pair["anchor"].to_numpy(float), anchor_pair["contrast"].to_numpy(float)), contrast_min=float(anchor_pair["contrast"].min()), contrast_max=float(anchor_pair["contrast"].max()))
    return result


def _column_semantics(columns: list[str], tokens: tuple[str, ...]) -> list[str]:
    output = []
    for column in columns:
        key = _key(column)
        if any(_key(token) in key for token in tokens):
            output.append(column)
    return sorted(output)


def _vdj_audit(vdj: pd.DataFrame | None, challenge_participants: set[str]) -> dict[str, Any]:
    if vdj is None:
        return {"available": False, "participant_coverage": 0, "antigen_specificity_columns": [], "cell_subset_columns": [], "external_tcr_teacher_applicable": False, "reason": "challenge_vdj_not_available"}
    require_columns(vdj, ["participant_id"], table_name="E07 challenge VDJ")
    frame = vdj.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame = frame.loc[frame["participant_id"].isin(challenge_participants)].copy()
    columns = [str(column) for column in frame.columns]
    antigen = _column_semantics(columns, ANTIGEN_SPECIFICITY_TOKENS)
    subset = _column_semantics(columns, CELL_SUBSET_TOKENS)
    covered = int(frame["participant_id"].nunique())
    chain_counts: dict[str, int] = {}
    chain_participants: dict[str, int] = {}
    if "chain" in frame.columns:
        chain = frame["chain"].fillna("").map(_text)
        chain_counts = {name: int(count) for name, count in chain.value_counts().sort_index().items() if name}
        for name in sorted(set(chain) - {""}):
            chain_participants[name] = int(frame.loc[chain.eq(name), "participant_id"].nunique())
    productive_rows = None
    if "productive" in frame.columns:
        productive = frame["productive"].astype(str).str.casefold().isin({"true", "1", "yes"})
        productive_rows = int(productive.sum())
    umi_summary = None
    if "umis" in frame.columns:
        umis = pd.to_numeric(frame["umis"], errors="coerce").dropna()
        if len(umis):
            umi_summary = {"numeric_rows": int(len(umis)), "median": float(umis.median()), "p25": float(umis.quantile(0.25)), "p75": float(umis.quantile(0.75))}
    beta_present = any(_key(name) in {"trb", "tcrb", "bet"} or "trb" in _key(name) for name in chain_counts)
    applicable = bool(covered >= MIN_EXTERNAL_TEACHER_COVERAGE and antigen and subset and beta_present)
    reason = "applicable" if applicable else "generic_repertoire_lacks_antigen_specific_cd4_memory_semantics"
    return {"available": True, "rows": int(len(frame)), "participant_coverage": covered, "coverage_fraction": float(covered / len(challenge_participants)) if challenge_participants else 0.0, "chain_row_counts": chain_counts, "chain_participant_counts": chain_participants, "productive_rows": productive_rows, "umi_summary": umi_summary, "antigen_specificity_columns": antigen, "cell_subset_columns": subset, "minimum_teacher_coverage": MIN_EXTERNAL_TEACHER_COVERAGE, "tcr_beta_semantics_present": bool(beta_present), "external_tcr_teacher_applicable": applicable, "reason": reason, "generic_clonality_used_as_teacher": False}


def _hla_audit(participants: pd.DataFrame | None, hla: pd.DataFrame | None, challenge_participants: set[str], *, epitope_hla_map_available: bool) -> dict[str, Any]:
    if participants is None or hla is None:
        return {"available": False, "participant_coverage": 0, "epitope_hla_map_available": bool(epitope_hla_map_available), "hla_correction_applicable": False, "reason": "participant_or_hla_metadata_not_available"}
    require_columns(participants, ["participant_id", "subject"], table_name="E07 participants")
    require_columns(hla, ["subject", "locus_name"], table_name="E07 HLA")
    p = participants[["participant_id", "subject"]].copy()
    p["participant_id"] = p["participant_id"].astype(str)
    p = p.loc[p["participant_id"].isin(challenge_participants)]
    h = hla.copy()
    joined = p.merge(h, on="subject", how="inner")
    covered = int(joined["participant_id"].nunique()) if len(joined) else 0
    loci = sorted(_text(value) for value in joined["locus_name"].dropna().unique() if _text(value))
    applicable = bool(covered >= MIN_EXTERNAL_TEACHER_COVERAGE and epitope_hla_map_available)
    reason = "applicable" if applicable else ("conserved_pool_epitope_hla_mapping_not_available" if not epitope_hla_map_available else "challenge_hla_coverage_below_threshold")
    return {"available": True, "participant_coverage": covered, "coverage_fraction": float(covered / len(challenge_participants)) if challenge_participants else 0.0, "locus_count": len(loci), "loci": loci, "minimum_teacher_coverage": MIN_EXTERNAL_TEACHER_COVERAGE, "epitope_hla_map_available": bool(epitope_hla_map_available), "hla_correction_applicable": applicable, "reason": reason}


def run_e07(*, challenge_aim: pd.DataFrame, challenge_vdj: pd.DataFrame | None = None, participants: pd.DataFrame | None = None, participant_hla: pd.DataFrame | None = None, expected_real_counts: bool = True, epitope_hla_map_available: bool = False) -> dict[str, Any]:
    prepared = _prepare_aim(challenge_aim)
    collapsed, metadata_conflicts = _collapsed_aim(prepared)
    challenge_participants = set(prepared["participant_id"])
    anchor = build_task_14_anchor(challenge_aim)[["participant_id", "anchor"]].copy()
    anchor["participant_id"] = anchor["participant_id"].astype(str)
    if anchor["participant_id"].duplicated().any():
        raise DataContractError("E07 Task1.4 anchor is not one row per participant")
    stimuli = sorted(set(prepared["_stim"].astype(str)))
    names = sorted(set(prepared["_name"].astype(str)))
    timepoints = sorted(set(prepared["_time"].astype(str)), key=lambda value: (value == "Pre-vacc", value))
    unit_categories = sorted(_text(value) for value in prepared["unit"].dropna().unique() if _text(value))
    material_categories = sorted(_text(value) for value in prepared["material"].dropna().unique() if _text(value))
    parent_hashes = sorted({_sha(_text(value).casefold()) for value in prepared["parent_population"]})
    population_definition_hashes = sorted({_sha(_text(value).casefold()) for value in prepared["population_definition"]})
    prevacc = _prevacc_arithmetic_audit(collapsed)
    repeats = _conserved_repeat_audit(collapsed)
    negative = _negative_control(collapsed, anchor)
    vdj = _vdj_audit(challenge_vdj, challenge_participants)
    hla = _hla_audit(participants, participant_hla, challenge_participants, epitope_hla_map_available=epitope_hla_map_available)
    real_contract = None
    status = "complete"
    if expected_real_counts:
        real_contract = {"aim_rows": int(len(prepared)), "challenge_subjects": int(len(challenge_participants)), "anchor_subjects": int(len(anchor)), "stimulation_count": len(stimuli), "baseline_timepoints": sorted(set(timepoints)), "metadata_conflicts": int(metadata_conflicts)}
        real_contract["all_pass"] = bool(real_contract["aim_rows"] == EXPECTED_AIM_ROWS and real_contract["challenge_subjects"] == EXPECTED_CHALLENGE_SUBJECTS and real_contract["anchor_subjects"] == EXPECTED_CHALLENGE_SUBJECTS and real_contract["stimulation_count"] == EXPECTED_STIMULATIONS and set(real_contract["baseline_timepoints"]) == set(EXPECTED_BASELINE_TIMEPOINTS) and real_contract["metadata_conflicts"] == 0 and prevacc["all_within_tolerance"])
        if not real_contract["all_pass"]:
            status = "real_contract_review_required"
    external_teacher_applicable = bool(vdj.get("external_tcr_teacher_applicable") or hla.get("hla_correction_applicable"))
    decision = {"direct_public_aim_teacher_available": False, "supervised_cv_available": False, "background_contrast_sensitivity_available": bool(negative.get("sensitivity_available")), "background_contrast_promotable_without_teacher": False, "external_tcr_teacher_applicable": bool(vdj.get("external_tcr_teacher_applicable")), "hla_correction_applicable": bool(hla.get("hla_correction_applicable")), "external_teacher_applicable": external_teacher_applicable, "recommended_task14_predictor": "raw_pre_vacc_conserved_anchor", "incumbent_changed": False, "public_probe_authorized": False, "decision": "external_teacher_requires_separate_predeclared_model" if external_teacher_applicable else "data_limited_hypothesis_retained"}
    result = {"schema_version": 1, "experiment": EXPERIMENT, "task": TASK, "contract_version": CONTRACT_VERSION, "status": status, "outcomes_accessed": False, "competition_submission_attempted": False, "leaderboard_used_for_selection": False, "automatic_compute_retries": 0, "public_training_aim_rows": 0, "public_training_aim_studies": 0, "aim_audit": {"rows": int(len(prepared)), "subjects": int(len(challenge_participants)), "stimulation_count": len(stimuli), "stimulations": stimuli, "name_count": len(names), "names": names, "timepoints": timepoints, "unit_categories": unit_categories, "material_categories": material_categories, "parent_population_hashes": parent_hashes, "population_definition_hashes": population_definition_hashes, "metadata_conflict_keys": int(metadata_conflicts), "anchor_subjects": int(len(anchor)), "anchor_unique_values": int(anchor["anchor"].nunique()), "anchor_min": float(anchor["anchor"].min()), "anchor_max": float(anchor["anchor"].max()), "prevacc_arithmetic_mean_check": prevacc, "conserved_repeat_reliability": repeats, "negative_control": negative}, "vdj_applicability": vdj, "hla_applicability": hla, "real_contract": real_contract, "decision": decision, "interpretation_limits": ["no_public_AIM_outcomes_for_supervised_training", "lower_baseline_predicts_fold_increase_does_not_imply_inverted_absolute_D7_rank", "generic_TCR_richness_is_not_influenza_specific_CD4_memory_richness", "background_subtraction_sensitivity_is_not_proof_of_official_target_preprocessing", "HLA_requires_conserved_pool_epitope_mapping_before_use"], "reentry_conditions": ["compatible_baseline_D7_AIM_or_T_cell_training_study", "influenza_specific_CD4_memory_TCR_annotation_with_sufficient_challenge_coverage", "paired_chain_epitope_or_multimer_metadata", "validated_conserved_pool_epitope_HLA_map_with_challenge_HLA_coverage", "new_independent_biological_measurement_teacher"]}
    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    for token in ('"participant_id"', '"subject"', '"barcode"', '"contig_id"', '"cdr3"'):
        if token in serialized:
            raise ValueError(f"E07 aggregate leaked row-level field:{token}")
    for participant in challenge_participants:
        if participant and participant in serialized:
            raise ValueError("E07 aggregate leaked participant identifier")
    return result
