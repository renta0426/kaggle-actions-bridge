"""Execution-contract repair for Strategy v3 V3-07.

This module does not change the frozen H1/H2 model families, features, alpha,
outer/inner CV, controls, proxy cohorts, or promotion criteria. It repairs two
pre-result execution-contract defects found by the first Kaggle run:

* H1: the global 256 fit guard was an initial engineering limit and stopped a
  deterministic nested-CV run at fit 257. We compute the exact fit count from
  outcome-independent split/panel support before fitting and use that exact
  count as the per-run limit, under a fixed absolute engineering ceiling.
* H2: V3-01's 11,913 count is the number of unique biological
  subject×strain D28/D365 pairs after collapsing repeated participant-years,
  while the actual retention teacher has 18,593 participant-year×strain rows.
  Both grains were present in the pre-result V3-01 audit. We validate both
  independently instead of treating them as the same row count.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import DataContractError
from . import strategy_v3_v07 as v1

REPAIR_VERSION = "strategy_v3_v07_v2_execution_contract_20260913"
H1_ENGINEERING_FIT_CEILING = 384
H1_EXPECTED_LEARNING_STUDIES = 48
H1_EXPECTED_LEARNING_ROWS = 62285
H2_EXPECTED_TEACHER_ROWS = 18593
H2_EXPECTED_TEACHER_PARTICIPANT_YEARS = 1067
H2_EXPECTED_TEACHER_STUDIES = 11
H2_EXPECTED_BIOLOGICAL_PAIRS = 11913
H2_EXPECTED_BIOLOGICAL_SUBJECTS = 567


def _dummy_panel_support(frame: pd.DataFrame, panel: Sequence[str]) -> pd.DataFrame:
    """Return donor/panel support using no fitted prediction or target selection."""
    if frame.empty:
        return pd.DataFrame()
    columns = ["participant_id", "subject_group", "study_group", "virus_strain", "log2_pre_hai"]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise DataContractError(f"V3-07 v2 panel support missing columns: {missing}")
    out = frame[columns].copy()
    if "post_hai" in frame:
        out["post_hai"] = pd.to_numeric(frame["post_hai"], errors="coerce").to_numpy(float)
    else:
        out["post_hai"] = 1.0
    out["post_prediction"] = 1.0
    return v1._panel_donors(out, official_panel=panel, prediction_column="post_prediction")


def _inner_validation_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    frame = frame.reset_index(drop=True)
    splits = v1._nested_splits(frame)
    indices = sorted({int(i) for split in splits for i in split.validation_indices})
    if len(indices) != len(frame):
        raise DataContractError(
            f"V3-07 v2 inner validation coverage changed: {len(indices)}!={len(frame)}"
        )
    return frame.iloc[indices].reset_index(drop=True), len(splits)


def _planned_h1_fit_count_from_frames(
    train: pd.DataFrame,
    challenge: pd.DataFrame,
    panel: Sequence[str],
) -> tuple[int, Mapping[str, Any]]:
    """Mirror H1 branch structure without fitting any model."""
    train = train.reset_index(drop=True)
    challenge = challenge.reset_index(drop=True)
    outer = v1.purged_leave_one_study_out(
        train["study_group"].astype(str), train["subject_group"].astype(str)
    )
    count = 0
    evaluable_outer = 0
    no_calibration_outer = 0
    no_held_panel_outer = 0
    inner_fit_count = 0
    for split in outer:
        tr = train.iloc[split.train_indices].reset_index(drop=True)
        va = train.iloc[split.validation_indices].reset_index(drop=True)
        inner_frame, inner_n = _inner_validation_frame(tr)
        count += inner_n
        inner_fit_count += inner_n
        cal = _dummy_panel_support(inner_frame, panel)
        if cal.empty:
            no_calibration_outer += 1
            continue
        count += 1  # held-study E05 prediction
        donors = _dummy_panel_support(va, panel)
        if donors.empty:
            no_held_panel_outer += 1
            continue
        count += 2  # H1 residual Ridge + ET control
        evaluable_outer += 1

    full_inner_frame, final_inner_n = _inner_validation_frame(train)
    final_cal = _dummy_panel_support(full_inner_frame, panel)
    if final_cal.empty:
        raise DataContractError("V3-07 v2 H1 final calibration support is empty")
    count += final_inner_n
    inner_fit_count += final_inner_n
    count += 1  # final E05 Challenge prediction
    challenge_support = _dummy_panel_support(challenge, panel)
    if challenge_support.empty:
        raise DataContractError("V3-07 v2 H1 Challenge panel support is empty")
    count += 1  # final H1 residual Ridge
    return count, {
        "outer_folds": int(len(outer)),
        "evaluable_outer_folds": int(evaluable_outer),
        "no_calibration_outer_folds": int(no_calibration_outer),
        "no_held_panel_outer_folds": int(no_held_panel_outer),
        "inner_fit_count": int(inner_fit_count),
        "final_inner_folds": int(final_inner_n),
        "planned_fit_count": int(count),
    }


def _build_h1_frames(
    inputs: Any,
    sequence_reference: pd.DataFrame,
    vaccine_reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    panel = v1._canonical_panel(inputs.challenge_strains)
    tables = inputs.tables
    base = v1.build_hai_model_dataset(
        tables["public_serology"],
        tables["challenge_serology"],
        tables["participants"],
        tables["investigations"],
        day=28,
        target_representation="residual",
        challenge_panel_strains=panel,
    )
    ridge = v1.add_hai_ontology_features(
        base,
        vaccine_reference=vaccine_reference,
        sequence_reference=v1.normalize_sequence_reference_schema(sequence_reference),
        vaccine_2025=inputs.vaccine_strains,
        condition="ontology_sequence_local",
    )
    return ridge.train.reset_index(drop=True), ridge.challenge.reset_index(drop=True), panel


def _planned_h1_fit_count(
    inputs: Any,
    sequence_reference: pd.DataFrame,
    vaccine_reference: pd.DataFrame,
) -> tuple[int, Mapping[str, Any]]:
    train, challenge, panel = _build_h1_frames(inputs, sequence_reference, vaccine_reference)
    studies = int(train["study_group"].astype(str).nunique())
    if studies != H1_EXPECTED_LEARNING_STUDIES or len(train) != H1_EXPECTED_LEARNING_ROWS:
        raise DataContractError(
            f"V3-07 v2 H1 learning support changed rows={len(train)} studies={studies}"
        )
    planned, audit = _planned_h1_fit_count_from_frames(train, challenge, panel)
    if planned <= 256 or planned > H1_ENGINEERING_FIT_CEILING:
        raise DataContractError(
            f"V3-07 v2 H1 planned fit count outside repair bound: {planned}"
        )
    return planned, {**dict(audit), "learning_rows": int(len(train)), "learning_studies": studies}


def _biological_pair_support(d28: pd.DataFrame, d365: pd.DataFrame) -> Mapping[str, int]:
    left = d28[["subject_group", "virus_strain"]].copy()
    right = d365[["subject_group", "virus_strain"]].copy()
    left["virus_strain"] = left["virus_strain"].map(v1.canonicalize_strain)
    right["virus_strain"] = right["virus_strain"].map(v1.canonicalize_strain)
    paired = left.drop_duplicates().merge(
        right.drop_duplicates(), on=["subject_group", "virus_strain"], how="inner", validate="one_to_one"
    )
    return {
        "pairs": int(len(paired)),
        "subjects": int(paired["subject_group"].astype(str).nunique()) if len(paired) else 0,
    }


def _h2_support_and_plan(config: Any, inputs: Any) -> tuple[Mapping[str, Any], int, Mapping[str, Any]]:
    panel = v1._canonical_panel(inputs.challenge_strains)
    tables = inputs.tables

    def build(day: int):
        return v1.build_hai_model_dataset(
            tables["public_serology"],
            tables["challenge_serology"],
            tables["participants"],
            tables["investigations"],
            day=day,
            target_representation="residual",
            challenge_panel_strains=panel,
        )

    d28 = build(28)
    d365 = build(365)
    full_pairs = v1._pair_d28_d365(d28.train, d365.train)
    participant_years = int(full_pairs["participant_id_d365"].astype(str).nunique())
    teacher_studies = int(full_pairs["study_group"].astype(str).nunique())
    biological = _biological_pair_support(d28.train, d365.train)
    observed = {
        "participant_year_strain_rows": int(len(full_pairs)),
        "participant_years": participant_years,
        "studies": teacher_studies,
        "biological_subject_strain_pairs": int(biological["pairs"]),
        "biological_subjects": int(biological["subjects"]),
    }
    expected = {
        "participant_year_strain_rows": H2_EXPECTED_TEACHER_ROWS,
        "participant_years": H2_EXPECTED_TEACHER_PARTICIPANT_YEARS,
        "studies": H2_EXPECTED_TEACHER_STUDIES,
        "biological_subject_strain_pairs": H2_EXPECTED_BIOLOGICAL_PAIRS,
        "biological_subjects": H2_EXPECTED_BIOLOGICAL_SUBJECTS,
    }
    if observed != expected:
        raise DataContractError(f"V3-07 v2 H2 support changed: {observed}")

    spec = v1._find_e06b_model(config)
    outer = v1.purged_leave_one_study_out(
        d365.train["study_group"].astype(str), d365.train["subject_group"].astype(str)
    )
    count = 0
    usable = 0
    inner_fit_count = 0
    for split in outer:
        tr365 = d365.train.iloc[split.train_indices].reset_index(drop=True)
        va = d365.train.iloc[split.validation_indices].reset_index(drop=True)
        held = str(split.held_out_group)
        heldsub = set(va["subject_group"].astype(str))
        tr28 = d28.train.loc[
            (d28.train["study_group"].astype(str) != held)
            & ~d28.train["subject_group"].astype(str).isin(heldsub)
        ].reset_index(drop=True)
        _, inner_n = _inner_validation_frame(tr28)
        count += inner_n
        inner_fit_count += inner_n
        pair = v1._pair_d28_d365(tr28, tr365)
        if len(pair) < 10:
            continue
        count += 4  # held D28 + retention + independent D365 + two-head control
        usable += 1

    _, final_inner_n = _inner_validation_frame(d28.train.reset_index(drop=True))
    count += final_inner_n + 4
    inner_fit_count += final_inner_n
    if count > 256:
        raise DataContractError(f"V3-07 v2 H2 planned fit count exceeds frozen bound: {count}>256")
    return observed, int(count), {
        "outer_folds": int(len(outer)),
        "usable_outer_folds": int(usable),
        "inner_fit_count": int(inner_fit_count),
        "final_inner_folds": int(final_inner_n),
        "planned_fit_count": int(count),
    }


def _run_h1(config: Any, inputs: Any, sequence_reference: pd.DataFrame, vaccine_reference: pd.DataFrame):
    planned, audit = _planned_h1_fit_count(inputs, sequence_reference, vaccine_reference)
    old_limit = v1.MAX_FITS_H1
    v1.MAX_FITS_H1 = int(planned)
    try:
        result, oof, challenge = v1._h1(config, inputs, sequence_reference, vaccine_reference)
    finally:
        v1.MAX_FITS_H1 = old_limit
    if int(result.get("fit_count", -1)) != planned:
        raise DataContractError(
            f"V3-07 v2 H1 fit-plan mismatch actual={result.get('fit_count')} planned={planned}"
        )
    result = dict(result)
    result["execution_repair"] = {
        "version": REPAIR_VERSION,
        "reason": "initial_256_engineering_limit_below_deterministic_nested_cv_requirement",
        "initial_fit_limit": 256,
        "fit_limit_policy": "exact_outcome_independent_planned_count",
        **dict(audit),
    }
    return result, oof, challenge


def _run_h2(config: Any, inputs: Any):
    support, planned, audit = _h2_support_and_plan(config, inputs)
    old_rows = v1.H2_EXPECTED_PAIR_ROWS
    old_limit = v1.MAX_FITS_H2
    v1.H2_EXPECTED_PAIR_ROWS = H2_EXPECTED_TEACHER_ROWS
    v1.MAX_FITS_H2 = int(planned)
    try:
        result, oof, challenge = v1._h2(config, inputs)
    finally:
        v1.H2_EXPECTED_PAIR_ROWS = old_rows
        v1.MAX_FITS_H2 = old_limit
    if int(result.get("fit_count", -1)) != planned:
        raise DataContractError(
            f"V3-07 v2 H2 fit-plan mismatch actual={result.get('fit_count')} planned={planned}"
        )
    result = dict(result)
    paired = dict(result.get("paired_teacher") or {})
    paired.update(support)
    paired["row_unit"] = "participant_year_strain"
    paired["biological_pair_unit"] = "subject_strain_deduplicated_across_participant_years"
    result["paired_teacher"] = paired
    contract = dict(result.get("contract") or {})
    contract["teacher_row_unit"] = "participant_year_strain"
    contract["biological_support_unit"] = "subject_strain_deduplicated_across_participant_years"
    result["contract"] = contract
    result["execution_repair"] = {
        "version": REPAIR_VERSION,
        "reason": "v3_01_biological_pair_count_was_mistaken_for_participant_year_teacher_row_count",
        "pre_result_v3_01_biological_pairs": H2_EXPECTED_BIOLOGICAL_PAIRS,
        "pre_result_v3_01_teacher_rows": H2_EXPECTED_TEACHER_ROWS,
        "fit_limit_policy": "exact_outcome_independent_planned_count_within_frozen_256_bound",
        **dict(audit),
    }
    return result, oof, challenge


def run_v3_07(
    condition: str,
    config: Any,
    inputs: Any,
    *,
    sequence_reference: pd.DataFrame | None = None,
    vaccine_reference: pd.DataFrame | None = None,
):
    if condition not in v1.CONDITIONS:
        raise DataContractError(f"unknown V3-07 condition: {condition}")
    if str(config.section("hai").get("target_representation", "residual")) != "residual":
        raise DataContractError("V3-07 requires residual frozen B2.1 config")
    if condition == "task22_panel_mean":
        if sequence_reference is None or vaccine_reference is None:
            raise DataContractError("V3-07 H1 requires sequence/vaccine references")
        result, oof, challenge = _run_h1(config, inputs, sequence_reference, vaccine_reference)
    else:
        result, oof, challenge = _run_h2(config, inputs)
    aggregate = {
        "experiment": v1.EXPERIMENT,
        "new_candidate_conditions": [condition],
        "result": result,
        "fit_count": int(result["fit_count"]),
        "competition_submission_attempted": False,
        "final_submission_selection_attempted": False,
        "public_leaderboard_used": False,
        "saved_outer_oof_reused_as_inner_teacher": False,
        "automatic_parameter_sweep": False,
        "execution_repair_version": REPAIR_VERSION,
    }
    return aggregate, oof, challenge


write_v3_07_outputs = v1.write_v3_07_outputs

__all__ = [
    "REPAIR_VERSION",
    "H1_ENGINEERING_FIT_CEILING",
    "H2_EXPECTED_TEACHER_ROWS",
    "H2_EXPECTED_BIOLOGICAL_PAIRS",
    "run_v3_07",
    "write_v3_07_outputs",
    "_planned_h1_fit_count_from_frames",
    "_biological_pair_support",
]
