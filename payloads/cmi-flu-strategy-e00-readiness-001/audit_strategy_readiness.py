#!/usr/bin/env python3
"""Aggregate-only E00 readiness audit; no fitting, network, or submission.

Standalone helpers are synthetic-tested. The repository/data integration must
still be tested inside the authorized CMI-Flu environment. Counts are diagnostic:
an ambiguous key is not proof of an erroneous biological measurement.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd

BASE_COMMIT = "2277401a332ebd704ad0e58eb1a3619f36264e79"


class ReadinessError(ValueError):
    """A failed contract, with no participant values in its message."""


def require(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    if not set(columns).issubset(frame.columns):
        raise ReadinessError("required_columns_missing")


def _cohort_contract(frame: pd.DataFrame) -> None:
    keys = ["participant_id", "subject_group", "study_group"]
    require(frame, keys)
    if frame.empty or frame[keys].isna().any().any():
        raise ReadinessError("empty_cohort_or_missing_group_key")
    if frame["participant_id"].duplicated().any():
        raise ReadinessError("nonunique_participant_in_cohort")


def audit_measurement_mixing(
    frame: pd.DataFrame,
    key_columns: Sequence[str],
    metadata_columns: Sequence[str],
) -> dict[str, Any]:
    """Count keys with conflicting metadata; never export the keys or values.

    Missing metadata is its own level; known+unknown is conservatively flagged.
    Identical repeats are not classified as heterogeneous merely for repeating.
    """
    if not key_columns or not metadata_columns:
        raise ReadinessError("empty_measurement_contract")
    require(frame, key_columns)
    if frame[list(key_columns)].isna().any().any():
        raise ReadinessError("missing_measurement_key")
    present = [c for c in metadata_columns if c in frame]
    grouped = frame.groupby(list(key_columns), dropna=False, observed=True)
    group_count = int(grouped.ngroups)
    mixed = pd.Series(False, index=grouped.size().index)
    details: dict[str, Any] = {}
    for column in present:
        count = grouped[column].nunique(dropna=False)
        conflicts = count.gt(1)
        mixed = mixed | conflicts
        text = frame[column].astype("string").str.strip().str.casefold()
        missing = frame[column].isna() | text.isin(["", "na", "unknown"])
        details[column] = {
            "ambiguous_keys": int(conflicts.sum()),
            "missing_or_unknown_rows": int(missing.sum()),
        }
    missing_columns = [c for c in metadata_columns if c not in frame]
    return {
        "rows": int(len(frame)),
        "aggregation_keys": group_count,
        "keys_with_multiple_rows": int(grouped.size().gt(1).sum()),
        "keys_with_any_metadata_conflict": int(mixed.sum()),
        "metadata_fields": details,
        "unavailable_metadata_columns": missing_columns,
        "status": (
            "no_rows" if frame.empty else
            "requires_review" if mixed.any() or missing_columns else
            "no_conflict_in_checked_columns"
        ),
        "does_not_establish_biological_equivalence": True,
    }


def _attach_view(
    cohort: pd.DataFrame, features: pd.DataFrame, feature_columns: Sequence[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    _cohort_contract(cohort)
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ReadinessError("empty_or_duplicate_feature_contract")
    require(features, ["participant_id", *feature_columns])
    if features["participant_id"].isna().any() or features["participant_id"].duplicated().any():
        raise ReadinessError("nonunique_or_missing_participant_in_view")
    extra = ["study_accession"] if "study_accession" in features else []
    merged = cohort[["participant_id", "subject_group", "study_group"]].merge(
        features[["participant_id", *extra, *feature_columns]],
        on="participant_id", how="left", validate="one_to_one", sort=False,
    )
    if extra:
        known = merged["study_accession"].notna()
        mismatch = known & merged["study_accession"].astype(str).ne(merged["study_group"].astype(str))
        if mismatch.any():
            raise ReadinessError("view_study_mismatch")
    values = merged[list(feature_columns)].apply(pd.to_numeric, errors="raise")
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ReadinessError("infinite_view_value")
    available = values.notna().any(axis=1).to_numpy(dtype=bool)
    merged["__has_view"] = available
    return merged, available


def coverage_summary(
    cohort: pd.DataFrame, features: pd.DataFrame, feature_columns: Sequence[str]
) -> dict[str, Any]:
    merged, available = _attach_view(cohort, features, feature_columns)
    studies = []
    for study, group in merged.groupby("study_group", observed=True, sort=True):
        has = group["__has_view"]
        studies.append({
            "study": str(study), "rows": int(len(group)),
            "subjects": int(group["subject_group"].nunique()),
            "rows_with_view": int(has.sum()),
            "subjects_with_view": int(group.loc[has, "subject_group"].nunique()),
            "rows_with_complete_view": int(group[list(feature_columns)].notna().all(axis=1).sum()),
        })
    return {
        "rows": int(len(merged)), "subjects": int(merged["subject_group"].nunique()),
        "rows_with_view": int(available.sum()), "rows_without_view": int((~available).sum()),
        "feature_count": len(feature_columns), "studies": studies,
        "availability_definition": "at_least_one_finite_feature; not biological compatibility",
        "complete_case_filter_applied": False,
    }


def purged_support(
    cohort: pd.DataFrame, features: pd.DataFrame, feature_columns: Sequence[str]
) -> list[dict[str, Any]]:
    """LOSO support after removing every held-study subject from all sources."""
    merged, _ = _attach_view(cohort, features, feature_columns)
    output = []
    for study in sorted(merged["study_group"].unique(), key=str):
        held = merged["study_group"].eq(study)
        held_subjects = set(merged.loc[held, "subject_group"])
        source = (~held) & (~merged["subject_group"].isin(held_subjects))
        supported = source & merged["__has_view"]
        output.append({
            "held_study": str(study),
            "held_rows": int(held.sum()),
            "held_rows_with_view": int((held & merged["__has_view"]).sum()),
            "source_rows_before_purge": int((~held).sum()),
            "source_rows_after_purge": int(source.sum()),
            "source_rows_with_view_after_purge": int(supported.sum()),
            "source_subjects_with_view_after_purge": int(merged.loc[supported, "subject_group"].nunique()),
            "source_studies_with_view_after_purge": int(merged.loc[supported, "study_group"].nunique()),
        })
    return output


def missing_view_fallback(base: Sequence[float], expert: Sequence[float], weight: float) -> np.ndarray:
    """Small reference helper for testing the missing-view contract, not a model."""
    b, e = np.asarray(base, dtype=float), np.asarray(expert, dtype=float)
    if b.ndim != 1 or b.shape != e.shape or not np.isfinite(b).all():
        raise ReadinessError("invalid_fusion_arrays")
    if not np.isfinite(weight) or not 0 <= weight <= 1 or np.isinf(e).any():
        raise ReadinessError("invalid_fusion_weight_or_expert")
    mask = np.isfinite(e)
    result = b.copy()
    result[mask] = (1 - weight) * b[mask] + weight * e[mask]
    return result


def encode_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def run_repository_audit(config_path: Path, data_dir: Path | None, external_dir: Path | None) -> dict[str, Any]:
    """Use the pinned repository API, not a copied or redefined target builder."""
    from cmi_flu.aliases import canonicalize_cytokine, canonicalize_flow_population, canonicalize_timepoint
    from cmi_flu.configuration import load_baseline_config
    from cmi_flu.datasets import build_task_11_dataset
    from cmi_flu.features.flow import build_flow_baseline_features
    from cmi_flu.runner import load_inputs
    from cmi_flu.task11_prior_immunity import build_prior_immunity_features, SEROLOGY_RANK_COLUMNS

    config = replace(load_baseline_config(config_path), verify_md5=True)
    if config.baseline != "b021_taskwise_robust":
        raise ReadinessError("requires_b021_config")
    if data_dir is not None:
        config = replace(config, data_dir=data_dir.resolve())
    if external_dir is not None:
        config = replace(config, external_dir=external_dir.resolve())
    inputs = load_inputs(config)  # Full MD5 is mandatory; failure propagates.
    if inputs.checksum_report is None:
        raise ReadinessError("checksum_verification_absent")
    tables = inputs.tables
    task = build_task_11_dataset(
        tables["public_cytokine"], tables["challenge_cytokine"],
        tables["participants"], tables["investigations"],
    )
    if len(task.train) != 127 or len(task.challenge) != 40:
        raise ReadinessError("snapshot_task11_cohort_count_changed")
    public_flow = build_flow_baseline_features(tables["public_flow"], mode="broad")
    challenge_flow = build_flow_baseline_features(tables["challenge_flow"], mode="broad")
    common = sorted(c for c in public_flow if c.startswith("flow_rank__") and c in challenge_flow)
    if not common:
        raise ReadinessError("no_shared_flow_features")
    innate = [c for c in common if any(x in c.casefold() for x in ("monocyte", "dendritic", "_dc", "_nk_", "natural_kill"))]
    priors = build_prior_immunity_features(tables["public_serology"])
    challenge_priors = build_prior_immunity_features(tables["challenge_serology"])
    view_reports = {}
    for name, public, challenge, columns in [
        ("all_shared_flow", public_flow, challenge_flow, common),
        ("innate_name_candidates_not_final_panel", public_flow, challenge_flow, innate),
        ("historical_hai_prior", priors, challenge_priors, list(SEROLOGY_RANK_COLUMNS)),
    ]:
        if not columns:
            view_reports[name] = {"status": "no_candidate_columns"}
            continue
        view_reports[name] = {
            "feature_columns": columns,
            "train": coverage_summary(task.train, public, columns),
            "challenge": coverage_summary(task.challenge, challenge, columns),
            "purged_support": purged_support(task.train, public, columns),
        }

    flow = tables["public_flow"].copy()
    flow["name"] = flow["name"].map(canonicalize_flow_population)
    flow["timepoint"] = flow["timepoint"].map(canonicalize_timepoint)
    require(flow, ["participant_id", "study_accession", "timepoint", "name", "population_definition"])
    gate = flow["population_definition"].fillna("").astype(str).str.upper()
    proxy = (
        flow["study_accession"].eq("SDY272") & flow["name"].str.casefold().eq("b_cells")
        & gate.str.contains("CD19+", regex=False)
        & gate.str.contains(r"(?:MS4A1|CD20)-", regex=True)
        & gate.str.contains("CD27+", regex=False) & gate.str.contains("CD38++", regex=False)
    )
    flow_keys = ["participant_id", "study_accession", "timepoint", "name"]
    flow_meta = ["unit", "material", "population_definition"]
    audits = {}
    for label, mask in [
        ("task12_day1", flow["name"].eq("Classical_monocytes") & flow["timepoint"].eq("1")),
        ("task13_strict_day7", flow["name"].eq("Antibody-secreting_cells_(ASC)") & flow["timepoint"].eq("7")),
        ("task13_sdy272_proxy_day7", proxy & flow["timepoint"].eq("7")),
    ]:
        audits[label] = audit_measurement_mixing(flow.loc[mask], flow_keys, flow_meta)
    numeric_time = pd.to_numeric(flow["timepoint"], errors="coerce")
    baseline_mask = flow["timepoint"].eq("Pre-vacc") | numeric_time.le(0)
    audits["public_flow_baseline_same_timepoint"] = audit_measurement_mixing(
        flow.loc[baseline_mask], flow_keys, flow_meta,
    )
    challenge = tables["challenge_flow"].copy()
    challenge["name"] = challenge["name"].map(canonicalize_flow_population)
    challenge["timepoint"] = challenge["timepoint"].map(canonicalize_timepoint)
    numeric_time = pd.to_numeric(challenge["timepoint"], errors="coerce")
    audits["challenge_flow_baseline_same_timepoint"] = audit_measurement_mixing(
        challenge.loc[challenge["timepoint"].eq("Pre-vacc") | numeric_time.le(0)], flow_keys, flow_meta,
    )
    cytokine = tables["public_cytokine"].copy()
    cytokine["analyte"] = cytokine["analyte"].map(canonicalize_cytokine)
    cytokine["timepoint"] = cytokine["timepoint"].map(canonicalize_timepoint)
    ct = pd.to_numeric(cytokine["timepoint"], errors="coerce")
    cmask = cytokine["analyte"].eq("CXCL10") & (ct.le(1) | cytokine["timepoint"].eq("Pre-vacc"))
    audits["ip10_baseline_day1_same_timepoint"] = audit_measurement_mixing(
        cytokine.loc[cmask], ["participant_id", "study_accession", "timepoint", "analyte"],
        ["unit", "material", "assay"],
    )
    return {
        "schema_version": 1, "audit": "strategy_20260907_E00_partial",
        "source_review_base_commit": BASE_COMMIT,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "md5_verification_requested": True, "md5_loader_completed": True,
        "views": view_reports, "measurement_ambiguity": audits,
        "ready_for_science_launch": False,
        "remaining_gates": ["review_actual_gate_and_parent_denominator", "frozen_runtime_smoke_test",
                            "paired_OOF_provenance", "live_rules_and_separate_run_approval"],
        "contains_participant_identifiers": False,
        "fitting_performed": False, "submission_created": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--external-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ReadinessError("output_already_exists")
        # Do not print upstream warning contents, which may enumerate private fields.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            report = run_repository_audit(args.config, args.data_dir, args.external_dir)
        report["captured_warning_count"] = len(captured)
        payload = encode_json(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents overwriting a prior immutable audit result.
        with args.output.open("xb") as handle:
            handle.write(payload)
        print(json.dumps({"status": "completed_aggregate_audit", "bytes": len(payload),
                          "sha256": hashlib.sha256(payload).hexdigest(), "science_ready": False}))
        return 0
    except Exception as exc:
        # No traceback or exception message: existing library errors can contain IDs.
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__,
                          "error_sha256": hashlib.sha256(str(exc).encode()).hexdigest()}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
