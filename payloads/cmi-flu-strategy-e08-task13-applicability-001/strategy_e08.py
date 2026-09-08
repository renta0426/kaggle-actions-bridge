"""Strategy-v2 E08: outcome-independent Task1.3 measurement applicability audit.

This module does not fit a predictive model and does not inspect outcome values.
It asks which *measurement contract* prevents a 2024UGA/2025LJI ASC bridge,
quantifies a narrowly predeclared PBMC/PBMCs lexical sensitivity, and screens
all historical flow studies for additional ASC-like paired baseline/D7 domains.
Only aggregate dictionaries are exportable.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

import pandas as pd

from .aliases import canonicalize_flow_population, canonicalize_timepoint
from .strategy_e04_contract import (
    ASC,
    MARKERS,
    STUDIES,
    gate_text,
    marker_state,
    processing_state,
    study_key,
    text,
)
from .task13_harmonization import sdy272_asc_proxy_mask

EXPERIMENT = "strategy_v2_e08_task13_applicability_audit"
ONTOLOGY_VERSION = "e08_task13_ontology_audit_v1"
MIN_BRIDGE_SUBJECTS = 8


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _compact(value: Any) -> str:
    return re.sub(r"[\s_-]+", "", text(value))


def material_plural_only(value: Any) -> str:
    """Canonicalize only the literal PBMC/PBMCs singular/plural spelling."""
    raw = text(value)
    return "pbmc" if _compact(raw) in {"pbmc", "pbmcs"} else raw


def parent_category(value: Any) -> str:
    key = _compact(value)
    aliases = {
        "pbmc": "PBMC",
        "pbmcs": "PBMC",
        "bcells": "B_cells",
        "lymphocytes": "lymphocytes",
        "livecells": "live_cells",
    }
    return aliases.get(key, "other_reported" if key else "not_reported")


def material_category(value: Any) -> str:
    key = _compact(value)
    return "PBMC" if key in {"pbmc", "pbmcs"} else ("other_reported" if key else "not_reported")


def unit_category(value: Any) -> str:
    raw = text(value)
    key = re.sub(r"\s+", "", raw)
    aliases = {
        "percentage": "percentage",
        "%ofparent": "% of parent",
        "cells": "cells",
        "cells/ul": "cells/ul",
        "percentile": "percentile",
    }
    return aliases.get(key, "other_reported" if key else "not_reported")


def _marker_tuple(row: pd.Series) -> tuple[str, ...]:
    return tuple(str(row[f"_marker_{marker}"]) for marker in MARKERS)


def _signature(row: pd.Series, level: str) -> str:
    gate = str(row["_gate"])
    processing = str(row["_processing"])
    if level == "exact_e04":
        parts: list[Any] = [gate, row["_parent"], row["_unit"], row["_material"], processing]
    elif level == "material_plural_only":
        parts = [gate, row["_parent"], row["_unit"], material_plural_only(row["_material"]), processing]
    elif level == "category_non_gate":
        parts = [
            gate,
            parent_category(row["_parent"]),
            unit_category(row["_unit"]),
            material_category(row["_material"]),
            processing,
        ]
    elif level == "marker_category_upper_bound":
        parts = [
            _marker_tuple(row),
            parent_category(row["_parent"]),
            unit_category(row["_unit"]),
            material_category(row["_material"]),
            processing,
        ]
    else:
        raise ValueError(f"unknown E08 compatibility level: {level}")
    return _sha(parts)


LEVELS = (
    "exact_e04",
    "material_plural_only",
    "category_non_gate",
    "marker_category_upper_bound",
)


def _prepare(public_flow: pd.DataFrame, challenge_flow: pd.DataFrame) -> pd.DataFrame:
    required = {
        "participant_id", "study_accession", "timepoint", "name",
        "population_definition", "parent_population", "unit", "material",
    }
    for frame in (public_flow, challenge_flow):
        if not required.issubset(frame.columns):
            raise ValueError("E08 flow metadata schema incomplete")

    source = public_flow.copy()
    source["_split"] = "historical"
    challenge = challenge_flow.copy()
    challenge["_split"] = "challenge"
    frame = pd.concat([source, challenge], ignore_index=True)
    frame["_study"] = frame["study_accession"].map(study_key)
    # Preserve unknown historical study accessions for the all-study screen.
    unknown = frame["_study"].eq("other_study")
    frame.loc[unknown, "_study"] = frame.loc[unknown, "study_accession"].map(lambda x: text(x).upper().replace(" ", ""))
    frame["_time"] = frame["timepoint"].map(canonicalize_timepoint)
    frame["_numeric_time"] = pd.to_numeric(frame["_time"], errors="coerce")
    frame["_name"] = frame["name"].map(canonicalize_flow_population)
    proxy = sdy272_asc_proxy_mask(frame)
    frame["_proxy"] = proxy
    frame.loc[proxy, "_name"] = ASC
    frame["_gate"] = frame["population_definition"].map(gate_text)
    frame["_parent"] = frame["parent_population"].map(text)
    frame["_unit"] = frame["unit"].map(text)
    frame["_material"] = frame["material"].map(text)
    comments = frame["comments"] if "comments" in frame.columns else pd.Series("", index=frame.index)
    frame["_processing"] = comments.map(processing_state)
    for marker in MARKERS:
        frame[f"_marker_{marker}"] = frame["population_definition"].map(lambda x, m=marker: marker_state(x, m))
    for level in LEVELS:
        frame[f"_sig_{level}"] = frame.apply(lambda row, lvl=level: _signature(row, lvl), axis=1)
    return frame


def _baseline_rows(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = frame.loc[frame["_time"].eq("Pre-vacc") | frame["_numeric_time"].le(0)].copy()
    chosen: list[pd.DataFrame] = []
    for _, group in candidates.groupby(["participant_id", "_study", "_name"], sort=True):
        literal = group.loc[group["_time"].eq("Pre-vacc")]
        chosen.append(literal if len(literal) else group)
    return pd.concat(chosen, ignore_index=True) if chosen else frame.iloc[:0].copy()


def _participant_signatures(rows: pd.DataFrame, level: str) -> pd.DataFrame:
    column = f"_sig_{level}"
    records = []
    for (participant, study), group in rows.groupby(["participant_id", "_study"], sort=True):
        signatures = sorted(set(group[column].astype(str)))
        records.append(
            {
                "participant_id": str(participant),
                "study": str(study),
                "signature": signatures[0] if len(signatures) == 1 else "conflict",
                "homogeneous": len(signatures) == 1,
            }
        )
    return pd.DataFrame(records, columns=["participant_id", "study", "signature", "homogeneous"])


def _challenge_signature(rows: pd.DataFrame, level: str) -> str | None:
    signatures = _participant_signatures(rows, level)
    if signatures.empty or not bool(signatures["homogeneous"].all()) or signatures["signature"].nunique() != 1:
        return None
    return str(signatures["signature"].iloc[0])


def _bridge_counts(baseline: pd.DataFrame) -> dict[str, Any]:
    strict = baseline.loc[baseline["_name"].eq(ASC) & baseline["_study"].isin(STUDIES)].copy()
    output: dict[str, Any] = {}
    for level in LEVELS:
        challenge_rows = strict.loc[strict["_study"].eq("2025LJI")]
        target = _challenge_signature(challenge_rows, level)
        subjects = _participant_signatures(strict.loc[strict["_study"].eq("2024UGA")], level)
        compatible = 0 if target is None else int((subjects["homogeneous"] & subjects["signature"].eq(target)).sum())
        output[level] = {
            "challenge_signature_defined": target is not None,
            "2024UGA_compatible_subjects": compatible,
            "meets_minimum_bridge": compatible >= MIN_BRIDGE_SUBJECTS,
        }
    return output


def _pair_metadata(rows: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    strict = rows.loc[rows["_name"].eq(ASC) & rows["_study"].isin(["2024UGA", "2025LJI"])]
    for study in ("2024UGA", "2025LJI"):
        group = strict.loc[strict["_study"].eq(study)]
        if group.empty:
            result[study] = {"rows": 0, "subjects": 0, "contracts": []}
            continue
        contracts = []
        for _, h in group.groupby("_sig_exact_e04", sort=True):
            row = h.iloc[0]
            contracts.append(
                {
                    "rows": int(len(h)),
                    "subjects": int(h["participant_id"].nunique()),
                    "gate_sha256": _sha(row["_gate"]),
                    "markers": {marker: row[f"_marker_{marker}"] for marker in MARKERS},
                    "parent_category": parent_category(row["_parent"]),
                    "parent_sha256": _sha(row["_parent"]) if row["_parent"] else None,
                    "unit_category": unit_category(row["_unit"]),
                    "unit_sha256": _sha(row["_unit"]) if row["_unit"] else None,
                    "material_category": material_category(row["_material"]),
                    "material_sha256": _sha(row["_material"]) if row["_material"] else None,
                    "processing": row["_processing"],
                }
            )
        result[study] = {
            "rows": int(len(group)),
            "subjects": int(group["participant_id"].nunique()),
            "contracts": contracts,
        }
    return result


def _strict_like(row: pd.Series) -> bool:
    return (
        row["_marker_CD20"] in {"low", "negative"}
        and row["_marker_CD27"] == "high"
        and row["_marker_CD38"] == "high"
        and row["_marker_CD71"] == "high"
    )


def _broad_like(row: pd.Series) -> bool:
    return (
        row["_marker_CD20"] in {"low", "negative"}
        and row["_marker_CD27"] in {"positive", "high"}
        and row["_marker_CD38"] == "high"
    )


def _study_candidate_screen(frame: pd.DataFrame) -> list[dict[str, Any]]:
    historical = frame.loc[frame["_split"].eq("historical")].copy()
    historical["_strict_like"] = historical.apply(_strict_like, axis=1)
    historical["_broad_like"] = historical.apply(_broad_like, axis=1)
    baseline = _baseline_rows(historical)
    day7 = historical.loc[historical["_time"].eq("7")].copy()
    records = []
    for study in sorted(set(historical["_study"].astype(str))):
        b = baseline.loc[baseline["_study"].eq(study)]
        d = day7.loc[day7["_study"].eq(study)]
        for family, column in (("strict_like", "_strict_like"), ("broad_like", "_broad_like")):
            bb = b.loc[b[column]].copy()
            dd = d.loc[d[column]].copy()
            bkeys = bb[["participant_id", "_gate"]].drop_duplicates()
            dkeys = dd[["participant_id", "_gate"]].drop_duplicates()
            paired = bkeys.merge(dkeys, on=["participant_id", "_gate"], how="inner")
            records.append(
                {
                    "study": study,
                    "family": family,
                    "baseline_subjects": int(bb["participant_id"].nunique()),
                    "day7_subjects": int(dd["participant_id"].nunique()),
                    "paired_same_gate_subjects": int(paired["participant_id"].nunique()),
                    "gate_count_baseline": int(bb["_gate"].nunique()),
                    "has_minimum_paired_support": int(paired["participant_id"].nunique()) >= MIN_BRIDGE_SUBJECTS,
                }
            )
    return records


def _proxy_intrinsic_pairing(frame: pd.DataFrame) -> dict[str, Any]:
    proxy = frame.loc[frame["_study"].eq("SDY272") & frame["_proxy"]].copy()
    baseline = _baseline_rows(proxy)
    day7 = proxy.loc[proxy["_time"].eq("7")]
    b = baseline[["participant_id", "_gate"]].drop_duplicates()
    d = day7[["participant_id", "_gate"]].drop_duplicates()
    paired = b.merge(d, on=["participant_id", "_gate"], how="inner")
    complete = baseline.loc[
        baseline["_parent"].ne("") & baseline["_unit"].ne("") & baseline["_material"].ne("")
    ]
    return {
        "baseline_subjects": int(baseline["participant_id"].nunique()),
        "day7_subjects": int(day7["participant_id"].nunique()),
        "paired_same_gate_subjects": int(paired["participant_id"].nunique()),
        "baseline_subjects_with_parent_unit_material": int(complete["participant_id"].nunique()),
        "cd71_state_counts": {
            str(key): int(value)
            for key, value in proxy["_marker_CD71"].value_counts(dropna=False).sort_index().items()
        },
    }


def _inventory_summary(file_inventory: Iterable[str] | None, external_manifest_text: str | None) -> dict[str, Any]:
    inventory = list(file_inventory or [])
    fcs = [path for path in inventory if str(path).casefold().endswith(".fcs")]
    manifest = external_manifest_text or ""
    manifest_fcs = len(re.findall(r"(?i)\.fcs(?:`|\s|\||$)", manifest))
    return {
        "competition_input_file_count": len(inventory),
        "competition_input_fcs_count": len(fcs),
        "managed_external_manifest_fcs_mentions": int(manifest_fcs),
        "raw_fcs_availability_established": bool(fcs or manifest_fcs),
    }


def run_e08(
    inputs: Any,
    *,
    file_inventory: Iterable[str] | None = None,
    external_manifest_text: str | None = None,
) -> dict[str, Any]:
    """Run E08 without reading any target/outcome values."""
    tables = inputs.tables
    prepared = _prepare(tables["public_flow"], tables["challenge_flow"])
    baseline = _baseline_rows(prepared)
    bridge = _bridge_counts(baseline)
    candidates = _study_candidate_screen(prepared)
    third_strict = [
        row for row in candidates
        if row["family"] == "strict_like"
        and row["study"] not in {"2024UGA", "SDY272", "2025LJI"}
        and row["has_minimum_paired_support"]
    ]
    alias_only = bool(bridge["material_plural_only"]["meets_minimum_bridge"])
    exact = bool(bridge["exact_e04"]["meets_minimum_bridge"])
    e04b_ready = alias_only and not exact
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "ontology_version": ONTOLOGY_VERSION,
        "task": "Task1.3",
        "status": "complete",
        "outcomes_accessed": False,
        "leaderboard_used_for_selection": False,
        "competition_submission_attempted": False,
        "automatic_compute_retries": 0,
        "minimum_bridge_subjects": MIN_BRIDGE_SUBJECTS,
        "compatibility_levels": list(LEVELS),
        "bridge_counts": bridge,
        "strict_pair_metadata": _pair_metadata(baseline),
        "sdy272_proxy_intrinsic": _proxy_intrinsic_pairing(prepared),
        "historical_candidate_screen": candidates,
        "third_strict_like_candidates": third_strict,
        "file_inventory": _inventory_summary(file_inventory, external_manifest_text),
        "decision": {
            "material_plural_alias_alone_recovers_bridge": e04b_ready,
            "exact_bridge_already_available": exact,
            "e04b_measurement_bridge_ready": e04b_ready,
            "third_strict_like_domain_found": bool(third_strict),
            "raw_fcs_available_in_managed_inputs": _inventory_summary(file_inventory, external_manifest_text)["raw_fcs_availability_established"],
        },
        "interpretation_limits": [
            "marker_category_upper_bound_is_not_gate_equivalence",
            "category_non_gate_compatibility_is_not_processing_equivalence",
            "material_plural_alias_is_a_new_ontology_sensitivity_not_a_reinterpretation_of_E04_v1",
            "flow_assay_presence_does_not_prove_raw_FCS_file_availability",
            "no_outcome_value_used_for_compatibility_or_candidate_screening",
        ],
    }
    # No participant identifiers or raw gate text may leave the private process.
    payload = json.dumps(result, sort_keys=True)
    if "participant_id" in payload or "population_definition" in payload:
        raise ValueError("E08 aggregate privacy contract violated")
    return result
