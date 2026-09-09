"""Strategy-v2 E04b: Task1.3 PBMC/PBMCs-only strict measurement bridge.

E04b deliberately reuses the frozen E04 model/evaluation code.  Its only
scientific change is an outcome-independent normalization of the plural
material literal ``PBMCs`` to ``PBMC`` on named strict ASC rows from 2024UGA
or 2025LJI before E04's measurement audit.  Non-ASC rows are untouched so the
experiment cannot silently add new auxiliary views.
"""
from __future__ import annotations

from types import SimpleNamespace
import re
from typing import Any

import numpy as np
import pandas as pd

from .aliases import canonicalize_flow_population
from .strategy_e04 import run_e04
from .strategy_e04_contract import ASC, study_key, text

EXPERIMENT = "strategy_v2_e04b_task13_material_plural_bridge"
PARENT_EXPERIMENT = "strategy_v2_e04_task13_anchor_preserving_rescue"
ONTOLOGY_VERSION = "e04b_task13_material_plural_v1"
ONTOLOGY_SCOPE = "named_strict_ASC_2024UGA_2025LJI_only"
REAL_EXPECTED_BASELINE_COMPATIBLE_2024UGA = 33
REAL_EXPECTED_CHALLENGE_COMPATIBLE_2025LJI = 40
REAL_EXPECTED_SUPERVISED_STRICT_SOURCE = 23
REAL_EXPECTED_CHALLENGE_CORRECTION_ROWS = 40
LOCAL_TOLERANCE = 2e-6
E04_STRICT_CANDIDATE = 0.3713227226429786
E04_STRICT_ANCHOR = 0.38158872734833993
E04_STRICT_DELTA = -0.01026600470536132


def _compact(value: Any) -> str:
    return re.sub(r"[\s_-]+", "", text(value))


def _plural_strict_asc_mask(frame: pd.DataFrame) -> pd.Series:
    required = {"study_accession", "name", "material"}
    if not required.issubset(frame.columns):
        raise ValueError("E04b flow metadata schema incomplete")
    studies = frame["study_accession"].map(study_key)
    names = frame["name"].map(canonicalize_flow_population)
    plural = frame["material"].map(_compact).eq("pbmcs")
    return studies.isin(["2024UGA", "2025LJI"]) & names.eq(ASC) & plural


def apply_material_plural_ontology(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Return copied tables with the single predeclared ASC material alias applied."""
    if "public_flow" not in tables or "challenge_flow" not in tables:
        raise ValueError("E04b requires public_flow and challenge_flow")
    copied = dict(tables)
    audit: dict[str, Any] = {
        "ontology_version": ONTOLOGY_VERSION,
        "scope": ONTOLOGY_SCOPE,
        "source_literal": "pbmcs",
        "canonical_literal": "pbmc",
        "outcomes_inspected": False,
        "non_material_columns_unchanged": True,
        "non_asc_rows_changed": 0,
        "changed_rows": 0,
        "changed_rows_by_split": {},
        "changed_rows_by_study": {},
    }
    for split, key in (("historical", "public_flow"), ("challenge", "challenge_flow")):
        original = tables[key]
        frame = original.copy(deep=True)
        mask = _plural_strict_asc_mask(frame)
        changed = int(mask.sum())
        if changed:
            frame.loc[mask, "material"] = "PBMC"
        # The wrapper may alter only material values on the exact mask.
        other = [column for column in frame.columns if column != "material"]
        if not frame[other].equals(original[other]):
            raise ValueError("E04b ontology modified non-material columns")
        material_changed = ~frame["material"].astype(str).eq(original["material"].astype(str))
        if not np.array_equal(material_changed.to_numpy(bool), mask.to_numpy(bool)):
            raise ValueError("E04b material changes escaped the strict alias mask")
        non_asc_changed = int((material_changed & ~frame["name"].map(canonicalize_flow_population).eq(ASC)).sum())
        if non_asc_changed:
            raise ValueError("E04b changed a non-ASC row")
        by_study: dict[str, int] = {}
        if changed:
            normalized_study = original.loc[mask, "study_accession"].map(study_key)
            by_study = {str(name): int(count) for name, count in normalized_study.value_counts().sort_index().items()}
        copied[key] = frame
        audit["changed_rows"] += changed
        audit["changed_rows_by_split"][split] = changed
        audit["changed_rows_by_study"].update(by_study)
    return copied, audit


def _coverage_by_study(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = (result.get("measurement_audit") or {}).get("paired_coverage") or []
    return {str(row.get("study")): row for row in rows}


def _strict_local_reproduction(result: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    deltas = []
    for name in ("proxy_weight_0.25", "proxy_weight_1"):
        item = (result.get("strict_cv") or {}).get(name) or {}
        metrics = item.get("metrics") or {}
        paired = item.get("paired_vs_anchor") or {}
        candidates.append(metrics.get("study_equal_spearman"))
        deltas.append(paired.get("paired_study_mean_delta"))
    anchor = (result.get("strict_anchor") or {}).get("study_equal_spearman")
    checks = [
        value is not None and abs(float(value) - E04_STRICT_CANDIDATE) <= LOCAL_TOLERANCE
        for value in candidates
    ]
    checks += [
        value is not None and abs(float(value) - E04_STRICT_DELTA) <= LOCAL_TOLERANCE
        for value in deltas
    ]
    checks.append(anchor is not None and abs(float(anchor) - E04_STRICT_ANCHOR) <= LOCAL_TOLERANCE)
    return {
        "tolerance": LOCAL_TOLERANCE,
        "expected_candidate_study_equal_spearman": E04_STRICT_CANDIDATE,
        "expected_anchor_study_equal_spearman": E04_STRICT_ANCHOR,
        "expected_delta_vs_anchor": E04_STRICT_DELTA,
        "candidate_values": candidates,
        "delta_values": deltas,
        "anchor_value": anchor,
        "all_pass": bool(all(checks)),
    }


def run_e04b(
    config: Any,
    inputs: Any,
    *,
    expected_counts: tuple[int, int, int] | None = (23, 68, 40),
) -> dict[str, Any]:
    """Run E04b using the unchanged E04 model after the narrow ontology transform."""
    tables, ontology = apply_material_plural_ontology(inputs.tables)
    result = dict(run_e04(config, SimpleNamespace(tables=tables), expected_counts=expected_counts))
    if result.get("experiment") != PARENT_EXPERIMENT:
        raise ValueError("E04b parent experiment identity changed")
    result["experiment"] = EXPERIMENT
    result["parent_experiment"] = PARENT_EXPERIMENT
    result["ontology_change"] = ontology
    result["ontology_version"] = ONTOLOGY_VERSION
    result.setdefault("interpretation_limits", []).extend(
        [
            "e04b_alias_applies_only_named_strict_ASC_rows",
            "e04_local_negative_result_remains_binding",
            "baseline_measurement_compatibility_is_not_supervised_sample_count",
        ]
    )

    coverage = _coverage_by_study(result)
    challenge_items = {
        name: (result.get("challenge") or {}).get(name) or {}
        for name in ("proxy_weight_0.25", "proxy_weight_1")
    }
    bridge_sources = [item.get("strict_bridge_source_subjects") for item in challenge_items.values()]
    correction_rows = [(item.get("fit") or {}).get("correction_applied") for item in challenge_items.values()]
    fit_states = [(item.get("fit") or {}).get("state") for item in challenge_items.values()]
    local = _strict_local_reproduction(result)

    real_preconditions = None
    if expected_counts is not None:
        uga = coverage.get("2024UGA") or {}
        lji = coverage.get("2025LJI") or {}
        real_preconditions = {
            "baseline_compatible_2024UGA": uga.get("challenge_gate_compatible_subjects"),
            "challenge_compatible_2025LJI": lji.get("challenge_gate_compatible_subjects"),
            "supervised_strict_source_subjects": bridge_sources,
            "challenge_correction_rows": correction_rows,
            "fit_states": fit_states,
            "selected_auxiliary_count": (result.get("measurement_audit") or {}).get("selected_auxiliary_count"),
        }
        expected_ok = (
            real_preconditions["baseline_compatible_2024UGA"] == REAL_EXPECTED_BASELINE_COMPATIBLE_2024UGA
            and real_preconditions["challenge_compatible_2025LJI"] == REAL_EXPECTED_CHALLENGE_COMPATIBLE_2025LJI
            and bridge_sources == [REAL_EXPECTED_SUPERVISED_STRICT_SOURCE] * 2
            and correction_rows == [REAL_EXPECTED_CHALLENGE_CORRECTION_ROWS] * 2
            and fit_states == ["evaluated", "evaluated"]
            and real_preconditions["selected_auxiliary_count"] == 0
            and local["all_pass"]
        )
        real_preconditions["all_pass"] = bool(expected_ok)
        if not expected_ok:
            result["status"] = "e04b_precondition_or_local_reproduction_review_required"

    any_candidate = any(bool(item.get("local_candidate_heuristic")) for item in challenge_items.values())
    bridge_recovered = all(
        isinstance(value, int) and value >= 8 for value in bridge_sources
    ) and all(state == "evaluated" for state in fit_states)
    strict_delta = local["delta_values"][0] if local["delta_values"] else None
    result["e04b_transfer_audit"] = {
        "baseline_compatible_2024UGA_subjects": (coverage.get("2024UGA") or {}).get("challenge_gate_compatible_subjects"),
        "baseline_compatible_2025LJI_subjects": (coverage.get("2025LJI") or {}).get("challenge_gate_compatible_subjects"),
        "supervised_strict_source_subjects": bridge_sources,
        "challenge_correction_rows": correction_rows,
        "fit_states": fit_states,
        "measurement_bridge_recovered": bool(bridge_recovered),
        "strict_local_reproduction": local,
        "real_preconditions": real_preconditions,
    }
    result["e04b_decision"] = {
        "measurement_bridge_recovered": bool(bridge_recovered),
        "strict_local_delta_vs_anchor": strict_delta,
        "strict_local_gate_passed": bool(strict_delta is not None and float(strict_delta) >= 0.02),
        "any_parent_candidate_heuristic": bool(any_candidate),
        "competition_candidate": bool(any_candidate and bridge_recovered and result.get("status") == "complete"),
        "public_probe_authorized": False,
        "decision": "candidate_requires_review" if any_candidate else "no_promotion",
        "reason": "parent_local_candidate_heuristic_passed" if any_candidate else "bridge_recovered_but_strict_local_residual_not_supported",
    }
    result["incumbent_changed"] = False
    result["competition_submission_attempted"] = False
    result["leaderboard_used_for_selection"] = False
    return result
