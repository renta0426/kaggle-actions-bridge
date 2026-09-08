"""Compatibility refinement for Strategy-v2 E08 same-gate pairing.

The E08 scientific contract is unchanged.  This layer replaces a pandas merge
on raw normalized gate strings with the contract's literal intended operation:
for each participant, require exactly one baseline gate, exactly one D7 gate,
and equality of those two normalized gates.  It is outcome-independent.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from . import strategy_e08 as base

PAIRING_IMPLEMENTATION = "participant_unique_gate_v2"


def _unique_gates_by_participant(rows: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for participant, group in rows.groupby("participant_id", sort=True):
        result[str(participant)] = tuple(sorted(set(group["_gate"].astype(str))))
    return result


def _paired_same_gate_subjects(baseline: pd.DataFrame, day7: pd.DataFrame) -> int:
    before = _unique_gates_by_participant(baseline)
    after = _unique_gates_by_participant(day7)
    return sum(
        len(before[participant]) == 1
        and len(after[participant]) == 1
        and before[participant][0] == after[participant][0]
        for participant in sorted(set(before) & set(after))
    )


def _study_candidate_screen(frame: pd.DataFrame) -> list[dict[str, Any]]:
    historical = frame.loc[frame["_split"].eq("historical")].copy()
    historical["_strict_like"] = historical.apply(base._strict_like, axis=1)
    historical["_broad_like"] = historical.apply(base._broad_like, axis=1)
    baseline = base._baseline_rows(historical)
    day7 = historical.loc[historical["_time"].eq("7")].copy()
    records: list[dict[str, Any]] = []
    for study in sorted(set(historical["_study"].astype(str))):
        b = baseline.loc[baseline["_study"].eq(study)]
        d = day7.loc[day7["_study"].eq(study)]
        for family, column in (("strict_like", "_strict_like"), ("broad_like", "_broad_like")):
            bb = b.loc[b[column]].copy()
            dd = d.loc[d[column]].copy()
            paired = _paired_same_gate_subjects(bb, dd)
            records.append(
                {
                    "study": study,
                    "family": family,
                    "baseline_subjects": int(bb["participant_id"].nunique()),
                    "day7_subjects": int(dd["participant_id"].nunique()),
                    "paired_same_gate_subjects": paired,
                    "gate_count_baseline": int(bb["_gate"].nunique()),
                    "has_minimum_paired_support": paired >= base.MIN_BRIDGE_SUBJECTS,
                }
            )
    return records


def _proxy_intrinsic_pairing(frame: pd.DataFrame) -> dict[str, Any]:
    proxy = frame.loc[frame["_study"].eq("SDY272") & frame["_proxy"]].copy()
    baseline = base._baseline_rows(proxy)
    day7 = proxy.loc[proxy["_time"].eq("7")].copy()
    complete = baseline.loc[
        baseline["_parent"].ne("")
        & baseline["_unit"].ne("")
        & baseline["_material"].ne("")
    ]
    return {
        "baseline_subjects": int(baseline["participant_id"].nunique()),
        "day7_subjects": int(day7["participant_id"].nunique()),
        "paired_same_gate_subjects": _paired_same_gate_subjects(baseline, day7),
        "baseline_subjects_with_parent_unit_material": int(complete["participant_id"].nunique()),
        "cd71_state_counts": {
            str(key): int(value)
            for key, value in proxy["_marker_CD71"].value_counts(dropna=False).sort_index().items()
        },
    }


def run_e08(
    inputs: Any,
    *,
    file_inventory: list[str] | None = None,
    external_manifest_text: str | None = None,
) -> dict[str, Any]:
    result = base.run_e08(
        inputs,
        file_inventory=file_inventory,
        external_manifest_text=external_manifest_text,
    )
    prepared = base._prepare(inputs.tables["public_flow"], inputs.tables["challenge_flow"])
    candidates = _study_candidate_screen(prepared)
    third_strict = [
        row
        for row in candidates
        if row["family"] == "strict_like"
        and row["study"] not in {"2024UGA", "SDY272", "2025LJI"}
        and row["has_minimum_paired_support"]
    ]
    result["pairing_implementation"] = PAIRING_IMPLEMENTATION
    result["historical_candidate_screen"] = candidates
    result["third_strict_like_candidates"] = third_strict
    result["sdy272_proxy_intrinsic"] = _proxy_intrinsic_pairing(prepared)
    result["decision"]["third_strict_like_domain_found"] = bool(third_strict)
    return result
