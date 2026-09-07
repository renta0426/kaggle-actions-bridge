#!/usr/bin/env python3
"""Regression for E05 007 compatibility binding and aggregate lifecycle."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

CONDITIONS = {
    "b21_reference": {},
    "phase_a_fixed_sequence": {},
    "ridge_main_effects": {},
    "ridge_donor_by_strain_interactions": {},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    ns = runpy.run_path(str(a.runtime), run_name="e05_v6_binding_lifecycle")
    assert ns["REQUEST_ID"] == "20260907-cmi-flu-strategy-e05-hai-donor-strain-007"
    assert ns["TARGET_KERNEL"] == "renta0426/cmi-flu-e05-hai-donor-strain-20260907-007"
    assert ns["SCIENCE_COMMIT"] == "92fac34e52485f6f73ea1b7aedb983d8e8e0ee27"
    assert ns["E05_V4_BLOB"] == "ab93726fe0810299892091b7e0e8f63dcacec4f3"
    assert callable(ns.get("json_safe"))
    for name, digest in ns["LOCKED_REFERENCE_SHA256"].items():
        assert hashlib.sha256(ns["locked_reference_bytes"](name)).hexdigest() == digest

    task_payload = {
        "routing": {
            "routing": "synthetic_contract_only",
            "positive_e05_signal": False,
            "historical_proxy_study_mean_delta_vs_anchor": None,
            "challenge_rank_changed_by_at_least_one_position": False,
            "interaction_panel_proxy_rmse": None,
            "main_effects_panel_proxy_rmse": None,
        },
        "study_out_conditions": CONDITIONS,
    }
    result = {
        "experiment": "strategy_v2_e05_hai_donor_strain",
        "tasks": {"Task2.1": task_payload, "Task2.2": task_payload},
        "feature_contract": {
            "ridge_alpha": 10.0,
            "interaction_count": 8,
            "free_participant_embedding": False,
            "raw_strain_id_one_hot": False,
            "donor_rank_unit": "participant_id_within_study",
            "subject_group_role": "leakage_purge_and_hierarchical_weighting_only",
            "sequence_reference_schema": "organizer_native_or_canonical_explicit_adapter",
            "organizer_sequence_reference_columns": ["Virus", "Sequence", "Status_of_sequence"],
            "legacy_evaluation_without_fold_fields_supported": True,
            "panel_proxy_fold_metrics_contract": "recomputed_from_oof_split_participant_panel",
        },
        "split_contract": {
            "fixed_subject_strain_simultaneous_holdouts": 4,
            "simultaneous_selection_uses_outcomes": False,
        },
        "weight_contract": {
            "study_total_weight": "equal",
            "subject_total_weight_within_study": "equal",
            "strain_total_weight_within_subject": "equal",
            "rows_are_not_counted_as_independent_donors": True,
        },
        "leaderboard_used_for_selection": False,
        "competition_submission_attempted": False,
    }
    safe = ns["json_safe"](dict(result))
    ns["validate_result"](safe)
    summary = ns["render_summary"](safe)

    out = a.output_dir
    if out.exists():
        raise SystemExit("binding lifecycle output already exists")
    out.mkdir(parents=True)
    metrics = out / "metrics.json"
    summary_path = out / "summary.md"
    bridge = out / "bridge-result.json"
    metrics.write_text(json.dumps(safe, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(summary)
    bridge.write_text(
        json.dumps(
            {
                "request_id": ns["REQUEST_ID"],
                "science_commit": ns["SCIENCE_COMMIT"],
                "strategy_e05_blob_sha": ns["E05_BLOB"],
                "strategy_e05_v2_blob_sha": ns["E05_V2_BLOB"],
                "strategy_e05_v3_blob_sha": ns["E05_V3_BLOB"],
                "strategy_e05_v4_blob_sha": ns["E05_V4_BLOB"],
                "hai_transfer_v2_blob_sha": ns["HAI_TRANSFER_V2_BLOB"],
                "competition_submission_attempted": False,
                "leaderboard_used_for_selection": False,
                "contains_participant_identifiers": False,
                "contains_row_level_predictions": False,
                "metrics_sha256": sha(metrics),
                "summary_sha256": sha(summary_path),
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    print("CMI_FLU_E05_V6_BINDING_LIFECYCLE_PASS compat=true validation=true files=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
