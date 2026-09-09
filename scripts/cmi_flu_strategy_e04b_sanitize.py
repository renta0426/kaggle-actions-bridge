#!/usr/bin/env python3
"""Validate and emit the aggregate-only E04b result for science closeout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260909-cmi-flu-strategy-e04b-task13-material-bridge-001"
SCIENCE_COMMIT = "cc51692a66055ed1716fd6d66ac574c4a22f0dcc"
E04B_BLOB = "72f936a7248dfa17338932077d17fb43123cb056"
E04B_SYNTH_BLOB = "6f3c081ea47a0562150398ab307ab1a6b4d4c000"
E04_BLOB = "73c112a8bb3b1ee7a6dd5bfbd5645a945bea6ebb"
CONTRACT_BLOB = "3982541febfb4641fbf895438ae1a465bdbb3d5e"
E04_SYNTHETIC_BLOB = "436b6a971622915cc5b335060f9a850feb6108fd"
TASK13_BLOB = "5c6725dc757a5ba9dd21289b1c4f09997e1afdb8"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
BANNED = (
    '"participant_id"',
    '"subject_group"',
    '"row_index"',
    '"oof_predictions"',
    '"challenge_predictions"',
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.input_dir.resolve()
    files = sorted(path.name for path in root.iterdir() if path.is_file())
    if files != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit(f"E04b safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E04b privacy contract failed:{name}")

    bridge = json.loads(texts["bridge-result.json"])
    result = json.loads(texts["metrics.json"])
    expected_bridge = {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e04b_blob_sha": E04B_BLOB,
        "strategy_e04b_synthetic_blob_sha": E04B_SYNTH_BLOB,
        "strategy_e04_blob_sha": E04_BLOB,
        "strategy_e04_contract_blob_sha": CONTRACT_BLOB,
        "strategy_e04_synthetic_blob_sha": E04_SYNTHETIC_BLOB,
        "task13_harmonization_blob_sha": TASK13_BLOB,
        "config_blob_sha": CONFIG_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
        "synthetic": False,
    }
    for key, value in expected_bridge.items():
        if bridge.get(key) != value:
            raise SystemExit(f"E04b bridge contract mismatch:{key}:{bridge.get(key)!r}")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json") or bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E04b output hash mismatch")
    if not isinstance(bridge.get("md5_verified_count"), int) or bridge.get("md5_verified_count") < 1:
        raise SystemExit("E04b MD5 verification missing")

    if result.get("schema_version") != 1 or result.get("experiment") != "strategy_v2_e04b_task13_material_plural_bridge" or result.get("task") != "Task1.3":
        raise SystemExit("E04b experiment identity mismatch")
    if result.get("parent_experiment") != "strategy_v2_e04_task13_anchor_preserving_rescue" or result.get("ontology_version") != "e04b_task13_material_plural_v1":
        raise SystemExit("E04b parent/ontology mismatch")
    if result.get("status") != "complete" or result.get("competition_submission_attempted") is not False or result.get("leaderboard_used_for_selection") is not False or result.get("automatic_compute_retries") != 0 or result.get("incumbent_changed") is not False:
        raise SystemExit("E04b execution-boundary mismatch")

    frozen = result.get("frozen_conditions") or {}
    expected_frozen = {
        "strict_b21": "pls_1",
        "anchor": "flow_rank__Antibody-secreting_cells_(ASC)",
        "historical_rank": "enet_a0.1_l0.5",
        "residual_alpha": 10.0,
        "residual_shrinkage": 0.25,
        "residual_score_cap": 0.05,
        "proxy_weights": [0.25, 1.0],
    }
    if frozen != expected_frozen:
        raise SystemExit("E04b frozen model contract mismatch")
    reproduction = result.get("frozen_control_reproduction") or {}
    if reproduction.get("all_pass") is not True:
        raise SystemExit("E04b frozen E04 controls did not reproduce")

    ontology = result.get("ontology_change") or {}
    if ontology.get("ontology_version") != "e04b_task13_material_plural_v1" or ontology.get("scope") != "named_strict_ASC_2024UGA_2025LJI_only":
        raise SystemExit("E04b ontology identity mismatch")
    if ontology.get("source_literal") != "pbmcs" or ontology.get("canonical_literal") != "pbmc" or ontology.get("outcomes_inspected") is not False:
        raise SystemExit("E04b ontology literal/outcome boundary mismatch")
    if ontology.get("non_material_columns_unchanged") is not True or ontology.get("non_asc_rows_changed") != 0:
        raise SystemExit("E04b ontology escaped its material/ASC boundary")
    if ontology.get("changed_rows_by_split") != {"challenge": 40, "historical": 0}:
        raise SystemExit(f"E04b unexpected ontology row changes:{ontology.get('changed_rows_by_split')}")
    if ontology.get("changed_rows") != 40 or ontology.get("changed_rows_by_study") != {"2025LJI": 40}:
        raise SystemExit("E04b ontology real-data change count mismatch")

    measurement = result.get("measurement_audit") or {}
    if measurement.get("selected_auxiliary_count") != 0:
        raise SystemExit("E04b unexpectedly selected an auxiliary view")
    if measurement.get("strict_gate_reconstructed") is not False or measurement.get("challenge_D7_observed") is not False or measurement.get("marker_not_reported_is_negative") is not False:
        raise SystemExit("E04b measurement boundary mismatch")

    transfer = result.get("e04b_transfer_audit") or {}
    pre = transfer.get("real_preconditions") or {}
    if pre.get("all_pass") is not True:
        raise SystemExit("E04b predeclared real-data conditions failed")
    if transfer.get("baseline_compatible_2024UGA_subjects") != 33 or transfer.get("baseline_compatible_2025LJI_subjects") != 40:
        raise SystemExit("E04b baseline bridge count mismatch")
    if transfer.get("supervised_strict_source_subjects") != [23, 23]:
        raise SystemExit("E04b supervised strict source count mismatch")
    if transfer.get("challenge_correction_rows") != [40, 40] or transfer.get("fit_states") != ["evaluated", "evaluated"]:
        raise SystemExit("E04b Challenge correction applicability mismatch")
    if transfer.get("measurement_bridge_recovered") is not True:
        raise SystemExit("E04b measurement bridge was not recovered")

    local = transfer.get("strict_local_reproduction") or {}
    if local.get("all_pass") is not True or local.get("tolerance") != 0.000002:
        raise SystemExit("E04b strict local reproduction failed")
    if not finite(local.get("anchor_value")) or abs(float(local["anchor_value"]) - 0.38158872734833993) > 2e-6:
        raise SystemExit("E04b strict anchor changed")
    candidates = local.get("candidate_values") or []
    deltas = local.get("delta_values") or []
    if len(candidates) != 2 or len(deltas) != 2:
        raise SystemExit("E04b strict local condition count mismatch")
    if any(not finite(value) or abs(float(value) - 0.3713227226429786) > 2e-6 for value in candidates):
        raise SystemExit("E04b strict candidate changed")
    if any(not finite(value) or abs(float(value) + 0.01026600470536132) > 2e-6 for value in deltas):
        raise SystemExit("E04b strict delta changed")

    decision = result.get("e04b_decision") or {}
    if decision.get("measurement_bridge_recovered") is not True or decision.get("strict_local_gate_passed") is not False:
        raise SystemExit("E04b bridge/local decision inconsistency")
    if decision.get("competition_candidate") is not False or decision.get("public_probe_authorized") is not False or decision.get("decision") != "no_promotion":
        raise SystemExit("E04b promotion boundary mismatch")
    if decision.get("reason") != "bridge_recovered_but_strict_local_residual_not_supported":
        raise SystemExit("E04b no-promotion reason mismatch")

    challenge_summary = {}
    for name in ("proxy_weight_0.25", "proxy_weight_1"):
        item = (result.get("challenge") or {}).get(name) or {}
        fit = item.get("fit") or {}
        if item.get("strict_bridge_source_subjects") != 23 or fit.get("state") != "evaluated" or fit.get("correction_applied") != 40:
            raise SystemExit(f"E04b challenge fit mismatch:{name}")
        movement = item.get("vs_anchor") or {}
        rank = movement.get("rank_spearman") or {}
        if rank.get("status") != "ok" or not finite(rank.get("value")):
            raise SystemExit(f"E04b Challenge movement undefined:{name}")
        challenge_summary[name] = {
            "strict_bridge_source_subjects": 23,
            "fit_state": fit.get("state"),
            "correction_applied": fit.get("correction_applied"),
            "active_features": fit.get("active_features"),
            "max_absolute_score_correction": fit.get("max_absolute_score_correction"),
            "mean_absolute_score_correction": fit.get("mean_absolute_score_correction"),
            "rank_spearman_vs_anchor": rank.get("value"),
            "changed_rank_count_vs_anchor": movement.get("changed_rank_count"),
            "mean_absolute_percentile_shift_vs_anchor": movement.get("mean_absolute_percentile_shift"),
            "max_absolute_percentile_shift_vs_anchor": movement.get("max_absolute_percentile_shift"),
            "large_movement_review": movement.get("large_movement_review"),
            "vs_b21": item.get("vs_b21"),
            "vs_historical_rank": item.get("vs_historical_rank"),
        }

    aggregate = {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "science_blobs": {
            "e04b": E04B_BLOB,
            "e04b_synthetic": E04B_SYNTH_BLOB,
            "e04": E04_BLOB,
            "contract": CONTRACT_BLOB,
            "e04_synthetic": E04_SYNTHETIC_BLOB,
            "task13": TASK13_BLOB,
            "config": CONFIG_BLOB,
        },
        "output_hashes": {
            "metrics": bridge.get("metrics_sha256"),
            "summary": bridge.get("summary_sha256"),
        },
        "md5_verified_count": bridge.get("md5_verified_count"),
        "status": result.get("status"),
        "ontology_change": ontology,
        "frozen_control_reproduction": reproduction,
        "strict_reference": result.get("strict_reference"),
        "strict_anchor": result.get("strict_anchor"),
        "strict_local_reproduction": local,
        "transfer_audit": transfer,
        "decision": decision,
        "challenge": challenge_summary,
        "proxy_weight_identifiability": result.get("proxy_weight_identifiability"),
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
    }
    print("CMI_FLU_E04B_AGGREGATE_SUMMARY " + json.dumps(aggregate, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    print(f"CMI_FLU_E04B_RESULT PASS aggregate_only=true submission=false public_probe=false science_commit={SCIENCE_COMMIT} e04b_blob={E04B_BLOB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
