#!/usr/bin/env python3
"""Validate and print the aggregate-only E04 result needed by the science repo."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260908-cmi-flu-strategy-e04-task13-rescue-001"
SCIENCE_COMMIT = "6e4d786cddc7b2d02dc4671d1c005db551d19ad6"
E04_BLOB = "73c112a8bb3b1ee7a6dd5bfbd5645a945bea6ebb"
CONTRACT_BLOB = "3982541febfb4641fbf895438ae1a465bdbb3d5e"
SYNTHETIC_BLOB = "436b6a971622915cc5b335060f9a850feb6108fd"
TASK13_BLOB = "5c6725dc757a5ba9dd21289b1c4f09997e1afdb8"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
BANNED = ('"participant_id"', '"subject_group"', '"row_index"', '"oof_predictions"', '"challenge_predictions"')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_or_none(value) -> bool:
    if value is None:
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def metric_summary(item: dict) -> dict:
    metrics = item.get("metrics") or {}
    pa = item.get("paired_vs_anchor") or {}
    pf = item.get("paired_vs_frozen_reference") or {}
    shuffle = item.get("label_shuffle") or {}
    shuffle_metrics = shuffle.get("metrics") or {}
    shuffle_anchor = shuffle.get("paired_vs_anchor") or {}
    return {
        "study_equal_spearman": metrics.get("study_equal_spearman"),
        "finite_study_mean_diagnostic": metrics.get("finite_study_mean_diagnostic"),
        "undefined_study_count": metrics.get("undefined_study_count"),
        "study_scores": metrics.get("study_scores"),
        "pooled_spearman": metrics.get("pooled_spearman"),
        "pooled_within_study_rank_spearman": metrics.get("pooled_within_study_rank_spearman"),
        "rank_rmse": metrics.get("rank_rmse"),
        "delta_vs_anchor": pa.get("paired_study_mean_delta"),
        "delta_vs_frozen_reference": pf.get("paired_study_mean_delta"),
        "large_studies_requiring_review_vs_anchor": pa.get("large_studies_requiring_review"),
        "large_studies_requiring_review_vs_frozen": pf.get("large_studies_requiring_review"),
        "shuffle_study_equal_spearman": shuffle_metrics.get("study_equal_spearman"),
        "shuffle_delta_vs_anchor": shuffle_anchor.get("paired_study_mean_delta"),
        "fit_diagnostics": item.get("fit_diagnostics"),
        "availability_adjusted_diagnostic": item.get("availability_adjusted_diagnostic"),
        "subset_28_sensitivity_vs_anchor": item.get("subset_28_sensitivity_vs_anchor"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.input_dir.resolve()
    files = sorted(x.name for x in root.iterdir() if x.is_file())
    if files != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit(f"E04 safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E04 privacy contract failed:{name}")
    bridge = json.loads(texts["bridge-result.json"])
    result = json.loads(texts["metrics.json"])
    expected = {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e04_blob_sha": E04_BLOB,
        "strategy_e04_contract_blob_sha": CONTRACT_BLOB,
        "strategy_e04_synthetic_blob_sha": SYNTHETIC_BLOB,
        "task13_harmonization_blob_sha": TASK13_BLOB,
        "config_blob_sha": CONFIG_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
        "synthetic": False,
    }
    for key, value in expected.items():
        if bridge.get(key) != value:
            raise SystemExit(f"E04 bridge contract mismatch:{key}:{bridge.get(key)!r}")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json") or bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E04 output hash mismatch")
    if result.get("schema_version") != 1 or result.get("experiment") != "strategy_v2_e04_task13_anchor_preserving_rescue" or result.get("task") != "Task1.3":
        raise SystemExit("E04 experiment identity mismatch")
    if result.get("competition_submission_attempted") is not False or result.get("leaderboard_used_for_selection") is not False or result.get("automatic_compute_retries") != 0:
        raise SystemExit("E04 selection/submission contract mismatch")
    frozen = result.get("frozen_conditions") or {}
    expected_frozen = {"strict_b21": "pls_1", "anchor": "flow_rank__Antibody-secreting_cells_(ASC)", "historical_rank": "enet_a0.1_l0.5", "residual_alpha": 10.0, "residual_shrinkage": 0.25, "residual_score_cap": 0.05, "proxy_weights": [0.25, 1.0]}
    if frozen != expected_frozen:
        raise SystemExit("E04 frozen condition mismatch")
    audit = result.get("measurement_audit") or {}
    if not audit or audit.get("challenge_D7_observed") is not False or audit.get("strict_gate_reconstructed") is not False or audit.get("marker_not_reported_is_negative") is not False:
        raise SystemExit("E04 measurement audit boundary mismatch")
    conditions = {}
    for group in ("strict_cv", "historical_loso", "strict_cv_plus_proxy"):
        payload = result.get(group) or {}
        if set(payload) != {"proxy_weight_0.25", "proxy_weight_1"}:
            raise SystemExit(f"E04 condition set mismatch:{group}:{sorted(payload)}")
        conditions[group] = {}
        for name, item in payload.items():
            summary = metric_summary(item)
            for field in ("study_equal_spearman", "delta_vs_anchor", "delta_vs_frozen_reference", "shuffle_study_equal_spearman", "shuffle_delta_vs_anchor"):
                if not finite_or_none(summary.get(field)):
                    raise SystemExit(f"E04 nonfinite aggregate:{group}:{name}:{field}")
            conditions[group][name] = summary
    challenge = {}
    for name in ("proxy_weight_0.25", "proxy_weight_1"):
        item = (result.get("challenge") or {}).get(name) or {}
        decision = item.get("decision")
        if decision not in {"candidate_requires_review", "no_promotion", "data_limited"}:
            raise SystemExit(f"E04 challenge decision invalid:{name}:{decision}")
        challenge[name] = {
            "decision": decision,
            "local_candidate_heuristic": item.get("local_candidate_heuristic"),
            "strict_bridge_source_subjects": item.get("strict_bridge_source_subjects"),
            "fit": item.get("fit"),
            "vs_anchor": item.get("vs_anchor"),
            "vs_b21": item.get("vs_b21"),
            "vs_historical_rank": item.get("vs_historical_rank"),
        }
    ident = result.get("proxy_weight_identifiability") or {}
    if ident.get("weight_selected") is not None or ident.get("one_source_study_cannot_identify_relative_proxy_weight") is not True:
        raise SystemExit("E04 proxy-weight identifiability contract mismatch")
    aggregate = {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "science_blobs": {"e04": E04_BLOB, "contract": CONTRACT_BLOB, "synthetic": SYNTHETIC_BLOB, "task13": TASK13_BLOB, "config": CONFIG_BLOB},
        "status": result.get("status"),
        "incumbent_changed": result.get("incumbent_changed"),
        "frozen_control_reproduction": result.get("frozen_control_reproduction"),
        "measurement_audit": audit,
        "controls": {
            "strict_reference": result.get("strict_reference"),
            "strict_anchor": result.get("strict_anchor"),
            "historical_rank_research_control": result.get("historical_rank_research_control"),
            "historical_target_only_diagnostic": result.get("historical_target_only_diagnostic"),
            "strict_anchor_vs_b21": result.get("strict_anchor_vs_b21"),
            "availability_only_legacy": result.get("availability_only_legacy"),
            "availability_only_audited": result.get("availability_only_audited"),
            "gate_eligibility_only": result.get("gate_eligibility_only"),
        },
        "conditions": conditions,
        "challenge": challenge,
        "challenge_frozen_controls": {
            "frozen_historical_rank_vs_b21": (result.get("challenge") or {}).get("frozen_historical_rank_vs_b21"),
            "frozen_historical_rank_vs_anchor": (result.get("challenge") or {}).get("frozen_historical_rank_vs_anchor"),
            "anchor_vs_b21": (result.get("challenge") or {}).get("anchor_vs_b21"),
        },
        "proxy_weight_identifiability": ident,
        "interpretation_limits": result.get("interpretation_limits"),
        "reentry_conditions": result.get("reentry_conditions"),
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
    }
    print("CMI_FLU_E04_AGGREGATE_SUMMARY " + json.dumps(aggregate, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    frozen_pass = (result.get("frozen_control_reproduction") or {}).get("all_pass") is True
    if result.get("status") != "complete" or not frozen_pass:
        raise SystemExit(f"E04 frozen reproduction/status requires review:{result.get('status')}")
    print(f"CMI_FLU_E04_RESULT PASS aggregate_only=true submission=false science_commit={SCIENCE_COMMIT} e04_blob={E04_BLOB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
