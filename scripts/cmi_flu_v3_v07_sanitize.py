#!/usr/bin/env python3
"""Validate V3-07 private outputs and emit aggregate-safe scientific diagnostics only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

SCIENCE_COMMIT = "9bfe05664e4d96f09631a3210f7951bf9dfefe71"
SCIENCE_BLOB = "4cf804e0c10889d0e037457ac096711b77200ea9"
CONTRACTS = {
    "task22_panel_mean": {"prefix":"v3_v07_h1", "request_id":"20260912-cmi-flu-strategy-v3-v07-h1-panel-mean-001", "candidate":"task22_panel_mean", "reference":"e05_main_effects"},
    "task23_retention": {"prefix":"v3_v07_h2", "request_id":"20260912-cmi-flu-strategy-v3-v07-h2-retention-001", "candidate":"task23_retention", "reference":"independent_d365"},
}
BANNED_AGGREGATE_KEYS = {"participant_id", "subject_group", "row_index", "oof_bank", "challenge_bank"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def assert_no_banned_keys(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in BANNED_AGGREGATE_KEYS:
                raise SystemExit(f"V3-07 aggregate privacy key:{key}")
            assert_no_banned_keys(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_banned_keys(item)


def metric_delta(block: dict, candidate: str, reference: str) -> dict:
    pooled = block["pooled"]
    equal = block["equal_study"]
    c = pooled[candidate]; r = pooled[reference]
    ce = equal[candidate]; re = equal[reference]
    return {
        "pooled_rmse_candidate": c["rmse"],
        "pooled_rmse_reference": r["rmse"],
        "pooled_rmse_relative_reduction": 1.0 - float(c["rmse"]) / float(r["rmse"]),
        "pooled_spearman_candidate": c["spearman"],
        "pooled_spearman_reference": r["spearman"],
        "pooled_spearman_delta": None if c["spearman"] is None or r["spearman"] is None else float(c["spearman"]) - float(r["spearman"]),
        "equal_study_rmse_candidate": ce["rmse_mean"],
        "equal_study_rmse_reference": re["rmse_mean"],
        "equal_study_rmse_relative_reduction": 1.0 - float(ce["rmse_mean"]) / float(re["rmse_mean"]),
        "median_study_rmse_candidate": ce["rmse_median"],
        "median_study_rmse_reference": re["rmse_median"],
        "worst_study_rmse_candidate": ce["rmse_worst"],
        "worst_study_rmse_reference": re["rmse_worst"],
        "equal_study_spearman_candidate": ce["spearman_mean"],
        "equal_study_spearman_reference": re["spearman_mean"],
        "equal_study_spearman_delta": None if ce["spearman_mean"] is None or re["spearman_mean"] is None else float(ce["spearman_mean"]) - float(re["spearman_mean"]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=sorted(CONTRACTS), required=True)
    p.add_argument("--input-dir", type=Path, required=True)
    args = p.parse_args()
    c = CONTRACTS[args.condition]
    root = args.input_dir.resolve(); prefix = c["prefix"]
    expected = sorted([f"{prefix}_oof_bank.csv", f"{prefix}_challenge_bank.csv", f"{prefix}_summary.json", f"{prefix}_manifest.json"])
    found = sorted(path.name for path in root.iterdir() if path.is_file())
    if found != expected:
        raise SystemExit(f"V3-07 safe-output file set mismatch:{found}")

    manifest = json.loads((root / f"{prefix}_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment") != "strategy_v3_v07_hai_panel_mean_and_retention" or manifest.get("condition") != args.condition or manifest.get("row_level_banks_private") is not True:
        raise SystemExit("V3-07 manifest contract mismatch")
    listed = {item["filename"]: item for item in manifest.get("files", [])}
    for name in (f"{prefix}_oof_bank.csv", f"{prefix}_challenge_bank.csv", f"{prefix}_summary.json"):
        if name not in listed:
            raise SystemExit(f"V3-07 manifest missing:{name}")
        path = root / name; item = listed[name]
        if path.stat().st_size != int(item["bytes"]) or sha(path) != item["sha256"]:
            raise SystemExit(f"V3-07 manifest hash mismatch:{name}")

    summary = json.loads((root / f"{prefix}_summary.json").read_text(encoding="utf-8"))
    assert_no_banned_keys(summary)
    if summary.get("experiment") != "strategy_v3_v07_hai_panel_mean_and_retention" or summary.get("new_candidate_conditions") != [args.condition]:
        raise SystemExit("V3-07 summary identity mismatch")
    if int(summary.get("fit_count", 999999)) > 256:
        raise SystemExit("V3-07 fit budget exceeded")
    for key in ("competition_submission_attempted","final_submission_selection_attempted","public_leaderboard_used","saved_outer_oof_reused_as_inner_teacher","automatic_parameter_sweep"):
        if summary.get(key) is not False:
            raise SystemExit(f"V3-07 forbidden boundary:{key}")
    runtime = summary.get("runtime") or {}
    if runtime.get("request_id") != c["request_id"] or runtime.get("science_commit") != SCIENCE_COMMIT or runtime.get("science_blob") != SCIENCE_BLOB or runtime.get("condition") != args.condition:
        raise SystemExit("V3-07 runtime provenance mismatch")
    if runtime.get("runtime_terminal_marker") != "CMI_FLU_V307_RUNTIME_PASS":
        raise SystemExit("V3-07 terminal marker missing")

    payload = summary.get("result") or {}; contract = payload.get("contract") or {}; proxy = payload.get("target_proxy") or {}
    if payload.get("condition") != args.condition:
        raise SystemExit("V3-07 result condition mismatch")
    if args.condition == "task22_panel_mean":
        if proxy != {"panel_size":9,"subjects":238,"studies":["2023_UGA","2024UGA"]}:
            raise SystemExit(f"V3-07 H1 proxy support mismatch:{proxy}")
        if float(contract.get("alpha", -1)) != 10.0 or contract.get("features") != ["log2_pre_panel_gm","pre_panel_log2_sd"] or contract.get("inner_oof_only") is not True or contract.get("positivity") is not True:
            raise SystemExit("V3-07 H1 scientific contract mismatch")
    else:
        teacher = payload.get("paired_teacher") or {}
        if teacher != {"rows":11913,"target":"log2(Y365/Y28)","unique_subjects":567}:
            raise SystemExit(f"V3-07 H2 paired support mismatch:{teacher}")
        if proxy != {"panel_size":8,"subjects":439,"studies":3}:
            raise SystemExit(f"V3-07 H2 proxy support mismatch:{proxy}")
        if float(contract.get("alpha", -1)) != 100.0 or contract.get("features") != ["baseline HAI","age","predicted D28"] or contract.get("inner_oof_d28_only") is not True or contract.get("held_observed_d28_used") is not False or contract.get("challenge_observed_d28_used") is not False:
            raise SystemExit("V3-07 H2 scientific contract mismatch")

    metrics = payload.get("metrics") or {}; proxy_metrics = payload.get("target_proxy_metrics") or {}
    for block in (metrics, proxy_metrics):
        for model in (c["candidate"], c["reference"]):
            for metric in (block.get("pooled") or {}).get(model, {}).values():
                if isinstance(metric, (int,float)) and metric is not None and not finite(metric):
                    raise SystemExit("V3-07 nonfinite aggregate metric")
    diagnostic = {
        "condition": args.condition,
        "request_id": c["request_id"],
        "science_commit": SCIENCE_COMMIT,
        "science_blob": SCIENCE_BLOB,
        "fit_count": summary["fit_count"],
        "target_proxy": proxy,
        "overall_vs_primary_control": metric_delta(metrics, c["candidate"], c["reference"]),
        "target_proxy_vs_primary_control": metric_delta(proxy_metrics, c["candidate"], c["reference"]),
        "by_study": metrics.get("by_study"),
        "panel_strata": metrics.get("panel_strata"),
        "target_proxy_by_study": proxy_metrics.get("by_study"),
        "target_proxy_panel_strata": proxy_metrics.get("panel_strata"),
        "paired": payload.get("paired_vs_e05") if args.condition == "task22_panel_mean" else payload.get("paired_vs_independent"),
        "paired_target_proxy": payload.get("paired_target_proxy_vs_e05") if args.condition == "task22_panel_mean" else payload.get("paired_target_proxy_vs_independent"),
        "challenge_rank_tie_shift": payload.get("challenge_rank_tie_shift_vs_e05") if args.condition == "task22_panel_mean" else payload.get("challenge_rank_tie_shift_vs_independent"),
        "challenge_rank_tie_shift_secondary": None if args.condition == "task22_panel_mean" else payload.get("challenge_rank_tie_shift_vs_e06b"),
        "controls": sorted((metrics.get("pooled") or {}).keys()),
        "competition_submission_attempted": False,
        "final_submission_selection_attempted": False,
        "public_leaderboard_used": False,
    }
    # Constant-correction stop check is aggregate-only: consume only the correction_sd column,
    # never identifiers or predictions, and emit its range rather than row values.
    if args.condition == "task22_panel_mean":
        corr = pd.read_csv(root / f"{prefix}_oof_bank.csv", usecols=["correction_sd"])["correction_sd"].dropna().astype(float)
        diagnostic["correction_sd_min"] = float(corr.min()) if len(corr) else None
        diagnostic["correction_sd_max"] = float(corr.max()) if len(corr) else None
        diagnostic["constant_correction_degenerate"] = bool(len(corr) == 0 or float(corr.max()) <= 1e-12)
    print("CMI_FLU_V307_AGGREGATE " + json.dumps(diagnostic, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
