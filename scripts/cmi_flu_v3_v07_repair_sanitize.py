#!/usr/bin/env python3
"""Validate repaired V3-07 private outputs and emit aggregate-safe diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

SCIENCE_COMMIT = "bfaef18b9f703be4d88b8a170794ace2c47a0e55"
SCIENCE_V1_BLOB = "4cf804e0c10889d0e037457ac096711b77200ea9"
SCIENCE_REPAIR_BLOB = "f547b894235542981a4ad48d523fe53393c52e34"
REPAIR_VERSION = "strategy_v3_v07_v2_execution_contract_20260913"
CONTRACTS = {
    "task22_panel_mean": {
        "prefix": "v3_v07_h1",
        "request_id": "20260913-cmi-flu-strategy-v3-v07-h1-panel-mean-002",
        "candidate": "task22_panel_mean",
        "reference": "e05_main_effects",
    },
    "task23_retention": {
        "prefix": "v3_v07_h2",
        "request_id": "20260913-cmi-flu-strategy-v3-v07-h2-retention-002",
        "candidate": "task23_retention",
        "reference": "independent_d365",
    },
}
BANNED_AGGREGATE_KEYS = {"participant_id", "subject_group", "row_index", "oof_bank", "challenge_bank"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> bool:
    try: return math.isfinite(float(value))
    except Exception: return False


def assert_no_banned_keys(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in BANNED_AGGREGATE_KEYS:
                raise SystemExit(f"V3-07 repair aggregate privacy key:{key}")
            assert_no_banned_keys(item)
    elif isinstance(value, list):
        for item in value: assert_no_banned_keys(item)


def metric_delta(block: dict, candidate: str, reference: str) -> dict:
    pooled = block["pooled"]; equal = block["equal_study"]
    c = pooled[candidate]; r = pooled[reference]; ce = equal[candidate]; re = equal[reference]
    return {
        "pooled_rmse_candidate": c["rmse"], "pooled_rmse_reference": r["rmse"],
        "pooled_rmse_relative_reduction": 1.0 - float(c["rmse"]) / float(r["rmse"]),
        "pooled_spearman_candidate": c["spearman"], "pooled_spearman_reference": r["spearman"],
        "pooled_spearman_delta": None if c["spearman"] is None or r["spearman"] is None else float(c["spearman"]) - float(r["spearman"]),
        "equal_study_rmse_candidate": ce["rmse_mean"], "equal_study_rmse_reference": re["rmse_mean"],
        "equal_study_rmse_relative_reduction": 1.0 - float(ce["rmse_mean"]) / float(re["rmse_mean"]),
        "median_study_rmse_candidate": ce["rmse_median"], "median_study_rmse_reference": re["rmse_median"],
        "worst_study_rmse_candidate": ce["rmse_worst"], "worst_study_rmse_reference": re["rmse_worst"],
        "equal_study_spearman_candidate": ce["spearman_mean"], "equal_study_spearman_reference": re["spearman_mean"],
        "equal_study_spearman_delta": None if ce["spearman_mean"] is None or re["spearman_mean"] is None else float(ce["spearman_mean"]) - float(re["spearman_mean"]),
    }


def correction_sd_range(path: Path) -> tuple[float | None, float | None]:
    values = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if "correction_sd" not in (reader.fieldnames or []):
            raise SystemExit("V3-07 repair H1 correction_sd column missing")
        for row in reader:
            raw = row.get("correction_sd")
            if raw not in (None, ""):
                value = float(raw)
                if not math.isfinite(value): raise SystemExit("V3-07 repair nonfinite correction_sd")
                values.append(value)
    return (min(values), max(values)) if values else (None, None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=sorted(CONTRACTS), required=True)
    p.add_argument("--input-dir", type=Path, required=True)
    args = p.parse_args(); c = CONTRACTS[args.condition]
    root = args.input_dir.resolve(); prefix = c["prefix"]
    expected = sorted([f"{prefix}_oof_bank.csv", f"{prefix}_challenge_bank.csv", f"{prefix}_summary.json", f"{prefix}_manifest.json"])
    found = sorted(path.name for path in root.iterdir() if path.is_file())
    if found != expected: raise SystemExit(f"V3-07 repair safe-output file set mismatch:{found}")

    manifest = json.loads((root / f"{prefix}_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment") != "strategy_v3_v07_hai_panel_mean_and_retention" or manifest.get("condition") != args.condition or manifest.get("row_level_banks_private") is not True:
        raise SystemExit("V3-07 repair manifest contract mismatch")
    listed = {item["filename"]: item for item in manifest.get("files", [])}
    for name in (f"{prefix}_oof_bank.csv", f"{prefix}_challenge_bank.csv", f"{prefix}_summary.json"):
        if name not in listed: raise SystemExit(f"V3-07 repair manifest missing:{name}")
        path = root / name; item = listed[name]
        if path.stat().st_size != int(item["bytes"]) or sha(path) != item["sha256"]:
            raise SystemExit(f"V3-07 repair manifest hash mismatch:{name}")

    summary = json.loads((root / f"{prefix}_summary.json").read_text(encoding="utf-8"))
    assert_no_banned_keys(summary)
    if summary.get("experiment") != "strategy_v3_v07_hai_panel_mean_and_retention" or summary.get("new_candidate_conditions") != [args.condition] or summary.get("execution_repair_version") != REPAIR_VERSION:
        raise SystemExit("V3-07 repair summary identity mismatch")
    for key in ("competition_submission_attempted", "final_submission_selection_attempted", "public_leaderboard_used", "saved_outer_oof_reused_as_inner_teacher", "automatic_parameter_sweep"):
        if summary.get(key) is not False: raise SystemExit(f"V3-07 repair forbidden boundary:{key}")
    runtime = summary.get("runtime") or {}
    if runtime.get("request_id") != c["request_id"] or runtime.get("science_commit") != SCIENCE_COMMIT or runtime.get("science_blob") != SCIENCE_V1_BLOB or runtime.get("science_repair_blob") != SCIENCE_REPAIR_BLOB or runtime.get("condition") != args.condition or runtime.get("runtime_terminal_marker") != "CMI_FLU_V307_RUNTIME_PASS":
        raise SystemExit("V3-07 repair runtime provenance mismatch")

    payload = summary.get("result") or {}; contract = payload.get("contract") or {}; proxy = payload.get("target_proxy") or {}; repair = payload.get("execution_repair") or {}
    fit = int(summary.get("fit_count", -1)); planned = int(repair.get("planned_fit_count", -2))
    if payload.get("condition") != args.condition or int(payload.get("fit_count", -3)) != fit or planned != fit or repair.get("version") != REPAIR_VERSION:
        raise SystemExit("V3-07 repair fit-plan identity mismatch")
    if args.condition == "task22_panel_mean":
        if not (256 < fit <= 384) or repair.get("fit_limit_policy") != "exact_outcome_independent_planned_count": raise SystemExit("V3-07 repair H1 fit contract")
        if proxy != {"panel_size":9,"subjects":238,"studies":["2023_UGA","2024UGA"]}: raise SystemExit(f"V3-07 repair H1 proxy mismatch:{proxy}")
        if float(contract.get("alpha", -1)) != 10.0 or contract.get("features") != ["log2_pre_panel_gm","pre_panel_log2_sd"] or contract.get("inner_oof_only") is not True or contract.get("positivity") is not True: raise SystemExit("V3-07 repair H1 science contract")
        if int(repair.get("learning_rows", -1)) != 62285 or int(repair.get("learning_studies", -1)) != 48: raise SystemExit("V3-07 repair H1 learning support")
    else:
        if not (0 < fit <= 256) or repair.get("fit_limit_policy") != "exact_outcome_independent_planned_count_within_frozen_256_bound": raise SystemExit("V3-07 repair H2 fit contract")
        teacher = payload.get("paired_teacher") or {}
        expected_teacher = {"rows":18593,"unique_subjects":567,"target":"log2(Y365/Y28)","participant_year_strain_rows":18593,"participant_years":1067,"studies":11,"biological_subject_strain_pairs":11913,"biological_subjects":567,"row_unit":"participant_year_strain","biological_pair_unit":"subject_strain_deduplicated_across_participant_years"}
        if any(teacher.get(k) != v for k,v in expected_teacher.items()): raise SystemExit(f"V3-07 repair H2 teacher mismatch:{teacher}")
        if proxy != {"panel_size":8,"subjects":439,"studies":3}: raise SystemExit(f"V3-07 repair H2 proxy mismatch:{proxy}")
        if float(contract.get("alpha", -1)) != 100.0 or contract.get("features") != ["baseline HAI","age","predicted D28"] or contract.get("inner_oof_d28_only") is not True or contract.get("held_observed_d28_used") is not False or contract.get("challenge_observed_d28_used") is not False or contract.get("teacher_row_unit") != "participant_year_strain" or contract.get("biological_support_unit") != "subject_strain_deduplicated_across_participant_years": raise SystemExit("V3-07 repair H2 science contract")

    metrics = payload.get("metrics") or {}; proxy_metrics = payload.get("target_proxy_metrics") or {}
    for block in (metrics, proxy_metrics):
        for model in (c["candidate"], c["reference"]):
            for metric in (block.get("pooled") or {}).get(model, {}).values():
                if isinstance(metric, (int, float)) and metric is not None and not finite(metric): raise SystemExit("V3-07 repair nonfinite aggregate metric")
    diagnostic = {
        "condition": args.condition, "request_id": c["request_id"], "science_commit": SCIENCE_COMMIT,
        "science_v1_blob": SCIENCE_V1_BLOB, "science_repair_blob": SCIENCE_REPAIR_BLOB, "repair_version": REPAIR_VERSION,
        "fit_count": fit, "fit_plan": repair,
        "target_proxy": proxy,
        "overall_vs_primary_control": metric_delta(metrics, c["candidate"], c["reference"]),
        "target_proxy_vs_primary_control": metric_delta(proxy_metrics, c["candidate"], c["reference"]),
        "by_study": metrics.get("by_study"), "panel_strata": metrics.get("panel_strata"),
        "target_proxy_by_study": proxy_metrics.get("by_study"), "target_proxy_panel_strata": proxy_metrics.get("panel_strata"),
        "paired": payload.get("paired_vs_e05") if args.condition == "task22_panel_mean" else payload.get("paired_vs_independent"),
        "paired_target_proxy": payload.get("paired_target_proxy_vs_e05") if args.condition == "task22_panel_mean" else payload.get("paired_target_proxy_vs_independent"),
        "challenge_rank_tie_shift": payload.get("challenge_rank_tie_shift_vs_e05") if args.condition == "task22_panel_mean" else payload.get("challenge_rank_tie_shift_vs_independent"),
        "challenge_rank_tie_shift_secondary": None if args.condition == "task22_panel_mean" else payload.get("challenge_rank_tie_shift_vs_e06b"),
        "controls": sorted((metrics.get("pooled") or {}).keys()),
        "competition_submission_attempted": False, "final_submission_selection_attempted": False, "public_leaderboard_used": False,
    }
    if args.condition == "task22_panel_mean":
        lo, hi = correction_sd_range(root / f"{prefix}_oof_bank.csv")
        diagnostic["correction_sd_min"] = lo; diagnostic["correction_sd_max"] = hi
        diagnostic["constant_correction_degenerate"] = bool(lo is None or hi is None or hi <= 1e-12)
    else:
        diagnostic["paired_teacher"] = payload.get("paired_teacher")
    print("CMI_FLU_V307_REPAIR_AGGREGATE " + json.dumps(diagnostic, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
