#!/usr/bin/env python3
"""Sanitize and validate aggregate-only CMI-Flu E12b prediction-freeze outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

REQUEST_ID = "20260911-cmi-flu-strategy-e12b-challenge-freeze-001"
TARGET = "renta0426/cmi-flu-e12b-challenge-freeze-20260911-001"
SCIENCE_COMMIT = "a411bf85a4a79fec2e9a2b9c2bc8cbe186adee5f"
E12B_BLOB = "af5df92ca81abc34a7f7046dba5bc284c98302d4"
E12B_CONTRACT_BLOB = "7e844b2d56bd6739793888bf6306946a0028d966"
E12A_V2_BLOB = "0c4c970c8bacfed61bfbb9587e0a0bfdec7903d9"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
HEADING = "# CMI-Flu Strategy-v2 E12b Challenge prediction freeze"
TASKS = ("Task1.1", "Task1.2", "Task1.3", "Task1.4", "Task2.1", "Task2.2", "Task2.3")
EXPECTED_PORTFOLIO = {
    "Task1.1": "b21_pls_2",
    "Task1.2": "task12_anchor_residual_et_d5_l5_sqrt_lambda0.5",
    "Task1.3": "strict_asc_anchor",
    "Task1.4": "raw_pre_vacc_conserved_anchor",
    "Task2.1": "b21_et_subtype_d3_l5",
    "Task2.2": "b21_et_subtype_d5_l10",
    "Task2.3": "b21_ridge_exact_a100",
}
BANNED_KEYS = {
    "participant_id", "subject_group", "row_index", "challenge_predictions",
    "submission_rows", "prediction_vector", "oof_predictions",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_no_row_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in BANNED_KEYS:
                raise SystemExit(f"E12b row-level key persisted:{key}")
            assert_no_row_keys(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_row_keys(item)


def require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise SystemExit(f"E12b invalid SHA-256:{label}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.expanduser().resolve()
    expected = {"bridge-result.json", "metrics.json", "summary.md"}
    files = {path.name for path in root.iterdir() if path.is_file()}
    if files != expected:
        raise SystemExit(f"E12b output file set mismatch:{sorted(files)}")
    if any(path.is_symlink() for path in root.iterdir()):
        raise SystemExit("E12b symlink output forbidden")

    bridge = json.loads((root / "bridge-result.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    summary = (root / "summary.md").read_text(encoding="utf-8")
    assert_no_row_keys(bridge)
    assert_no_row_keys(metrics)

    if bridge.get("request_id") != REQUEST_ID or bridge.get("target_kernel") != TARGET:
        raise SystemExit("E12b bridge identity mismatch")
    if bridge.get("science_commit") != SCIENCE_COMMIT:
        raise SystemExit("E12b bridge science commit mismatch")
    if bridge.get("strategy_e12b_blob_sha") != E12B_BLOB:
        raise SystemExit("E12b bridge science blob mismatch")
    if bridge.get("strategy_e12b_contract_blob_sha") != E12B_CONTRACT_BLOB:
        raise SystemExit("E12b bridge contract blob mismatch")
    if bridge.get("strategy_e12a_v2_blob_sha") != E12A_V2_BLOB:
        raise SystemExit("E12b E12a-v2 parent blob mismatch")
    if bridge.get("config_blob_sha") != CONFIG_BLOB:
        raise SystemExit("E12b config blob mismatch")
    if bridge.get("metrics_sha256") != sha256(root / "metrics.json"):
        raise SystemExit("E12b metrics hash link mismatch")
    if bridge.get("summary_sha256") != sha256(root / "summary.md"):
        raise SystemExit("E12b summary hash link mismatch")
    if bridge.get("frozen_portfolio") != EXPECTED_PORTFOLIO:
        raise SystemExit("E12b bridge portfolio mismatch")
    if bridge.get("next_step") != "ready_for_separate_submission_authorization":
        raise SystemExit("E12b bridge next-step mismatch")
    for key in (
        "row_level_candidate_persisted", "competition_submission_attempted",
        "leaderboard_used_for_selection", "contains_participant_identifiers",
        "contains_row_level_predictions",
    ):
        if bridge.get(key) is not False:
            raise SystemExit(f"E12b bridge boundary mismatch:{key}")

    if metrics.get("experiment") != "strategy_v2_e12b_challenge_prediction_freeze":
        raise SystemExit("E12b metrics identity mismatch")
    if metrics.get("final_portfolio") != EXPECTED_PORTFOLIO or metrics.get("portfolio_task_count") != 7:
        raise SystemExit("E12b metrics portfolio mismatch")
    if metrics.get("challenge_rows") != 40:
        raise SystemExit("E12b Challenge row-count mismatch")
    for key in (
        "new_model_selection_performed", "leaderboard_used_for_selection",
        "public_probe_performed", "competition_submission_attempted",
        "competition_submission_authorized", "contains_participant_identifiers",
        "contains_row_level_predictions", "row_level_candidate_persisted",
    ):
        if metrics.get(key) is not False:
            raise SystemExit(f"E12b metrics boundary mismatch:{key}")
    if metrics.get("next_step") != "ready_for_separate_submission_authorization":
        raise SystemExit("E12b metrics next-step mismatch")

    validation = metrics.get("submission_validation") or {}
    if validation.get("rows") != 40:
        raise SystemExit("E12b submission validation row mismatch")
    if validation.get("columns") != ["participant_id", *TASKS]:
        raise SystemExit("E12b submission validation column mismatch")
    if validation.get("minus99_tasks") != []:
        raise SystemExit("E12b submission retains -99")
    unique_counts = validation.get("task_unique_counts") or {}
    if set(unique_counts) != set(TASKS):
        raise SystemExit("E12b submission task set mismatch")
    for task in TASKS[:-1]:
        if int(unique_counts.get(task, 0)) < 2:
            raise SystemExit(f"E12b Public-scored task constant:{task}")

    fp = metrics.get("fingerprint_contract") or {}
    if fp.get("semantic_hash_version") != "cmi-flu-e12b-semantic-v1":
        raise SystemExit("E12b semantic hash version mismatch")
    if fp.get("canonical_csv_float_format") != "%.17g" or fp.get("canonical_csv_lineterminator") != "LF":
        raise SystemExit("E12b canonical CSV contract mismatch")
    semantic = require_hash(fp.get("semantic_submission_sha256"), "semantic")
    canonical = require_hash(fp.get("canonical_csv_sha256"), "canonical_csv")
    if bridge.get("semantic_submission_sha256") != semantic or bridge.get("canonical_csv_sha256") != canonical:
        raise SystemExit("E12b bridge/fingerprint mismatch")
    if int(fp.get("canonical_csv_bytes", 0)) <= 0:
        raise SystemExit("E12b canonical CSV size invalid")

    task_summaries = metrics.get("task_prediction_summaries") or {}
    if set(task_summaries) != set(TASKS):
        raise SystemExit("E12b per-task summary set mismatch")
    for task in TASKS:
        row = task_summaries[task]
        if row.get("task") != task or int(row.get("rows", -1)) != 40:
            raise SystemExit(f"E12b per-task identity mismatch:{task}")
        minimum = float(row.get("prediction_min"))
        maximum = float(row.get("prediction_max"))
        unique = int(row.get("prediction_unique", 0))
        tie = float(row.get("tie_fraction"))
        if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum < minimum:
            raise SystemExit(f"E12b prediction range invalid:{task}")
        if unique < 1 or unique > 40 or abs(tie - (1.0 - unique / 40.0)) > 1e-12:
            raise SystemExit(f"E12b prediction uniqueness invalid:{task}")
        require_hash(row.get("prediction_sha256"), f"task:{task}")
    if int(task_summaries["Task1.3"].get("prediction_unique", 0)) != 36:
        raise SystemExit("E12b Task1.3 unique-count mismatch")

    controls = metrics.get("structural_controls") or {}
    if controls.get("final_changed_tasks_vs_regenerated_b21") != ["Task1.2", "Task1.3"]:
        raise SystemExit("E12b changed-task set vs B2.1 mismatch")
    if controls.get("final_changed_tasks_vs_regenerated_task12_only") != ["Task1.3"]:
        raise SystemExit("E12b changed-task set vs Task1.2-only mismatch")
    if controls.get("historical_public_0_218_is_not_assumed_byte_identical") is not True:
        raise SystemExit("E12b historical Public boundary mismatch")
    require_hash(controls.get("regenerated_b21_semantic_sha256"), "b21_control")
    require_hash(controls.get("regenerated_task12_only_semantic_sha256"), "task12_control")
    for key, expected_task in (
        ("task12_final_vs_b21", "Task1.2"),
        ("task13_final_vs_b21", "Task1.3"),
        ("task13_final_vs_task12_only", "Task1.3"),
    ):
        comp = controls.get(key) or {}
        if comp.get("task") != expected_task:
            raise SystemExit(f"E12b rank-comparison identity mismatch:{key}")
        if int(comp.get("changed_rank_count", -1)) < 0 or int(comp.get("changed_rank_count", -1)) > 40:
            raise SystemExit(f"E12b rank-comparison count invalid:{key}")
        for scalar in ("mean_absolute_percentile_shift", "maximum_absolute_percentile_shift"):
            value = float(comp.get(scalar, -1))
            if not math.isfinite(value) or value < 0:
                raise SystemExit(f"E12b rank-comparison scalar invalid:{key}:{scalar}")

    task13 = metrics.get("task13_reconciliation") or {}
    if task13 != {
        "final_incumbent": "strict_asc_anchor",
        "predictor_feature": "flow_rank__Antibody-secreting_cells_(ASC)",
        "challenge_unique_values": 36,
        "e12a_v2_reproduced": True,
    }:
        raise SystemExit("E12b Task1.3 reconciliation mismatch")

    if not summary.startswith(HEADING + "\n") or "None" in summary:
        raise SystemExit("E12b summary identity/presentation mismatch")
    for token in (
        f"semantic submission SHA-256: `{semantic}`",
        f"canonical CSV SHA-256: `{canonical}`",
        "changed vs regenerated B2.1: `Task1.2,Task1.3`",
        "changed vs regenerated Task1.2-only: `Task1.3`",
        "- Task1.3: `strict_asc_anchor`",
        "row-level candidate was not persisted",
    ):
        if token not in summary:
            raise SystemExit("E12b summary/metrics mismatch")

    raw = "\n".join((root / name).read_text(errors="replace") for name in sorted(expected))
    for banned in ("KGAT_", "KAGGLE_API_TOKEN", "SUB_", "ROW_"):
        if banned in raw:
            raise SystemExit("E12b sensitive/row-level token in persistent output")

    print(
        "CMI_FLU_E12B_SANITIZE PASS "
        f"aggregate_only=true challenge_rows=40 semantic_sha256={semantic} "
        "row_persisted=false submission=false next_step=ready_for_separate_submission_authorization"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
