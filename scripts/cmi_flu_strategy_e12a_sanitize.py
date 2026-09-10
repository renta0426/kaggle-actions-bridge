#!/usr/bin/env python3
"""Validate E12a aggregate outputs and the human-readable summary contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260910-cmi-flu-strategy-e12a-final-reproduction-001"
TARGET = "renta0426/cmi-flu-e12a-final-reproduction-20260910-001"
SCIENCE_COMMIT = "f227694aee240a05de1b4e318a8c17acc2db5651"
E12A_BLOB = "84c9368d520b2b7f88f189d9d9a92a3306e54b0e"
HEADING = "# CMI-Flu Strategy-v2 E12a final-system reproduction audit"
EXPECTED_PORTFOLIO = {
    "Task1.1":"b21_pls_2",
    "Task1.2":"task12_anchor_residual_et_d5_l5_sqrt_lambda0.5",
    "Task1.3":"b21_pls_1",
    "Task1.4":"raw_pre_vacc_conserved_anchor",
    "Task2.1":"b21_et_subtype_d3_l5",
    "Task2.2":"b21_et_subtype_d5_l10",
    "Task2.3":"b21_ridge_exact_a100",
}
EXPECTED_SCORES = {
    "Task1.1":0.099707180,
    "Task1.2":0.527103,
    "Task1.3":0.106304108,
    "Task2.1":0.623449,
    "Task2.2":0.576506,
    "Task2.3":0.695731,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,required=True)
    args=p.parse_args(); root=args.input_dir.resolve()
    expected={"bridge-result.json","metrics.json","summary.md"}
    files={x.name for x in root.iterdir() if x.is_file()}
    if files!=expected: raise SystemExit(f"E12a output file set mismatch:{sorted(files)}")
    for path in root.iterdir():
        if path.is_symlink(): raise SystemExit("E12a symlink output forbidden")
    bridge=json.loads((root/"bridge-result.json").read_text())
    metrics=json.loads((root/"metrics.json").read_text())
    summary=(root/"summary.md").read_text()
    if bridge.get("request_id")!=REQUEST_ID or bridge.get("target_kernel")!=TARGET: raise SystemExit("E12a bridge identity mismatch")
    if bridge.get("science_commit")!=SCIENCE_COMMIT or bridge.get("strategy_e12a_blob_sha")!=E12A_BLOB: raise SystemExit("E12a bridge science identity mismatch")
    if bridge.get("metrics_sha256")!=sha(root/"metrics.json") or bridge.get("summary_sha256")!=sha(root/"summary.md"): raise SystemExit("E12a output hash link mismatch")
    for key in ("competition_submission_attempted","leaderboard_used_for_selection","contains_participant_identifiers","contains_row_level_predictions"):
        if bridge.get(key) is not False: raise SystemExit(f"E12a bridge boundary mismatch:{key}")
    if metrics.get("experiment")!="strategy_v2_e12a_final_system_reproduction" or metrics.get("comparison_contract")!="paired_subject_purged_v2": raise SystemExit("E12a metrics identity mismatch")
    if metrics.get("frozen_portfolio")!=EXPECTED_PORTFOLIO or metrics.get("portfolio_task_count")!=7: raise SystemExit("E12a portfolio mismatch")
    if metrics.get("e11_task11_candidates_carried")!=[]: raise SystemExit("E12a E11 carry mismatch")
    for key in ("new_model_selection_performed","incumbent_changed","public_probe_authorized","competition_submission_authorized","leaderboard_used_for_selection","competition_submission_attempted","contains_participant_identifiers","contains_row_level_predictions"):
        if metrics.get(key) is not False: raise SystemExit(f"E12a metrics boundary mismatch:{key}")
    rows=metrics.get("supervised_reproduction") or []
    if len(rows)!=6: raise SystemExit("E12a supervised reproduction count mismatch")
    by={row.get("task"):row for row in rows}
    if set(by)!=set(EXPECTED_SCORES): raise SystemExit("E12a supervised task set mismatch")
    for task,expected_score in EXPECTED_SCORES.items():
        row=by[task]
        if row.get("incumbent")!=EXPECTED_PORTFOLIO[task]: raise SystemExit("E12a incumbent identity mismatch")
        exp=float(row.get("expected_study_equal_spearman")); obs=float(row.get("observed_study_equal_spearman")); dev=float(row.get("deviation")); absolute=float(row.get("absolute_deviation"))
        if not all(math.isfinite(x) for x in (exp,obs,dev,absolute)): raise SystemExit("E12a nonfinite reproduction value")
        if abs(exp-expected_score)>1e-12 or abs((obs-exp)-dev)>1e-12 or abs(abs(dev)-absolute)>1e-12: raise SystemExit("E12a reproduction arithmetic mismatch")
        if bool(row.get("reproduced")) is not (absolute<=2e-6): raise SystemExit("E12a tolerance mismatch")
    all_pass=all(bool(by[t].get("reproduced")) for t in EXPECTED_SCORES)
    if metrics.get("all_supervised_tasks_reproduced") is not all_pass: raise SystemExit("E12a all-pass mismatch")
    expected_next="proceed_to_E12b_challenge_prediction_freeze" if all_pass else "investigate_reproduction_contract_no_model_selection"
    if metrics.get("next_step")!=expected_next: raise SystemExit("E12a next-step mismatch")
    t14=metrics.get("task14_contract") or {}
    if t14.get("incumbent")!="raw_pre_vacc_conserved_anchor" or t14.get("supervised_cv_available") is not False or t14.get("outcomes_accessed") is not False: raise SystemExit("E12a Task1.4 contract mismatch")

    if not summary.startswith(HEADING+"\n"): raise SystemExit("E12a summary heading mismatch")
    if "None" in summary or "# CMI-Flu strategy E01" in summary or "E11b" in summary: raise SystemExit("E12a stale summary presentation detected")
    required_summary=(
        f"- all supervised tasks reproduced: `{str(all_pass).lower()}`",
        f"- next step: `{expected_next}`",
        "- incumbent: `raw_pre_vacc_conserved_anchor`",
    )
    if any(token not in summary for token in required_summary): raise SystemExit("E12a summary/metrics mismatch")
    for task,row in by.items():
        token=f"- {task}: incumbent `{row['incumbent']}`, expected={float(row['expected_study_equal_spearman']):.9f}, observed={float(row['observed_study_equal_spearman']):.9f}, abs_delta={float(row['absolute_deviation']):.9g}, reproduced={str(bool(row['reproduced'])).lower()}"
        if token not in summary: raise SystemExit(f"E12a summary task mismatch:{task}")
    raw="\n".join((root/name).read_text(errors="replace") for name in sorted(expected))
    for banned in ('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"','KGAT_','KAGGLE_API_TOKEN'):
        if banned in raw: raise SystemExit("E12a sensitive/row-level token in persistent output")
    print("CMI_FLU_E12A_SANITIZE PASS aggregate_only=true summary_identity=true submission=false all_reproduced="+str(all_pass).lower()+" next_step="+expected_next)
    return 0


if __name__=="__main__": raise SystemExit(main())
