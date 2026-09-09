#!/usr/bin/env python3
"""Validate and emit the aggregate-only E07 Task1.4 result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUEST_ID = "20260909-cmi-flu-strategy-e07-task14-formal-closure-001"
SCIENCE_COMMIT = "a6cdc7c43bc487d88e9c15f2c63565ff8c702dac"
BLOBS = {
    "strategy_e07_blob_sha": "44ebdd2a1dc9ceeb771eaeb295c74545eb3cc1a4",
    "strategy_e07_synthetic_blob_sha": "b8971d05d0d1f094b03f861ec3438c6f17c8587a",
    "aliases_blob_sha": "5b9930c8e26f2b462ee8ce64fdfbdb76df2d1af3",
    "contracts_blob_sha": "6e2791e526255d9c534e71c9c0886f0c300df7bd",
    "targets_blob_sha": "154681d9a51e36abd635b596c6d8f7cee22c2d96",
}
BANNED = ('"participant_id"', '"subject"', '"barcode"', '"contig_id"', '"cdr3"', 'SYN_ONLY_E07_')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    files = sorted(path.name for path in root.iterdir() if path.is_file())
    if files != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit(f"E07 safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E07 privacy contract failed:{name}")

    bridge = json.loads(texts["bridge-result.json"])
    result = json.loads(texts["metrics.json"])
    expected = {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        **BLOBS,
        "synthetic": False,
        "outcomes_accessed": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
    for key, value in expected.items():
        if bridge.get(key) != value:
            raise SystemExit(f"E07 bridge contract mismatch:{key}:{bridge.get(key)!r}")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json") or bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E07 output hash mismatch")
    if bridge.get("md5_verified_count") != 4:
        raise SystemExit("E07 required-file MD5 verification mismatch")

    if result.get("schema_version") != 1 or result.get("experiment") != "strategy_v2_e07_task14_formal_closure" or result.get("task") != "Task1.4":
        raise SystemExit("E07 experiment identity mismatch")
    if result.get("contract_version") != "e07_task14_zero_shot_audit_v1":
        raise SystemExit("E07 contract version mismatch")
    if result.get("status") not in {"complete", "real_contract_review_required"}:
        raise SystemExit("E07 unexpected status")
    if result.get("outcomes_accessed") is not False or result.get("competition_submission_attempted") is not False or result.get("leaderboard_used_for_selection") is not False or result.get("automatic_compute_retries") != 0:
        raise SystemExit("E07 execution boundary mismatch")
    if result.get("public_training_aim_rows") != 0 or result.get("public_training_aim_studies") != 0:
        raise SystemExit("E07 fabricated training AIM evidence")

    real = result.get("real_contract") or {}
    if real.get("aim_rows") != 600 or real.get("challenge_subjects") != 40 or real.get("anchor_subjects") != 40 or real.get("stimulation_count") != 5:
        raise SystemExit("E07 real structural counts mismatch")
    if set(real.get("baseline_timepoints") or []) != {"-14", "0", "Pre-vacc"}:
        raise SystemExit("E07 baseline timepoints changed")

    aim = result.get("aim_audit") or {}
    if aim.get("rows") != 600 or aim.get("subjects") != 40 or aim.get("anchor_subjects") != 40 or aim.get("stimulation_count") != 5:
        raise SystemExit("E07 AIM aggregate mismatch")
    if "Conserved" not in (aim.get("stimulations") or []):
        raise SystemExit("E07 conserved stimulation missing")
    prevacc = aim.get("prevacc_arithmetic_mean_check") or {}
    if prevacc.get("tolerance") != 1e-8 or not isinstance(prevacc.get("complete_measure_keys"), int):
        raise SystemExit("E07 Pre-vacc processing audit malformed")
    repeat = aim.get("conserved_repeat_reliability") or {}
    if repeat.get("paired_subjects") != 40:
        raise SystemExit("E07 repeat reliability coverage mismatch")
    negative = aim.get("negative_control") or {}
    if not isinstance(negative.get("candidate_count"), int) or not isinstance(negative.get("candidate_names"), list):
        raise SystemExit("E07 negative-control audit malformed")
    if negative.get("background_contrast_target_semantics_established") is not False:
        raise SystemExit("E07 background contrast incorrectly promoted to target semantics")

    vdj = result.get("vdj_applicability") or {}
    if vdj.get("generic_clonality_used_as_teacher") is not False:
        raise SystemExit("E07 generic clonality was used as teacher")
    if vdj.get("external_tcr_teacher_applicable"):
        if int(vdj.get("participant_coverage") or 0) < 28 or not vdj.get("antigen_specificity_columns") or not vdj.get("cell_subset_columns") or vdj.get("tcr_beta_semantics_present") is not True:
            raise SystemExit("E07 TCR applicability inconsistent")
    hla = result.get("hla_applicability") or {}
    if hla.get("epitope_hla_map_available") is not False or hla.get("hla_correction_applicable") is not False:
        raise SystemExit("E07 v1 HLA boundary changed")

    decision = result.get("decision") or {}
    if decision.get("direct_public_aim_teacher_available") is not False or decision.get("supervised_cv_available") is not False:
        raise SystemExit("E07 supervised boundary changed")
    if decision.get("recommended_task14_predictor") != "raw_pre_vacc_conserved_anchor" or decision.get("incumbent_changed") is not False or decision.get("public_probe_authorized") is not False:
        raise SystemExit("E07 incumbent/public boundary mismatch")
    external = bool(vdj.get("external_tcr_teacher_applicable"))
    expected_decision = "external_teacher_requires_separate_predeclared_model" if external else "data_limited_hypothesis_retained"
    if decision.get("external_teacher_applicable") is not external or decision.get("decision") != expected_decision:
        raise SystemExit("E07 external-teacher decision mismatch")

    aggregate = {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "science_blobs": BLOBS,
        "output_hashes": {"metrics": bridge.get("metrics_sha256"), "summary": bridge.get("summary_sha256")},
        "md5_verified_count": bridge.get("md5_verified_count"),
        "status": result.get("status"),
        "real_contract": real,
        "aim_audit": aim,
        "vdj_applicability": vdj,
        "hla_applicability": hla,
        "decision": decision,
        "interpretation_limits": result.get("interpretation_limits"),
        "reentry_conditions": result.get("reentry_conditions"),
        "outcomes_accessed": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
    }
    print("CMI_FLU_E07_AGGREGATE_SUMMARY " + json.dumps(aggregate, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    print(f"CMI_FLU_E07_RESULT PASS aggregate_only=true submission=false public_probe=false science_commit={SCIENCE_COMMIT} e07_blob={BLOBS['strategy_e07_blob_sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
