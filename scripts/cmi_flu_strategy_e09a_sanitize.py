#!/usr/bin/env python3
"""Validate aggregate-only E09a applicability audit outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUEST_ID = "20260909-cmi-flu-strategy-e09a-module-teacher-audit-001"
SCIENCE_COMMIT = "537cc7fbcf668d8729010c8be304c4b1bb9b63a5"
E09A_BLOB = "37270a0a8aafc070d91b5f43fbea3ba169cd61b0"
E09A_SYNTH_BLOB = "231a3739fd6015c61d389cce12ffa1a7aefb3f38"
ALIASES_BLOB = "5b9930c8e26f2b462ee8ce64fdfbdb76df2d1af3"
CONTRACTS_BLOB = "6e2791e526255d9c534e71c9c0886f0c300df7bd"
E04B_BLOB = "72f936a7248dfa17338932077d17fb43123cb056"
E04_BLOB = "73c112a8bb3b1ee7a6dd5bfbd5645a945bea6ebb"
E04_CONTRACT_BLOB = "3982541febfb4641fbf895438ae1a465bdbb3d5e"
TASK13_BLOB = "5c6725dc757a5ba9dd21289b1c4f09997e1afdb8"
BANNED = (
    '"participant_id"',
    '"subject_group"',
    '"row_index"',
    '"oof_predictions"',
    '"challenge_predictions"',
    '"cdr3"',
    '"sequence_id"',
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    root = p.parse_args().input_dir.resolve()
    files = sorted(path.name for path in root.iterdir() if path.is_file())
    if files != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit(f"E09a safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E09a aggregate privacy contract failed:{name}")

    bridge = json.loads(texts["bridge-result.json"])
    metrics = json.loads(texts["metrics.json"])
    expected_bridge = {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e09a_blob_sha": E09A_BLOB,
        "strategy_e09a_synthetic_blob_sha": E09A_SYNTH_BLOB,
        "aliases_blob_sha": ALIASES_BLOB,
        "contracts_blob_sha": CONTRACTS_BLOB,
        "strategy_e04b_blob_sha": E04B_BLOB,
        "strategy_e04_blob_sha": E04_BLOB,
        "strategy_e04_contract_blob_sha": E04_CONTRACT_BLOB,
        "task13_harmonization_blob_sha": TASK13_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
        "synthetic": False,
    }
    for key, expected in expected_bridge.items():
        if bridge.get(key) != expected:
            raise SystemExit(f"E09a bridge contract mismatch:{key}:{bridge.get(key)!r}")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json"):
        raise SystemExit("E09a metrics hash mismatch")
    if bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E09a summary hash mismatch")

    if metrics.get("schema_version") != 1 or metrics.get("experiment") != "strategy_v2_e09a_module_teacher_applicability_audit":
        raise SystemExit("E09a experiment identity mismatch")
    if metrics.get("audit_type") != "outcome_independent_presence_schema_and_measurement_compatibility":
        raise SystemExit("E09a audit type mismatch")
    if metrics.get("outcome_values_accessed") is not False or metrics.get("public_leaderboard_used_for_selection") is not False or metrics.get("competition_submission_attempted") is not False:
        raise SystemExit("E09a outcome/selection/submission boundary")
    if metrics.get("manifest_entries_present_for_audited_files") is not True:
        raise SystemExit("E09a manifest coverage missing")

    challenge = metrics.get("challenge_rna") or {}
    raw = challenge.get("raw") or {}
    corrected = challenge.get("batch_corrected") or {}
    if int(challenge.get("expected_donors", -1)) != 40 or int(raw.get("all_participants", -1)) != 40:
        raise SystemExit("E09a Challenge donor count mismatch")
    if raw.get("value_column") != "tpm" or int(raw.get("negative_values", -1)) != 0 or int(raw.get("numeric_parse_failures", -1)) != 0:
        raise SystemExit("E09a raw RNA contract mismatch")
    if corrected.get("value_column") != "batch_corrected_expression" or int(corrected.get("numeric_parse_failures", -1)) != 0:
        raise SystemExit("E09a corrected RNA contract mismatch")
    if challenge.get("batch_corrected_has_negative_values") is not True:
        raise SystemExit("E09a corrected negative-value evidence missing")
    if challenge.get("public_batch_corrected_counterpart_available") is not False or challenge.get("batch_corrected_form_admissible_for_cross_study_module_training") is not False:
        raise SystemExit("E09a corrected representation boundary mismatch")
    raw_day0 = int(challenge.get("raw_day0_donors", -1))
    raw_negative = int(challenge.get("raw_negative_baseline_donors", -1))
    raw_union = int(challenge.get("raw_day0_or_negative_baseline_donors", -1))
    raw_intersection = int(challenge.get("raw_day0_and_negative_baseline_donors", -1))
    if not (0 <= raw_day0 <= 40 and 0 <= raw_negative <= 40 and 0 <= raw_intersection <= min(raw_day0, raw_negative) and max(raw_day0, raw_negative) <= raw_union <= 40):
        raise SystemExit("E09a Challenge baseline coverage accounting mismatch")
    if int(challenge.get("raw_day0_missing_donors_require_frozen_base_fallback", -1)) != 40 - raw_day0:
        raise SystemExit("E09a Challenge missing-view fallback mismatch")

    public_rna = metrics.get("public_rna") or {}
    expected_public = {"2019_UGA", "2020_UGA", "2024_UGA", "SDY224", "SDY2867", "SDY2941"}
    if set(public_rna) != expected_public:
        raise SystemExit("E09a public RNA study set mismatch")
    for study, payload in public_rna.items():
        if payload.get("study") != study or payload.get("value_column") != "tpm":
            raise SystemExit(f"E09a public RNA identity mismatch:{study}")
        if int(payload.get("numeric_parse_failures", -1)) != 0 or int(payload.get("day0_gene_count", -1)) < 0:
            raise SystemExit(f"E09a public RNA schema/count mismatch:{study}")

    tasks = metrics.get("task_readiness") or {}
    if set(tasks) != {"Task1.3", "Task2.1", "Task2.2", "Task2.3"}:
        raise SystemExit("E09a task readiness set mismatch")
    aggregate = {}
    for task, payload in tasks.items():
        if payload.get("task") != task or not isinstance(payload.get("direct_rna_teacher_data_ready"), bool):
            raise SystemExit(f"E09a readiness type mismatch:{task}")
        studies = payload.get("source_studies") or []
        if len(studies) != 6:
            raise SystemExit(f"E09a readiness source-study count mismatch:{task}")
        if int(payload.get("studies_meeting_per_study_gate", -1)) < 0 or int(payload.get("total_material_compatible_paired_subjects", -1)) < 0:
            raise SystemExit(f"E09a readiness count mismatch:{task}")
        if task == "Task1.3":
            if payload.get("official_strict_gate_teacher_required") is not True or payload.get("proxy_teacher_promoted_to_strict") is not False:
                raise SystemExit("E09a Task1.3 strict teacher boundary mismatch")
        else:
            if payload.get("readiness_rule") != {"minimum_studies": 2, "minimum_subjects_per_study": 8, "minimum_total_subjects": 24, "minimum_common_genes": 1000}:
                raise SystemExit(f"E09a readiness threshold mismatch:{task}")
            if payload.get("official_complete_panel_cv_claimed") is not False:
                raise SystemExit(f"E09a official CV claim mismatch:{task}")
        aggregate[task] = {
            "ready": payload.get("direct_rna_teacher_data_ready"),
            "ready_studies": payload.get("studies_meeting_per_study_gate"),
            "paired": payload.get("total_material_compatible_paired_subjects"),
        }

    decision = metrics.get("decision") or {}
    if decision.get("module_gene_sets_frozen") is not False or decision.get("model_fitting_authorized_by_e09a") is not False:
        raise SystemExit("E09a model authorization boundary mismatch")
    if decision.get("batch_corrected_challenge_only_form_selected") is not False or decision.get("missing_view_policy") != "fallback_to_frozen_base":
        raise SystemExit("E09a representation/missing-view boundary mismatch")
    if not isinstance(decision.get("e09b_hai_module_data_gate_ready"), bool) or not isinstance(decision.get("task13_rna_module_data_gate_ready"), bool):
        raise SystemExit("E09a decision readiness booleans missing")

    bridge13 = metrics.get("task13_measurement_bridge") or {}
    if bridge13.get("ontology_version") != "e04b_task13_material_plural_v1" or bridge13.get("strict_gate_reconstructed") is not False:
        raise SystemExit("E09a Task1.3 measurement bridge mismatch")
    repertoire = metrics.get("repertoire_applicability") or {}
    if repertoire.get("single_cell_vdj_external_teacher_available") is not False or repertoire.get("generic_clonality_promoted_to_antigen_specific_teacher") is not False:
        raise SystemExit("E09a repertoire boundary mismatch")

    print(
        "CMI_FLU_E09A_RESULT PASS aggregate_only=true outcomes=false submission=false "
        f"raw_day0={raw_day0} raw_negative={raw_negative} raw_union={raw_union} "
        f"hai_ready={decision.get('e09b_hai_module_data_gate_ready')} "
        f"task13_ready={decision.get('task13_rna_module_data_gate_ready')} "
        f"tasks={json.dumps(aggregate, sort_keys=True, separators=(',', ':'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
