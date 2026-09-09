#!/usr/bin/env python3
"""Validate aggregate-only E09b HIPC9 RNA residual outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260909-cmi-flu-strategy-e09b-hipc9-rna-residual-001"
SCIENCE_COMMIT = "8f36ba0fa0d5fae6809ff09e0bf15bb132dd1ff5"
E09B_BLOB = "6a58b1f0a09ac6b2b07473627af37847d9ac41c4"
E09B_SYNTH_BLOB = "3c5a08f5bd209ae094e4fcc1d66f65abe949c018"
MAPPING_BLOB = "ae42cab3dfffb242dd695a6b1df0357fc90cf2d0"
E09A_BLOB = "37270a0a8aafc070d91b5f43fbea3ba169cd61b0"
EXPECTED_GENES = ["RAB24", "GRB2", "DPP3", "ACTB", "MVP", "DPP7", "ARPC4", "PLEKHB2", "ARRB1"]
EXPECTED_ENSEMBL = [
    "ENSG00000169228", "ENSG00000177885", "ENSG00000254986",
    "ENSG00000075624", "ENSG00000013364", "ENSG00000176978",
    "ENSG00000241553", "ENSG00000115762", "ENSG00000137486",
]
BANNED = (
    '"participant_id"', '"subject_group"', '"row_index"', '"oof_predictions"',
    '"challenge_predictions"', '"predictions"', '"sequence_id"', '"cdr3"',
    "SYN_E09B_",
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
    root = p.parse_args().input_dir.expanduser().resolve()
    files = sorted(path.name for path in root.iterdir() if path.is_file())
    if files != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit(f"E09b safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E09b aggregate privacy contract failed:{name}")

    bridge = json.loads(texts["bridge-result.json"])
    metrics = json.loads(texts["metrics.json"])
    for key, value in {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e09a_blob_sha": E09A_BLOB,
        "strategy_e09b_blob_sha": E09B_BLOB,
        "strategy_e09b_synthetic_blob_sha": E09B_SYNTH_BLOB,
        "strategy_e09b_mapping_blob_sha": MAPPING_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }.items():
        if bridge.get(key) != value:
            raise SystemExit(f"E09b bridge provenance mismatch:{key}:{bridge.get(key)!r}")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json"):
        raise SystemExit("E09b metrics hash mismatch")
    if bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E09b summary hash mismatch")

    if metrics.get("schema_version") != 1 or metrics.get("experiment") != "strategy_v2_e09b_hipc9_rna_residual":
        raise SystemExit("E09b experiment identity mismatch")
    if metrics.get("competition_submission_attempted") is not False or metrics.get("public_leaderboard_used_for_selection") is not False or metrics.get("automatic_compute_retries") != 0 or metrics.get("incumbent_changed") is not False:
        raise SystemExit("E09b execution boundary mismatch")
    feature = metrics.get("feature_contract") or {}
    expected_feature = {
        "source_doi": "10.1126/sciimmunol.aal4656",
        "source_pmid": "28842433",
        "gene_symbols": EXPECTED_GENES,
        "ensembl_ids": EXPECTED_ENSEMBL,
        "validation_subset": ["GRB2", "ACTB", "MVP", "DPP7", "ARPC4", "PLEKHB2", "ARRB1"],
        "day0_only": True,
        "raw_tpm_only": True,
        "within_study_gene_rank": True,
        "minimum_gene_coverage": 7,
        "ridge_alpha": 10.0,
        "residual_shrinkage": 0.25,
        "max_score_correction": 0.1,
        "minimum_fold_subjects": 24,
        "outcome_based_gene_selection": False,
        "batch_corrected_challenge_expression_used": False,
    }
    for key, value in expected_feature.items():
        if feature.get(key) != value:
            raise SystemExit(f"E09b feature contract mismatch:{key}")
    evaluation = metrics.get("evaluation_contract") or {}
    for key, value in {
        "qualifying_studies": ["2020_UGA", "2024_UGA"],
        "historical_panels_are_incomplete_proxies": True,
        "official_complete_2025_panel_cv_claimed": False,
        "promotion_mean_delta": 0.02,
        "promotion_min_study_delta": -0.05,
        "shuffle_seed": 20260909,
        "challenge_missing_rna_policy": "exact_frozen_b21_fallback",
    }.items():
        if evaluation.get(key) != value:
            raise SystemExit(f"E09b evaluation contract mismatch:{key}")

    rna = metrics.get("rna_audit") or {}
    public = rna.get("public") or {}
    challenge = rna.get("challenge") or {}
    if set(public) != {"2020_UGA", "2024_UGA"}:
        raise SystemExit("E09b public RNA study set mismatch")
    for study, audit in public.items():
        if audit.get("study") != study or audit.get("day0_only") is not True or audit.get("raw_tpm_only") is not True or audit.get("material_category") != "PBMC":
            raise SystemExit(f"E09b public RNA identity mismatch:{study}")
        if int(audit.get("eligible_participants", -1)) < 24 or not 7 <= int(audit.get("present_gene_count", -1)) <= 9:
            raise SystemExit(f"E09b public RNA coverage mismatch:{study}")
        symbols = audit.get("present_symbols") or []
        if not set(symbols).issubset(EXPECTED_GENES) or len(symbols) < 7:
            raise SystemExit(f"E09b public RNA symbol contract mismatch:{study}")
    if challenge.get("study") != "2025LJI" or challenge.get("day0_only") is not True or challenge.get("raw_tpm_only") is not True or challenge.get("material_category") != "PBMC":
        raise SystemExit("E09b Challenge RNA identity mismatch")
    challenge_eligible = int(challenge.get("eligible_participants", -1))
    if not 35 <= challenge_eligible <= 40 or not 7 <= int(challenge.get("present_gene_count", -1)) <= 9:
        raise SystemExit("E09b Challenge RNA coverage mismatch")

    tasks = metrics.get("tasks") or {}
    if set(tasks) != {"Task2.1", "Task2.2"}:
        raise SystemExit("E09b task set mismatch")
    safe_summary = {}
    for task, payload in tasks.items():
        if payload.get("task") != task or int(payload.get("historical_joined_subjects", -1)) < 48:
            raise SystemExit(f"E09b historical support mismatch:{task}")
        studies = payload.get("study_metrics") or []
        if [row.get("study") for row in studies] != ["2020_UGA", "2024_UGA"]:
            raise SystemExit(f"E09b study order/set mismatch:{task}")
        for row in studies:
            if int(row.get("n", -1)) < 24:
                raise SystemExit(f"E09b study support mismatch:{task}:{row.get('study')}")
            if any(not finite(row.get(key)) for key in ("base_spearman", "candidate_spearman", "hipc7_zero_shot_spearman", "shuffle_spearman", "delta_vs_base")):
                raise SystemExit(f"E09b nonfinite study metric:{task}:{row.get('study')}")
        if any(not finite(payload.get(key)) for key in ("study_equal_base_spearman", "study_equal_candidate_spearman", "study_equal_shuffle_spearman", "study_equal_delta_vs_base", "minimum_study_delta_vs_base")):
            raise SystemExit(f"E09b nonfinite aggregate metric:{task}")
        movement = payload.get("challenge_movement") or {}
        donors = int(movement.get("donors", -1)); available = int(movement.get("rna_available_donors", -1)); fallback = int(movement.get("frozen_base_fallback_donors", -1)); changed = int(movement.get("changed_rank_count_vs_base", -1))
        if donors != 40 or available != challenge_eligible or fallback != 40 - available or not 0 <= changed <= 40:
            raise SystemExit(f"E09b Challenge movement accounting:{task}")
        if not finite(movement.get("rank_spearman_vs_base")) or not finite(movement.get("mean_absolute_percentile_shift")) or not finite(movement.get("max_absolute_percentile_shift")):
            raise SystemExit(f"E09b Challenge movement metric missing:{task}")
        if not finite(movement.get("max_absolute_score_correction")) or float(movement["max_absolute_score_correction"]) > 0.100000000001:
            raise SystemExit(f"E09b correction cap mismatch:{task}")
        decision = payload.get("decision") or {}
        mean_delta = float(payload["study_equal_delta_vs_base"]); minimum = float(payload["minimum_study_delta_vs_base"]); candidate = float(payload["study_equal_candidate_spearman"]); shuffled = float(payload["study_equal_shuffle_spearman"])
        expected_local = bool(mean_delta >= 0.02 and minimum >= -0.05 and candidate >= shuffled)
        expected_competition = bool(expected_local and changed > 0)
        if decision.get("local_promotion_gate_passed") is not expected_local or decision.get("competition_candidate") is not expected_competition:
            raise SystemExit(f"E09b promotion boolean inconsistency:{task}")
        if decision.get("public_probe_authorized") is not False or decision.get("incumbent_changed") is not False:
            raise SystemExit(f"E09b promotion boundary mismatch:{task}")
        safe_summary[task] = {
            "joined": payload["historical_joined_subjects"],
            "base": payload["study_equal_base_spearman"],
            "candidate": payload["study_equal_candidate_spearman"],
            "delta": payload["study_equal_delta_vs_base"],
            "min_delta": payload["minimum_study_delta_vs_base"],
            "shuffle": payload["study_equal_shuffle_spearman"],
            "changed": changed,
            "rank_agreement": movement["rank_spearman_vs_base"],
            "local_gate": expected_local,
            "competition_candidate": expected_competition,
        }

    print(
        "CMI_FLU_E09B_RESULT PASS aggregate_only=true submission=false public_probe=false "
        f"challenge_rna={challenge_eligible} tasks={json.dumps(safe_summary, sort_keys=True, separators=(',', ':'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
