#!/usr/bin/env python3
"""Aggregate-only sanitizer for E05 006 real-reference correction."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-006"
SCIENCE_COMMIT = "8209ee237dace5ee383d7983d3d56f2e845dfdca"
E05_BLOB = "78cde9e3a6b6e3b332352218c4b4545f769aa6c4"
E05_V2_BLOB = "50951807d28418bb50f4e9dba656fa2b3e4d86ff"
E05_V3_BLOB = "129a4a5713a758d58adbb02489105e17a1c13922"
HAI_TRANSFER_V2_BLOB = "ebf4110df4ab49d726d80af03c1203f93e9c89c0"
BANNED = ('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"','"predictions"')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_or_none(value) -> bool:
    if value is None:
        return True
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    root = p.parse_args().input_dir.resolve()
    files = sorted(x.name for x in root.iterdir() if x.is_file())
    if files != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit(f"E05 v5 safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text() for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E05 v5 privacy contract failed:{name}")

    bridge = json.loads(texts["bridge-result.json"])
    metrics = json.loads(texts["metrics.json"])
    expected = {
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e05_blob_sha": E05_BLOB,
        "strategy_e05_v2_blob_sha": E05_V2_BLOB,
        "strategy_e05_v3_blob_sha": E05_V3_BLOB,
        "hai_transfer_v2_blob_sha": HAI_TRANSFER_V2_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
    for key, value in expected.items():
        if bridge.get(key) != value:
            raise SystemExit(f"E05 v5 bridge contract mismatch:{key}:{bridge.get(key)!r}")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json") or bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E05 v5 output hash mismatch")

    if metrics.get("experiment") != "strategy_v2_e05_hai_donor_strain" or set((metrics.get("tasks") or {})) != {"Task2.1", "Task2.2"}:
        raise SystemExit("E05 v5 experiment/task identity mismatch")
    fc = metrics.get("feature_contract") or {}
    sc = metrics.get("split_contract") or {}
    wc = metrics.get("weight_contract") or {}
    if fc.get("ridge_alpha") != 10.0 or fc.get("interaction_count") != 8 or fc.get("free_participant_embedding") is not False or fc.get("raw_strain_id_one_hot") is not False:
        raise SystemExit("E05 v5 feature contract mismatch")
    if fc.get("donor_rank_unit") != "participant_id_within_study":
        raise SystemExit("E05 v5 donor rank unit mismatch")
    if fc.get("subject_group_role") != "leakage_purge_and_hierarchical_weighting_only":
        raise SystemExit("E05 v5 subject-group role mismatch")
    if fc.get("sequence_reference_schema") != "organizer_native_or_canonical_explicit_adapter":
        raise SystemExit("E05 v5 sequence reference schema mismatch")
    if fc.get("organizer_sequence_reference_columns") != ["Virus", "Sequence", "Status_of_sequence"]:
        raise SystemExit("E05 v5 organizer sequence columns mismatch")
    if sc.get("fixed_subject_strain_simultaneous_holdouts") != 4 or sc.get("simultaneous_selection_uses_outcomes") is not False:
        raise SystemExit("E05 v5 split contract mismatch")
    if wc.get("rows_are_not_counted_as_independent_donors") is not True:
        raise SystemExit("E05 v5 weight contract mismatch")

    expected_conditions = {
        "b21_reference", "phase_a_fixed_sequence", "ridge_main_effects", "ridge_donor_by_strain_interactions"
    }
    for task, payload in metrics["tasks"].items():
        route = payload.get("routing") or {}
        delta = route.get("historical_proxy_study_mean_delta_vs_anchor")
        if not finite_or_none(delta):
            raise SystemExit(f"E05 v5 nonfinite route delta:{task}")
        if set(payload.get("study_out_conditions") or {}) != expected_conditions:
            raise SystemExit(f"E05 v5 condition set mismatch:{task}")
        print(
            f"{task} route={route.get('routing')} positive={route.get('positive_e05_signal')} "
            f"delta_vs_anchor={delta} challenge_rank_changed={route.get('challenge_rank_changed_by_at_least_one_position')} "
            f"interaction_rmse={route.get('interaction_panel_proxy_rmse')} main_rmse={route.get('main_effects_panel_proxy_rmse')}"
        )
    print(
        "CMI_FLU_E05_V5_RESULT PASS aggregate_only=true submission=false "
        f"science_commit={SCIENCE_COMMIT} e05_v3_blob={E05_V3_BLOB} hai_v2_blob={HAI_TRANSFER_V2_BLOB}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
