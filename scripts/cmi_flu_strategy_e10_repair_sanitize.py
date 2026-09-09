#!/usr/bin/env python3
"""Validate aggregate-only repaired E10 pooled domain-correction outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REQUEST_ID = "20260909-cmi-flu-strategy-e10-pooled-domain-correction-002"
SCIENCE_COMMIT = "3a8728879179115487f5ce0ba4a4a0467e1910bb"
E10_BLOB = "638b09860f85689714cb0b354f4f377a405deeac"
E10_SYNTH_BLOB = "f1f33f3909e64e69fb859d98b4163cdb4860d64d"
ANCHOR_BLOB = "9a7814ecdf70e0b38b2740ade21e9db588869379"
RANK_TRANSFER_BLOB = "d5e07cdd09d2eabdc935eb1733ec238e26ab4c17"
STUDY_SIMILARITY_BLOB = "27351df3d9187899c4bce2ff1a24b06efc160185"
TARGET = "renta0426/cmi-flu-e10-pooled-domain-correction-20260909-002"
MIN_SOURCE_SUBJECTS = 5
CANDIDATES = {"xonly_weighted_shared", "shrunk_study_deviation"}
BANNED = (
    '"participant_id"', '"subject_group"', '"row_index"',
    '"oof_predictions"', '"challenge_predictions"', 'ROW_', 'SUB_',
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SystemExit(f"E10 nonfinite numeric value:{value!r}")
    return number


def validate_weight_audit(audit: dict, label: str) -> None:
    weights = audit.get("source_weights") or {}
    counts = audit.get("source_counts") or {}
    if set(weights) != set(counts) or len(weights) < 2:
        raise SystemExit(f"E10 source domain mismatch:{label}")
    if min(int(value) for value in counts.values()) < MIN_SOURCE_SUBJECTS:
        raise SystemExit(f"E10 insufficient source support:{label}")
    if min(finite(value) for value in weights.values()) < 0.5 - 1e-12:
        raise SystemExit(f"E10 source weight below clip:{label}")
    if max(finite(value) for value in weights.values()) > 2.0 + 1e-12:
        raise SystemExit(f"E10 source weight above clip:{label}")
    if abs(finite(audit.get("row_weight_mean")) - 1.0) > 1e-8:
        raise SystemExit(f"E10 row weight mean mismatch:{label}")
    ess = finite(audit.get("row_weight_ess_fraction"))
    if not 0.0 < ess <= 1.0 + 1e-12:
        raise SystemExit(f"E10 ESS fraction mismatch:{label}")
    if audit.get("held_outcomes_used") is not False:
        raise SystemExit(f"E10 outcome-independent weighting boundary:{label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    root = parser.parse_args().input_dir.resolve()
    files = sorted(path.name for path in root.iterdir() if path.is_file())
    if files != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit(f"E10 safe-output file set mismatch:{files}")
    texts = {name: (root / name).read_text(encoding="utf-8") for name in files}
    for name, text in texts.items():
        if any(token in text for token in BANNED):
            raise SystemExit(f"E10 aggregate privacy contract failed:{name}")

    bridge = json.loads(texts["bridge-result.json"])
    metrics = json.loads(texts["metrics.json"])
    expected_bridge = {
        "request_id": REQUEST_ID,
        "target_kernel": TARGET,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e10_blob_sha": E10_BLOB,
        "strategy_e10_synthetic_blob_sha": E10_SYNTH_BLOB,
        "anchor_residual_blob_sha": ANCHOR_BLOB,
        "rank_transfer_blob_sha": RANK_TRANSFER_BLOB,
        "study_similarity_blob_sha": STUDY_SIMILARITY_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
        "synthetic": False,
    }
    for key, expected in expected_bridge.items():
        if bridge.get(key) != expected:
            raise SystemExit(f"E10 bridge contract mismatch:{key}:{bridge.get(key)!r}")
    if bridge.get("metrics_sha256") != sha(root / "metrics.json"):
        raise SystemExit("E10 metrics hash mismatch")
    if bridge.get("summary_sha256") != sha(root / "summary.md"):
        raise SystemExit("E10 summary hash mismatch")

    if int(metrics.get("schema_version", -1)) != 1 or metrics.get("experiment") != "strategy_v2_e10_pooled_domain_conditioned_residual":
        raise SystemExit("E10 experiment identity mismatch")
    if metrics.get("comparison_contract") != "paired_subject_purged_v2" or set((metrics.get("tasks") or {})) != {"Task1.1", "Task1.2"}:
        raise SystemExit("E10 task/comparison set mismatch")
    if metrics.get("held_target_outcomes_used_for_domain_weights") is not False or metrics.get("isolated_source_study_models_allowed") is not False:
        raise SystemExit("E10 domain adaptation boundary mismatch")
    for key in ("leaderboard_used_for_selection", "competition_submission_attempted", "incumbent_changed", "public_probe_authorized"):
        if metrics.get(key) is not False:
            raise SystemExit(f"E10 execution boundary mismatch:{key}")
    if int(metrics.get("automatic_compute_retries", -1)) != 0:
        raise SystemExit("E10 retry contract mismatch")

    frozen = metrics.get("frozen_conditions") or {}
    if frozen.get("ridge_alpha") != 10.0 or frozen.get("correction_shrinkage") != 0.25 or frozen.get("absolute_rank_correction_cap") != 0.05:
        raise SystemExit("E10 ridge/shrinkage/cap contract mismatch")
    if frozen.get("source_weight_clip") != [0.5, 2.0] or frozen.get("deviation_scale") != 0.25 or frozen.get("minimum_source_subjects") != MIN_SOURCE_SUBJECTS:
        raise SystemExit("E10 domain parameter contract mismatch")
    if frozen.get("promotion_mean_delta") != 0.02 or frozen.get("promotion_minimum_study_delta") != -0.10:
        raise SystemExit("E10 promotion threshold contract mismatch")

    expected_bases = {
        "Task1.1": {"kind": "b21", "model": "pls_2", "anchor_column": None, "anchor_lambda": None, "current_competition_incumbent": True},
        "Task1.2": {"kind": "anchor_residual", "model": "et_d5_l5_sqrt", "anchor_column": "flow_rank__Classical_monocytes", "anchor_lambda": 0.5, "current_competition_incumbent": True},
    }
    expected_folds = {"Task1.1": 4, "Task1.2": 3}
    aggregate = {}
    for task, payload in metrics["tasks"].items():
        if payload.get("base_contract") != expected_bases[task]:
            raise SystemExit(f"E10 base contract mismatch:{task}")
        folds = payload.get("folds") or []
        if len(folds) != expected_folds[task]:
            raise SystemExit(f"E10 fold count mismatch:{task}:{len(folds)}")
        if set((payload.get("candidates") or {})) != CANDIDATES:
            raise SystemExit(f"E10 candidate set mismatch:{task}")
        if int(payload.get("challenge_rows", -1)) != 40:
            raise SystemExit(f"E10 challenge row mismatch:{task}")
        for fold in folds:
            if int(fold.get("n", 0)) < 3 or int(fold.get("training_studies", 0)) < 2:
                raise SystemExit(f"E10 fold support mismatch:{task}")
            for key in (
                "base_spearman", "xonly_weighted_shared_spearman", "shrunk_study_deviation_spearman",
                "xonly_weighted_shared_delta", "shrunk_study_deviation_delta",
            ):
                finite(fold.get(key))
            validate_weight_audit(fold.get("xonly_weight_audit") or {}, f"{task}/{fold.get('held_study')}")
            for movement_key in ("xonly_weighted_movement", "shrunk_study_deviation_movement"):
                if finite((fold.get(movement_key) or {}).get("max_absolute_rank_correction")) > 0.050000000001:
                    raise SystemExit(f"E10 movement cap:{task}:{movement_key}")
            wf = fold.get("xonly_weighted_fit") or {}
            hf = fold.get("shrunk_study_deviation_fit") or {}
            if wf.get("pooled_model_count") != 1 or wf.get("isolated_source_models_fit") != 0:
                raise SystemExit(f"E10 weighted pooling mismatch:{task}")
            if hf.get("pooled_model_count") != 1 or hf.get("isolated_source_models_fit") != 0 or hf.get("unseen_target_deviation_columns") != "all_zero":
                raise SystemExit(f"E10 hierarchical pooling mismatch:{task}")

        task_aggregate = {}
        passing = []
        for candidate_name, candidate in payload["candidates"].items():
            promotion = candidate.get("promotion") or {}
            deltas = [finite(fold[f"{candidate_name}_delta"]) for fold in folds]
            wins = sum(delta > 0 for delta in deltas)
            required = len(deltas) // 2 + 1
            mean_delta = sum(deltas) / len(deltas)
            min_delta = min(deltas)
            expected_pass = bool(mean_delta >= 0.02 and min_delta >= -0.10 and wins >= required)
            if promotion.get("passed") is not expected_pass:
                raise SystemExit(f"E10 promotion boolean mismatch:{task}:{candidate_name}")
            if abs(finite(promotion.get("mean_delta")) - mean_delta) > 1e-12 or abs(finite(promotion.get("minimum_delta")) - min_delta) > 1e-12:
                raise SystemExit(f"E10 promotion metric mismatch:{task}:{candidate_name}")
            if int(promotion.get("wins", -1)) != wins or int(promotion.get("required_wins", -1)) != required:
                raise SystemExit(f"E10 promotion win mismatch:{task}:{candidate_name}")
            if expected_pass:
                passing.append(candidate_name)
            agreement = candidate.get("challenge_agreement_vs_base") or {}
            rank = finite((agreement.get("rank_spearman") or {}).get("value"))
            changed = int(agreement.get("changed_rank_count", -1))
            if not 0 <= changed <= 40:
                raise SystemExit(f"E10 changed-rank mismatch:{task}:{candidate_name}")
            if finite((candidate.get("challenge_movement") or {}).get("max_absolute_rank_correction")) > 0.050000000001:
                raise SystemExit(f"E10 challenge movement cap:{task}:{candidate_name}")
            task_aggregate[candidate_name] = {
                "pass": expected_pass,
                "mean_delta": mean_delta,
                "min_delta": min_delta,
                "wins": wins,
                "required": required,
                "challenge_rank_agreement": rank,
                "challenge_changed_ranks": changed,
            }
        selected = payload.get("selected_local_candidate")
        if (selected is None) != (len(passing) == 0) or (selected is not None and selected not in passing):
            raise SystemExit(f"E10 local selection mismatch:{task}")
        if payload.get("competition_candidate") is not False or payload.get("public_probe_authorized") is not False:
            raise SystemExit(f"E10 competition boundary mismatch:{task}")
        validate_weight_audit(payload.get("challenge_source_weight_audit") or {}, f"{task}/challenge")
        aggregate[task] = task_aggregate

    print(
        "CMI_FLU_E10_RESULT PASS aggregate_only=true submission=false public_probe=false "
        f"tasks={json.dumps(aggregate, sort_keys=True, separators=(',', ':'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
