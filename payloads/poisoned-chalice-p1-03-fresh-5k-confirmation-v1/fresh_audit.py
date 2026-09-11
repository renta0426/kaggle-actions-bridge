#!/usr/bin/env python3
"""Frozen P1-03 fresh-disjoint 5k confirmation audit.

This script consumes the fresh hidden-anchor OOF predictions produced in the same
Kaggle evaluation run, the exact fresh hidden-state cache, and the frozen Stage1
10k feature cache.  It reuses the already-reviewed P1-03 development-audit
implementations for Stage1 reproduction, content controls, and portable hidden
scalars, while narrowing the fusion candidate set to the two formulas frozen
before the fresh cache existed.

No target-model forward pass or competition submission occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_AUDIT_PATH = ROOT / "scripts/run_p1_03_post_5k_development_audit_v1.py"
EXPECTED_BASE_AUDIT_GIT_BLOB = "3d0a45b20c93554e293b4db9c243cd1133f1fef9"

TASK_ID = "P1-03-fresh-disjoint-5k-confirmation-evaluation-v1"
FRESH_CACHE_MANIFEST_SHA256 = "f2b9c6048f1496b5f0c96a369d9773754f13ebace14aacfce8c0ecec92ba4481"
FRESH_COHORT_SET_SHA256 = "8fd03b98b46ffd9b7ef9a69ea58eaaeb6e0307cdafecf9d1c35a1562bbeaad27"
FRESH_CACHE_TASK_ID = "P1-03-lumia-hidden-state-fresh-cache-5000-v1"
FRESH_SELECTION_TASK_ID = "P1-03-fresh-disjoint-5k-confirmation-v1"
FRESH_HIDDEN_EVAL_TASK_ID = "P1-03-fresh-hidden-anchor-evaluation-5000-v1"
FROZEN_FUSION_WEIGHTS = (0.75, 0.50)
PRIMARY_FUSION = "rank_fusion_hidden_0.75"
SECONDARY_FUSION = "rank_fusion_hidden_0.50"
EXPECTED_ROWS = 5000
EXPECTED_STAGE1_ROWS = 10000
EXPECTED_SHARD_BYTES = 4512840643
EXPECTED_LANGUAGE_LABEL = 500

FUSION_TPR_DELTA_MIN = 0.01
FUSION_TPR_BOOTSTRAP_LOWER_MIN = 0.0
FUSION_AUC_DELTA_MIN = -0.005


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_base_audit():
    data = BASE_AUDIT_PATH.read_bytes()
    if git_blob_sha(data) != EXPECTED_BASE_AUDIT_GIT_BLOB:
        raise RuntimeError("frozen development-audit implementation identity changed")
    spec = importlib.util.spec_from_file_location("p1_03_frozen_development_audit", BASE_AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen development-audit implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if tuple(module.FUSION_HIDDEN_WEIGHTS) != (0.25, 0.50, 0.75):
        raise RuntimeError("development fusion implementation contract changed")
    if module.MEAN_TOP5 != "mean_only_top5_ensemble":
        raise RuntimeError("hidden anchor name changed")
    return module


def validate_fresh_cache(cache_root: Path) -> dict:
    manifest_path = cache_root / "cache_manifest.json"
    if sha256_file(manifest_path) != FRESH_CACHE_MANIFEST_SHA256:
        raise RuntimeError("fresh cache manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config") or {}
    exact = {
        "task_id": FRESH_CACHE_TASK_ID,
        "rows": EXPECTED_ROWS,
        "num_layers": 30,
        "hidden_size": 3072,
        "labels_present_during_model_scoring": False,
        "performance_metrics_computed": False,
        "raw_token_layer_activations_persisted": False,
        "logits_persisted": False,
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"fresh cache manifest changed: {key}")
    config_exact = {
        "rows": EXPECTED_ROWS,
        "rows_per_language_label": EXPECTED_LANGUAGE_LABEL,
        "seed": 20260909,
        "max_length": 8192,
        "shard_size": 250,
        "model_id": "bigcode/starcoder2-3b",
        "model_revision": "733247c55e3f73af49ce8e9c7949bf14af205928",
        "dataset_id": "Poisoned-Chalice/ICSE-2027-public",
        "dataset_revision": "2ed5468723efa5457a3665782c6979ea4dbac7c2",
        "dtype": "float32",
        "compression": "npz_compressed",
    }
    for key, expected in config_exact.items():
        if config.get(key) != expected:
            raise RuntimeError(f"fresh cache config changed: {key}")
    shards = manifest.get("shards") or []
    if len(shards) != 20:
        raise RuntimeError("fresh cache shard count changed")
    if sum(int(item.get("bytes", -1)) for item in shards) != EXPECTED_SHARD_BYTES:
        raise RuntimeError("fresh cache declared total shard bytes changed")
    expected_start = 0
    for index, item in enumerate(shards):
        if int(item.get("shard_index", -1)) != index:
            raise RuntimeError("fresh cache shard index changed")
        if int(item.get("start_index", -1)) != expected_start:
            raise RuntimeError("fresh cache shard coverage is not contiguous")
        if int(item.get("rows", -1)) != 250:
            raise RuntimeError("fresh cache shard row count changed")
        if list(item.get("shape_per_variant") or []) != [250, 30, 3072]:
            raise RuntimeError("fresh cache shard shape changed")
        path = cache_root / str(item.get("path"))
        if not path.is_file() or path.stat().st_size != int(item.get("bytes", -1)):
            raise RuntimeError(f"fresh cache shard missing/size mismatch: {index}")
        if sha256_file(path) != item.get("sha256"):
            raise RuntimeError(f"fresh cache shard SHA-256 mismatch: {index}")
        expected_start = int(item.get("stop_index_exclusive", -1))
    if expected_start != EXPECTED_ROWS:
        raise RuntimeError("fresh cache shard coverage incomplete")

    selection_path = cache_root / "cohort_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    required_selection = {
        "task_id": FRESH_SELECTION_TASK_ID,
        "fresh_rows": EXPECTED_ROWS,
        "fresh_full_stage1_overlap": True,
        "fresh_current_hidden_overlap_rows": 0,
        "fresh_prior_1k_overlap_rows": 0,
        "fresh_prior_50_overlap_rows": 0,
        "labels_present_during_model_scoring": False,
        "fresh_cohort_set_sha256": FRESH_COHORT_SET_SHA256,
        "stage1_rows": EXPECTED_STAGE1_ROWS,
    }
    for key, expected in required_selection.items():
        if selection.get(key) != expected:
            raise RuntimeError(f"fresh cohort selection changed: {key}")
    cell_counts = selection.get("fresh_language_label_counts") or {}
    if len(cell_counts) != 10 or any(int(v) != EXPECTED_LANGUAGE_LABEL for v in cell_counts.values()):
        raise RuntimeError("fresh cohort selection language-label balance changed")

    run_manifest = json.loads((cache_root / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete":
        raise RuntimeError("fresh cache run manifest is not complete")
    if run_manifest.get("cohort_set_sha256") != FRESH_COHORT_SET_SHA256:
        raise RuntimeError("fresh cache run cohort hash changed")
    if run_manifest.get("automatic_compute_retries") != 0:
        raise RuntimeError("fresh cache retry contract changed")

    metadata_info = manifest.get("metadata") or {}
    sample_meta_path = cache_root / str(metadata_info.get("sample_metadata"))
    cohort_path = cache_root / str(metadata_info.get("cohort_manifest"))
    if sha256_file(sample_meta_path) != metadata_info.get("sample_metadata_sha256"):
        raise RuntimeError("fresh sample metadata hash mismatch")
    if sha256_file(cohort_path) != metadata_info.get("cohort_manifest_sha256"):
        raise RuntimeError("fresh cohort manifest hash mismatch")
    sample_rows = [json.loads(x) for x in sample_meta_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    cohort_rows = [json.loads(x) for x in cohort_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(sample_rows) != EXPECTED_ROWS or len(cohort_rows) != EXPECTED_ROWS:
        raise RuntimeError("fresh metadata coverage changed")
    if [x["sample_id"] for x in sample_rows] != [x["sample_id"] for x in cohort_rows]:
        raise RuntimeError("fresh metadata/cohort order mismatch")
    if len({x["sample_id"] for x in sample_rows}) != EXPECTED_ROWS:
        raise RuntimeError("fresh metadata duplicate sample ids")
    helper_fallback = [
        {
            "execution_index": int(x["execution_index"]),
            "language": str(x["language"]),
            "sample_id": str(x["sample_id"]),
            "seq_len": int(x["seq_len"]),
        }
        for x in sample_rows
        if x.get("helper_ast_backend") != "tree_sitter"
    ]
    return {
        "cache_manifest_sha256": FRESH_CACHE_MANIFEST_SHA256,
        "cohort_set_sha256": FRESH_COHORT_SET_SHA256,
        "rows": EXPECTED_ROWS,
        "shard_count": 20,
        "total_shard_bytes": EXPECTED_SHARD_BYTES,
        "helper_ast_fallback_rows": helper_fallback,
    }


def load_hidden_oof(hidden_eval_dir: Path) -> tuple[pd.DataFrame, dict]:
    run_manifest_path = hidden_eval_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete":
        raise RuntimeError("fresh hidden evaluation not complete")
    if run_manifest.get("task_id") != FRESH_HIDDEN_EVAL_TASK_ID:
        raise RuntimeError("fresh hidden evaluation task identity changed")
    if run_manifest.get("cache_manifest_sha256") != FRESH_CACHE_MANIFEST_SHA256:
        raise RuntimeError("fresh hidden evaluation used wrong cache")
    if run_manifest.get("expected_cache_manifest_sha256") != FRESH_CACHE_MANIFEST_SHA256:
        raise RuntimeError("fresh hidden evaluation cache expectation changed")
    if run_manifest.get("rows") != EXPECTED_ROWS or run_manifest.get("outer_oof_coverage") is not True:
        raise RuntimeError("fresh hidden evaluation OOF coverage incomplete")

    path = hidden_eval_dir / "oof_predictions.csv"
    frame = pd.read_csv(path)
    required = {
        "sample_id", "language", "label",
        "mean_only_top5_ensemble",
        "lumia_caller_literal_top5_ensemble",
        "lumia_helper_language_aware_top5_ensemble",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(f"fresh hidden OOF columns missing: {sorted(required-set(frame.columns))}")
    if len(frame) != EXPECTED_ROWS or frame.sample_id.duplicated().any():
        raise RuntimeError("fresh hidden OOF coverage changed")
    frame["label"] = frame.label.astype(int)
    counts = frame.groupby(["language", "label"]).size()
    if len(counts) != 10 or not counts.eq(EXPECTED_LANGUAGE_LABEL).all():
        raise RuntimeError("fresh hidden OOF balance changed")
    return frame.reset_index(drop=True), {
        "oof_predictions_sha256": sha256_file(path),
        "hidden_run_manifest_sha256": sha256_file(run_manifest_path),
        "hidden_metrics_sha256": sha256_file(hidden_eval_dir / "metrics.json"),
        "hidden_decision_sha256": sha256_file(hidden_eval_dir / "decision.json"),
        "hidden_fold_selection_sha256": sha256_file(hidden_eval_dir / "fold_selection.json"),
    }


def fusion_confirmation_decision(base, frame: pd.DataFrame, fusion: dict) -> dict:
    hidden = fusion["metrics"][base.MEAN_TOP5]
    candidates = {}
    for weight in FROZEN_FUSION_WEIGHTS:
        name = f"rank_fusion_hidden_{weight:.2f}"
        metrics = fusion["metrics"][name]
        bootstrap = fusion["paired_bootstrap"][f"{name}_minus_{base.MEAN_TOP5}"]
        tpr_delta = float(metrics["tpr_at_0.01_fpr"] - hidden["tpr_at_0.01_fpr"])
        auc_delta = float(metrics["auc"] - hidden["auc"])
        tpr_lower = float(bootstrap["delta"]["tpr_at_0.01_fpr"]["lower_95"])
        tpr_upper = float(bootstrap["delta"]["tpr_at_0.01_fpr"]["upper_95"])
        auc_lower = float(bootstrap["delta"]["auc"]["lower_95"])
        auc_upper = float(bootstrap["delta"]["auc"]["upper_95"])
        gates = {
            "tpr_at_1pct_delta_at_least_0_01": bool(tpr_delta >= FUSION_TPR_DELTA_MIN),
            "paired_bootstrap_tpr_delta_lower_95_nonnegative": bool(
                tpr_lower >= FUSION_TPR_BOOTSTRAP_LOWER_MIN
            ),
            "auc_delta_at_least_minus_0_005": bool(auc_delta >= FUSION_AUC_DELTA_MIN),
        }
        candidates[name] = {
            "hidden_rank_weight": weight,
            "metrics": metrics,
            "delta_vs_hidden": {
                "auc": auc_delta,
                "auc_lower_95": auc_lower,
                "auc_upper_95": auc_upper,
                "tpr_at_0.01_fpr": tpr_delta,
                "tpr_at_0.01_fpr_lower_95": tpr_lower,
                "tpr_at_0.01_fpr_upper_95": tpr_upper,
            },
            "promotion_gates": gates,
            "passes_all_promotion_gates": bool(all(gates.values())),
        }
    return {
        "primary": PRIMARY_FUSION,
        "secondary": SECONDARY_FUSION,
        "hidden_reference": base.MEAN_TOP5,
        "hidden_metrics": hidden,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "no_additional_weights_evaluated": True,
    }


def target_specificity_summary(base, controls: dict) -> dict:
    random_oof = controls["random_outer_oof"]
    hidden = random_oof["hidden_mean_top5"]
    c0 = random_oof["C0_static_logistic"]
    c1 = random_oof["C1_hashed_char_token_ngram_logistic"]
    by_auc = "C0_static_logistic" if c0["auc"] >= c1["auc"] else "C1_hashed_char_token_ngram_logistic"
    strongest = random_oof[by_auc]
    paired_name = "hidden_minus_C0" if by_auc.startswith("C0") else "hidden_minus_C1"
    return {
        "hidden": hidden,
        "C0": c0,
        "C1": c1,
        "strongest_control_by_auc": by_auc,
        "hidden_minus_strongest_control_auc": float(hidden["auc"] - strongest["auc"]),
        "paired_bootstrap_hidden_minus_strongest_control": controls["paired_bootstrap"][paired_name],
        "strict_content_group_cv": controls["strict_content_group_cv"],
        "interpretation_boundary": (
            "Same-cohort target-specificity diagnostic only; this is not cross-model transfer evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-eval-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--stage1-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)

    base = load_base_audit()
    cache_diag = validate_fresh_cache(args.cache_dir.resolve())
    hidden, hidden_identity = load_hidden_oof(args.hidden_eval_dir.resolve())

    # Verify the fresh OOF order against the sealed fresh-cache execution order.
    cache_manifest = json.loads((args.cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    cache_metadata = base.load_cache_metadata(args.cache_dir.resolve(), cache_manifest)
    if cache_metadata.sample_id.tolist() != hidden.sample_id.tolist():
        raise RuntimeError("fresh hidden OOF order differs from cache metadata")
    if cache_metadata.language.tolist() != hidden.language.tolist():
        raise RuntimeError("fresh hidden OOF language order differs from cache metadata")

    # Reproduce the already-frozen Stage1 branch from all 10k cached features.
    stage1_frame = base.load_stage1_train(args.stage1_root.resolve())
    stage1_score = base.stage1_interaction_oof(stage1_frame)
    stage1_scored = stage1_frame[["sample_id", "language", "label"]].copy()
    stage1_scored["stage1_plus_length_interactions"] = stage1_score
    stage1_reproduction = base.low_fpr_metrics(stage1_scored.label, stage1_score)

    # Narrow the development implementation to exactly the two pre-frozen fusion formulas.
    base.FUSION_HIDDEN_WEIGHTS = FROZEN_FUSION_WEIGHTS
    fusion, common = base.fusion_audit(stage1_scored, hidden)
    if fusion["intersection_rows"] != EXPECTED_ROWS:
        raise RuntimeError(f"fresh Stage1/hidden intersection must be 5000, got {fusion['intersection_rows']}")
    if set(name for name in fusion["metrics"] if name.startswith("rank_fusion_hidden_")) != {
        PRIMARY_FUSION, SECONDARY_FUSION
    }:
        raise RuntimeError("unexpected fusion formula evaluated")
    fusion["fusion_candidates_are_development_only"] = False
    fusion["fresh_confirmation_only"] = True
    fusion["frozen_before_fresh_result"] = True
    fusion["cross_fitted_two_score_logistic_not_run"] = True

    # Unchanged content-only controls on the exact same 5k fresh cohort.
    content_frame = base.load_public_content(hidden)
    controls, static_table, control_scores = base.control_audit(content_frame)
    controls["fresh_confirmation_only"] = True
    controls["frozen_before_fresh_result"] = True

    # Portable scalar extraction: raw 3072-d hidden coordinates never enter the scalar learner.
    base.CACHE_MANIFEST_SHA256 = FRESH_CACHE_MANIFEST_SHA256
    scalar_table, scalar_diag = base.extract_portable_scalars(args.cache_dir.resolve(), hidden)
    portable, portable_scores = base.portable_audit(hidden, scalar_table, scalar_diag)
    portable["development_only"] = False
    portable["fresh_confirmation_only"] = True
    portable["frozen_primary"] = "mean"
    portable["frozen_primary_feature_count"] = 96
    portable["caller_helper_are_ablation_only"] = True
    if int(portable["diagnostics"]["scalar_features_per_pooling_variant"]) != 96:
        raise RuntimeError("portable scalar feature count changed")

    fusion_decision = fusion_confirmation_decision(base, hidden, fusion)
    target_specificity = target_specificity_summary(base, controls)
    portable_primary = portable["metrics"]["mean"]

    hidden_metrics = json.loads((args.hidden_eval_dir / "metrics.json").read_text(encoding="utf-8"))
    hidden_decision = json.loads((args.hidden_eval_dir / "decision.json").read_text(encoding="utf-8"))
    fold_selection = json.loads((args.hidden_eval_dir / "fold_selection.json").read_text(encoding="utf-8"))

    decision = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "complete",
        "freshness_scope": {
            "fresh_for_hidden_probe_fusion_and_portable_scalar": True,
            "unseen_by_stage1_branch": False,
            "validation_rows_used": False,
            "public_lb_feedback_used": False,
            "competition_submission": False,
        },
        "cache_provenance": cache_diag,
        "hidden_anchor": {
            "evaluation_task_id": FRESH_HIDDEN_EVAL_TASK_ID,
            "metrics": hidden_metrics,
            "decision": hidden_decision,
            "fold_selection": fold_selection,
            "identity": hidden_identity,
        },
        "fusion_confirmation": fusion_decision,
        "target_specificity": target_specificity,
        "portable_scalar_confirmation": {
            "primary": "mean",
            "feature_count": 96,
            "metrics": portable_primary,
            "paired_bootstrap_primary_vs_raw_hidden_probe": portable[
                "paired_bootstrap_primary_vs_raw_hidden_probe"
            ],
            "caller_helper_are_ablation_only": True,
            "no_threshold_selected_from_fresh_result": True,
        },
        "anti_posthoc": {
            "fusion_weights_evaluated": [0.75, 0.50],
            "additional_fusion_weights_evaluated": False,
            "learned_two_score_stack_evaluated": False,
            "portable_primary_pooling": "mean",
            "new_scalar_features_added_after_fresh_result": False,
            "layer_choice_from_fresh_outer_holdout": False,
        },
        "claim_boundary": (
            "Fresh same-model confirmation on StarCoder2-3B public-train membership proxy; "
            "no Stage2 cross-model transfer or arbitrary-model training-membership claim."
        ),
    }

    json_dump(out / "fusion_confirmation.json", fusion)
    json_dump(out / "content_control_confirmation.json", controls)
    json_dump(out / "portable_scalar_confirmation.json", portable)
    json_dump(out / "confirmation_decision.json", decision)
    common.to_parquet(out / "fusion_scores.parquet", index=False)
    control_scores.to_parquet(out / "content_control_scores.parquet", index=False)
    scalar_table.to_parquet(out / "portable_scalar_table.parquet", index=False)
    portable_scores.to_parquet(out / "portable_scalar_scores.parquet", index=False)
    static_table.to_parquet(out / "static_feature_table.parquet", index=False)

    # Preserve the sealed hidden-anchor evaluation as small reproducibility outputs.
    for src, dst in (
        ("metrics.json", "hidden_metrics.json"),
        ("decision.json", "hidden_decision.json"),
        ("fold_selection.json", "hidden_fold_selection.json"),
        ("run_manifest.json", "hidden_run_manifest.json"),
        ("oof_predictions.csv", "hidden_oof_predictions.csv"),
    ):
        shutil.copy2(args.hidden_eval_dir / src, out / dst)

    persistent = sorted(path.name for path in out.iterdir()) + ["run_manifest.json"]
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "complete",
        "rows": EXPECTED_ROWS,
        "cache_manifest_sha256": FRESH_CACHE_MANIFEST_SHA256,
        "cohort_set_sha256": FRESH_COHORT_SET_SHA256,
        "stage1_rows": EXPECTED_STAGE1_ROWS,
        "stage1_reproduction": stage1_reproduction,
        "fusion_intersection_rows": int(fusion["intersection_rows"]),
        "frozen_fusion_weights": [0.75, 0.50],
        "portable_primary": "mean_96_feature_scalar",
        "content_controls": [
            "C0_static_logistic_C0.2",
            "C1_hashed_char3-5_token1-2_liblinear_C1",
            "token5gram_SimHash64_HammingLE6_group_diagnostic",
        ],
        "hidden_oof_sha256": hidden_identity["oof_predictions_sha256"],
        "competition_submission": False,
        "validation_rows_used": False,
        "public_lb_feedback_used": False,
        "persistent_outputs": persistent,
    }
    json_dump(out / "run_manifest.json", manifest)
    print(json.dumps({
        "task_id": TASK_ID,
        "status": "complete",
        "rows": EXPECTED_ROWS,
        "fusion_intersection_rows": int(fusion["intersection_rows"]),
        "fusion_candidates": {
            name: value["passes_all_promotion_gates"]
            for name, value in fusion_decision["candidates"].items()
        },
        "portable_mean_auc": portable_primary["auc"],
        "target_specificity_auc_gap": target_specificity["hidden_minus_strongest_control_auc"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
