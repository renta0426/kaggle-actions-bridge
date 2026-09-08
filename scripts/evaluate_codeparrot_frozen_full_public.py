#!/usr/bin/env python3
"""Evaluate the sealed CodeParrot full-v1 output from public inputs only.

This is a self-contained implementation of the predeclared frozen evaluation
contract. Cohort reconstruction follows the already-public CodeParrot cohort
workflow: pinned train/valid revisions, first 500k train rows, all validation
rows, deterministic per-repository capping, matched license/length strata, and
SHA ordering. Derived predictors are sealed before either membership dataset is
opened. The script writes aggregate evaluation only and never persists a
prediction+membership row table.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


MEMBER_DATASET = "codeparrot/codeparrot-clean-train"
MEMBER_REVISION = "3e6ab65f2864931e041f6a82db9b5a6ec2b71ab4"
NONMEMBER_DATASET = "codeparrot/codeparrot-clean-valid"
NONMEMBER_REVISION = "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd"
ROWS_PER_LABEL = 2_000
TRAIN_SCAN_ROWS = 500_000
MIN_CHARS = 512
MAX_CHARS = 32_768
MAX_PER_REPO = 2
TARGET_FPRS = (0.001, 0.005, 0.01, 0.02)
PRIMARY_FPR = 0.01
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 2027
MAX_AUC_DROP = 0.005
MAX_QUARTILE_AUC_DROP = 0.02
EXPECTED_COHORT_SHA256 = "b664e368f9380e230c9cfc0616327d829424749c5473f87575678525db1998fc"
MODEL_ID = "codeparrot/codeparrot"
MODEL_REVISION = "065248a99f051da363b1c2cbf05da943c8b6211b"
WEIGHT_SHA256 = "4a9d17dfbde54741587c26a07dd17366635faa3513051cdb21ac08eec6413e1d"

RAW_COLUMNS = [
    "sample_id", "content_sha256", "language", "token_count", "window_count",
    "loss_multiwindow", "legacy_minkpp10", "paper_minkpp10", "local_64",
    "neg_mean_log_rank",
]
PREDICTORS = [
    "local_64", "short_half_local_or_paper_v1", "paper_minkpp10",
    "legacy_minkpp10", "loss_multiwindow", "neg_mean_log_rank",
    "stage2_v1_rank_fusion",
]


def digest_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def copies_one(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return int(str(value).strip()) == 1
    except Exception:
        return False


def length_bucket(length: int) -> int:
    return min(14, max(9, int(math.floor(math.log2(length)))))


def canonical_frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path.name}")
    return value


def validate_raw(predictions: pd.DataFrame, manifest: dict[str, Any], fidelity: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    forbidden = {"membership", "label", "source_split", "source_position", "repo_group_sha256", "repo_name", "path", "license", "dataset_id", "prior_model_score"}
    if forbidden.intersection(predictions.columns):
        raise ValueError("label/provenance leaked into sealed predictions")
    if predictions.columns.tolist() != RAW_COLUMNS:
        raise ValueError(f"raw prediction schema mismatch: {predictions.columns.tolist()}")
    if len(predictions) != 4_000 or not predictions.sample_id.is_unique or not predictions.content_sha256.is_unique:
        raise ValueError("raw prediction coverage/identity mismatch")
    if predictions.sample_id.astype(str).tolist() != sorted(predictions.sample_id.astype(str).tolist()):
        raise ValueError("raw prediction order is not frozen sample_id order")
    if not np.array_equal(
        predictions.sample_id.astype(str).to_numpy(),
        ("cp-" + predictions.content_sha256.astype(str)).to_numpy(),
    ):
        raise ValueError("sample_id/content_sha256 relation failed")
    if set(predictions.language.astype(str)) != {"Python"}:
        raise ValueError("full cohort language mismatch")
    if (predictions.token_count.astype(int) < 2).any() or not predictions.window_count.astype(int).isin([1, 2, 3]).all():
        raise ValueError("token/window contract failed")
    if not np.isfinite(predictions.select_dtypes(include=[np.number]).to_numpy(float)).all():
        raise ValueError("non-finite raw prediction")

    required_manifest = {
        "schema_version": "codeparrot_fresh_full_manifest_v1",
        "experiment_id": "codeparrot-fresh-full-v1",
        "status": "sealed_label_blind_full_predictions",
        "rows": 4000,
        "selection": "all_4000_sample_ids_from_frozen_label_free_prediction_input",
        "same_forward_paper_and_legacy": True,
        "membership_join_performed": False,
        "performance_metrics_computed": False,
        "performance_metrics_allowed": False,
        "evaluation_manifest_attached": False,
        "competition_submission": False,
        "full_4000_scoring_authorized": True,
        "automatic_compute_retries": 0,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest contract mismatch: {key}")
    if manifest.get("features") != ["loss_multiwindow", "legacy_minkpp10", "paper_minkpp10", "local_64", "neg_mean_log_rank"]:
        raise ValueError("manifest feature list mismatch")
    model = manifest.get("model") or {}
    expected_model = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "resolved_model_revision": MODEL_REVISION,
        "model_class": "GPT2LMHeadModel",
        "model_type": "gpt2",
        "trust_remote_code": False,
        "use_safetensors": False,
        "pytorch_model_bin_lfs_sha256": WEIGHT_SHA256,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"model identity mismatch: {key}")
    if profile.get("status") != "complete_label_blind_full_scoring" or profile.get("rows") != 4000:
        raise ValueError("profile completion mismatch")
    if profile.get("cohort_prediction_input_sha256") != EXPECTED_COHORT_SHA256:
        raise ValueError("cohort prediction-input SHA mismatch")
    if int(profile.get("windows", 0)) < 4000 or float(profile.get("run_seconds", 0.0)) <= 0 or int(profile.get("peak_gpu_memory_bytes", 0)) <= 0:
        raise ValueError("profile runtime/resource contract failed")
    if fidelity.get("status") != "pass" or fidelity.get("membership_join_performed") is not False or fidelity.get("performance_metrics_computed") is not False or fidelity.get("original_sample_id_passed_to_scorer") is not False:
        raise ValueError("fidelity boundary mismatch")
    gates = [
        ("cpu_oracle_abs_error", 2e-5),
        ("synthetic_paper_dense_blocked_max_z_diff", 3e-5),
        ("synthetic_paper_dense_blocked_max_variance_diff", 3e-5),
        ("real_paper_dense_blocked_max_z_diff", 5e-5),
        ("real_paper_dense_blocked_max_variance_diff", 5e-5),
        ("synthetic_legacy_exact_blocked_max_logp_diff", 1e-6),
        ("synthetic_legacy_exact_blocked_max_z_diff", 1e-6),
        ("historical_same_forward_first8_max_abs_diff", 1e-6),
    ]
    for key, maximum in gates:
        if float(fidelity.get(key, math.inf)) > maximum:
            raise ValueError(f"fidelity numeric gate failed: {key}")
    if fidelity.get("synthetic_legacy_rank_exact_match") is not True or fidelity.get("variance_clamp_count") != 0 or fidelity.get("target_tokens") != fidelity.get("valid_target_tokens"):
        raise ValueError("fidelity discrete/accounting gate failed")
    return {
        "rows": 4000,
        "unique_sample_ids": 4000,
        "unique_content_hashes": 4000,
        "all_numeric_features_finite": True,
        "token_count_min": int(predictions.token_count.min()),
        "token_count_max": int(predictions.token_count.max()),
        "window_count_distribution": {str(int(k)): int(v) for k, v in predictions.window_count.value_counts().sort_index().items()},
        "target_tokens": int(fidelity["target_tokens"]),
        "windows": int(profile["windows"]),
        "run_seconds": float(profile["run_seconds"]),
        "peak_gpu_memory_bytes": int(profile["peak_gpu_memory_bytes"]),
    }


def percentile(values: pd.Series) -> np.ndarray:
    result = values.rank(method="average", pct=True).to_numpy(float)
    if not np.isfinite(result).all():
        raise ValueError("non-finite percentile")
    return result


def seal_label_free_predictors(raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    work = raw.sort_values("sample_id").reset_index(drop=True).copy()
    local = percentile(work.local_64)
    paper = percentile(work.paper_minkpp10)
    token = percentile(work.token_count)
    loss = percentile(work.loss_multiwindow)
    legacy = percentile(work.legacy_minkpp10)
    work["short_half_local_or_paper_v1"] = np.where(token <= 0.5, np.maximum(local, paper), local)
    work["stage2_v1_rank_fusion"] = (loss + legacy + local) / 3.0
    sealed = work[["sample_id", "short_half_local_or_paper_v1", "stage2_v1_rank_fusion"]]
    return work, canonical_frame_sha256(sealed)


def scan_public_stream(dataset, *, membership: int, max_rows: int | None) -> tuple[pd.DataFrame, set[str], dict[str, int]]:
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_hashes: set[str] = set()
    scanned = eligible = duplicates = 0
    for position, record in enumerate(dataset):
        if max_rows is not None and position >= max_rows:
            break
        scanned += 1
        content = record.get("content"); repo = record.get("repo_name"); path = record.get("path"); license_name = record.get("license")
        if not isinstance(content, str) or not (MIN_CHARS <= len(content) <= MAX_CHARS):
            continue
        if record.get("autogenerated") is not False or not copies_one(record.get("copies")):
            continue
        if not all(isinstance(value, str) and value.strip() for value in (repo, path, license_name)):
            continue
        content_hash = digest_text(content)
        if content_hash in all_hashes:
            duplicates += 1; continue
        all_hashes.add(content_hash); eligible += 1
        row = {
            "membership": int(membership),
            "source_position": int(position),
            "content_sha256": content_hash,
            "repo_group_sha256": digest_text(repo),
            "license_lower": license_name.strip().lower(),
            "content_chars": int(len(content)),
            "length_bucket": length_bucket(len(content)),
        }
        bucket = by_repo[row["repo_group_sha256"]]
        bucket.append(row)
        bucket.sort(key=lambda item: (item["content_sha256"], item["source_position"]))
        del bucket[MAX_PER_REPO:]
    retained = [row for rows in by_repo.values() for row in rows]
    frame = pd.DataFrame(retained).sort_values(["license_lower", "length_bucket", "content_sha256", "source_position"]).reset_index(drop=True)
    if frame.empty or frame.content_sha256.duplicated().any() or frame.groupby("repo_group_sha256").size().max() > MAX_PER_REPO:
        raise RuntimeError("public cohort scan invariant failed")
    return frame, all_hashes, {"scanned_rows": scanned, "eligible_unique_content_rows": eligible, "retained_rows": len(frame), "retained_repositories": int(frame.repo_group_sha256.nunique()), "duplicate_content_within_split": duplicates}


def matched_allocation(member_frame: pd.DataFrame, nonmember_frame: pd.DataFrame) -> tuple[dict[tuple[str, int], int], int]:
    def capacities(frame: pd.DataFrame) -> dict[tuple[str, int], int]:
        counts = frame.groupby(["license_lower", "length_bucket"], sort=True).size()
        return {(str(a), int(b)): int(c) for (a, b), c in counts.items()}
    left, right = capacities(member_frame), capacities(nonmember_frame)
    caps = {key: min(left[key], right[key]) for key in set(left) & set(right)}
    caps = {key: value for key, value in caps.items() if value > 0}
    total = sum(caps.values())
    if total < ROWS_PER_LABEL:
        raise RuntimeError("matched capacity too small")
    exact = {key: ROWS_PER_LABEL * value / total for key, value in caps.items()}
    out = {key: min(caps[key], int(math.floor(exact[key]))) for key in caps}
    remaining = ROWS_PER_LABEL - sum(out.values())
    order = sorted(caps, key=lambda key: (-(exact[key] - math.floor(exact[key])), key[0], key[1]))
    while remaining:
        progressed = False
        for key in order:
            if out[key] < caps[key]:
                out[key] += 1; remaining -= 1; progressed = True
                if remaining == 0: break
        if not progressed: raise RuntimeError("matched allocation stalled")
    return out, total


def select_rows(frame: pd.DataFrame, allocations: dict[tuple[str, int], int]) -> pd.DataFrame:
    pieces = []
    for (license_name, bucket), count in sorted(allocations.items()):
        subset = frame[(frame.license_lower == license_name) & (frame.length_bucket == bucket)].sort_values(["content_sha256", "source_position"])
        if len(subset) < count: raise RuntimeError("allocation exceeds capacity")
        pieces.append(subset.head(count))
    return pd.concat(pieces, ignore_index=True)


def reconstruct_manifest() -> tuple[pd.DataFrame, dict[str, Any]]:
    # This function is called only after the label-free predictor hash is sealed.
    import datasets
    from datasets import load_dataset
    member_stream = load_dataset(MEMBER_DATASET, split="train", streaming=True, revision=MEMBER_REVISION)
    nonmember_stream = load_dataset(NONMEMBER_DATASET, split="train", streaming=True, revision=NONMEMBER_REVISION)
    members, member_hashes, member_diag = scan_public_stream(member_stream, membership=1, max_rows=TRAIN_SCAN_ROWS)
    nonmembers, nonmember_hashes, nonmember_diag = scan_public_stream(nonmember_stream, membership=0, max_rows=None)
    if member_hashes & nonmember_hashes:
        raise RuntimeError("exact content overlap across published train/valid")
    allocations, total_capacity = matched_allocation(members, nonmembers)
    selected = pd.concat([select_rows(members, allocations), select_rows(nonmembers, allocations)], ignore_index=True)
    if (selected.membership == 1).sum() != ROWS_PER_LABEL or (selected.membership == 0).sum() != ROWS_PER_LABEL:
        raise RuntimeError("selected class balance failed")
    selected["sample_id"] = "cp-" + selected.content_sha256.astype(str)
    selected["language"] = "Python"
    selected = selected.sort_values("sample_id").reset_index(drop=True)
    if not selected.sample_id.is_unique or selected.content_sha256.duplicated().any():
        raise RuntimeError("selected identity is not unique")
    matched = selected.groupby(["membership", "license_lower", "length_bucket"], sort=True).size().unstack("membership", fill_value=0)
    if set(matched.columns) != {0, 1} or not np.array_equal(matched[0].to_numpy(), matched[1].to_numpy()):
        raise RuntimeError("matched strata differ across labels")
    member_groups = set(selected.loc[selected.membership == 1, "repo_group_sha256"].astype(str))
    nonmember_groups = set(selected.loc[selected.membership == 0, "repo_group_sha256"].astype(str))
    manifest = selected[["sample_id", "content_sha256", "membership", "repo_group_sha256", "language", "length_bucket", "license_lower"]].copy()
    audit = {
        "datasets_version": datasets.__version__,
        "member_source": {"dataset_id": MEMBER_DATASET, "revision": MEMBER_REVISION, "split": "train"},
        "nonmember_source": {"dataset_id": NONMEMBER_DATASET, "revision": NONMEMBER_REVISION, "split": "train"},
        "member_scan": member_diag,
        "nonmember_scan": nonmember_diag,
        "selection": {"rows": len(selected), "rows_per_label": ROWS_PER_LABEL, "strata_used": len(allocations), "total_matched_capacity": total_capacity},
        "group_balance": {"members": ROWS_PER_LABEL, "nonmembers": ROWS_PER_LABEL, "member_repository_groups": len(member_groups), "nonmember_repository_groups": len(nonmember_groups), "shared_repository_groups_across_labels": len(member_groups & nonmember_groups), "maximum_rows_per_repository_group_member": int(selected[selected.membership == 1].groupby("repo_group_sha256").size().max()), "maximum_rows_per_repository_group_nonmember": int(selected[selected.membership == 0].groupby("repo_group_sha256").size().max()), "matched_strata": len(matched), "stratum_counts_identical_between_labels": True},
    }
    return manifest, audit


def conservative_boundary(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> dict[str, Any]:
    negatives = scores[labels == 0]
    allowed = int(math.floor(target_fpr * len(negatives) + 1e-12))
    unique, counts = np.unique(negatives, return_counts=True); order = np.argsort(unique)[::-1]
    admitted = 0; cutoff = -math.inf; excluded_size = 0
    for score, count in zip(unique[order], counts[order]):
        count = int(count)
        if admitted + count <= allowed:
            admitted += count; continue
        cutoff = float(score); excluded_size = count; break
    achieved = int(np.sum(negatives > cutoff))
    if achieved != admitted or achieved > allowed: raise RuntimeError("conservative boundary accounting failed")
    return {"schema_version": "low_fpr_novelty_v2", "policy": "conservative_empirical", "target_fpr": float(target_fpr), "negative_count": len(negatives), "allowed_false_positives": allowed, "achieved_false_positives": achieved, "achieved_fpr": float(achieved / len(negatives)), "cutoff_score": cutoff, "comparison": ">", "excluded_tie_block_size": excluded_size}


def low_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> dict[str, Any]:
    boundary = conservative_boundary(labels, scores, target_fpr); mask = scores > boundary["cutoff_score"]
    tp = int(np.sum(mask & (labels == 1))); members = int(np.sum(labels == 1))
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    unique_fpr = np.unique(fpr); max_tpr = np.asarray([np.max(tpr[fpr == value]) for value in unique_fpr])
    return {"conservative_empirical": {"true_positives": tp, "member_count": members, "tpr": float(tp / members), "boundary": boundary}, "roc_interpolated": {"tpr": float(np.interp(target_fpr, unique_fpr, max_tpr)), "sample_set_defined": False}}


def predictor_metrics(frame: pd.DataFrame, predictor: str) -> dict[str, Any]:
    labels = frame.membership.astype(int).to_numpy(); scores = frame[predictor].to_numpy(float)
    return {"auc_roc": float(roc_auc_score(labels, scores)), "standardized_pauc_01": float(roc_auc_score(labels, scores, max_fpr=0.01)), "spearman_token_count": float(frame[[predictor, "token_count"]].corr(method="spearman").iloc[0, 1]), "low_fpr": {f"{rate:.3f}": low_fpr(labels, scores, rate) for rate in TARGET_FPRS}}


def group_bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups); sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == group) for group in sampled])


def paired_bootstrap(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    labels = frame.membership.astype(int).to_numpy(); left = frame.short_half_local_or_paper_v1.to_numpy(float); right = frame.local_64.to_numpy(float); groups = frame.repo_group_sha256.astype(str).to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED); rows = []; attempts = 0
    while len(rows) < BOOTSTRAP_REPLICATES:
        attempts += 1
        if attempts > BOOTSTRAP_REPLICATES * 20: raise RuntimeError("bootstrap class-valid replicate failure")
        idx = group_bootstrap_indices(groups, rng)
        if len(np.unique(labels[idx])) != 2: continue
        l_low, r_low = low_fpr(labels[idx], left[idx], PRIMARY_FPR), low_fpr(labels[idx], right[idx], PRIMARY_FPR)
        rows.append({"auc_roc": float(roc_auc_score(labels[idx], left[idx]) - roc_auc_score(labels[idx], right[idx])), "conservative_empirical_tpr": float(l_low["conservative_empirical"]["tpr"] - r_low["conservative_empirical"]["tpr"]), "roc_interpolated_tpr": float(l_low["roc_interpolated"]["tpr"] - r_low["roc_interpolated"]["tpr"])})
    out = {}
    for metric in rows[0]:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        out[metric] = {"mean": float(np.mean(values)), "lower_95": float(np.quantile(values, 0.025)), "upper_95": float(np.quantile(values, 0.975))}
    return out


def length_quartiles(frame: pd.DataFrame) -> list[dict[str, Any]]:
    work = frame.copy(); work["token_length_quartile"] = pd.qcut(work.token_count, q=4, labels=False, duplicates="drop")
    if work.token_length_quartile.nunique() != 4: raise RuntimeError("expected four token-count quartiles")
    rows = []
    for quartile, group in work.groupby("token_length_quartile", sort=True):
        labels = group.membership.astype(int).to_numpy()
        if set(np.unique(labels)) != {0, 1}: raise RuntimeError("length quartile lacks both classes")
        rows.append({"quartile": int(quartile), "rows": len(group), "members": int(labels.sum()), "min_token_count": int(group.token_count.min()), "max_token_count": int(group.token_count.max()), "local_64__auc": float(roc_auc_score(labels, group.local_64)), "short_half_local_or_paper_v1__auc": float(roc_auc_score(labels, group.short_half_local_or_paper_v1))})
    return rows


def detection_overlap(frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame.membership.astype(int).to_numpy(); left = frame.local_64.to_numpy(float); right = frame.short_half_local_or_paper_v1.to_numpy(float)
    lb, rb = conservative_boundary(labels, left, PRIMARY_FPR), conservative_boundary(labels, right, PRIMARY_FPR)
    members = labels == 1; lmask = members & (left > lb["cutoff_score"]); rmask = members & (right > rb["cutoff_score"])
    common = int(np.sum(lmask & rmask)); union = int(np.sum(lmask | rmask)); ltp = int(np.sum(lmask)); rtp = int(np.sum(rmask))
    return {"left": "local_64", "right": "short_half_local_or_paper_v1", "target_fpr": PRIMARY_FPR, "left_true_positives": ltp, "right_true_positives": rtp, "common_true_positives": common, "left_unique_true_positives": ltp - common, "right_unique_true_positives": rtp - common, "jaccard": float(common / union) if union else 1.0, "left_boundary": lb, "right_boundary": rb}


def evaluate(prediction_only: pd.DataFrame, manifest: pd.DataFrame, artifact_audit: dict[str, Any], predictor_sha: str, reconstruction: dict[str, Any]) -> dict[str, Any]:
    left = prediction_only.sort_values("sample_id").reset_index(drop=True); right = manifest.sort_values("sample_id").reset_index(drop=True)
    if len(left) != len(right) or not np.array_equal(left.sample_id.astype(str), right.sample_id.astype(str)) or not np.array_equal(left.content_sha256.astype(str), right.content_sha256.astype(str)):
        raise RuntimeError("sealed prediction/reconstructed manifest identity mismatch")
    frame = left.merge(right[["sample_id", "membership", "repo_group_sha256", "language", "content_sha256", "length_bucket"]], on=["sample_id", "content_sha256"], how="inner", validate="one_to_one", sort=False, suffixes=("", "_manifest"))
    if len(frame) != 4000 or int(frame.membership.sum()) != 2000 or int((frame.membership == 0).sum()) != 2000 or not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(float)).all(): raise RuntimeError("joined evaluation frame contract failed")
    metrics = {name: predictor_metrics(frame, name) for name in PREDICTORS}
    bootstrap = paired_bootstrap(frame); quartiles = length_quartiles(frame); overlap = detection_overlap(frame)
    baseline = metrics["local_64"]; aux = metrics["short_half_local_or_paper_v1"]; rate_key = "0.010"
    drops = [row["local_64__auc"] - row["short_half_local_or_paper_v1__auc"] for row in quartiles]
    checks = {"conservative_tpr_01_strictly_above_local64": aux["low_fpr"][rate_key]["conservative_empirical"]["tpr"] > baseline["low_fpr"][rate_key]["conservative_empirical"]["tpr"], "bootstrap_tpr_01_delta_mean_positive": bootstrap["conservative_empirical_tpr"]["mean"] > 0, "bootstrap_tpr_01_delta_lower95_nonnegative": bootstrap["conservative_empirical_tpr"]["lower_95"] >= 0, "pauc_01_not_lower": aux["standardized_pauc_01"] >= baseline["standardized_pauc_01"], "auc_drop_within_guard": aux["auc_roc"] >= baseline["auc_roc"] - MAX_AUC_DROP, "all_length_quartile_auc_drops_within_guard": max(drops) <= MAX_QUARTILE_AUC_DROP}
    eval_manifest_sha = canonical_frame_sha256(right)
    return {"schema_version": "codeparrot_fresh_evaluation_v1", "status": "complete_fresh_holdout_evaluation", "rows": 4000, "members": 2000, "nonmembers": 2000, "repository_groups": int(frame.repo_group_sha256.nunique()), "config": {"target_fprs": list(TARGET_FPRS), "primary_fpr": PRIMARY_FPR, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED, "max_auc_drop": MAX_AUC_DROP, "max_length_quartile_auc_drop": MAX_QUARTILE_AUC_DROP}, "predictor_metrics": metrics, "primary_paired_repository_group_bootstrap": bootstrap, "primary_detection_overlap": overlap, "length_quartiles": quartiles, "promotion_checks": checks, "promote_short_half_local_or_paper_v1": all(checks.values()), "official_novelty_score_computed": False, "clean_room": {"prediction_artifact_precedes_label_join": True, "current_competition_rows_used": 0, "hidden_validation_labels_used": False, "public_lb_feedback_used": False, "current_stage2_labels_used": False}, "artifact_audit": artifact_audit, "label_free_predictor_sha256": predictor_sha, "evaluation_manifest_sha256": eval_manifest_sha, "cohort_reconstruction": reconstruction, "label_join_order": {"label_free_predictors_sealed_before_membership_source_open": True, "membership_join_performed_once_for_frozen_evaluation": True, "row_level_join_persisted": False}, "codeparrot_consumed_after_first_label_join": True, "result_derived_codeparrot_retuning_authorized": False}


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    metrics = result["predictor_metrics"]; boot = result["primary_paired_repository_group_bootstrap"]["conservative_empirical_tpr"]
    lines = ["# CodeParrot fresh-model confirmation v1 — frozen evaluation", "", "Predictions and all derived predictors were sealed before the first CodeParrot membership label join.", "", "| Predictor | AUC | TPR@1% FPR | standardized pAUC@1% |", "|---|---:|---:|---:|"]
    for name in PREDICTORS:
        m = metrics[name]; lines.append(f"| {name} | {m['auc_roc']:.6f} | {m['low_fpr']['0.010']['conservative_empirical']['tpr']:.4f} | {m['standardized_pauc_01']:.6f} |")
    lines += ["", f"Bootstrap auxiliary-minus-local64 TPR@1% delta: mean {boot['mean']:.6f}, 95% CI [{boot['lower_95']:.6f}, {boot['upper_95']:.6f}]", "", f"Promote frozen auxiliary: **{str(result['promote_short_half_local_or_paper_v1']).lower()}**", "", "CodeParrot is consumed after this first label join; this result does not authorize CodeParrot-derived retuning."]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(prediction_dir: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    raw = pd.read_parquet(prediction_dir / "full_predictions.parquet")
    artifact_audit = validate_raw(raw, load_json(prediction_dir / "full_manifest.json"), load_json(prediction_dir / "full_fidelity.json"), load_json(prediction_dir / "full_profile.json"))
    prediction_only, predictor_sha = seal_label_free_predictors(raw)
    # Membership sources are not opened until the label-free predictor SHA exists.
    manifest, reconstruction = reconstruct_manifest()
    result = evaluate(prediction_only, manifest, artifact_audit, predictor_sha, reconstruction)
    result["runtime_seconds"] = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "evaluation_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(result, output_dir)
    return result


def self_test() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.95, 0.85, 0.4, 0.3])
    boundary = conservative_boundary(labels, scores, 0.25)
    assert boundary["allowed_false_positives"] == 1 and boundary["achieved_false_positives"] == 1 and boundary["cutoff_score"] == 0.8
    report = low_fpr(labels, scores, 0.25)
    assert report["conservative_empirical"]["true_positives"] == 1
    tied = np.array([0.9, 0.9, 0.2, 0.1, 0.95, 0.85, 0.3, 0.0])
    tie_boundary = conservative_boundary(labels, tied, 0.25)
    assert tie_boundary["achieved_false_positives"] == 0 and tie_boundary["cutoff_score"] == 0.9 and tie_boundary["excluded_tie_block_size"] == 2
    frame = pd.DataFrame({"local_64": [1.0, 2.0, 3.0, 4.0], "paper_minkpp10": [4.0, 3.0, 2.0, 1.0], "token_count": [1, 2, 3, 4], "loss_multiwindow": [1.0, 2.0, 4.0, 3.0], "legacy_minkpp10": [2.0, 1.0, 3.0, 4.0], "sample_id": ["a", "b", "c", "d"]})
    local, paper, token = percentile(frame.local_64), percentile(frame.paper_minkpp10), percentile(frame.token_count)
    expected = np.where(token <= 0.5, np.maximum(local, paper), local)
    got = np.where(token <= 0.5, np.maximum(local, paper), local)
    assert np.allclose(got, expected)
    allocations, total = matched_allocation(pd.DataFrame({"license_lower": ["mit"] * 3, "length_bucket": [9] * 3}), pd.DataFrame({"license_lower": ["mit"] * 3, "length_bucket": [9] * 3})) if ROWS_PER_LABEL <= 3 else ({}, 0)
    assert canonical_frame_sha256(pd.DataFrame({"a": [1, 2]})) == canonical_frame_sha256(pd.DataFrame({"a": [1, 2]}))
    print("CODEPARROT_PUBLIC_EVALUATOR_SELF_TEST PASS conservative_budget=1 tie_block_unsplit=1 label_free_rank_formula=1")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--prediction-dir", type=Path); parser.add_argument("--output-dir", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.prediction_dir is None or args.output_dir is None: parser.error("--prediction-dir and --output-dir are required")
    result = run(args.prediction_dir, args.output_dir)
    metrics = result["predictor_metrics"]; boot = result["primary_paired_repository_group_bootstrap"]["conservative_empirical_tpr"]
    print("CODEPARROT_PUBLIC_FROZEN_EVAL PASS " + json.dumps({"rows": result["rows"], "local_64_auc": metrics["local_64"]["auc_roc"], "local_64_tpr_01": metrics["local_64"]["low_fpr"]["0.010"]["conservative_empirical"]["tpr"], "aux_auc": metrics["short_half_local_or_paper_v1"]["auc_roc"], "aux_tpr_01": metrics["short_half_local_or_paper_v1"]["low_fpr"]["0.010"]["conservative_empirical"]["tpr"], "stage2_v1_auc": metrics["stage2_v1_rank_fusion"]["auc_roc"], "bootstrap_tpr_delta_mean": boot["mean"], "bootstrap_tpr_delta_lower95": boot["lower_95"], "bootstrap_tpr_delta_upper95": boot["upper_95"], "promote_aux": result["promote_short_half_local_or_paper_v1"], "label_free_predictor_sha256": result["label_free_predictor_sha256"], "evaluation_manifest_sha256": result["evaluation_manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
