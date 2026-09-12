"""One-shot same-model deployment experiment for STAGE2-DEPLOYMENT-VALIDATION-V1.

Order is enforced: source-only fit -> bundle seal/reload fidelity -> holdout freeze ->
label-free inference/prediction hash -> label join/evaluation.  No holdout result can
feed back into fitting, layer choice, score direction, or fusion weights.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from transformers import AutoModelForCausalLM, AutoTokenizer

from poisoned_chalice.lumia_hidden_state_pilot import (
    DATASET_ID, DATASET_REVISION, LumiaRuntimePilotConfig, _layer_modules,
    extract_one_sample, validate_public_train,
)
from poisoned_chalice.lumia_hidden_state_scale_cache import load_sealed_cache_5000, join_public_labels_5000
from poisoned_chalice.sersem_author_runtime import load_author_runtime
from poisoned_chalice.stage2_api import Stage2RuntimeConfig, score_samples_detailed
from poisoned_chalice.stage2_deployment_evaluation import evaluate_prediction_frame
from poisoned_chalice.stage2_deployment_validation import (
    DEPLOYMENT_HOLDOUT_SEED, LANGUAGES, build_prediction_frame, content_groups,
    equal_rank_fusion, fit_gr_source, fit_hr_mean_top5_source, load_gr_bundle,
    load_hr_bundle, predict_gr, predict_hr, save_gr_bundle, save_hr_bundle,
    sha256_file,
)

TASK_ID = "STAGE2-DEPLOYMENT-VALIDATION-V1"
MODEL_ID = "bigcode/starcoder2-3b"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
SOURCE_CACHE_MANIFEST_SHA256 = "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa"
OUTPUT = Path("/kaggle/working/stage2_deployment_validation_v1")
BUNDLE = OUTPUT / "sealed_source_bundle"
HIDDEN_DIR = OUTPUT / "holdout_hidden_mean"
PREDICTIONS = OUTPUT / "predictions_label_free.csv"
SHARD_SIZE = 250
MAX_LENGTH_HR = 8192


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_hash(values) -> str:
    return sha256("\n".join(sorted(map(str, values))).encode("utf-8")).hexdigest()


def locate_source_cache(root: Path) -> Path:
    matches = [p for p in root.rglob("cache_manifest.json") if sha256_file(p) == SOURCE_CACHE_MANIFEST_SHA256]
    if len(matches) != 1:
        raise RuntimeError(f"expected one source cache manifest, found {len(matches)}")
    return matches[0].parent


def load_public_train() -> pd.DataFrame:
    expected = {"Go": 10000, "Java": 10000, "Python": 10000, "Ruby": 10000, "Rust": 9040}
    parts = []
    for language, rows in expected.items():
        part = load_dataset(DATASET_ID, language, split="train", revision=DATASET_REVISION).to_pandas()
        if len(part) != rows:
            raise RuntimeError(f"public train row count changed: {language}={len(part)}")
        part["language"] = language
        parts.append(part[["sample_id", "content", "membership", "language"]])
    frame = pd.concat(parts, ignore_index=True)
    validate_public_train(frame)
    normalized = frame.membership.astype("string").str.strip().str.lower().str.replace("_", "-", regex=False)
    frame["label"] = normalized.map({"member": 1, "non-member": 0, "nonmember": 0})
    if frame.label.isna().any() or frame.sample_id.duplicated().any():
        raise RuntimeError("public train label identity changed")
    frame["label"] = frame.label.astype(int)
    return frame


def load_stage1_ids(root: Path) -> set[str]:
    paths = sorted(p for p in root.rglob("features.part*.parquet") if "/train_10k/parts/" in p.as_posix())
    if len(paths) != 40:
        raise RuntimeError(f"expected 40 Stage1 shards, found {len(paths)}")
    frame = pd.concat([pd.read_parquet(p, columns=["sample_id", "language", "label"]) for p in paths], ignore_index=True)
    if len(frame) != 10000 or frame.sample_id.duplicated().any():
        raise RuntimeError("Stage1 10k identity changed")
    counts = frame.groupby(["language", "label"]).size()
    if len(counts) != 10 or not counts.eq(1000).all():
        raise RuntimeError("Stage1 10k balance changed")
    return set(frame.sample_id.astype(str))


def fit_c1(source: pd.DataFrame):
    union = FeatureUnion([
        ("char", HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=2**16, alternate_sign=True, norm="l2", lowercase=False)),
        ("token", HashingVectorizer(analyzer="word", ngram_range=(1, 2), n_features=2**16, alternate_sign=True, norm="l2", lowercase=False, token_pattern=r"(?u)\b\w+\b")),
    ])
    X = union.fit_transform(source.content.astype(str).tolist())
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=20260909)
    model.fit(X, source.label.to_numpy(int))
    return union, model


def seal_source_bundle(source_mean: np.ndarray, source_frame: pd.DataFrame, source_content: pd.DataFrame):
    started = time.perf_counter()
    hr, hr_audit = fit_hr_mean_top5_source(source_mean, source_frame, device="cuda:0")
    hr_hash = save_hr_bundle(hr, BUNDLE)
    gr, _ = fit_gr_source(source_mean, source_frame.label.to_numpy(int))
    gr_hash = save_gr_bundle(gr, BUNDLE)
    c1_union, c1_model = fit_c1(source_content)
    c1_path = BUNDLE / "c1_source_model.joblib"
    joblib.dump((c1_union, c1_model), c1_path, compress=3)
    audit_path = BUNDLE / "hr_training_audit.json"
    write_json(audit_path, hr_audit)

    # The actual deployment objects are reloaded from sealed bytes.
    hr_reload = load_hr_bundle(BUNDLE)
    gr_reload = load_gr_bundle(BUNDLE)
    c1_union_reload, c1_model_reload = joblib.load(c1_path)
    n = min(32, len(source_frame))
    hr_err = float(np.max(np.abs(predict_hr(hr, source_mean[:n], device="cuda:0") - predict_hr(hr_reload, source_mean[:n], device="cuda:0"))))
    gr_err = float(np.max(np.abs(predict_gr(gr, source_mean[:n]) - predict_gr(gr_reload, source_mean[:n]))))
    texts = source_content.content.astype(str).iloc[:n].tolist()
    c1_err = float(np.max(np.abs(c1_model.predict_proba(c1_union.transform(texts))[:, 1] - c1_model_reload.predict_proba(c1_union_reload.transform(texts))[:, 1])))
    if hr_err > 1e-7 or gr_err > 1e-12 or c1_err != 0.0:
        raise RuntimeError(f"bundle reload fidelity failed: HR={hr_err} GR={gr_err} C1={c1_err}")

    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_rows": 5000,
        "source_role": "source development",
        "source_cache_manifest_sha256": SOURCE_CACHE_MANIFEST_SHA256,
        "hr": {"pooling": "mean", "max_length": MAX_LENGTH_HR, "top_layers": hr["top_layers"], "fit_split": "3000/1000/1000 train/early/select", "training_audit_sha256": sha256_file(audit_path), **hr_hash},
        "gr": {"features": "exact mean 96-feature scalar", "preprocessing": "median imputer -> StandardScaler", "learner": "LogisticRegression(C=0.2,max_iter=2000,random_state=20260909)", **gr_hash},
        "c1": {"type": "hashed char(3,5)+token(1,2) logistic", "sha256": sha256_file(c1_path)},
        "labels_used": "source labels only",
        "holdout_rows_seen_during_fit": 0,
        "reload_fidelity": {"rows": n, "hr_max_abs_error": hr_err, "gr_max_abs_error": gr_err, "c1_max_abs_error": c1_err, "passed": True},
        "fit_seconds": time.perf_counter() - started,
    }
    write_json(BUNDLE / "bundle_manifest.json", manifest)
    manifest_sha = sha256_file(BUNDLE / "bundle_manifest.json")
    (BUNDLE / "SEALED.sha256").write_text(manifest_sha + "  bundle_manifest.json\n", encoding="utf-8")
    manifest["bundle_manifest_sha256"] = manifest_sha
    return hr_reload, gr_reload, c1_union_reload, c1_model_reload, manifest


def select_holdout(train: pd.DataFrame, stage1_ids: set[str], source_ids: set[str]):
    excluded_ids = stage1_ids | source_ids
    groups, group_diag = content_groups(train[["content"]])
    work = train.copy()
    work["content_group"] = groups
    excluded_groups = set(work.loc[work.sample_id.astype(str).isin(excluded_ids), "content_group"].astype(int))
    work["near_excluded"] = work.content_group.astype(int).isin(excluded_groups)
    eligible = work[~work.near_excluded].copy()
    counts = eligible.groupby(["language", "label"]).size()
    if len(counts) != 10 or not counts.ge(500).all():
        raise RuntimeError(f"holdout lacks 500 rows/cell after exclusions: {counts.to_dict()}")
    pieces = []
    for language_index, language in enumerate(LANGUAGES):
        for label in (0, 1):
            pool = eligible[(eligible.language == language) & (eligible.label == label)]
            pieces.append(pool.sample(n=500, random_state=DEPLOYMENT_HOLDOUT_SEED + 100 * language_index + label, replace=False))
    selected = pd.concat(pieces, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    selected_ids = set(selected.sample_id.astype(str))
    if len(selected) != 5000 or len(selected_ids) != 5000 or selected_ids & excluded_ids or set(selected.content_group.astype(int)) & excluded_groups:
        raise RuntimeError("holdout disjointness failed")
    selected_counts = selected.groupby(["language", "label"]).size()
    if len(selected_counts) != 10 or not selected_counts.eq(500).all():
        raise RuntimeError("holdout balance changed")
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "role": "same-model deployment holdout",
        "selection_seed": DEPLOYMENT_HOLDOUT_SEED,
        "rows": 5000,
        "rows_per_language_label": 500,
        "official_train_only": True,
        "validation_rows_used": 0,
        "stage1_10k_excluded_rows": len(stage1_ids),
        "source_hidden_5k_excluded_rows": len(source_ids),
        "direct_union_excluded_rows": len(excluded_ids),
        "direct_union_set_sha256": set_hash(excluded_ids),
        "near_duplicate_group_rule": group_diag,
        "groups_touching_excluded_ids": len(excluded_groups),
        "eligible_language_label_counts": {f"{k[0]}|{int(k[1])}": int(v) for k, v in counts.items()},
        "holdout_set_sha256": set_hash(selected_ids),
        "holdout_language_label_counts": {f"{k[0]}|{int(k[1])}": int(v) for k, v in selected_counts.items()},
        "labels_used_for_balanced_selection": True,
        "performance_or_error_used_for_selection": False,
        "individual_error_inspection_before_freeze": False,
        "freshness_claim": "performance-unseen deployment cohort; public labels are not claimed unseen",
    }
    return selected, manifest, groups


def extract_hidden(model, tokenizer, runtime, scoring: pd.DataFrame):
    HIDDEN_DIR.mkdir(parents=True, exist_ok=False)
    names = ("zsigmoid_single_sequence", "caller_literal_output_weighted_single_sequence", "helper_language_aware_output_weighted_single_sequence")
    legacy = {name: np.empty(len(scoring), dtype=np.float64) for name in names}
    seq_len = np.empty(len(scoring), dtype=np.int32)
    records = []
    config = LumiaRuntimePilotConfig(model_id=MODEL_ID, model_revision=MODEL_REVISION, dataset_id=DATASET_ID, dataset_revision=DATASET_REVISION, rows=len(scoring), rows_per_language_label=500, seed=20260909, max_length=MAX_LENGTH_HR, output_dir=str(HIDDEN_DIR), performance_metrics_computed=False)
    started = time.perf_counter()
    if (len(_layer_modules(model)), int(model.config.hidden_size)) != (30, 3072):
        raise RuntimeError("frozen R architecture changed")
    for shard_index, start in enumerate(range(0, len(scoring), SHARD_SIZE)):
        stop = min(start + SHARD_SIZE, len(scoring))
        mean = np.empty((stop - start, 30, 3072), dtype=np.float32)
        shard_started = time.perf_counter()
        for local_index, row in enumerate(scoring.iloc[start:stop].itertuples(index=False)):
            result = extract_one_sample(model, tokenizer, str(row.content), str(row.language), runtime, config)
            mean[local_index] = np.stack([result["activations"][layer]["mean"] for layer in range(30)], axis=0)
            seq_len[start + local_index] = int(result["seq_len"])
            for name in names:
                legacy[name][start + local_index] = float(result["output_scores"][name])
        path = HIDDEN_DIR / f"mean.part{shard_index:02d}.npz"
        np.savez_compressed(path, mean=mean)
        records.append({"shard_index": shard_index, "start_index": start, "stop_index_exclusive": stop, "rows": stop - start, "sha256": sha256_file(path), "seconds": time.perf_counter() - shard_started})
        del mean
    manifest = {
        "rows": len(scoring), "shards": records, "dtype": "float32", "shape_per_row": [30, 3072],
        "pooling": "ordinary unmasked mean on the single unpadded/truncated sequence", "max_length": MAX_LENGTH_HR,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "labels_used": False,
        "sample_order_sha256": sha256("\n".join(scoring.sample_id.astype(str)).encode("utf-8")).hexdigest(),
        "runtime_seconds": time.perf_counter() - started, "seq_len_min": int(seq_len.min()),
        "seq_len_median": float(np.median(seq_len)), "seq_len_max": int(seq_len.max()),
    }
    write_json(HIDDEN_DIR / "manifest.json", manifest)
    return seq_len, legacy, manifest


def predict_hidden_shards(hr_bundle, gr_bundle, rows: int):
    manifest = json.loads((HIDDEN_DIR / "manifest.json").read_text(encoding="utf-8"))
    hr = np.empty(rows, dtype=np.float64)
    gr = np.empty(rows, dtype=np.float64)
    for record in manifest["shards"]:
        start, stop = int(record["start_index"]), int(record["stop_index_exclusive"])
        path = HIDDEN_DIR / f"mean.part{int(record['shard_index']):02d}.npz"
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError("hidden shard hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            mean = np.asarray(archive["mean"], dtype=np.float32)
        hr[start:stop] = predict_hr(hr_bundle, mean, device="cuda:0")
        gr[start:stop] = predict_gr(gr_bundle, mean)
    if not np.isfinite(hr).all() or not np.isfinite(gr).all():
        raise RuntimeError("HR/GR prediction incomplete")
    return hr, gr


def limited_fidelity(hr_bundle, gr_bundle) -> dict:
    manifest = json.loads((HIDDEN_DIR / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["shards"][0]
    path = HIDDEN_DIR / f"mean.part{int(record['shard_index']):02d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        mean = np.asarray(archive["mean"], dtype=np.float32)[:8]
    hr_full = predict_hr(hr_bundle, mean, device="cuda:0")
    hr_split = np.concatenate([predict_hr(hr_bundle, mean[:4], device="cuda:0"), predict_hr(hr_bundle, mean[4:], device="cuda:0")])
    gr_full = predict_gr(gr_bundle, mean)
    gr_split = np.concatenate([predict_gr(gr_bundle, mean[:4]), predict_gr(gr_bundle, mean[4:])])
    hr_err = float(np.max(np.abs(hr_full - hr_split)))
    gr_err = float(np.max(np.abs(gr_full - gr_split)))
    if hr_err > 1e-7 or gr_err > 1e-12:
        raise RuntimeError(f"batch/shard fidelity failed: HR={hr_err} GR={gr_err}")
    result = {"rows": 8, "hr_batch_split_max_abs_error": hr_err, "gr_batch_split_max_abs_error": gr_err, "passed": True}
    write_json(HIDDEN_DIR / "limited_fidelity.json", result)
    return result


def evaluate_after_seal(predictions: pd.DataFrame, train: pd.DataFrame, stage2_features: pd.DataFrame, all_groups: np.ndarray) -> dict:
    frame = predictions.merge(train[["sample_id", "language", "label"]], on=["sample_id", "language"], how="left", validate="one_to_one")
    if frame.label.isna().any() or len(frame) != len(predictions):
        raise RuntimeError("post-seal label join failed")
    frame["label"] = frame.label.astype(int)
    features = stage2_features.sort_values("sample_index").reset_index(drop=True)
    if len(features) != len(frame):
        raise RuntimeError("TR feature order/coverage changed")
    frame["token_count"] = features.token_count.to_numpy(int)
    lookup = pd.Series(np.asarray(all_groups, dtype=int), index=train.sample_id.astype(str))
    frame["content_group"] = frame.sample_id.astype(str).map(lookup)
    sizes = pd.Series(np.asarray(all_groups, dtype=int)).value_counts()
    frame["content_group_non_singleton"] = frame.content_group.astype(int).map(sizes).fillna(1).astype(int) > 1
    score_names = [c for c in predictions.columns if c not in {"sample_id", "language"}]
    result = evaluate_prediction_frame(
        frame,
        score_names=score_names,
        primary_candidate="TR_HR_rank_50_50",
        primary_baseline="TR_stage2_v1",
        extra_bootstrap_pairs=(("HR_mean_top5", "TR_stage2_v1"), ("HR_mean_top5", "C1_source_content")),
    )
    result.update({"schema_version": 1, "task_id": TASK_ID, "rows": len(frame), "prediction_sha256": sha256_file(PREDICTIONS), "performance_labels_joined_only_after_prediction_hash": True})
    return result


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"fresh output directory already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    BUNDLE.mkdir(parents=True)
    started = time.perf_counter()
    input_root = Path("/kaggle/input")

    train = load_public_train()
    source_arrays, source_metadata, _ = load_sealed_cache_5000(locate_source_cache(input_root))
    source_frame = join_public_labels_5000(source_metadata, train)
    source_mean = source_arrays["mean"]
    source_content = source_frame[["sample_id", "language", "label"]].merge(train[["sample_id", "language", "content"]], on=["sample_id", "language"], how="left", validate="one_to_one")
    if source_content.content.isna().any():
        raise RuntimeError("source content join failed")
    source_ids = set(source_frame.sample_id.astype(str))
    stage1_ids = load_stage1_ids(input_root)

    hr_bundle, gr_bundle, c1_union, c1_model, bundle_manifest = seal_source_bundle(source_mean, source_frame, source_content)
    del source_arrays, source_mean, source_frame, source_content
    torch.cuda.empty_cache()

    selected, selection_manifest, all_groups = select_holdout(train, stage1_ids, source_ids)
    write_json(OUTPUT / "holdout_selection_manifest.json", selection_manifest)
    scoring = selected[["sample_id", "language", "content"]].copy()
    scoring.to_parquet(OUTPUT / "holdout_scoring_manifest.parquet", index=False)
    scoring_hash = sha256_file(OUTPUT / "holdout_scoring_manifest.parquet")
    (OUTPUT / "holdout_scoring_manifest.sha256").write_text(scoring_hash + "  holdout_scoring_manifest.parquet\n", encoding="utf-8")
    del selected

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda:0").eval()
    runtime = load_author_runtime(require_exact_versions=True, require_full_components=True)

    _, legacy, hidden_manifest = extract_hidden(model, tokenizer, runtime, scoring)
    fidelity = limited_fidelity(hr_bundle, gr_bundle)
    hr, gr = predict_hidden_shards(hr_bundle, gr_bundle, len(scoring))
    c1 = c1_model.predict_proba(c1_union.transform(scoring.content.astype(str).tolist()))[:, 1]

    tr_started = time.perf_counter()
    tr_result = score_samples_detailed(model=model, tokenizer=tokenizer, samples=scoring.content.astype(str).tolist(), languages=scoring.language.astype(str).tolist(), runtime_config=Stage2RuntimeConfig())
    tr_seconds = time.perf_counter() - tr_started
    tr = tr_result.scores
    stage2_features = tr_result.features.sort_values("sample_index").reset_index(drop=True)
    stage2_features.to_parquet(OUTPUT / "tr_stage2_features.parquet", index=False)
    write_json(OUTPUT / "tr_stage2_manifest.json", tr_result.manifest)

    predictions = build_prediction_frame(scoring.sample_id, scoring.language, HR_mean_top5=hr, GR_mean96=gr, TR_stage2_v1=tr, TR_HR_rank_50_50=equal_rank_fusion(tr, hr), C1_source_content=c1, **legacy)
    predictions.to_csv(PREDICTIONS, index=False)
    prediction_sha = sha256_file(PREDICTIONS)
    (OUTPUT / "predictions_label_free.sha256").write_text(prediction_sha + "  predictions_label_free.csv\n", encoding="utf-8")

    evaluation = evaluate_after_seal(predictions, train, stage2_features, all_groups)
    evaluation["runtime"] = {
        "total_seconds": time.perf_counter() - started,
        "hidden_seconds": hidden_manifest["runtime_seconds"],
        "tr_stage2_seconds": tr_seconds,
        "cuda_peak_memory_bytes": max(int(torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())),
        "limited_fidelity": fidelity,
    }
    evaluation["source_bundle_manifest"] = bundle_manifest
    evaluation["holdout_selection_manifest_sha256"] = sha256_file(OUTPUT / "holdout_selection_manifest.json")
    evaluation["scoring_manifest_sha256"] = scoring_hash
    write_json(OUTPUT / "evaluation.json", evaluation)

    report = {
        "task_id": TASK_ID, "status": "complete", "bundle_manifest_sha256": bundle_manifest["bundle_manifest_sha256"],
        "holdout_set_sha256": selection_manifest["holdout_set_sha256"], "prediction_sha256": prediction_sha,
        "evaluation_sha256": sha256_file(OUTPUT / "evaluation.json"), "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION, "competition_submission": False,
        "validation_used": False, "holdout_retraining_or_reselection": False, "performance_rows_persisted_with_labels": False,
    }
    write_json(OUTPUT / "run_manifest.json", report)
    print("STAGE2_DEPLOYMENT_VALIDATION_V1 COMPLETE " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
