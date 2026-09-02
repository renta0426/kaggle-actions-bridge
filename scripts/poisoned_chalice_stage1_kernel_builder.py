"""Assemble the frozen Poisoned Chalice Stage 1 notebook from private Kaggle source kernels.

The bridge repository is public while the competition research repository is private.
To avoid copying the full private research implementation into this repository, the
launch workflow pulls two already-existing private Kaggle notebooks owned by the user.
This builder extracts only their definition cells and adds the frozen orchestration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _find_definition_cell(notebook: dict, markers: tuple[str, ...], label: str) -> str:
    matches = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = _source(cell)
        if all(marker in source for marker in markers):
            matches.append(source)
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {label} definition cell, got {len(matches)}")
    return matches[0]


EVALUATION_SOURCE = r'''
from __future__ import annotations

import numpy as np
import pandas as pd

CDF_SOURCES = ("score_loss_mean__max", "min_k_10__max", "min_kpp_10__max")
EXCLUDED_COLUMNS = {
    "label", "validation_order", "window_count", "token_count", "window_token_count",
    "window_start", "file_token_count",
}


def numeric_feature_columns(frame):
    return [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in EXCLUDED_COLUMNS and not column.startswith("cdf_")
    ]


def add_fold_cdf(reference, target, columns=CDF_SOURCES, min_language_rows=20):
    output = target.copy()
    nonmember = reference[reference.label == 0]
    if nonmember.empty:
        raise ValueError("CDF reference has no non-members")
    for column in columns:
        global_values = np.sort(nonmember[column].dropna().to_numpy(float))
        language_values = {
            language: np.sort(group[column].dropna().to_numpy(float))
            for language, group in nonmember.groupby("language")
        }
        calibrated = []
        for language, value in zip(output.language, output[column]):
            values = language_values.get(language, np.array([], dtype=float))
            if len(values) < min_language_rows:
                values = global_values
            calibrated.append(np.searchsorted(values, value, side="right") / len(values))
        output[f"cdf_{column}"] = calibrated
    return output


def _pipeline(seed):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.2, max_iter=2000, random_state=seed),
    )


def stratified_splits(frame, n_splits, seed):
    from sklearn.model_selection import StratifiedKFold
    strata = frame.language.astype(str) + "_" + frame.label.astype(int).astype(str)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(frame, strata))


def cross_fit_meta(frame, splits, base_features, seed):
    cdf_features = [f"cdf_{column}" for column in CDF_SOURCES if column in base_features]
    model_features = list(base_features) + cdf_features
    prediction = np.full(len(frame), np.nan, dtype=float)
    seen = np.zeros(len(frame), dtype=int)
    labels = frame.label.astype(int).to_numpy()
    for fold, (fit_index, hold_index) in enumerate(splits):
        fit = add_fold_cdf(frame.iloc[fit_index], frame.iloc[fit_index])
        hold = add_fold_cdf(frame.iloc[fit_index], frame.iloc[hold_index])
        model = _pipeline(seed + fold)
        model.fit(fit[model_features], labels[fit_index])
        prediction[hold_index] = model.predict_proba(hold[model_features])[:, 1]
        seen[hold_index] += 1
    if not np.isfinite(prediction).all() or not np.all(seen == 1):
        raise RuntimeError("OOF coverage invalid")
    return prediction


def low_fpr_metrics(y_true, score):
    from sklearn.metrics import roc_auc_score, roc_curve
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    fpr, tpr, _ = roc_curve(y, score)
    result = {
        "auc": float(roc_auc_score(y, score)),
        "pauc_01": float(roc_auc_score(y, score, max_fpr=0.01)),
    }
    for rate in (0.001, 0.005, 0.01, 0.02):
        result[f"tpr_at_{rate:g}_fpr"] = float(tpr[fpr <= rate].max())
    return result
'''


RUNNER_SOURCE = r'''
from __future__ import annotations

from collections import Counter
from pathlib import Path
import gc
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

METHOD = "raw_plus_fim"
CORE_COMMIT = "bcf6d44db86fdda4ac9dd07b2d0172de471ab67a"
SOURCE_COMMIT = "28c34307c71db733dab2744ec9ec46549e54d424"
SEED = 2027
TRAIN_PER_LANGUAGE = 2000
SHARD_SIZE = 250
EXPECTED_TRAIN_ROWS = 10000
EXPECTED_VALIDATION_ROWS = 5000
EXPECTED_BASE_FEATURES = 113
EXPECTED_STRUCTURE_FEATURES = 50
EXPECTED_FIM_FEATURES = 11
EXPECTED_OOF_AUC = 0.664524
OOF_AUC_TOLERANCE = 0.002


def configs(output_root):
    output_root = str(output_root)
    return (
        FeatureCacheV2Config(
            train_samples_per_language=TRAIN_PER_LANGUAGE,
            samples_per_shard=SHARD_SIZE,
            output_dir=f"{output_root}/base",
        ),
        FIMConfig(
            samples_per_language=TRAIN_PER_LANGUAGE,
            samples_per_shard=SHARD_SIZE,
            output_dir=f"{output_root}/fim",
        ),
    )


def parse_frame(frame, tokenizer, parsers, update_counts):
    from tqdm.auto import tqdm
    parsed_by_id = {}
    counts = Counter()
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="Parse/tokenize"):
        parsed = parse_token_categories(row.content, row.language, tokenizer, parsers[row.language])
        parsed_by_id[row.sample_id] = parsed
        if update_counts:
            counts.update(int(token) for token in parsed.token_ids)
    return parsed_by_id, counts


def build_fim_records(frame, parsed_by_id, token_counts, special_ids, fim_config):
    records = []
    for row in frame.itertuples(index=False):
        parsed = parsed_by_id[row.sample_id]
        span = select_fim_span(parsed.token_ids, parsed.categories, token_counts, fim_config)
        sequence = build_fim_sequence(parsed.token_ids, span, special_ids, fim_config)
        label = float(row.label) if pd.notna(row.label) else np.nan
        records.append({
            "sample_id": row.sample_id,
            "language": row.language,
            "membership": row.membership,
            "label": label,
        } | span | sequence)
    return records


def fidelity_gates(train, train_parsed, token_counts, model, tokenizer, special_ids, base_config, fim_config):
    pilot_frame = train.iloc[:2].copy()
    pilot_windows = make_window_records(pilot_frame, tokenizer, base_config.max_length)
    pilot_config = StarterPlusConfig(
        max_length=base_config.max_length,
        max_batch_tokens=base_config.max_batch_tokens,
        vocab_chunk_tokens=base_config.vocab_chunk_tokens,
        train_samples_per_language=base_config.train_samples_per_language,
    )
    base_reference = extract_window_features(pilot_windows, model, tokenizer, pilot_config)
    base_optimized, _, base_profile = extract_window_features_optimized(
        pilot_windows, model, tokenizer, base_config, progress=False
    )
    base_fidelity = compare_feature_frames(base_reference, base_optimized, atol=1e-6)

    pilot_fim = build_fim_records(pilot_frame, train_parsed, token_counts, special_ids, fim_config)
    fim_reference, fim_reference_profile = extract_fim_features(
        pilot_fim, model, tokenizer, fim_config, use_logits_to_keep=False, progress=False
    )
    use_logits_to_keep = True
    fim_optimization_error = None
    fim_optimized_profile = None
    fim_optimized = None
    try:
        fim_optimized, fim_optimized_profile = extract_fim_features(
            pilot_fim, model, tokenizer, fim_config, use_logits_to_keep=True, progress=False
        )
        fim_columns = [column for column in fim_reference.columns if column.startswith("fim_")]
        fim_max_difference = float(np.nanmax(np.abs(
            fim_reference[fim_columns].to_numpy(float)
            - fim_optimized[fim_columns].to_numpy(float)
        )))
        if fim_max_difference > 1e-6:
            use_logits_to_keep = False
    except (TypeError, ValueError) as error:
        use_logits_to_keep = False
        fim_max_difference = None
        fim_optimization_error = repr(error)

    fidelity = {
        "base": base_fidelity,
        "base_profile": base_profile,
        "fim_gate": 1e-6,
        "fim_max_absolute_difference": fim_max_difference,
        "fim_use_logits_to_keep": use_logits_to_keep,
        "fim_optimization_error": fim_optimization_error,
        "fim_reference_profile": fim_reference_profile,
        "fim_optimized_profile": fim_optimized_profile,
    }
    if not base_fidelity["passed"]:
        raise RuntimeError("Base optimized extractor failed exact fidelity gate")
    del base_reference, base_optimized, fim_reference, fim_optimized
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return fidelity, use_logits_to_keep


def extract_feature_shard(shard, parsed_by_id, token_counts, model, tokenizer, special_ids,
                          base_config, fim_config, use_logits_to_keep):
    windows = make_window_records(shard, tokenizer, base_config.max_length)
    window_features, token_rows, _ = extract_window_features_optimized(
        windows, model, tokenizer, base_config, progress=False
    )
    base = aggregate_windows(window_features)
    token_groups = {
        sample_id: group for sample_id, group in token_rows.groupby("sample_id", sort=False)
    }
    ast_rows = []
    for row in shard.itertuples(index=False):
        summary = summarize_sample_categories(token_groups[row.sample_id], parsed_by_id[row.sample_id])
        ast_rows.append({"sample_id": row.sample_id, "language": row.language} | summary)
    ast = pd.DataFrame(ast_rows)

    fim_records = build_fim_records(shard, parsed_by_id, token_counts, special_ids, fim_config)
    fim, _ = extract_fim_features(
        fim_records, model, tokenizer, fim_config,
        use_logits_to_keep=use_logits_to_keep, progress=False
    )
    fim_keep = ["sample_id"] + [column for column in fim.columns if column.startswith("fim_")]
    metadata_keep = ["sample_id", "language", "membership", "label"]
    if "validation_order" in shard.columns:
        metadata_keep.append("validation_order")
    merged = base.merge(ast, on=["sample_id", "language"], validate="one_to_one")
    merged = merged.merge(fim[fim_keep], on="sample_id", validate="one_to_one")
    merged = merged.merge(
        shard[metadata_keep], on=["sample_id", "language"], validate="one_to_one"
    )
    return merged.sort_values("sample_id").reset_index(drop=True)


def extract_feature_frame(frame, parsed_by_id, name, cache_root, token_counts, model, tokenizer,
                          special_ids, base_config, fim_config, use_logits_to_keep):
    parts_dir = cache_root / name / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for shard_index, start in enumerate(range(0, len(frame), SHARD_SIZE)):
        stop = min(len(frame), start + SHARD_SIZE)
        path = parts_dir / f"features.part{shard_index:03d}.parquet"
        shard_features = extract_feature_shard(
            frame.iloc[start:stop].copy(), parsed_by_id, token_counts, model, tokenizer,
            special_ids, base_config, fim_config, use_logits_to_keep
        )
        if len(shard_features) != stop - start or not shard_features.sample_id.is_unique:
            raise RuntimeError(f"Feature coverage failure in {name} shard {shard_index}")
        temporary = path.with_suffix(".tmp.parquet")
        shard_features.to_parquet(temporary, index=False)
        temporary.replace(path)
        parts.append(shard_features)
        print({"split": name, "shard": shard_index, "rows": len(shard_features)})
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    result = pd.concat(parts, ignore_index=True)
    if len(result) != len(frame) or not result.sample_id.is_unique:
        raise RuntimeError(f"Feature coverage failure for {name}")
    return result


def fit_and_score(train_features, validation_features):
    all_numeric = numeric_feature_columns(train_features)
    base_columns = [column for column in all_numeric if not column.startswith(("ast_", "fim_"))]
    structure_columns = [column for column in all_numeric if column.startswith("ast_")]
    fim_columns = [column for column in all_numeric if column.startswith("fim_")]
    feature_columns = base_columns + structure_columns + fim_columns
    if len(base_columns) != EXPECTED_BASE_FEATURES:
        raise RuntimeError(f"Expected {EXPECTED_BASE_FEATURES} base features, got {len(base_columns)}")
    if len(structure_columns) != EXPECTED_STRUCTURE_FEATURES:
        raise RuntimeError(f"Expected {EXPECTED_STRUCTURE_FEATURES} structure features, got {len(structure_columns)}")
    if len(fim_columns) != EXPECTED_FIM_FEATURES:
        raise RuntimeError(f"Expected {EXPECTED_FIM_FEATURES} FIM features, got {len(fim_columns)}")

    train_features = train_features.sort_values("sample_id").reset_index(drop=True).copy()
    train_features["label"] = train_features.label.astype(int)
    splits = stratified_splits(train_features, 5, SEED)
    oof_score = cross_fit_meta(train_features, splits, feature_columns, SEED)
    oof_metrics = low_fpr_metrics(train_features.label, oof_score)
    oof_delta = float(oof_metrics["auc"] - EXPECTED_OOF_AUC)
    if abs(oof_delta) > OOF_AUC_TOLERANCE:
        raise RuntimeError(
            f"Frozen raw_plus_fim reproduction gate failed: AUC={oof_metrics['auc']:.6f}, "
            f"expected={EXPECTED_OOF_AUC:.6f}+/-{OOF_AUC_TOLERANCE:.6f}"
        )

    cdf_features = [f"cdf_{column}" for column in CDF_SOURCES if column in feature_columns]
    model_features = feature_columns + cdf_features
    fit_frame = add_fold_cdf(train_features, train_features)
    target_frame = add_fold_cdf(train_features, validation_features)
    meta_model = _pipeline(SEED)
    meta_model.fit(fit_frame[model_features], train_features.label.to_numpy(int))
    validation_score = meta_model.predict_proba(target_frame[model_features])[:, 1]
    if not np.isfinite(validation_score).all():
        raise RuntimeError("Non-finite validation membership scores")
    return validation_score, {
        "feature_counts": {
            "base": len(base_columns),
            "structure_raw": len(structure_columns),
            "fim": len(fim_columns),
            "total": len(feature_columns),
            "cdf_added_at_fit": len(cdf_features),
        },
        "reproduced_oof": oof_metrics,
        "expected_oof_auc": EXPECTED_OOF_AUC,
        "oof_auc_delta": oof_delta,
    }


def main(output_dir="/kaggle/working"):
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tree_sitter_languages import get_parser

    if not torch.cuda.is_available():
        raise RuntimeError("Frozen reproduction requires CUDA; TPU is intentionally not substituted")
    output = Path(output_dir)
    cache_root = output / "stage1_raw_fim_v1"
    cache_root.mkdir(parents=True, exist_ok=True)
    base_config, fim_config = configs(cache_root)
    seed_everything(SEED)
    started = time.perf_counter()

    train_full, validation = load_canonical_data(base_config)
    train = sample_training_rows(train_full, TRAIN_PER_LANGUAGE, SEED)
    train = train.sort_values("sample_id").reset_index(drop=True)
    del train_full
    gc.collect()
    validation = validation.reset_index(drop=True).copy()
    validation["validation_order"] = np.arange(len(validation), dtype=np.int32)
    if len(train) != EXPECTED_TRAIN_ROWS or len(validation) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError("Unexpected train/validation row count")
    if not train.groupby(["language", "label"]).size().eq(TRAIN_PER_LANGUAGE // 2).all():
        raise RuntimeError("Deterministic 10k training cohort is not balanced")

    tokenizer = AutoTokenizer.from_pretrained(
        base_config.model_id, revision=base_config.model_revision, use_fast=True
    )
    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer required for exact AST/token offset alignment")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_ids = fim_special_token_ids(tokenizer)
    parser_names = {"Go": "go", "Java": "java", "Python": "python", "Ruby": "ruby", "Rust": "rust"}
    parsers = {language: get_parser(name) for language, name in parser_names.items()}

    train_parsed, train_token_counts = parse_frame(train, tokenizer, parsers, True)
    validation_parsed, _ = parse_frame(validation, tokenizer, parsers, False)

    model = AutoModelForCausalLM.from_pretrained(
        base_config.model_id,
        revision=base_config.model_revision,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation=base_config.attention_implementation,
    ).to("cuda").eval()
    print({"gpu": torch.cuda.get_device_name(0), "method": METHOD})

    fidelity, use_logits_to_keep = fidelity_gates(
        train, train_parsed, train_token_counts, model, tokenizer,
        special_ids, base_config, fim_config
    )
    train_features = extract_feature_frame(
        train, train_parsed, "train_10k", cache_root, train_token_counts, model, tokenizer,
        special_ids, base_config, fim_config, use_logits_to_keep
    )
    del train_parsed
    gc.collect()
    validation_features = extract_feature_frame(
        validation, validation_parsed, "validation_5k", cache_root, train_token_counts,
        model, tokenizer, special_ids, base_config, fim_config, use_logits_to_keep
    )

    validation_score, diagnostics = fit_and_score(train_features, validation_features)
    validation_features = validation_features.copy()
    validation_features["membership_score"] = validation_score
    visible_mask = validation_features.label.notna().to_numpy()
    visible_metrics = low_fpr_metrics(
        validation_features.loc[visible_mask, "label"].astype(int),
        validation_features.loc[visible_mask, "membership_score"],
    )

    submission = validation_features.sort_values("validation_order")[["sample_id", "membership_score"]].copy()
    if len(submission) != EXPECTED_VALIDATION_ROWS or not submission.sample_id.is_unique:
        raise RuntimeError("Submission row/id coverage mismatch")
    if set(submission.sample_id) != set(validation.sample_id):
        raise RuntimeError("Submission sample IDs do not match validation")
    if not np.isfinite(submission.membership_score.to_numpy(float)).all():
        raise RuntimeError("Submission contains non-finite scores")

    submission_path = output / "submission.csv"
    submission.to_csv(submission_path, index=False)
    submission_sha256 = hashlib.sha256(submission_path.read_bytes()).hexdigest()
    manifest = {
        "status": "complete",
        "method": METHOD,
        "core_commit": CORE_COMMIT,
        "source_commit": SOURCE_COMMIT,
        "model_id": base_config.model_id,
        "model_revision": base_config.model_revision,
        "dataset_id": base_config.dataset_id,
        "dataset_revision": base_config.dataset_revision,
        "train_rows": len(train_features),
        "validation_rows": len(validation_features),
        "visible_validation_rows": int(visible_mask.sum()),
        **diagnostics,
        "visible_validation_diagnostic_only": visible_metrics,
        "fidelity": fidelity,
        "submission_file": submission_path.name,
        "submission_rows": len(submission),
        "submission_sha256": submission_sha256,
        "runtime_seconds": time.perf_counter() - started,
        "hidden_validation_labels_used": False,
        "public_leaderboard_tuning_used": False,
        "validation_labels_used_for_fit_or_feature_selection": False,
        "assembled_from_private_kaggle_sources": True,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return manifest


manifest = main()
manifest
'''


def build(base_notebook: Path, fim_notebook: Path, output_dir: Path) -> None:
    base = _load_notebook(base_notebook)
    fim = _load_notebook(fim_notebook)
    starter = _find_definition_cell(
        base,
        ("class StarterPlusConfig", "def make_window_records", "def aggregate_windows", "def extract_window_features"),
        "Starter++",
    )
    feature_cache = _find_definition_cell(
        base,
        ("class FeatureCacheV2Config", "def extract_window_features_optimized", "def compare_feature_frames"),
        "feature-cache-v2",
    )
    structure = _find_definition_cell(
        fim,
        ("class ParsedCategories", "def parse_token_categories", "def summarize_sample_categories"),
        "structure",
    )
    fim_features = _find_definition_cell(
        fim,
        ("class FIMConfig", "def select_fim_span", "def extract_fim_features", "def fim_special_token_ids"),
        "selective-FIM",
    )

    # The original cache builder removes this package-relative import when it
    # creates its self-contained notebook. Keep the operation idempotent here.
    feature_cache = feature_cache.replace(
        "from .starter_plus import StarterPlusConfig, dynamic_batches, summarize_tokens\n",
        "",
    )
    checks = {
        "starter_model": "bigcode/starcoder2-3b" in starter,
        "dataset_revision": "2ed5468723efa5457a3665782c6979ea4dbac7c2" in starter,
        "feature_cache_revision": "733247c55e3f73af49ce8e9c7949bf14af205928" in feature_cache,
        "feature_cache_10k": "train_samples_per_language: int = 2_000" in feature_cache,
        "structure_categories": all(name in structure for name in ("comment", "string", "identifier", "punctuation")),
        "fim_target": "target_tokens: int = 32" in fim_features,
        "fim_rank_block": "rank_vocab_block_size: int = 8_192" in fim_features,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise SystemExit(f"private Kaggle source notebook contract changed: {failed}")

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Stage 1 raw-structure + selective-FIM submission v1\n",
                    "\n",
                    "Frozen 10k Stage 1 reproduction. Source definitions are assembled from the user's already-completed private Kaggle notebooks; no hidden-label recovery and no Public-LB tuning are used.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "%pip install -q datasets transformers==5.0.0 accelerate pyarrow tree-sitter==0.20.4 tree-sitter-languages==1.10.2\n"
                ],
            },
        ]
        + [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
            for source in (starter, feature_cache, structure, fim_features, textwrap.dedent(EVALUATION_SOURCE), textwrap.dedent(RUNNER_SOURCE))
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage1-raw-fim-submission-v1.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata = {
        "id": "renta0426/stage1-raw-fim-submission-v1",
        "title": "Stage1 Raw FIM Submission V1",
        "code_file": "stage1-raw-fim-submission-v1.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "built", "source_contract": checks, "cells": len(notebook["cells"])}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-notebook", type=Path, required=True)
    parser.add_argument("--fim-notebook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.base_notebook, args.fim_notebook, args.output_dir)


if __name__ == "__main__":
    main()
