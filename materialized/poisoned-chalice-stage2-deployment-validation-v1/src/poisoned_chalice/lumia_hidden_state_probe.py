"""Frozen outer-OOF probe evaluation for P1-03 LUMIA pooled hidden states.

The GPU cache is label-free.  Labels are joined only after the cache is sealed and
validated.  Each outer fold uses a strict 60/20/20 split of the outer-training rows
for probe weight updates, early stopping, and layer selection respectively; the
outer holdout is score-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import copy
import json
import math

import numpy as np
import pandas as pd

from poisoned_chalice.evaluation import conservative_detection_set, low_fpr_metrics
from poisoned_chalice.lumia_hidden_state_cache import OUTPUT_BASELINES, POOLING_VARIANTS, sha256_file


HIDDEN_SCORE_NAMES = (
    "mean_only_best_layer",
    "mean_only_top5_ensemble",
    "lumia_caller_literal_best_layer",
    "lumia_caller_literal_top5_ensemble",
    "lumia_helper_language_aware_best_layer",
    "lumia_helper_language_aware_top5_ensemble",
)


@dataclass(frozen=True)
class LumiaProbeConfig:
    seed: int = 20260909
    outer_folds: int = 5
    hidden_dim: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs_max: int = 100
    batch_size: int = 32
    validation_every_epochs: int = 5
    early_stopping_patience_checks: int = 5
    ensemble_size: int = 5
    bootstrap_replicates: int = 1000


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def load_sealed_cache(root: str | Path) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    root = Path(root)
    manifest = json.loads((root / "cache_manifest.json").read_text(encoding="utf-8"))
    rows = int(manifest["rows"])
    layers = int(manifest["num_layers"])
    hidden = int(manifest["hidden_size"])
    if rows != 1000 or layers <= 0 or hidden <= 0:
        raise ValueError("unexpected sealed LUMIA cache dimensions")
    if manifest.get("labels_present_during_model_scoring") is not False:
        raise ValueError("sealed cache label boundary changed")
    if manifest.get("performance_metrics_computed") is not False:
        raise ValueError("sealed cache unexpectedly contains performance evaluation")
    if manifest.get("raw_token_layer_activations_persisted") is not False or manifest.get("logits_persisted") is not False:
        raise ValueError("sealed cache persistence boundary changed")
    if tuple(manifest.get("pooling_variants") or []) != POOLING_VARIANTS:
        raise ValueError("sealed cache pooling variants changed")
    if tuple(manifest.get("output_baselines") or []) != OUTPUT_BASELINES:
        raise ValueError("sealed cache output baselines changed")

    metadata_path = root / manifest["metadata"]["sample_metadata"]
    cohort_path = root / manifest["metadata"]["cohort_manifest"]
    if sha256_file(metadata_path) != manifest["metadata"]["sample_metadata_sha256"]:
        raise ValueError("sample metadata hash mismatch")
    if sha256_file(cohort_path) != manifest["metadata"]["cohort_manifest_sha256"]:
        raise ValueError("cohort manifest hash mismatch")
    metadata = _read_jsonl(metadata_path)
    cohort = _read_jsonl(cohort_path)
    if len(metadata) != rows or len(cohort) != rows:
        raise ValueError("sealed cache metadata coverage changed")
    if not np.array_equal(metadata.execution_index.to_numpy(), np.arange(rows)):
        raise ValueError("sample metadata execution order changed")
    if not np.array_equal(cohort.execution_index.to_numpy(), np.arange(rows)):
        raise ValueError("cohort manifest execution order changed")
    if metadata.sample_id.tolist() != cohort.sample_id.tolist() or metadata.language.tolist() != cohort.language.tolist():
        raise ValueError("cache metadata/cohort identity mismatch")
    if metadata.sample_id.duplicated().any():
        raise ValueError("sealed cache contains duplicate sample IDs")

    buffers = {name: [] for name in POOLING_VARIANTS}
    expected_start = 0
    for shard in manifest["shards"]:
        if int(shard["start_index"]) != expected_start:
            raise ValueError("cache shard coverage is not contiguous")
        path = root / shard["path"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError("cache shard hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            for name in POOLING_VARIANTS:
                values = np.asarray(archive[name], dtype=np.float32)
                if values.ndim != 3 or values.shape[1:] != (layers, hidden):
                    raise ValueError("cache shard hidden dimensions changed")
                if not np.isfinite(values).all():
                    raise ValueError("cache shard contains nonfinite values")
                buffers[name].append(values)
            shard_rows = len(archive[POOLING_VARIANTS[0]])
        if shard_rows != int(shard["rows"]):
            raise ValueError("cache shard row count changed")
        expected_start = int(shard["stop_index_exclusive"])
    if expected_start != rows:
        raise ValueError("cache shard coverage incomplete")
    arrays = {name: np.concatenate(buffers[name], axis=0) for name in POOLING_VARIANTS}
    for name, values in arrays.items():
        if values.shape != (rows, layers, hidden) or values.dtype != np.float32:
            raise ValueError(f"sealed cache array changed: {name} {values.shape} {values.dtype}")
    return arrays, metadata, manifest


def join_public_labels(metadata: pd.DataFrame, public_train: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "language", "membership"}
    missing = required.difference(public_train.columns)
    if missing:
        raise ValueError(f"public label table missing columns: {sorted(missing)}")
    labels = public_train[list(required)].copy()
    normalized = labels.membership.astype("string").str.strip().str.lower().str.replace("_", "-", regex=False)
    labels["label"] = normalized.map({"member": 1, "non-member": 0, "nonmember": 0})
    if labels.label.isna().any() or labels.sample_id.duplicated().any():
        raise ValueError("public train label identity invalid")
    merged = metadata.merge(labels[["sample_id", "language", "label"]], on="sample_id", how="left", suffixes=("", "_public"), validate="one_to_one")
    if merged.label.isna().any():
        raise ValueError("sealed cache sample missing public label")
    if not (merged.language == merged.language_public).all():
        raise ValueError("sealed cache/public language mismatch")
    merged = merged.drop(columns=["language_public"])
    merged["label"] = merged.label.astype(int)
    counts = merged.groupby(["language", "label"]).size().to_dict()
    if sorted(counts.values()) != [100] * 10:
        raise ValueError(f"unexpected frozen 1k language-label balance: {counts}")
    return merged


def outer_and_inner_splits(frame: pd.DataFrame, config: LumiaProbeConfig):
    from sklearn.model_selection import StratifiedKFold, train_test_split

    indices = np.arange(len(frame))
    strata = frame.language.astype(str) + "_" + frame.label.astype(str)
    splitter = StratifiedKFold(n_splits=config.outer_folds, shuffle=True, random_state=config.seed)
    result = []
    for fold, (outer_fit, outer_hold) in enumerate(splitter.split(indices, strata)):
        fit_strata = strata.iloc[outer_fit].to_numpy()
        trainval, select = train_test_split(
            outer_fit,
            test_size=0.20,
            random_state=config.seed + 1000 + fold,
            stratify=fit_strata,
        )
        trainval_strata = strata.iloc[trainval].to_numpy()
        train, early = train_test_split(
            trainval,
            test_size=0.25,
            random_state=config.seed + 2000 + fold,
            stratify=trainval_strata,
        )
        # 800 outer-fit rows -> 480/160/160 and 200 outer-hold rows.
        if (len(train), len(early), len(select), len(outer_hold)) != (480, 160, 160, 200):
            raise RuntimeError("frozen outer/inner split sizes changed")
        for subset in (train, early, select, outer_hold):
            observed = frame.iloc[subset].groupby(["language", "label"]).size().to_numpy()
            if len(observed) != 10 or len(set(observed.tolist())) != 1:
                raise RuntimeError("language-label stratification changed within a split")
        if set(train) & set(early) or set(train) & set(select) or set(train) & set(outer_hold) or set(early) & set(select) or set(early) & set(outer_hold) or set(select) & set(outer_hold):
            raise RuntimeError("outer/inner split overlap detected")
        if set(train) | set(early) | set(select) | set(outer_hold) != set(indices):
            raise RuntimeError("outer/inner split coverage changed")
        result.append((train, early, select, outer_hold))
    return result


def variant_layer_matrix(arrays: dict[str, np.ndarray], variant: str, layer: int) -> np.ndarray:
    mean = arrays["mean"][:, layer, :]
    if variant == "mean_only":
        return mean
    if variant == "lumia_caller_literal":
        return np.concatenate([mean, arrays["caller_weighted"][:, layer, :]], axis=1)
    if variant == "lumia_helper_language_aware":
        return np.concatenate([mean, arrays["helper_weighted"][:, layer, :]], axis=1)
    raise ValueError(f"unknown hidden-state variant: {variant}")


def _make_probe(torch: Any, input_dim: int, hidden_dim: int):
    nn = torch.nn
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid(),
    )


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_early: np.ndarray,
    y_early: np.ndarray,
    *,
    config: LumiaProbeConfig,
    seed: int,
    device: str,
):
    import torch
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    probe = _make_probe(torch, X_train.shape[1], config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = torch.nn.BCELoss()
    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    X_early_t = torch.as_tensor(X_early, dtype=torch.float32, device=device)

    best_auc = -math.inf
    best_state = None
    patience = 0
    best_epoch = -1
    for epoch in range(config.epochs_max):
        probe.train()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch)
        order = torch.randperm(len(X_train_t), generator=generator)
        for start in range(0, len(order), config.batch_size):
            batch_index = order[start:start + config.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = probe(X_train_t[batch_index]).squeeze(-1)
            loss = criterion(prediction, y_train_t[batch_index])
            loss.backward()
            optimizer.step()
        if epoch % config.validation_every_epochs == 0:
            probe.eval()
            with torch.no_grad():
                early_prediction = probe(X_early_t).squeeze(-1).detach().cpu().numpy()
            auc = float(roc_auc_score(y_early, early_prediction))
            if auc > best_auc:
                best_auc = auc
                best_state = copy.deepcopy(probe.state_dict())
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
                if patience >= config.early_stopping_patience_checks:
                    break
    if best_state is None:
        raise RuntimeError("probe early-stopping state was never captured")
    probe.load_state_dict(best_state)
    probe.eval()
    return probe, {"early_stop_auc": best_auc, "best_epoch": best_epoch}


def predict_probe(probe: Any, X: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    import torch

    values = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            values.append(probe(batch).squeeze(-1).detach().cpu().numpy())
    result = np.concatenate(values).astype(np.float64, copy=False)
    if result.shape != (len(X),) or not np.isfinite(result).all():
        raise RuntimeError("probe prediction coverage/finite check failed")
    return result


def run_outer_oof(
    arrays: dict[str, np.ndarray],
    frame: pd.DataFrame,
    config: LumiaProbeConfig,
    *,
    device: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    from sklearn.metrics import roc_auc_score

    if len(frame) != 1000 or sorted(frame.label.unique()) != [0, 1]:
        raise ValueError("frozen LUMIA evaluation frame changed")
    layers = arrays["mean"].shape[1]
    splits = outer_and_inner_splits(frame, config)
    predictions = {name: np.full(len(frame), np.nan, dtype=np.float64) for name in HIDDEN_SCORE_NAMES}
    fold_records: list[dict[str, Any]] = []
    y = frame.label.to_numpy(int)
    variants = ("mean_only", "lumia_caller_literal", "lumia_helper_language_aware")

    for fold, (train_index, early_index, select_index, hold_index) in enumerate(splits):
        for variant_index, variant in enumerate(variants):
            selection = []
            hold_predictions: dict[int, np.ndarray] = {}
            layer_records = []
            for layer in range(layers):
                X = variant_layer_matrix(arrays, variant, layer)
                seed = config.seed + fold * 10000 + variant_index * 1000 + layer
                probe, training_record = train_probe(
                    X[train_index], y[train_index], X[early_index], y[early_index],
                    config=config, seed=seed, device=device,
                )
                select_prediction = predict_probe(
                    probe, X[select_index], batch_size=config.batch_size, device=device
                )
                selection_auc = float(roc_auc_score(y[select_index], select_prediction))
                hold_prediction = predict_probe(
                    probe, X[hold_index], batch_size=config.batch_size, device=device
                )
                selection.append((layer, selection_auc))
                hold_predictions[layer] = hold_prediction
                layer_records.append({
                    "layer": layer,
                    "selection_auc": selection_auc,
                    **training_record,
                })
                del probe
            ordered = sorted(selection, key=lambda item: (-item[1], item[0]))
            best_layer = ordered[0][0]
            top_layers = [layer for layer, _ in ordered[:config.ensemble_size]]
            best_name = f"{variant}_best_layer"
            top_name = f"{variant}_top5_ensemble"
            predictions[best_name][hold_index] = hold_predictions[best_layer]
            predictions[top_name][hold_index] = np.mean(
                np.stack([hold_predictions[layer] for layer in top_layers], axis=0), axis=0
            )
            fold_records.append({
                "fold": fold,
                "variant": variant,
                "best_layer": best_layer,
                "top5_layers": top_layers,
                "layer_records": layer_records,
            })
    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"OOF prediction coverage failed: {name}")
    return predictions, fold_records


def per_language_auc(frame: pd.DataFrame, score: Sequence[float]) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    values = np.asarray(score, dtype=float)
    return {
        language: float(roc_auc_score(group.label, values[group.index.to_numpy()]))
        for language, group in frame.groupby("language", sort=True)
    }


def paired_bootstrap_delta(
    frame: pd.DataFrame,
    left: Sequence[float],
    right: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    y = frame.label.to_numpy(int)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    strata = [group.index.to_numpy() for _, group in frame.groupby(["language", "label"], sort=True)]
    auc_delta = []
    tpr_delta = []
    for _ in range(replicates):
        sampled = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in strata])
        left_metrics = low_fpr_metrics(y[sampled], left[sampled])
        right_metrics = low_fpr_metrics(y[sampled], right[sampled])
        auc_delta.append(left_metrics["auc"] - right_metrics["auc"])
        tpr_delta.append(left_metrics["tpr_at_0.01_fpr"] - right_metrics["tpr_at_0.01_fpr"])
    def summarize(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(array.mean()),
            "lower_95": float(np.quantile(array, 0.025)),
            "upper_95": float(np.quantile(array, 0.975)),
        }
    return {"auc_delta": summarize(auc_delta), "tpr_at_0.01_fpr_delta": summarize(tpr_delta)}


def summarize_evaluation(
    frame: pd.DataFrame,
    hidden_scores: dict[str, np.ndarray],
    config: LumiaProbeConfig,
) -> dict[str, Any]:
    scores = {name: np.asarray(values, dtype=float) for name, values in hidden_scores.items()}
    for baseline in OUTPUT_BASELINES:
        scores[baseline] = frame[baseline].to_numpy(float)
    metrics = {}
    for name, values in scores.items():
        row = low_fpr_metrics(frame.label, values)
        row["conservative_tp_at_0.01_fpr"] = len(
            conservative_detection_set(frame.sample_id, frame.label, values, 0.01)
        )
        row["per_language_auc"] = per_language_auc(frame, values)
        metrics[name] = row
    reference = scores["zsigmoid_single_sequence"]
    overlap = {}
    reference_set = conservative_detection_set(frame.sample_id, frame.label, reference, 0.01)
    for name in HIDDEN_SCORE_NAMES:
        observed_set = conservative_detection_set(frame.sample_id, frame.label, scores[name], 0.01)
        union = observed_set | reference_set
        overlap[name] = {
            "true_positives": len(observed_set),
            "unique_vs_zsigmoid": len(observed_set - reference_set),
            "jaccard_vs_zsigmoid": len(observed_set & reference_set) / len(union) if union else 1.0,
        }
    bootstrap = {
        name: paired_bootstrap_delta(
            frame, scores[name], reference,
            replicates=config.bootstrap_replicates,
            seed=config.seed + 50000 + index,
        )
        for index, name in enumerate(HIDDEN_SCORE_NAMES)
    }
    return {"metrics": metrics, "overlap_vs_zsigmoid": overlap, "paired_bootstrap_vs_zsigmoid": bootstrap}
