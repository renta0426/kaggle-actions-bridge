"""Frozen deployment-validation helpers for STAGE2-DEPLOYMENT-VALIDATION-V1.

This module contains only source-fitted / label-free scoring primitives. Holdout
labels are intentionally absent from every prediction API.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence
import hashlib
import json
import re

import numpy as np
import pandas as pd

from poisoned_chalice.lumia_hidden_state_probe import (
    LumiaProbeConfig,
    _make_probe,
    predict_probe,
    train_probe,
)

LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
SOURCE_ROWS = 5000
SOURCE_ROWS_PER_CELL = 500
DEPLOYMENT_ROWS = 5000
DEPLOYMENT_ROWS_PER_CELL = 500
DEPLOYMENT_SPLIT_SEED = 20260909
DEPLOYMENT_HOLDOUT_SEED = 20260911
PORTABLE_FEATURE_COUNT = 96
PORTABLE_GRID_POINTS = 16
PORTABLE_SUMMARIES = ("mean", "std", "min", "max", "q25", "q50", "q75", "slope")
SIMHASH_BITS = 64
SIMHASH_MAX_HAMMING = 6
SIMHASH_BLOCK_WIDTHS = (10, 9, 9, 9, 9, 9, 9)


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_membership(values: Sequence[object]) -> np.ndarray:
    normalized = pd.Series(values, dtype="string").str.strip().str.lower().str.replace("_", "-", regex=False)
    labels = normalized.map({"member": 1, "non-member": 0, "nonmember": 0})
    if labels.isna().any():
        raise ValueError("membership labels contain an unsupported value")
    return labels.to_numpy(dtype=int)


def validate_balanced_source_frame(frame: pd.DataFrame) -> None:
    required = {"sample_id", "language", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"source frame missing columns: {sorted(missing)}")
    if len(frame) != SOURCE_ROWS or frame.sample_id.duplicated().any():
        raise ValueError("source frame identity/coverage changed")
    counts = frame.groupby(["language", "label"]).size().to_dict()
    expected = {(language, label): SOURCE_ROWS_PER_CELL for language in LANGUAGES for label in (0, 1)}
    if counts != expected:
        raise ValueError(f"source frame language-label balance changed: {counts}")


def source_train_early_select_split(
    frame: pd.DataFrame,
    seed: int = DEPLOYMENT_SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reuse the historical 60/20/20 inner split recipe on the full source 5k."""
    from sklearn.model_selection import train_test_split

    validate_balanced_source_frame(frame)
    indices = np.arange(len(frame))
    strata = frame.language.astype(str) + "_" + frame.label.astype(int).astype(str)
    trainval, select = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed + 1000,
        stratify=strata.to_numpy(),
    )
    trainval_strata = strata.iloc[trainval].to_numpy()
    train, early = train_test_split(
        trainval,
        test_size=0.25,
        random_state=seed + 2000,
        stratify=trainval_strata,
    )
    if (len(train), len(early), len(select)) != (3000, 1000, 1000):
        raise RuntimeError("deployment source split sizes changed")
    for subset, rows_per_cell in ((train, 300), (early, 100), (select, 100)):
        counts = frame.iloc[subset].groupby(["language", "label"]).size().to_dict()
        expected = {(language, label): rows_per_cell for language in LANGUAGES for label in (0, 1)}
        if counts != expected:
            raise RuntimeError(f"deployment source stratification changed: {counts}")
    if set(train) & set(early) or set(train) & set(select) or set(early) & set(select):
        raise RuntimeError("deployment source split overlap detected")
    if set(train) | set(early) | set(select) != set(indices):
        raise RuntimeError("deployment source split coverage changed")
    return np.asarray(train), np.asarray(early), np.asarray(select)


def profile_features(values: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("portable profile must be [rows, depth]")
    _, depth = values.shape
    if depth < 2:
        raise ValueError(f"portable profile too short: {prefix} depth={depth}")
    x = np.linspace(0.0, 1.0, depth)
    grid = np.linspace(0.0, 1.0, PORTABLE_GRID_POINTS)
    interp = np.vstack([np.interp(grid, x, row) for row in values])
    q = np.quantile(values, [0.25, 0.50, 0.75], axis=1).T
    xm = x - x.mean()
    slope = ((values - values.mean(axis=1, keepdims=True)) @ xm) / float(np.dot(xm, xm))
    stats = np.column_stack([
        values.mean(axis=1),
        values.std(axis=1),
        values.min(axis=1),
        values.max(axis=1),
        q,
        slope,
    ])
    names = [f"{prefix}_depth_{i:02d}" for i in range(PORTABLE_GRID_POINTS)]
    names += [f"{prefix}_{name}" for name in PORTABLE_SUMMARIES]
    return np.column_stack([interp, stats]), names


def scalarize_hidden_mean_96(array: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Exact 96-feature mean-hidden scalarization used by the post-5k audit."""
    x = np.asarray(array, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("hidden array must be [rows, layers, hidden]")
    if x.shape[1] < 3 or x.shape[2] < 1:
        raise ValueError("hidden array dimensions are too small")
    eps = 1e-12
    rms = np.sqrt(np.mean(x * x, axis=2))
    left, right = x[:, :-1, :], x[:, 1:, :]
    left_norm = np.linalg.norm(left, axis=2)
    right_norm = np.linalg.norm(right, axis=2)
    adjacent_cos = np.sum(left * right, axis=2) / np.maximum(left_norm * right_norm, eps)
    delta = right - left
    rel_delta = np.linalg.norm(delta, axis=2) / np.maximum(right_norm, eps)
    dleft, dright = delta[:, :-1, :], delta[:, 1:, :]
    delta_direction_cos = np.sum(dleft * dright, axis=2) / np.maximum(
        np.linalg.norm(dleft, axis=2) * np.linalg.norm(dright, axis=2), eps
    )
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for values, prefix in (
        (rms, "rms"),
        (adjacent_cos, "adjacent_cos"),
        (rel_delta, "relative_delta_norm"),
        (delta_direction_cos, "delta_direction_cos"),
    ):
        block, block_names = profile_features(values, prefix)
        blocks.append(block)
        names.extend(block_names)
    result = np.column_stack(blocks)
    if result.shape != (x.shape[0], PORTABLE_FEATURE_COUNT) or len(names) != PORTABLE_FEATURE_COUNT:
        raise RuntimeError("portable 96-feature schema changed")
    if len(set(names)) != PORTABLE_FEATURE_COUNT or not np.isfinite(result).all():
        raise RuntimeError("portable 96-feature finite/identity check failed")
    return result, names


def fit_gr_source(
    mean_hidden: np.ndarray,
    labels: Sequence[int],
    *,
    seed: int = DEPLOYMENT_SPLIT_SEED,
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit exact median -> StandardScaler -> LR(C=.2) GR on source only."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, names = scalarize_hidden_mean_96(mean_hidden)
    y = np.asarray(labels, dtype=int)
    if X.shape[0] != len(y) or sorted(np.unique(y).tolist()) != [0, 1]:
        raise ValueError("GR source labels/coverage invalid")
    imputer = SimpleImputer(strategy="median")
    Xi = imputer.fit_transform(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xi)
    learner = LogisticRegression(C=0.2, max_iter=2000, random_state=seed)
    learner.fit(Xs, y)
    direct = learner.predict_proba(Xs)[:, 1]
    bundle = {
        "schema_version": 1,
        "feature_names": names,
        "imputer_statistics": np.asarray(imputer.statistics_, dtype=np.float64),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "coef": np.asarray(learner.coef_[0], dtype=np.float64),
        "intercept": float(learner.intercept_[0]),
        "learner": {"type": "LogisticRegression", "C": 0.2, "max_iter": 2000, "random_state": seed},
        "score_direction": "higher_is_member",
    }
    replay = predict_gr_from_features(bundle, X)
    if not np.allclose(direct, replay, rtol=0.0, atol=1e-12):
        raise RuntimeError("GR exported bundle replay mismatch")
    return bundle, direct


def predict_gr_from_features(bundle: dict[str, Any], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64).copy()
    if X.ndim != 2 or X.shape[1] != PORTABLE_FEATURE_COUNT:
        raise ValueError("GR feature shape changed")
    stats = np.asarray(bundle["imputer_statistics"], dtype=np.float64)
    if stats.shape != (PORTABLE_FEATURE_COUNT,):
        raise ValueError("GR imputer shape changed")
    rows, cols = np.where(~np.isfinite(X))
    if len(rows):
        X[rows, cols] = stats[cols]
    mean = np.asarray(bundle["scaler_mean"], dtype=np.float64)
    scale = np.asarray(bundle["scaler_scale"], dtype=np.float64)
    coef = np.asarray(bundle["coef"], dtype=np.float64)
    if np.any(scale <= 0) or mean.shape != scale.shape or coef.shape != scale.shape:
        raise ValueError("GR scaler/coef shape changed")
    standardized = (X - mean) / scale
    decision = standardized @ coef + float(bundle["intercept"])
    probability = np.empty_like(decision, dtype=np.float64)
    positive = decision >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-decision[positive]))
    exp_value = np.exp(decision[~positive])
    probability[~positive] = exp_value / (1.0 + exp_value)
    if not np.isfinite(probability).all():
        raise RuntimeError("GR prediction contains nonfinite values")
    return probability


def predict_gr(bundle: dict[str, Any], mean_hidden: np.ndarray) -> np.ndarray:
    X, names = scalarize_hidden_mean_96(mean_hidden)
    if names != list(bundle["feature_names"]):
        raise RuntimeError("GR feature order changed")
    return predict_gr_from_features(bundle, X)


def save_gr_bundle(bundle: dict[str, Any], root: str | Path) -> dict[str, str]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    array_fields = {"imputer_statistics", "scaler_mean", "scaler_scale", "coef"}
    metadata = {k: v for k, v in bundle.items() if k not in array_fields}
    metadata_path = root / "gr_metadata.json"
    weights_path = root / "gr_parameters.npz"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        weights_path,
        imputer_statistics=np.asarray(bundle["imputer_statistics"], dtype=np.float64),
        scaler_mean=np.asarray(bundle["scaler_mean"], dtype=np.float64),
        scaler_scale=np.asarray(bundle["scaler_scale"], dtype=np.float64),
        coef=np.asarray(bundle["coef"], dtype=np.float64),
    )
    return {"metadata_sha256": sha256_file(metadata_path), "parameters_sha256": sha256_file(weights_path)}


def load_gr_bundle(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    bundle = json.loads((root / "gr_metadata.json").read_text(encoding="utf-8"))
    with np.load(root / "gr_parameters.npz", allow_pickle=False) as archive:
        for name in ("imputer_statistics", "scaler_mean", "scaler_scale", "coef"):
            bundle[name] = np.asarray(archive[name], dtype=np.float64)
    return bundle


def fit_hr_mean_top5_source(
    mean_hidden: np.ndarray,
    frame: pd.DataFrame,
    *,
    config: LumiaProbeConfig | None = None,
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train all layers on source train/early, select top-5 on source select."""
    from sklearn.metrics import roc_auc_score

    config = config or LumiaProbeConfig()
    validate_balanced_source_frame(frame)
    X = np.asarray(mean_hidden, dtype=np.float32)
    if X.shape[0] != len(frame) or X.ndim != 3:
        raise ValueError("HR mean-hidden/source-frame shape changed")
    train_index, early_index, select_index = source_train_early_select_split(frame, config.seed)
    y = frame.label.to_numpy(dtype=int)
    selection: list[tuple[int, float]] = []
    state_by_layer: dict[int, dict[str, np.ndarray]] = {}
    layer_records: list[dict[str, Any]] = []
    for layer in range(X.shape[1]):
        seed = config.seed + layer
        probe, training_record = train_probe(
            X[train_index, layer, :],
            y[train_index],
            X[early_index, layer, :],
            y[early_index],
            config=config,
            seed=seed,
            device=device,
        )
        select_prediction = predict_probe(
            probe,
            X[select_index, layer, :],
            batch_size=config.batch_size,
            device=device,
        )
        selection_auc = float(roc_auc_score(y[select_index], select_prediction))
        selection.append((layer, selection_auc))
        state_by_layer[layer] = {
            key: value.detach().cpu().numpy().copy() for key, value in probe.state_dict().items()
        }
        layer_records.append({"layer": layer, "selection_auc": selection_auc, **training_record, "seed": seed})
        del probe
    ordered = sorted(selection, key=lambda item: (-item[1], item[0]))
    top_layers = [int(layer) for layer, _ in ordered[: config.ensemble_size]]
    bundle = {
        "schema_version": 1,
        "input_dim": int(X.shape[2]),
        "hidden_dim": int(config.hidden_dim),
        "top_layers": top_layers,
        "state_dicts": {str(layer): state_by_layer[layer] for layer in top_layers},
        "config": asdict(config),
        "source_split": {
            "train_rows": int(len(train_index)),
            "early_rows": int(len(early_index)),
            "select_rows": int(len(select_index)),
            "seed": int(config.seed),
            "outer_oof_holdout_rows": 0,
        },
        "score_direction": "higher_is_member",
        "ensemble": "arithmetic_mean_of_top5_probe_probabilities",
        "pooling": "mean",
    }
    audit = {"top_layers": top_layers, "layer_records": layer_records}
    return bundle, audit


def _probe_from_numpy_state(
    state: dict[str, np.ndarray],
    input_dim: int,
    hidden_dim: int,
    device: str,
):
    import torch

    probe = _make_probe(torch, input_dim, hidden_dim).to(device)
    tensor_state = {name: torch.as_tensor(value, device=device) for name, value in state.items()}
    probe.load_state_dict(tensor_state)
    probe.eval()
    return probe


def predict_hr(bundle: dict[str, Any], mean_hidden: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    X = np.asarray(mean_hidden, dtype=np.float32)
    if X.ndim != 3 or X.shape[2] != int(bundle["input_dim"]):
        raise ValueError("HR mean-hidden shape changed")
    config = LumiaProbeConfig(**bundle["config"])
    predictions = []
    for layer in bundle["top_layers"]:
        state = bundle["state_dicts"][str(layer)]
        probe = _probe_from_numpy_state(
            state,
            int(bundle["input_dim"]),
            int(bundle["hidden_dim"]),
            device,
        )
        predictions.append(
            predict_probe(
                probe,
                X[:, int(layer), :],
                batch_size=config.batch_size,
                device=device,
            )
        )
        del probe
    result = np.mean(np.stack(predictions, axis=0), axis=0)
    if result.shape != (len(X),) or not np.isfinite(result).all():
        raise RuntimeError("HR prediction coverage/finite check failed")
    return result


def save_hr_bundle(bundle: dict[str, Any], root: str | Path) -> dict[str, str]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {k: v for k, v in bundle.items() if k != "state_dicts"}
    metadata_path = root / "hr_metadata.json"
    weights_path = root / "hr_parameters.npz"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arrays = {}
    for layer in bundle["top_layers"]:
        for name, value in bundle["state_dicts"][str(layer)].items():
            arrays[f"layer_{int(layer):02d}__{name}"] = np.asarray(value)
    np.savez_compressed(weights_path, **arrays)
    return {"metadata_sha256": sha256_file(metadata_path), "parameters_sha256": sha256_file(weights_path)}


def load_hr_bundle(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    bundle = json.loads((root / "hr_metadata.json").read_text(encoding="utf-8"))
    states: dict[str, dict[str, np.ndarray]] = {str(layer): {} for layer in bundle["top_layers"]}
    with np.load(root / "hr_parameters.npz", allow_pickle=False) as archive:
        for layer in bundle["top_layers"]:
            prefix = f"layer_{int(layer):02d}__"
            matched = [key for key in archive.files if key.startswith(prefix)]
            if not matched:
                raise ValueError(f"HR saved state missing layer {layer}")
            for key in matched:
                states[str(layer)][key[len(prefix):]] = np.asarray(archive[key])
    bundle["state_dicts"] = states
    return bundle


def global_rank01(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("global rank input must be finite 1-D")
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def equal_rank_fusion(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("rank-fusion score lengths differ")
    return 0.5 * global_rank01(left) + 0.5 * global_rank01(right)


def build_prediction_frame(
    sample_ids: Sequence[str],
    languages: Sequence[str],
    **scores: Sequence[float],
) -> pd.DataFrame:
    frame = pd.DataFrame({"sample_id": list(sample_ids), "language": list(languages)})
    if frame.sample_id.duplicated().any():
        raise ValueError("prediction identity contains duplicate sample IDs")
    for name, values in scores.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(frame),) or not np.isfinite(array).all():
            raise ValueError(f"prediction column invalid: {name}")
        frame[name] = array
    return frame


TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|==|!=|<=|>=|=>|->|::|&&|\|\||\S"
)


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.sz = [1] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.p[b] = a
        self.sz[a] += self.sz[b]


def simhash64(text: str) -> int:
    tokens = TOKEN_RE.findall(str(text))
    if len(tokens) < 5:
        shingles = [" ".join(tokens)]
    else:
        total = len(tokens) - 4
        starts = range(total) if total <= 256 else np.linspace(0, total - 1, 256, dtype=int)
        shingles = [" ".join(tokens[int(i):int(i) + 5]) for i in starts]
    digests = b"".join(
        hashlib.blake2b(shingle.encode("utf-8", errors="ignore"), digest_size=8).digest()
        for shingle in shingles
    )
    if not digests:
        return 0
    bits = np.unpackbits(
        np.frombuffer(digests, dtype=np.uint8).reshape(-1, 8),
        axis=1,
        bitorder="little",
    )
    majority = bits.sum(axis=0) * 2 >= bits.shape[0]
    value = 0
    for bit, flag in enumerate(majority.tolist()):
        if flag:
            value |= 1 << bit
    return value


def _simhash_block_values(value: int) -> tuple[int, ...]:
    """Seven disjoint blocks make Hamming<=6 candidate generation exact.

    By the pigeonhole principle, if at most six of 64 bits differ, at least one
    of seven disjoint blocks is unchanged. Therefore every true <=6 pair shares
    at least one bucket and is compared below; no bucket-size cutoff is allowed.
    """
    if sum(SIMHASH_BLOCK_WIDTHS) != SIMHASH_BITS or len(SIMHASH_BLOCK_WIDTHS) != SIMHASH_MAX_HAMMING + 1:
        raise RuntimeError("SimHash exact-blocking contract changed")
    output = []
    offset = 0
    for width in SIMHASH_BLOCK_WIDTHS:
        output.append((int(value) >> offset) & ((1 << width) - 1))
        offset += width
    return tuple(output)


def _content_groups_from_hashes(hashes: Sequence[int]) -> tuple[np.ndarray, dict[str, Any]]:
    hashes = [int(value) & ((1 << SIMHASH_BITS) - 1) for value in hashes]
    block_values = [_simhash_block_values(value) for value in hashes]
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, values in enumerate(block_values):
        for block_index, block_value in enumerate(values):
            buckets[(block_index, block_value)].append(idx)

    dsu = DSU(len(hashes))
    candidate_pairs_checked = 0
    for (block_index, _), items in buckets.items():
        for pos, left in enumerate(items):
            for right in items[pos + 1:]:
                # A pair can share several blocks. Compare it only in the first
                # shared block, avoiding a large global `checked` set while
                # keeping exhaustive candidate coverage.
                if any(
                    block_values[left][previous] == block_values[right][previous]
                    for previous in range(block_index)
                ):
                    continue
                candidate_pairs_checked += 1
                if (hashes[left] ^ hashes[right]).bit_count() <= SIMHASH_MAX_HAMMING:
                    dsu.union(left, right)

    groups = np.asarray([dsu.find(i) for i in range(len(hashes))], dtype=int)
    sizes = pd.Series(groups).value_counts()
    return groups, {
        "method": "token_5gram_simhash64_hamming_le_6",
        "blocking": "7_disjoint_blocks_exact_for_hamming_le_6_no_bucket_cutoff",
        "block_widths": list(SIMHASH_BLOCK_WIDTHS),
        "non_singleton_groups": int((sizes > 1).sum()),
        "rows_in_non_singleton_groups": int(sizes[sizes > 1].sum()) if (sizes > 1).any() else 0,
        "max_group_size": int(sizes.max()) if len(sizes) else 0,
        "candidate_pairs_checked": int(candidate_pairs_checked),
    }


def content_groups(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    if "content" not in frame:
        raise ValueError("content_groups requires content")
    hashes = [simhash64(text) for text in frame.content]
    return _content_groups_from_hashes(hashes)
