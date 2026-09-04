"""Gold controlled architecture-transfer scoring and attack selection.

This module is deliberately downstream of model training.  It scores checkpoints
without membership labels, retains a small predeclared primitive matrix, fits or
selects an attack using only the Shadow-A development partition, freezes that
attack, and applies it unchanged to the paired Shadow-A/Shadow-B holdout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .shadow_protocol import ByteCodeTokenizer
from .shadow_scoring import load_scoring_bundle, load_shadow_checkpoint
from .stage2_api import Stage2RuntimeConfig, score_samples_detailed


SHADOW_GOLD_TRANSFER_VERSION = "shadow-gold-architecture-transfer-v1"
GOLD_PRIMITIVES: Mapping[str, tuple[str, float]] = {
    "loss": ("score_logp_mean__max", 1.0),
    "min_kpp": ("min_kpp_zselect_10__max", 1.0),
    "local_64": ("best_local_64__max", 1.0),
    "log_rank": ("mean_log_rank__mean", -1.0),
}
GOLD_CANDIDATE_ORDER = ("loss", "min_kpp", "local_64", "log_rank", "stage2_v1", "logistic4")


class ShadowGoldTransferError(RuntimeError):
    """Raised when the Gold transfer boundary or frozen feature contract fails."""


@dataclass(frozen=True)
class GoldScoringConfig:
    max_batch_tokens: int = 4096
    vocab_chunk_tokens: int = 64
    rank_vocab_block_size: int = 256
    language_calibration_min_rows: int = 20
    length_calibration_min_rows: int = 12
    fidelity_tokens: int = 24
    fidelity_atol: float = 1e-5


@dataclass(frozen=True)
class FrozenGoldAttack:
    version: str
    selected_candidate: str
    primitive_columns: tuple[str, ...]
    primitive_orientations: tuple[float, ...]
    scaler_mean: tuple[float, ...] | None
    scaler_scale: tuple[float, ...] | None
    logistic_coef: tuple[float, ...] | None
    logistic_intercept: float | None
    development_metrics: Mapping[str, Mapping[str, float]]
    development_rows: int
    holdout_rows: int
    selection_rule: str


def _stable_hash(seed: int, value: str) -> int:
    payload = f"{seed}\0{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def score_gold_shadow_features(
    training_output_directory: str | Path,
    scoring_bundle_directory: str | Path,
    *,
    backend: str = "cuda",
    config: GoldScoringConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return content-free primitive features without reading membership labels."""

    runtime = config or GoldScoringConfig()
    bundle = load_scoring_bundle(scoring_bundle_directory)
    checkpoint = load_shadow_checkpoint(training_output_directory, backend=backend)
    stage2 = Stage2RuntimeConfig(
        max_length=checkpoint.architecture.max_position_embeddings,
        max_batch_tokens=runtime.max_batch_tokens,
        vocab_chunk_tokens=runtime.vocab_chunk_tokens,
        rank_vocab_block_size=runtime.rank_vocab_block_size,
        language_calibration_min_rows=runtime.language_calibration_min_rows,
        length_calibration_min_rows=runtime.length_calibration_min_rows,
        fidelity_gate=True,
        fidelity_tokens=runtime.fidelity_tokens,
        fidelity_atol=runtime.fidelity_atol,
        device="auto",
        move_model=False,
    )
    source = bundle.frame.reset_index(drop=True)
    detailed = score_samples_detailed(
        model=checkpoint.model,
        tokenizer=ByteCodeTokenizer(),
        samples=source.content.astype(str).tolist(),
        languages=source.language.astype(str).tolist(),
        runtime_config=stage2,
    )
    features = detailed.features.sort_values("sample_index").reset_index(drop=True)
    if features.sample_index.tolist() != list(range(len(source))):
        raise ShadowGoldTransferError("Stage2 detailed features lost original row order")
    required = [column for column, _ in GOLD_PRIMITIVES.values()]
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise ShadowGoldTransferError(f"predeclared primitive columns missing: {missing}")
    output = source[["benchmark_id", "language", "length_bin", "character_count"]].copy()
    for name, (column, orientation) in GOLD_PRIMITIVES.items():
        values = features[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ShadowGoldTransferError(f"non-finite primitive: {name}")
        output[name] = values * float(orientation)
    generic = np.asarray(detailed.scores, dtype=np.float64)
    if len(generic) != len(output) or not np.isfinite(generic).all():
        raise ShadowGoldTransferError("generic Stage2 score is incomplete or non-finite")
    output["stage2_v1"] = generic
    if output.benchmark_id.duplicated().any():
        raise ShadowGoldTransferError("feature output benchmark IDs are not unique")
    manifest = {
        "status": "sealed",
        "version": SHADOW_GOLD_TRANSFER_VERSION,
        "architecture_slot": checkpoint.training_manifest["architecture_slot"],
        "rows": len(output),
        "primitive_contract": {
            name: {"source_column": column, "orientation": orientation}
            for name, (column, orientation) in GOLD_PRIMITIVES.items()
        },
        "generic_stage2_score_retained": True,
        "membership_labels_read": False,
        "benchmark_ids_passed_to_model_scorer": False,
        "evaluation_input_sha256": bundle.input_sha256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "model_state_sha256": checkpoint.model_state_sha256,
        "stage2_manifest": detailed.manifest,
    }
    return output, manifest


def attach_exact_labels(feature_frame: pd.DataFrame, label_frame: pd.DataFrame) -> pd.DataFrame:
    required_labels = {"benchmark_id", "membership", "matched_pair_id"}
    if set(label_frame.columns) != required_labels:
        raise ShadowGoldTransferError("Gold label frame columns changed")
    labels = label_frame.copy()
    labels["membership"] = labels.membership.astype(int)
    if set(labels.membership.unique()) != {0, 1}:
        raise ShadowGoldTransferError("Gold labels are not binary")
    if labels.benchmark_id.duplicated().any():
        raise ShadowGoldTransferError("Gold label benchmark IDs are not unique")
    merged = feature_frame.merge(labels, on="benchmark_id", how="inner", validate="one_to_one")
    if len(merged) != len(feature_frame) or len(merged) != len(labels):
        raise ShadowGoldTransferError("Gold feature/label coverage mismatch")
    return merged


def split_development_holdout(frame: pd.DataFrame, *, seed: int = 2027) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split matched pairs exactly in half within each language."""

    required = {"language", "membership", "matched_pair_id"}
    if not required.issubset(frame.columns):
        raise ShadowGoldTransferError("Gold split frame lacks language/membership/pair metadata")
    pair_rows = []
    for pair_id, group in frame.groupby("matched_pair_id", sort=False):
        if len(group) != 2 or set(group.membership.astype(int)) != {0, 1}:
            raise ShadowGoldTransferError(f"matched pair is not one member plus one nonmember: {pair_id}")
        languages = set(group.language.astype(str))
        if len(languages) != 1:
            raise ShadowGoldTransferError(f"matched pair crosses languages: {pair_id}")
        pair_rows.append((str(pair_id), next(iter(languages))))
    pairs = pd.DataFrame(pair_rows, columns=["matched_pair_id", "language"])
    development_pairs: set[str] = set()
    holdout_pairs: set[str] = set()
    for language, group in pairs.groupby("language", sort=True):
        ordered = sorted(
            group.matched_pair_id.astype(str).tolist(),
            key=lambda value: (_stable_hash(seed, f"{language}\0{value}"), value),
        )
        if len(ordered) % 2:
            raise ShadowGoldTransferError(f"matched-pair count is not even for {language}")
        midpoint = len(ordered) // 2
        development_pairs.update(ordered[:midpoint])
        holdout_pairs.update(ordered[midpoint:])
    if development_pairs & holdout_pairs:
        raise ShadowGoldTransferError("development/holdout matched pairs overlap")
    development = frame[frame.matched_pair_id.astype(str).isin(development_pairs)].copy()
    holdout = frame[frame.matched_pair_id.astype(str).isin(holdout_pairs)].copy()
    if len(development) != len(holdout) or len(development) + len(holdout) != len(frame):
        raise ShadowGoldTransferError("development/holdout coverage is not an exact half split")
    for subset in (development, holdout):
        counts = subset.groupby(["language", "membership"]).size()
        if counts.nunique() != 1:
            raise ShadowGoldTransferError("Gold split lost language/class balance")
    return development.reset_index(drop=True), holdout.reset_index(drop=True)


def _metrics(y: np.ndarray, score: np.ndarray, *, false_positive_rate: float = 0.01) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=np.float64)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(score).all():
        raise ShadowGoldTransferError("metric inputs are invalid")
    auc = float(roc_auc_score(y, score))
    pauc = float(roc_auc_score(y, score, max_fpr=false_positive_rate))
    fpr, tpr, _ = roc_curve(y, score)
    eligible = tpr[fpr <= false_positive_rate + 1e-12]
    tpr_at = float(eligible.max()) if len(eligible) else 0.0
    return {"auc": auc, "partial_auc_standardized": pauc, "tpr_at_fpr": tpr_at}


def _logistic_oof(development: pd.DataFrame, columns: list[str], *, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = development[columns].to_numpy(dtype=np.float64)
    y = development.membership.to_numpy(dtype=int)
    pair_ids = development.matched_pair_id.astype(str).tolist()
    folds = np.asarray([_stable_hash(seed + 71, value) % 5 for value in pair_ids], dtype=int)
    output = np.full(len(development), np.nan, dtype=np.float64)
    for fold in range(5):
        test = folds == fold
        train = ~test
        if not test.any() or set(np.unique(y[train])) != {0, 1}:
            raise ShadowGoldTransferError("deterministic logistic fold is empty or single-class")
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
        model.fit(scaler.transform(x[train]), y[train])
        output[test] = model.predict_proba(scaler.transform(x[test]))[:, 1]
    if not np.isfinite(output).all():
        raise ShadowGoldTransferError("logistic OOF prediction coverage failed")
    return output


def fit_frozen_gold_attack(
    left_labeled_features: pd.DataFrame,
    *,
    seed: int = 2027,
    false_positive_rate: float = 0.01,
) -> tuple[FrozenGoldAttack, pd.DataFrame]:
    """Fit/select on Shadow-A development rows only and return untouched holdout."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    development, holdout = split_development_holdout(left_labeled_features, seed=seed)
    primitive_columns = list(GOLD_PRIMITIVES)
    candidate_scores: dict[str, np.ndarray] = {
        name: development[name].to_numpy(dtype=np.float64)
        for name in primitive_columns
    }
    candidate_scores["stage2_v1"] = development.stage2_v1.to_numpy(dtype=np.float64)
    candidate_scores["logistic4"] = _logistic_oof(development, primitive_columns, seed=seed)
    y = development.membership.to_numpy(dtype=int)
    metrics = {
        name: _metrics(y, score, false_positive_rate=false_positive_rate)
        for name, score in candidate_scores.items()
    }
    order_index = {name: index for index, name in enumerate(GOLD_CANDIDATE_ORDER)}
    selected = max(
        GOLD_CANDIDATE_ORDER,
        key=lambda name: (
            metrics[name]["partial_auc_standardized"],
            metrics[name]["tpr_at_fpr"],
            metrics[name]["auc"],
            -order_index[name],
        ),
    )
    scaler_mean = scaler_scale = logistic_coef = None
    logistic_intercept = None
    if selected == "logistic4":
        x = development[primitive_columns].to_numpy(dtype=np.float64)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
        model.fit(scaler.transform(x), y)
        scaler_mean = tuple(float(value) for value in scaler.mean_)
        scaler_scale = tuple(float(value) for value in scaler.scale_)
        logistic_coef = tuple(float(value) for value in model.coef_[0])
        logistic_intercept = float(model.intercept_[0])
    attack = FrozenGoldAttack(
        version=SHADOW_GOLD_TRANSFER_VERSION,
        selected_candidate=selected,
        primitive_columns=tuple(primitive_columns),
        primitive_orientations=tuple(float(GOLD_PRIMITIVES[name][1]) for name in primitive_columns),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        logistic_coef=logistic_coef,
        logistic_intercept=logistic_intercept,
        development_metrics=metrics,
        development_rows=len(development),
        holdout_rows=len(holdout),
        selection_rule="maximize development 5-fold-OOF standardized pAUC@1% FPR; then TPR@1%; then AUC; fixed candidate order tie-break",
    )
    return attack, holdout


def apply_frozen_gold_attack(feature_frame: pd.DataFrame, attack: FrozenGoldAttack) -> np.ndarray:
    selected = attack.selected_candidate
    if selected in GOLD_PRIMITIVES or selected == "stage2_v1":
        if selected not in feature_frame.columns:
            raise ShadowGoldTransferError(f"frozen candidate column missing: {selected}")
        score = feature_frame[selected].to_numpy(dtype=np.float64)
    elif selected == "logistic4":
        if None in (attack.scaler_mean, attack.scaler_scale, attack.logistic_coef, attack.logistic_intercept):
            raise ShadowGoldTransferError("frozen logistic parameters are incomplete")
        x = feature_frame[list(attack.primitive_columns)].to_numpy(dtype=np.float64)
        mean = np.asarray(attack.scaler_mean, dtype=np.float64)
        scale = np.asarray(attack.scaler_scale, dtype=np.float64)
        coef = np.asarray(attack.logistic_coef, dtype=np.float64)
        z = ((x - mean) / scale) @ coef + float(attack.logistic_intercept)
        score = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    else:
        raise ShadowGoldTransferError(f"unknown frozen candidate: {selected}")
    if not np.isfinite(score).all():
        raise ShadowGoldTransferError("frozen attack produced non-finite scores")
    return score


def evaluate_frozen_gold_holdout(
    labeled_holdout: pd.DataFrame,
    score: np.ndarray,
    *,
    false_positive_rate: float = 0.01,
) -> dict[str, Any]:
    if len(labeled_holdout) != len(score):
        raise ShadowGoldTransferError("holdout score length mismatch")
    result: dict[str, Any] = {
        "overall": _metrics(
            labeled_holdout.membership.to_numpy(dtype=int),
            np.asarray(score, dtype=np.float64),
            false_positive_rate=false_positive_rate,
        ),
        "rows": len(labeled_holdout),
        "false_positive_rate": false_positive_rate,
        "by_language": {},
        "by_length_bin": {},
    }
    scored = labeled_holdout.copy()
    scored["frozen_score"] = np.asarray(score, dtype=np.float64)
    for language, group in scored.groupby("language", sort=True):
        result["by_language"][str(language)] = _metrics(
            group.membership.to_numpy(dtype=int), group.frozen_score.to_numpy(dtype=np.float64),
            false_positive_rate=false_positive_rate,
        )
    for length_bin, group in scored.groupby("length_bin", sort=True):
        if set(group.membership.astype(int).unique()) == {0, 1}:
            result["by_length_bin"][str(length_bin)] = _metrics(
                group.membership.to_numpy(dtype=int), group.frozen_score.to_numpy(dtype=np.float64),
                false_positive_rate=false_positive_rate,
            )
    pairs = []
    for _, group in scored.groupby("matched_pair_id", sort=False):
        member = group.loc[group.membership == 1, "frozen_score"]
        nonmember = group.loc[group.membership == 0, "frozen_score"]
        if len(member) == 1 and len(nonmember) == 1:
            delta = float(member.iloc[0] - nonmember.iloc[0])
            pairs.append(1.0 if delta > 0 else (0.5 if delta == 0 else 0.0))
    result["matched_pair_accuracy"] = float(np.mean(pairs)) if pairs else float("nan")
    return result


def frozen_attack_to_dict(attack: FrozenGoldAttack) -> dict[str, Any]:
    return asdict(attack)


__all__ = [
    "GOLD_CANDIDATE_ORDER",
    "GOLD_PRIMITIVES",
    "GoldScoringConfig",
    "FrozenGoldAttack",
    "ShadowGoldTransferError",
    "apply_frozen_gold_attack",
    "attach_exact_labels",
    "evaluate_frozen_gold_holdout",
    "fit_frozen_gold_attack",
    "frozen_attack_to_dict",
    "score_gold_shadow_features",
    "split_development_holdout",
]
