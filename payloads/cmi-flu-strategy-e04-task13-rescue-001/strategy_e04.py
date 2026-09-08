"""Strategy-v2 E04: audited, anchor-preserving Task1.3 rescue (aggregate only).

The five predeclared conditions are evaluated without model/weight selection.
Strict repeated CV, historical LOSO and strict-CV-plus-proxy are distinct
estimands; the latter does NOT create another independent strict study.
"""
from __future__ import annotations

from dataclasses import replace
import inspect
from typing import Any

import numpy as np
import pandas as pd

from .strategy_e04_contract import (
    ALPHA, ASC, MAX_CORRECTION, MIN_SOURCE, PROXY_WEIGHTS, REENTRY, SEED,
    SHRINKAGE, MeasurementAudit, audit_measurements, bounded_residual,
    correlation, percentile, study_key,
)

EXPERIMENT = "strategy_v2_e04_task13_anchor_preserving_rescue"
ANCHOR_COLUMN = "flow_rank__" + ASC
FROZEN_STRICT_MODEL = "pls_1"
FROZEN_HISTORICAL_MODEL = "enet_a0.1_l0.5"
FROZEN_TARGET_ONLY_MODEL = "pls_3"
DELTA_THRESHOLD = 0.02
DECLINE_THRESHOLD = -0.10


def _summary(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    y = frame["target"].to_numpy(float)
    prediction = np.asarray(prediction, float)
    if len(y) != len(prediction) or not np.isfinite(prediction).all():
        raise ValueError("E04 summary has unaligned or nonfinite predictions")
    yr, pr = np.empty(len(y)), np.empty(len(y))
    studies = frame["study"].to_numpy()
    records = []
    for study in sorted(set(studies)):
        idx = np.flatnonzero(studies == study)
        yr[idx], pr[idx] = percentile(y[idx]), percentile(prediction[idx])
        metric = correlation(y[idx], prediction[idx])
        records.append({"study": study, "n": len(idx), "spearman": metric["value"], "status": metric["status"], "constant_prediction": bool(len(np.unique(prediction[idx])) < 2), "prediction_unique": int(len(np.unique(prediction[idx])))})
    finite = [row["spearman"] for row in records if row["status"] == "ok"]
    return {"rows": int(len(y)), "subjects": int(frame["subject_group"].nunique()), "studies": len(records),
            "study_equal_spearman": float(np.mean(finite)) if len(finite) == len(records) else None,
            "finite_study_mean_diagnostic": float(np.mean(finite)) if finite else None,
            "undefined_study_count": len(records) - len(finite), "study_scores": records,
            "pooled_spearman": correlation(y, prediction), "pooled_within_study_rank_spearman": correlation(yr, pr),
            "rank_rmse": float(np.sqrt(np.mean((yr-pr)**2)))}


def _paired(frame: pd.DataFrame, candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    c, r = _summary(frame, candidate), _summary(frame, reference)
    folds, delta, review = [], [], []
    for a, b in zip(c["study_scores"], r["study_scores"], strict=True):
        d = a["spearman"]-b["spearman"] if a["status"] == b["status"] == "ok" else None
        if d is not None:
            delta.append(d)
            if a["n"] >= 10 and d < DECLINE_THRESHOLD:
                review.append(a["study"])
        folds.append({"study": a["study"], "n": a["n"], "candidate": a["spearman"], "reference": b["spearman"], "delta": d, "candidate_status": a["status"], "reference_status": b["status"]})
    mean = float(np.mean(delta)) if len(delta) == len(folds) else None
    return {"paired_study_mean_delta": mean, "passes_delta_heuristic": mean is not None and mean >= DELTA_THRESHOLD,
            "large_studies_requiring_review": review, "all_studies_defined": len(delta) == len(folds), "study_scores": folds}


def _movement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    if reference.shape != candidate.shape:
        raise ValueError("E04 movement alignment mismatch")
    delta = np.abs(percentile(reference) - percentile(candidate))
    return {"n": len(reference), "rank_spearman": correlation(reference, candidate),
            "mean_absolute_percentile_shift": float(delta.mean()), "max_absolute_percentile_shift": float(delta.max()),
            "changed_rank_count": int(np.sum(delta > 1e-12)),
            "large_movement_review": bool(delta.mean() > 0.10 or delta.max() > 0.25)}


def _frame(dataset: Any, *, challenge: bool = False, require_anchor: bool = True) -> pd.DataFrame:
    raw = dataset.challenge if challenge else dataset.train
    out = raw[["participant_id", "subject_group", "study_group"]].copy().reset_index(drop=True)
    out["participant_id"] = out["participant_id"].astype(str)
    out["subject_group"] = out["subject_group"].astype(str)
    out["study"] = out["study_group"].map(study_key)
    out["anchor"] = pd.to_numeric(raw[ANCHOR_COLUMN], errors="coerce").to_numpy(float)
    if not challenge:
        out["target"] = pd.to_numeric(raw[dataset.target_column], errors="coerce").to_numpy(float)
    if out["participant_id"].duplicated().any() or out.duplicated(["study", "subject_group"]).any():
        raise ValueError("E04 requires one row per participant and study/subject")
    if (require_anchor and not np.isfinite(out["anchor"]).all()) or (not challenge and not np.isfinite(out["target"]).all()):
        raise ValueError("E04 frozen anchor or target nonfinite")
    return out


def _view(frame: pd.DataFrame, audit: MeasurementAudit, *, challenge: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = audit.eligibility.set_index(["participant_id", "study"])
    keys = pd.MultiIndex.from_frame(frame[["participant_id", "study"]])
    aligned = pairs.reindex(keys)
    eligible = aligned["valid_baseline"].fillna(False).to_numpy(bool) if challenge else aligned["within_gate_compatible"].fillna(False).to_numpy(bool)
    if not challenge:
        is_proxy = frame["study"].eq("SDY272").to_numpy()
        eligible &= ~is_proxy | aligned["proxy_allowed"].fillna(False).to_numpy(bool)
    else:
        eligible &= aligned["challenge_gate_compatible"].fillna(False).to_numpy(bool)
    columns = [frame["anchor"].to_numpy(float)]
    b = audit.baseline
    for name in audit.auxiliary_names:
        g = b.loc[b["name"].eq(name) & b["valid"]].copy()
        challenge_g = g.loc[g["study"].eq("2025LJI")]
        signature = challenge_g["signature"].iloc[0]
        g = g.loc[g["signature"].eq(signature)].copy()
        g["rank"] = g.groupby("study", observed=True)["value"].transform(lambda x: percentile(x.to_numpy(float)))
        feature = g.set_index(["participant_id", "study"])["rank"].reindex(keys).to_numpy(float)
        columns.append(feature)
    signature = aligned["signature_baseline"].fillna("missing").astype(str).to_numpy()
    return np.column_stack(columns), eligible, signature


def _align_prediction(dataset: Any, result: Any, *, challenge: bool = False) -> np.ndarray:
    if challenge:
        rows = result.challenge_predictions.set_index("participant_id")
        keys = dataset.challenge["participant_id"].astype(str)
        if rows.index.duplicated().any() or set(rows.index.astype(str)) != set(keys):
            raise ValueError("E04 frozen Challenge control alignment mismatch")
        return rows.reindex(keys)["prediction"].to_numpy(float)
    rows = result.oof_predictions
    if rows["row_index"].duplicated().any() or set(rows["row_index"].astype(int)) != set(range(len(dataset.train))):
        raise ValueError("E04 frozen OOF must aggregate repeats to one row per source")
    aligned = rows.set_index("row_index").reindex(range(len(dataset.train)))
    if not np.allclose(aligned["target"].to_numpy(float), dataset.train[dataset.target_column].to_numpy(float), rtol=0, atol=1e-10):
        raise ValueError("E04 frozen OOF target alignment mismatch")
    return aligned["prediction"].to_numpy(float)


def _purge(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    if set(train["subject_group"]) & set(validation["subject_group"]):
        raise ValueError("E04 subject purge violation")


def _fold_score(y: np.ndarray, pred: np.ndarray, base: np.ndarray, number: int) -> dict[str, Any]:
    c, b = correlation(y, pred), correlation(y, base)
    return {"split_index": number, "n": len(y), "spearman": c["value"], "status": c["status"],
            "anchor_spearman": b["value"], "anchor_status": b["status"],
            "delta_vs_anchor": c["value"]-b["value"] if c["status"] == b["status"] == "ok" else None,
            "constant_prediction": len(np.unique(pred)) < 2}


def _residual_oof(
    frame: pd.DataFrame, x: np.ndarray, eligible: np.ndarray, splits: list[Any], weight: float,
    *, proxy_frame: pd.DataFrame | None = None, proxy_x: np.ndarray | None = None,
    proxy_eligible: np.ndarray | None = None, shuffle: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    sums, counts = np.zeros(len(frame)), np.zeros(len(frame), int)
    folds, fits = [], []
    for number, split in enumerate(splits):
        ti, vi = np.asarray(split.train_indices, int), np.asarray(split.validation_indices, int)
        train, val = frame.iloc[ti].copy(), frame.iloc[vi]
        tx, good = x[ti], eligible[ti]
        if proxy_frame is not None:
            keep = ~proxy_frame["subject_group"].isin(set(val["subject_group"])).to_numpy()
            train = pd.concat([train, proxy_frame.loc[keep]], ignore_index=True)
            tx, good = np.vstack([tx, proxy_x[keep]]), np.r_[good, proxy_eligible[keep]]
        _purge(train, val)
        subject_count = int(train.loc[good, "subject_group"].nunique())
        if subject_count < MIN_SOURCE:
            good = np.zeros(len(train), bool)
        pred, diagnostic = bounded_residual(tx, x[vi], train["target"].to_numpy(float), train["anchor"].to_numpy(float), val["anchor"].to_numpy(float), train["study"].to_numpy(), proxy_weight=weight, eligible_train=good, eligible_prediction=eligible[vi], shuffle_seed=SEED+number if shuffle else None)
        diagnostic["source_subjects"] = subject_count
        diagnostic["subject_purge_passed"] = True
        sums[vi] += pred
        counts[vi] += 1
        fits.append(diagnostic)
        folds.append(_fold_score(val["target"].to_numpy(float), pred, val["anchor"].to_numpy(float), number))
    if not len(counts) or counts.min() < 1 or len(np.unique(counts)) != 1:
        raise ValueError("E04 OOF coverage/repeat count mismatch")
    return sums/counts, folds, fits


def _subgroups(frame: pd.DataFrame, candidate: np.ndarray, reference: np.ndarray, x: np.ndarray, eligible: np.ndarray, signatures: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for kind, groups in (("audited_feature_count", np.isfinite(x).sum(axis=1).astype(str)), ("gate_eligible", eligible.astype(int).astype(str)), ("gate_signature", signatures)):
        for study in sorted(frame["study"].unique()):
            for group in sorted(set(groups[frame["study"].eq(study)])):
                mask = frame["study"].eq(study).to_numpy() & (groups == group)
                n = int(mask.sum())
                record = {"study": study, "group_type": kind, "group": group, "n": n, "status": "suppressed_n_lt_5" if n < 5 else "evaluated"}
                if n >= 5:
                    record["candidate"] = correlation(frame.loc[mask, "target"], candidate[mask])
                    record["anchor"] = correlation(frame.loc[mask, "target"], reference[mask])
                rows.append(record)
    return rows


def _paired_split_scores(folds: list[dict[str, Any]], frozen: Any, splits: list[Any]) -> list[dict[str, Any]]:
    reference = frozen.fold_metrics.set_index("split")
    if reference.index.duplicated().any() or set(reference.index) != {s.name for s in splits}:
        raise ValueError("E04 frozen fold names do not match paired split contract")
    result = []
    for row, split in zip(folds, splits, strict=True):
        ref = reference.loc[split.name]
        if int(ref["n"]) != row["n"]:
            raise ValueError("E04 paired validation fold counts differ")
        value = float(ref["spearman"]) if ref["spearman_status"] == "ok" and np.isfinite(ref["spearman"]) else None
        item = dict(row, frozen_reference_spearman=value, frozen_reference_status=str(ref["spearman_status"]))
        item["delta_vs_frozen_reference"] = row["spearman"]-value if row["status"] == "ok" and value is not None else None
        result.append(item)
    return result


def _group_adjusted_diagnostic(frame: pd.DataFrame, prediction: np.ndarray, availability: np.ndarray) -> dict[str, Any]:
    yr, pr = np.empty(len(frame)), np.empty(len(frame))
    for study in sorted(frame["study"].unique()):
        pos = np.flatnonzero(frame["study"].eq(study).to_numpy())
        yr[pos], pr[pos] = percentile(frame.iloc[pos]["target"].to_numpy(float)), percentile(prediction[pos])
        for group in sorted(set(availability[pos])):
            idx = pos[availability[pos] == group]
            yr[idx] -= yr[idx].mean()
            pr[idx] -= pr[idx].mean()
    return {"metric": correlation(yr, pr), "scope": "within_study_rank_after_availability_group_centering", "interpretation": "diagnostic_not_independent_evidence"}


def _subset28(frame: pd.DataFrame, candidate: np.ndarray, reference: np.ndarray) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    output = []
    for study in sorted(frame["study"].unique()):
        idx = np.flatnonzero(frame["study"].eq(study).to_numpy())
        if len(idx) < 28:
            output.append({"study": study, "n": len(idx), "status": "not_run_n_lt_28"})
            continue
        delta = []
        for _ in range(200):
            sub = rng.choice(idx, size=28, replace=False)
            y = frame.iloc[sub]["target"].to_numpy(float)
            c, r = correlation(y, candidate[sub]), correlation(y, reference[sub])
            if c["status"] == r["status"] == "ok":
                delta.append(c["value"]-r["value"])
        output.append({"study": study, "n": len(idx), "status": "sensitivity_not_confidence_interval", "subset_n": 28, "repetitions": 200, "defined_repetitions": len(delta), "seed": SEED, "delta_p10": float(np.quantile(delta, 0.1)) if delta else None, "delta_median": float(np.median(delta)) if delta else None, "delta_p90": float(np.quantile(delta, 0.9)) if delta else None, "positive_fraction": float(np.mean(np.asarray(delta) > 0)) if delta else None})
    return output


def run_e04(config: Any, inputs: Any, *, expected_counts: tuple[int, int, int] | None = (23, 68, 40)) -> dict[str, Any]:
    """Run one frozen CPU E04; only aggregate dictionaries leave this function."""
    from .datasets import build_task_13_dataset
    from .evaluation import default_splits_for_task
    from .task13_harmonization import harmonize_sdy272_asc_proxy, to_within_study_rank_target
    from . import runner

    tables = inputs.tables
    audit = audit_measurements(tables["public_flow"], tables["challenge_flow"])
    # Audit has completed before any dataset/model construction and no y-based choice.
    common = (tables["challenge_flow"], tables["participants"], tables["investigations"])
    strict = build_task_13_dataset(tables["public_flow"], *common, mode="broad", include_sdy272_asc_proxy=False)
    harmonized_flow, _ = harmonize_sdy272_asc_proxy(tables["public_flow"])
    historical = build_task_13_dataset(harmonized_flow, *common, mode="broad", include_sdy272_asc_proxy=False)
    target_only = build_task_13_dataset(tables["public_flow"], *common, mode="broad", include_sdy272_asc_proxy=True)
    observed = (len(strict.train), len(historical.train), len(strict.challenge))
    if expected_counts is not None and observed != expected_counts:
        raise ValueError("E04 cohort counts differ from pinned input contract")
    sf, hf, cf = _frame(strict), _frame(historical), _frame(strict, challenge=True)
    if set(sf["study"]) != {"2024UGA"} or set(hf["study"]) != {"2024UGA", "SDY272"}:
        raise ValueError("E04 unexpected historical study membership")
    if not np.array_equal(historical.challenge["participant_id"].astype(str), strict.challenge["participant_id"].astype(str)):
        raise ValueError("E04 frozen control Challenge cohorts differ")
    strict_splits = default_splits_for_task(strict, random_state=config.random_state)
    historical_splits = default_splits_for_task(historical, random_state=config.random_state)
    for ds, splits in ((sf, strict_splits), (hf, historical_splits)):
        for split in splits:
            _purge(ds.iloc[split.train_indices], ds.iloc[split.validation_indices])
    specs = {s.name: s for s in config.model_specs("task_13")}
    control = runner.run_compact_task  # Explicit B2.1 adapter namespace, not legacy evaluation binding.
    if "selection_policy" not in inspect.signature(control).parameters:
        raise ValueError("E04 frozen runtime robust selector binding missing")
    b21 = control(strict, specs=[specs[FROZEN_STRICT_MODEL]], splits=strict_splits, random_state=config.random_state, selection_policy="robust_v1")
    ranked = to_within_study_rank_target(historical)
    rank_control = control(ranked, specs=[replace(specs[FROZEN_HISTORICAL_MODEL], target_transform="identity", clip_min=None)], splits=historical_splits, random_state=config.random_state, selection_policy="robust_v1")
    old_target_control = control(target_only, specs=[specs[FROZEN_TARGET_ONLY_MODEL]], random_state=config.random_state, selection_policy="robust_v1")
    b21_oof, historical_oof = _align_prediction(strict, b21), _align_prediction(ranked, rank_control)
    base = sf["anchor"].to_numpy(float)
    sx, se, ss = _view(sf, audit)
    hx, he, hs = _view(hf, audit)
    cx, ce, cs = _view(cf, audit, challenge=True)
    proxy_mask = hf["study"].eq("SDY272").to_numpy()
    # Prove the strict rows used by the supplemental experiment are identical.
    aligned = hf.set_index("participant_id").reindex(sf["participant_id"])
    if not np.allclose(aligned[["target", "anchor"]].to_numpy(float), sf[["target", "anchor"]].to_numpy(float), rtol=0, atol=1e-12):
        raise ValueError("E04 strict/harmonized strict-row ontology mismatch")
    strict_available = strict.train[strict.feature_columns(strict.train)].notna().sum(axis=1).to_numpy(float)
    out: dict[str, Any] = {
        "schema_version": 1, "experiment": EXPERIMENT, "task": "Task1.3", "status": "complete",
        "cv_contract": "paired_subject_purged_v2", "split_seed": int(config.random_state), "shuffle_seed": SEED,
        "competition_submission_attempted": False, "leaderboard_used_for_selection": False,
        "incumbent_changed": False, "automatic_compute_retries": 0,
        "measurement_audit": audit.aggregate,
        "frozen_conditions": {"strict_b21": FROZEN_STRICT_MODEL, "anchor": ANCHOR_COLUMN, "historical_rank": FROZEN_HISTORICAL_MODEL, "residual_alpha": ALPHA, "residual_shrinkage": SHRINKAGE, "residual_score_cap": MAX_CORRECTION, "proxy_weights": list(PROXY_WEIGHTS)},
        "strict_reference": _summary(sf, b21_oof), "strict_anchor": _summary(sf, base),
        "historical_rank_research_control": _summary(hf, historical_oof),
        "historical_target_only_diagnostic": _summary(_frame(target_only, require_anchor=False), _align_prediction(target_only, old_target_control)),
        "strict_anchor_vs_b21": _paired(sf, base, b21_oof),
        "availability_only_legacy": _summary(sf, strict_available),
        "availability_only_audited": _summary(sf, np.isfinite(sx).sum(axis=1)),
        "gate_eligibility_only": _summary(sf, se.astype(float)),
        "strict_cv": {}, "historical_loso": {}, "strict_cv_plus_proxy": {}, "challenge": {},
        "interpretation_limits": ["one_independent_strict_study", "two_historical_gate_domains", "proxy_is_not_strict_gate_equivalence", "single_shuffle_is_not_null_distribution", "score_cap_does_not_bound_rank_movement", "no_automatic_competition_promotion"],
        "reentry_conditions": list(REENTRY),
    }
    challenge_b21 = _align_prediction(strict, b21, challenge=True)
    challenge_old = _align_prediction(ranked, rank_control, challenge=True)
    out["challenge"]["frozen_historical_rank_vs_b21"] = _movement(challenge_b21, challenge_old)
    out["challenge"]["frozen_historical_rank_vs_anchor"] = _movement(cf["anchor"].to_numpy(float), challenge_old)
    out["challenge"]["anchor_vs_b21"] = _movement(challenge_b21, cf["anchor"].to_numpy(float))
    if expected_counts is not None:
        expected = ((out["strict_reference"]["study_equal_spearman"], 0.106304), (out["strict_anchor"]["study_equal_spearman"], 0.381589), (out["historical_rank_research_control"]["study_equal_spearman"], 0.365798), (out["availability_only_legacy"]["study_equal_spearman"], 0.422933))
        out["frozen_control_reproduction"] = {"tolerance": 0.000002, "all_pass": all(v is not None and abs(v-e) <= 0.000002 for v, e in expected)}
        if not out["frozen_control_reproduction"]["all_pass"]:
            out["status"] = "reference_mismatch_review_required"
            return out
    else:
        out["frozen_control_reproduction"] = {"all_pass": None, "status": "synthetic_not_real_data"}
    hist_predictions = []
    for weight in PROXY_WEIGHTS:
        name = f"proxy_weight_{weight:g}"
        primary, pfolds, pfits = _residual_oof(sf, sx, se, strict_splits, weight)
        hp, hfolds, hfits = _residual_oof(hf, hx, he, historical_splits, weight)
        plus, plusfolds, plusfits = _residual_oof(sf, sx, se, strict_splits, weight, proxy_frame=hf.loc[proxy_mask], proxy_x=hx[proxy_mask], proxy_eligible=he[proxy_mask])
        for group, frame, pred, x, eligible, sig, folds, fits, splits, reference in (
            ("strict_cv", sf, primary, sx, se, ss, pfolds, pfits, strict_splits, b21_oof),
            ("historical_loso", hf, hp, hx, he, hs, hfolds, hfits, historical_splits, historical_oof),
            ("strict_cv_plus_proxy", sf, plus, sx, se, ss, plusfolds, plusfits, strict_splits, b21_oof),
        ):
            kw = {"proxy_frame": hf.loc[proxy_mask], "proxy_x": hx[proxy_mask], "proxy_eligible": he[proxy_mask]} if group == "strict_cv_plus_proxy" else {}
            shuffled, _, _ = _residual_oof(frame, x, eligible, splits, weight, shuffle=True, **kw)
            anchor = frame["anchor"].to_numpy(float)
            out[group][name] = {"metrics": _summary(frame, pred), "paired_vs_frozen_reference": _paired(frame, pred, reference), "paired_vs_anchor": _paired(frame, pred, anchor),
                                "split_scores": _paired_split_scores(folds, rank_control if group == "historical_loso" else b21, splits), "fit_diagnostics": fits, "prediction_rank_agreement_with_reference": _movement(reference, pred), "prediction_rank_agreement_with_anchor": _movement(anchor, pred),
                                "label_shuffle": {"metrics": _summary(frame, shuffled), "paired_vs_anchor": _paired(frame, shuffled, anchor)},
                                "subgroups": _subgroups(frame, pred, anchor, x, eligible, sig),
                                "availability_adjusted_diagnostic": _group_adjusted_diagnostic(frame, pred, strict_available if group != "historical_loso" else np.isfinite(x).sum(axis=1)),
                                "subset_28_sensitivity_vs_anchor": _subset28(frame, pred, anchor)}
        # A Challenge correction additionally needs strict source metadata matching
        # its baseline gate. Proxy data alone do not establish this bridge.
        strict_matches = audit.eligibility.set_index(["participant_id", "study"])["challenge_gate_compatible"].reindex(pd.MultiIndex.from_frame(hf[["participant_id", "study"]])).fillna(False).to_numpy(bool)
        bridge_support = int(hf.loc[he & strict_matches & ~proxy_mask, "subject_group"].nunique())
        predict_eligible = ce if bridge_support >= MIN_SOURCE else np.zeros(len(cf), bool)
        challenge_prediction, fit = bounded_residual(hx, cx, hf["target"].to_numpy(float), hf["anchor"].to_numpy(float), cf["anchor"].to_numpy(float), hf["study"].to_numpy(), proxy_weight=weight, eligible_train=he, eligible_prediction=predict_eligible)
        movement = _movement(cf["anchor"].to_numpy(float), challenge_prediction)
        out["challenge"][name] = {"rows": len(cf), "strict_bridge_source_subjects": bridge_support, "fit": fit, "vs_anchor": movement, "vs_b21": _movement(challenge_b21, challenge_prediction), "vs_historical_rank": _movement(challenge_old, challenge_prediction)}
        strict_delta = out["strict_cv_plus_proxy"][name]["paired_vs_anchor"]["paired_study_mean_delta"]
        delta_b21 = out["strict_cv_plus_proxy"][name]["paired_vs_frozen_reference"]["paired_study_mean_delta"]
        historical_scores = out["historical_loso"][name]["metrics"]["study_scores"]
        historical_guard = all(row["status"] == "ok" and row["spearman"] > 0 for row in historical_scores) and not out["historical_loso"][name]["paired_vs_anchor"]["large_studies_requiring_review"]
        candidate = (strict_delta is not None and strict_delta >= DELTA_THRESHOLD and delta_b21 is not None and delta_b21 >= DELTA_THRESHOLD and historical_guard and not movement["large_movement_review"] and bridge_support >= MIN_SOURCE)
        out["challenge"][name]["local_candidate_heuristic"] = bool(candidate)
        out["challenge"][name]["decision"] = "candidate_requires_review" if candidate else ("data_limited" if fit["state"] == "data_limited" else "no_promotion")
        hist_predictions.append(hp)
    out["proxy_weight_identifiability"] = {"historical_loso_max_absolute_prediction_difference": float(np.max(np.abs(hist_predictions[0]-hist_predictions[1]))), "training_weights_mean_normalized": True, "one_source_study_cannot_identify_relative_proxy_weight": True, "mixed_training_evidence_is_sensitivity_only": True, "weight_selected": None}
    if any(len(np.unique(hf.iloc[s.train_indices]["study"])) != 1 for s in historical_splits):
        raise ValueError("E04 LOSO weight-identifiability assumption violated")
    if out["proxy_weight_identifiability"]["historical_loso_max_absolute_prediction_difference"] > 1e-12:
        raise ValueError("E04 uniform source weighting changed normalized Ridge unexpectedly")
    return out
