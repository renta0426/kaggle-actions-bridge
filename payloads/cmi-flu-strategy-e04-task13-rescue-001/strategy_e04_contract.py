"""Outcome-independent E04 measurement contract and bounded residual primitive.

All identifiers in private return values are transient alignment keys. Only the
``aggregate`` member of the audit is exportable. Marker absence means not reported
in the supplied definition, NOT a negative stain or an unmeasured FCS channel.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ASC = "Antibody-secreting_cells_(ASC)"
MARKERS = ("CD19", "CD20", "CD27", "CD38", "CD56", "CD71")
STUDIES = ("2024UGA", "SDY272", "2025LJI")
ALPHA = 10.0
SHRINKAGE = 0.25
MAX_CORRECTION = 0.05
PROXY_WEIGHTS = (0.25, 1.0)
SEED = 20260907
MIN_SOURCE = 8
MAX_AUXILIARY = 6
REENTRY = (
    "third_historical_study", "paired_broad_narrow_gate",
    "raw_fcs_with_required_markers", "organizer_metadata_correction",
    "validated_gate_ontology", "new_compatible_labeled_data",
)


def text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    s = re.sub(r"\s+", " ", str(value).strip()).casefold()
    return "" if s in {"na", "n/a", "nan", "none", "null", "unknown", "not reported", "not available"} else s


def study_key(value: Any) -> str:
    key = re.sub(r"[\s_-]+", "", text(value)).upper()
    return key if key in STUDIES else "other_study"


def definition_text(value: Any) -> str:
    s = text(value).upper().replace("−", "-").replace("–", "-")
    s = re.sub(r"MS4A1\s*\(\s*CD20\s*\)|CD20\s*/\s*MS4A1", "CD20", s)
    s = s.replace("MS4A1", "CD20")
    # Only explicit notation aliases; never equate low with negative.
    s = re.sub(r"(CD\d+|IGD)\s*(?:HIGH|HI)(?=$|[^A-Z]|CD\d|HCD|IGD)", r"\1++", s)
    s = re.sub(r"(CD\d+|IGD)\s*(?:LOW|LO)(?=$|[^A-Z]|CD\d|HCD|IGD)", r"\1LO", s)
    return re.sub(r"\s+", "", s)


def marker_state(value: Any, marker: str) -> str:
    s = definition_text(value)
    found = re.findall(re.escape(marker) + r"(?!\d)(\+\+|\+|\-|LO)?", s)
    if not found:
        return "not_reported"
    states = {"++": "high", "+": "positive", "-": "negative", "LO": "low", "": "present_unspecified"}
    unique = {states[item] for item in found}
    return next(iter(unique)) if len(unique) == 1 else "conflicting"


def gate_text(value: Any) -> str:
    s = definition_text(value)
    tokens = re.findall(r"(?:CD\d+|IGD)(?:\+\+|\+|-|LO)", s)
    remainder = re.sub(r"(?:CD\d+|IGD)(?:\+\+|\+|-|LO)", "", s)
    remainder = re.sub(r"[,;:/()\[\]{}&]+", "", remainder)
    # Unparsed content is retained: different non-marker constraints cannot collapse.
    return "|".join(sorted(tokens)) + "::" + remainder if s else ""


def processing_state(value: Any) -> str:
    s = text(value)
    fresh = bool(re.search(r"\bfresh(?:ly)?\b", s))
    frozen = bool(re.search(r"cryopreserv|\bfrozen\b|\bthaw", s))
    if fresh and frozen:
        return "mixed_or_ambiguous"
    return "fresh_reported" if fresh else ("frozen_or_thawed_reported" if frozen else "not_reported")


def percentile(values: Any) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("E04 rank requires a nonempty finite vector")
    return np.array([0.5]) if len(a) == 1 else (rankdata(a, method="average") - 1) / (len(a) - 1)


def correlation(target: Any, prediction: Any) -> dict[str, Any]:
    a, b = np.asarray(target, float), np.asarray(prediction, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("E04 paired metric shape mismatch")
    status = "ok"
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        status = "nonfinite"
    elif len(a) < 3:
        status = "insufficient_n"
    elif len(np.unique(a)) < 2:
        status = "constant_target"
    elif len(np.unique(b)) < 2:
        status = "constant_prediction"
    value = float(spearmanr(a, b).statistic) if status == "ok" else None
    if value is not None and not np.isfinite(value):
        value, status = None, "undefined"
    return {"value": value, "status": status, "n": int(len(a))}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _parent_category(value: str) -> str:
    normalized = re.sub(r"[\s_-]+", "", value)
    allowed = {"pbmc": "PBMC", "pbmcs": "PBMC", "lymphocytes": "lymphocytes", "bcells": "B_cells", "livecells": "live_cells"}
    return allowed.get(normalized, "other_reported" if value else "not_reported")


@dataclass
class MeasurementAudit:
    aggregate: dict[str, Any]
    baseline: pd.DataFrame
    day7: pd.DataFrame
    eligibility: pd.DataFrame
    auxiliary_names: tuple[str, ...]
    challenge_signature: str | None


def audit_measurements(public_flow: pd.DataFrame, challenge_flow: pd.DataFrame) -> MeasurementAudit:
    """Inspect metadata before modeling; never export raw definitions/comments/IDs."""
    from .aliases import canonicalize_flow_population, canonicalize_timepoint
    from .task13_harmonization import sdy272_asc_proxy_mask

    required = {"participant_id", "study_accession", "timepoint", "name", "value", "population_definition", "parent_population", "unit", "material"}
    for frame in (public_flow, challenge_flow):
        if not required.issubset(frame.columns):
            raise ValueError("E04 flow metadata schema incomplete")
    challenge_times = challenge_flow["timepoint"].map(canonicalize_timepoint)
    if not (challenge_times.eq("Pre-vacc") | pd.to_numeric(challenge_times, errors="coerce").le(0)).all():
        raise ValueError("E04 Challenge contains non-baseline timepoints")
    f = pd.concat([public_flow, challenge_flow], ignore_index=True).copy()
    f["_study"] = f["study_accession"].map(study_key)
    f = f.loc[f["_study"].isin(STUDIES)].copy()
    f["_name"] = f["name"].map(canonicalize_flow_population)
    f["_proxy"] = sdy272_asc_proxy_mask(f)
    f.loc[f["_proxy"], "_name"] = ASC
    f["_time"] = f["timepoint"].map(canonicalize_timepoint)
    f["_numeric_time"] = pd.to_numeric(f["_time"], errors="coerce")
    f["_value"] = pd.to_numeric(f["value"], errors="coerce")
    f["_gate"] = f["population_definition"].map(gate_text)
    for field in ("parent_population", "unit", "material"):
        f["_" + field] = f[field].map(text)
    comments = f["comments"] if "comments" in f else pd.Series("", index=f.index)
    f["_processing"] = comments.map(processing_state)
    for marker in MARKERS:
        f["_" + marker] = f["population_definition"].map(lambda x, m=marker: marker_state(x, m))
    sigcols = ["_gate", "_parent_population", "_unit", "_material", "_processing"]
    f["_signature"] = [_sha(list(x)) for x in f[sigcols].itertuples(index=False, name=None)]
    f["_complete"] = f[["_gate", "_parent_population", "_unit", "_material"]].ne("").all(axis=1)
    f["_complete"] &= ~f[["_" + m for m in MARKERS]].eq("conflicting").any(axis=1)
    f["_complete"] &= f["_processing"].ne("mixed_or_ambiguous")

    records: list[dict[str, Any]] = []
    asc = f.loc[f["_name"].eq(ASC)]
    for study in STUDIES:
        for point in ("Pre-vacc", "0", "negative_baseline", "7"):
            mask = asc["_numeric_time"].lt(0) if point == "negative_baseline" else asc["_time"].eq(point)
            g = asc.loc[asc["_study"].eq(study) & mask]
            contracts = []
            for signature, h in g.groupby("_signature", sort=True):
                row = h.iloc[0]
                contracts.append({
                    "signature_sha256": signature, "rows": int(len(h)),
                    "subjects": int(h["participant_id"].nunique()),
                    "markers": {m: row["_" + m] for m in MARKERS},
                    "parent_category": _parent_category(row["_parent_population"]),
                    "parent_sha256": _sha(row["_parent_population"]) if row["_parent_population"] else None,
                    "denominator_status": "reported_parent_only" if row["_parent_population"] else "not_reported",
                    "unit_category": row["_unit"] if row["_unit"] in {"percentage", "% of parent", "cells", "cells/ul", "percentile"} else "other_or_missing",
                    "unit_sha256": _sha(row["_unit"]) if row["_unit"] else None,
                    "material_category": "PBMC" if row["_material"].startswith("pbmc") else "other_or_missing",
                    "material_sha256": _sha(row["_material"]) if row["_material"] else None,
                    "processing": row["_processing"], "metadata_complete": bool(row["_complete"]),
                    "population_name_class": "SDY272_locked_B_cell_proxy" if bool(row["_proxy"]) else "named_ASC",
                })
            records.append({"study": study, "timepoint": point, "rows": int(len(g)), "subjects": int(g["participant_id"].nunique()), "signature_count": len(contracts), "contracts": contracts})

    def collapse(group: pd.DataFrame, phase: str) -> list[dict[str, Any]]:
        out = []
        keys = ["participant_id", "_study", "_name"]
        for (pid, study, name), g in group.groupby(keys, sort=True):
            # Do not average heterogeneous gates, units, denominators or processing.
            signatures = g["_signature"].unique()
            finite = np.isfinite(g["_value"].to_numpy(float))
            good = len(signatures) == 1 and bool(g["_complete"].all()) and bool(finite.all()) and bool(g["_value"].ge(0).all())
            out.append({"participant_id": str(pid), "study": str(study), "name": name,
                        "signature": str(signatures[0]) if len(signatures) == 1 else "conflict",
                        "valid": good, "value": float(g["_value"].mean()) if good else np.nan,
                        "phase": phase, "metadata_conflict": len(signatures) > 1,
                        "proxy": bool(g["_proxy"].all()), "source_rows": int(len(g))})
        return out

    # Literal Pre-vacc wins; only its absence permits a homogeneous <=0 fallback.
    pre = f.loc[f["_time"].eq("Pre-vacc") | f["_numeric_time"].le(0)].copy()
    chosen = []
    for _, g in pre.groupby(["participant_id", "_study", "_name"], sort=True):
        literal = g.loc[g["_time"].eq("Pre-vacc")]
        chosen.append(literal if len(literal) else g)
    pre = pd.concat(chosen, ignore_index=True) if chosen else f.iloc[:0]
    cols = ["participant_id", "study", "name", "signature", "valid", "value", "phase", "metadata_conflict", "proxy", "source_rows"]
    baseline = pd.DataFrame(collapse(pre, "baseline"), columns=cols)
    day7 = pd.DataFrame(collapse(f.loc[f["_time"].eq("7")], "D7"), columns=cols)
    cb = baseline.loc[baseline["study"].eq("2025LJI") & baseline["name"].eq(ASC) & baseline["valid"]]
    challenge_signature = str(cb["signature"].iloc[0]) if len(cb) and cb["signature"].nunique() == 1 else None
    b = baseline.loc[baseline["name"].eq(ASC)].copy()
    d = day7.loc[day7["name"].eq(ASC)]
    pairs = b.merge(d[["participant_id", "study", "signature", "valid"]], on=["participant_id", "study"], how="left", suffixes=("_baseline", "_D7"), validate="one_to_one")
    pairs["within_gate_compatible"] = pairs["valid_baseline"] & pairs["valid_D7"].fillna(False).astype(bool) & pairs["signature_baseline"].eq(pairs["signature_D7"])
    pairs["challenge_gate_compatible"] = pairs["valid_baseline"] & pairs["signature_baseline"].eq(challenge_signature)
    pairs["strict_reconstructed"] = False  # Neither metadata matching nor this model reconstructs FCS gates.
    pairs["proxy_allowed"] = pairs["study"].eq("SDY272") & pairs["proxy"] & pairs["within_gate_compatible"]

    # B-cell-family panel membership and ordering use X/metadata only, not y or scores.
    candidates = sorted(n for n in baseline["name"].unique() if n != ASC and re.search(r"b[_-]?cells?", str(n), re.I))
    auxiliary = []
    for name in candidates:
        g = baseline.loc[baseline["name"].eq(name) & baseline["valid"]]
        challenge = g.loc[g["study"].eq("2025LJI")]
        if challenge["signature"].nunique() != 1:
            continue
        sig = challenge["signature"].iloc[0]
        if len(g.loc[g["study"].ne("2025LJI") & g["signature"].eq(sig)]) >= MIN_SOURCE:
            auxiliary.append(name)
    auxiliary = auxiliary[:MAX_AUXILIARY]
    diagnostics = []
    for study in STUDIES:
        g = pairs.loc[pairs["study"].eq(study)]
        diagnostics.append({"study": study, "baseline_subjects": int(len(g)),
                            "paired_gate_compatible_subjects": int(g["within_gate_compatible"].sum()),
                            "challenge_gate_compatible_subjects": int(g["challenge_gate_compatible"].sum()),
                            "proxy_admissible_subjects": int(g["proxy_allowed"].sum())})
    aggregate = {
        "schema_version": 1, "measurement_contract_version": "e04_metadata_v1",
        "endpoint_contracts": records, "paired_coverage": diagnostics,
        "baseline_conflict_keys": int(baseline["metadata_conflict"].sum()),
        "day7_conflict_keys": int(day7["metadata_conflict"].sum()),
        "selected_auxiliary_count": len(auxiliary),
        "selected_auxiliary_name_hashes": [_sha(n) for n in auxiliary],
        "challenge_D7_observed": bool(len(day7.loc[day7["study"].eq("2025LJI")])),
        "raw_fcs_availability": "not_established_by_TSV_audit",
        "strict_gate_reconstructed": False,
        "marker_not_reported_is_negative": False,
        "processing_equivalence_established": False,
        "cross_study_gate_equivalence_claim": False,
        "reentry_conditions": list(REENTRY),
    }
    if aggregate["challenge_D7_observed"]:
        raise ValueError("E04 baseline-only Challenge boundary violated")
    return MeasurementAudit(aggregate, baseline, day7, pairs, tuple(auxiliary), challenge_signature)


def normalized_proxy_weights(studies: Any, proxy_weight: float) -> np.ndarray:
    if float(proxy_weight) not in PROXY_WEIGHTS:
        raise ValueError("E04 proxy weight is frozen; no search permitted")
    labels = np.asarray(studies)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("E04 weights require nonempty study labels")
    weights = np.where(labels == "SDY272", float(proxy_weight), 1.0)
    return weights / float(weights.mean())


def bounded_residual(
    train_x: np.ndarray, prediction_x: np.ndarray, train_y: np.ndarray,
    train_anchor: np.ndarray, prediction_anchor: np.ndarray, studies: np.ndarray,
    *, proxy_weight: float, eligible_train: np.ndarray, eligible_prediction: np.ndarray,
    shuffle_seed: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fixed alpha, train-only y ranks, bounded correction and exact row fallback.

    Missing X is median-imputed using eligible training rows; no missingness
    indicators enter the learned model. Weight normalization avoids changing
    effective Ridge regularization when one whole source study is downweighted.
    """
    tx, px = np.asarray(train_x, float), np.asarray(prediction_x, float)
    y, a, pa = np.asarray(train_y, float), np.asarray(train_anchor, float), np.asarray(prediction_anchor, float)
    studies = np.asarray(studies)
    et, ep = np.asarray(eligible_train, bool), np.asarray(eligible_prediction, bool)
    if tx.ndim != 2 or px.ndim != 2 or tx.shape[1] != px.shape[1] or len(tx) != len(y) or len(px) != len(pa):
        raise ValueError("E04 residual matrix shape mismatch")
    if len(studies) != len(y) or len(a) != len(y) or et.shape != y.shape or ep.shape != pa.shape:
        raise ValueError("E04 residual metadata shape mismatch")
    if not np.isfinite(pa).all():
        raise ValueError("E04 preserved anchor must be finite")
    good = et & np.isfinite(y) & np.isfinite(a)
    selected = np.flatnonzero(good)
    result = pa.copy()
    info = {"source_rows": int(len(selected)), "source_studies": int(len(np.unique(studies[good]))), "prediction_eligible": int(ep.sum()), "alpha": ALPHA, "shrinkage": SHRINKAGE, "correction_cap": MAX_CORRECTION, "proxy_weight": float(proxy_weight), "shuffle": shuffle_seed is not None, "correction_applied": 0, "state": "data_limited"}
    weights = normalized_proxy_weights(studies[good], proxy_weight) if len(selected) else np.array([])
    info["normalized_weight_min"] = float(weights.min()) if len(weights) else None
    info["normalized_weight_max"] = float(weights.max()) if len(weights) else None
    if len(selected) < MIN_SOURCE or not ep.any():
        return result, info
    target = np.empty(len(selected))
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    for study in sorted(set(studies[good])):
        positions = np.flatnonzero(studies[good] == study)
        values = y[selected[positions]]
        if rng is not None:
            values = rng.permutation(values)
        target[positions] = percentile(values) - a[selected[positions]]
    available = np.isfinite(tx[good]).any(axis=0)
    # Keep the primitive stable when every optional panel is missing.
    if not available.any():
        return result, info
    x = tx[good][:, available]
    medians = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    x = np.where(np.isfinite(x), x, medians)
    p = px[:, available]
    p = np.where(np.isfinite(p), p, medians)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=ALPHA).fit(scaler.transform(x), target, sample_weight=weights)
    correction = model.predict(scaler.transform(p))
    if not np.isfinite(correction).all():
        raise ValueError("E04 nonfinite learned correction")
    delta = np.clip(SHRINKAGE * correction, -MAX_CORRECTION, MAX_CORRECTION)
    result[ep] = pa[ep] + delta[ep]
    info.update(state="evaluated", correction_applied=int(ep.sum()), active_features=int(available.sum()), max_absolute_score_correction=float(np.max(np.abs(result - pa))), mean_absolute_score_correction=float(np.mean(np.abs(result - pa))))
    if not np.array_equal(result[~ep], pa[~ep]):
        raise ValueError("E04 missing/gate fallback is not exact")
    return result, info
