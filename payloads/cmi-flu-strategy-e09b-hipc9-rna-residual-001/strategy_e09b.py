"""Strategy-v2 E09b: fixed HIPC young-signature RNA residual for D28 HAI.

This experiment is deliberately small.  It uses the nine baseline genes from the
2017 HIPC multicohort influenza-vaccination study as a frozen prior and asks
whether within-study RNA ranks add donor-order information beyond the frozen
B2.1 HAI model for Task2.1/Task2.2.

No gene is selected from the CMI-Flu outcomes.  Public/Challenge RNA uses raw
Day-0 TPM only; the Challenge-only batch-corrected representation is excluded.
Participant-level predictions are transient and never returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .aliases import canonicalize_timepoint
from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns, require_finite
from .cv import purged_leave_one_study_out
from .datasets import build_hai_model_dataset
from .evaluation import aggregate_hai_task_predictions
from .metrics import percentile_rank, safe_spearman
from .strategy_e05 import B21_MODELS, _canonical_panel, _control_run, _find_model, _panel_donor_frame
from .strategy_e09a import _material_category

EXPERIMENT = "strategy_v2_e09b_hipc9_rna_residual"
SOURCE_DOI = "10.1126/sciimmunol.aal4656"
SOURCE_PMID = "28842433"
QUALIFYING_STUDIES = ("2020_UGA", "2024_UGA")
TASKS = ("Task2.1", "Task2.2")
RIDGE_ALPHA = 10.0
RESIDUAL_SHRINKAGE = 0.25
MAX_SCORE_CORRECTION = 0.10
MIN_GENE_COVERAGE = 7
MIN_FOLD_SUBJECTS = 24
PROMOTION_MEAN_DELTA = 0.02
PROMOTION_MIN_STUDY_DELTA = -0.05
SHUFFLE_SEED = 20260909
CHUNK_ROWS = 500_000

# NCBI Gene -> Ensembl cross-references checked 2026-09-09.  The paper's
# independent validation used the seven-gene subset excluding RAB24/DPP3 when
# those genes were not measured, motivating MIN_GENE_COVERAGE=7 rather than a
# data-dependent feature search.
HIPC9: tuple[tuple[str, str], ...] = (
    ("RAB24", "ENSG00000169228"),
    ("GRB2", "ENSG00000177885"),
    ("DPP3", "ENSG00000254986"),
    ("ACTB", "ENSG00000075624"),
    ("MVP", "ENSG00000013364"),
    ("DPP7", "ENSG00000176978"),
    ("ARPC4", "ENSG00000241553"),
    ("PLEKHB2", "ENSG00000115762"),
    ("ARRB1", "ENSG00000137486"),
)
HIPC7_VALIDATION = ("GRB2", "ACTB", "MVP", "DPP7", "ARPC4", "PLEKHB2", "ARRB1")
GENE_COLUMNS = tuple(f"rna_rank__{symbol}" for symbol, _ in HIPC9)
PUBLIC_RNA_FILES = {
    "2020_UGA": "publicData_rnaseq_2020_UGA.tsv",
    "2024_UGA": "publicData_rnaseq_2024_UGA.tsv",
}
CHALLENGE_RNA_FILE = "2025LJI_rnaseq.tsv"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _ensembl_base(value: object) -> str:
    raw = "" if value is None or pd.isna(value) else str(value).strip().upper()
    return raw.split(".", 1)[0]


def _study_rank(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    finite = np.isfinite(numeric)
    if finite.any():
        result[finite] = percentile_rank(numeric[finite])
    return result


def load_hipc9_day0(path: str | Path, *, expected_study: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only the frozen HIPC genes, then rank each gene within study."""
    source = Path(path)
    if not source.is_file():
        raise DataContractError(f"E09b RNA file missing:{source.name}")
    wanted = {ensembl: symbol for symbol, ensembl in HIPC9}
    parts: list[pd.DataFrame] = []
    rows_scanned = 0
    selected_rows = 0
    for chunk in pd.read_csv(
        source,
        sep="\t",
        usecols=["participant_id", "timepoint", "ensembl_gene_id", "tpm", "material", "study_accession"],
        dtype={"participant_id": "string", "timepoint": "string", "ensembl_gene_id": "string", "material": "string", "study_accession": "string"},
        chunksize=CHUNK_ROWS,
        low_memory=False,
    ):
        rows_scanned += int(len(chunk))
        study = chunk["study_accession"].astype("string").str.strip()
        time = chunk["timepoint"].map(canonicalize_timepoint)
        gene = chunk["ensembl_gene_id"].map(_ensembl_base)
        material = chunk["material"].map(_material_category)
        mask = study.eq(expected_study) & time.eq("0") & gene.isin(wanted) & material.eq("PBMC")
        if not mask.any():
            continue
        selected = chunk.loc[mask, ["participant_id", "tpm"]].copy()
        selected["gene"] = gene.loc[mask].map(wanted)
        selected["tpm"] = pd.to_numeric(selected["tpm"], errors="coerce")
        selected = selected.loc[selected["tpm"].notna() & selected["tpm"].ge(0)].copy()
        selected_rows += int(len(selected))
        parts.append(selected)
    if not parts:
        raise DataContractError(f"E09b no frozen HIPC genes found:{expected_study}")
    selected = pd.concat(parts, ignore_index=True)
    grouped = (
        selected.groupby(["participant_id", "gene"], observed=True, dropna=False)["tpm"]
        .mean()
        .reset_index()
    )
    wide = grouped.pivot(index="participant_id", columns="gene", values="tpm").reset_index()
    for symbol, _ in HIPC9:
        if symbol not in wide:
            wide[symbol] = np.nan
    out = wide[["participant_id", *[symbol for symbol, _ in HIPC9]]].copy()
    coverage = out[[symbol for symbol, _ in HIPC9]].notna().sum(axis=1)
    out["gene_coverage"] = coverage.astype(int)
    for symbol, _ in HIPC9:
        ranks = _study_rank(np.log1p(pd.to_numeric(out[symbol], errors="coerce")))
        out[f"rna_rank__{symbol}"] = np.where(np.isfinite(ranks), ranks, 0.5)
    out["hipc7_score"] = out[[f"rna_rank__{symbol}" for symbol in HIPC7_VALIDATION]].mean(axis=1)
    out["rna_eligible"] = out["gene_coverage"].ge(MIN_GENE_COVERAGE)
    out["study_group"] = expected_study
    present_symbols = [symbol for symbol, _ in HIPC9 if out[symbol].notna().any()]
    result = out[["participant_id", "study_group", *GENE_COLUMNS, "hipc7_score", "gene_coverage", "rna_eligible"]].copy()
    audit = {
        "study": expected_study,
        "rows_scanned": rows_scanned,
        "selected_rows": selected_rows,
        "participants": int(len(result)),
        "eligible_participants": int(result["rna_eligible"].sum()),
        "present_gene_count": len(present_symbols),
        "present_symbols": present_symbols,
        "minimum_gene_coverage": int(result["gene_coverage"].min()) if len(result) else 0,
        "maximum_gene_coverage": int(result["gene_coverage"].max()) if len(result) else 0,
        "day0_only": True,
        "raw_tpm_only": True,
        "material_category": "PBMC",
    }
    return result, audit


def _rank_donor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, ["study_group", "participant_id", "target", "prediction"], table_name="E09b donor frame")
    out = frame.copy().reset_index(drop=True)
    out["target_rank"] = np.nan
    out["base_rank"] = np.nan
    for _, positions in out.groupby("study_group", observed=True, dropna=False).indices.items():
        pos = np.asarray(positions, dtype=int)
        target = pd.to_numeric(out.iloc[pos]["target"], errors="coerce").to_numpy(dtype=float)
        base = pd.to_numeric(out.iloc[pos]["prediction"], errors="coerce").to_numpy(dtype=float)
        require_finite(target, name="E09b panel target")
        require_finite(base, name="E09b B2.1 panel prediction")
        out.loc[out.index[pos], "target_rank"] = percentile_rank(target)
        out.loc[out.index[pos], "base_rank"] = percentile_rank(base)
    return out


def _fit_residual(train: pd.DataFrame, prediction: pd.DataFrame, *, shuffle: bool = False) -> np.ndarray:
    require_columns(train, [*GENE_COLUMNS, "target_rank", "base_rank", "study_group"], table_name="E09b residual train")
    require_columns(prediction, GENE_COLUMNS, table_name="E09b residual prediction")
    x = train[list(GENE_COLUMNS)].to_numpy(dtype=float)
    px = prediction[list(GENE_COLUMNS)].to_numpy(dtype=float)
    require_finite(x, name="E09b RNA train ranks")
    require_finite(px, name="E09b RNA prediction ranks")
    target = train["target_rank"].to_numpy(dtype=float) - train["base_rank"].to_numpy(dtype=float)
    require_finite(target, name="E09b residual target")
    if shuffle:
        target = np.random.default_rng(SHUFFLE_SEED).permutation(target)
    if len(train) < MIN_FOLD_SUBJECTS:
        raise DataContractError(f"E09b residual source below minimum:{len(train)}")
    scaler = StandardScaler().fit(x)
    studies = train["study_group"].astype(str)
    counts = studies.value_counts()
    weights = np.asarray([1.0 / float(counts.loc[study]) for study in studies], dtype=float)
    weights *= len(weights) / float(weights.sum())
    model = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(x), target, sample_weight=weights)
    raw = model.predict(scaler.transform(px))
    require_finite(raw, name="E09b residual prediction")
    return np.clip(RESIDUAL_SHRINKAGE * raw, -MAX_SCORE_CORRECTION, MAX_SCORE_CORRECTION)


def _spearman_value(target: Sequence[float], prediction: Sequence[float]) -> float | None:
    metric = safe_spearman(target, prediction)
    return float(metric.value) if metric.status == "ok" and np.isfinite(metric.value) else None


def _study_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for study, group in frame.groupby("study_group", observed=True, sort=True):
        base = _spearman_value(group["target_rank"], group["base_rank"])
        candidate = _spearman_value(group["target_rank"], group["candidate_score"])
        hipc7 = _spearman_value(group["target_rank"], group["hipc7_score"])
        shuffled = _spearman_value(group["target_rank"], group["shuffle_score"])
        rows.append({
            "study": str(study),
            "n": int(len(group)),
            "base_spearman": base,
            "candidate_spearman": candidate,
            "hipc7_zero_shot_spearman": hipc7,
            "shuffle_spearman": shuffled,
            "delta_vs_base": None if base is None or candidate is None else float(candidate - base),
        })
    return rows


def _challenge_movement(base: np.ndarray, candidate: np.ndarray, *, rna_available: int) -> dict[str, Any]:
    require_finite(base, name="E09b Challenge base")
    require_finite(candidate, name="E09b Challenge candidate")
    if base.shape != candidate.shape:
        raise DataContractError("E09b Challenge candidate shape mismatch")
    br = percentile_rank(base)
    cr = percentile_rank(candidate)
    delta = np.abs(br - cr)
    return {
        "donors": int(len(base)),
        "rna_available_donors": int(rna_available),
        "frozen_base_fallback_donors": int(len(base) - rna_available),
        "rank_spearman_vs_base": _spearman_value(base, candidate),
        "changed_rank_count_vs_base": int(np.sum(np.abs(br - cr) > 1e-12)),
        "mean_absolute_percentile_shift": float(delta.mean()),
        "max_absolute_percentile_shift": float(delta.max()),
        "max_absolute_score_correction": float(np.max(np.abs(candidate - base))),
    }


def evaluate_task_from_donor_frames(
    *,
    task: str,
    historical_base: pd.DataFrame,
    challenge_base: pd.DataFrame,
    historical_rna: pd.DataFrame,
    challenge_rna: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate fixed HIPC9 residual given already-cross-fitted base donor predictions."""
    if task not in TASKS:
        raise DataContractError(f"E09b unsupported task:{task}")
    hist = _rank_donor_frame(historical_base)
    hist = hist.loc[hist["study_group"].astype(str).isin(QUALIFYING_STUDIES)].copy()
    merged = hist.merge(
        historical_rna,
        on=["participant_id", "study_group"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.loc[merged["rna_eligible"].astype(bool)].copy()
    if set(merged["study_group"].astype(str)) != set(QUALIFYING_STUDIES):
        raise DataContractError("E09b qualifying study RNA join incomplete")

    folds = []
    for held in QUALIFYING_STUDIES:
        train = merged.loc[merged["study_group"].astype(str).ne(held)].copy()
        valid = merged.loc[merged["study_group"].astype(str).eq(held)].copy()
        if len(train) < MIN_FOLD_SUBJECTS or len(valid) < MIN_FOLD_SUBJECTS:
            raise DataContractError(f"E09b fold support below minimum:{held}:{len(train)}/{len(valid)}")
        correction = _fit_residual(train, valid, shuffle=False)
        shuffle = _fit_residual(train, valid, shuffle=True)
        valid["candidate_score"] = valid["base_rank"].to_numpy(dtype=float) + correction
        valid["shuffle_score"] = valid["base_rank"].to_numpy(dtype=float) + shuffle
        folds.append(valid)
    oof = pd.concat(folds, ignore_index=True)
    studies = _study_metrics(oof)
    deltas = [row["delta_vs_base"] for row in studies if row["delta_vs_base"] is not None]
    base_values = [row["base_spearman"] for row in studies if row["base_spearman"] is not None]
    candidate_values = [row["candidate_spearman"] for row in studies if row["candidate_spearman"] is not None]
    shuffle_values = [row["shuffle_spearman"] for row in studies if row["shuffle_spearman"] is not None]
    if len(deltas) != 2 or len(base_values) != 2 or len(candidate_values) != 2:
        raise DataContractError("E09b study metrics undefined")
    study_equal_base = float(np.mean(base_values))
    study_equal_candidate = float(np.mean(candidate_values))
    study_equal_shuffle = float(np.mean(shuffle_values)) if len(shuffle_values) == 2 else None
    mean_delta = float(np.mean(deltas))
    min_delta = float(np.min(deltas))

    cb = challenge_base[["participant_id", "prediction"]].copy()
    cb["base_rank"] = percentile_rank(pd.to_numeric(cb["prediction"], errors="coerce").to_numpy(dtype=float))
    crna = challenge_rna.loc[challenge_rna["rna_eligible"].astype(bool)].copy()
    joined = cb.merge(crna, on="participant_id", how="left", validate="one_to_one")
    eligible = joined["rna_eligible"].fillna(False).astype(bool).to_numpy()
    final_train = merged.copy()
    correction = np.zeros(len(joined), dtype=float)
    if eligible.any():
        correction[eligible] = _fit_residual(final_train, joined.loc[eligible], shuffle=False)
    candidate = joined["base_rank"].to_numpy(dtype=float) + correction
    movement = _challenge_movement(joined["base_rank"].to_numpy(dtype=float), candidate, rna_available=int(eligible.sum()))
    local_gate = bool(
        mean_delta >= PROMOTION_MEAN_DELTA
        and min_delta >= PROMOTION_MIN_STUDY_DELTA
        and (study_equal_shuffle is None or study_equal_candidate >= study_equal_shuffle)
    )
    competition_candidate = bool(local_gate and movement["changed_rank_count_vs_base"] > 0)
    return _json_safe({
        "task": task,
        "historical_joined_subjects": int(len(oof)),
        "study_metrics": studies,
        "study_equal_base_spearman": study_equal_base,
        "study_equal_candidate_spearman": study_equal_candidate,
        "study_equal_shuffle_spearman": study_equal_shuffle,
        "study_equal_delta_vs_base": mean_delta,
        "minimum_study_delta_vs_base": min_delta,
        "challenge_movement": movement,
        "decision": {
            "local_promotion_gate_passed": local_gate,
            "competition_candidate": competition_candidate,
            "public_probe_authorized": False,
            "incumbent_changed": False,
        },
    })


def _selected_study_splits(dataset: Any) -> list[Any]:
    splits = purged_leave_one_study_out(
        dataset.train["study_group"].astype(str),
        dataset.train["subject_group"].astype(str),
    )
    selected = [split for split in splits if str(split.held_out_group) in QUALIFYING_STUDIES]
    if {str(split.held_out_group) for split in selected} != set(QUALIFYING_STUDIES):
        raise DataContractError("E09b B2.1 qualifying study splits unavailable")
    return selected


def run_strategy_e09b(config: BaselineConfig, inputs: Any, *, data_dir: str | Path) -> dict[str, Any]:
    """Run Task2.1/2.2 E09b and return aggregate-only diagnostics."""
    root = Path(data_dir).expanduser().resolve()
    public_rna = {}
    rna_audit = {}
    for study, filename in PUBLIC_RNA_FILES.items():
        frame, audit = load_hipc9_day0(root / filename, expected_study=study)
        public_rna[study] = frame
        rna_audit[study] = audit
    challenge_rna, challenge_audit = load_hipc9_day0(root / CHALLENGE_RNA_FILE, expected_study="2025LJI")
    historical_rna = pd.concat(public_rna.values(), ignore_index=True)

    base_dataset = build_hai_model_dataset(
        inputs.tables["public_serology"],
        inputs.tables["challenge_serology"],
        inputs.tables["participants"],
        inputs.tables["investigations"],
        day=28,
        target_representation="residual",
        challenge_panel_strains=inputs.challenge_strains,
    )
    splits = _selected_study_splits(base_dataset)
    tasks = {}
    for task, panel in (("Task2.1", inputs.vaccine_strains), ("Task2.2", inputs.challenge_strains)):
        spec = _find_model(config, B21_MODELS[task])
        run = _control_run(
            base_dataset,
            spec=spec,
            splits=splits,
            panel_strains=_canonical_panel(panel),
            name="b21_reference",
            fit_challenge=True,
        )
        historical_base = _panel_donor_frame(run.oof, panel_strains=panel)
        challenge_base = aggregate_hai_task_predictions(
            run.challenge_strain_predictions,
            panel_strains=panel,
            task=task,
        )
        tasks[task] = evaluate_task_from_donor_frames(
            task=task,
            historical_base=historical_base,
            challenge_base=challenge_base,
            historical_rna=historical_rna,
            challenge_rna=challenge_rna,
        )

    return _json_safe({
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "tasks": tasks,
        "feature_contract": {
            "source_doi": SOURCE_DOI,
            "source_pmid": SOURCE_PMID,
            "gene_symbols": [symbol for symbol, _ in HIPC9],
            "ensembl_ids": [ensembl for _, ensembl in HIPC9],
            "validation_subset": list(HIPC7_VALIDATION),
            "day0_only": True,
            "raw_tpm_only": True,
            "within_study_gene_rank": True,
            "minimum_gene_coverage": MIN_GENE_COVERAGE,
            "ridge_alpha": RIDGE_ALPHA,
            "residual_shrinkage": RESIDUAL_SHRINKAGE,
            "max_score_correction": MAX_SCORE_CORRECTION,
            "minimum_fold_subjects": MIN_FOLD_SUBJECTS,
            "outcome_based_gene_selection": False,
            "batch_corrected_challenge_expression_used": False,
        },
        "rna_audit": {"public": rna_audit, "challenge": challenge_audit},
        "evaluation_contract": {
            "qualifying_studies": list(QUALIFYING_STUDIES),
            "historical_panels_are_incomplete_proxies": True,
            "official_complete_2025_panel_cv_claimed": False,
            "promotion_mean_delta": PROMOTION_MEAN_DELTA,
            "promotion_min_study_delta": PROMOTION_MIN_STUDY_DELTA,
            "shuffle_seed": SHUFFLE_SEED,
            "challenge_missing_rna_policy": "exact_frozen_b21_fallback",
        },
        "competition_submission_attempted": False,
        "public_leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
        "incumbent_changed": False,
    })


__all__ = [
    "EXPERIMENT",
    "HIPC9",
    "HIPC7_VALIDATION",
    "load_hipc9_day0",
    "evaluate_task_from_donor_frames",
    "run_strategy_e09b",
]
