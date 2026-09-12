"""Strategy v3 V3-02: fixed-reference validation harness and private prediction bank.

This stage introduces no new scientific candidate conditions. It regenerates only
predeclared frozen references/candidates, evaluates them under explicit paired
contracts, and persists row-level OOF/challenge prediction banks privately for
later candidate-role evaluation. Public leaderboard scores are not consumed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .aliases import canonicalize_strain
from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns, require_finite
from .cv import NamedSplit, purged_leave_one_study_out
from .datasets import HAIModelDataset, TaskDataset
from .evaluation import default_splits_for_task, evaluate_hai_spec
from .metrics import safe_spearman
from .models import ModelSpec, evaluate_model_spec, fit_final_model
from .runner import InputBundle, build_b02_datasets
from .strategy_e01 import (
    EXPECTED_B21_MODELS,
    TASK12_AR_LAMBDA,
    TASK12_AR_MODEL,
    TASK_ANCHORS,
    _anchor_frame,
    _compact_base_frame,
    _find_spec,
    _identity_spec,
    _metric_summary,
    _oriented_anchor,
    _rank_within,
)
from .targets import geometric_mean

EXPERIMENT = "strategy_v3_v02_paired_validation_and_private_prediction_store"
SCHEMA_VERSION = 1
RANDOM_SEED = 20260911
PSEUDO_REPETITIONS = 200
PSEUDO_COHORT_N = 40
PSEUDO_PUBLIC_N = 12
PSEUDO_PRIVATE_N = 28
MAX_FITS = 256

EXPECTED_E01_STRICT = {"Task1.1": 0.099707, "Task1.2": 0.527103, "Task1.3": 0.106304, "Task2.1": 0.623449, "Task2.2": 0.576506, "Task2.3": 0.695731}
EXPECTED_E01_ANCHOR_STRICT = {"Task1.1": 0.120854, "Task1.2": 0.425988, "Task1.3": 0.381589, "Task2.1": 0.649251, "Task2.2": 0.572427, "Task2.3": 0.677594}
TASK_UNITS = {"Task1.1": "fold_change", "Task1.2": "official_absolute_flow_target_unresolved_strict_unit", "Task1.3": "official_absolute_flow_target", "Task1.4": "official_absolute_aim_target", "Task2.1": "HAI_titer_geometric_mean", "Task2.2": "HAI_titer_geometric_mean", "Task2.3": "HAI_titer_geometric_mean"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)): return value
    if isinstance(value, float): return value if np.isfinite(value) else None
    if isinstance(value, np.generic): return _json_safe(value.item())
    if isinstance(value, Mapping): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    return value


def _sha256_bytes(raw: bytes) -> str: return hashlib.sha256(raw).hexdigest()


@dataclass
class _FitCounter:
    count: int = 0
    def add(self, n: int, *, label: str) -> None:
        if n < 0: raise DataContractError(f"negative fit count:{label}")
        self.count += int(n)
        if self.count > MAX_FITS: raise DataContractError(f"V3-02 fit budget exceeded:{self.count}>{MAX_FITS}:{label}")


def _study_subject_key(frame: pd.DataFrame) -> pd.Series:
    require_columns(frame, ["study_group", "subject_group"]); return frame["study_group"].astype(str) + "\x1f" + frame["subject_group"].astype(str)


def _metric_from_frame(frame: pd.DataFrame, *, prediction_scale: str) -> dict[str, Any]:
    require_columns(frame, ["participant_id", "study_group", "subject_group", "target", "prediction"])
    if _study_subject_key(frame).duplicated().any(): raise DataContractError("V3-02 metric frame is not one row per study/subject")
    work = frame.copy(); work["participant_id"] = [f"__v302_row_{i}" for i in range(len(work))]
    result = dict(_metric_summary(work, prediction_scale=prediction_scale)); result["participant_years"] = int(len(frame)); result["unique_subjects"] = int(frame["subject_group"].astype(str).nunique()); return _json_safe(result)


def _compact_fixed_oof(dataset: TaskDataset, *, spec: ModelSpec, counter: _FitCounter) -> tuple[pd.DataFrame, Sequence[NamedSplit]]:
    splits = default_splits_for_task(dataset, random_state=42)
    evaluation = evaluate_model_spec(dataset.train, target_column=dataset.target_column, splits=splits, spec=spec, excluded_columns=dataset.excluded_columns, aggregate_repeats=int(dataset.train["study_group"].nunique()) < 2)
    counter.add(len(splits), label=f"{dataset.task}:{spec.name}:oof"); return _compact_base_frame(dataset, evaluation), splits


def _task12_anchor_residual_oof(dataset: TaskDataset, *, config: BaselineConfig, splits: Sequence[NamedSplit], counter: _FitCounter) -> pd.DataFrame:
    if dataset.task != "Task1.2": raise DataContractError("Task1.2 AR requested for another task")
    anchor_column = TASK_ANCHORS["Task1.2"]; spec = _identity_spec(_find_spec(config, "task_12", TASK12_AR_MODEL)); excluded = tuple(dict.fromkeys([*dataset.excluded_columns, dataset.target_column])); parts = []
    for split in splits:
        train = dataset.train.iloc[split.train_indices].copy(); validation = dataset.train.iloc[split.validation_indices].copy()
        train_anchor = _oriented_anchor("Task1.2", train, train, anchor_column=anchor_column, target_column=dataset.target_column); validation_anchor = _oriented_anchor("Task1.2", train, validation, anchor_column=anchor_column, target_column=dataset.target_column)
        train["__v302_ar_residual"] = _rank_within(train, dataset.target_column) - train_anchor
        _, correction = fit_final_model(train, validation, target_column="__v302_ar_residual", spec=spec, excluded_columns=excluded); counter.add(1, label="Task1.2:anchor_residual:oof")
        parts.append(pd.DataFrame({"row_index": split.validation_indices, "participant_id": validation["participant_id"].astype(str).to_numpy(), "subject_group": validation["subject_group"].astype(str).to_numpy(), "study_group": validation["study_group"].astype(str).to_numpy(), "target": pd.to_numeric(validation[dataset.target_column], errors="coerce").to_numpy(dtype=float), "prediction": validation_anchor + TASK12_AR_LAMBDA * np.asarray(correction, dtype=float)}))
    result = pd.concat(parts, ignore_index=True)
    if result["row_index"].duplicated().any() or set(result["row_index"]) != set(range(len(dataset.train))): raise DataContractError("Task1.2 AR OOF coverage mismatch")
    return result.sort_values("row_index").reset_index(drop=True)


def _compact_challenge_prediction(dataset: TaskDataset, *, spec: ModelSpec, counter: _FitCounter) -> np.ndarray:
    _, pred = fit_final_model(dataset.train, dataset.challenge, target_column=dataset.target_column, spec=spec, excluded_columns=dataset.excluded_columns); counter.add(1, label=f"{dataset.task}:{spec.name}:challenge"); return np.asarray(pred, dtype=float)


def _task12_anchor_residual_challenge(dataset: TaskDataset, *, config: BaselineConfig, counter: _FitCounter) -> np.ndarray:
    anchor_column = TASK_ANCHORS["Task1.2"]; spec = _identity_spec(_find_spec(config, "task_12", TASK12_AR_MODEL)); train = dataset.train.copy(); challenge = dataset.challenge.copy()
    train_anchor = _oriented_anchor("Task1.2", train, train, anchor_column=anchor_column, target_column=dataset.target_column); challenge_anchor = _oriented_anchor("Task1.2", train, challenge, anchor_column=anchor_column, target_column=dataset.target_column)
    train["__v302_ar_residual"] = _rank_within(train, dataset.target_column) - train_anchor; excluded = tuple(dict.fromkeys([*dataset.excluded_columns, dataset.target_column]))
    _, correction = fit_final_model(train, challenge, target_column="__v302_ar_residual", spec=spec, excluded_columns=excluded); counter.add(1, label="Task1.2:anchor_residual:challenge"); return challenge_anchor + TASK12_AR_LAMBDA * np.asarray(correction, dtype=float)


def _compact_anchor_challenge(dataset: TaskDataset) -> np.ndarray:
    anchor_column = TASK_ANCHORS[dataset.task]; return np.asarray(_oriented_anchor(dataset.task, dataset.train, dataset.challenge, anchor_column=anchor_column, target_column=dataset.target_column), dtype=float)


def _hai_oof_frame(dataset: HAIModelDataset, *, spec: ModelSpec, panel: Sequence[str], counter: _FitCounter) -> tuple[pd.DataFrame, pd.DataFrame, Sequence[NamedSplit]]:
    splits = purged_leave_one_study_out(dataset.train["study_group"].astype(str), dataset.train["subject_group"].astype(str)); evaluation = evaluate_hai_spec(dataset, spec=spec, splits=splits, panel_strains=panel); counter.add(len(splits), label=f"HAI_D{dataset.day}:{spec.name}:oof")
    enriched = evaluation.enriched_oof.copy(); panel_set = {canonicalize_strain(x) for x in panel}; enriched["virus_strain"] = enriched["virus_strain"].map(canonicalize_strain); work = enriched.loc[enriched["virus_strain"].isin(panel_set)].copy()
    grouped = work.groupby(["study_group", "participant_id", "subject_group"], observed=True, dropna=False).agg(target=("post_hai", geometric_mean), prediction=("post_prediction", geometric_mean)).reset_index(); return grouped, enriched, splits


def _hai_anchor_from_enriched(enriched: pd.DataFrame, *, panel: Sequence[str]) -> pd.DataFrame:
    panel_set = {canonicalize_strain(x) for x in panel}; work = enriched.copy(); work["virus_strain"] = work["virus_strain"].map(canonicalize_strain); work = work.loc[work["virus_strain"].isin(panel_set)].copy(); work["pre_hai"] = np.exp2(pd.to_numeric(work["log2_pre_hai"], errors="coerce").to_numpy(dtype=float))
    grouped = work.groupby(["study_group", "participant_id", "subject_group"], observed=True, dropna=False).agg(target=("post_hai", geometric_mean), pre_hai=("pre_hai", geometric_mean)).reset_index(); grouped["prediction"] = np.nan
    for _, positions in grouped.groupby("study_group", observed=True, dropna=False).indices.items():
        pos = np.asarray(positions, dtype=int); grouped.loc[grouped.index[pos], "prediction"] = pd.Series(grouped.iloc[pos]["pre_hai"].to_numpy(dtype=float)).rank(method="average", pct=True).to_numpy(dtype=float)
    return grouped[["participant_id", "subject_group", "study_group", "target", "prediction"]]


def _hai_challenge_raw(dataset: HAIModelDataset, *, spec: ModelSpec, panel: Sequence[str], counter: _FitCounter) -> pd.DataFrame:
    _, target_pred = fit_final_model(dataset.train, dataset.challenge, target_column=dataset.target_column, spec=spec, excluded_columns=dataset.excluded_columns); counter.add(1, label=f"HAI_D{dataset.day}:{spec.name}:challenge")
    strain_pred = dataset.challenge[["participant_id", "subject_group", "study_group", "virus_strain"]].copy(); strain_pred["virus_strain"] = strain_pred["virus_strain"].map(canonicalize_strain); strain_pred["prediction"] = dataset.target_prediction_to_post_hai(target_pred, frame=dataset.challenge); panel_set = {canonicalize_strain(x) for x in panel}; strain_pred = strain_pred.loc[strain_pred["virus_strain"].isin(panel_set)].copy()
    return strain_pred.groupby(["participant_id", "subject_group", "study_group"], observed=True, dropna=False).agg(prediction=("prediction", geometric_mean)).reset_index()


def _hai_challenge_anchor(dataset: HAIModelDataset, *, panel: Sequence[str]) -> pd.DataFrame:
    work = dataset.challenge[["participant_id", "subject_group", "study_group", "virus_strain", "log2_pre_hai"]].copy(); work["virus_strain"] = work["virus_strain"].map(canonicalize_strain); panel_set = {canonicalize_strain(x) for x in panel}; work = work.loc[work["virus_strain"].isin(panel_set)].copy(); work["pre_hai"] = np.exp2(pd.to_numeric(work["log2_pre_hai"], errors="coerce").to_numpy(dtype=float))
    grouped = work.groupby(["participant_id", "subject_group", "study_group"], observed=True, dropna=False).agg(pre_hai=("pre_hai", geometric_mean)).reset_index(); grouped["prediction"] = pd.Series(grouped["pre_hai"]).rank(method="average", pct=True).to_numpy(dtype=float); return grouped[["participant_id", "subject_group", "study_group", "prediction"]]


def _panel_signature_candidates(enriched: pd.DataFrame, *, challenge_panel: Sequence[str], max_panels: int = 3) -> list[dict[str, Any]]:
    panel = {canonicalize_strain(x) for x in challenge_panel}; work = enriched.copy(); work["virus_strain"] = work["virus_strain"].map(canonicalize_strain); work = work.loc[work["virus_strain"].isin(panel)].copy(); rows = []; signatures_by_study = {}
    for study, group in work.groupby("study_group", observed=True, sort=True):
        signatures = group.groupby(["participant_id", "subject_group"], observed=True)["virus_strain"].apply(lambda x: tuple(sorted(set(x)))); signatures_by_study[str(study)] = signatures
        if signatures.empty: continue
        candidates = [(tuple(sig), int(n)) for sig, n in signatures.value_counts().items() if len(sig)]
        if not candidates: continue
        chosen, support = sorted(candidates, key=lambda item: (-len(item[0]), -item[1], item[0]))[0]; rows.append({"study": str(study), "panel": list(chosen), "panel_size": len(chosen), "support": support, "panel_sha256": _sha256_bytes("\n".join(chosen).encode())})
    rows.sort(key=lambda r: (-r["panel_size"], -r["support"], r["study"], r["panel"])); unique = []; seen = set()
    for row in rows:
        if row["panel_sha256"] in seen: continue
        seen.add(row["panel_sha256"]); signature = tuple(row["panel"]); per_study = {study: int((sig == signature).sum()) for study, sig in signatures_by_study.items() if int((sig == signature).sum()) > 0}; unique.append({**row, "support_by_study": per_study, "support_total": int(sum(per_study.values()))})
        if len(unique) >= max_panels: break
    return unique


def _p1_complete_panel_frame(enriched: pd.DataFrame, *, panel: Sequence[str], prediction_column: str) -> pd.DataFrame:
    panel_tuple = tuple(sorted(canonicalize_strain(x) for x in panel)); panel_set = set(panel_tuple); work = enriched.copy(); work["virus_strain"] = work["virus_strain"].map(canonicalize_strain); work = work.loc[work["virus_strain"].isin(panel_set)].copy(); keys = ["study_group", "participant_id", "subject_group"]
    signatures = work.groupby(keys, observed=True)["virus_strain"].apply(lambda x: tuple(sorted(set(x)))); complete_index = signatures.loc[signatures == panel_tuple].index
    if not len(complete_index): return pd.DataFrame(columns=["participant_id", "subject_group", "study_group", "target", "prediction"])
    selected = work.set_index(keys).loc[complete_index].reset_index(); return selected.groupby(keys, observed=True, dropna=False).agg(target=("post_hai", geometric_mean), prediction=(prediction_column, geometric_mean)).reset_index()


def _p1_panel_metrics(enriched: pd.DataFrame, *, challenge_panel: Sequence[str]) -> list[dict[str, Any]]:
    work = enriched.copy(); work["pre_hai"] = np.exp2(pd.to_numeric(work["log2_pre_hai"], errors="coerce").to_numpy(dtype=float)); out = []
    for candidate in _panel_signature_candidates(work, challenge_panel=challenge_panel):
        model = _p1_complete_panel_frame(work, panel=candidate["panel"], prediction_column="post_prediction"); anchor_raw = _p1_complete_panel_frame(work, panel=candidate["panel"], prediction_column="pre_hai"); keys = ["study_group", "participant_id", "subject_group", "target"]
        aligned = model[keys + ["prediction"]].merge(anchor_raw[keys + ["prediction"]].rename(columns={"prediction": "anchor_raw"}), on=keys, how="inner", validate="one_to_one")
        if len(aligned) != len(model) or len(model) != len(anchor_raw): raise DataContractError("P1 model/anchor donor-panel mismatch")
        anchor_frame = aligned[keys].copy(); anchor_frame["prediction"] = aligned["anchor_raw"].to_numpy(dtype=float); studies = {str(s): int(n) for s, n in model.groupby("study_group", observed=True)["subject_group"].nunique().sort_index().items()}
        out.append({"metric_id": f"P1_complete_panel_{candidate['panel_sha256'][:12]}", "panel_sha256": candidate["panel_sha256"], "panel_size": candidate["panel_size"], "panel": candidate["panel"], "support_by_study": studies, "support_total": int(sum(studies.values())), "model": _metric_from_frame(model, prediction_scale="raw_target"), "anchor": _metric_from_frame(anchor_frame, prediction_scale="raw_target")})
    return _json_safe(out)


def _spearman_value(target: Sequence[float], pred: Sequence[float]) -> float | None:
    metric = safe_spearman(target, pred); return float(metric.value) if metric.status == "ok" else None


def _pseudo40_replay(candidate: pd.DataFrame, reference: pd.DataFrame, *, task: str) -> dict[str, Any]:
    keys = ["participant_id", "subject_group", "study_group", "target"]; aligned = candidate[keys + ["prediction"]].rename(columns={"prediction": "candidate"}).merge(reference[keys + ["prediction"]].rename(columns={"prediction": "reference"}), on=keys, how="inner", validate="one_to_one")
    if len(aligned) != len(candidate) or len(aligned) != len(reference): raise DataContractError("pseudo40 candidate/reference alignment mismatch")
    rng = np.random.default_rng(RANDOM_SEED); studies = []
    for study, group in aligned.groupby("study_group", observed=True, sort=True):
        if group["subject_group"].astype(str).duplicated().any(): raise DataContractError(f"pseudo40 requires one row per subject:{task}:{study}")
        n = len(group)
        if n < 40:
            status = "sensitivity_28_only" if n >= 28 else ("small_subset_only" if n >= 12 else "descriptive_only"); studies.append({"study": str(study), "n_subjects": int(n), "status": status, "repetitions": 0}); continue
        public_delta = []; private_delta = []; same_sign = 0; valid = 0; indices = np.arange(n)
        for _ in range(PSEUDO_REPETITIONS):
            cohort = rng.choice(indices, size=PSEUDO_COHORT_N, replace=False); public_local = rng.choice(np.arange(PSEUDO_COHORT_N), size=PSEUDO_PUBLIC_N, replace=False); mask = np.ones(PSEUDO_COHORT_N, dtype=bool); mask[public_local] = False; private_local = np.flatnonzero(mask)
            if len(private_local) != PSEUDO_PRIVATE_N or set(public_local).intersection(set(private_local)): raise DataContractError("pseudo40 12/28 partition is not disjoint")
            c = group.iloc[cohort].reset_index(drop=True); pub, prv = c.iloc[public_local], c.iloc[private_local]; cp, rp = _spearman_value(pub["target"], pub["candidate"]), _spearman_value(pub["target"], pub["reference"]); cv, rv = _spearman_value(prv["target"], prv["candidate"]), _spearman_value(prv["target"], prv["reference"])
            if None in (cp, rp, cv, rv): continue
            dp, dv = float(cp-rp), float(cv-rv); public_delta.append(dp); private_delta.append(dv); same_sign += int((dp >= 0 and dv >= 0) or (dp < 0 and dv < 0)); valid += 1
        pub_arr, prv_arr = np.asarray(public_delta), np.asarray(private_delta); studies.append({"study": str(study), "n_subjects": int(n), "status": "ok" if valid else "all_invalid", "repetitions": PSEUDO_REPETITIONS, "valid_repetitions": valid, "cohort_n": 40, "public_n": 12, "private_n": 28, "public_delta_mean": float(pub_arr.mean()) if valid else None, "public_delta_q10": float(np.quantile(pub_arr,.10)) if valid else None, "public_delta_q90": float(np.quantile(pub_arr,.90)) if valid else None, "private_delta_mean": float(prv_arr.mean()) if valid else None, "private_delta_q10": float(np.quantile(prv_arr,.10)) if valid else None, "private_delta_q90": float(np.quantile(prv_arr,.90)) if valid else None, "public_private_delta_spearman": _spearman_value(pub_arr,prv_arr) if valid>=2 else None, "delta_sign_agreement_fraction": float(same_sign/valid) if valid else None})
    return _json_safe({"task": task, "seed": RANDOM_SEED, "without_replacement": True, "repetitions": PSEUDO_REPETITIONS, "studies": studies})


def _oof_bank_rows(frame: pd.DataFrame, *, task: str, candidate: str, prediction_space: str, provenance: str) -> pd.DataFrame:
    require_columns(frame, ["participant_id","subject_group","study_group","target","prediction"]); out=frame[["participant_id","subject_group","study_group","target","prediction"]].copy(); out.insert(0,"task",task); out.insert(1,"candidate",candidate); out.insert(2,"prediction_space",prediction_space); out.insert(3,"target_unit",TASK_UNITS[task]); out.insert(4,"provenance",provenance); return out


def _challenge_bank_rows(ids: pd.DataFrame, prediction: Sequence[float], *, task: str, candidate: str, prediction_space: str, provenance: str) -> pd.DataFrame:
    require_columns(ids,["participant_id"]); values=np.asarray(prediction,dtype=float); require_finite(values,name=f"V3-02 challenge:{task}:{candidate}")
    if len(values)!=len(ids): raise DataContractError("challenge prediction length mismatch")
    return pd.DataFrame({"task":task,"candidate":candidate,"prediction_space":prediction_space,"target_unit":TASK_UNITS[task],"provenance":provenance,"participant_id":ids["participant_id"].astype(str).to_numpy(),"prediction":values})


def _source_b_long(source_csv: pd.DataFrame) -> pd.DataFrame:
    require_columns(source_csv,["participant_id",*TASK_UNITS.keys()],table_name="V3-02 source B"); pieces=[]
    for task in TASK_UNITS:
        values=pd.to_numeric(source_csv[task],errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all(): raise DataContractError(f"source B non-finite:{task}")
        pieces.append(pd.DataFrame({"task":task,"candidate":"source_B_frozen_current","prediction_space":"source_artifact_unknown_or_mixed","target_unit":TASK_UNITS[task],"provenance":"E12c-v2 exact immutable CSV","participant_id":source_csv["participant_id"].astype(str),"prediction":values.astype(float)}))
    return pd.concat(pieces,ignore_index=True)


def _prediction_fingerprint(frame: pd.DataFrame) -> str:
    require_columns(frame,["task","candidate","participant_id","prediction"]); work=frame[["task","candidate","participant_id","prediction"]].copy(); work["prediction"]=pd.to_numeric(work["prediction"],errors="raise"); work=work.sort_values(["task","candidate","participant_id"],kind="mergesort"); return _sha256_bytes(work.to_csv(index=False,lineterminator="\n",float_format="%.17g").encode())


def _challenge_reproducibility(challenge_bank: pd.DataFrame) -> dict[str, Any]:
    source=challenge_bank.loc[challenge_bank["candidate"].eq("source_B_frozen_current")]; rows=[]; mapping={"Task1.1":"b21_pls_2_fresh","Task1.2":"task12_anchor_residual_fresh","Task2.1":"b21_et_subtype_d3_l5_fresh","Task2.2":"b21_et_subtype_d5_l10_fresh","Task2.3":"b21_ridge_exact_a100_fresh"}
    for task,fresh_name in mapping.items():
        left=source.loc[source["task"].eq(task),["participant_id","prediction"]].rename(columns={"prediction":"source"}); right=challenge_bank.loc[challenge_bank["task"].eq(task)&challenge_bank["candidate"].eq(fresh_name),["participant_id","prediction"]].rename(columns={"prediction":"fresh"}); aligned=left.merge(right,on="participant_id",how="inner",validate="one_to_one")
        if len(aligned)!=40: raise DataContractError(f"challenge reproducibility expected 40 rows:{task}")
        diff=aligned["fresh"].to_numpy(dtype=float)-aligned["source"].to_numpy(dtype=float); weak=safe_spearman(aligned["source"],aligned["fresh"]); rows.append({"task":task,"source_candidate":"source_B_frozen_current","fresh_candidate":fresh_name,"max_absolute_difference":float(np.max(np.abs(diff))),"mean_absolute_difference":float(np.mean(np.abs(diff))),"spearman":weak.value,"spearman_status":weak.status,"exact_numeric_match":bool(np.array_equal(aligned["source"].to_numpy(),aligned["fresh"].to_numpy())),"source_hash":_prediction_fingerprint(source.loc[source["task"].eq(task)]),"fresh_hash":_prediction_fingerprint(challenge_bank.loc[challenge_bank["task"].eq(task)&challenge_bank["candidate"].eq(fresh_name)])})
    return {"comparisons":_json_safe(rows),"uncompared":{"Task1.3":"source B is strict ASC while E01 frozen reference is PLS1","Task1.4":"no supervised public teacher/fresh fit"}}


def _e01_reproduction_status(metrics: Mapping[str, Any]) -> dict[str, Any]:
    rows=[]
    for task,expected in EXPECTED_E01_STRICT.items():
        observed=metrics[task]["candidate"]["study_equal_weight_spearman_mean_strict"]; anchor=metrics[task]["anchor"]["study_equal_weight_spearman_mean_strict"]; exp_anchor=EXPECTED_E01_ANCHOR_STRICT[task]; rows.append({"task":task,"expected_candidate":expected,"observed_candidate":observed,"candidate_abs_delta":None if observed is None else abs(float(observed)-expected),"expected_anchor":exp_anchor,"observed_anchor":anchor,"anchor_abs_delta":None if anchor is None else abs(float(anchor)-exp_anchor)})
    max_delta=max(v for row in rows for v in (row["candidate_abs_delta"],row["anchor_abs_delta"]) if v is not None); return {"rows":rows,"max_absolute_metric_delta":max_delta,"status":"reproduced_rounding_compatible" if max_delta<=5e-6 else "mismatch_requires_diagnosis","tolerance":5e-6,"historical_values_are_report_rounded":True}


def run_v3_02(config: BaselineConfig, inputs: InputBundle, *, source_b_csv: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if config.baseline!="b021_taskwise_robust" or str(config.section("selection").get("policy",""))!="robust_v1": raise DataContractError("V3-02 requires frozen b021_taskwise_robust/robust_v1 config")
    datasets=build_b02_datasets(config,inputs); counter=_FitCounter(); oof_parts=[]; challenge_parts=[_source_b_long(source_b_csv)]; task_metrics={}; pseudo={}
    for task,model_set in (("Task1.1","task_11"),("Task1.2","task_12"),("Task1.3","task_13")):
        dataset=datasets[task]
        if not isinstance(dataset,TaskDataset): raise DataContractError(f"V3-02 expected TaskDataset:{task}")
        spec=_find_spec(config,model_set,EXPECTED_B21_MODELS[task]); b21,splits=_compact_fixed_oof(dataset,spec=spec,counter=counter); anchor=_anchor_frame(dataset,splits); candidate_frame,candidate_name,prediction_space=b21,f"b21_{EXPECTED_B21_MODELS[task]}","raw_target"
        if task=="Task1.2":
            ar=_task12_anchor_residual_oof(dataset,config=config,splits=splits,counter=counter); candidate_frame,candidate_name,prediction_space=ar,"task12_anchor_residual_et_d5_l5_sqrt_lambda0.5","rank_score"; oof_parts.append(_oof_bank_rows(b21,task=task,candidate=f"b21_{EXPECTED_B21_MODELS[task]}",prediction_space="raw_target",provenance="fresh fixed E01 reference reproduction"))
        oof_parts.extend([_oof_bank_rows(candidate_frame,task=task,candidate=candidate_name,prediction_space=prediction_space,provenance="fresh fixed E01 candidate reproduction"),_oof_bank_rows(anchor,task=task,candidate="same_readout_anchor",prediction_space="rank_score",provenance="no-fit same-readout anchor")]); task_metrics[task]={"candidate_name":candidate_name,"candidate":_metric_from_frame(candidate_frame,prediction_scale=prediction_space),"anchor":_metric_from_frame(anchor,prediction_scale="rank_score")}
        if task in {"Task1.1","Task1.2"}: pseudo[task]=_pseudo40_replay(candidate_frame,anchor,task=task)
        fresh_b21=_compact_challenge_prediction(dataset,spec=spec,counter=counter); challenge_parts.append(_challenge_bank_rows(dataset.challenge,fresh_b21,task=task,candidate=f"b21_{EXPECTED_B21_MODELS[task]}_fresh",prediction_space="raw_target",provenance="fresh fixed full-source refit for reproducibility diagnostic"))
        if task=="Task1.2": challenge_parts.append(_challenge_bank_rows(dataset.challenge,_task12_anchor_residual_challenge(dataset,config=config,counter=counter),task=task,candidate="task12_anchor_residual_fresh",prediction_space="rank_score",provenance="fresh fixed full-source refit for reproducibility diagnostic"))
        challenge_parts.append(_challenge_bank_rows(dataset.challenge,_compact_anchor_challenge(dataset),task=task,candidate="same_readout_anchor_fresh",prediction_space="rank_score",provenance="no-fit challenge anchor"))
    hai_specs={"Task2.1":("HAI_D28","et_subtype_d3_l5",inputs.vaccine_strains),"Task2.2":("HAI_D28","et_subtype_d5_l10",inputs.challenge_strains),"Task2.3":("HAI_D365","ridge_exact_a100",inputs.challenge_strains)}; p1={}
    for task,(dataset_key,spec_name,panel) in hai_specs.items():
        dataset=datasets[dataset_key]
        if not isinstance(dataset,HAIModelDataset): raise DataContractError(f"V3-02 expected HAIModelDataset:{dataset_key}")
        spec=_find_spec(config,"hai",spec_name); model_frame,enriched,_=_hai_oof_frame(dataset,spec=spec,panel=panel,counter=counter); anchor=_hai_anchor_from_enriched(enriched,panel=panel); task_metrics[task]={"candidate_name":f"b21_{spec_name}","candidate":_metric_from_frame(model_frame,prediction_scale="raw_target"),"anchor":_metric_from_frame(anchor,prediction_scale="rank_score")}; oof_parts.extend([_oof_bank_rows(model_frame,task=task,candidate=f"b21_{spec_name}",prediction_space="raw_target",provenance="fresh fixed E01 candidate reproduction"),_oof_bank_rows(anchor,task=task,candidate="pre_hai_panel_geometric_mean_rank",prediction_space="rank_score",provenance="no-fit pre-HAI anchor")]); p1[task]=_p1_panel_metrics(enriched,challenge_panel=panel)
        challenge_model=_hai_challenge_raw(dataset,spec=spec,panel=panel,counter=counter); challenge_parts.append(_challenge_bank_rows(challenge_model,challenge_model["prediction"],task=task,candidate=f"b21_{spec_name}_fresh",prediction_space="raw_target",provenance="fresh fixed full-source refit for reproducibility diagnostic")); challenge_anchor=_hai_challenge_anchor(dataset,panel=panel); challenge_parts.append(_challenge_bank_rows(challenge_anchor,challenge_anchor["prediction"],task=task,candidate="pre_hai_panel_geometric_mean_rank_fresh",prediction_space="rank_score",provenance="no-fit pre-HAI challenge anchor"))
    oof_bank=pd.concat(oof_parts,ignore_index=True); challenge_bank=pd.concat(challenge_parts,ignore_index=True)
    aggregate={"schema_version":SCHEMA_VERSION,"experiment":EXPERIMENT,"stage":"V3-02","new_candidate_conditions":0,"fit_count":counter.count,"fit_limit":MAX_FITS,"public_leaderboard_used":False,"competition_submission_attempted":False,"private_bank_contains_row_level_values":True,"aggregate_contains_row_level_values":False,"e01_reproduction":_e01_reproduction_status(task_metrics),"task_metrics":_json_safe(task_metrics),"pseudo40_12_28":_json_safe(pseudo),"hai_complete_local_panel_p1":_json_safe(p1),"bank":{"oof_rows":int(len(oof_bank)),"oof_candidates":int(oof_bank[["task","candidate"]].drop_duplicates().shape[0]),"challenge_rows":int(len(challenge_bank)),"challenge_candidates":int(challenge_bank[["task","candidate"]].drop_duplicates().shape[0]),"oof_fingerprint":_prediction_fingerprint(oof_bank),"challenge_fingerprint":_prediction_fingerprint(challenge_bank)},"challenge_reproducibility":_challenge_reproducibility(challenge_bank),"schema_caveat":{"hai_metric_subjects_legacy_field":"participant_years_not_unique_biological_subjects","use_explicit_fields":True}}
    return _json_safe(aggregate),oof_bank,challenge_bank


def write_v3_02_outputs(aggregate: Mapping[str, Any], oof_bank: pd.DataFrame, challenge_bank: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); files=[]
    for name,frame in {"v3_v02_oof_bank.csv":oof_bank,"v3_v02_challenge_bank.csv":challenge_bank}.items():
        path=out/name
        if path.exists(): raise DataContractError(f"V3-02 refuses to overwrite:{name}")
        frame.to_csv(path,index=False,float_format="%.17g",lineterminator="\n"); raw=path.read_bytes(); files.append({"filename":name,"bytes":len(raw),"sha256":_sha256_bytes(raw),"rows":int(len(frame)),"private_row_level":True})
    summary,manifest=out/"v3_v02_summary.json",out/"v3_v02_bank_manifest.json"
    if summary.exists() or manifest.exists(): raise DataContractError("V3-02 aggregate output already exists")
    summary.write_text(json.dumps(_json_safe(aggregate),indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8"); safe_manifest={"schema_version":1,"experiment":EXPERIMENT,"files":files,"row_level_contents_must_not_be_publicly_emitted":True,"competition_submission_attempted":False}; manifest.write_text(json.dumps(safe_manifest,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8"); return safe_manifest
