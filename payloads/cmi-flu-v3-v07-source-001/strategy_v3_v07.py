"""Strategy v3 V3-07: HAI panel-conditioned scale and cross-fitted retention heads.

Frozen finite family (2026-09-12):
* task22_panel_mean: E05 main-effects fixed offset + two donor/panel baseline features,
  Ridge(alpha=10), trained on raw panel-GM residuals from nested inner OOF only.
* task23_retention: log2(D365/D28) retention Ridge(alpha=100) with exactly
  baseline log2 HAI, age and cross-fitted predicted log2 D28.

Outer evaluation is biological-subject-purged study-out. A saved outer OOF bank
is never accepted as an inner teacher. No Competition submission path exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .aliases import canonicalize_strain
from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns, require_finite
from .cv import NamedSplit, purged_leave_one_study_out
from .datasets import HAIModelDataset, build_hai_model_dataset
from .evaluation import aggregate_hai_task_predictions
from .hai_transfer import add_hai_ontology_features
from .hai_transfer_v2 import normalize_sequence_reference_schema
from .metrics import safe_spearman
from .models import fit_final_model
from .strategy_e05 import B21_MODELS, _canonical_panel, _e05_design, _find_model as _find_e05_model, _fit_ridge as _fit_e05_ridge, _hierarchical_sample_weights
from .strategy_e06b import _find_model as _find_e06b_model
try:
    from .strategy_e06b_v2 import _fit_two_head_ridge
except Exception:  # pragma: no cover
    from .strategy_e06b import _fit_two_head_ridge

EXPERIMENT = "strategy_v3_v07_hai_panel_mean_and_retention"
CONDITIONS = ("task22_panel_mean", "task23_retention")
H1_ALPHA = 10.0
H2_ALPHA = 100.0
INNER_STUDY_FOLDS = 3
MAX_FITS_H1 = 256
MAX_FITS_H2 = 256
EPS = 1e-8
H1_TARGET_PROXY_SIZE = 9
H1_EXPECTED_PROXY_SUBJECTS = 238
H1_EXPECTED_PROXY_STUDIES = ("2023_UGA", "2024UGA")
H2_TARGET_PROXY_SIZE = 8
H2_EXPECTED_PROXY_SUBJECTS = 439
H2_EXPECTED_PROXY_STUDIES = 3
H2_EXPECTED_PAIR_ROWS = 11913
H2_EXPECTED_PAIR_SUBJECTS = 567

@dataclass
class _FitCounter:
    count: int = 0
    limit: int = 256
    def add(self, n: int = 1) -> None:
        self.count += int(n)
        if self.count > self.limit:
            raise DataContractError(f"V3-07 fit budget exceeded: {self.count}>{self.limit}")

def _gm(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    require_finite(x, name="V3-07 geometric-mean values")
    if x.size == 0 or (x <= 0).any():
        raise DataContractError("V3-07 geometric mean requires positive values")
    return float(np.exp(np.mean(np.log(x))))

def _nested_splits(frame: pd.DataFrame) -> list[NamedSplit]:
    require_columns(frame, ["study_group", "subject_group"], table_name="V3-07 inner source")
    study = frame["study_group"].astype(str).to_numpy()
    subject = frame["subject_group"].astype(str).to_numpy()
    studies = sorted(set(study))
    if len(studies) >= 3:
        counts = {s: len(set(subject[study == s])) for s in studies}
        bins = [[] for _ in range(INNER_STUDY_FOLDS)]
        mass = [0, 0, 0]
        for s in sorted(studies, key=lambda z: (-counts[z], z)):
            j = min(range(3), key=lambda k: (mass[k], k))
            bins[j].append(s); mass[j] += counts[s]
        masks = [np.isin(study, sorted(x)) for x in bins if x]
        kind = "balanced_study_3fold"
    elif len(studies) == 2:
        masks = [study == s for s in studies]; kind = "source_study_loso"
    else:
        unique_subjects = sorted(set(subject))
        if len(unique_subjects) < 4:
            raise DataContractError("V3-07 one-study inner CV has insufficient subjects")
        masks = [np.isin(subject, unique_subjects[i::3]) for i in range(3) if unique_subjects[i::3]]
        kind = "subject_grouped_3fold"
    result = []
    for i, val_mask in enumerate(masks):
        held_subjects = set(subject[val_mask])
        train_mask = (~val_mask) & (~np.isin(subject, list(held_subjects)))
        tr = np.flatnonzero(train_mask); va = np.flatnonzero(val_mask)
        if len(tr) < 2 or len(va) < 2:
            continue
        if set(subject[tr]).intersection(subject[va]):
            raise DataContractError("V3-07 inner split leaks biological subjects")
        result.append(NamedSplit(name=f"inner={kind}/{i}", train_indices=tr, validation_indices=va,
                                 held_out_group=",".join(sorted(set(study[va]))), purged_subjects=tuple(sorted(held_subjects))))
    if len(result) < 2:
        raise DataContractError("V3-07 produced fewer than two usable inner folds")
    return result

def _e05_predict(train: pd.DataFrame, pred: pd.DataFrame, counter: _FitCounter) -> np.ndarray:
    train = train.reset_index(drop=True); pred = pred.reset_index(drop=True)
    both = pd.concat([train.assign(__part=0), pred.assign(__part=1)], ignore_index=True)
    design = _e05_design(both, interactions=False)
    y = pd.to_numeric(train["target_log2_post"], errors="coerce").to_numpy(float)
    weights = _hierarchical_sample_weights(train)
    _, out = _fit_e05_ridge(design.iloc[:len(train)].reset_index(drop=True), y, weights,
                            design.iloc[len(train):].reset_index(drop=True), interactions=False)
    counter.add()
    return np.exp2(np.clip(out, -20.0, 30.0))

def _e05_inner_oof(frame: pd.DataFrame, counter: _FitCounter) -> pd.DataFrame:
    frame = frame.reset_index(drop=True); parts = []
    for split in _nested_splits(frame):
        tr = frame.iloc[split.train_indices].reset_index(drop=True)
        va = frame.iloc[split.validation_indices].reset_index(drop=True)
        p = _e05_predict(tr, va, counter)
        q = va[["participant_id","subject_group","study_group","virus_strain","post_hai","log2_pre_hai"]].copy()
        q["post_prediction"] = p; q["inner_split"] = split.name; parts.append(q)
    if not parts:
        raise DataContractError("V3-07 H1 inner E05 OOF is empty")
    return pd.concat(parts, ignore_index=True)

def _study_local_panels(frame: pd.DataFrame, official_panel: Sequence[str]) -> dict[str, tuple[str,...]]:
    panel = set(_canonical_panel(official_panel))
    work = frame[["study_group","virus_strain"]].copy(); work["virus_strain"] = work["virus_strain"].map(canonicalize_strain)
    work = work.loc[work["virus_strain"].isin(panel)]
    return {str(study): tuple(sorted(set(g["virus_strain"].astype(str)))) for study,g in work.groupby("study_group", observed=True) if len(g)}

def _panel_donors(frame: pd.DataFrame, *, official_panel: Sequence[str], prediction_column: str) -> pd.DataFrame:
    require_columns(frame, ["participant_id","subject_group","study_group","virus_strain","post_hai","log2_pre_hai",prediction_column])
    work = frame.copy(); work["virus_strain"] = work["virus_strain"].map(canonicalize_strain)
    panels = _study_local_panels(work, official_panel); rows = []
    for study, local in panels.items():
        s = work.loc[(work["study_group"].astype(str)==study) & work["virus_strain"].isin(local)].copy()
        per = s.groupby(["participant_id","subject_group","virus_strain"], observed=True).agg(
            actual=("post_hai", _gm), pred=(prediction_column, _gm), pre_log2=("log2_pre_hai","mean")).reset_index()
        for (pid, subj), g in per.groupby(["participant_id","subject_group"], observed=True):
            if set(g["virus_strain"].astype(str)) != set(local):
                continue
            pre = g["pre_log2"].to_numpy(float)
            rows.append({"participant_id":str(pid),"subject_group":str(subj),"study_group":study,
                         "panel_size":len(local),"panel_composition":"|".join(local),"actual":_gm(g["actual"]),
                         "base_prediction":_gm(g["pred"]),"log2_pre_panel_gm":float(np.mean(pre)),
                         "pre_panel_log2_sd":float(np.std(pre, ddof=0))})
    return pd.DataFrame(rows)

def _fit_panel_residual(cal: pd.DataFrame, pred: pd.DataFrame, counter: _FitCounter) -> tuple[np.ndarray, Mapping[str,Any]]:
    features = ["log2_pre_panel_gm","pre_panel_log2_sd"]
    require_columns(cal, [*features,"actual","base_prediction"]); require_columns(pred, [*features,"base_prediction"])
    x = cal[features].to_numpy(float); xp = pred[features].to_numpy(float)
    require_finite(x, name="V3-07 H1 calibrator features"); require_finite(xp, name="V3-07 H1 prediction features")
    target = cal["actual"].to_numpy(float) - cal["base_prediction"].to_numpy(float)
    scaler = StandardScaler().fit(x); model = Ridge(alpha=H1_ALPHA).fit(scaler.transform(x), target); counter.add()
    corr = model.predict(scaler.transform(xp)); out = np.maximum(EPS, pred["base_prediction"].to_numpy(float) + corr)
    return out, {"alpha":H1_ALPHA,"features":features,"raw_squared_loss":True,"offset":"inner_oof_E05_main_effects_raw_panel_GM",
                 "correction_sd":float(np.std(corr)),"intercept":float(model.intercept_),"coef":[float(v) for v in np.ravel(model.coef_)]}

def _metric_rows(frame: pd.DataFrame, pred_cols: Sequence[str]) -> Mapping[str,Any]:
    require_columns(frame, ["actual","study_group","panel_size","panel_composition",*pred_cols])
    def one(g: pd.DataFrame, col: str) -> dict[str,Any]:
        y=g["actual"].to_numpy(float); p=g[col].to_numpy(float); require_finite(y,name="actual"); require_finite(p,name=col)
        sp=safe_spearman(y,p).to_dict(); return {"n":int(len(g)),"rmse":float(np.sqrt(np.mean((y-p)**2))),"spearman":sp.get("value"),"spearman_detail":sp}
    pooled={c:one(frame,c) for c in pred_cols}; by_study=[]
    for study,g in frame.groupby("study_group", observed=True):
        row={"study_group":str(study),"n":int(len(g)),"panel_size":int(g["panel_size"].iloc[0]),"panel_composition":str(g["panel_composition"].iloc[0])}
        for c in pred_cols: row[c]=one(g,c)
        by_study.append(row)
    equal={}
    for c in pred_cols:
        rms=np.asarray([r[c]["rmse"] for r in by_study],float); sps=np.asarray([r[c]["spearman"] for r in by_study if r[c]["spearman"] is not None],float)
        equal[c]={"rmse_mean":float(rms.mean()),"rmse_median":float(np.median(rms)),"rmse_worst":float(rms.max()),"spearman_mean":float(sps.mean()) if len(sps) else None}
    strata=[]
    for (size,comp),g in frame.groupby(["panel_size","panel_composition"], observed=True):
        row={"panel_size":int(size),"panel_composition":str(comp),"n":int(len(g)),"studies":int(g["study_group"].nunique())}
        for c in pred_cols: row[c]=one(g,c)
        strata.append(row)
    return {"pooled":pooled,"equal_study":equal,"by_study":by_study,"panel_strata":strata}

def _paired_delta(frame: pd.DataFrame, candidate: str, reference: str) -> Mapping[str,Any]:
    e0=np.abs(frame["actual"].to_numpy(float)-frame[reference].to_numpy(float)); e1=np.abs(frame["actual"].to_numpy(float)-frame[candidate].to_numpy(float)); gain=e0-e1
    return {"n":int(len(frame)),"candidate_better":int((gain>0).sum()),"reference_better":int((gain<0).sum()),"tie":int((gain==0).sum()),
            "mean_absolute_error_gain":float(gain.mean()),"error_complementarity_correlation":float(np.corrcoef(e0,e1)[0,1]) if len(frame)>1 and np.std(e0)>0 and np.std(e1)>0 else None}

def _rank_tie_shift(reference: pd.DataFrame, candidate: pd.DataFrame) -> Mapping[str,Any]:
    x=reference[["participant_id","prediction"]].rename(columns={"prediction":"ref"}).merge(candidate[["participant_id","prediction"]].rename(columns={"prediction":"cand"}),on="participant_id",validate="one_to_one")
    if len(x)!=len(reference) or len(x)!=len(candidate): raise DataContractError("V3-07 Challenge donor alignment failed")
    rr=pd.Series(x.ref).rank(method="average",pct=True).to_numpy(float); rc=pd.Series(x.cand).rank(method="average",pct=True).to_numpy(float)
    return {"donors":int(len(x)),"rank_spearman":safe_spearman(x.ref,x.cand).to_dict(),"mean_abs_weak_rank_shift":float(np.mean(np.abs(rr-rc))),
            "max_abs_weak_rank_shift":float(np.max(np.abs(rr-rc))),"reference_tie_groups":int(pd.Series(x.ref).value_counts().gt(1).sum()),"candidate_tie_groups":int(pd.Series(x.cand).value_counts().gt(1).sum())}

def _h1(config: BaselineConfig, inputs: Any, sequence_reference: pd.DataFrame, vaccine_reference: pd.DataFrame) -> tuple[Mapping[str,Any],pd.DataFrame,pd.DataFrame]:
    panel=_canonical_panel(inputs.challenge_strains); t=inputs.tables
    base=build_hai_model_dataset(t["public_serology"],t["challenge_serology"],t["participants"],t["investigations"],day=28,target_representation="residual",challenge_panel_strains=panel)
    ridge=add_hai_ontology_features(base,vaccine_reference=vaccine_reference,sequence_reference=normalize_sequence_reference_schema(sequence_reference),vaccine_2025=inputs.vaccine_strains,condition="ontology_sequence_local")
    outer=purged_leave_one_study_out(ridge.train["study_group"].astype(str),ridge.train["subject_group"].astype(str)); et_spec=_find_e05_model(config,B21_MODELS["Task2.2"])
    if et_spec.name!="et_subtype_d5_l10": raise DataContractError("V3-07 H1 ET control changed")
    counter=_FitCounter(limit=MAX_FITS_H1); pieces=[]
    for split in outer:
        tr=ridge.train.iloc[split.train_indices].reset_index(drop=True); va=ridge.train.iloc[split.validation_indices].reset_index(drop=True)
        cal=_panel_donors(_e05_inner_oof(tr,counter),official_panel=panel,prediction_column="post_prediction")
        if cal.empty: continue
        hold=va[["participant_id","subject_group","study_group","virus_strain","post_hai","log2_pre_hai"]].copy(); hold["post_prediction"]=_e05_predict(tr,va,counter)
        donors=_panel_donors(hold,official_panel=panel,prediction_column="post_prediction")
        if donors.empty: continue
        cand,contract=_fit_panel_residual(cal,donors,counter); donors["task22_panel_mean"]=cand; donors["e05_main_effects"]=donors["base_prediction"]
        held=str(split.held_out_group); heldsub=set(va.subject_group.astype(str)); b=base.train.reset_index(drop=True)
        btr=b.loc[(b.study_group.astype(str)!=held)&~b.subject_group.astype(str).isin(heldsub)].reset_index(drop=True); bva=b.loc[b.study_group.astype(str)==held].reset_index(drop=True)
        _,ett=fit_final_model(btr,bva,target_column=base.target_column,spec=et_spec,excluded_columns=base.excluded_columns); counter.add()
        ef=bva[["participant_id","subject_group","study_group","virus_strain","post_hai","log2_pre_hai"]].copy(); ef["post_prediction"]=base.target_prediction_to_post_hai(ett,frame=bva)
        etd=_panel_donors(ef,official_panel=panel,prediction_column="post_prediction")[["participant_id","study_group","base_prediction"]].rename(columns={"base_prediction":"et_subtype_d5_l10"})
        donors=donors.merge(etd,on=["participant_id","study_group"],how="left",validate="one_to_one"); donors["pre_hai"]=np.exp2(donors["log2_pre_panel_gm"])
        donors["outer_fold"]=split.name; donors["calibration_n"]=len(cal); donors["correction_sd"]=contract["correction_sd"]; pieces.append(donors)
    bank=pd.concat(pieces,ignore_index=True); proxy=bank.loc[bank.panel_size.eq(H1_TARGET_PROXY_SIZE)]; proxy_studies=tuple(sorted(set(proxy.study_group.astype(str))))
    if int(proxy.subject_group.nunique())!=H1_EXPECTED_PROXY_SUBJECTS or proxy_studies!=H1_EXPECTED_PROXY_STUDIES:
        raise DataContractError(f"V3-07 H1 fixed proxy support changed: subjects={proxy.subject_group.nunique()} studies={proxy_studies}")
    metrics=_metric_rows(bank,["task22_panel_mean","e05_main_effects","et_subtype_d5_l10","pre_hai"]); pm=_metric_rows(proxy,["task22_panel_mean","e05_main_effects","et_subtype_d5_l10","pre_hai"])
    cal=_panel_donors(_e05_inner_oof(ridge.train.reset_index(drop=True),counter),official_panel=panel,prediction_column="post_prediction"); cp=_e05_predict(ridge.train.reset_index(drop=True),ridge.challenge.reset_index(drop=True),counter)
    cf=ridge.challenge[["participant_id","subject_group","study_group","virus_strain","log2_pre_hai"]].copy(); cf["post_hai"]=1.0; cf["post_prediction"]=cp
    cd=_panel_donors(cf,official_panel=panel,prediction_column="post_prediction"); cval,_=_fit_panel_residual(cal,cd,counter)
    br=ridge.challenge[["participant_id","virus_strain"]].copy(); br["prediction"]=cp; bt=aggregate_hai_task_predictions(br,panel_strains=panel,task="Task2.2")
    challenge=cd[["participant_id"]].copy(); challenge["prediction"]=cval
    summary={"condition":"task22_panel_mean","fit_count":counter.count,"fit_limit":counter.limit,"metrics":metrics,"target_proxy_metrics":pm,
             "target_proxy":{"panel_size":9,"subjects":int(proxy.subject_group.nunique()),"studies":list(proxy_studies)},"paired_vs_e05":_paired_delta(bank,"task22_panel_mean","e05_main_effects"),
             "paired_target_proxy_vs_e05":_paired_delta(proxy,"task22_panel_mean","e05_main_effects"),"challenge_rank_tie_shift_vs_e05":_rank_tie_shift(bt[["participant_id","prediction"]],challenge),
             "contract":{"alpha":10.0,"features":["log2_pre_panel_gm","pre_panel_log2_sd"],"fixed_offset":"nested_inner_OOF_E05_main_effects","outer_subject_purge":True,"inner_oof_only":True,"positivity":True,"panel_strata_not_pooled_for_decision":True},
             "competition_submission_attempted":False,"public_leaderboard_used":False}
    return summary,bank,challenge

def _inner_d28_oof(day28: HAIModelDataset, source: pd.DataFrame, spec: Any, counter:_FitCounter) -> pd.DataFrame:
    source=source.reset_index(drop=True); parts=[]
    for split in _nested_splits(source):
        tr=source.iloc[split.train_indices].reset_index(drop=True); va=source.iloc[split.validation_indices].reset_index(drop=True)
        _,tar=fit_final_model(tr,va,target_column=day28.target_column,spec=spec,excluded_columns=day28.excluded_columns); counter.add()
        q=va[["participant_id","subject_group","study_group","virus_strain"]].copy(); q["predicted_d28"]=day28.target_prediction_to_post_hai(tar,frame=va); q["inner_split"]=split.name; parts.append(q)
    return pd.concat(parts,ignore_index=True)

def _pair_d28_d365(d28: pd.DataFrame,d365: pd.DataFrame,pred28: pd.DataFrame|None=None) -> pd.DataFrame:
    keys=["study_group","subject_group","virus_strain"]; a=d28.copy(); b=d365.copy(); a["virus_strain"]=a.virus_strain.map(canonicalize_strain); b["virus_strain"]=b.virus_strain.map(canonicalize_strain)
    aa=a[keys+["participant_id","post_hai"]].rename(columns={"participant_id":"participant_id_d28","post_hai":"d28_observed"}); bb=b[keys+["participant_id","post_hai","log2_pre_hai","age"]].rename(columns={"participant_id":"participant_id_d365","post_hai":"d365_observed"})
    out=aa.merge(bb,on=keys,how="inner",validate="one_to_one")
    if not out.participant_id_d28.astype(str).eq(out.participant_id_d365.astype(str)).all(): raise DataContractError("V3-07 H2 participant-year mismatch inside same-subject pair")
    if pred28 is not None:
        p=pred28.copy(); p["virus_strain"]=p.virus_strain.map(canonicalize_strain); out=out.merge(p[keys+["predicted_d28"]],on=keys,how="inner",validate="one_to_one")
    out["retention_target"]=np.log2(out.d365_observed.to_numpy(float)/out.d28_observed.to_numpy(float)); return out

def _fit_retention(train_pairs: pd.DataFrame,pred_frame: pd.DataFrame,counter:_FitCounter)->tuple[np.ndarray,Mapping[str,Any]]:
    feats=["log2_pre_hai","age","predicted_log2_d28"]
    x=pd.DataFrame({"log2_pre_hai":pd.to_numeric(train_pairs.log2_pre_hai,errors="coerce"),"age":pd.to_numeric(train_pairs.age,errors="coerce"),"predicted_log2_d28":np.log2(np.maximum(EPS,train_pairs.predicted_d28))})
    xp=pred_frame[feats].apply(pd.to_numeric,errors="coerce").copy()
    med=x.median(axis=0,skipna=True)
    if med.isna().any(): raise DataContractError("V3-07 H2 source feature median undefined")
    x=x.fillna(med); xp=xp.fillna(med)
    require_finite(x.to_numpy(),name="V3-07 H2 retention features"); require_finite(xp.to_numpy(),name="V3-07 H2 held features")
    y=train_pairs.retention_target.to_numpy(float); require_finite(y,name="V3-07 retention target"); sc=StandardScaler().fit(x); model=Ridge(alpha=H2_ALPHA).fit(sc.transform(x),y); counter.add()
    return model.predict(sc.transform(xp)),{"alpha":100.0,"features":["baseline HAI (log2)","age","predicted D28 (log2 representation)"],"observed_d28_feature":False,"imputation":"source_training_median"}

def _h2(config:BaselineConfig,inputs:Any)->tuple[Mapping[str,Any],pd.DataFrame,pd.DataFrame]:
    panel=_canonical_panel(inputs.challenge_strains); t=inputs.tables
    def build(day:int)->HAIModelDataset: return build_hai_model_dataset(t["public_serology"],t["challenge_serology"],t["participants"],t["investigations"],day=day,target_representation="residual",challenge_panel_strains=panel)
    d28=build(28); d365=build(365); spec=_find_e06b_model(config)
    if spec.name!="ridge_exact_a100" or float(spec.params.get("alpha",-1))!=100.0: raise DataContractError("V3-07 H2 frozen Ridge control changed")
    full_pairs=_pair_d28_d365(d28.train,d365.train)
    if len(full_pairs)!=H2_EXPECTED_PAIR_ROWS or full_pairs.subject_group.nunique()!=H2_EXPECTED_PAIR_SUBJECTS: raise DataContractError(f"V3-07 H2 paired support changed rows={len(full_pairs)} subjects={full_pairs.subject_group.nunique()}")
    outer=purged_leave_one_study_out(d365.train.study_group.astype(str),d365.train.subject_group.astype(str)); counter=_FitCounter(limit=MAX_FITS_H2); pieces=[]
    for split in outer:
        tr365=d365.train.iloc[split.train_indices].reset_index(drop=True); va=d365.train.iloc[split.validation_indices].reset_index(drop=True); held=str(split.held_out_group); heldsub=set(va.subject_group.astype(str))
        tr28=d28.train.loc[(d28.train.study_group.astype(str)!=held)&~d28.train.subject_group.astype(str).isin(heldsub)].reset_index(drop=True)
        pair=_pair_d28_d365(tr28,tr365,_inner_d28_oof(d28,tr28,spec,counter))
        if len(pair)<10: continue
        _,d28tar=fit_final_model(tr28,va,target_column=d28.target_column,spec=spec,excluded_columns=d28.excluded_columns); counter.add(); pd28=d28.target_prediction_to_post_hai(d28tar,frame=va)
        hf=pd.DataFrame({"log2_pre_hai":va.log2_pre_hai.to_numpy(float),"age":pd.to_numeric(va.age,errors="coerce").to_numpy(float),"predicted_log2_d28":np.log2(np.maximum(EPS,pd28))})
        retention,_=_fit_retention(pair,hf,counter); cand=np.exp2(np.clip(np.log2(np.maximum(EPS,pd28))+retention,-20,30))
        _,indtar=fit_final_model(tr365,va,target_column=d365.target_column,spec=spec,excluded_columns=d365.excluded_columns); counter.add(); ind=d365.target_prediction_to_post_hai(indtar,frame=va)
        twotar,_=_fit_two_head_ridge(tr28,tr365,va,day28=d28,day365=d365,spec=spec); counter.add(); two=d365.target_prediction_to_post_hai(twotar,frame=va)
        q=va[["participant_id","subject_group","study_group","virus_strain","post_hai","log2_pre_hai"]].copy(); q["task23_retention_strain"]=cand; q["independent_d365_strain"]=ind; q["e06b_two_head_strain"]=two
        c=_panel_donors(q.rename(columns={"task23_retention_strain":"post_prediction"}),official_panel=panel,prediction_column="post_prediction").rename(columns={"base_prediction":"task23_retention"})
        for src,dst in [("independent_d365_strain","independent_d365"),("e06b_two_head_strain","e06b_two_head")]:
            d=_panel_donors(q.rename(columns={src:"post_prediction"}),official_panel=panel,prediction_column="post_prediction")[["participant_id","study_group","base_prediction"]].rename(columns={"base_prediction":dst}); c=c.merge(d,on=["participant_id","study_group"],validate="one_to_one")
        c["outer_fold"]=split.name; pieces.append(c)
    bank=pd.concat(pieces,ignore_index=True); proxy=bank.loc[bank.panel_size.eq(H2_TARGET_PROXY_SIZE)]
    if proxy.subject_group.nunique()!=H2_EXPECTED_PROXY_SUBJECTS or proxy.study_group.nunique()!=H2_EXPECTED_PROXY_STUDIES: raise DataContractError(f"V3-07 H2 fixed proxy support changed subjects={proxy.subject_group.nunique()} studies={proxy.study_group.nunique()}")
    metrics=_metric_rows(bank,["task23_retention","independent_d365","e06b_two_head"]); pm=_metric_rows(proxy,["task23_retention","independent_d365","e06b_two_head"])
    pair=_pair_d28_d365(d28.train,d365.train,_inner_d28_oof(d28,d28.train.reset_index(drop=True),spec,counter))
    _,d28tar=fit_final_model(d28.train,d365.challenge,target_column=d28.target_column,spec=spec,excluded_columns=d28.excluded_columns); counter.add(); pd28=d28.target_prediction_to_post_hai(d28tar,frame=d365.challenge)
    hf=pd.DataFrame({"log2_pre_hai":d365.challenge.log2_pre_hai.to_numpy(float),"age":pd.to_numeric(d365.challenge.age,errors="coerce").to_numpy(float),"predicted_log2_d28":np.log2(np.maximum(EPS,pd28))})
    ret,_=_fit_retention(pair,hf,counter); cand=np.exp2(np.clip(np.log2(np.maximum(EPS,pd28))+ret,-20,30))
    _,indtar=fit_final_model(d365.train,d365.challenge,target_column=d365.target_column,spec=spec,excluded_columns=d365.excluded_columns); counter.add(); ind=d365.target_prediction_to_post_hai(indtar,frame=d365.challenge)
    twotar,_=_fit_two_head_ridge(d28.train,d365.train,d365.challenge,day28=d28,day365=d365,spec=spec); counter.add(); two=d365.target_prediction_to_post_hai(twotar,frame=d365.challenge)
    def task(pred):
        r=d365.challenge[["participant_id","virus_strain"]].copy(); r["prediction"]=pred; return aggregate_hai_task_predictions(r,panel_strains=panel,task="Task2.3")
    ct=task(cand); it=task(ind); tt=task(two); challenge=ct[["participant_id","prediction"]].copy()
    summary={"condition":"task23_retention","fit_count":counter.count,"fit_limit":counter.limit,"paired_teacher":{"rows":len(full_pairs),"unique_subjects":int(full_pairs.subject_group.nunique()),"target":"log2(Y365/Y28)"},
             "metrics":metrics,"target_proxy_metrics":pm,"target_proxy":{"panel_size":8,"subjects":int(proxy.subject_group.nunique()),"studies":int(proxy.study_group.nunique())},
             "paired_vs_independent":_paired_delta(bank,"task23_retention","independent_d365"),"paired_target_proxy_vs_independent":_paired_delta(proxy,"task23_retention","independent_d365"),
             "challenge_rank_tie_shift_vs_independent":_rank_tie_shift(it[["participant_id","prediction"]],ct[["participant_id","prediction"]]),"challenge_rank_tie_shift_vs_e06b":_rank_tie_shift(tt[["participant_id","prediction"]],ct[["participant_id","prediction"]]),
             "contract":{"alpha":100.0,"features":["baseline HAI","age","predicted D28"],"retention_target":"log2(Y365/Y28)","inner_oof_d28_only":True,"held_observed_d28_used":False,"challenge_observed_d28_used":False,"outer_subject_purge":True},
             "competition_submission_attempted":False,"public_leaderboard_used":False}
    return summary,bank,challenge

def run_v3_07(condition:str,config:BaselineConfig,inputs:Any,*,sequence_reference:pd.DataFrame|None=None,vaccine_reference:pd.DataFrame|None=None)->tuple[Mapping[str,Any],pd.DataFrame,pd.DataFrame]:
    if condition not in CONDITIONS: raise DataContractError(f"unknown V3-07 condition: {condition}")
    if str(config.section("hai").get("target_representation","residual"))!="residual": raise DataContractError("V3-07 requires residual frozen B2.1 config")
    if condition=="task22_panel_mean":
        if sequence_reference is None or vaccine_reference is None: raise DataContractError("V3-07 H1 requires sequence/vaccine references")
        result,oof,challenge=_h1(config,inputs,sequence_reference,vaccine_reference)
    else: result,oof,challenge=_h2(config,inputs)
    return {"experiment":EXPERIMENT,"new_candidate_conditions":[condition],"result":result,"fit_count":int(result["fit_count"]),"competition_submission_attempted":False,"final_submission_selection_attempted":False,"public_leaderboard_used":False,"saved_outer_oof_reused_as_inner_teacher":False,"automatic_parameter_sweep":False},oof,challenge

def write_v3_07_outputs(condition:str,aggregate:Mapping[str,Any],oof:pd.DataFrame,challenge:pd.DataFrame,output_dir:Path)->Mapping[str,Any]:
    if condition not in CONDITIONS: raise DataContractError("V3-07 output condition invalid")
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); prefix="v3_v07_h1" if condition=="task22_panel_mean" else "v3_v07_h2"
    paths={"oof":output_dir/f"{prefix}_oof_bank.csv","challenge":output_dir/f"{prefix}_challenge_bank.csv","summary":output_dir/f"{prefix}_summary.json"}
    if any(p.exists() for p in paths.values()): raise DataContractError("V3-07 outputs require empty destination names")
    oof.to_csv(paths["oof"],index=False); challenge.to_csv(paths["challenge"],index=False); paths["summary"].write_text(json.dumps(aggregate,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    files=[]
    for role,p in paths.items(): files.append({"role":role,"filename":p.name,"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"rows":int(len(oof) if role=="oof" else len(challenge)) if role!="summary" else None})
    manifest={"experiment":EXPERIMENT,"condition":condition,"files":files,"row_level_banks_private":True}; mp=output_dir/f"{prefix}_manifest.json"; mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return manifest

__all__=["EXPERIMENT","CONDITIONS","run_v3_07","write_v3_07_outputs","_nested_splits","_panel_donors","_fit_panel_residual","_pair_d28_d365"]
