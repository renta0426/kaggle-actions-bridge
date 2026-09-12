#!/usr/bin/env python3
"""V3-06 runtime compatibility using explicit frozen calibration helpers."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile

import cmi_flu_v3_v06_prepare as parent
from cmi_flu_v3_batch1_prepare import replace_function


def build_runtime(root: Path, output: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="v306-v3-") as td:
        raw = parent.build_runtime(root, Path(td) / "parent.py")
    replacement = r'''def _v306_json_safe(value):
    import numpy as np
    if value is None or isinstance(value,(str,bool,int)): return value
    if isinstance(value,float): return value if np.isfinite(value) else None
    if isinstance(value,np.generic): return _v306_json_safe(value.item())
    if isinstance(value,dict): return {str(k):_v306_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_v306_json_safe(v) for v in value]
    if hasattr(value,"to_dict"): return _v306_json_safe(value.to_dict())
    return value

def _v306_canonical_panel(values):
    from cmi_flu.aliases import canonicalize_strain
    from cmi_flu.contracts import DataContractError
    panel=tuple(dict.fromkeys(canonicalize_strain(value) for value in values))
    if not panel or any(not value for value in panel): raise DataContractError("V306 compat panel empty")
    return panel

def _v306_panel_frame(frame,prediction,*,panel_strains,split):
    import numpy as np, pandas as pd
    from cmi_flu.aliases import canonicalize_strain
    from cmi_flu.contracts import DataContractError, require_columns, require_finite
    from cmi_flu.targets import geometric_mean
    require_columns(frame,["participant_id","study_group","subject_group","virus_strain","post_hai"])
    values=np.asarray(prediction,dtype=float).reshape(-1); require_finite(values,name="V306 compat panel prediction")
    if len(values)!=len(frame) or (values<=0).any(): raise DataContractError("V306 compat panel prediction invalid")
    work=frame[["participant_id","study_group","subject_group","virus_strain","post_hai"]].copy(); work["virus_strain"]=work["virus_strain"].map(canonicalize_strain); work["prediction"]=values
    if isinstance(split,str): work["split"]=split
    else:
        labels=np.asarray(list(split),dtype=object)
        if len(labels)!=len(work): raise DataContractError("V306 compat split length")
        work["split"]=labels.astype(str)
    panel=set(_v306_canonical_panel(panel_strains)); work=work.loc[work["virus_strain"].isin(panel)].copy()
    if work.empty: raise DataContractError("V306 compat zero panel overlap")
    work["post_hai"]=pd.to_numeric(work["post_hai"],errors="coerce")
    if not np.isfinite(work["post_hai"].to_numpy(float)).all() or (work["post_hai"].to_numpy(float)<=0).any(): raise DataContractError("V306 compat invalid target")
    per=work.groupby(["split","study_group","participant_id","subject_group","virus_strain"],dropna=False,observed=True).agg(target=("post_hai",geometric_mean),prediction=("prediction",geometric_mean)).reset_index()
    return per.groupby(["split","study_group","participant_id","subject_group"],dropna=False,observed=True).agg(target=("target",geometric_mean),prediction=("prediction",geometric_mean),panel_size=("virus_strain","nunique")).reset_index()

def _v306_panel_frame_from_oof(evaluation,*,panel_strains):
    from cmi_flu.contracts import require_columns
    oof=evaluation.enriched_oof; require_columns(oof,["split","participant_id","study_group","subject_group","virus_strain","post_hai","post_prediction"])
    return _v306_panel_frame(oof,oof["post_prediction"].to_numpy(float),panel_strains=panel_strains,split=oof["split"].astype(str).tolist())

def _v306_fit_calibrator(frame,*,kind):
    import numpy as np, pandas as pd
    from scipy.optimize import lsq_linear
    from cmi_flu.contracts import DataContractError, require_columns, require_finite
    if kind!="log2_affine": raise DataContractError("V306 compat only log2_affine authorized")
    require_columns(frame,["study_group","target","prediction"])
    studies=frame["study_group"].astype(str); counts=studies.value_counts()
    if int(studies.nunique())<2: raise DataContractError("V306 calibration requires >=2 studies")
    x=pd.to_numeric(frame["prediction"],errors="coerce").to_numpy(float); y=pd.to_numeric(frame["target"],errors="coerce").to_numpy(float); require_finite(x,name="V306 x"); require_finite(y,name="V306 y")
    if len(x)<4 or (x<=0).any() or (y<=0).any(): raise DataContractError("V306 calibration invalid values")
    w=np.asarray([1.0/float(counts.loc[s]) for s in studies],dtype=float); w*=len(w)/float(w.sum()); root=np.sqrt(w); design=np.column_stack([np.ones(len(x)),np.log2(x)])
    result=lsq_linear(design*root[:,None],np.log2(y)*root,bounds=([-np.inf,1e-6],[np.inf,np.inf]),method="trf")
    if not result.success: raise DataContractError("V306 log2 calibration fit failed")
    a,b=map(float,result.x); return {"kind":"log2_affine","intercept":a,"slope":b,"training_panel_donors":int(len(frame)),"training_studies":int(studies.nunique())}

def _v306_apply_calibrator(values,params):
    import numpy as np
    from cmi_flu.contracts import DataContractError, require_finite
    x=np.asarray(values,dtype=float).reshape(-1); require_finite(x,name="V306 calibration input"); a=float(params["intercept"]); b=float(params["slope"])
    if (x<=0).any() or b<1e-6: raise DataContractError("V306 calibration apply invalid")
    out=np.exp2(a+b*np.log2(x)); require_finite(out,name="V306 calibrated prediction"); return out

def _v306_metric_bundle(frame):
    import numpy as np, pandas as pd
    from cmi_flu.metrics import evaluate_predictions, grouped_metrics
    folds=grouped_metrics(frame,group_columns=["study_group"],target_column="target",prediction_column="prediction")
    def summary(x):
        sp=pd.to_numeric(x["spearman"],errors="coerce").to_numpy(float); rm=pd.to_numeric(x["rmse"],errors="coerce").to_numpy(float); sp=sp[np.isfinite(sp)]; rm=rm[np.isfinite(rm)]
        return {"count":int(len(x)),"spearman_mean":float(np.mean(sp)) if sp.size else None,"spearman_median":float(np.median(sp)) if sp.size else None,"spearman_min":float(np.min(sp)) if sp.size else None,"rmse_mean":float(np.mean(rm)) if rm.size else None,"rmse_median":float(np.median(rm)) if rm.size else None,"rmse_max":float(np.max(rm)) if rm.size else None}
    return _v306_json_safe({"pooled":evaluate_predictions(frame["target"],frame["prediction"]),"equal_study_summary":summary(folds),"study_metrics":folds.to_dict(orient="records")})

def _v306_rank_preservation(raw,calibrated):
    import numpy as np
    from cmi_flu.metrics import percentile_rank, safe_spearman
    keys=["split","study_group","participant_id","subject_group"]; aligned=raw[keys+["prediction"]].rename(columns={"prediction":"raw_prediction"}).merge(calibrated[keys+["prediction"]].rename(columns={"prediction":"calibrated_prediction"}),on=keys,how="inner",validate="one_to_one"); maximum=0.0; rows=[]
    for study,g in aligned.groupby("study_group",observed=True):
        a=percentile_rank(g["raw_prediction"].to_numpy(float)); b=percentile_rank(g["calibrated_prediction"].to_numpy(float)); d=np.abs(a-b); maximum=max(maximum,float(d.max()) if len(d) else 0.0); rows.append({"study":str(study),"n":int(len(g)),"rank_spearman_raw_vs_calibrated":safe_spearman(g["raw_prediction"],g["calibrated_prediction"]).to_dict(),"max_absolute_percentile_difference":float(d.max()) if len(d) else 0.0})
    return _v306_json_safe({"max_absolute_percentile_difference":maximum,"preserved_within_tolerance":bool(maximum<=1e-12),"by_study":rows})

def _v306_rmse_diagnostics(raw_metrics,calibrated_metrics):
    import numpy as np
    from cmi_flu.contracts import DataContractError
    raw={str(r["study_group"]):r for r in raw_metrics["study_metrics"]}; cal={str(r["study_group"]):r for r in calibrated_metrics["study_metrics"]}
    if set(raw)!=set(cal): raise DataContractError("V306 RMSE study mismatch")
    reductions=[]; ratios=[]; rows=[]
    for study in sorted(raw):
        rr=float(raw[study]["rmse"]); cr=float(cal[study]["rmse"]); reductions.append(1-cr/rr); ratios.append(cr/rr); rows.append({"study":study,"raw_rmse":rr,"calibrated_rmse":cr,"relative_reduction":1-cr/rr})
    rmean=float(raw_metrics["equal_study_summary"]["rmse_mean"]); cmean=float(calibrated_metrics["equal_study_summary"]["rmse_mean"])
    return _v306_json_safe({"equal_study_mean_relative_reduction":1-cmean/rmean,"median_study_relative_reduction":float(np.median(reductions)),"worst_study_rmse_ratio":float(np.max(ratios)),"by_study":rows})

def _v306_historical_panel_coverage(dataset,*,panel_strains):
    from cmi_flu.aliases import canonicalize_strain
    panel=set(_v306_canonical_panel(panel_strains)); work=dataset.train[["study_group","participant_id","virus_strain"]].copy(); work["virus_strain"]=work["virus_strain"].map(canonicalize_strain); work=work.loc[work["virus_strain"].isin(panel)].drop_duplicates(); sizes=work.groupby(["study_group","participant_id"],observed=True)["virus_strain"].nunique()
    return {"requested_strains":int(len(panel)),"historically_observed_requested_strains":int(work["virus_strain"].nunique()),"panel_proxy_donors":int(len(sizes)),"panel_size_min":int(sizes.min()) if len(sizes) else 0,"panel_size_max":int(sizes.max()) if len(sizes) else 0,"complete_requested_panel_donors":int((sizes==len(panel)).sum())}

def _install_v306_calibration_compat():
    import sys, types
    base=types.ModuleType("cmi_flu.strategy_e06"); base.__package__="cmi_flu"; base._json_safe=_v306_json_safe; base._canonical_panel=_v306_canonical_panel; base._panel_frame=_v306_panel_frame; base._panel_frame_from_oof=_v306_panel_frame_from_oof; base._fit_calibrator=_v306_fit_calibrator; base._apply_calibrator=_v306_apply_calibrator; base._metric_bundle=_v306_metric_bundle; base._rank_preservation=_v306_rank_preservation; base._rmse_diagnostics=_v306_rmse_diagnostics; base._historical_panel_coverage=_v306_historical_panel_coverage; sys.modules["cmi_flu.strategy_e06"]=base
    compat=types.ModuleType("cmi_flu.strategy_e06_v2"); compat.__package__="cmi_flu"; compat._fit_calibrator=_v306_fit_calibrator; sys.modules["cmi_flu.strategy_e06_v2"]=compat

def load_v3_v06_module() -> object:
    import sys, types
    load_e12c_v2_module(); load_v3_dependency_closure(); _install_v306_calibration_compat()
    module=types.ModuleType("cmi_flu.strategy_v3_v06"); module.__file__="<cmi_flu.strategy_v3_v06>"; module.__package__="cmi_flu"; sys.modules["cmi_flu.strategy_v3_v06"]=module
    exec(compile(V306_SOURCE,"cmi_flu/strategy_v3_v06.py","exec"),module.__dict__,module.__dict__)
    for name in ("run_v3_06","write_v3_06_outputs"):
        if not callable(getattr(module,name,None)): raise BridgeContractError("v306_entry_missing:"+name)
    if module.MODEL_BY_TASK!={"Task2.1":"et_subtype_d3_l5","Task2.2":"et_subtype_d5_l10"} or module.CALIBRATION_KIND!="log2_affine" or module.INNER_STUDY_FOLDS!=3 or module.MAX_FITS!=256: raise BridgeContractError("v306_contract_changed")
    return module
'''
    raw=replace_function(raw,"load_v3_v06_module",replacement)
    compile(raw,"v306-runtime-v3.py","exec"); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(raw,encoding="utf-8"); return raw


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--repository-root",required=True,type=Path); p.add_argument("--output",required=True,type=Path); a=p.parse_args(); raw=build_runtime(a.repository_root.resolve(),a.output.resolve()).encode(); print(f"V306_PREPARE_V3 PASS bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()} compat=e06_log2_primitives_only submission=false"); return 0

if __name__=="__main__": raise SystemExit(main())
