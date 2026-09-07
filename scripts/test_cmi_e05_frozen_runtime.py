#!/usr/bin/env python3
"""Execute the frozen E05 scientific module on synthetic Challenge-shaped HAI rows."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

def load_runtime(path:Path):
    spec=importlib.util.spec_from_file_location("e05_frozen_runtime",path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def synthetic_frame():
    strains=("A/California/07/2009_H1N1_MDCK","A/Switzerland/9715293/2013_H3N2_MDCK","B/Phuket/3073/2013_Yamagata_MDCK")
    rows=[]
    for si,study in enumerate(("SDY_A","SDY_B")):
        for d in range(4):
            mean=3.0+.2*d+.1*si; std=.5+.05*d; breadth=1+d; age=25+8*d+si
            for j,strain in enumerate(strains):
                rows.append({"participant_id":f"{study}-p{d}","study_group":study,"subject_group":f"{study}-s{d}","virus_strain":strain,"log2_pre_hai":2.5+.3*d+.2*j,"hai_log2_mean":mean,"hai_log2_std":std,"hai_breadth_ge_40":breadth,"age":age,"strain_subtype":("H1N1","H3N2","B")[j],"strain_substrate":"MDCK","ontology_official_component_vaccine":("yes","no","unknown")[j],"ontology_seq_distance_to_study_vaccine":.1+.2*j})
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--runtime",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False); runtime=load_runtime(a.runtime.resolve()); run,e05,_=runtime.load_e05_module(); assert callable(run)
    frame=synthetic_frame(); main=e05._e05_design(frame,interactions=False); inter=e05._e05_design(frame,interactions=True); assert inter.shape[1]-main.shape[1]==8; weights=e05._hierarchical_sample_weights(frame); target=3.5+.6*main["g_log2_pre_hai"].to_numpy()+.2*main["z_age_rank"].to_numpy(); _,pred=e05._fit_ridge(inter.iloc[:18],target[:18],weights[:18],inter.iloc[18:],interactions=True); assert len(pred)==6 and np.isfinite(pred).all(); splits=e05._simultaneous_subject_strain_splits(frame,panel_strains=tuple(frame["virus_strain"].unique())); assert splits
    result={"schema_version":1,"experiment":"strategy_v2_e05_hai_donor_strain","science_commit":runtime.SCIENCE_COMMIT,"e05_blob":runtime.E05_BLOB,"rows":len(frame),"interaction_count":8,"prediction_rows":len(pred),"prediction_finite":True,"split_count":len(splits),"competition_submission_attempted":False,"leaderboard_used_for_selection":False}
    (out/"synthetic-result.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n"); manifest={"final_files":["synthetic-result.json","runtime-manifest.json","release-receipt.json"],"runtime_sha256":hashlib.sha256(a.runtime.read_bytes()).hexdigest(),"science_commit":runtime.SCIENCE_COMMIT}; (out/"runtime-manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n"); receipt={"status":"PASS","model_executed":True,"result_validated":True,"all_final_files_saved":True,"cleanup_completed":True,"aggregate_only":True}; (out/"release-receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n"); allowed=set(manifest["final_files"]); assert {p.name for p in out.iterdir() if p.is_file()}==allowed; print("CMI_FLU_E05_SYNTHETIC PASS model=true validate=true final_files=true receipt=true cleanup=true")
    return 0
if __name__=="__main__": raise SystemExit(main())
