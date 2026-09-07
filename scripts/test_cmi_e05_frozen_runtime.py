#!/usr/bin/env python3
"""Execute the frozen E05 scientific module on synthetic Challenge-shaped HAI rows."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

def load_runtime(path: Path):
    spec=importlib.util.spec_from_file_location("e05_frozen_runtime",path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def synthetic_frame():
    strains=("A/California/07/2009_H1N1_MDCK","A/Switzerland/9715293/2013_H3N2_MDCK","B/Phuket/3073/2013_Yamagata_MDCK")
    rows=[]
    for si,study in enumerate(("SDY_A","SDY_B")):
        for donor in range(4):
            mean=3.0+.2*donor+.1*si; std=.5+.05*donor; breadth=1+donor; age=25+8*donor+si
            for j,strain in enumerate(strains):
                rows.append({"participant_id":f"{study}-p{donor}","study_group":study,"subject_group":f"{study}-s{donor}","virus_strain":strain,"log2_pre_hai":2.5+.3*donor+.2*j,"hai_log2_mean":mean,"hai_log2_std":std,"hai_breadth_ge_40":breadth,"age":age,"strain_subtype":("H1N1","H3N2","B")[j],"strain_substrate":"MDCK","ontology_official_component_vaccine":("yes","no","unknown")[j],"ontology_seq_distance_to_study_vaccine":.1+.2*j})
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--runtime",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args()
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False); runtime=load_runtime(args.runtime.resolve())
    with tempfile.TemporaryDirectory(prefix="e05-synthetic-package-") as tmp:
        package_path=Path(tmp)/"cmi_flu_bundle.zip"; package_path.write_bytes(runtime.package_bytes()); sys.path.insert(0,str(package_path))
        try:
            run,e05=runtime.load_e05_module(); assert callable(run)
            frame=synthetic_frame(); main=e05._e05_design(frame,interactions=False); interaction=e05._e05_design(frame,interactions=True); assert interaction.shape[1]-main.shape[1]==8
            target=3.5+.6*main["g_log2_pre_hai"].to_numpy()+.2*main["z_age_rank"].to_numpy(); train_rows=18; weights=e05._hierarchical_sample_weights(frame.iloc[:train_rows]); _,prediction=e05._fit_ridge(interaction.iloc[:train_rows],target[:train_rows],weights,interaction.iloc[train_rows:],interactions=True); assert len(prediction)==6 and np.isfinite(prediction).all()
            splits=e05._simultaneous_subject_strain_splits(frame,panel_strains=tuple(frame["virus_strain"].unique())); assert splits
        finally:
            sys.path.remove(str(package_path))
    result={"schema_version":1,"experiment":"strategy_v2_e05_hai_donor_strain","science_commit":runtime.SCIENCE_COMMIT,"e05_blob":runtime.E05_BLOB,"rows":len(frame),"interaction_count":8,"prediction_rows":len(prediction),"prediction_finite":True,"split_count":len(splits),"competition_submission_attempted":False,"leaderboard_used_for_selection":False}
    (out/"synthetic-result.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    manifest={"final_files":["synthetic-result.json","runtime-manifest.json","release-receipt.json"],"runtime_sha256":hashlib.sha256(args.runtime.read_bytes()).hexdigest(),"science_commit":runtime.SCIENCE_COMMIT}; (out/"runtime-manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    receipt={"status":"PASS","model_executed":True,"result_validated":True,"all_final_files_saved":True,"cleanup_completed":True,"aggregate_only":True}; (out/"release-receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    assert {path.name for path in out.iterdir() if path.is_file()}==set(manifest["final_files"])
    print("CMI_FLU_E05_SYNTHETIC PASS model=true validate=true final_files=true receipt=true cleanup=true")
    return 0
if __name__=="__main__": raise SystemExit(main())
