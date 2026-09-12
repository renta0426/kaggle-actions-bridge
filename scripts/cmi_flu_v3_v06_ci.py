#!/usr/bin/env python3
"""Credential-free real-library full-path regression for V3-06."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import numpy as np
import pandas as pd

import cmi_flu_v3_v06_prepare as prepare


def load(path: Path, name: str):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module


@dataclass
class FakeConfig:
    specs: list
    baseline: str = "b021_taskwise_robust"
    random_state: int = 42
    def section(self,name: str, required: bool=True):
        if name=="hai": return {"target_representation":"residual"}
        if name=="selection": return {"policy":"robust_v1"}
        return {}
    def model_specs(self,name: str):
        return list(self.specs) if name=="hai" else []


def hai_dataset(rt):
    strains=("H1N1 A/Victoria/4897/2022","Vic B/Austria/1359417/2021","H3N2 A/Tasmania/503/2020")
    rows=[]
    for sidx,study in enumerate(("Y1","Y2","Y3","Y4","Y5")):
        for i in range(12):
            for k,strain in enumerate(strains):
                pre=3.0+.04*i+.08*k+.02*sidx
                fold=.25+.012*sidx+.006*i+.008*k
                rows.append({"participant_id":f"{study}_P{i}","subject_group":f"{study}_U{i}","study_group":study,"virus_strain":strain,"strain_subtype":"H1" if k==0 else "H3","log2_pre_hai":pre,"target_log2_fold":fold,"post_hai":float(2**(pre+fold)),"age":20+i})
    challenge=[]
    for i in range(40):
        for k,strain in enumerate(strains):
            challenge.append({"participant_id":f"C{i:02d}","subject_group":f"CU{i}","study_group":"2025LJI","virus_strain":strain,"strain_subtype":"H1" if k==0 else "H3","log2_pre_hai":3.2+.02*i+.08*k,"age":20+i%40})
    return rt.HAIModelDataset(day=28,train=pd.DataFrame(rows),challenge=pd.DataFrame(challenge),target_representation="residual"),strains


def full_science(runtime: Path):
    generated=load(runtime,"v306_generated_runtime")
    with tempfile.TemporaryDirectory(prefix="v306-pkg-") as td:
        package=Path(td)/"bundle.zip"; package.write_bytes(generated.package_bytes()); sys.path.insert(0,str(package))
        try:
            rt=generated.load_v3_v06_module()
            from cmi_flu.models import ModelSpec
            dataset,strains=hai_dataset(rt)
            specs=[
                ModelSpec("et_subtype_d3_l5","extra_trees",{"n_estimators":20,"max_depth":3,"min_samples_leaf":5,"max_features":"sqrt"},drop_columns=("virus_strain",)),
                ModelSpec("et_subtype_d5_l10","extra_trees",{"n_estimators":20,"max_depth":5,"min_samples_leaf":10,"max_features":"sqrt"},drop_columns=("virus_strain",)),
            ]
            config=FakeConfig(specs)
            inputs=SimpleNamespace(tables={"public_serology":pd.DataFrame(),"challenge_serology":pd.DataFrame(),"participants":pd.DataFrame(),"investigations":pd.DataFrame()},vaccine_strains=strains[:2],challenge_strains=strains)
            native=rt.build_hai_model_dataset; rt.build_hai_model_dataset=lambda *args,**kwargs: dataset
            try:
                aggregate,oof,challenge=rt.run_v3_06(config,inputs)
            finally:
                rt.build_hai_model_dataset=native
            assert aggregate["new_candidate_conditions"]==["task21_et_log_affine","task22_et_log_affine"]
            assert aggregate["fit_count"]==48 and aggregate["fit_count"]<=256
            assert aggregate["calibration_contract"]["inner_study_folds"]==3
            assert aggregate["calibration_contract"]["affine_or_power_grid_reopened"] is False
            assert len(challenge)==80
            for task in ("Task2.1","Task2.2"):
                assert aggregate["tasks"][task]["challenge"]["rank_contract"]["same_rank_vector"] is True
                assert aggregate["tasks"][task]["challenge"]["rank_contract"]["same_tie_equivalence"] is True
            with tempfile.TemporaryDirectory(prefix="v306-out-") as out:
                manifest=rt.write_v3_06_outputs(aggregate,oof,challenge,out)
                assert len(manifest["files"])==2
                safe=(Path(out)/"v3_v06_summary.json").read_text()+(Path(out)/"v3_v06_bank_manifest.json").read_text()
                assert "C00" not in safe and "Y1_P0" not in safe
            print(f"V306_FULL_SCIENCE_SYNTHETIC PASS fit_count={aggregate['fit_count']} oof_rows={len(oof)} challenge_rows={len(challenge)}")
        finally:
            if sys.path and sys.path[0]==str(package): sys.path.pop(0)


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--repository-root",required=True,type=Path); p.add_argument("--workdir",required=True,type=Path); a=p.parse_args()
    root=a.repository_root.resolve(); work=a.workdir.resolve(); work.mkdir(parents=True,exist_ok=True)
    one,two=work/"runtime.py",work/"runtime2.py"
    prepare.build_runtime(root,one); prepare.build_runtime(root,two)
    assert one.read_bytes()==two.read_bytes()
    compile(one.read_text(),str(one),"exec")
    mod=load(one,"v306_selftest_runtime"); assert mod.self_test()==0
    full_science(one)
    raw=one.read_bytes(); print(f"V306_CI PASS bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()} real_libraries=true full_science_path=true kaggle_write=0 competition_submit=0")
    return 0

if __name__=="__main__": raise SystemExit(main())
