#!/usr/bin/env python3
"""Exercise the exact generated V3-07 runtime, writer, contracts and failure cleanup."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCIENCE_COMMIT = "9bfe05664e4d96f09631a3210f7951bf9dfefe71"
SCIENCE_BLOB = "4cf804e0c10889d0e037457ac096711b77200ea9"


def load_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("v307_generated_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metric(n: int, rmse: float, spearman: float) -> dict:
    return {"n": n, "rmse": rmse, "spearman": spearman, "spearman_detail": {"value": spearman, "n": n}}


def block(condition: str) -> dict:
    if condition == "task22_panel_mean":
        vals = {
            "task22_panel_mean": metric(24, 8.0, 0.42),
            "e05_main_effects": metric(24, 10.0, 0.40),
            "et_subtype_d5_l10": metric(24, 11.0, 0.38),
            "pre_hai": metric(24, 12.0, 0.35),
        }
    else:
        vals = {
            "task23_retention": metric(24, 7.0, 0.44),
            "independent_d365": metric(24, 8.0, 0.41),
            "e06b_two_head": metric(24, 7.8, 0.42),
        }
    equal = {name: {"rmse_mean": row["rmse"], "rmse_median": row["rmse"], "rmse_worst": row["rmse"] * 1.1, "spearman_mean": row["spearman"]} for name, row in vals.items()}
    study = {"study_group":"SYN_A","n":12,"panel_size":9 if condition=="task22_panel_mean" else 8,"panel_composition":"S1|S2"}
    for name, row in vals.items(): study[name] = row
    stratum = {"panel_size":9 if condition=="task22_panel_mean" else 8,"panel_composition":"S1|S2","n":24,"studies":2}
    for name, row in vals.items(): stratum[name] = row
    return {"pooled":vals,"equal_study":equal,"by_study":[study],"panel_strata":[stratum]}


def synthetic_aggregate(runtime, condition: str) -> dict:
    metrics = block(condition)
    if condition == "task22_panel_mean":
        payload = {
            "condition": condition,
            "fit_count": 9,
            "fit_limit": 256,
            "metrics": metrics,
            "target_proxy_metrics": metrics,
            "target_proxy": {"panel_size":9,"subjects":238,"studies":["2023_UGA","2024UGA"]},
            "paired_vs_e05": {"n":24,"candidate_better":15,"reference_better":9,"tie":0,"mean_absolute_error_gain":1.0,"error_complementarity_correlation":0.7},
            "paired_target_proxy_vs_e05": {"n":24,"candidate_better":15,"reference_better":9,"tie":0,"mean_absolute_error_gain":1.0,"error_complementarity_correlation":0.7},
            "challenge_rank_tie_shift_vs_e05": {"donors":40,"rank_spearman":{"value":0.98},"mean_abs_weak_rank_shift":0.03,"max_abs_weak_rank_shift":0.1,"reference_tie_groups":0,"candidate_tie_groups":0},
            "contract": {"alpha":10.0,"features":["log2_pre_panel_gm","pre_panel_log2_sd"],"fixed_offset":"nested_inner_OOF_E05_main_effects","outer_subject_purge":True,"inner_oof_only":True,"positivity":True,"panel_strata_not_pooled_for_decision":True},
            "competition_submission_attempted":False,"public_leaderboard_used":False,
        }
    else:
        payload = {
            "condition": condition,
            "fit_count": 12,
            "fit_limit": 256,
            "paired_teacher": {"rows":11913,"unique_subjects":567,"target":"log2(Y365/Y28)"},
            "metrics": metrics,
            "target_proxy_metrics": metrics,
            "target_proxy": {"panel_size":8,"subjects":439,"studies":3},
            "paired_vs_independent": {"n":24,"candidate_better":14,"reference_better":10,"tie":0,"mean_absolute_error_gain":0.8,"error_complementarity_correlation":0.72},
            "paired_target_proxy_vs_independent": {"n":24,"candidate_better":14,"reference_better":10,"tie":0,"mean_absolute_error_gain":0.8,"error_complementarity_correlation":0.72},
            "challenge_rank_tie_shift_vs_independent": {"donors":40,"rank_spearman":{"value":0.97},"mean_abs_weak_rank_shift":0.04,"max_abs_weak_rank_shift":0.12,"reference_tie_groups":0,"candidate_tie_groups":0},
            "challenge_rank_tie_shift_vs_e06b": {"donors":40,"rank_spearman":{"value":0.96},"mean_abs_weak_rank_shift":0.05,"max_abs_weak_rank_shift":0.13,"reference_tie_groups":0,"candidate_tie_groups":0},
            "contract": {"alpha":100.0,"features":["baseline HAI","age","predicted D28"],"retention_target":"log2(Y365/Y28)","inner_oof_d28_only":True,"held_observed_d28_used":False,"challenge_observed_d28_used":False,"outer_subject_purge":True},
            "competition_submission_attempted":False,"public_leaderboard_used":False,
        }
    result = {
        "experiment":"strategy_v3_v07_hai_panel_mean_and_retention",
        "new_candidate_conditions":[condition],
        "result":payload,
        "fit_count":payload["fit_count"],
        "competition_submission_attempted":False,
        "final_submission_selection_attempted":False,
        "public_leaderboard_used":False,
        "saved_outer_oof_reused_as_inner_teacher":False,
        "automatic_parameter_sweep":False,
        "runtime": {"request_id":runtime.V307_REQUEST_ID,"target_kernel":runtime.V307_TARGET_KERNEL,"science_commit":SCIENCE_COMMIT,"science_blob":SCIENCE_BLOB,"condition":condition,"library_versions":{},"runtime_terminal_marker":"CMI_FLU_V307_RUNTIME_PASS"},
    }
    return result


def exercise_science_helpers(runtime, condition: str) -> None:
    with tempfile.TemporaryDirectory(prefix="v307-synthetic-package-") as tmp:
        package = Path(tmp) / "cmi_flu_bundle.zip"; package.write_bytes(runtime.package_bytes()); sys.path.insert(0, str(package))
        try:
            v307, _hai = runtime.load_v307_module()
            # Subject-purge inner split contract.
            frame = pd.DataFrame({"study_group":["A"]*6+["B"]*6+["C"]*6,
                                  "subject_group":[f"A{i}" for i in range(6)]+[f"B{i}" for i in range(6)]+[f"C{i}" for i in range(6)]})
            splits = v307._nested_splits(frame)
            assert len(splits) == 3
            for split in splits:
                tr=set(frame.iloc[split.train_indices].subject_group); va=set(frame.iloc[split.validation_indices].subject_group); assert not tr.intersection(va)
            if condition == "task22_panel_mean":
                cal = pd.DataFrame({"log2_pre_panel_gm":np.linspace(4,7,18),"pre_panel_log2_sd":np.linspace(.1,.8,18),"base_prediction":np.linspace(20,80,18)})
                cal["actual"] = cal["base_prediction"] + 2.0*cal["log2_pre_panel_gm"] - 1.5*cal["pre_panel_log2_sd"]
                pred = pd.DataFrame({"log2_pre_panel_gm":np.linspace(4.2,6.8,7),"pre_panel_log2_sd":np.linspace(.15,.7,7),"base_prediction":np.linspace(25,70,7)})
                counter=v307._FitCounter(limit=256); out, contract=v307._fit_panel_residual(cal,pred,counter)
                assert counter.count == 1 and np.all(out > 0) and contract["features"] == ["log2_pre_panel_gm","pre_panel_log2_sd"] and contract["correction_sd"] > 0
            else:
                n=24
                train=pd.DataFrame({"log2_pre_hai":np.linspace(3,7,n),"age":np.linspace(20,70,n),"predicted_d28":np.linspace(20,120,n),"retention_target":np.linspace(-1,.2,n)})
                pred=pd.DataFrame({"log2_pre_hai":np.linspace(3.2,6.8,8),"age":np.linspace(25,65,8),"predicted_log2_d28":np.linspace(4.5,6.5,8)})
                counter=v307._FitCounter(limit=256); out, contract=v307._fit_retention(train,pred,counter)
                assert counter.count == 1 and len(out) == 8 and contract["observed_d28_feature"] is False
        finally:
            sys.path.remove(str(package))


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime",type=Path,required=True)
    p.add_argument("--condition",choices=["task22_panel_mean","task23_retention"],required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args(); runtime=load_runtime(args.runtime.resolve())
    assert runtime.V307_CONDITION == args.condition
    exercise_science_helpers(runtime,args.condition)
    aggregate=synthetic_aggregate(runtime,args.condition); runtime.validate_result(aggregate); assert "participant_id" not in json.dumps(aggregate)
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    if args.condition == "task22_panel_mean":
        oof=pd.DataFrame({"participant_id":["p1","p2"],"subject_group":["s1","s2"],"correction_sd":[1.25,1.25],"prediction":[1.0,2.0]})
    else:
        oof=pd.DataFrame({"participant_id":["p1","p2"],"subject_group":["s1","s2"],"prediction":[1.0,2.0]})
    challenge=pd.DataFrame({"participant_id":["c1","c2"],"prediction":[1.5,2.5]})
    with tempfile.TemporaryDirectory(prefix="v307-writer-") as tmp:
        stage=Path(tmp); v307=sys.modules.get("cmi_flu.strategy_v3_v07")
        if v307 is None:
            package=Path(tmp)/"pkg.zip"; package.write_bytes(runtime.package_bytes()); sys.path.insert(0,str(package))
            try: v307,_=runtime.load_v307_module()
            finally: sys.path.remove(str(package))
        manifest=v307.write_v3_07_outputs(args.condition,aggregate,oof,challenge,stage)
        assert manifest["row_level_banks_private"] is True
        for path in stage.iterdir():
            if path.is_file(): (out/path.name).write_bytes(path.read_bytes())
    # Failure cleanup: empty Competition tree must fail before publishing final files.
    with tempfile.TemporaryDirectory(prefix="v307-failure-input-") as tmpin, tempfile.TemporaryDirectory(prefix="v307-failure-out-parent-") as tmpout:
        failure_out=Path(tmpout)/"final"
        code=runtime.execute(Path(tmpin),failure_out)
        assert code != 0
        assert not failure_out.exists() or not any(failure_out.iterdir())
    digest=hashlib.sha256(args.runtime.read_bytes()).hexdigest()
    print(f"CMI_FLU_V307_SYNTHETIC PASS condition={args.condition} helper_science=true writer=true manifest=true failure_cleanup=true fit_budget=true runtime_sha256={digest} submission=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
