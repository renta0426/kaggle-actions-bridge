#!/usr/bin/env python3
"""Build exact E11a runtime on the proven E10-003 frozen-bundle ancestry."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REQUEST_ID = "20260910-cmi-flu-strategy-e11a-task11-pairwise-001"
TARGET_KERNEL = "renta0426/cmi-flu-e11a-task11-pairwise-20260910-001"
SCIENCE_COMMIT = "4d8c568e3b26ad6ba48c076bcb28fda64a0dc72f"
E11_BLOB = "ec7e709ac83f9792b6cc4d63ad398a8a68e16e7b"
E11_SYNTH_BLOB = "ff5c7cbc5aea7ebdbb9e16853d26ec8179cf2d6c"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e11a-task11-pairwise-001"
E11_PATH = f"{PAYLOAD_ROOT}/strategy_e11.py"
E11_SYNTH_PATH = f"{PAYLOAD_ROOT}/strategy_e11_synthetic.py"
REQUEST_PATH = "requests/cmi-flu-strategy-e11a-task11-pairwise-001.json"
BASE_PREPARE = "scripts/cmi_flu_strategy_e10_validator_repair_prepare.py"

OLD_REQUEST = "20260910-cmi-flu-strategy-e10-pooled-domain-correction-003"
OLD_TARGET = "renta0426/cmi-flu-e10-pooled-domain-correction-20260910-003"
OLD_SCIENCE = "3a8728879179115487f5ce0ba4a4a0467e1910bb"


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def require_text(root: Path, path: str, sha: str) -> str:
    data=(root/path).read_bytes()
    found=git_blob_sha(data)
    if found!=sha:
        raise SystemExit(f"E11a relay mismatch:{path}:{found}")
    text=data.decode()
    compile(text,path,"exec")
    if "kaggle competitions submit" in text or "competition_submit" in text:
        raise SystemExit("E11a relayed source contains submission path")
    return text


def validate_request(root: Path) -> None:
    r=json.loads((root/REQUEST_PATH).read_text())
    exact={
        "schema_version":1,
        "request_id":REQUEST_ID,
        "competition":"cmi-flu-first-prediction-challenge",
        "operation":"kernel_run_and_current_output_read",
        "target":TARGET_KERNEL,
        "science_repository":"renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit":SCIENCE_COMMIT,
        "science_transport":"agent_relay_exact_blobs_plus_proven_e10_validator_runtime",
        "strategy_e11_blob_sha":E11_BLOB,
        "strategy_e11_synthetic_blob_sha":E11_SYNTH_BLOB,
        "expected_kernel_version":1,
        "enable_internet":False,
        "competition_submission_attempted":False,
        "leaderboard_used_for_selection":False,
        "automatic_compute_retries":0,
    }
    for k,v in exact.items():
        if r.get(k)!=v: raise SystemExit(f"E11a request mismatch:{k}")
    if r.get("dependency_blobs")!={"baseline_b021_robust.yaml":CONFIG_BLOB}:
        raise SystemExit("E11a dependency blob mismatch")
    if r.get("resource")!={"accelerator":"cpu","expected_runtime_minutes":15,"hard_timeout_minutes":45,"max_active_runs":1}:
        raise SystemExit("E11a resource contract mismatch")
    if r.get("api_budget")!={"max_calls":70,"poll_interval_seconds":120,"max_polls":25,"max_pages":2}:
        raise SystemExit("E11a API budget mismatch")
    if r.get("side_effects")!=["create exactly one fresh private CPU Notebook version and read only its current aggregate outputs"]:
        raise SystemExit("E11a side effects mismatch")
    if r.get("allowed_output_paths")!=["bridge-result.json","metrics.json","summary.md"]:
        raise SystemExit("E11a output allowlist mismatch")
    c=r.get("experiment_contract") or {}
    locked={
        "experiment":"strategy_v2_e11a_task11_pairwise_ranker",
        "task":"Task1.1",
        "comparison_contract":"paired_subject_purged_v2",
        "base_model_set":"task_11",
        "base_model":"pls_2",
        "expected_train_rows":127,
        "expected_train_studies":4,
        "expected_challenge_rows":40,
        "pair_policy":"within_source_study_only_symmetric",
        "pair_tie_tolerance":1e-12,
        "max_rows_per_study":80,
        "max_transformed_features":512,
        "minimum_directed_pairs":100,
        "reference_aggregation":"study_equal",
        "classifier":"HistGradientBoostingClassifier",
        "classifier_params":{"learning_rate":0.05,"max_iter":150,"max_leaf_nodes":15,"min_samples_leaf":10,"l2_regularization":1.0,"early_stopping":False,"random_state":20260910},
        "cross_study_pair_labels_used":False,
        "held_outcomes_used_for_fit":False,
        "held_outcomes_used_for_scoring":False,
        "promotion_mean_delta":0.02,
        "promotion_minimum_study_delta":-0.1,
        "promotion_requires_strict_majority_wins":True,
        "public_probe_authorized":False,
        "automatic_incumbent_change_authorized":False,
    }
    if c!=locked: raise SystemExit("E11a experiment contract mismatch")


def replace_function(text: str, name: str, replacement: str) -> str:
    tree=ast.parse(text)
    matches=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    if len(matches)!=1: raise SystemExit(f"E11a function binding changed:{name}:{len(matches)}")
    node=matches[0]
    lines=text.splitlines(keepends=True)
    return "".join(lines[:node.lineno-1])+replacement.rstrip()+"\n"+"".join(lines[node.end_lineno:])


def patch_runtime(runtime: str, e11: str, synth: str) -> str:
    for old,new in ((OLD_REQUEST,REQUEST_ID),(OLD_TARGET,TARGET_KERNEL),(OLD_SCIENCE,SCIENCE_COMMIT)):
        if old not in runtime: raise SystemExit(f"E11a identity anchor missing:{old}")
        runtime=runtime.replace(old,new)

    marker="def load_e10_modules() -> tuple[object, object, object]:\n"
    if runtime.count(marker)!=1: raise SystemExit("E11a module injection anchor changed")
    injected=f'''E11_BLOB = "{E11_BLOB}"
E11_SYNTH_BLOB = "{E11_SYNTH_BLOB}"
E11_SOURCE = {e11!r}
E11_SYNTH_SOURCE = {synth!r}

def load_e11_modules() -> tuple[object, object, object]:
    load_e05_module()
    from cmi_flu import models as _e11_models
    if not hasattr(_e11_models, "summarize_metric_frame"):
        def _e11_summarize_metric_frame(frame):
            if frame.empty:
                return {{"count":0,"spearman_mean":None,"spearman_median":None,"spearman_min":None,"spearman_max":None,"spearman_std":None,"rmse_mean":None,"rmse_median":None,"rmse_max":None,"rmse_std":None}}
            spearman=pd.to_numeric(frame["spearman"],errors="coerce").to_numpy(dtype=float)
            rmse=pd.to_numeric(frame["rmse"],errors="coerce").to_numpy(dtype=float)
            spearman=spearman[np.isfinite(spearman)]; rmse=rmse[np.isfinite(rmse)]
            return {{"count":int(spearman.size),"spearman_mean":float(np.mean(spearman)) if spearman.size else None,"spearman_median":float(np.median(spearman)) if spearman.size else None,"spearman_min":float(np.min(spearman)) if spearman.size else None,"spearman_max":float(np.max(spearman)) if spearman.size else None,"spearman_std":float(np.std(spearman)) if spearman.size else None,"rmse_mean":float(np.mean(rmse)) if rmse.size else None,"rmse_median":float(np.median(rmse)) if rmse.size else None,"rmse_max":float(np.max(rmse)) if rmse.size else None,"rmse_std":float(np.std(rmse)) if rmse.size else None}}
        _e11_models.summarize_metric_frame=_e11_summarize_metric_frame
    def install(name, source):
        module=types.ModuleType(name); module.__file__=f"<{{name}}>"; module.__package__="cmi_flu"; sys.modules[name]=module
        exec(compile(source,name.replace(".","/")+".py","exec"),module.__dict__,module.__dict__)
        return module
    e11=install("cmi_flu.strategy_e11",E11_SOURCE)
    synthetic=install("cmi_flu.strategy_e11_synthetic",E11_SYNTH_SOURCE)
    run=getattr(e11,"run_e11a",None); run_synthetic=getattr(synthetic,"run_synthetic",None)
    if not callable(run) or not callable(run_synthetic):
        raise BridgeContractError("e11a_entry_missing")
    return run,run_synthetic,e11

'''
    runtime=runtime.replace(marker,injected+marker,1)

    validate=r'''def validate_result(result: dict, *, synthetic: bool = False) -> None:
    params={"learning_rate":0.05,"max_iter":150,"max_leaf_nodes":15,"min_samples_leaf":10,"l2_regularization":1.0,"early_stopping":False,"random_state":20260910}
    if synthetic:
        if int(result.get("schema_version",-1))!=1 or result.get("experiment")!="synthetic_e11a_pairwise_contract":
            raise BridgeContractError("e11a_synthetic_identity")
        if result.get("study_counts")!={"SDY180":34,"SDY515":16,"SDY519":17,"SDY56":60}:
            raise BridgeContractError("e11a_synthetic_counts")
        if result.get("feasibility_ready") is not True or int(result.get("outer_split_count",-1))!=4 or int(result.get("subject_overlap_max",-1))!=0:
            raise BridgeContractError("e11a_synthetic_feasibility")
        if result.get("pair_policy") is not True or result.get("cross_study_pair_labels_used") is not False:
            raise BridgeContractError("e11a_synthetic_pair_boundary")
        if int(result.get("positive_pairs",-1))!=int(result.get("negative_pairs",-2)) or int(result.get("directed_pairs",0))<100:
            raise BridgeContractError("e11a_synthetic_pair_balance")
        if float(result.get("antisymmetric_design_max_error",99))>1e-12 or float(result.get("held_spearman",-9))<0.80:
            raise BridgeContractError("e11a_synthetic_rank_contract")
        if float(result.get("fit_and_score_seconds",999))>float(result.get("cpu_feasibility_seconds",0)):
            raise BridgeContractError("e11a_synthetic_cpu_budget")
        if int(result.get("transformed_features",9999))>512 or result.get("classifier_params")!=params:
            raise BridgeContractError("e11a_synthetic_model_contract")
        if result.get("held_outcomes_used_for_fit") is not False or result.get("held_outcomes_used_for_scoring") is not False:
            raise BridgeContractError("e11a_synthetic_leakage")
        if result.get("competition_submission_attempted") is not False or result.get("leaderboard_used_for_selection") is not False:
            raise BridgeContractError("e11a_synthetic_execution_boundary")
        return

    if int(result.get("schema_version",-1))!=1 or result.get("experiment")!="strategy_v2_e11a_task11_pairwise_ranker":
        raise BridgeContractError("e11a_identity")
    if result.get("comparison_contract")!="paired_subject_purged_v2":
        raise BridgeContractError("e11a_comparison_contract")
    for key in ("cross_study_pair_labels_used","held_outcomes_used_for_fit","held_outcomes_used_for_scoring","leaderboard_used_for_selection","competition_submission_attempted","incumbent_changed","public_probe_authorized"):
        if result.get(key) is not False: raise BridgeContractError(f"e11a_boundary:{key}")
    if int(result.get("automatic_compute_retries",-1))!=0: raise BridgeContractError("e11a_retry")
    frozen=result.get("frozen_conditions") or {}
    expected={"task":"Task1.1","base_model":"pls_2","pair_policy":"within_source_study_only_symmetric","pair_tie_tolerance":1e-12,"max_rows_per_study":80,"max_transformed_features":512,"minimum_directed_pairs":100,"reference_aggregation":"study_equal","classifier":"HistGradientBoostingClassifier","classifier_params":params,"promotion_mean_delta":0.02,"promotion_minimum_study_delta":-0.10,"promotion_requires_strict_majority_wins":True}
    if frozen!=expected: raise BridgeContractError("e11a_frozen_contract")
    task=result.get("task") or {}
    if task.get("task")!="Task1.1" or task.get("base_contract")!={"kind":"b21","model":"pls_2","current_competition_incumbent":True}:
        raise BridgeContractError("e11a_task_base")
    folds=task.get("folds") or []
    if len(folds)!=4: raise BridgeContractError("e11a_fold_count")
    if {str(f.get("held_study")) for f in folds}!={"SDY180","SDY515","SDY519","SDY56"}:
        raise BridgeContractError("e11a_held_studies")
    deltas=[]
    for fold in folds:
        if int(fold.get("n",0))<3: raise BridgeContractError("e11a_fold_support")
        for key in ("base_spearman","pairwise_spearman","pairwise_delta"):
            if not math.isfinite(float(fold.get(key,float("nan")))): raise BridgeContractError(f"e11a_nonfinite:{key}")
        if fold.get("held_outcomes_used_for_fit") is not False or fold.get("held_outcomes_used_for_scoring") is not False:
            raise BridgeContractError("e11a_fold_leakage")
        audit=fold.get("fit_audit") or {}
        if audit.get("pairs_within_study_only") is not True or audit.get("cross_study_pair_labels_used") is not False:
            raise BridgeContractError("e11a_pair_boundary")
        if int(audit.get("positive_pairs",-1))!=int(audit.get("negative_pairs",-2)) or int(audit.get("directed_pairs",0))<100:
            raise BridgeContractError("e11a_pair_balance")
        if int(audit.get("training_studies",-1))!=3 or int(audit.get("transformed_features",9999))>512:
            raise BridgeContractError("e11a_fit_shape")
        if audit.get("reference_aggregation")!="study_equal" or audit.get("held_outcomes_used_for_fit") is not False:
            raise BridgeContractError("e11a_fit_boundary")
        if audit.get("classifier")!="HistGradientBoostingClassifier" or audit.get("classifier_params")!=params:
            raise BridgeContractError("e11a_classifier")
        deltas.append(float(fold["pairwise_delta"]))
    promotion=(task.get("candidate") or {}).get("promotion") or {}
    wins=sum(v>0 for v in deltas); required=len(deltas)//2+1
    mean=sum(deltas)/len(deltas); minimum=min(deltas)
    passed=bool(mean>=0.02 and minimum>=-0.10 and wins>=required)
    if promotion.get("passed") is not passed or abs(float(promotion.get("mean_delta"))-mean)>1e-12 or abs(float(promotion.get("minimum_delta"))-minimum)>1e-12:
        raise BridgeContractError("e11a_promotion")
    if int(promotion.get("wins",-1))!=wins or int(promotion.get("required_wins",-1))!=required:
        raise BridgeContractError("e11a_promotion_wins")
    candidate=task.get("candidate") or {}
    if candidate.get("name")!="pairwise_hgb": raise BridgeContractError("e11a_candidate")
    agreement=candidate.get("challenge_agreement_vs_base") or {}
    if not math.isfinite(float((agreement.get("rank_spearman") or {}).get("value",float("nan")))) or not (0<=int(agreement.get("changed_rank_count",-1))<=40):
        raise BridgeContractError("e11a_challenge_agreement")
    fit=candidate.get("fit") or {}
    if int(fit.get("training_rows",-1))!=127 or int(fit.get("training_studies",-1))!=4 or fit.get("pairs_within_study_only") is not True or fit.get("cross_study_pair_labels_used") is not False:
        raise BridgeContractError("e11a_final_fit")
    if int(fit.get("positive_pairs",-1))!=int(fit.get("negative_pairs",-2)) or fit.get("classifier_params")!=params:
        raise BridgeContractError("e11a_final_pair_contract")
    feasibility=task.get("feasibility") or {}
    if feasibility.get("study_counts")!={"SDY180":34,"SDY515":16,"SDY519":17,"SDY56":60} or len(feasibility.get("outer_splits") or [])!=4:
        raise BridgeContractError("e11a_feasibility")
    if any(int(s.get("subject_overlap",-1))!=0 for s in feasibility["outer_splits"]):
        raise BridgeContractError("e11a_subject_purge")
    selected=task.get("selected_local_candidate")
    if (selected=="pairwise_hgb") is not passed: raise BridgeContractError("e11a_selection")
    if task.get("competition_candidate") is not False or task.get("public_probe_authorized") is not False or int(task.get("challenge_rows",-1))!=40:
        raise BridgeContractError("e11a_competition_boundary")
    serialized=json.dumps(result,sort_keys=True,ensure_ascii=False)
    banned=('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"','ROW_','SUB_')
    if any(token in serialized for token in banned): raise BridgeContractError("e11a_aggregate_privacy")
'''
    runtime=replace_function(runtime,"validate_result",validate)

    summary=r'''def render_summary(result: dict) -> str:
    task=result.get("task") or {}; candidate=task.get("candidate") or {}; p=candidate.get("promotion") or {}; agreement=candidate.get("challenge_agreement_vs_base") or {}; fit=candidate.get("fit") or {}
    lines=["# CMI-Flu strategy E11a Task1.1 pairwise ranker","",
        "One fixed within-study pairwise-HGB comparison against B2.1 PLS-2; aggregate-only; no submission/Public probe.","",
        f"- science commit: `{SCIENCE_COMMIT}`",f"- E11 blob: `{E11_BLOB}`",
        f"- promotion: `{p.get('passed')}`; mean delta `{p.get('mean_delta')}`; minimum `{p.get('minimum_delta')}`; wins `{p.get('wins')}/{p.get('required_wins')}`",
        f"- Challenge rank agreement: `{(agreement.get('rank_spearman') or {}).get('value')}`; changed ranks `{agreement.get('changed_rank_count')}`",
        f"- final directed pairs: `{fit.get('directed_pairs')}`; transformed features `{fit.get('transformed_features')}`; fit seconds `{fit.get('fit_seconds')}`",""]
    for fold in task.get("folds") or []:
        lines.append(f"- {fold.get('held_study')}: n=`{fold.get('n')}` base=`{fold.get('base_spearman')}` pairwise=`{fold.get('pairwise_spearman')}` delta=`{fold.get('pairwise_delta')}`")
    return "\n".join(lines)+"\n"
'''
    runtime=replace_function(runtime,"render_summary",summary)

    selftest=f'''def self_test() -> int:
    package=package_bytes()
    for source,expected,label in ((E11_SOURCE,E11_BLOB,"e11"),(E11_SYNTH_SOURCE,E11_SYNTH_BLOB,"e11_synthetic")):
        if git_blob_sha(source.encode("utf-8"))!=expected: raise BridgeContractError(f"{{label}}_blob_mismatch")
        compile(source,f"cmi_flu/{{label}}.py","exec")
    print(f"CMI_FLU_E11A_RUNTIME_SELF_TEST PASS request_id={{REQUEST_ID}} package_bytes={{len(package)}} science_commit={{SCIENCE_COMMIT}} e11_blob={{E11_BLOB}}")
    return 0'''
    runtime=replace_function(runtime,"self_test",selftest)

    main=r'''def main() -> int:
    args=parse_args()
    if args.self_test: return self_test()
    if args.synthetic:
        output_dir=args.output_dir.expanduser().resolve(); output_dir.mkdir(parents=True,exist_ok=False)
        runtime_root=Path("/tmp")/"cmi-flu-e11a-synthetic-runtime"
        if runtime_root.exists(): shutil.rmtree(runtime_root)
        runtime_root.mkdir(parents=True)
        package_path=runtime_root/"cmi_flu_bundle.zip"; package_path.write_bytes(package_bytes()); sys.path.insert(0,str(package_path))
        try:
            run_e11a,run_synthetic,e11=load_e11_modules()
            result=json_safe(dict(run_synthetic()))
            validate_result(result,synthetic=True)
            metrics_path=output_dir/"metrics.json"; summary_path=output_dir/"summary.md"; bridge_path=output_dir/"bridge-result.json"
            metrics_path.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")
            summary_path.write_text("# CMI-Flu E11a synthetic contract\n\nCompetition-Data-free CPU/leakage validation; no submission.\n",encoding="utf-8")
            bridge={"schema_version":1,"request_id":REQUEST_ID,"competition":COMPETITION,"target_kernel":TARGET_KERNEL,"science_commit":SCIENCE_COMMIT,"strategy_e11_blob_sha":E11_BLOB,"strategy_e11_synthetic_blob_sha":E11_SYNTH_BLOB,"metrics_sha256":sha256_file(metrics_path),"summary_sha256":sha256_file(summary_path),"competition_submission_attempted":False,"leaderboard_used_for_selection":False,"contains_participant_identifiers":False,"contains_row_level_predictions":False,"synthetic":True}
            bridge_path.write_text(json.dumps(bridge,indent=2,sort_keys=True)+"\n",encoding="utf-8")
            print("CMI_FLU_E11A_COMPLETE synthetic=true submission=false")
            return 0
        finally:
            shutil.rmtree(runtime_root,ignore_errors=True)
    try:
        input_dir=locate_competition_data(args.input_dir)
    except Exception as exc:
        code=hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode()).hexdigest()[:20]
        print(f"CMI_FLU_E11A_FAILED stage=locate_competition_data exception_type={type(exc).__name__} error_code={code}",file=sys.stderr)
        return 2
    return execute(input_dir,args.output_dir)
'''
    runtime=replace_function(runtime,"main",main)

    old='''        stage = "load_e10"
        run_e10, run_synthetic, e10_module = load_e10_modules()
        stage = "run_e10"
        result = json_safe(dict(run_e10(config, inputs)))
        stage = "validate_e10"
        validate_result(result, synthetic=False)
'''
    new='''        stage = "load_e11a"
        run_e11a, run_synthetic, e11_module = load_e11_modules()
        stage = "run_e11a"
        result = json_safe(dict(run_e11a(config, inputs)))
        stage = "validate_e11a"
        validate_result(result, synthetic=False)
'''
    if runtime.count(old)!=1: raise SystemExit("E11a execute-stage anchor changed")
    runtime=runtime.replace(old,new,1)

    prov='            "strategy_e10_synthetic_blob_sha": E10_SYNTH_BLOB,\n'
    if runtime.count(prov)!=1: raise SystemExit("E11a bridge provenance anchor changed")
    runtime=runtime.replace(prov,prov+'            "strategy_e11_blob_sha": E11_BLOB,\n            "strategy_e11_synthetic_blob_sha": E11_SYNTH_BLOB,\n',1)
    runtime=runtime.replace("CMI-Flu strategy E10 pooled domain correction. Aggregate outputs only; no submission.","CMI-Flu strategy E11a Task1.1 pairwise ranker. Aggregate outputs only; no submission.")
    runtime=runtime.replace("CMI_FLU_E10_FAILED","CMI_FLU_E11A_FAILED").replace("CMI_FLU_E10_COMPLETE","CMI_FLU_E11A_COMPLETE")
    if "kaggle competitions submit" in runtime or "competition_submit" in runtime:
        raise SystemExit("E11a generated runtime contains submission path")
    compile(runtime,"generated_e11a_runtime.py","exec")
    return runtime


def main() -> int:
    args=parse_args(); root=args.repository_root.resolve(); refs=args.reference_dir.resolve()
    validate_request(root)
    e11=require_text(root,E11_PATH,E11_BLOB); synth=require_text(root,E11_SYNTH_PATH,E11_SYNTH_BLOB)
    with tempfile.TemporaryDirectory(prefix="cmi-flu-e11a-base-") as tmp:
        base=Path(tmp)/"e10_runtime.py"
        subprocess.run([sys.executable,str(root/BASE_PREPARE),"--repository-root",str(root),"--reference-dir",str(refs),"--output",str(base)],check=True)
        runtime=patch_runtime(base.read_text(),e11,synth)
    out=args.output.resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(runtime)
    subprocess.run([sys.executable,str(out),"--self-test"],check=True)
    print(f"CMI_FLU_E11A_PREPARE_PASS science_commit={SCIENCE_COMMIT} e11_blob={E11_BLOB} target={TARGET_KERNEL} runtime_sha256={hashlib.sha256(runtime.encode()).hexdigest()} submission=false")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
