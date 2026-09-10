#!/usr/bin/env python3
"""Build the exact E11b TabPFN runtime on proven E11a transport ancestry."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REQUEST_ID = "20260910-cmi-flu-strategy-e11b-tabpfn3-001"
TARGET_KERNEL = "renta0426/cmi-flu-e11b-tabpfn3-20260910-001"
SCIENCE_COMMIT = "95c692bf2e27f6d07ea31a45fd788293666ea94a"
E11B_BLOB = "8a291ae4952786bf16ee667caa5557599310e3fd"
E11B_SYNTH_BLOB = "f25d04d75db584db78d26a232ab84dc5ab1bbf7d"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e11b-tabpfn3-001"
E11B_PATH = f"{PAYLOAD_ROOT}/strategy_e11b.py"
E11B_SYNTH_PATH = f"{PAYLOAD_ROOT}/strategy_e11b_synthetic.py"
REQUEST_PATH = "requests/cmi-flu-strategy-e11b-tabpfn3-001.json"
BASE_PREPARE = "scripts/cmi_flu_strategy_e11a_prepare.py"

OLD_REQUEST = "20260910-cmi-flu-strategy-e11a-task11-pairwise-001"
OLD_TARGET = "renta0426/cmi-flu-e11a-task11-pairwise-20260910-001"
OLD_SCIENCE = "4d8c568e3b26ad6ba48c076bcb28fda64a0dc72f"

TABPFN_PACKAGE_VERSION = "8.5.0"
TABPFN_SOURCE_COMMIT = "9ed44abd5882140b88c9f2816c5791987ce059b9"
TABPFN_WHEEL_FILENAME = "tabpfn-8.5.0-py3-none-any.whl"
TABPFN_WHEEL_SHA256 = "4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0"
MODEL_SOURCE = "prior-labsai/tabpfn-3/pytorch/default/1"
CHECKPOINT_FILENAME = "tabpfn-v3-regressor-v3_default.ckpt"
CHECKPOINT_BYTES = 233_289_807
CHECKPOINT_SHA256 = "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301"
MODEL_MOUNT = "/kaggle/input/models/prior-labsai/tabpfn-3/pytorch/default/1"
TABPFN_PARAMS = {
    "n_estimators": 8,
    "device": "cpu",
    "fit_mode": "fit_preprocessors",
    "memory_saving_mode": "auto",
    "random_state": 20260910,
    "n_preprocessing_jobs": 1,
    "show_progress_bar": False,
    "ignore_pretraining_limits": False,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--tabpfn-wheel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def require_text(root: Path, path: str, sha: str) -> str:
    data = (root / path).read_bytes()
    found = git_blob_sha(data)
    if found != sha:
        raise SystemExit(f"E11b relay mismatch:{path}:{found}")
    text = data.decode("utf-8")
    compile(text, path, "exec")
    if "kaggle competitions submit" in text or "competition_submit" in text:
        raise SystemExit("E11b relayed source contains submission path")
    return text


def require_wheel(path: Path) -> bytes:
    path = path.resolve()
    if path.name != TABPFN_WHEEL_FILENAME or not path.is_file():
        raise SystemExit("E11b wheel filename/path contract failed")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != TABPFN_WHEEL_SHA256:
        raise SystemExit(f"E11b wheel SHA-256 mismatch:{digest}")
    if not 100_000 < len(data) < 5_000_000:
        raise SystemExit("E11b wheel byte contract failed")
    return data


def validate_request(root: Path) -> None:
    r = json.loads((root / REQUEST_PATH).read_text())
    exact = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_repository": "renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": "agent_relay_exact_blobs_plus_proven_e11a_transport_runtime",
        "strategy_e11b_blob_sha": E11B_BLOB,
        "strategy_e11b_synthetic_blob_sha": E11B_SYNTH_BLOB,
        "expected_kernel_version": 1,
        "enable_internet": False,
        "automatic_compute_retries": 0,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "publish_only_sanitized_aggregate": True,
    }
    for key, value in exact.items():
        if r.get(key) != value:
            raise SystemExit(f"E11b request mismatch:{key}")
    if r.get("dependency_blobs") != {"baseline_b021_robust.yaml": CONFIG_BLOB}:
        raise SystemExit("E11b dependency blob mismatch")
    if r.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 30,
        "hard_timeout_minutes": 60,
        "max_active_runs": 1,
    }:
        raise SystemExit("E11b resource contract mismatch")
    if r.get("api_budget") != {
        "max_calls": 80,
        "poll_interval_seconds": 120,
        "max_polls": 35,
        "max_pages": 2,
    }:
        raise SystemExit("E11b API budget mismatch")
    if r.get("allowed_output_paths") != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E11b output allowlist mismatch")
    preflight = r.get("model_access_preflight") or {}
    if preflight != {
        "required_before_kernel_write": True,
        "official_kaggle_model_source": MODEL_SOURCE,
        "failure_classification": "competition_input_model_access_staging",
        "write_on_failure": False,
    }:
        raise SystemExit("E11b model access preflight mismatch")
    external = r.get("external_model") or {}
    if external != {
        "package": "tabpfn",
        "package_version": TABPFN_PACKAGE_VERSION,
        "source_commit": TABPFN_SOURCE_COMMIT,
        "wheel_filename": TABPFN_WHEEL_FILENAME,
        "wheel_sha256": TABPFN_WHEEL_SHA256,
        "kaggle_model_source": MODEL_SOURCE,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_license": "tabpfn-3-license-v1.0",
        "project_redistributes_weights": False,
    }:
        raise SystemExit("E11b external model contract mismatch")
    c = r.get("experiment_contract") or {}
    locked = {
        "experiment": "strategy_v2_e11b_task11_tabpfn3",
        "task": "Task1.1",
        "comparison_contract": "paired_subject_purged_v2",
        "base_model_set": "task_11",
        "base_model": "pls_2",
        "target_transform": "log",
        "feature_space": "unchanged_compact_b21_task11",
        "max_raw_features": 200,
        "expected_train_rows": 127,
        "expected_train_studies": 4,
        "expected_challenge_rows": 40,
        "candidate": "tabpfn3_default_cpu",
        "tabpfn_params": TABPFN_PARAMS,
        "held_outcomes_used_for_fit": False,
        "held_outcomes_used_for_scoring": False,
        "promotion_mean_delta": 0.02,
        "promotion_minimum_study_delta": -0.1,
        "promotion_requires_strict_majority_wins": True,
        "public_probe_authorized": False,
        "automatic_incumbent_change_authorized": False,
    }
    if c != locked:
        raise SystemExit("E11b experiment contract mismatch")
    if r.get("side_effects") != [
        "after read-only model access preflight, create exactly one fresh private CPU Notebook version with one official Kaggle Model input and read only its current aggregate outputs"
    ]:
        raise SystemExit("E11b side effects mismatch")


def replace_function(text: str, name: str, replacement: str) -> str:
    tree = ast.parse(text)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise SystemExit(f"E11b function binding changed:{name}:{len(matches)}")
    node = matches[0]
    lines = text.splitlines(keepends=True)
    return "".join(lines[: node.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno :])


def patch_runtime(runtime: str, e11b: str, synth: str, wheel: bytes) -> str:
    for old, new in ((OLD_REQUEST, REQUEST_ID), (OLD_TARGET, TARGET_KERNEL), (OLD_SCIENCE, SCIENCE_COMMIT)):
        if old not in runtime:
            raise SystemExit(f"E11b identity anchor missing:{old}")
        runtime = runtime.replace(old, new)

    marker = "def load_e10_modules() -> tuple[object, object, object]:\n"
    if runtime.count(marker) != 1:
        raise SystemExit("E11b module injection anchor changed")
    wheel_b64 = base64.b64encode(wheel).decode("ascii")
    injected = f'''E11B_BLOB = "{E11B_BLOB}"
E11B_SYNTH_BLOB = "{E11B_SYNTH_BLOB}"
E11B_SOURCE = {e11b!r}
E11B_SYNTH_SOURCE = {synth!r}
TABPFN_PACKAGE_VERSION = "{TABPFN_PACKAGE_VERSION}"
TABPFN_SOURCE_COMMIT = "{TABPFN_SOURCE_COMMIT}"
TABPFN_WHEEL_FILENAME = "{TABPFN_WHEEL_FILENAME}"
TABPFN_WHEEL_SHA256 = "{TABPFN_WHEEL_SHA256}"
TABPFN_WHEEL_B64 = {wheel_b64!r}
TABPFN_MODEL_SOURCE = "{MODEL_SOURCE}"
TABPFN_CHECKPOINT_FILENAME = "{CHECKPOINT_FILENAME}"
TABPFN_CHECKPOINT_BYTES = {CHECKPOINT_BYTES}
TABPFN_CHECKPOINT_SHA256 = "{CHECKPOINT_SHA256}"
TABPFN_MODEL_MOUNT = Path("{MODEL_MOUNT}")

def load_e11b_modules() -> tuple[object, object, object]:
    load_e05_module()
    def install(name, source):
        module=types.ModuleType(name); module.__file__=f"<{{name}}>"; module.__package__="cmi_flu"; sys.modules[name]=module
        exec(compile(source,name.replace(".","/")+".py","exec"),module.__dict__,module.__dict__)
        return module
    e11b=install("cmi_flu.strategy_e11b",E11B_SOURCE)
    synthetic=install("cmi_flu.strategy_e11b_synthetic",E11B_SYNTH_SOURCE)
    run=getattr(e11b,"run_e11b",None); run_synthetic=getattr(synthetic,"run_synthetic",None)
    if not callable(run) or not callable(run_synthetic):
        raise BridgeContractError("e11b_entry_missing")
    return run,run_synthetic,e11b

def install_tabpfn_wheel() -> dict:
    import base64 as _b64
    import importlib.metadata as _metadata
    root=Path("/tmp")/"cmi-flu-e11b-tabpfn-site"
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    wheel_path=root/TABPFN_WHEEL_FILENAME
    wheel_bytes=_b64.b64decode(TABPFN_WHEEL_B64.encode("ascii"),validate=True)
    if hashlib.sha256(wheel_bytes).hexdigest()!=TABPFN_WHEEL_SHA256:
        raise BridgeContractError("e11b_embedded_wheel_sha")
    wheel_path.write_bytes(wheel_bytes)
    site=root/"site"; site.mkdir()
    completed=subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check","--no-input","--no-deps","--target",str(site),str(wheel_path)],capture_output=True,text=True,timeout=180,check=False)
    if completed.returncode!=0:
        code=hashlib.sha256((completed.stdout+completed.stderr).encode("utf-8",errors="replace")).hexdigest()[:20]
        raise BridgeContractError(f"e11b_tabpfn_install_failed:{{code}}")
    sys.path.insert(0,str(site))
    found=_metadata.version("tabpfn")
    if found!=TABPFN_PACKAGE_VERSION:
        raise BridgeContractError("e11b_tabpfn_version")
    return {{"version":found,"wheel_sha256":TABPFN_WHEEL_SHA256,"site":str(site)}}

def locate_tabpfn_checkpoint() -> Path:
    root=TABPFN_MODEL_MOUNT
    if not root.is_dir():
        raise BridgeContractError("e11b_model_mount_missing")
    matches=[p for p in root.rglob(TABPFN_CHECKPOINT_FILENAME) if p.is_file() and not p.is_symlink()]
    if len(matches)!=1:
        raise BridgeContractError(f"e11b_checkpoint_count:{{len(matches)}}")
    path=matches[0]
    if path.stat().st_size!=TABPFN_CHECKPOINT_BYTES:
        raise BridgeContractError("e11b_checkpoint_bytes")
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=TABPFN_CHECKPOINT_SHA256:
        raise BridgeContractError("e11b_checkpoint_sha")
    return path

'''
    runtime = runtime.replace(marker, injected + marker, 1)

    validate = r'''def validate_result(result: dict, *, synthetic: bool = False) -> None:
    params={"n_estimators":8,"device":"cpu","fit_mode":"fit_preprocessors","memory_saving_mode":"auto","random_state":20260910,"n_preprocessing_jobs":1,"show_progress_bar":False,"ignore_pretraining_limits":False}
    if synthetic:
        if int(result.get("schema_version",-1))!=1 or result.get("experiment")!="synthetic_e11b_tabpfn_contract": raise BridgeContractError("e11b_synth_identity")
        if result.get("study_counts")!={"SDY180":34,"SDY515":16,"SDY519":17,"SDY56":60}: raise BridgeContractError("e11b_synth_counts")
        if result.get("feasibility_ready") is not True or int(result.get("outer_split_count",-1))!=4 or int(result.get("subject_overlap_max",-1))!=0: raise BridgeContractError("e11b_synth_feasibility")
        if result.get("target_transform")!="log" or result.get("package_version")!="8.5.0" or result.get("source_commit")!="9ed44abd5882140b88c9f2816c5791987ce059b9": raise BridgeContractError("e11b_synth_package")
        if result.get("wheel_sha256")!="4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0": raise BridgeContractError("e11b_synth_wheel")
        if result.get("kaggle_model_source")!="prior-labsai/tabpfn-3/pytorch/default/1": raise BridgeContractError("e11b_synth_model_source")
        if result.get("checkpoint_filename")!="tabpfn-v3-regressor-v3_default.ckpt" or int(result.get("checkpoint_bytes",0))!=233289807 or result.get("checkpoint_sha256")!="311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301": raise BridgeContractError("e11b_synth_checkpoint")
        if result.get("tabpfn_params")!=params or int(result.get("fold_count",-1))!=4 or result.get("finite_fold_scores") is not True: raise BridgeContractError("e11b_synth_model")
        if result.get("all_synthetic_backends") is not True or result.get("checkpoint_verified_in_synthetic") is not False: raise BridgeContractError("e11b_synth_backend_boundary")
        if float(result.get("fit_and_score_seconds",999))>float(result.get("cpu_feasibility_seconds",0)): raise BridgeContractError("e11b_synth_cpu")
        if result.get("challenge_rank_agreement_status")!="ok": raise BridgeContractError("e11b_synth_challenge")
        if result.get("held_outcomes_used_for_fit") is not False or result.get("held_outcomes_used_for_scoring") is not False or result.get("competition_submission_attempted") is not False or result.get("leaderboard_used_for_selection") is not False: raise BridgeContractError("e11b_synth_boundary")
        return
    if int(result.get("schema_version",-1))!=1 or result.get("experiment")!="strategy_v2_e11b_task11_tabpfn3": raise BridgeContractError("e11b_identity")
    if result.get("comparison_contract")!="paired_subject_purged_v2": raise BridgeContractError("e11b_comparison")
    for key in ("cross_study_target_scale_redefined","held_outcomes_used_for_fit","held_outcomes_used_for_scoring","leaderboard_used_for_selection","public_probe_authorized","competition_submission_attempted","incumbent_changed"):
        if result.get(key) is not False: raise BridgeContractError(f"e11b_boundary:{{key}}")
    if int(result.get("automatic_compute_retries",-1))!=0: raise BridgeContractError("e11b_retry")
    frozen=result.get("frozen_conditions") or {}
    expected={"task":"Task1.1","base_model":"pls_2","target_transform":"log","feature_space":"unchanged_compact_b21_task11","max_raw_features":200,"tabpfn_package_version":"8.5.0","tabpfn_source_commit":"9ed44abd5882140b88c9f2816c5791987ce059b9","tabpfn_wheel_filename":"tabpfn-8.5.0-py3-none-any.whl","tabpfn_wheel_sha256":"4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0","kaggle_model_source":"prior-labsai/tabpfn-3/pytorch/default/1","checkpoint_filename":"tabpfn-v3-regressor-v3_default.ckpt","checkpoint_bytes":233289807,"checkpoint_sha256":"311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301","tabpfn_params":params,"promotion_mean_delta":0.02,"promotion_minimum_study_delta":-0.10,"promotion_requires_strict_majority_wins":True}
    if frozen!=expected: raise BridgeContractError("e11b_frozen")
    lic=result.get("external_model_license") or {}
    if lic.get("model_license")!="tabpfn-3-license-v1.0" or lic.get("weight_redistribution_by_project") is not False or lic.get("official_kaggle_model_input") is not True: raise BridgeContractError("e11b_license")
    task=result.get("task") or {}
    if task.get("task")!="Task1.1" or task.get("base_contract")!={"kind":"b21","model":"pls_2","target_transform":"log","current_competition_incumbent":True}: raise BridgeContractError("e11b_base")
    folds=task.get("folds") or []
    if len(folds)!=4 or {str(f.get("held_study")) for f in folds}!={"SDY180","SDY515","SDY519","SDY56"}: raise BridgeContractError("e11b_folds")
    deltas=[]
    for fold in folds:
        if int(fold.get("n",0))<3: raise BridgeContractError("e11b_fold_support")
        for key in ("base_spearman","tabpfn_spearman","tabpfn_delta"):
            if not math.isfinite(float(fold.get(key,float("nan")))): raise BridgeContractError(f"e11b_nonfinite:{{key}}")
        if abs(float(fold["tabpfn_spearman"])-float(fold["base_spearman"])-float(fold["tabpfn_delta"]))>1e-12: raise BridgeContractError("e11b_delta")
        if fold.get("held_outcomes_used_for_fit") is not False or fold.get("held_outcomes_used_for_scoring") is not False: raise BridgeContractError("e11b_fold_leakage")
        audit=fold.get("fit_audit") or {}; checkpoint=audit.get("checkpoint") or {}
        if audit.get("backend")!="tabpfn" or audit.get("package_verified") is not True or audit.get("package_version_expected")!="8.5.0" or audit.get("package_source_commit")!="9ed44abd5882140b88c9f2816c5791987ce059b9": raise BridgeContractError("e11b_fold_package")
        if audit.get("wheel_filename")!="tabpfn-8.5.0-py3-none-any.whl" or audit.get("wheel_sha256")!="4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0" or audit.get("model_source")!="prior-labsai/tabpfn-3/pytorch/default/1": raise BridgeContractError("e11b_fold_artifact")
        if checkpoint.get("verified") is not True or checkpoint.get("filename")!="tabpfn-v3-regressor-v3_default.ckpt" or int(checkpoint.get("bytes",0))!=233289807 or checkpoint.get("sha256")!="311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301": raise BridgeContractError("e11b_fold_checkpoint")
        if int(audit.get("training_studies",-1))!=3 or int(audit.get("raw_features",9999))>200 or audit.get("target_transform")!="log" or audit.get("tabpfn_params")!=params or audit.get("held_outcomes_used_for_fit") is not False: raise BridgeContractError("e11b_fold_fit")
        deltas.append(float(fold["tabpfn_delta"]))
    candidate=task.get("candidate") or {}; promotion=candidate.get("promotion") or {}
    mean=sum(deltas)/len(deltas); minimum=min(deltas); wins=sum(v>0 for v in deltas); required=len(deltas)//2+1; passed=bool(mean>=0.02 and minimum>=-0.10 and wins>=required)
    if promotion.get("passed") is not passed or abs(float(promotion.get("mean_delta"))-mean)>1e-12 or abs(float(promotion.get("minimum_delta"))-minimum)>1e-12 or int(promotion.get("wins",-1))!=wins or int(promotion.get("required_wins",-1))!=required: raise BridgeContractError("e11b_promotion")
    if candidate.get("name")!="tabpfn3_default_cpu" or candidate.get("competition_candidate") is not passed or candidate.get("public_probe_authorized") is not False: raise BridgeContractError("e11b_candidate")
    agreement=candidate.get("challenge_agreement_vs_base") or {}; rank=(agreement.get("rank_spearman") or {}).get("value")
    if not math.isfinite(float(rank)) or not 0<=int(agreement.get("changed_rank_count",-1))<=40: raise BridgeContractError("e11b_challenge")
    final=task.get("final_fit_audit") or {}; checkpoint=final.get("checkpoint") or {}
    if final.get("backend")!="tabpfn" or final.get("package_verified") is not True or int(final.get("training_rows",-1))!=127 or int(final.get("training_studies",-1))!=4 or int(final.get("raw_features",9999))>200 or final.get("tabpfn_params")!=params: raise BridgeContractError("e11b_final_fit")
    if checkpoint.get("verified") is not True or checkpoint.get("sha256")!="311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301": raise BridgeContractError("e11b_final_checkpoint")
    feasibility=task.get("feasibility") or {}
    if feasibility.get("ready_for_one_cpu_tabpfn_comparison") is not True or int(feasibility.get("train_rows",-1))!=127 or int(feasibility.get("train_studies",-1))!=4 or int(feasibility.get("challenge_rows",-1))!=40 or len(feasibility.get("outer_splits") or [])!=4: raise BridgeContractError("e11b_feasibility")
    if any(int(s.get("subject_overlap",-1))!=0 for s in feasibility["outer_splits"]): raise BridgeContractError("e11b_purge")
    if int(task.get("challenge_rows",-1))!=40: raise BridgeContractError("e11b_challenge_rows")
    serialized=json.dumps(result,sort_keys=True,ensure_ascii=False)
    banned=('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"','ROW_','SUB_')
    if any(token in serialized for token in banned): raise BridgeContractError("e11b_aggregate_privacy")
'''
    runtime = replace_function(runtime, "validate_result", validate)

    summary = r'''def render_summary(result: dict) -> str:
    task=result.get("task") or {}; candidate=task.get("candidate") or {}; p=candidate.get("promotion") or {}; agreement=candidate.get("challenge_agreement_vs_base") or {}; fit=task.get("final_fit_audit") or {}
    lines=["# CMI-Flu strategy E11b Task1.1 TabPFN-3","","One fixed TabPFN-3 default CPU comparison against B2.1 PLS-2; aggregate-only; no submission/Public probe.","",f"- science commit: `{SCIENCE_COMMIT}`",f"- E11b blob: `{E11B_BLOB}`",f"- promotion: `{p.get('passed')}`; mean delta `{p.get('mean_delta')}`; minimum `{p.get('minimum_delta')}`; wins `{p.get('wins')}/{p.get('required_wins')}`",f"- Challenge rank agreement: `{(agreement.get('rank_spearman') or {}).get('value')}`; changed ranks `{agreement.get('changed_rank_count')}`",f"- final raw features: `{fit.get('raw_features')}`; fit seconds `{fit.get('fit_seconds')}`",""]
    for fold in task.get("folds") or []:
        lines.append(f"- {fold.get('held_study')}: n=`{fold.get('n')}` base=`{fold.get('base_spearman')}` tabpfn=`{fold.get('tabpfn_spearman')}` delta=`{fold.get('tabpfn_delta')}`")
    return "\n".join(lines)+"\n"
'''
    runtime = replace_function(runtime, "render_summary", summary)

    selftest = f'''def self_test() -> int:
    package=package_bytes()
    for source,expected,label in ((E11B_SOURCE,E11B_BLOB,"e11b"),(E11B_SYNTH_SOURCE,E11B_SYNTH_BLOB,"e11b_synthetic")):
        if git_blob_sha(source.encode("utf-8"))!=expected: raise BridgeContractError(f"{{label}}_blob_mismatch")
        compile(source,f"cmi_flu/{{label}}.py","exec")
    wheel=base64.b64decode(TABPFN_WHEEL_B64.encode("ascii"),validate=True)
    if hashlib.sha256(wheel).hexdigest()!=TABPFN_WHEEL_SHA256: raise BridgeContractError("e11b_wheel_selftest")
    if "result['tasks']" in globals().get("__loader_source__",""): raise BridgeContractError("e11b_writer_legacy")
    print(f"CMI_FLU_E11B_RUNTIME_SELF_TEST PASS request_id={{REQUEST_ID}} package_bytes={{len(package)}} science_commit={{SCIENCE_COMMIT}} e11b_blob={{E11B_BLOB}} wheel_bytes={{len(wheel)}}")
    return 0'''
    # selftest needs base64 import in its local body.
    selftest = selftest.replace("wheel=base64.b64decode", "import base64 as _b64\n    wheel=_b64.b64decode")
    runtime = replace_function(runtime, "self_test", selftest)

    main = r'''def main() -> int:
    args=parse_args()
    if args.self_test: return self_test()
    if args.synthetic:
        output_dir=args.output_dir.expanduser().resolve(); output_dir.mkdir(parents=True,exist_ok=False)
        runtime_root=Path("/tmp")/"cmi-flu-e11b-synthetic-runtime"
        if runtime_root.exists(): shutil.rmtree(runtime_root)
        runtime_root.mkdir(parents=True)
        package_path=runtime_root/"cmi_flu_bundle.zip"; package_path.write_bytes(package_bytes()); sys.path.insert(0,str(package_path))
        try:
            run_e11b,run_synthetic,e11b=load_e11b_modules()
            result=json_safe(dict(run_synthetic()))
            validate_result(result,synthetic=True)
            metrics_path=output_dir/"metrics.json"; summary_path=output_dir/"summary.md"; bridge_path=output_dir/"bridge-result.json"
            metrics_path.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")
            summary_path.write_text("# CMI-Flu E11b synthetic contract\n\nCompetition-Data-free TabPFN adapter/leakage validation; no submission.\n",encoding="utf-8")
            bridge={"schema_version":1,"request_id":REQUEST_ID,"competition":COMPETITION,"target_kernel":TARGET_KERNEL,"science_commit":SCIENCE_COMMIT,"strategy_e11b_blob_sha":E11B_BLOB,"strategy_e11b_synthetic_blob_sha":E11B_SYNTH_BLOB,"tabpfn_package_version":TABPFN_PACKAGE_VERSION,"tabpfn_wheel_sha256":TABPFN_WHEEL_SHA256,"tabpfn_model_source":TABPFN_MODEL_SOURCE,"tabpfn_checkpoint_sha256":TABPFN_CHECKPOINT_SHA256,"metrics_sha256":sha256_file(metrics_path),"summary_sha256":sha256_file(summary_path),"competition_submission_attempted":False,"leaderboard_used_for_selection":False,"contains_participant_identifiers":False,"contains_row_level_predictions":False,"synthetic":True}
            bridge_path.write_text(json.dumps(bridge,indent=2,sort_keys=True)+"\n",encoding="utf-8")
            print("CMI_FLU_E11B_COMPLETE synthetic=true submission=false")
            return 0
        finally: shutil.rmtree(runtime_root,ignore_errors=True)
    try:
        input_dir=locate_competition_data(args.input_dir)
    except Exception as exc:
        code=hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode()).hexdigest()[:20]
        print(f"CMI_FLU_E11B_FAILED stage=locate_competition_data exception_type={type(exc).__name__} error_code={code}",file=sys.stderr)
        return 2
    return execute(input_dir,args.output_dir)
'''
    runtime = replace_function(runtime, "main", main)

    old = '''        stage = "load_e11a"\n        run_e11a, run_synthetic, e11_module = load_e11_modules()\n        stage = "run_e11a"\n        result = json_safe(dict(run_e11a(config, inputs)))\n        stage = "validate_e11a"\n        validate_result(result, synthetic=False)\n'''
    new = '''        stage = "install_tabpfn"\n        install_tabpfn_wheel()\n        stage = "locate_tabpfn_checkpoint"\n        checkpoint_path = locate_tabpfn_checkpoint()\n        stage = "load_e11b"\n        run_e11b, run_synthetic, e11b_module = load_e11b_modules()\n        stage = "run_e11b"\n        result = json_safe(dict(run_e11b(config, inputs, checkpoint_path=checkpoint_path)))\n        stage = "validate_e11b"\n        validate_result(result, synthetic=False)\n'''
    if runtime.count(old) != 1:
        raise SystemExit("E11b execute-stage anchor changed")
    runtime = runtime.replace(old, new, 1)

    prov = '            "strategy_e11_synthetic_blob_sha": E11_SYNTH_BLOB,\n'
    if runtime.count(prov) != 1:
        raise SystemExit("E11b bridge provenance anchor changed")
    runtime = runtime.replace(
        prov,
        prov
        + '            "strategy_e11b_blob_sha": E11B_BLOB,\n'
        + '            "strategy_e11b_synthetic_blob_sha": E11B_SYNTH_BLOB,\n'
        + '            "tabpfn_package_version": TABPFN_PACKAGE_VERSION,\n'
        + '            "tabpfn_wheel_sha256": TABPFN_WHEEL_SHA256,\n'
        + '            "tabpfn_model_source": TABPFN_MODEL_SOURCE,\n'
        + '            "tabpfn_checkpoint_sha256": TABPFN_CHECKPOINT_SHA256,\n',
        1,
    )

    legacy = "len(result['tasks'])"
    if runtime.count(legacy) != 1:
        raise SystemExit(f"E11b expected exactly one stale writer expression, found {runtime.count(legacy)}")
    runtime = runtime.replace(legacy, "1", 1)
    runtime = runtime.replace("CMI-Flu strategy E11a Task1.1 pairwise ranker. Aggregate outputs only; no submission.", "CMI-Flu strategy E11b Task1.1 TabPFN-3. Aggregate outputs only; no submission.")
    runtime = runtime.replace("CMI_FLU_E11A_FAILED", "CMI_FLU_E11B_FAILED").replace("CMI_FLU_E11A_COMPLETE", "CMI_FLU_E11B_COMPLETE")
    if "result['tasks']" in runtime:
        raise SystemExit("E11b generated runtime retains stale tasks writer access")
    if "kaggle competitions submit" in runtime or "competition_submit" in runtime:
        raise SystemExit("E11b generated runtime contains submission path")
    compile(runtime, "generated_e11b_runtime.py", "exec")
    return runtime


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    refs = args.reference_dir.resolve()
    validate_request(root)
    e11b = require_text(root, E11B_PATH, E11B_BLOB)
    synth = require_text(root, E11B_SYNTH_PATH, E11B_SYNTH_BLOB)
    wheel = require_wheel(args.tabpfn_wheel)
    with tempfile.TemporaryDirectory(prefix="cmi-flu-e11b-base-") as tmp:
        base = Path(tmp) / "e11a_runtime.py"
        subprocess.run(
            [sys.executable, str(root / BASE_PREPARE), "--repository-root", str(root), "--reference-dir", str(refs), "--output", str(base)],
            check=True,
        )
        runtime = patch_runtime(base.read_text(), e11b, synth, wheel)
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime)
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        f"CMI_FLU_E11B_PREPARE_PASS science_commit={SCIENCE_COMMIT} e11b_blob={E11B_BLOB} "
        f"target={TARGET_KERNEL} runtime_sha256={hashlib.sha256(runtime.encode()).hexdigest()} "
        f"wheel_sha256={TABPFN_WHEEL_SHA256} model_source={MODEL_SOURCE} submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
