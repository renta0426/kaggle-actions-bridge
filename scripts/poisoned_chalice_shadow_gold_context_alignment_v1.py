"""Launch the frozen Gold training-context alignment ablation on Kaggle T4 x2."""
from __future__ import annotations

import argparse, ast, base64, hashlib, importlib.util, json, ssl, subprocess, time, urllib.request
from pathlib import Path

EXECUTION_ID="shadow-gold-context-alignment-v1"
TARGET="renta0426/shadow-gold-context-alignment-v1"
RESEARCH_COMMIT="15786822673e68ed0340a7975f17a3c749cc1889"
CONTENT_REFERENCE_RESULT_COMMIT="f09c8b9c4f69f6b89ba33f6a831216f25623411f"
PROBE_LOCAL_RESULT_COMMIT="844d83992ff364ee7bc43ae3560fb06540e2bd99"
BASE_LAUNCHER_BLOB="fff2d72e0a00480ad6cb9d5523dd059c81812da7"
CONTEXT_SOURCE_BLOB="24bf1c9cfff0a6688fea41803709e822b7974391"
CONTEXT_CONFIG_BLOB="1e831f4d4f00ab26870344e05ef3bfb7399f7f15"
MACHINE_SHAPE="NvidiaTeslaT4"
NOTEBOOK_NAME="shadow-gold-context-alignment-v1.ipynb"
OUTPUTS=("context_alignment_selected.json","context_alignment_metrics.json","context_alignment_predictions.jsonl","context_alignment_runtime_manifest.json")


def blob(data:bytes)->str: return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()

def load_base(path:Path):
    data=path.read_bytes()
    if len(data)>262144 or blob(data)!=BASE_LAUNCHER_BLOB: raise RuntimeError("base launcher identity mismatch")
    spec=importlib.util.spec_from_file_location("gold_content_ref_v1_builder",path)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load base builder")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def fetch_public(relative:str,want:str,maximum:int)->bytes:
    url=f"https://raw.githubusercontent.com/renta0426/The-Poisoned-Chalice-of-LLM-Evaluation/{RESEARCH_COMMIT}/{relative}"
    ctx=ssl.create_default_context(); last=None
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"kaggle-actions-bridge/1"})
            with urllib.request.urlopen(req,timeout=30,context=ctx) as response: data=response.read(maximum+1)
            if len(data)>maximum or blob(data)!=want: raise RuntimeError("public snapshot identity mismatch")
            return data
        except Exception as exc:
            last=exc
            if attempt<2: time.sleep(attempt+1)
    raise RuntimeError(f"public snapshot fetch failed: {last}")

def load_files(root:Path,base_launcher:Path)->dict[str,bytes]:
    base=load_base(base_launcher); files=base.load_snapshots(root)
    files.pop("poisoned_chalice/shadow_gold_content_reference.py",None); files.pop("shadow_gold_content_reference_v1.json",None)
    files["poisoned_chalice/shadow_gold_context_alignment.py"]=fetch_public("src/poisoned_chalice/shadow_gold_context_alignment.py",CONTEXT_SOURCE_BLOB,65536)
    files["shadow_gold_context_alignment_v1.json"]=fetch_public("configs/shadow_gold_context_alignment_v1.json",CONTEXT_CONFIG_BLOB,32768)
    return files

def validate(files:dict[str,bytes])->tuple[str,str]:
    for name,data in files.items():
        if name.endswith('.py'): compile(data.decode(),name,'exec')
    source=files["poisoned_chalice/shadow_gold_context_alignment.py"].decode()
    for marker in ("run_context_alignment_benchmark","build_training_aligned_suffix_plan","required_gpu_name_fragment=\"T4\"","probe_boundary_used","stage2_v3_selection_allowed"):
        if marker not in source: raise RuntimeError(f"context source marker missing: {marker}")
    positive=files["shadow_gold_positive_control_v1.json"].decode(); config=files["shadow_gold_context_alignment_v1.json"].decode(); p=json.loads(positive); c=json.loads(config)
    if p.get("experiment_id")!="shadow-gold-positive-control-v1" or p["protocol"].get("exposure_repeats")!=16 or p["protocol"].get("expected_optimizer_steps_per_architecture")!=1280: raise RuntimeError("positive-control contract changed")
    if c.get("execution_id")!=EXECUTION_ID or c.get("role")!="gold_training_context_alignment_ablation": raise RuntimeError("context config identity changed")
    if c["base_evidence"].get("content_reference_result_commit")!=CONTENT_REFERENCE_RESULT_COMMIT or c["base_evidence"].get("probe_local_result_commit")!=PROBE_LOCAL_RESULT_COMMIT: raise RuntimeError("evidence chain changed")
    if c["candidate_features"]!=["aligned_loss_mean","aligned_min_k_01","aligned_min_k_02","aligned_min_k_05","aligned_min_k_10","aligned_local08_max","aligned_local16_max","aligned_local32_max","aligned_local64_max"]: raise RuntimeError("candidate matrix changed")
    if c["aligned_scoring"].get("context_construction")!="bos_plus_last_254_payload_plus_eos_then_right_pad" or c["aligned_scoring"].get("probe_boundary_used") is not False or c["aligned_scoring"].get("probe_text_used") is not False: raise RuntimeError("aligned scoring contract changed")
    r=c["recovery_criterion"]
    if (r.get("minimum_holdout_auc_each_architecture"),r.get("minimum_auc_gain_vs_raw_stage2_loss_max_each_architecture"),r.get("minimum_tpr_at_1pct_fpr_each_architecture"),r.get("all_conditions_required"))!=(0.56,0.03,0.015,True): raise RuntimeError("recovery gate changed")
    g=c["scientific_guards"]
    if g.get("features_sealed_before_label_reveal") is not True or any(g.get(k) is not False for k in ("known_probe_boundary_used","probe_text_used","stage2_v3_selection_allowed","competition_feature_promotion_allowed","external_model_holdout_consumed","fourth_external_holdout_consumed")) or g.get("competition_rows_used")!=0 or g.get("external_rows_used")!=0: raise RuntimeError("scientific guard changed")
    return positive,config

def build_code(files:dict[str,bytes])->str:
    positive,config=validate(files); payloads={k:base64.b64encode(v).decode() for k,v in files.items() if not k.endswith('.json')}
    code=r'''from __future__ import annotations
import base64,json,os,shutil,sys
from pathlib import Path
EXECUTION_ID=__E__; RESEARCH_COMMIT=__R__; CONTENT_REFERENCE_RESULT_COMMIT=__C__; PROBE_LOCAL_RESULT_COMMIT=__P__; SOURCE_PAYLOADS=__S__; POSITIVE_CONFIG_TEXT=__PC__; ALIGNMENT_CONFIG_TEXT=__AC__; OUTPUTS=__O__
working=Path('/kaggle/working'); working.mkdir(parents=True,exist_ok=True)
for path in list(working.iterdir()): shutil.rmtree(path) if path.is_dir() else path.unlink()
scratch=Path('/tmp')/EXECUTION_ID
if scratch.exists(): shutil.rmtree(scratch)
scratch.mkdir(parents=True); runtime=scratch/'runtime'
for relative,encoded in SOURCE_PAYLOADS.items():
    dst=runtime/relative; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(base64.b64decode(encoded))
(runtime/'poisoned_chalice/__init__.py').write_text('')
positive=json.loads(POSITIVE_CONFIG_TEXT); alignment=json.loads(ALIGNMENT_CONFIG_TEXT)
if alignment['aligned_scoring']['probe_boundary_used'] is not False or alignment['aligned_scoring']['probe_text_used'] is not False or alignment['scientific_guards']['stage2_v3_selection_allowed'] is not False: raise RuntimeError('scientific boundary changed')
sys.path.insert(0,str(runtime)); os.environ['PYTHONPATH']=str(runtime)+(os.pathsep+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
from poisoned_chalice.shadow_gold_context_alignment import run_context_alignment_benchmark
def default(value):
    if hasattr(value,'item'): return value.item()
    if isinstance(value,Path): return str(value)
    raise TypeError(type(value).__name__)
def write_json(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True,default=default)+'\n')
def publish(source,destination):
    temp=destination.with_suffix(destination.suffix+'.partial'); shutil.copyfile(source,temp); os.replace(temp,destination)
try:
    metrics,predictions,manifest=run_context_alignment_benchmark(scratch_directory=scratch/'benchmark',positive_control_config=positive,context_alignment_config=alignment,timeout_seconds=5400)
    if metrics.get('status')!='complete' or manifest.get('status')!='complete' or manifest.get('training_context_alignment_used') is not True or manifest.get('features_sealed_before_label_reveal') is not True: raise RuntimeError('context-alignment incomplete')
    if manifest.get('probe_boundary_used') is not False or manifest.get('probe_text_used') is not False or manifest.get('stage2_v3_selection_allowed') is not False or manifest.get('competition_feature_promotion_allowed') is not False: raise RuntimeError('scientific boundary failed')
    if manifest.get('competition_rows_used')!=0 or manifest.get('external_rows_used')!=0 or manifest.get('automatic_compute_retries')!=0 or len(predictions)!=2560: raise RuntimeError('runtime boundary failed')
    manifest=dict(manifest); manifest['research_commit']=RESEARCH_COMMIT; manifest['content_reference_result_commit']=CONTENT_REFERENCE_RESULT_COMMIT; manifest['probe_local_result_commit']=PROBE_LOCAL_RESULT_COMMIT
    selected={'selected_candidate':metrics['selected_candidate'],'recovery_passed':metrics['recovery_passed'],'recovery_checks':metrics['recovery_checks'],'selected_auc_gain_vs_raw_stage2_loss_max':metrics['selected_auc_gain_vs_raw_stage2_loss_max'],'selected_holdout':metrics['selected_holdout'],'raw_stage2_loss_max_holdout':metrics['raw_stage2_loss_max_holdout']}
    staged=scratch/'published'; staged.mkdir(); write_json(staged/'context_alignment_selected.json',selected); write_json(staged/'context_alignment_metrics.json',metrics); predictions.to_json(staged/'context_alignment_predictions.jsonl',orient='records',lines=True,force_ascii=False); write_json(staged/'context_alignment_runtime_manifest.json',manifest)
    if {p.name for p in staged.iterdir()}!=set(OUTPUTS): raise RuntimeError('output allowlist changed')
    for name in OUTPUTS: publish(staged/name,working/name)
    print(json.dumps({'execution_id':EXECUTION_ID,'status':'complete','selected_candidate':metrics['selected_candidate'],'recovery_passed':metrics['recovery_passed'],'left_auc':metrics['selected_holdout']['left']['auc'],'right_auc':metrics['selected_holdout']['right']['auc'],'stage2_v3_selection_allowed':False},sort_keys=True))
finally: shutil.rmtree(scratch,ignore_errors=True)
'''
    for marker,value in {"__E__":repr(EXECUTION_ID),"__R__":repr(RESEARCH_COMMIT),"__C__":repr(CONTENT_REFERENCE_RESULT_COMMIT),"__P__":repr(PROBE_LOCAL_RESULT_COMMIT),"__S__":repr(payloads),"__PC__":repr(positive),"__AC__":repr(config),"__O__":repr(OUTPUTS)}.items(): code=code.replace(marker,value)
    compile(code,NOTEBOOK_NAME,'exec'); return code

def validate_bundle(kernel_dir:Path)->None:
    if {p.name for p in kernel_dir.iterdir()}!={NOTEBOOK_NAME,'kernel-metadata.json'}: raise RuntimeError('bundle allowlist changed')
    nb=json.loads((kernel_dir/NOTEBOOK_NAME).read_text()); cells=nb.get('cells') or []
    if len(cells)!=2 or any(not c.get('id') for c in cells): raise RuntimeError('stable notebook cell IDs missing')
    code=''.join(cells[1]['source']); compile(code,NOTEBOOK_NAME,'exec')
    for marker in (RESEARCH_COMMIT,'run_context_alignment_benchmark','training_context_alignment_used','context_alignment_metrics.json'):
        if marker not in code: raise RuntimeError(f'generated marker missing: {marker}')
    meta=json.loads((kernel_dir/'kernel-metadata.json').read_text())
    if (meta.get('id'),meta.get('title'),meta.get('enable_gpu'),meta.get('enable_tpu'),meta.get('enable_internet'),meta.get('machine_shape'))!=(TARGET,'Gold training-context alignment v1',True,False,False,MACHINE_SHAPE): raise RuntimeError('kernel metadata changed')
    if any(meta.get(k) for k in ('dataset_sources','kernel_sources','competition_sources','model_sources')): raise RuntimeError('kernel sources changed')

def materialize(root:Path,kernel_dir:Path,base_launcher:Path)->None:
    code=build_code(load_files(root,base_launcher)); kernel_dir.mkdir(parents=True,exist_ok=True)
    nb={'cells':[{'cell_type':'markdown','id':'context-alignment-intro','metadata':{},'source':['# Gold training-context alignment v1\n']},{'cell_type':'code','id':'run-context-alignment','execution_count':None,'metadata':{},'outputs':[],'source':code.splitlines(keepends=True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    (kernel_dir/NOTEBOOK_NAME).write_text(json.dumps(nb,indent=2)+'\n')
    meta={'id':TARGET,'title':'Gold training-context alignment v1','code_file':NOTEBOOK_NAME,'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':MACHINE_SHAPE,'dataset_sources':[],'kernel_sources':[],'competition_sources':[],'model_sources':[]}; (kernel_dir/'kernel-metadata.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n'); validate_bundle(kernel_dir)
    print('SHADOW_GOLD_CONTEXT_ALIGNMENT_V1_MATERIALIZE PASS base_snapshots=15 replaced=2 public_snapshots=2 candidates=9 outputs=4 gpu=2 steps=1280 cell_ids=stable')

def validate_static(path:Path)->None:
    tree=ast.parse(path.read_text()); commands=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='subprocess' and node.func.attr=='run' and node.args and isinstance(node.args[0],(ast.List,ast.Tuple)):
            commands.append([x.value if isinstance(x,ast.Constant) and isinstance(x.value,str) else '<dynamic>' for x in node.args[0].elts])
    def has(cmd,a,b): return any(cmd[i:i+2]==[a,b] for i in range(len(cmd)-1))
    if sum(has(c,'kernels','push') for c in commands)!=1: raise RuntimeError('launcher write count changed')
    if any(has(c,a,b) for c in commands for a,b in (("competitions","submit"),("kernels","cancel"),("kernels","delete"),("datasets","create"),("datasets","version"),("models","create"),("models","version"))): raise RuntimeError('forbidden write present')

def execute(kaggle_bin:Path,kernel_dir:Path)->None:
    validate_bundle(kernel_dir); result=subprocess.run([str(kaggle_bin),'kernels','push','-p',str(kernel_dir),'--timeout','3600'],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    if result.returncode!=0: raise RuntimeError(f'Kaggle push returned {result.returncode}; outcome ambiguous and retry forbidden')
    print('SHADOW_GOLD_CONTEXT_ALIGNMENT_V1_LAUNCH_ACCEPTED target=renta0426/shadow-gold-context-alignment-v1 accelerator=gpu retries=0 submissions=0 private_repo_access=0')

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--launcher',type=Path); p.add_argument('--base-launcher',type=Path); p.add_argument('--snapshot-root',type=Path); p.add_argument('--kernel-dir',type=Path); p.add_argument('--kaggle-bin',type=Path); p.add_argument('--static',action='store_true'); p.add_argument('--materialize',action='store_true'); p.add_argument('--execute',action='store_true'); a=p.parse_args()
    if sum((a.static,a.materialize,a.execute))!=1: raise SystemExit('select exactly one operation')
    if a.static: validate_static(a.launcher); print('SHADOW_GOLD_CONTEXT_ALIGNMENT_V1_STATIC PASS write_calls=1 retries=0 submissions=0'); return
    if a.materialize: materialize(a.snapshot_root,a.kernel_dir,a.base_launcher); return
    execute(a.kaggle_bin,a.kernel_dir)
if __name__=='__main__': main()
