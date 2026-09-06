"""Launch the frozen Gold generic aggregation ablation on Kaggle T4 x2."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

EXECUTION_ID = "shadow-gold-generic-aggregation-v1"
TARGET = "renta0426/shadow-gold-generic-aggregation-v1"
RESEARCH_COMMIT = "3f0f6475e582067cb52c1070b6e3b48781935bd5"
PROBE_LOCAL_RESULT_COMMIT = "844d83992ff364ee7bc43ae3560fb06540e2bd99"
MACHINE_SHAPE = "NvidiaTeslaT4"
NOTEBOOK_NAME = "shadow-gold-generic-aggregation-v1.ipynb"
PERSISTENT_OUTPUTS = (
    "generic_aggregation_selected.json",
    "generic_aggregation_metrics.json",
    "generic_aggregation_predictions.jsonl",
    "generic_aggregation_runtime_manifest.json",
)

SNAPSHOTS = {
    "poisoned_chalice/controlled_transfer.py": ("materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/controlled_transfer.py", "09b04c10d934654044817bb9b36fd7358a2aa942", 65536),
    "poisoned_chalice/shadow_gold_corpus.py": ("materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_gold_corpus.py", "241e3481f7dc56f5fc7495436f3cfc33274a0823", 32768),
    "poisoned_chalice/shadow_gold_kaggle.py": ("materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_gold_kaggle.py", "e68eb320605a943bb7d46b0e9ce51bb06142ffef", 65536),
    "poisoned_chalice/shadow_gold_transfer.py": ("materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_gold_transfer.py", "1ee969fe852f8a18d00a5c0249f4a8a90d61338a", 65536),
    "poisoned_chalice/shadow_protocol.py": ("materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_protocol.py", "1c47b80696050aa2e5e7c62384617df61ecb80da", 65536),
    "poisoned_chalice/shadow_scoring.py": ("materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_scoring.py", "6bc2b55bec2a32cbb3b0cfe870e426f80260730d", 65536),
    "poisoned_chalice/shadow_training.py": ("materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_training.py", "2fe3cf2079659ae10e1ebe0d5973f3ce96c26a03", 65536),
    "poisoned_chalice/shadow_training_dual_gpu.py": ("materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_training_dual_gpu.py", "c75a5963fa930916024cb16b4fbf562ba4afa818", 65536),
    "poisoned_chalice/stage2_api.py": ("materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/stage2_api.py", "16578d3e01b6471e5bfb2b04cebb99e1b3126c7c", 65536),
    "poisoned_chalice/shadow_gold_kaggle_v2.py": ("materialized/poisoned-chalice-shadow-gold-v2/poisoned_chalice/shadow_gold_kaggle_v2.py", "df6e67f7d0e67d2585fd6226b9d2592223f808d2", 32768),
    "poisoned_chalice/shadow_gold_positive_control.py": ("materialized/poisoned-chalice-shadow-gold-positive-control-v1/poisoned_chalice/shadow_gold_positive_control.py", "8b14ba725a939b8a7c21aff93f70165c304fde79", 65536),
    "shadow_gold_positive_control_v1.json": ("materialized/poisoned-chalice-shadow-gold-positive-control-v1/shadow_gold_positive_control_v1.json", "424fe2674daebc1bba51b560d9ad057dff2b82c2", 32768),
    "poisoned_chalice/shadow_gold_positive_control_v2.py": ("materialized/poisoned-chalice-shadow-gold-positive-control-v2/poisoned_chalice/shadow_gold_positive_control_v2.py", "374a6197b4238f4ea7279b27f08edb2a6f366e50", 32768),
    "poisoned_chalice/shadow_gold_generic_aggregation.py": ("materialized/poisoned-chalice-shadow-gold-generic-aggregation-v1/poisoned_chalice/shadow_gold_generic_aggregation.py", "48fa30d20b048fab14b6bb0167d4f5b6cd52e31b", 65536),
    "shadow_gold_generic_aggregation_v1.json": ("materialized/poisoned-chalice-shadow-gold-generic-aggregation-v1/shadow_gold_generic_aggregation_v1.json", "e3da785732f630cc24c21734a0814b5d6ae455a5", 32768),
}


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def load_snapshots(root: Path) -> dict[str, bytes]:
    result = {}
    for logical, (relative, expected, maximum) in SNAPSHOTS.items():
        data = (root / relative).read_bytes()
        if len(data) > maximum or blob_sha(data) != expected:
            raise RuntimeError(f"snapshot identity mismatch: {relative}")
        result[logical] = data
    return result


def validate_sources(files: dict[str, bytes]) -> None:
    for logical, data in files.items():
        if logical.endswith(".py"):
            compile(data.decode(), logical, "exec")
    source = files["poisoned_chalice/shadow_gold_generic_aggregation.py"].decode()
    for marker in (
        "run_generic_aggregation_benchmark",
        "probe_boundary_used",
        "probe_local_score_used",
        "stage2_v3_selection_allowed",
        "score_samples_detailed",
    ):
        if marker not in source:
            raise RuntimeError(f"generic aggregation marker missing: {marker}")


def validate_configs(positive: dict, aggregation: dict) -> None:
    if positive.get("experiment_id") != "shadow-gold-positive-control-v1":
        raise RuntimeError("positive-control config identity changed")
    protocol = positive.get("protocol") or {}
    if (protocol.get("exposure_repeats"), protocol.get("expected_training_sequences"), protocol.get("expected_optimizer_steps_per_architecture")) != (16, 81920, 1280):
        raise RuntimeError("positive-control training contract changed")
    if aggregation.get("execution_id") != EXECUTION_ID:
        raise RuntimeError("aggregation config identity changed")
    if len(aggregation.get("candidate_features") or []) != 18:
        raise RuntimeError("aggregation candidate count changed")
    scoring = aggregation.get("generic_scoring") or {}
    if scoring.get("min_k_percents") != [1,2,5,10] or scoring.get("local_widths") != [8,16,32,64]:
        raise RuntimeError("aggregation scale matrix changed")
    for key in ("probe_boundary_used", "probe_text_used_to_select_tokens", "membership_used_for_scoring", "benchmark_id_used_as_model_input"):
        if scoring.get(key) is not False:
            raise RuntimeError(f"generic scorer guard changed: {key}")
    guards = aggregation.get("scientific_guards") or {}
    for key in ("probe_local_score_used_as_candidate", "known_probe_boundary_used", "stage2_v3_selection_allowed", "competition_feature_promotion_allowed", "external_model_holdout_consumed", "fourth_external_holdout_consumed"):
        if guards.get(key) is not False:
            raise RuntimeError(f"aggregation guard changed: {key}")
    recovery = aggregation.get("recovery_criterion") or {}
    if recovery.get("baseline_candidate") != "loss_max" or recovery.get("minimum_auc_gain_vs_baseline_each_architecture") != 0.03 or recovery.get("both_architectures_required") is not True:
        raise RuntimeError("aggregation recovery criterion changed")


def build_code(files: dict[str, bytes]) -> str:
    validate_sources(files)
    positive_text = files["shadow_gold_positive_control_v1.json"].decode()
    aggregation_text = files["shadow_gold_generic_aggregation_v1.json"].decode()
    validate_configs(json.loads(positive_text), json.loads(aggregation_text))
    payloads = {key: base64.b64encode(value).decode("ascii") for key, value in files.items() if not key.endswith(".json")}
    template = r'''from __future__ import annotations
import base64, json, os, shutil, sys
from pathlib import Path
EXECUTION_ID = __EXECUTION_ID__
RESEARCH_COMMIT = __RESEARCH_COMMIT__
PROBE_LOCAL_RESULT_COMMIT = __PROBE_RESULT__
SOURCE_PAYLOADS = __PAYLOADS__
POSITIVE_CONFIG_TEXT = __POSITIVE__
AGGREGATION_CONFIG_TEXT = __AGGREGATION__
PERSISTENT_OUTPUTS = __OUTPUTS__
working = Path('/kaggle/working'); working.mkdir(parents=True, exist_ok=True)
for path in list(working.iterdir()): shutil.rmtree(path) if path.is_dir() else path.unlink()
scratch = Path('/tmp') / EXECUTION_ID
if scratch.exists(): shutil.rmtree(scratch)
scratch.mkdir(parents=True)
runtime_root = scratch / 'runtime'
for relative, encoded in SOURCE_PAYLOADS.items():
    destination = runtime_root / relative; destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(base64.b64decode(encoded))
(runtime_root / 'poisoned_chalice/__init__.py').write_text('', encoding='utf-8')
positive_config = json.loads(POSITIVE_CONFIG_TEXT); aggregation_config = json.loads(AGGREGATION_CONFIG_TEXT)
if positive_config['protocol']['exposure_repeats'] != 16 or positive_config['protocol']['expected_optimizer_steps_per_architecture'] != 1280: raise RuntimeError('training contract changed')
if aggregation_config['scientific_guards']['known_probe_boundary_used'] is not False: raise RuntimeError('probe boundary leaked')
if aggregation_config['scientific_guards']['stage2_v3_selection_allowed'] is not False: raise RuntimeError('Stage2-v3 guard changed')
sys.path.insert(0, str(runtime_root)); os.environ['PYTHONPATH'] = str(runtime_root) + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
from poisoned_chalice.shadow_gold_generic_aggregation import run_generic_aggregation_benchmark

def default(value):
    if hasattr(value, 'item'): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)
def write_json(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True, default=default) + '\n', encoding='utf-8')
def publish(source, destination):
    temporary = destination.with_suffix(destination.suffix + '.partial'); shutil.copyfile(source, temporary); os.replace(temporary, destination)
try:
    metrics, predictions, manifest = run_generic_aggregation_benchmark(scratch_directory=scratch/'benchmark', positive_control_config=positive_config, aggregation_config=aggregation_config, timeout_seconds=5400)
    if manifest.get('status') != 'complete' or metrics.get('status') != 'complete': raise RuntimeError('generic aggregation did not complete')
    if manifest.get('probe_boundary_used') is not False or manifest.get('probe_local_score_used') is not False: raise RuntimeError('probe-specific information leaked')
    if manifest.get('stage2_v3_selection_allowed') is not False or manifest.get('competition_feature_promotion_allowed') is not False: raise RuntimeError('promotion guard failed')
    if manifest.get('features_sealed_before_label_reveal') is not True: raise RuntimeError('label boundary failed')
    if len(predictions) != 2560: raise RuntimeError('prediction count changed')
    manifest = dict(manifest); manifest['research_commit'] = RESEARCH_COMMIT; manifest['probe_local_result_commit'] = PROBE_LOCAL_RESULT_COMMIT
    selected = {'selected_candidate': metrics['selected_candidate'], 'recovery_criterion': metrics['recovery_criterion'], 'selected_auc_gain_vs_loss_max': metrics['selected_auc_gain_vs_loss_max']}
    staged = scratch/'published'; staged.mkdir()
    write_json(staged/'generic_aggregation_selected.json', selected)
    write_json(staged/'generic_aggregation_metrics.json', metrics)
    predictions.to_json(staged/'generic_aggregation_predictions.jsonl', orient='records', lines=True, force_ascii=False)
    write_json(staged/'generic_aggregation_runtime_manifest.json', manifest)
    if {p.name for p in staged.iterdir()} != set(PERSISTENT_OUTPUTS): raise RuntimeError('staged output allowlist changed')
    for name in PERSISTENT_OUTPUTS: publish(staged/name, working/name)
    for path in list(working.iterdir()):
        if path.name not in PERSISTENT_OUTPUTS: shutil.rmtree(path) if path.is_dir() else path.unlink()
    if {p.name for p in working.iterdir()} != set(PERSISTENT_OUTPUTS): raise RuntimeError('persistent output allowlist changed')
    print(json.dumps({'execution_id': EXECUTION_ID, 'status':'complete', 'selected_candidate':metrics['selected_candidate'], 'recovery_passed':metrics['recovery_criterion']['passed'], 'left_auc_gain':metrics['selected_auc_gain_vs_loss_max']['left'], 'right_auc_gain':metrics['selected_auc_gain_vs_loss_max']['right'], 'stage2_v3_selection_allowed':False}, sort_keys=True))
finally:
    shutil.rmtree(scratch, ignore_errors=True)
'''
    for marker, value in {
        "__EXECUTION_ID__": repr(EXECUTION_ID), "__RESEARCH_COMMIT__": repr(RESEARCH_COMMIT), "__PROBE_RESULT__": repr(PROBE_LOCAL_RESULT_COMMIT),
        "__PAYLOADS__": repr(payloads), "__POSITIVE__": repr(positive_text), "__AGGREGATION__": repr(aggregation_text), "__OUTPUTS__": repr(PERSISTENT_OUTPUTS),
    }.items():
        template = template.replace(marker, value)
    compile(template, NOTEBOOK_NAME, "exec")
    return template


def materialize(root: Path, kernel_dir: Path) -> None:
    files = load_snapshots(root); code = build_code(files)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    notebook = {"cells":[{"cell_type":"markdown","metadata":{},"source":["# Gold generic aggregation v1\n"]},{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":code.splitlines(keepends=True)}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
    (kernel_dir/NOTEBOOK_NAME).write_text(json.dumps(notebook, indent=2)+'\n', encoding='utf-8')
    metadata = {"id":TARGET,"title":"Gold generic aggregation ablation v1","code_file":NOTEBOOK_NAME,"language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":False,"machine_shape":MACHINE_SHAPE,"dataset_sources":[],"kernel_sources":[],"competition_sources":[],"model_sources":[]}
    (kernel_dir/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    validate_bundle(kernel_dir)
    print('SHADOW_GOLD_GENERIC_AGG_V1_MATERIALIZE PASS snapshots=15 candidates=18 outputs=4 gpu=2 steps=1280 eval=5120 retries=0')


def validate_bundle(kernel_dir: Path) -> None:
    if {p.name for p in kernel_dir.iterdir()} != {NOTEBOOK_NAME,'kernel-metadata.json'}: raise RuntimeError('kernel bundle allowlist changed')
    notebook=json.loads((kernel_dir/NOTEBOOK_NAME).read_text()); code=''.join(notebook['cells'][1]['source']); compile(code, NOTEBOOK_NAME, 'exec')
    for marker in (RESEARCH_COMMIT,'run_generic_aggregation_benchmark','known_probe_boundary_used','stage2_v3_selection_allowed','generic_aggregation_metrics.json'):
        if marker not in code: raise RuntimeError(f'generated marker missing: {marker}')
    metadata=json.loads((kernel_dir/'kernel-metadata.json').read_text())
    if metadata.get('id')!=TARGET or metadata.get('enable_gpu') is not True or metadata.get('enable_tpu') is not False or metadata.get('enable_internet') is not False or metadata.get('machine_shape')!=MACHINE_SHAPE: raise RuntimeError('kernel metadata changed')


def validate_static(path: Path) -> None:
    source=path.read_text(); tree=ast.parse(source); count=0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id=='subprocess' and node.func.attr=='run' and node.args and isinstance(node.args[0], (ast.List,ast.Tuple)):
            vals=[x.value if isinstance(x,ast.Constant) and isinstance(x.value,str) else '<dynamic>' for x in node.args[0].elts]
            if any(vals[i:i+2]==['kernels','push'] for i in range(len(vals)-1)): count+=1
    if count!=1: raise RuntimeError(f'launcher must contain one kernels push, found {count}')


def execute(kaggle_bin: Path, kernel_dir: Path) -> None:
    validate_bundle(kernel_dir)
    result=subprocess.run([str(kaggle_bin),'kernels','push','-p',str(kernel_dir),'--timeout','3600'],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    if result.returncode!=0: raise RuntimeError(f'Kaggle push returned {result.returncode}; outcome ambiguous and retry forbidden')
    print('SHADOW_GOLD_GENERIC_AGG_V1_LAUNCH_ACCEPTED target=renta0426/shadow-gold-generic-aggregation-v1 accelerator=gpu retries=0 submissions=0 private_repo_access=0')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--launcher',type=Path); parser.add_argument('--snapshot-root',type=Path); parser.add_argument('--kernel-dir',type=Path); parser.add_argument('--kaggle-bin',type=Path); parser.add_argument('--static',action='store_true'); parser.add_argument('--materialize',action='store_true'); parser.add_argument('--execute',action='store_true'); args=parser.parse_args()
    if sum((args.static,args.materialize,args.execute))!=1: raise SystemExit('select exactly one operation')
    if args.static: validate_static(args.launcher); print('SHADOW_GOLD_GENERIC_AGG_V1_STATIC PASS write_calls=1 retries=0 submissions=0 outputs=4'); return
    if args.materialize: materialize(args.snapshot_root,args.kernel_dir); return
    execute(args.kaggle_bin,args.kernel_dir)
if __name__=='__main__': main()
