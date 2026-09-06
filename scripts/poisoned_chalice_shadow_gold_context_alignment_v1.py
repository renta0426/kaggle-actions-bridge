"""Launch the frozen Gold training-context alignment ablation on Kaggle T4 x2.

The private research repository is never read by the protected job.  This
launcher consumes only exact public bridge snapshots.  The new research runtime
is stored as three bounded public parts; they are concatenated and accepted only
if the reconstructed Git blob is exactly the research blob pinned below.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

EXECUTION_ID = "shadow-gold-context-alignment-v1"
REQUEST_ID = "20260906-poisoned-chalice-shadow-gold-context-alignment-v1-001"
TARGET = "renta0426/shadow-gold-context-alignment-v1"
RESEARCH_COMMIT = "15786822673e68ed0340a7975f17a3c749cc1889"
CONTENT_REFERENCE_RESULT_COMMIT = "f09c8b9c4f69f6b89ba33f6a831216f25623411f"
PROBE_LOCAL_RESULT_COMMIT = "844d83992ff364ee7bc43ae3560fb06540e2bd99"
CONTEXT_RUNTIME_BLOB = "24bf1c9cfff0a6688fea41803709e822b7974391"
CONTEXT_CONFIG_BLOB = "1e831f4d4f00ab26870344e05ef3bfb7399f7f15"
MACHINE_SHAPE = "NvidiaTeslaT4"
NOTEBOOK_NAME = "shadow-gold-context-alignment-v1.ipynb"
PERSISTENT_OUTPUTS = (
    "context_alignment_selected.json",
    "context_alignment_metrics.json",
    "context_alignment_predictions.jsonl",
    "context_alignment_runtime_manifest.json",
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
    "shadow_gold_context_alignment_v1.json": ("materialized/poisoned-chalice-shadow-gold-context-alignment-v1/shadow_gold_context_alignment_v1.json", CONTEXT_CONFIG_BLOB, 32768),
}
SOURCE_PARTS = (
    ("materialized/poisoned-chalice-shadow-gold-context-alignment-v1/shadow_gold_context_alignment.py.part01", "a44c1d3ee9552e4d4343b3e939f16499752cf431", 16384),
    ("materialized/poisoned-chalice-shadow-gold-context-alignment-v1/shadow_gold_context_alignment.py.part02", "c9fb1932458456d00612014b18e8400d8f72f613", 16384),
    ("materialized/poisoned-chalice-shadow-gold-context-alignment-v1/shadow_gold_context_alignment.py.part03", "67fdcd0f09b11eb35559dd263f308b7bc524cf46", 16384),
)


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _load_exact(path: Path, expected: str, maximum: int) -> bytes:
    if not path.is_file():
        raise RuntimeError(f"snapshot missing: {path}")
    data = path.read_bytes()
    if len(data) > maximum or blob_sha(data) != expected:
        raise RuntimeError(f"snapshot identity mismatch: {path}")
    return data


def load_snapshots(root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for logical, (relative, expected, maximum) in SNAPSHOTS.items():
        files[logical] = _load_exact(root / relative, expected, maximum)
    parts = [_load_exact(root / relative, expected, maximum) for relative, expected, maximum in SOURCE_PARTS]
    source = b"".join(parts)
    if blob_sha(source) != CONTEXT_RUNTIME_BLOB:
        raise RuntimeError("reconstructed context-alignment source blob mismatch")
    files["poisoned_chalice/shadow_gold_context_alignment.py"] = source
    return files


def validate_sources(files: dict[str, bytes]) -> None:
    for logical, data in files.items():
        if logical.endswith(".py"):
            compile(data.decode("utf-8"), logical, "exec")
    source = files["poisoned_chalice/shadow_gold_context_alignment.py"].decode("utf-8")
    required = (
        "run_context_alignment_benchmark",
        "build_training_aligned_suffix_plan",
        "bos_plus_last_254_payload_plus_eos_then_right_pad",
        'required_gpu_name_fragment="T4"',
        '"probe_boundary_used": False',
        '"stage2_v3_selection_allowed": False',
    )
    for marker in required:
        if marker not in source:
            raise RuntimeError(f"context-alignment source marker missing: {marker}")
    if "required_name_fragment=\"T4\"" in source.split("run_dual_gpu_shadow(", 1)[-1]:
        raise RuntimeError("historical dual-GPU keyword typo leaked into context-alignment runtime")


def validate_configs(positive: dict, aligned: dict) -> None:
    if positive.get("experiment_id") != "shadow-gold-positive-control-v1":
        raise RuntimeError("positive-control config identity changed")
    protocol = positive.get("protocol") or {}
    if (protocol.get("exposure_repeats"), protocol.get("expected_training_sequences"), protocol.get("expected_optimizer_steps_per_architecture")) != (16, 81920, 1280):
        raise RuntimeError("positive-control training contract changed")
    if aligned.get("execution_id") != EXECUTION_ID or aligned.get("role") != "gold_training_context_alignment_ablation":
        raise RuntimeError("context-alignment config identity changed")
    if (aligned.get("base_evidence") or {}).get("content_reference_result_commit") != CONTENT_REFERENCE_RESULT_COMMIT:
        raise RuntimeError("context-alignment content-reference evidence changed")
    if (aligned.get("base_evidence") or {}).get("probe_local_result_commit") != PROBE_LOCAL_RESULT_COMMIT:
        raise RuntimeError("context-alignment probe-local evidence changed")
    scoring = aligned.get("aligned_scoring") or {}
    frozen_scoring = {
        "version":"training-aligned-suffix-generic-v1",
        "max_sequence_tokens":256,
        "payload_capacity_tokens":254,
        "context_construction":"bos_plus_last_254_payload_plus_eos_then_right_pad",
        "score_payload_tokens_only":True,
        "score_eos":False,
        "probe_boundary_used":False,
        "probe_text_used":False,
        "language_used_to_locate_probe":False,
        "benchmark_id_used_as_model_input":False,
        "membership_used_for_scoring":False,
        "min_k_percents":[1,2,5,10],
        "local_widths":[8,16,32,64],
        "batch_size":64,
    }
    for key, value in frozen_scoring.items():
        if scoring.get(key) != value:
            raise RuntimeError(f"context-alignment scoring contract changed: {key}")
    candidates = [
        "aligned_loss_mean","aligned_min_k_01","aligned_min_k_02","aligned_min_k_05","aligned_min_k_10",
        "aligned_local08_max","aligned_local16_max","aligned_local32_max","aligned_local64_max",
    ]
    if aligned.get("candidate_features") != candidates:
        raise RuntimeError("context-alignment candidate matrix changed")
    recovery = aligned.get("recovery_criterion") or {}
    frozen_recovery = {
        "minimum_holdout_auc_each_architecture":0.56,
        "minimum_auc_gain_vs_raw_stage2_loss_max_each_architecture":0.03,
        "minimum_tpr_at_1pct_fpr_each_architecture":0.015,
        "both_architectures_required":True,
        "all_conditions_required":True,
        "failed_gate_is_valid_scientific_result_not_runtime_failure":True,
    }
    for key, value in frozen_recovery.items():
        if recovery.get(key) != value:
            raise RuntimeError(f"context-alignment recovery contract changed: {key}")
    guards = aligned.get("scientific_guards") or {}
    for key in ("known_probe_boundary_used","probe_text_used","probe_local_score_used_as_candidate","evaluation_labels_passed_to_training_children","stage2_v3_selection_allowed","competition_feature_promotion_allowed","external_model_holdout_consumed","fourth_external_holdout_consumed"):
        if guards.get(key) is not False:
            raise RuntimeError(f"context-alignment scientific guard changed: {key}")
    if guards.get("features_sealed_before_label_reveal") is not True or guards.get("competition_rows_used") != 0 or guards.get("external_rows_used") != 0:
        raise RuntimeError("context-alignment clean-room boundary changed")


def build_code(files: dict[str, bytes]) -> str:
    validate_sources(files)
    positive_text = files["shadow_gold_positive_control_v1.json"].decode("utf-8")
    aligned_text = files["shadow_gold_context_alignment_v1.json"].decode("utf-8")
    validate_configs(json.loads(positive_text), json.loads(aligned_text))
    payloads = {
        key: base64.b64encode(value).decode("ascii")
        for key, value in sorted(files.items())
        if not key.endswith(".json")
    }
    template = r'''from __future__ import annotations
import base64, json, os, shutil, sys
from pathlib import Path
EXECUTION_ID = __EXECUTION_ID__
RESEARCH_COMMIT = __RESEARCH_COMMIT__
CONTENT_REFERENCE_RESULT_COMMIT = __CONTENT_RESULT__
PROBE_LOCAL_RESULT_COMMIT = __PROBE_RESULT__
SOURCE_PAYLOADS = __PAYLOADS__
POSITIVE_CONFIG_TEXT = __POSITIVE__
ALIGNED_CONFIG_TEXT = __ALIGNED__
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
positive_config = json.loads(POSITIVE_CONFIG_TEXT); aligned_config = json.loads(ALIGNED_CONFIG_TEXT)
if positive_config['protocol']['exposure_repeats'] != 16 or positive_config['protocol']['expected_optimizer_steps_per_architecture'] != 1280: raise RuntimeError('training contract changed')
if aligned_config['aligned_scoring']['payload_capacity_tokens'] != 254 or aligned_config['aligned_scoring']['probe_boundary_used'] is not False: raise RuntimeError('aligned scorer contract changed')
if aligned_config['scientific_guards']['probe_text_used'] is not False or aligned_config['scientific_guards']['stage2_v3_selection_allowed'] is not False: raise RuntimeError('scientific guard changed')
sys.path.insert(0, str(runtime_root)); os.environ['PYTHONPATH'] = str(runtime_root) + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
from poisoned_chalice.shadow_gold_context_alignment import run_context_alignment_benchmark

def default(value):
    if hasattr(value, 'item'): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)
def write_json(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True, default=default) + '\n', encoding='utf-8')
def publish(source, destination):
    temporary = destination.with_suffix(destination.suffix + '.partial'); shutil.copyfile(source, temporary); os.replace(temporary, destination)
try:
    metrics, predictions, manifest = run_context_alignment_benchmark(scratch_directory=scratch/'benchmark', positive_control_config=positive_config, context_alignment_config=aligned_config, timeout_seconds=5400)
    if manifest.get('status') != 'complete' or metrics.get('status') != 'complete': raise RuntimeError('context-alignment benchmark did not complete')
    if manifest.get('training_protocol_changed_from_positive_control_v2') is not False: raise RuntimeError('training protocol changed')
    if manifest.get('training_context_alignment_used') is not True or manifest.get('features_sealed_before_label_reveal') is not True: raise RuntimeError('context/label boundary failed')
    if manifest.get('probe_boundary_used') is not False or manifest.get('probe_text_used') is not False: raise RuntimeError('probe-specific information leaked')
    if manifest.get('competition_rows_used') != 0 or manifest.get('external_rows_used') != 0: raise RuntimeError('clean-room row boundary changed')
    if manifest.get('pretrained_weights_used') is not False or manifest.get('automatic_compute_retries') != 0: raise RuntimeError('runtime guard changed')
    if manifest.get('stage2_v3_selection_allowed') is not False or manifest.get('competition_feature_promotion_allowed') is not False: raise RuntimeError('promotion guard failed')
    if len(predictions) != 2560: raise RuntimeError('prediction count changed')
    manifest = dict(manifest); manifest['research_commit'] = RESEARCH_COMMIT; manifest['content_reference_result_commit'] = CONTENT_REFERENCE_RESULT_COMMIT; manifest['probe_local_result_commit'] = PROBE_LOCAL_RESULT_COMMIT
    selected = {'selected_candidate':metrics['selected_candidate'], 'recovery_passed':metrics['recovery_passed'], 'recovery_checks':metrics['recovery_checks'], 'selected_auc_gain_vs_raw_stage2_loss_max':metrics['selected_auc_gain_vs_raw_stage2_loss_max'], 'raw_stage2_loss_max_holdout':metrics['raw_stage2_loss_max_holdout']}
    staged = scratch/'published'; staged.mkdir()
    write_json(staged/'context_alignment_selected.json', selected)
    write_json(staged/'context_alignment_metrics.json', metrics)
    predictions.to_json(staged/'context_alignment_predictions.jsonl', orient='records', lines=True, force_ascii=False)
    write_json(staged/'context_alignment_runtime_manifest.json', manifest)
    if {p.name for p in staged.iterdir()} != set(PERSISTENT_OUTPUTS): raise RuntimeError('staged output allowlist changed')
    for name in PERSISTENT_OUTPUTS: publish(staged/name, working/name)
    for path in list(working.iterdir()):
        if path.name not in PERSISTENT_OUTPUTS: shutil.rmtree(path) if path.is_dir() else path.unlink()
    if {p.name for p in working.iterdir()} != set(PERSISTENT_OUTPUTS): raise RuntimeError('persistent output allowlist changed')
    print(json.dumps({'execution_id':EXECUTION_ID, 'status':'complete', 'selected_candidate':metrics['selected_candidate'], 'recovery_passed':metrics['recovery_passed'], 'left_auc':metrics['selected_holdout']['left']['auc'], 'right_auc':metrics['selected_holdout']['right']['auc'], 'left_raw_loss_auc':metrics['raw_stage2_loss_max_holdout']['left']['auc'], 'right_raw_loss_auc':metrics['raw_stage2_loss_max_holdout']['right']['auc'], 'stage2_v3_selection_allowed':False}, sort_keys=True))
finally:
    shutil.rmtree(scratch, ignore_errors=True)
'''
    replacements = {
        "__EXECUTION_ID__":repr(EXECUTION_ID), "__RESEARCH_COMMIT__":repr(RESEARCH_COMMIT),
        "__CONTENT_RESULT__":repr(CONTENT_REFERENCE_RESULT_COMMIT), "__PROBE_RESULT__":repr(PROBE_LOCAL_RESULT_COMMIT),
        "__PAYLOADS__":repr(payloads), "__POSITIVE__":repr(positive_text), "__ALIGNED__":repr(aligned_text), "__OUTPUTS__":repr(PERSISTENT_OUTPUTS),
    }
    for marker, value in replacements.items():
        if marker not in template: raise RuntimeError(f"template marker missing: {marker}")
        template = template.replace(marker, value)
    compile(template, NOTEBOOK_NAME, "exec")
    return template


def validate_generated_code(code: str) -> None:
    tree = ast.parse(code)
    for marker in (RESEARCH_COMMIT, CONTENT_REFERENCE_RESULT_COMMIT, "run_context_alignment_benchmark", "training_context_alignment_used", "context_alignment_metrics.json", "stage2_v3_selection_allowed"):
        if marker not in code: raise RuntimeError(f"generated Notebook marker missing: {marker}")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_context_alignment_benchmark"]
    if len(calls) != 1: raise RuntimeError("generated Notebook must invoke benchmark exactly once")
    for forbidden in ("competitions submit", "kernels push", "RESEARCH_REPO_READ_TOKEN"):
        if forbidden in code: raise RuntimeError(f"forbidden Notebook marker: {forbidden}")


def materialize(snapshot_root: Path, kernel_dir: Path) -> None:
    if kernel_dir.exists() and any(kernel_dir.iterdir()): raise RuntimeError("kernel directory is not empty")
    files = load_snapshots(snapshot_root)
    code = build_code(files); validate_generated_code(code)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells":[
            {"cell_type":"markdown","id":"context-alignment-intro","metadata":{},"source":["# Gold training-context alignment ablation v1\n","Probe-boundary-free mechanistic calibration; not Stage2-v3 tuning.\n"]},
            {"cell_type":"code","id":"context-alignment-run","execution_count":None,"metadata":{},"outputs":[],"source":code.splitlines(keepends=True)},
        ],
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},
        "nbformat":4,"nbformat_minor":5,
    }
    (kernel_dir/NOTEBOOK_NAME).write_text(json.dumps(notebook, indent=2)+"\n", encoding="utf-8")
    metadata = {
        "id":TARGET,"title":"Gold training-context alignment ablation v1","code_file":NOTEBOOK_NAME,
        "language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_tpu":False,
        "enable_internet":False,"machine_shape":MACHINE_SHAPE,"dataset_sources":[],"kernel_sources":[],"competition_sources":[],"model_sources":[],
    }
    (kernel_dir/"kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    validate_bundle(kernel_dir)
    print("SHADOW_GOLD_CONTEXT_ALIGN_V1_MATERIALIZE PASS physical_snapshots=17 logical_snapshots=15 candidates=9 outputs=4 gpu=2 steps=1280 eval=5120 retries=0")


def validate_bundle(kernel_dir: Path) -> None:
    if {p.name for p in kernel_dir.iterdir() if p.is_file()} != {NOTEBOOK_NAME,"kernel-metadata.json"}: raise RuntimeError("kernel bundle allowlist changed")
    notebook=json.loads((kernel_dir/NOTEBOOK_NAME).read_text(encoding="utf-8")); metadata=json.loads((kernel_dir/"kernel-metadata.json").read_text(encoding="utf-8"))
    code="".join(notebook["cells"][1]["source"]); compile(code, NOTEBOOK_NAME, "exec"); validate_generated_code(code)
    expected={"id":TARGET,"code_file":NOTEBOOK_NAME,"kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":False,"machine_shape":MACHINE_SHAPE,"dataset_sources":[],"kernel_sources":[],"competition_sources":[],"model_sources":[]}
    for key,value in expected.items():
        if metadata.get(key)!=value: raise RuntimeError(f"kernel metadata mismatch: {key}")


def literal_commands(source: str) -> list[tuple[str,...]]:
    commands=[]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=="subprocess" and node.func.attr=="run" and node.args and isinstance(node.args[0],(ast.List,ast.Tuple)):
            commands.append(tuple(item.value if isinstance(item,ast.Constant) and isinstance(item.value,str) else "<dynamic>" for item in node.args[0].elts))
    return commands


def validate_static(path: Path) -> None:
    source=path.read_text(encoding="utf-8"); compile(source,str(path),"exec"); commands=literal_commands(source)
    def pair(command,left,right): return any(command[i:i+2]==(left,right) for i in range(len(command)-1))
    if sum(pair(command,"kernels","push") for command in commands)!=1: raise RuntimeError("launcher must contain exactly one kernels push")
    for forbidden in (("competitions","submit"),("datasets","create"),("datasets","version"),("models","create"),("models","version"),("kernels","delete"),("kernels","cancel")):
        if any(pair(command,*forbidden) for command in commands): raise RuntimeError(f"forbidden write: {' '.join(forbidden)}")


def execute(kaggle_bin: Path, kernel_dir: Path) -> None:
    validate_bundle(kernel_dir)
    result=subprocess.run([str(kaggle_bin),"kernels","push","-p",str(kernel_dir),"--timeout","3600"],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    if result.returncode!=0: raise RuntimeError(f"Kaggle context-alignment push returned nonzero ({result.returncode}); outcome ambiguous and retry forbidden")
    print("SHADOW_GOLD_CONTEXT_ALIGN_V1_LAUNCH_ACCEPTED target=renta0426/shadow-gold-context-alignment-v1 accelerator=gpu machine=NvidiaTeslaT4 retries=0 submissions=0 private_repo_access=0 post_push_getkernel=0")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--launcher",type=Path); parser.add_argument("--snapshot-root",type=Path); parser.add_argument("--kernel-dir",type=Path); parser.add_argument("--kaggle-bin",type=Path); parser.add_argument("--static",action="store_true"); parser.add_argument("--materialize",action="store_true"); parser.add_argument("--execute",action="store_true"); args=parser.parse_args()
    if sum((args.static,args.materialize,args.execute))!=1: raise SystemExit("select exactly one operation")
    if args.static:
        if args.launcher is None: raise SystemExit("--launcher required")
        validate_static(args.launcher); print("SHADOW_GOLD_CONTEXT_ALIGN_V1_STATIC PASS write_calls=1 retries=0 submissions=0 outputs=4"); return
    if args.materialize:
        if args.snapshot_root is None or args.kernel_dir is None: raise SystemExit("--snapshot-root and --kernel-dir required")
        materialize(args.snapshot_root,args.kernel_dir); return
    if args.kaggle_bin is None or args.kernel_dir is None: raise SystemExit("--kaggle-bin and --kernel-dir required")
    execute(args.kaggle_bin,args.kernel_dir)


if __name__ == "__main__":
    main()
