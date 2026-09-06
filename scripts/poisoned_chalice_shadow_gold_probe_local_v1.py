"""Launch the frozen Gold probe-local sensitivity calibration on Kaggle T4 x2."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

EXECUTION_ID = "shadow-gold-probe-local-v1"
REQUEST_ID = "20260906-poisoned-chalice-shadow-gold-probe-local-v1-001"
TARGET = "renta0426/shadow-gold-probe-local-v1"
RESEARCH_COMMIT = "80c2ebb548bce6aad6381c378d88a59135ed5bf3"
POSITIVE_CONTROL_RESULT_COMMIT = "9c4a837d45e2731783d65db954cdca2aa4329441"
POSITIVE_CONTROL_SCIENCE_COMMIT = "4a0e00d96964b1203447c944d803455fd2c7b890"
POSITIVE_CONTROL_REPAIR_COMMIT = "a98a088b1369d02325884296526eeec797ddc97b"
MACHINE_SHAPE = "NvidiaTeslaT4"
NOTEBOOK_NAME = "shadow-gold-probe-local-v1.ipynb"
PERSISTENT_OUTPUTS = (
    "probe_local_generic_attack.json",
    "probe_local_metrics.json",
    "probe_local_predictions.jsonl",
    "probe_local_runtime_manifest.json",
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
    "poisoned_chalice/shadow_gold_probe_local.py": ("materialized/poisoned-chalice-shadow-gold-probe-local-v1/poisoned_chalice/shadow_gold_probe_local.py", "b1eb555a631485659cb4202c3345c7a14b472b1d", 65536),
    "shadow_gold_probe_local_v1.json": ("materialized/poisoned-chalice-shadow-gold-probe-local-v1/shadow_gold_probe_local_v1.json", "da2377a38e7ee013c05c025bc2a78e6a255b34ff", 32768),
}


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def load_snapshots(root: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for logical, (bridge_path, expected_blob, maximum) in SNAPSHOTS.items():
        path = root / bridge_path
        if not path.is_file():
            raise RuntimeError(f"snapshot missing: {bridge_path}")
        data = path.read_bytes()
        if len(data) > maximum or blob_sha(data) != expected_blob:
            raise RuntimeError(f"snapshot identity mismatch: {bridge_path}")
        result[logical] = data
    return result


def validate_sources(files: dict[str, bytes]) -> None:
    for logical, data in files.items():
        if logical.endswith(".py"):
            compile(data.decode("utf-8"), logical, "exec")
    source = files["poisoned_chalice/shadow_gold_probe_local.py"].decode("utf-8")
    for marker in (
        "run_probe_local_benchmark",
        "probe_digest_logp_mean",
        "probe_scores_sealed_before_label_reveal",
        "probe_local_score_may_become_stage2_feature",
        "build_probe_token_plan",
    ):
        if marker not in source:
            raise RuntimeError(f"probe-local source marker missing: {marker}")


def validate_configs(positive: dict, probe: dict) -> None:
    if positive.get("experiment_id") != "shadow-gold-positive-control-v1":
        raise RuntimeError("positive-control base config identity changed")
    protocol = positive.get("protocol") or {}
    if protocol.get("exposure_repeats") != 16 or protocol.get("expected_training_sequences") != 81920 or protocol.get("expected_optimizer_steps_per_architecture") != 1280:
        raise RuntimeError("positive-control training protocol changed")
    if probe.get("execution_id") != EXECUTION_ID or probe.get("role") != "gold_positive_control_probe_local_calibration":
        raise RuntimeError("probe-local config identity changed")
    scoring = probe.get("probe_local_scoring") or {}
    expected_scoring = {
        "version": "gold-probe-local-byte-logp-v1",
        "primary_predictor": "probe_digest_logp_mean",
        "secondary_predictor": "probe_full_logp_mean",
        "digest_token_count": 16,
        "probe_boundary_derived_from_content_and_language_only": True,
        "membership_used_for_scoring": False,
        "benchmark_id_used_as_model_input": False,
        "score_eos": False,
        "batch_size": 64,
    }
    for key, value in expected_scoring.items():
        if scoring.get(key) != value:
            raise RuntimeError(f"probe-local scoring contract changed: {key}")
    gate = probe.get("sensitivity_gate") or {}
    expected_gate = {
        "primary_predictor": "probe_digest_logp_mean",
        "shadow_a_holdout_min_auc": 0.60,
        "shadow_b_blind_holdout_min_auc": 0.60,
        "shadow_a_holdout_min_tpr_at_1pct_fpr": 0.03,
        "shadow_b_blind_holdout_min_tpr_at_1pct_fpr": 0.03,
        "all_four_thresholds_required": True,
        "secondary_predictor_may_rescue_failed_primary_gate": False,
        "failed_gate_is_valid_scientific_result_not_runtime_failure": True,
    }
    for key, value in expected_gate.items():
        if gate.get(key) != value:
            raise RuntimeError(f"probe-local gate changed: {key}")
    guards = probe.get("scientific_guards") or {}
    for key in ("stage2_v3_selection_allowed", "probe_local_score_may_become_stage2_feature", "external_model_holdout_consumed", "fourth_external_holdout_consumed"):
        if guards.get(key) is not False:
            raise RuntimeError(f"probe-local scientific guard changed: {key}")
    if guards.get("competition_rows_used") != 0 or guards.get("external_rows_used") != 0:
        raise RuntimeError("probe-local clean-room row contract changed")


def build_notebook_code(files: dict[str, bytes]) -> str:
    validate_sources(files)
    positive_text = files["shadow_gold_positive_control_v1.json"].decode("utf-8")
    probe_text = files["shadow_gold_probe_local_v1.json"].decode("utf-8")
    validate_configs(json.loads(positive_text), json.loads(probe_text))
    payloads = {path: base64.b64encode(data).decode("ascii") for path, data in sorted(files.items()) if not path.endswith(".json")}
    template = r'''from __future__ import annotations
import base64, json, os, shutil, sys
from pathlib import Path

EXECUTION_ID = __EXECUTION_ID__
RESEARCH_COMMIT = __RESEARCH_COMMIT__
POSITIVE_CONTROL_RESULT_COMMIT = __RESULT_COMMIT__
SOURCE_PAYLOADS = __SOURCE_PAYLOADS__
POSITIVE_CONFIG_TEXT = __POSITIVE_CONFIG_TEXT__
PROBE_CONFIG_TEXT = __PROBE_CONFIG_TEXT__
PERSISTENT_OUTPUTS = __PERSISTENT_OUTPUTS__

working = Path('/kaggle/working')
working.mkdir(parents=True, exist_ok=True)
for path in list(working.iterdir()):
    shutil.rmtree(path) if path.is_dir() else path.unlink()
scratch = Path('/tmp') / EXECUTION_ID
if scratch.exists():
    shutil.rmtree(scratch)
scratch.mkdir(parents=True)
runtime_root = scratch / 'runtime'
for relative, encoded in SOURCE_PAYLOADS.items():
    destination = runtime_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(encoded))
(runtime_root / 'poisoned_chalice/__init__.py').write_text('', encoding='utf-8')
positive_config = json.loads(POSITIVE_CONFIG_TEXT)
probe_config = json.loads(PROBE_CONFIG_TEXT)
if positive_config.get('protocol', {}).get('exposure_repeats') != 16:
    raise RuntimeError('probe-local base exposure count changed')
if positive_config.get('protocol', {}).get('expected_optimizer_steps_per_architecture') != 1280:
    raise RuntimeError('probe-local base optimizer-step count changed')
if probe_config.get('probe_local_scoring', {}).get('primary_predictor') != 'probe_digest_logp_mean':
    raise RuntimeError('probe-local primary predictor changed')
if probe_config.get('scientific_guards', {}).get('stage2_v3_selection_allowed') is not False:
    raise RuntimeError('probe-local Stage2-v3 guard changed')
if probe_config.get('scientific_guards', {}).get('probe_local_score_may_become_stage2_feature') is not False:
    raise RuntimeError('probe-local promotion guard changed')
sys.path.insert(0, str(runtime_root))
existing_pythonpath = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = str(runtime_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else '')
from poisoned_chalice.shadow_gold_probe_local import run_probe_local_benchmark


def json_default(value):
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + '\n', encoding='utf-8')


def publish_atomic(source, destination):
    temporary = destination.with_suffix(destination.suffix + '.partial')
    if temporary.exists():
        temporary.unlink()
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)

try:
    generic_attack, metrics, predictions, manifest = run_probe_local_benchmark(
        scratch_directory=scratch / 'benchmark',
        positive_control_config=positive_config,
        probe_local_config=probe_config,
        timeout_seconds=5400,
    )
    if manifest.get('status') != 'complete' or metrics.get('status') != 'complete':
        raise RuntimeError('probe-local calibration did not complete')
    gate = metrics.get('probe_local_gate') or {}
    if gate.get('status') not in {'pass', 'fail'} or gate.get('primary_predictor') != 'probe_digest_logp_mean':
        raise RuntimeError('probe-local gate result missing')
    if manifest.get('probe_local_gate_status') != gate.get('status'):
        raise RuntimeError('probe-local gate status mismatch')
    if manifest.get('training_protocol_changed_from_positive_control_v2') is not False:
        raise RuntimeError('probe-local training protocol changed')
    if manifest.get('probe_scores_sealed_before_label_reveal') is not True:
        raise RuntimeError('probe-local label boundary failed')
    if manifest.get('stage2_v3_selection_allowed') is not False or manifest.get('probe_local_score_may_become_stage2_feature') is not False:
        raise RuntimeError('probe-local scientific guard failed')
    if manifest.get('competition_rows_used') != 0 or manifest.get('external_rows_used') != 0:
        raise RuntimeError('probe-local clean-room rows changed')
    if len(predictions) != 2560:
        raise RuntimeError(f'probe-local prediction count changed: {len(predictions)}')
    manifest = dict(manifest)
    manifest['research_commit'] = RESEARCH_COMMIT
    manifest['positive_control_result_commit'] = POSITIVE_CONTROL_RESULT_COMMIT
    staged = scratch / 'published'
    staged.mkdir()
    write_json(staged / 'probe_local_generic_attack.json', generic_attack)
    write_json(staged / 'probe_local_metrics.json', metrics)
    predictions.to_json(staged / 'probe_local_predictions.jsonl', orient='records', lines=True, force_ascii=False)
    write_json(staged / 'probe_local_runtime_manifest.json', manifest)
    if {p.name for p in staged.iterdir()} != set(PERSISTENT_OUTPUTS):
        raise RuntimeError('probe-local staged output allowlist changed')
    for name in PERSISTENT_OUTPUTS:
        publish_atomic(staged / name, working / name)
    for path in list(working.iterdir()):
        if path.name not in PERSISTENT_OUTPUTS:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    if {p.name for p in working.iterdir()} != set(PERSISTENT_OUTPUTS):
        raise RuntimeError('probe-local persistent output allowlist changed')
    observed = gate['metrics']
    print(json.dumps({
        'execution_id': EXECUTION_ID,
        'status': 'complete',
        'probe_local_gate_status': gate['status'],
        'primary_predictor': 'probe_digest_logp_mean',
        'shadow_a_auc': observed['left']['probe_digest_logp_mean']['auc'],
        'shadow_b_auc': observed['right']['probe_digest_logp_mean']['auc'],
        'shadow_a_tpr_1pct': observed['left']['probe_digest_logp_mean']['tpr_at_fpr'],
        'shadow_b_tpr_1pct': observed['right']['probe_digest_logp_mean']['tpr_at_fpr'],
        'stage2_v3_selection_allowed': False,
    }, sort_keys=True))
except Exception:
    for name in PERSISTENT_OUTPUTS:
        path = working / name
        if path.exists():
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    raise
finally:
    shutil.rmtree(scratch, ignore_errors=True)
'''
    replacements = {
        "__EXECUTION_ID__": repr(EXECUTION_ID),
        "__RESEARCH_COMMIT__": repr(RESEARCH_COMMIT),
        "__RESULT_COMMIT__": repr(POSITIVE_CONTROL_RESULT_COMMIT),
        "__SOURCE_PAYLOADS__": repr(payloads),
        "__POSITIVE_CONFIG_TEXT__": repr(positive_text),
        "__PROBE_CONFIG_TEXT__": repr(probe_text),
        "__PERSISTENT_OUTPUTS__": repr(PERSISTENT_OUTPUTS),
    }
    for marker, value in replacements.items():
        if marker not in template:
            raise RuntimeError(f"Notebook template marker missing: {marker}")
        template = template.replace(marker, value)
    compile(template, NOTEBOOK_NAME, "exec")
    return template


def validate_generated_code(code: str) -> None:
    tree = ast.parse(code)
    for marker in (
        RESEARCH_COMMIT,
        POSITIVE_CONTROL_RESULT_COMMIT,
        "run_probe_local_benchmark",
        "probe_digest_logp_mean",
        "probe_scores_sealed_before_label_reveal",
        "probe_local_score_may_become_stage2_feature",
        "probe_local_metrics.json",
        "timeout_seconds=5400",
    ):
        if marker not in code:
            raise RuntimeError(f"generated probe-local Notebook marker missing: {marker}")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_probe_local_benchmark"]
    if len(calls) != 1:
        raise RuntimeError("generated probe-local Notebook must invoke benchmark exactly once")
    for forbidden in ("competitions submit", "kernels push", "RESEARCH_REPO_READ_TOKEN"):
        if forbidden in code:
            raise RuntimeError(f"forbidden generated-code marker: {forbidden}")


def materialize(snapshot_root: Path, kernel_dir: Path) -> None:
    if kernel_dir.exists() and any(kernel_dir.iterdir()):
        raise RuntimeError("kernel directory already exists and is not empty")
    files = load_snapshots(snapshot_root)
    validate_sources(files)
    code = build_notebook_code(files)
    validate_generated_code(code)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type":"markdown","id":"gold-probe-local-intro","metadata":{},"source":["# Gold probe-local calibration v1\n","Calibration-only digest log-likelihood; not a Stage2 feature.\n"]},
            {"cell_type":"code","id":"gold-probe-local-run","execution_count":None,"metadata":{},"outputs":[],"source":code.splitlines(keepends=True)},
        ],
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},
        "nbformat":4,"nbformat_minor":5,
    }
    (kernel_dir / NOTEBOOK_NAME).write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "id": TARGET,
        "title": "Gold probe-local calibration v1",
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": MACHINE_SHAPE,
        "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_bundle(kernel_dir)
    print("SHADOW_GOLD_PROBE_LOCAL_V1_MATERIALIZE PASS snapshots=15 outputs=4 gpu=2 steps=1280 eval=5120 primary=digest retries=0")


def validate_bundle(kernel_dir: Path) -> None:
    if {p.name for p in kernel_dir.iterdir() if p.is_file()} != {"kernel-metadata.json", NOTEBOOK_NAME}:
        raise RuntimeError("probe-local kernel bundle allowlist changed")
    notebook = json.loads((kernel_dir / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][1]["source"])
    compile(code, NOTEBOOK_NAME, "exec")
    validate_generated_code(code)
    expected = {"id":TARGET,"code_file":NOTEBOOK_NAME,"kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":False,"machine_shape":MACHINE_SHAPE,"dataset_sources":[],"kernel_sources":[],"competition_sources":[],"model_sources":[]}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"probe-local metadata mismatch: {key}")


def literal_commands(source: str) -> list[tuple[str, ...]]:
    commands = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess" and node.func.attr == "run" and node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
            commands.append(tuple(item.value if isinstance(item, ast.Constant) and isinstance(item.value, str) else "<dynamic>" for item in node.args[0].elts))
    return commands


def validate_static(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    commands = literal_commands(source)
    def pair(command, left, right):
        return any(command[index:index+2] == (left, right) for index in range(len(command)-1))
    if sum(pair(command, "kernels", "push") for command in commands) != 1:
        raise RuntimeError("probe-local launcher must contain exactly one kernels push")
    for forbidden in (("competitions","submit"),("datasets","create"),("datasets","version"),("models","create"),("models","version"),("kernels","delete"),("kernels","cancel")):
        if any(pair(command, *forbidden) for command in commands):
            raise RuntimeError(f"forbidden write: {' '.join(forbidden)}")


def execute(kaggle_bin: Path, kernel_dir: Path) -> None:
    validate_bundle(kernel_dir)
    result = subprocess.run([str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir), "--timeout", "3600"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle probe-local push returned nonzero ({result.returncode}); outcome ambiguous and retry forbidden")
    print("SHADOW_GOLD_PROBE_LOCAL_V1_LAUNCH_ACCEPTED target=renta0426/shadow-gold-probe-local-v1 accelerator=gpu machine=NvidiaTeslaT4 retries=0 submissions=0 private_repo_access=0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path); parser.add_argument("--snapshot-root", type=Path); parser.add_argument("--kernel-dir", type=Path); parser.add_argument("--kaggle-bin", type=Path)
    parser.add_argument("--static", action="store_true"); parser.add_argument("--materialize", action="store_true"); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if sum((args.static, args.materialize, args.execute)) != 1:
        raise SystemExit("select exactly one operation")
    if args.static:
        if args.launcher is None: raise SystemExit("--launcher required")
        validate_static(args.launcher); print("SHADOW_GOLD_PROBE_LOCAL_V1_STATIC PASS write_calls=1 retries=0 submissions=0 outputs=4"); return
    if args.materialize:
        if args.snapshot_root is None or args.kernel_dir is None: raise SystemExit("--snapshot-root and --kernel-dir required")
        materialize(args.snapshot_root, args.kernel_dir); return
    if args.kaggle_bin is None or args.kernel_dir is None: raise SystemExit("--kaggle-bin and --kernel-dir required")
    execute(args.kaggle_bin, args.kernel_dir)


if __name__ == "__main__":
    main()
