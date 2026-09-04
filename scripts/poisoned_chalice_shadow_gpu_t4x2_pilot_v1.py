"""Self-contained launcher for the controlled-shadow Kaggle T4 x2 pilot.

This is the GPU replacement for the terminated TPU compatibility line.  The
protected job consumes exact public bridge snapshots only; the generated private
Notebook uses the unchanged CUDA training runtime and assigns one architecture
to each of two visible T4 GPUs without DDP/NCCL.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

EXPERIMENT_ID = "shadow-gpu-t4x2-pilot-v1"
REQUEST_ID = "20260904-poisoned-chalice-shadow-gpu-t4x2-pilot-v1-001"
TARGET = "renta0426/shadow-gpu-t4x2-pilot-v1"
RESEARCH_COMMIT = "251e9f0b7063dc2288a4fa258b9a314963f6ad19"
MACHINE_SHAPE = "NvidiaTeslaT4"
SUMMARY_NAME = "shadow_gpu_pilot_manifest.json"
NOTEBOOK_NAME = "shadow-gpu-t4x2-pilot-v1.ipynb"

SNAPSHOTS = {
    "poisoned_chalice/shadow_protocol.py": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v1/poisoned_chalice/shadow_protocol.py",
        "1c47b80696050aa2e5e7c62384617df61ecb80da", 131072,
    ),
    "poisoned_chalice/shadow_training.py": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v1/poisoned_chalice/shadow_training.py",
        "2fe3cf2079659ae10e1ebe0d5973f3ce96c26a03", 131072,
    ),
    "poisoned_chalice/shadow_training_dual_gpu.py": (
        "materialized/poisoned-chalice-shadow-gpu-t4x2-pilot-v1/poisoned_chalice/shadow_training_dual_gpu.py",
        "c75a5963fa930916024cb16b4fbf562ba4afa818", 65536,
    ),
    "shadow_gpu_t4x2_pilot_v1.json": (
        "materialized/poisoned-chalice-shadow-gpu-t4x2-pilot-v1/shadow_gpu_t4x2_pilot_v1.json",
        "1edcea3100f1d873ee015da44b2fdba002424476", 32768,
    ),
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


def build_notebook_code(files: dict[str, bytes]) -> str:
    payloads = {
        path: base64.b64encode(data).decode("ascii")
        for path, data in sorted(files.items())
        if path != "shadow_gpu_t4x2_pilot_v1.json"
    }
    config_text = files["shadow_gpu_t4x2_pilot_v1.json"].decode("utf-8")
    template = r'''from __future__ import annotations
import base64, hashlib, importlib.metadata, json, math, os, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

EXPERIMENT_ID = __EXPERIMENT_ID__
RESEARCH_COMMIT = __RESEARCH_COMMIT__
SUMMARY_NAME = __SUMMARY_NAME__
SOURCE_PAYLOADS = __SOURCE_PAYLOADS__
CONFIG_TEXT = __CONFIG_TEXT__

working = Path('/kaggle/working')
scratch = Path('/tmp') / EXPERIMENT_ID
if scratch.exists():
    shutil.rmtree(scratch)
scratch.mkdir(parents=True)
runtime_root = scratch / 'runtime'
for relative, encoded in SOURCE_PAYLOADS.items():
    destination = runtime_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(encoded))
(runtime_root / 'poisoned_chalice/__init__.py').write_text('', encoding='utf-8')

config = json.loads(CONFIG_TEXT)
if config.get('experiment_id') != EXPERIMENT_ID:
    raise RuntimeError('GPU pilot config ID changed')
if config.get('transition_from') != 'shadow-tpu-pilot-v6':
    raise RuntimeError('GPU transition parent changed')
runtime = config.get('runtime') or {}
if runtime.get('backend') != 'cuda' or runtime.get('accelerator') != 'gpu':
    raise RuntimeError('GPU runtime backend changed')
if runtime.get('kaggle_machine_shape') != 'NvidiaTeslaT4':
    raise RuntimeError('Kaggle GPU machine shape changed')
if runtime.get('expected_visible_gpu_count') != 2:
    raise RuntimeError('visible GPU count contract changed')
if runtime.get('assignment') != {'left': 0, 'right': 1}:
    raise RuntimeError('GPU architecture assignment changed')
if runtime.get('distributed_training') is not False or runtime.get('ddp_used') is not False or runtime.get('nccl_used') is not False:
    raise RuntimeError('distributed GPU training is forbidden in this pilot')
if runtime.get('per_architecture_world_size') != 1 or runtime.get('max_steps_per_architecture') != 2:
    raise RuntimeError('per-architecture CUDA runtime changed')
if runtime.get('checkpoint_reload_atol') != 1e-5:
    raise RuntimeError('checkpoint reload tolerance changed')
if config['data']['synthetic_training_rows'] != 128:
    raise RuntimeError('synthetic row count changed')

sys.path.insert(0, str(runtime_root))
existing_pythonpath = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = str(runtime_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else '')
from poisoned_chalice.shadow_protocol import ByteCodeTokenizer, ShadowSequenceConfig, ShadowTrainingSpec, build_shadow_protocol_manifest, build_shadow_training_sequences, default_shadow_pair
from poisoned_chalice.shadow_training_dual_gpu import run_dual_gpu_shadow, validate_gpu_inventory, visible_cuda_device_names

# Fail before protocol/model construction unless the expected two-T4 inventory is present.
gpu_names = visible_cuda_device_names()
validate_gpu_inventory(gpu_names, expected_count=2, required_name_fragment='T4')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_content(index):
    if index % 2 == 0:
        return f'def pilot_function_{index:03d}(value):\n    shifted = value + {index + 11}\n    return shifted * {index % 17 + 1}\n'
    return f'fn pilot_function_{index:03d}(value: i64) -> i64 {{\n    let shifted = value + {index + 11};\n    shifted * {index % 17 + 1}\n}}\n'


protocol_dir = scratch / 'protocol'
protocol_dir.mkdir()
train = pd.DataFrame({
    'benchmark_id': [f'tpu-pilot-{i:04d}' for i in range(128)],
    'content': [synthetic_content(i) for i in range(128)],
})
schedule = pd.DataFrame({
    'benchmark_id': train['benchmark_id'].tolist(),
    'exposure_round': [0] * 128,
    'sequence_index': list(range(128)),
})
tokenizer = ByteCodeTokenizer()
sequence_config = ShadowSequenceConfig(max_sequence_tokens=256)
training_spec = ShadowTrainingSpec(
    seed=2027,
    learning_rate=3e-4,
    weight_decay=0.1,
    warmup_fraction=0.05,
    global_batch_size=64,
    gradient_clip_norm=1.0,
)
sequences = build_shadow_training_sequences(
    train,
    schedule,
    tokenizer=tokenizer,
    sequence_config=sequence_config,
    seed=2027,
)
left, right = default_shadow_pair(max_position_embeddings=256)
protocol = build_shadow_protocol_manifest(
    left=left,
    right=right,
    sequence_config=sequence_config,
    training_spec=training_spec,
    training_sequences=sequences,
    tokenizer=tokenizer,
    max_parameter_ratio=1.15,
)
input_ids = np.asarray(sequences.input_ids.tolist(), dtype=np.uint16)
if input_ids.shape != (128, 256):
    raise RuntimeError(f'pilot input shape mismatch: {input_ids.shape}')
input_path = protocol_dir / 'training_input_ids.npy'
with input_path.open('wb') as handle:
    np.save(handle, input_ids, allow_pickle=False)
meta_path = protocol_dir / 'training_sequence_metadata.jsonl'
meta_cols = [
    'benchmark_id', 'exposure_round', 'sequence_index', 'source_token_count',
    'window_start', 'window_name', 'selected_payload_tokens',
]
with meta_path.open('w', encoding='utf-8', newline='\n') as handle:
    for row in sequences[meta_cols].itertuples(index=False, name=None):
        handle.write(json.dumps(dict(zip(meta_cols, row)), sort_keys=True, separators=(',', ':'), default=int) + '\n')
tokenizer.save_pretrained(protocol_dir)
manifest = {
    'status': 'frozen',
    'operation': 'freeze_synthetic_shadow_gpu_compatibility_protocol',
    'experiment_id': EXPERIMENT_ID,
    'research_commit': RESEARCH_COMMIT,
    'synthetic_compatibility_only': True,
    'protocol': protocol,
    'training_input_shape': list(input_ids.shape),
    'training_input_dtype': str(input_ids.dtype),
    'pad_token_id': tokenizer.pad_token_id,
    'attention_mask_rule': 'training_input_ids != pad_token_id',
    'label_rule': 'input id where attention=1, otherwise -100',
    'output_sha256': {
        'training_input_ids.npy': sha256_file(input_path),
        'training_sequence_metadata.jsonl': sha256_file(meta_path),
        'byte_tokenizer.json': sha256_file(protocol_dir / 'byte_tokenizer.json'),
    },
    'optimiser_steps_performed': 0,
    'model_compute_started': False,
    'accelerator_selected': False,
    'kaggle_operation_performed': False,
}
(protocol_dir / 'shadow_protocol_manifest.json').write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)

versions = {}
for package in ('torch', 'transformers', 'numpy', 'pandas'):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None

started = time.time()
run_error = None
result = None
try:
    result = run_dual_gpu_shadow(
        protocol_dir,
        scratch / 'training',
        max_steps=2,
        checkpoint_reload_atol=1e-5,
        parameter_sync_atol=1e-5,
        expected_gpu_count=2,
        required_gpu_name_fragment='T4',
        timeout_seconds=900,
    )
except Exception as exc:
    run_error = {
        'type': type(exc).__name__,
        'message': str(exc)[-10000:],
    }

acceptance = {'left': False, 'right': False}
if isinstance(result, dict) and result.get('status') == 'complete':
    for slot in ('left', 'right'):
        value = result.get(slot) or {}
        finite_loss = all(
            isinstance(value.get(key), (int, float)) and math.isfinite(float(value[key]))
            for key in ('initial_loss', 'final_loss', 'minimum_loss', 'maximum_loss')
        )
        acceptance[slot] = bool(
            value.get('status') == 'complete'
            and value.get('backend') == 'cuda'
            and value.get('world_size') == 1
            and value.get('completed_steps') == 2
            and finite_loss
            and value.get('checkpoint_reload_passed') is True
            and float(value.get('checkpoint_reload_max_absolute_logit_difference', float('inf'))) <= 1e-5
            and value.get('random_initialisation') is True
            and value.get('pretrained_weights_used') is False
            and value.get('evaluation_inputs_read') is False
            and value.get('evaluation_labels_read') is False
            and value.get('automatic_compute_retries') == 0
        )
summary = {
    'schema_version': 1,
    'experiment_id': EXPERIMENT_ID,
    'status': 'pass' if all(acceptance.values()) else 'fail',
    'purpose': 'CUDA/T4 x2 compatibility only; not Stage2-v3 selection evidence',
    'transition_from': 'shadow-tpu-pilot-v6',
    'research_commit': RESEARCH_COMMIT,
    'config_sha256': hashlib.sha256(CONFIG_TEXT.encode()).hexdigest(),
    'backend': 'cuda',
    'machine_shape': 'NvidiaTeslaT4',
    'gpu_names': gpu_names,
    'expected_visible_gpu_count': 2,
    'per_architecture_world_size': 1,
    'distributed_training': False,
    'ddp_used': False,
    'nccl_used': False,
    'architecture_gpu_assignment': {'left': 0, 'right': 1},
    'synthetic_training_rows': 128,
    'max_sequence_tokens': 256,
    'global_batch_size': 64,
    'max_steps_per_architecture': 2,
    'package_versions': versions,
    'acceptance': acceptance,
    'architecture_results': result,
    'error': run_error,
    'evaluation_inputs_read': False,
    'evaluation_labels_read': False,
    'smollm2_labels_used': False,
    'competition_train_rows_used': 0,
    'stage2_v3_selection_allowed': False,
    'automatic_compute_retries': 0,
    'protected_job_private_repository_access': False,
    'total_runtime_seconds': round(time.time() - started, 3),
    'persistent_outputs': [SUMMARY_NAME],
}
(working / SUMMARY_NAME).write_text(
    json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)
for path in working.iterdir():
    if path.name != SUMMARY_NAME:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
shutil.rmtree(scratch, ignore_errors=True)
print(json.dumps({
    'experiment_id': EXPERIMENT_ID,
    'status': summary['status'],
    'acceptance': acceptance,
    'gpu_count': len(gpu_names),
}, sort_keys=True))
if summary['status'] != 'pass':
    raise RuntimeError('shadow GPU T4 x2 compatibility pilot failed acceptance gates; do not retry automatically')
'''
    replacements = {
        "__EXPERIMENT_ID__": repr(EXPERIMENT_ID),
        "__RESEARCH_COMMIT__": repr(RESEARCH_COMMIT),
        "__SUMMARY_NAME__": repr(SUMMARY_NAME),
        "__SOURCE_PAYLOADS__": repr(payloads),
        "__CONFIG_TEXT__": repr(config_text),
    }
    for marker, value in replacements.items():
        if marker not in template:
            raise RuntimeError(f"Notebook template marker missing: {marker}")
        template = template.replace(marker, value)
    compile(template, NOTEBOOK_NAME, "exec")
    return template


def validate_generated_code(code: str) -> None:
    tree = ast.parse(code)
    required = (
        RESEARCH_COMMIT,
        "shadow-gpt2-byte-5m",
        "shadow-llama-byte-5m",
        "CUDA_VISIBLE_DEVICES",
        "expected_gpu_count=2",
        "required_gpu_name_fragment='T4'",
        "distributed_training': False",
        "ddp_used': False",
        "nccl_used': False",
        "os.environ['PYTHONPATH']",
        "checkpoint_reload_atol=1e-5",
        SUMMARY_NAME,
    )
    for marker in required:
        if marker not in code:
            raise RuntimeError(f"generated Notebook marker missing: {marker}")
    forbidden = (
        "torch_xla",
        "PJRT_DEVICE",
        "TPU_PROCESS_ADDRESSES",
        "shadow-xla-compat",
        "shadow-xla-spawn-safe",
        "shadow-kaggle-tpu-pjrt-env",
        "torch.distributed",
        "DistributedDataParallel",
        "torchrun",
    )
    for marker in forbidden:
        if marker in code:
            raise RuntimeError(f"TPU/distributed marker leaked into GPU Notebook: {marker}")
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_dual_gpu_shadow"
    ]
    if len(calls) != 1:
        raise RuntimeError("generated Notebook must invoke run_dual_gpu_shadow exactly once")


def materialize(snapshot_root: Path, kernel_dir: Path) -> None:
    if kernel_dir.exists() and any(kernel_dir.iterdir()):
        raise RuntimeError("kernel directory already exists and is not empty")
    files = load_snapshots(snapshot_root)
    config = json.loads(files["shadow_gpu_t4x2_pilot_v1.json"])
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("transition_from") != "shadow-tpu-pilot-v6":
        raise RuntimeError("GPU pilot identity changed")
    runtime = config.get("runtime") or {}
    expected_runtime = {
        "backend": "cuda",
        "accelerator": "gpu",
        "kaggle_machine_shape": MACHINE_SHAPE,
        "expected_visible_gpu_count": 2,
        "per_architecture_world_size": 1,
        "distributed_training": False,
        "ddp_used": False,
        "nccl_used": False,
        "run_architectures_concurrently": True,
        "max_steps_per_architecture": 2,
        "automatic_compute_retries": 0,
        "notebook_internet": False,
    }
    for key, value in expected_runtime.items():
        if runtime.get(key) != value:
            raise RuntimeError(f"GPU runtime contract changed: {key}")
    if runtime.get("assignment") != {"left": 0, "right": 1}:
        raise RuntimeError("GPU assignment changed")
    if runtime.get("checkpoint_reload_atol") != 1e-5:
        raise RuntimeError("GPU checkpoint tolerance changed")

    code = build_notebook_code(files)
    validate_generated_code(code)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "shadow-gpu-t4x2-intro",
                "metadata": {},
                "source": [
                    "# Controlled shadow GPU T4 x2 compatibility pilot v1\n",
                    "Synthetic compatibility-only GPU replacement for the terminated TPU pilot line.\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "shadow-gpu-t4x2-run",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (kernel_dir / NOTEBOOK_NAME).write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "id": TARGET,
        "title": "Controlled shadow GPU T4 x2 compatibility pilot v1",
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": MACHINE_SHAPE,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_bundle(kernel_dir)
    print("SHADOW_GPU_T4X2_MATERIALIZE PASS snapshots=4 tpu_layers=0 ddp=0 private_repo_access=0")


def validate_bundle(kernel_dir: Path) -> None:
    expected_files = {"kernel-metadata.json", NOTEBOOK_NAME}
    actual = {path.name for path in kernel_dir.iterdir() if path.is_file()}
    if actual != expected_files:
        raise RuntimeError(f"kernel bundle allowlist changed: {sorted(actual)}")
    notebook = json.loads((kernel_dir / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][1]["source"])
    compile(code, NOTEBOOK_NAME, "exec")
    validate_generated_code(code)
    expected = {
        "id": TARGET,
        "code_file": NOTEBOOK_NAME,
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": MACHINE_SHAPE,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"metadata mismatch: {key}")


def literal_commands(source: str) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr == "run"
            and node.args
            and isinstance(node.args[0], (ast.List, ast.Tuple))
        ):
            commands.append(
                tuple(
                    item.value if isinstance(item, ast.Constant) and isinstance(item.value, str) else "<dynamic>"
                    for item in node.args[0].elts
                )
            )
    return commands


def validate_static(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    commands = literal_commands(source)
    def pair(command: tuple[str, ...], left: str, right: str) -> bool:
        return any(command[index:index+2] == (left, right) for index in range(len(command)-1))
    if sum(pair(command, "kernels", "push") for command in commands) != 1:
        raise RuntimeError("launcher must contain exactly one kernels push")
    for forbidden in (
        ("competitions", "submit"),
        ("datasets", "create"),
        ("models", "create"),
        ("kernels", "delete"),
        ("kernels", "cancel"),
    ):
        if any(pair(command, *forbidden) for command in commands):
            raise RuntimeError(f"forbidden write: {' '.join(forbidden)}")


def execute(kaggle_bin: Path, kernel_dir: Path) -> None:
    validate_bundle(kernel_dir)
    result = subprocess.run(
        [str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir), "--timeout", "3600"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle GPU push returned nonzero ({result.returncode}); outcome ambiguous and retry forbidden"
        )
    print(
        "SHADOW_GPU_T4X2_LAUNCH_ACCEPTED "
        "target=renta0426/shadow-gpu-t4x2-pilot-v1 accelerator=gpu machine=NvidiaTeslaT4 "
        "retries=0 submissions=0 private_repo_access=0 post_push_getkernel=0"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--kernel-dir", type=Path)
    parser.add_argument("--kaggle-bin", type=Path)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if sum((args.static, args.materialize, args.execute)) != 1:
        raise SystemExit("select exactly one operation")
    if args.static:
        if args.launcher is None:
            raise SystemExit("--launcher required")
        validate_static(args.launcher)
        print("SHADOW_GPU_T4X2_STATIC PASS write_calls=1 retries=0 submissions=0 tpu_layers=0 private_repo_access=0")
        return
    if args.materialize:
        if args.snapshot_root is None or args.kernel_dir is None:
            raise SystemExit("--snapshot-root and --kernel-dir required")
        materialize(args.snapshot_root, args.kernel_dir)
        return
    if args.kaggle_bin is None or args.kernel_dir is None:
        raise SystemExit("--kaggle-bin and --kernel-dir required")
    execute(args.kaggle_bin, args.kernel_dir)


if __name__ == "__main__":
    main()
