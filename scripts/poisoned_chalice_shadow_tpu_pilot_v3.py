"""Self-contained launcher for repaired controlled-shadow TPU compatibility pilot v2.

The protected job consumes only exact snapshots already present in the approved
public bridge commit.  It never reads the private research repository.  The
scientific contract matches pilot v1; only the declared PyTorch/XLA API
compatibility shim is added.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess

EXPERIMENT_ID = "shadow-tpu-pilot-v2"
REQUEST_ID = "20260904-poisoned-chalice-shadow-tpu-pilot-v2-001"
TARGET = "renta0426/shadow-tpu-pilot-v2"
RESEARCH_COMMIT = "39a5d1b9d3852edeff69f598666494654d4607f9"
MACHINE_SHAPE = "Tpu1VmV38"
SUMMARY_NAME = "shadow_tpu_pilot_manifest.json"
NOTEBOOK_NAME = "shadow-tpu-pilot-v2.ipynb"

SNAPSHOTS = {
    "poisoned_chalice/shadow_protocol.py": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v1/poisoned_chalice/shadow_protocol.py",
        "1c47b80696050aa2e5e7c62384617df61ecb80da",
        131072,
    ),
    "poisoned_chalice/shadow_training.py": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v1/poisoned_chalice/shadow_training.py",
        "2fe3cf2079659ae10e1ebe0d5973f3ce96c26a03",
        131072,
    ),
    "poisoned_chalice/shadow_xla_compat.py": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v2/poisoned_chalice/shadow_xla_compat.py",
        "896606c8a3ae2a353a3a8619da9c66df1e2e918b",
        32768,
    ),
    "shadow_tpu_pilot_v2.json": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v2/shadow_tpu_pilot_v2.json",
        "be23c26463ec0136acbb0e14477c22459a862465",
        32768,
    ),
}


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def load_snapshots(root: Path) -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    for logical, (bridge_path, expected_blob, maximum) in SNAPSHOTS.items():
        path = root / bridge_path
        if not path.is_file():
            raise RuntimeError(f"materialized snapshot missing: {bridge_path}")
        data = path.read_bytes()
        if len(data) > maximum or blob_sha(data) != expected_blob:
            raise RuntimeError(f"materialized snapshot identity mismatch: {bridge_path}")
        values[logical] = data
    return values


def build_notebook_code(files: dict[str, bytes]) -> str:
    payload = {
        path: base64.b64encode(data).decode("ascii")
        for path, data in sorted(files.items())
        if path != "shadow_tpu_pilot_v2.json"
    }
    config_text = files["shadow_tpu_pilot_v2.json"].decode("utf-8")
    driver_source = (
        "from __future__ import annotations\n"
        "import json, sys\n"
        "from poisoned_chalice import XLA_COMPAT\n"
        "from poisoned_chalice.shadow_training import ShadowRuntimeConfig, train_shadow_model\n"
        "slot, protocol_dir, output_dir = sys.argv[1:]\n"
        "runtime = ShadowRuntimeConfig(backend='xla', architecture_slot=slot, max_steps=2, "
        "checkpoint_reload_atol=1e-5, parameter_sync_atol=1e-5, save_optimizer_state=True)\n"
        "manifest = train_shadow_model(protocol_dir, output_dir, runtime)\n"
        "if manifest is None: raise RuntimeError('XLA training produced no master manifest')\n"
        "print(json.dumps({'status': manifest['status'], 'slot': slot, "
        "'completed_steps': manifest['completed_steps'], 'xla_compat': XLA_COMPAT}, sort_keys=True))\n"
    )
    template = r'''from __future__ import annotations
import base64, hashlib, importlib.metadata, json, math, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

EXPERIMENT_ID = __EXPERIMENT_ID__
RESEARCH_COMMIT = __RESEARCH_COMMIT__
SUMMARY_NAME = __SUMMARY_NAME__
SOURCE_PAYLOADS = __SOURCE_PAYLOADS__
CONFIG_TEXT = __CONFIG_TEXT__
DRIVER_SOURCE = __DRIVER_SOURCE__

os.environ.setdefault('PJRT_DEVICE', 'TPU')
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
(runtime_root / 'poisoned_chalice/__init__.py').write_text(
    "from .shadow_xla_compat import install_shadow_xla_compat\n"
    "XLA_COMPAT = install_shadow_xla_compat()\n",
    encoding='utf-8',
)
config = json.loads(CONFIG_TEXT)
if config.get('experiment_id') != EXPERIMENT_ID:
    raise RuntimeError('pilot config ID changed')
if config.get('repair_from') != 'shadow-tpu-pilot-v1':
    raise RuntimeError('pilot repair parent changed')
repair = config.get('repair_scope') or {}
if repair.get('compatibility_layer') != 'shadow-xla-compat-v1' or repair.get('scientific_protocol_changed') is not False:
    raise RuntimeError('pilot repair scope changed')
if config['data']['synthetic_training_rows'] != 128:
    raise RuntimeError('pilot row count changed')
if config['runtime']['backend'] != 'xla' or config['runtime']['max_steps_per_architecture'] != 2:
    raise RuntimeError('pilot runtime changed')
if config['runtime']['expected_world_size'] != 8:
    raise RuntimeError('pilot world-size contract changed')

sys.path.insert(0, str(runtime_root))
from poisoned_chalice import XLA_COMPAT
from poisoned_chalice.shadow_protocol import ByteCodeTokenizer, ShadowSequenceConfig, ShadowTrainingSpec, build_shadow_protocol_manifest, build_shadow_training_sequences, default_shadow_pair


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
    'operation': 'freeze_synthetic_shadow_tpu_compatibility_protocol',
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
for package in ('torch', 'torch-xla', 'transformers', 'numpy', 'pandas'):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
if versions['torch-xla'] is None:
    raise RuntimeError('Kaggle TPU image lacks torch-xla')

driver = scratch / 'train_one_shadow.py'
driver.write_text(DRIVER_SOURCE, encoding='utf-8')
env = os.environ.copy()
env['PYTHONPATH'] = str(runtime_root)
env.setdefault('PJRT_DEVICE', 'TPU')
records = {}
started = time.time()
for slot in ('left', 'right'):
    output_dir = scratch / f'training-{slot}'
    result = subprocess.run(
        [sys.executable, str(driver), slot, str(protocol_dir), str(output_dir)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=1200,
        check=False,
    )
    record = {
        'returncode': result.returncode,
        'stdout_tail': result.stdout[-2000:],
        'stderr_tail': result.stderr[-4000:],
    }
    manifest_path = output_dir / 'training_manifest.json'
    if manifest_path.is_file():
        record['training_manifest'] = json.loads(manifest_path.read_text(encoding='utf-8'))
    records[slot] = record
    if result.returncode != 0:
        print(json.dumps({
            'event': 'shadow_tpu_slot_failure',
            'slot': slot,
            'returncode': result.returncode,
            'stdout_tail': record['stdout_tail'],
            'stderr_tail': record['stderr_tail'],
        }, sort_keys=True))

acceptance = {}
for slot in ('left', 'right'):
    record = records[slot]
    value = record.get('training_manifest') or {}
    finite_loss = all(
        isinstance(value.get(k), (int, float)) and math.isfinite(float(value[k]))
        for k in ('initial_loss', 'final_loss', 'minimum_loss', 'maximum_loss')
    )
    acceptance[slot] = bool(
        record['returncode'] == 0
        and value.get('status') == 'complete'
        and value.get('backend') == 'xla'
        and value.get('world_size') == 8
        and value.get('completed_steps') == 2
        and finite_loss
        and float(value.get('parameter_checksum_spread', float('inf'))) <= 1e-5
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
    'purpose': 'TPU/XLA compatibility only; not Stage2-v3 selection evidence',
    'repair_from': 'shadow-tpu-pilot-v1',
    'research_commit': RESEARCH_COMMIT,
    'config_sha256': hashlib.sha256(CONFIG_TEXT.encode()).hexdigest(),
    'xla_compat': XLA_COMPAT,
    'backend': 'xla',
    'machine_shape': 'Tpu1VmV38',
    'expected_world_size': 8,
    'synthetic_training_rows': 128,
    'max_sequence_tokens': 256,
    'global_batch_size': 64,
    'max_steps_per_architecture': 2,
    'package_versions': versions,
    'acceptance': acceptance,
    'architecture_results': records,
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
    'xla_compat': XLA_COMPAT,
}, sort_keys=True))
if summary['status'] != 'pass':
    raise RuntimeError('shadow TPU compatibility pilot v2 failed acceptance gates; do not retry automatically')
'''
    replacements = {
        "__EXPERIMENT_ID__": repr(EXPERIMENT_ID),
        "__RESEARCH_COMMIT__": repr(RESEARCH_COMMIT),
        "__SUMMARY_NAME__": repr(SUMMARY_NAME),
        "__SOURCE_PAYLOADS__": repr(payload),
        "__CONFIG_TEXT__": repr(config_text),
        "__DRIVER_SOURCE__": repr(driver_source),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template


def validate_bundle(kernel_dir: Path) -> None:
    expected_names = {"kernel-metadata.json", NOTEBOOK_NAME}
    actual = {path.name for path in kernel_dir.iterdir() if path.is_file()}
    if actual != expected_names:
        raise RuntimeError(f"kernel bundle allowlist changed: {sorted(actual)}")
    notebook = json.loads((kernel_dir / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    if len(notebook.get("cells", [])) != 2:
        raise RuntimeError("Notebook cell count changed")
    code = "".join(notebook["cells"][1]["source"])
    compile(code, NOTEBOOK_NAME, "exec")
    for marker in (
        RESEARCH_COMMIT,
        "shadow-gpt2-byte-5m",
        "shadow-llama-byte-5m",
        "shadow-xla-compat-v1",
        SUMMARY_NAME,
        "PJRT_DEVICE",
    ):
        if marker not in code:
            raise RuntimeError(f"Notebook marker missing: {marker}")
    for forbidden in (
        "KAGGLE_API_TOKEN",
        "RESEARCH_REPO_READ_TOKEN",
        "api.github.com/repos/renta0426/The-Poisoned-Chalice-of-LLM-Evaluation",
        "raw.githubusercontent.com/renta0426/The-Poisoned-Chalice-of-LLM-Evaluation",
        "competitions submit",
        "HuggingFaceTB/SmolLM2",
    ):
        if forbidden.casefold() in code.casefold():
            raise RuntimeError(f"Notebook gained forbidden marker: {forbidden}")
    expected = {
        "id": TARGET,
        "code_file": NOTEBOOK_NAME,
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
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


def materialize(snapshot_root: Path, kernel_dir: Path) -> None:
    if kernel_dir.exists() and any(kernel_dir.iterdir()):
        raise RuntimeError("kernel directory already exists and is not empty")
    kernel_dir.mkdir(parents=True, exist_ok=True)
    files = load_snapshots(snapshot_root)
    config = json.loads(files["shadow_tpu_pilot_v2.json"])
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("pilot config ID changed")
    if config.get("repair_from") != "shadow-tpu-pilot-v1":
        raise RuntimeError("pilot repair parent changed")
    if config.get("runtime", {}).get("machine_shape") != MACHINE_SHAPE:
        raise RuntimeError("pilot machine shape changed")
    if config.get("interpretation", {}).get("stage2_v3_selection_allowed") is not False:
        raise RuntimeError("pilot scientific guard changed")
    code = build_notebook_code(files)
    compile(code, NOTEBOOK_NAME, "exec")
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "shadow-tpu-v2-intro",
                "metadata": {},
                "source": [
                    "# Controlled shadow TPU compatibility pilot v2\n",
                    "Synthetic compatibility-only re-test after one scoped PyTorch/XLA API repair.\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "shadow-tpu-v2-run",
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
        "title": "Controlled shadow TPU compatibility pilot v2",
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
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
    print("SHADOW_TPU_V2_MATERIALIZE PASS snapshots=4 private_repo_access=0")


def _literal_commands(source: str) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr == "run"
            and node.args
            and isinstance(node.args[0], (ast.List, ast.Tuple))
        ):
            continue
        commands.append(tuple(
            item.value if isinstance(item, ast.Constant) and isinstance(item.value, str) else "<dynamic>"
            for item in node.args[0].elts
        ))
    return commands


def validate_static(launcher: Path) -> None:
    source = launcher.read_text(encoding="utf-8")
    compile(source, str(launcher), "exec")
    commands = _literal_commands(source)
    def has_pair(command: tuple[str, ...], left: str, right: str) -> bool:
        return any(command[index:index + 2] == (left, right) for index in range(len(command) - 1))
    if sum(has_pair(command, "kernels", "push") for command in commands) != 1:
        raise RuntimeError("launcher must contain exactly one kernels push")
    for pair in (
        ("competitions", "submit"),
        ("datasets", "create"),
        ("datasets", "version"),
        ("models", "create"),
        ("kernels", "delete"),
        ("kernels", "cancel"),
    ):
        if any(has_pair(command, *pair) for command in commands):
            raise RuntimeError(f"launcher gained forbidden write: {' '.join(pair)}")


def execute(kaggle_bin: Path, kernel_dir: Path) -> None:
    validate_bundle(kernel_dir)
    result = subprocess.run(
        [str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir), "--timeout", "3600"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle push returned nonzero ({result.returncode}); outcome ambiguous and retry forbidden"
        )
    print(
        "SHADOW_TPU_V2_LAUNCH_ACCEPTED target=renta0426/shadow-tpu-pilot-v2 "
        "accelerator=tpu retries=0 submissions=0 private_repo_access=0 post_push_getkernel=0"
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
    selected = sum((args.static, args.materialize, args.execute))
    if selected != 1:
        raise SystemExit("select exactly one operation")
    if args.static:
        if args.launcher is None:
            raise SystemExit("--launcher required")
        validate_static(args.launcher)
        print("SHADOW_TPU_V2_STATIC PASS write_calls=1 retries=0 submissions=0 private_repo_access=0")
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
