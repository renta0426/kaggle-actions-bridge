"""Launch the frozen Gold positive-control sensitivity calibration on Kaggle T4 x2.

This launcher consumes only exact public bridge snapshots, builds one private
self-contained Notebook, persists four declared result artifacts, and owns
exactly one Kaggle kernels push.  It performs no private-research-repository
read, no competition submission, and no automatic compute retry.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

EXPERIMENT_ID = "shadow-gold-positive-control-v1"
REQUEST_ID = "20260906-poisoned-chalice-shadow-gold-positive-control-v1-001"
TARGET = "renta0426/shadow-gold-positive-control-v1"
RESEARCH_COMMIT = "4a0e00d96964b1203447c944d803455fd2c7b890"
BASE_SCIENCE_COMMIT = "bc0661fa0307b3e27b31f2ab4c83e90e74647390"
READBACK_REPAIR_COMMIT = "f8678f3d5d0a99763e02906f24c7a34e09e12f5e"
MACHINE_SHAPE = "NvidiaTeslaT4"
NOTEBOOK_NAME = "shadow-gold-positive-control-v1.ipynb"
PERSISTENT_OUTPUTS = (
    "positive_control_attack.json",
    "positive_control_metrics.json",
    "positive_control_predictions.jsonl",
    "positive_control_runtime_manifest.json",
)

SNAPSHOTS = {
    "poisoned_chalice/controlled_transfer.py": (
        "materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/controlled_transfer.py",
        "09b04c10d934654044817bb9b36fd7358a2aa942", 65536,
    ),
    "poisoned_chalice/shadow_gold_corpus.py": (
        "materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_gold_corpus.py",
        "241e3481f7dc56f5fc7495436f3cfc33274a0823", 32768,
    ),
    "poisoned_chalice/shadow_gold_kaggle.py": (
        "materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_gold_kaggle.py",
        "e68eb320605a943bb7d46b0e9ce51bb06142ffef", 65536,
    ),
    "poisoned_chalice/shadow_gold_transfer.py": (
        "materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_gold_transfer.py",
        "1ee969fe852f8a18d00a5c0249f4a8a90d61338a", 65536,
    ),
    "poisoned_chalice/shadow_protocol.py": (
        "materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_protocol.py",
        "1c47b80696050aa2e5e7c62384617df61ecb80da", 65536,
    ),
    "poisoned_chalice/shadow_scoring.py": (
        "materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_scoring.py",
        "6bc2b55bec2a32cbb3b0cfe870e426f80260730d", 65536,
    ),
    "poisoned_chalice/shadow_training.py": (
        "materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_training.py",
        "2fe3cf2079659ae10e1ebe0d5973f3ce96c26a03", 65536,
    ),
    "poisoned_chalice/shadow_training_dual_gpu.py": (
        "materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/shadow_training_dual_gpu.py",
        "c75a5963fa930916024cb16b4fbf562ba4afa818", 65536,
    ),
    "poisoned_chalice/stage2_api.py": (
        "materialized/poisoned-chalice-shadow-gold-v1/poisoned_chalice/stage2_api.py",
        "16578d3e01b6471e5bfb2b04cebb99e1b3126c7c", 65536,
    ),
    "poisoned_chalice/shadow_gold_kaggle_v2.py": (
        "materialized/poisoned-chalice-shadow-gold-v2/poisoned_chalice/shadow_gold_kaggle_v2.py",
        "df6e67f7d0e67d2585fd6226b9d2592223f808d2", 32768,
    ),
    "poisoned_chalice/shadow_gold_positive_control.py": (
        "materialized/poisoned-chalice-shadow-gold-positive-control-v1/poisoned_chalice/shadow_gold_positive_control.py",
        "8b14ba725a939b8a7c21aff93f70165c304fde79", 65536,
    ),
    "shadow_gold_positive_control_v1.json": (
        "materialized/poisoned-chalice-shadow-gold-positive-control-v1/shadow_gold_positive_control_v1.json",
        "424fe2674daebc1bba51b560d9ad057dff2b82c2", 32768,
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


def validate_snapshot_sources(files: dict[str, bytes]) -> None:
    for logical, data in files.items():
        if logical.endswith(".py"):
            compile(data.decode("utf-8"), logical, "exec")
    positive = files["poisoned_chalice/shadow_gold_positive_control.py"].decode("utf-8")
    for marker in (
        "run_positive_control_benchmark",
        "build_suffix_exposure_schedule",
        "positive_control_probe",
        '"primary_predictor": "loss"',
        '"stage2_v3_selection_allowed": False',
        "failed_gate_is_valid_scientific_result",
    ):
        if marker not in positive:
            raise RuntimeError(f"positive-control source marker missing: {marker}")
    repair = files["poisoned_chalice/shadow_gold_kaggle_v2.py"].decode("utf-8")
    for marker in ("canonicalize_exact_schema", "finalize_gold_results", "exact-schema"):
        if marker not in repair:
            raise RuntimeError(f"readback repair marker missing: {marker}")
    dual = files["poisoned_chalice/shadow_training_dual_gpu.py"].decode("utf-8")
    for forbidden in ("torch.distributed", "DistributedDataParallel", "torchrun"):
        if forbidden in dual:
            raise RuntimeError(f"distributed marker leaked into dual-GPU runtime: {forbidden}")


def validate_config(config: dict) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("role") != "gold_positive_control_calibration":
        raise RuntimeError("positive-control identity changed")
    data = config.get("data") or {}
    if data.get("candidate_rows_total") != 10240 or data.get("competition_rows_used") != 0 or data.get("external_rows_used") != 0:
        raise RuntimeError("positive-control data contract changed")
    split = config.get("controlled_split") or {}
    if split.get("expected_training_corpus_rows") != 5120 or split.get("expected_evaluation_rows") != 5120:
        raise RuntimeError("positive-control split scale changed")
    if split.get("split_is_built_before_probe_injection") is not True:
        raise RuntimeError("positive-control split/probe order changed")
    probe = config.get("positive_control_probe") or {}
    expected_probe = {
        "version": "content-hash-comment-probe-v1",
        "digest": "blake2b-64bit-of-original-content",
        "hex_characters": 16,
        "applied_to_members": True,
        "applied_to_nonmembers": True,
        "membership_used_to_construct_probe": False,
        "benchmark_id_used_to_construct_probe": False,
        "source_content_used_to_construct_probe": True,
        "post_injection_length_bin_must_equal_preinjection_length_bin": True,
    }
    for key, value in expected_probe.items():
        if probe.get(key) != value:
            raise RuntimeError(f"positive-control probe contract changed: {key}")
    protocol = config.get("protocol") or {}
    expected_protocol = {
        "max_sequence_tokens": 256,
        "global_batch_size": 64,
        "exposure_repeats": 16,
        "forced_training_window": "suffix",
        "exposure_round_mapping": "2_plus_4_times_repeat_index",
        "expected_training_sequences": 81920,
        "expected_optimizer_steps_per_architecture": 1280,
        "architecture_pair": ["shadow-gpt2-byte-5m", "shadow-llama-byte-5m"],
        "random_initialisation": True,
        "pretrained_weights_used": False,
    }
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"positive-control protocol changed: {key}")
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
        "automatic_compute_retries": 0,
        "notebook_internet": False,
    }
    for key, value in expected_runtime.items():
        if runtime.get(key) != value:
            raise RuntimeError(f"positive-control runtime changed: {key}")
    gate = config.get("sensitivity_gate") or {}
    expected_gate = {
        "primary_predictor": "loss",
        "shadow_a_holdout_min_auc": 0.60,
        "shadow_b_blind_holdout_min_auc": 0.60,
        "shadow_a_holdout_min_tpr_at_1pct_fpr": 0.03,
        "shadow_b_blind_holdout_min_tpr_at_1pct_fpr": 0.03,
        "all_four_thresholds_required": True,
        "secondary_predictors_may_rescue_failed_primary_gate": False,
        "failed_gate_is_valid_scientific_result_not_runtime_failure": True,
    }
    for key, value in expected_gate.items():
        if gate.get(key) != value:
            raise RuntimeError(f"positive-control sensitivity gate changed: {key}")
    guards = config.get("scientific_guards") or {}
    for key in ("stage2_v3_selection_allowed", "external_model_holdout_consumed", "fourth_external_holdout_consumed"):
        if guards.get(key) is not False:
            raise RuntimeError(f"positive-control scientific guard changed: {key}")


def build_notebook_code(files: dict[str, bytes]) -> str:
    validate_snapshot_sources(files)
    config_text = files["shadow_gold_positive_control_v1.json"].decode("utf-8")
    config = json.loads(config_text)
    validate_config(config)
    payloads = {
        path: base64.b64encode(data).decode("ascii")
        for path, data in sorted(files.items())
        if path != "shadow_gold_positive_control_v1.json"
    }
    template = r'''from __future__ import annotations
import base64, json, os, shutil, sys
from pathlib import Path

EXPERIMENT_ID = __EXPERIMENT_ID__
RESEARCH_COMMIT = __RESEARCH_COMMIT__
BASE_SCIENCE_COMMIT = __BASE_SCIENCE_COMMIT__
READBACK_REPAIR_COMMIT = __READBACK_REPAIR_COMMIT__
SOURCE_PAYLOADS = __SOURCE_PAYLOADS__
CONFIG_TEXT = __CONFIG_TEXT__
PERSISTENT_OUTPUTS = __PERSISTENT_OUTPUTS__

working = Path('/kaggle/working')
working.mkdir(parents=True, exist_ok=True)
for path in list(working.iterdir()):
    shutil.rmtree(path) if path.is_dir() else path.unlink()
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
if config.get('experiment_id') != EXPERIMENT_ID or config.get('role') != 'gold_positive_control_calibration':
    raise RuntimeError('positive-control config identity changed')
if config.get('protocol', {}).get('exposure_repeats') != 16:
    raise RuntimeError('positive-control exposure repeat contract changed')
if config.get('protocol', {}).get('forced_training_window') != 'suffix':
    raise RuntimeError('positive-control window contract changed')
if config.get('protocol', {}).get('expected_training_sequences') != 81920:
    raise RuntimeError('positive-control sequence contract changed')
if config.get('protocol', {}).get('expected_optimizer_steps_per_architecture') != 1280:
    raise RuntimeError('positive-control optimizer-step contract changed')
gate = config.get('sensitivity_gate', {})
if gate.get('primary_predictor') != 'loss':
    raise RuntimeError('positive-control primary predictor changed')
if [gate.get('shadow_a_holdout_min_auc'), gate.get('shadow_b_blind_holdout_min_auc')] != [0.60, 0.60]:
    raise RuntimeError('positive-control AUC gate changed')
if [gate.get('shadow_a_holdout_min_tpr_at_1pct_fpr'), gate.get('shadow_b_blind_holdout_min_tpr_at_1pct_fpr')] != [0.03, 0.03]:
    raise RuntimeError('positive-control low-FPR gate changed')
if config.get('data', {}).get('competition_rows_used') != 0 or config.get('data', {}).get('external_rows_used') != 0:
    raise RuntimeError('external rows leaked into positive control')
if config.get('scientific_guards', {}).get('stage2_v3_selection_allowed') is not False:
    raise RuntimeError('Stage2-v3 positive-control guard changed')

sys.path.insert(0, str(runtime_root))
existing_pythonpath = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = str(runtime_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else '')
from poisoned_chalice.shadow_gold_positive_control import run_positive_control_benchmark


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
    attack, metrics, predictions, manifest = run_positive_control_benchmark(
        scratch_directory=scratch / 'benchmark',
        config=config,
        timeout_seconds=5400,
    )
    if manifest.get('status') != 'complete':
        raise RuntimeError('positive-control benchmark did not complete')
    sensitivity = metrics.get('positive_control_sensitivity_gate') or {}
    if sensitivity.get('status') not in {'pass', 'fail'} or sensitivity.get('primary_predictor') != 'loss':
        raise RuntimeError('positive-control sensitivity result missing')
    if manifest.get('sensitivity_gate_status') != sensitivity.get('status'):
        raise RuntimeError('positive-control sensitivity status mismatch')
    if manifest.get('scientific_result_valid_even_if_gate_fails') is not True:
        raise RuntimeError('positive-control failed-gate interpretation changed')
    if manifest.get('evaluation_labels_passed_to_children') is not False:
        raise RuntimeError('positive-control label boundary failed')
    if manifest.get('competition_rows_used') != 0 or manifest.get('external_rows_used') != 0:
        raise RuntimeError('positive-control clean-room rows changed')
    if manifest.get('pretrained_weights_used') is not False or manifest.get('automatic_compute_retries') != 0:
        raise RuntimeError('positive-control training/retry guard changed')
    if manifest.get('stage2_v3_selection_allowed') is not False or manifest.get('fourth_external_holdout_consumed') is not False:
        raise RuntimeError('positive-control scientific guard failed')
    if len(predictions) != 2560:
        raise RuntimeError(f'positive-control prediction count changed: {len(predictions)}')

    manifest = dict(manifest)
    manifest['research_commit'] = RESEARCH_COMMIT
    manifest['base_science_commit'] = BASE_SCIENCE_COMMIT
    manifest['readback_repair_commit'] = READBACK_REPAIR_COMMIT

    staged = scratch / 'published'
    staged.mkdir()
    write_json(staged / 'positive_control_attack.json', attack)
    write_json(staged / 'positive_control_metrics.json', metrics)
    predictions.to_json(staged / 'positive_control_predictions.jsonl', orient='records', lines=True, force_ascii=False)
    write_json(staged / 'positive_control_runtime_manifest.json', manifest)
    if {p.name for p in staged.iterdir()} != set(PERSISTENT_OUTPUTS):
        raise RuntimeError('positive-control staged output allowlist changed')
    for name in PERSISTENT_OUTPUTS:
        publish_atomic(staged / name, working / name)
    for path in list(working.iterdir()):
        if path.name not in PERSISTENT_OUTPUTS:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    if {p.name for p in working.iterdir()} != set(PERSISTENT_OUTPUTS):
        raise RuntimeError('positive-control persistent output allowlist changed')

    observed = sensitivity['observed']
    print(json.dumps({
        'experiment_id': EXPERIMENT_ID,
        'research_commit': RESEARCH_COMMIT,
        'status': 'complete',
        'sensitivity_gate_status': sensitivity['status'],
        'primary_predictor': 'loss',
        'shadow_a_loss_auc': observed['shadow_a_auc'],
        'shadow_b_loss_auc': observed['shadow_b_auc'],
        'shadow_a_loss_tpr_1pct': observed['shadow_a_tpr_at_1pct_fpr'],
        'shadow_b_loss_tpr_1pct': observed['shadow_b_tpr_at_1pct_fpr'],
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
        "__EXPERIMENT_ID__": repr(EXPERIMENT_ID),
        "__RESEARCH_COMMIT__": repr(RESEARCH_COMMIT),
        "__BASE_SCIENCE_COMMIT__": repr(BASE_SCIENCE_COMMIT),
        "__READBACK_REPAIR_COMMIT__": repr(READBACK_REPAIR_COMMIT),
        "__SOURCE_PAYLOADS__": repr(payloads),
        "__CONFIG_TEXT__": repr(config_text),
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
        BASE_SCIENCE_COMMIT,
        READBACK_REPAIR_COMMIT,
        "run_positive_control_benchmark",
        "timeout_seconds=5400",
        "positive_control_attack.json",
        "positive_control_metrics.json",
        "positive_control_predictions.jsonl",
        "positive_control_runtime_manifest.json",
        "sensitivity_gate_status",
        "stage2_v3_selection_allowed",
    ):
        if marker not in code:
            raise RuntimeError(f"generated positive-control Notebook marker missing: {marker}")
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_positive_control_benchmark"
    ]
    if len(calls) != 1:
        raise RuntimeError("generated positive-control Notebook must invoke benchmark exactly once")
    for forbidden in ("competitions submit", "kernels push", "RESEARCH_REPO_READ_TOKEN"):
        if forbidden in code:
            raise RuntimeError(f"forbidden generated-code marker: {forbidden}")


def materialize(snapshot_root: Path, kernel_dir: Path) -> None:
    if kernel_dir.exists() and any(kernel_dir.iterdir()):
        raise RuntimeError("kernel directory already exists and is not empty")
    files = load_snapshots(snapshot_root)
    validate_snapshot_sources(files)
    config = json.loads(files["shadow_gold_positive_control_v1.json"])
    validate_config(config)
    code = build_notebook_code(files)
    validate_generated_code(code)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "shadow-gold-positive-control-intro",
                "metadata": {},
                "source": [
                    "# Gold positive-control sensitivity v1\n",
                    "Predeclared calibration of controlled membership sensitivity; not Stage2-v3 attack tuning.\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "shadow-gold-positive-control-run",
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
        "title": "Gold positive-control sensitivity v1",
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
    print("SHADOW_GOLD_PC_V1_MATERIALIZE PASS snapshots=12 outputs=4 gpu=2 steps=1280 eval=5120 retries=0")


def validate_bundle(kernel_dir: Path) -> None:
    expected_files = {"kernel-metadata.json", NOTEBOOK_NAME}
    actual = {path.name for path in kernel_dir.iterdir() if path.is_file()}
    if actual != expected_files:
        raise RuntimeError(f"positive-control kernel bundle allowlist changed: {sorted(actual)}")
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
            raise RuntimeError(f"positive-control metadata mismatch: {key}")


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
            commands.append(tuple(
                item.value if isinstance(item, ast.Constant) and isinstance(item.value, str) else "<dynamic>"
                for item in node.args[0].elts
            ))
    return commands


def validate_static(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    commands = literal_commands(source)
    def pair(command: tuple[str, ...], left: str, right: str) -> bool:
        return any(command[index:index+2] == (left, right) for index in range(len(command) - 1))
    if sum(pair(command, "kernels", "push") for command in commands) != 1:
        raise RuntimeError("positive-control launcher must contain exactly one kernels push")
    for forbidden in (
        ("competitions", "submit"),
        ("datasets", "create"),
        ("datasets", "version"),
        ("models", "create"),
        ("models", "version"),
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
            f"Kaggle positive-control push returned nonzero ({result.returncode}); outcome ambiguous and retry forbidden"
        )
    print(
        "SHADOW_GOLD_PC_V1_LAUNCH_ACCEPTED "
        "target=renta0426/shadow-gold-positive-control-v1 accelerator=gpu machine=NvidiaTeslaT4 "
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
        print("SHADOW_GOLD_PC_V1_STATIC PASS write_calls=1 retries=0 submissions=0 outputs=4")
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
