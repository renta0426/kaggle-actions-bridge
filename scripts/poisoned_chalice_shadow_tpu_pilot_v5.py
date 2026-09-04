"""Self-contained launcher for controlled-shadow TPU compatibility pilot v4.

Pilot v4 preserves the v3 scientific contract and repairs only Kaggle/PJRT
process-environment compatibility.  Before any torch_xla import it removes the
two topology hints known to break multi-device PJRT initialization on Kaggle.
The topology-variable values are never emitted or retained.

The protected job reads only exact snapshots already present in the approved
public bridge commit.  It never reads the private research repository.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

EXPERIMENT_ID = "shadow-tpu-pilot-v4"
REQUEST_ID = "20260904-poisoned-chalice-shadow-tpu-pilot-v4-001"
TARGET = "renta0426/shadow-tpu-pilot-v4"
RESEARCH_COMMIT = "54552b71ab195542250100a26f42bcea663fc95b"
MACHINE_SHAPE = "Tpu1VmV38"
SUMMARY_NAME = "shadow_tpu_pilot_manifest.json"
NOTEBOOK_NAME = "shadow-tpu-pilot-v4.ipynb"
V3_LAUNCHER_BLOB_SHA = "317af66a052472410a8d33b5c54b8353c8acbbba"
V2_LAUNCHER_BLOB_SHA = "272f5e39eeb9695e1551bdcc5bffff5a2ef6c28a"

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
    "poisoned_chalice/shadow_training_spawn.py": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v3/poisoned_chalice/shadow_training_spawn.py",
        "86ecba35b15b2971f61a970677732afbc6ffd6a1",
        32768,
    ),
    "poisoned_chalice/shadow_kaggle_tpu_env.py": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v4/poisoned_chalice/shadow_kaggle_tpu_env.py",
        "03403f0e6acf381e82626e46191b5145e3df08dc",
        32768,
    ),
    "shadow_tpu_pilot_v4.json": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v4/shadow_tpu_pilot_v4.json",
        "13595883858867638508c9b68eaefd61c3a8f8fc",
        32768,
    ),
}


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _load_pinned_module(path: Path, expected_blob: str, module_name: str):
    data = path.read_bytes()
    if blob_sha(data) != expected_blob:
        raise RuntimeError(f"{module_name} Git blob mismatch")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _driver_source() -> str:
    return (
        "from __future__ import annotations\n"
        "import json, sys\n"
        "from poisoned_chalice import KAGGLE_TPU_ENV, XLA_COMPAT\n"
        "from poisoned_chalice.shadow_training import ShadowRuntimeConfig\n"
        "from poisoned_chalice.shadow_training_spawn import "
        "SHADOW_XLA_SPAWN_COMPAT_VERSION, train_shadow_model_spawn_safe\n"
        "\n"
        "def main():\n"
        "    if len(sys.argv) != 4: raise RuntimeError('expected slot, protocol_dir, output_dir')\n"
        "    slot, protocol_dir, output_dir = sys.argv[1:]\n"
        "    runtime = ShadowRuntimeConfig(backend='xla', architecture_slot=slot, max_steps=2, "
        "checkpoint_reload_atol=1e-5, parameter_sync_atol=1e-5, save_optimizer_state=True)\n"
        "    manifest = train_shadow_model_spawn_safe(protocol_dir, output_dir, runtime)\n"
        "    if manifest is None: raise RuntimeError('XLA training produced no master manifest')\n"
        "    print(json.dumps({'status': manifest['status'], 'slot': slot, "
        "'completed_steps': manifest['completed_steps'], 'xla_compat': XLA_COMPAT, "
        "'kaggle_tpu_env': KAGGLE_TPU_ENV, "
        "'xla_spawn_compat': SHADOW_XLA_SPAWN_COMPAT_VERSION}, sort_keys=True))\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


def build_notebook_code(
    files: dict[str, bytes],
    v3_launcher_path: Path,
    v2_launcher_path: Path,
) -> str:
    v3 = _load_pinned_module(
        v3_launcher_path, V3_LAUNCHER_BLOB_SHA, "shadow_tpu_pilot_v3_builder"
    )
    if blob_sha(v2_launcher_path.read_bytes()) != V2_LAUNCHER_BLOB_SHA:
        raise RuntimeError("v2 foundation launcher Git blob mismatch")
    v3.EXPERIMENT_ID = EXPERIMENT_ID
    v3.RESEARCH_COMMIT = RESEARCH_COMMIT
    v3.SUMMARY_NAME = SUMMARY_NAME
    v3.NOTEBOOK_NAME = NOTEBOOK_NAME

    adapted = {
        key: value for key, value in files.items() if key != "shadow_tpu_pilot_v4.json"
    }
    adapted["shadow_tpu_pilot_v3.json"] = files["shadow_tpu_pilot_v4.json"]
    code = v3.build_notebook_code(adapted, v2_launcher_path)

    lines = code.splitlines()
    driver_replaced = False
    import_replaced = False
    init_injected = False
    for index, line in enumerate(lines):
        if line.startswith("DRIVER_SOURCE = "):
            lines[index] = "DRIVER_SOURCE = " + repr(_driver_source())
            driver_replaced = True
        if line == "from poisoned_chalice import XLA_COMPAT":
            lines[index] = "from poisoned_chalice import KAGGLE_TPU_ENV, XLA_COMPAT"
            import_replaced = True
        if line.strip() == '"from .shadow_xla_compat import install_shadow_xla_compat\\n"':
            lines[index:index] = [
                '    "from .shadow_kaggle_tpu_env import sanitize_kaggle_tpu_pjrt_environment\\n"',
                '    "KAGGLE_TPU_ENV = sanitize_kaggle_tpu_pjrt_environment()\\n"',
            ]
            init_injected = True
            break
    if not driver_replaced or not import_replaced or not init_injected:
        raise RuntimeError(
            "pinned v3 builder structure changed: "
            f"driver={driver_replaced} import={import_replaced} init={init_injected}"
        )
    code = "\n".join(lines) + "\n"

    replacements = (
        (
            "if config.get('repair_from') != 'shadow-tpu-pilot-v2':",
            "if config.get('repair_from') != 'shadow-tpu-pilot-v3':",
            "repair-parent guard",
        ),
        (
            "    'repair_from': 'shadow-tpu-pilot-v2',",
            "    'repair_from': 'shadow-tpu-pilot-v3',",
            "summary repair parent",
        ),
        (
            "shadow TPU compatibility pilot v3 failed acceptance gates; do not retry automatically",
            "shadow TPU compatibility pilot v4 failed acceptance gates; do not retry automatically",
            "failure marker",
        ),
    )
    for old, new, label in replacements:
        if old not in code:
            raise RuntimeError(f"pinned v3 builder {label} not found")
        code = code.replace(old, new, 1)

    old_scope = (
        "if repair.get('compatibility_layer') != 'shadow-xla-compat-v1' or "
        "repair.get('spawn_layer') != 'shadow-xla-spawn-safe-v1' or "
        "repair.get('scientific_protocol_changed') is not False:"
    )
    new_scope = (
        "if repair.get('compatibility_layer') != 'shadow-xla-compat-v1' or "
        "repair.get('spawn_layer') != 'shadow-xla-spawn-safe-v1' or "
        "repair.get('kaggle_tpu_environment_layer') != 'shadow-kaggle-tpu-pjrt-env-v1' or "
        "repair.get('removed_environment_variables') != ['TPU_PROCESS_ADDRESSES', 'CLOUD_TPU_TASK_ID'] or "
        "repair.get('environment_variable_values_must_not_be_logged') is not True or "
        "repair.get('scientific_protocol_changed') is not False:"
    )
    if old_scope not in code:
        raise RuntimeError("pinned v3 builder repair-scope guard not found")
    code = code.replace(old_scope, new_scope, 1)

    summary_anchor = "    'xla_compat': XLA_COMPAT,"
    if summary_anchor not in code:
        raise RuntimeError("pinned v3 builder summary XLA marker not found")
    code = code.replace(
        summary_anchor,
        summary_anchor + "\n    'kaggle_tpu_env': KAGGLE_TPU_ENV,",
        1,
    )

    compile(code, NOTEBOOK_NAME, "exec")
    return code


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
    markers = (
        RESEARCH_COMMIT,
        "shadow-gpt2-byte-5m",
        "shadow-llama-byte-5m",
        "shadow-xla-compat-v1",
        "shadow-xla-spawn-safe-v1",
        "shadow-kaggle-tpu-pjrt-env-v1",
        "sanitize_kaggle_tpu_pjrt_environment",
        "TPU_PROCESS_ADDRESSES",
        "CLOUD_TPU_TASK_ID",
        "train_shadow_model_spawn_safe",
        "if __name__ == '__main__':",
        SUMMARY_NAME,
        "PJRT_DEVICE",
    )
    for marker in markers:
        if marker not in code:
            raise RuntimeError(f"Notebook marker missing: {marker}")
    if code.index("sanitize_kaggle_tpu_pjrt_environment()") > code.index(
        "install_shadow_xla_compat()"
    ):
        raise RuntimeError("Kaggle TPU environment sanitizer must precede torch_xla compatibility import")
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


def materialize(
    snapshot_root: Path,
    v3_launcher_path: Path,
    v2_launcher_path: Path,
    kernel_dir: Path,
) -> None:
    if kernel_dir.exists() and any(kernel_dir.iterdir()):
        raise RuntimeError("kernel directory already exists and is not empty")
    kernel_dir.mkdir(parents=True, exist_ok=True)
    files = load_snapshots(snapshot_root)
    config = json.loads(files["shadow_tpu_pilot_v4.json"])
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("pilot config ID changed")
    if config.get("repair_from") != "shadow-tpu-pilot-v3":
        raise RuntimeError("pilot repair parent changed")
    repair = config.get("repair_scope") or {}
    if repair != {
        "compatibility_layer": "shadow-xla-compat-v1",
        "spawn_layer": "shadow-xla-spawn-safe-v1",
        "kaggle_tpu_environment_layer": "shadow-kaggle-tpu-pjrt-env-v1",
        "removed_environment_variables": ["TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"],
        "environment_variable_values_must_not_be_logged": True,
        "scientific_protocol_changed": False,
    }:
        raise RuntimeError("pilot repair scope changed")
    runtime = config.get("runtime") or {}
    if runtime.get("machine_shape") != MACHINE_SHAPE:
        raise RuntimeError("pilot machine shape changed")
    if runtime.get("max_steps_per_architecture") != 2:
        raise RuntimeError("pilot step count changed")
    if runtime.get("expected_world_size") != 8:
        raise RuntimeError("pilot world-size contract changed")
    if config.get("interpretation", {}).get("stage2_v3_selection_allowed") is not False:
        raise RuntimeError("pilot scientific guard changed")

    code = build_notebook_code(files, v3_launcher_path, v2_launcher_path)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "shadow-tpu-v4-intro",
                "metadata": {},
                "source": [
                    "# Controlled shadow TPU compatibility pilot v4\n",
                    "Synthetic compatibility-only re-test after one scoped Kaggle PJRT topology-environment repair.\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "shadow-tpu-v4-run",
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
    (kernel_dir / NOTEBOOK_NAME).write_text(
        json.dumps(notebook, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "id": TARGET,
        "title": "Controlled shadow TPU compatibility pilot v4",
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
    print(
        "SHADOW_TPU_V4_MATERIALIZE PASS snapshots=6 private_repo_access=0 "
        "spawn_safe=1 topology_env_sanitized=1"
    )


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
        commands.append(
            tuple(
                item.value
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
                else "<dynamic>"
                for item in node.args[0].elts
            )
        )
    return commands


def validate_static(launcher: Path, v3_launcher: Path, v2_launcher: Path) -> None:
    source = launcher.read_text(encoding="utf-8")
    compile(source, str(launcher), "exec")
    _load_pinned_module(v3_launcher, V3_LAUNCHER_BLOB_SHA, "shadow_tpu_pilot_v3_builder")
    if blob_sha(v2_launcher.read_bytes()) != V2_LAUNCHER_BLOB_SHA:
        raise RuntimeError("v2 foundation launcher Git blob mismatch")
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
        "SHADOW_TPU_V4_LAUNCH_ACCEPTED target=renta0426/shadow-tpu-pilot-v4 "
        "accelerator=tpu retries=0 submissions=0 private_repo_access=0 post_push_getkernel=0"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--v3-launcher", type=Path)
    parser.add_argument("--v2-launcher", type=Path)
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
        if args.launcher is None or args.v3_launcher is None or args.v2_launcher is None:
            raise SystemExit("--launcher, --v3-launcher and --v2-launcher required")
        validate_static(args.launcher, args.v3_launcher, args.v2_launcher)
        print(
            "SHADOW_TPU_V4_STATIC PASS write_calls=1 retries=0 submissions=0 "
            "private_repo_access=0"
        )
        return
    if args.materialize:
        if (
            args.snapshot_root is None
            or args.v3_launcher is None
            or args.v2_launcher is None
            or args.kernel_dir is None
        ):
            raise SystemExit(
                "--snapshot-root, --v3-launcher, --v2-launcher and --kernel-dir required"
            )
        materialize(
            args.snapshot_root, args.v3_launcher, args.v2_launcher, args.kernel_dir
        )
        return
    if args.kaggle_bin is None or args.kernel_dir is None:
        raise SystemExit("--kaggle-bin and --kernel-dir required")
    execute(args.kaggle_bin, args.kernel_dir)


if __name__ == "__main__":
    main()
