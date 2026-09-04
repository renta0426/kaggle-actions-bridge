"""Self-contained launcher for controlled-shadow TPU compatibility pilot v5.

Pilot v5 preserves the v4 scientific contract and changes only fixed-logit XLA
forward context: PyTorch/XLA uses torch.no_grad instead of
``torch.inference_mode`` for reference/checkpoint-parity forwards.  The launcher
reuses the exact v4 builder path only in materialize mode, adapts the generated
Notebook to the new frozen research snapshots, and owns the sole Kaggle write.
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
import sys

EXPERIMENT_ID = "shadow-tpu-pilot-v5"
REQUEST_ID = "20260904-poisoned-chalice-shadow-tpu-pilot-v5-001"
TARGET = "renta0426/shadow-tpu-pilot-v5"
RESEARCH_COMMIT = "9af6ea4c5e974533a410a1dc3a10d99fc107f981"
MACHINE_SHAPE = "Tpu1VmV38"
SUMMARY_NAME = "shadow_tpu_pilot_manifest.json"
NOTEBOOK_NAME = "shadow-tpu-pilot-v5.ipynb"
V6_BLOB = "3f7ca6222fa3273aa7dca6a0c92f5997d33e6ce4"
V5_BLOB = "ea0ee793fb5afd500ab9341332dea534fa771e96"
V4_BLOB = "317af66a052472410a8d33b5c54b8353c8acbbba"
V3_BLOB = "272f5e39eeb9695e1551bdcc5bffff5a2ef6c28a"

SNAPSHOTS = {
    "poisoned_chalice/shadow_protocol.py": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v1/poisoned_chalice/shadow_protocol.py",
        "1c47b80696050aa2e5e7c62384617df61ecb80da", 131072,
    ),
    "poisoned_chalice/shadow_training.py": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v1/poisoned_chalice/shadow_training.py",
        "2fe3cf2079659ae10e1ebe0d5973f3ce96c26a03", 131072,
    ),
    "poisoned_chalice/shadow_xla_compat.py": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v2/poisoned_chalice/shadow_xla_compat.py",
        "896606c8a3ae2a353a3a8619da9c66df1e2e918b", 32768,
    ),
    "poisoned_chalice/shadow_training_spawn.py": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v3/poisoned_chalice/shadow_training_spawn.py",
        "86ecba35b15b2971f61a970677732afbc6ffd6a1", 32768,
    ),
    "poisoned_chalice/shadow_kaggle_tpu_env.py": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v4/poisoned_chalice/shadow_kaggle_tpu_env.py",
        "03403f0e6acf381e82626e46191b5145e3df08dc", 32768,
    ),
    "shadow_tpu_pilot_v4.json": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v4/shadow_tpu_pilot_v4.json",
        "13595883858867638508c9b68eaefd61c3a8f8fc", 32768,
    ),
    "poisoned_chalice/shadow_xla_inference_compat.py": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v5/poisoned_chalice/shadow_xla_inference_compat.py",
        "cdfd8ef997e251358fd9abba84ad21e1a2a07301", 32768,
    ),
    "shadow_tpu_pilot_v5.json": (
        "materialized/poisoned-chalice-shadow-tpu-pilot-v5/shadow_tpu_pilot_v5.json",
        "ff6349e35caa8168506499f730827e2525487f8d", 32768,
    ),
}


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def verify_file(path: Path, expected: str, maximum: int = 262144) -> None:
    data = path.read_bytes()
    if len(data) > maximum or blob_sha(data) != expected:
        raise RuntimeError(f"Git blob mismatch: {path}")


def load_snapshots(root: Path) -> dict[str, bytes]:
    result = {}
    for logical, (bridge_path, expected, maximum) in SNAPSHOTS.items():
        path = root / bridge_path
        if not path.is_file():
            raise RuntimeError(f"snapshot missing: {bridge_path}")
        data = path.read_bytes()
        if len(data) > maximum or blob_sha(data) != expected:
            raise RuntimeError(f"snapshot identity mismatch: {bridge_path}")
        result[logical] = data
    return result


def build_v4_baseline(
    *, root: Path, wrapper_v6: Path, launcher_v5: Path,
    launcher_v4: Path, launcher_v3: Path, output: Path,
) -> None:
    verify_file(wrapper_v6, V6_BLOB, 65536)
    verify_file(launcher_v5, V5_BLOB)
    verify_file(launcher_v4, V4_BLOB)
    verify_file(launcher_v3, V3_BLOB)
    result = subprocess.run(
        [
            sys.executable, str(wrapper_v6), str(launcher_v5),
            "--materialize", "--snapshot-root", str(root),
            "--v3-launcher", str(launcher_v4),
            "--v2-launcher", str(launcher_v3),
            "--kernel-dir", str(output),
        ],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=120, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "pinned v4 materializer failed before Kaggle write: "
            + result.stderr[-3000:]
        )


def adapt_notebook_code(code: str, files: dict[str, bytes]) -> str:
    lines = code.splitlines()
    flags = {
        "experiment": False, "research": False, "config": False,
        "payload": False, "import": False, "driver": False, "init": False,
    }
    for index, line in enumerate(lines):
        if line.startswith("EXPERIMENT_ID = "):
            lines[index] = f"EXPERIMENT_ID = {EXPERIMENT_ID!r}"; flags["experiment"] = True
        elif line.startswith("RESEARCH_COMMIT = "):
            lines[index] = f"RESEARCH_COMMIT = {RESEARCH_COMMIT!r}"; flags["research"] = True
        elif line.startswith("CONFIG_TEXT = "):
            lines[index] = "CONFIG_TEXT = " + repr(files["shadow_tpu_pilot_v5.json"].decode("utf-8")); flags["config"] = True
        elif line.startswith("SOURCE_PAYLOADS = "):
            payload = ast.literal_eval(line.split("=", 1)[1].strip())
            payload["poisoned_chalice/shadow_xla_inference_compat.py"] = base64.b64encode(
                files["poisoned_chalice/shadow_xla_inference_compat.py"]
            ).decode("ascii")
            lines[index] = "SOURCE_PAYLOADS = " + repr(dict(sorted(payload.items())))
            flags["payload"] = True
        elif line == "from poisoned_chalice import KAGGLE_TPU_ENV, XLA_COMPAT":
            lines[index] = "from poisoned_chalice import KAGGLE_TPU_ENV, XLA_COMPAT, XLA_INFERENCE_COMPAT"; flags["import"] = True
        elif line.startswith("DRIVER_SOURCE = "):
            driver = ast.literal_eval(line.split("=", 1)[1].strip())
            driver = driver.replace(
                "from poisoned_chalice import KAGGLE_TPU_ENV, XLA_COMPAT\n",
                "from poisoned_chalice import KAGGLE_TPU_ENV, XLA_COMPAT, XLA_INFERENCE_COMPAT\n",
            )
            driver = driver.replace(
                "'xla_compat': XLA_COMPAT, 'kaggle_tpu_env': KAGGLE_TPU_ENV, ",
                "'xla_compat': XLA_COMPAT, 'kaggle_tpu_env': KAGGLE_TPU_ENV, 'xla_inference_compat': XLA_INFERENCE_COMPAT, ",
            )
            lines[index] = "DRIVER_SOURCE = " + repr(driver); flags["driver"] = True

    # Package init is represented as adjacent string literals in write_text().
    for index, line in enumerate(lines):
        if line.strip() == '"XLA_COMPAT = install_shadow_xla_compat()\\n",':
            lines[index + 1:index + 1] = [
                '    "from .shadow_xla_inference_compat import install_shadow_xla_inference_compat\\n"',
                '    "XLA_INFERENCE_COMPAT = install_shadow_xla_inference_compat()\\n",',
            ]
            flags["init"] = True
            break
    if not all(flags.values()):
        raise RuntimeError(f"v5 Notebook adaptation markers missing: {flags}")
    code = "\n".join(lines) + "\n"

    replacements = [
        (
            "if config.get('repair_from') != 'shadow-tpu-pilot-v3':",
            "if config.get('repair_from') != 'shadow-tpu-pilot-v4':",
            "repair parent",
        ),
        (
            "    'repair_from': 'shadow-tpu-pilot-v3',",
            "    'repair_from': 'shadow-tpu-pilot-v4',",
            "summary parent",
        ),
        (
            "shadow TPU compatibility pilot v4 failed acceptance gates; do not retry automatically",
            "shadow TPU compatibility pilot v5 failed acceptance gates; do not retry automatically",
            "failure marker",
        ),
    ]
    for old, new, label in replacements:
        if old not in code:
            raise RuntimeError(f"v4 generated {label} missing")
        code = code.replace(old, new, 1)

    old_scope = (
        "if repair.get('compatibility_layer') != 'shadow-xla-compat-v1' or "
        "repair.get('spawn_layer') != 'shadow-xla-spawn-safe-v1' or "
        "repair.get('kaggle_tpu_environment_layer') != 'shadow-kaggle-tpu-pjrt-env-v1' or "
        "repair.get('removed_environment_variables') != ['TPU_PROCESS_ADDRESSES', 'CLOUD_TPU_TASK_ID'] or "
        "repair.get('environment_variable_values_must_not_be_logged') is not True or "
        "repair.get('scientific_protocol_changed') is not False:"
    )
    new_scope = (
        "if repair.get('compatibility_layer') != 'shadow-xla-compat-v1' or "
        "repair.get('spawn_layer') != 'shadow-xla-spawn-safe-v1' or "
        "repair.get('kaggle_tpu_environment_layer') != 'shadow-kaggle-tpu-pjrt-env-v1' or "
        "repair.get('xla_inference_layer') != 'shadow-xla-no-grad-forward-v1' or "
        "repair.get('xla_fixed_forward_context') != 'torch.no_grad' or "
        "repair.get('non_xla_fixed_forward_context') != 'unchanged' or "
        "repair.get('optimizer_step_context_changed') is not False or "
        "repair.get('scientific_protocol_changed') is not False:"
    )
    if old_scope not in code:
        raise RuntimeError("v4 generated repair-scope guard missing")
    code = code.replace(old_scope, new_scope, 1)

    anchor = "    'xla_compat': XLA_COMPAT,"
    if anchor not in code:
        raise RuntimeError("summary XLA compatibility anchor missing")
    code = code.replace(
        anchor,
        anchor + "\n    'xla_inference_compat': XLA_INFERENCE_COMPAT,",
        1,
    )
    compile(code, NOTEBOOK_NAME, "exec")
    return code


def validate_bundle(kernel_dir: Path) -> None:
    expected_files = {"kernel-metadata.json", NOTEBOOK_NAME}
    if {p.name for p in kernel_dir.iterdir() if p.is_file()} != expected_files:
        raise RuntimeError("kernel bundle allowlist changed")
    notebook = json.loads((kernel_dir / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][1]["source"])
    compile(code, NOTEBOOK_NAME, "exec")
    for marker in (
        RESEARCH_COMMIT, "shadow-gpt2-byte-5m", "shadow-llama-byte-5m",
        "shadow-xla-compat-v1", "shadow-xla-spawn-safe-v1",
        "shadow-kaggle-tpu-pjrt-env-v1", "shadow-xla-no-grad-forward-v1",
        "install_shadow_xla_inference_compat", "torch.no_grad", SUMMARY_NAME,
    ):
        if marker not in code:
            raise RuntimeError(f"Notebook marker missing: {marker}")
    if code.index("sanitize_kaggle_tpu_pjrt_environment()") > code.index("install_shadow_xla_compat()"):
        raise RuntimeError("Kaggle topology sanitizer ordering changed")
    if code.index("install_shadow_xla_compat()") > code.index("install_shadow_xla_inference_compat()"):
        raise RuntimeError("XLA inference compatibility ordering changed")
    expected = {
        "id": TARGET, "code_file": NOTEBOOK_NAME, "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": True,
        "enable_internet": False, "machine_shape": MACHINE_SHAPE,
        "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
        "model_sources": [],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"metadata mismatch: {key}")


def materialize(
    snapshot_root: Path, wrapper_v6: Path, launcher_v5: Path,
    launcher_v4: Path, launcher_v3: Path, kernel_dir: Path,
) -> None:
    if kernel_dir.exists() and any(kernel_dir.iterdir()):
        raise RuntimeError("kernel directory already exists and is not empty")
    files = load_snapshots(snapshot_root)
    config = json.loads(files["shadow_tpu_pilot_v5.json"])
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("repair_from") != "shadow-tpu-pilot-v4":
        raise RuntimeError("pilot v5 identity changed")
    runtime = config.get("runtime") or {}
    if runtime.get("machine_shape") != MACHINE_SHAPE or runtime.get("max_steps_per_architecture") != 2 or runtime.get("expected_world_size") != 8:
        raise RuntimeError("pilot runtime contract changed")
    scope = config.get("repair_scope") or {}
    if scope.get("xla_inference_layer") != "shadow-xla-no-grad-forward-v1" or scope.get("optimizer_step_context_changed") is not False or scope.get("scientific_protocol_changed") is not False:
        raise RuntimeError("pilot repair scope changed")

    baseline = kernel_dir.parent / (kernel_dir.name + "-v4-baseline")
    shutil.rmtree(baseline, ignore_errors=True)
    build_v4_baseline(
        root=snapshot_root, wrapper_v6=wrapper_v6, launcher_v5=launcher_v5,
        launcher_v4=launcher_v4, launcher_v3=launcher_v3, output=baseline,
    )
    base_notebook = json.loads((baseline / "shadow-tpu-pilot-v4.ipynb").read_text(encoding="utf-8"))
    code = adapt_notebook_code("".join(base_notebook["cells"][1]["source"]), files)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type":"markdown","id":"shadow-tpu-v5-intro","metadata":{},"source":["# Controlled shadow TPU compatibility pilot v5\n","Synthetic compatibility-only re-test after one scoped XLA fixed-forward context repair.\n"]},
            {"cell_type":"code","id":"shadow-tpu-v5-run","execution_count":None,"metadata":{},"outputs":[],"source":code.splitlines(keepends=True)},
        ],
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},
        "nbformat":4,"nbformat_minor":5,
    }
    (kernel_dir / NOTEBOOK_NAME).write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "id": TARGET, "title": "Controlled shadow TPU compatibility pilot v5",
        "code_file": NOTEBOOK_NAME, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": True,
        "enable_internet": False, "machine_shape": MACHINE_SHAPE,
        "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.rmtree(baseline, ignore_errors=True)
    validate_bundle(kernel_dir)
    print("SHADOW_TPU_V5_MATERIALIZE PASS snapshots=8 private_repo_access=0 xla_no_grad=1")


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
        return any(command[i:i+2] == (left, right) for i in range(len(command)-1))
    if sum(pair(c, "kernels", "push") for c in commands) != 1:
        raise RuntimeError("launcher must contain exactly one kernels push")
    for forbidden in (("competitions","submit"),("datasets","create"),("models","create"),("kernels","delete"),("kernels","cancel")):
        if any(pair(c, *forbidden) for c in commands):
            raise RuntimeError(f"forbidden write: {' '.join(forbidden)}")


def execute(kaggle_bin: Path, kernel_dir: Path) -> None:
    validate_bundle(kernel_dir)
    result = subprocess.run(
        [str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir), "--timeout", "3600"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle push returned nonzero ({result.returncode}); outcome ambiguous and retry forbidden")
    print("SHADOW_TPU_V5_LAUNCH_ACCEPTED target=renta0426/shadow-tpu-pilot-v5 accelerator=tpu retries=0 submissions=0 private_repo_access=0 post_push_getkernel=0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--wrapper-v6", type=Path)
    parser.add_argument("--launcher-v5", type=Path)
    parser.add_argument("--launcher-v4", type=Path)
    parser.add_argument("--launcher-v3", type=Path)
    parser.add_argument("--kernel-dir", type=Path)
    parser.add_argument("--kaggle-bin", type=Path)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if sum((args.static, args.materialize, args.execute)) != 1:
        raise SystemExit("select exactly one operation")
    if args.static:
        if args.launcher is None: raise SystemExit("--launcher required")
        validate_static(args.launcher)
        print("SHADOW_TPU_V5_STATIC PASS write_calls=1 retries=0 submissions=0 private_repo_access=0")
        return
    if args.materialize:
        required = (args.snapshot_root,args.wrapper_v6,args.launcher_v5,args.launcher_v4,args.launcher_v3,args.kernel_dir)
        if any(v is None for v in required): raise SystemExit("materialize paths required")
        materialize(args.snapshot_root,args.wrapper_v6,args.launcher_v5,args.launcher_v4,args.launcher_v3,args.kernel_dir)
        return
    if args.kaggle_bin is None or args.kernel_dir is None: raise SystemExit("execute paths required")
    execute(args.kaggle_bin,args.kernel_dir)


if __name__ == "__main__":
    main()
