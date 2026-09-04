"""Builder-only repair launcher for controlled-shadow TPU pilot v6.

v6 preserves the complete v5 scientific/XLA contract.  It uses the pinned v7
launcher only in materialize mode, repairs the generated package-init
``Path.write_text`` call so it has one data positional argument, upgrades the
frozen config/identity to v6, validates the generated Python AST, and owns the
sole Kaggle write.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

EXPERIMENT_ID = "shadow-tpu-pilot-v6"
REQUEST_ID = "20260904-poisoned-chalice-shadow-tpu-pilot-v6-001"
TARGET = "renta0426/shadow-tpu-pilot-v6"
RESEARCH_COMMIT = "686645b724152b46dfb8e7f1e674c0594efb7b70"
MACHINE_SHAPE = "Tpu1VmV38"
NOTEBOOK_NAME = "shadow-tpu-pilot-v6.ipynb"
SUMMARY_NAME = "shadow_tpu_pilot_manifest.json"
BUILDER_REPAIR = "package-init-write-text-single-data-arg-v1"

V7_BLOB = "bdf89be8cc873a42b06cf127fec9e33d434a3da2"
V6_BLOB = "3f7ca6222fa3273aa7dca6a0c92f5997d33e6ce4"
V5_BLOB = "ea0ee793fb5afd500ab9341332dea534fa771e96"
V4_BLOB = "317af66a052472410a8d33b5c54b8353c8acbbba"
V3_BLOB = "272f5e39eeb9695e1551bdcc5bffff5a2ef6c28a"
V6_CONFIG_PATH = "materialized/poisoned-chalice-shadow-tpu-pilot-v6/shadow_tpu_pilot_v6.json"
V6_CONFIG_BLOB = "02877fa3b9fce2cad4a95b454c56a7fe4163b1c0"


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def verify_file(path: Path, expected: str, maximum: int = 262144) -> None:
    data = path.read_bytes()
    if len(data) > maximum or blob_sha(data) != expected:
        raise RuntimeError(f"Git blob mismatch: {path}")


def build_v5_baseline(
    *, root: Path, launcher_v7: Path, wrapper_v6: Path, launcher_v5: Path,
    launcher_v4: Path, launcher_v3: Path, output: Path,
) -> None:
    verify_file(launcher_v7, V7_BLOB)
    verify_file(wrapper_v6, V6_BLOB, 65536)
    verify_file(launcher_v5, V5_BLOB)
    verify_file(launcher_v4, V4_BLOB)
    verify_file(launcher_v3, V3_BLOB)
    result = subprocess.run(
        [
            sys.executable, str(launcher_v7),
            "--materialize", "--snapshot-root", str(root),
            "--wrapper-v6", str(wrapper_v6),
            "--launcher-v5", str(launcher_v5),
            "--launcher-v4", str(launcher_v4),
            "--launcher-v3", str(launcher_v3),
            "--kernel-dir", str(output),
        ],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=120, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("pinned v5 materializer failed before Kaggle write: " + result.stderr[-3000:])


def validate_write_text_ast(code: str) -> None:
    tree = ast.parse(code)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_text"
    ]
    if not calls:
        raise RuntimeError("generated Notebook has no write_text calls")
    for call in calls:
        if len(call.args) != 1:
            segment = ast.get_source_segment(code, call) or "<unknown>"
            raise RuntimeError(f"generated write_text call has {len(call.args)} positional args: {segment[:240]}")
    init_calls = [
        call for call in calls
        if "poisoned_chalice/__init__.py" in (ast.get_source_segment(code, call) or "")
    ]
    if len(init_calls) != 1 or len(init_calls[0].args) != 1:
        raise RuntimeError("package-init write_text single-data-arg invariant failed")


def adapt_v5_code(code: str, config_text: str) -> str:
    lines = code.splitlines()
    flags = {"experiment": False, "research": False, "config": False, "init": False}
    for index, line in enumerate(lines):
        if line.startswith("EXPERIMENT_ID = "):
            lines[index] = f"EXPERIMENT_ID = {EXPERIMENT_ID!r}"; flags["experiment"] = True
        elif line.startswith("RESEARCH_COMMIT = "):
            lines[index] = f"RESEARCH_COMMIT = {RESEARCH_COMMIT!r}"; flags["research"] = True
        elif line.startswith("CONFIG_TEXT = "):
            lines[index] = "CONFIG_TEXT = " + repr(config_text); flags["config"] = True

    for index, line in enumerate(lines):
        if line.strip() == '"XLA_COMPAT = install_shadow_xla_compat()\\n",':
            if index + 2 >= len(lines):
                raise RuntimeError("package-init repair anchor truncated")
            if lines[index + 1].strip() != '"from .shadow_xla_inference_compat import install_shadow_xla_inference_compat\\n"':
                raise RuntimeError("package-init inference import anchor changed")
            if lines[index + 2].strip() != '"XLA_INFERENCE_COMPAT = install_shadow_xla_inference_compat()\\n",':
                raise RuntimeError("package-init inference install anchor changed")
            # v7 left a comma after XLA_COMPAT, creating two positional args.
            lines[index] = line.rsplit(",", 1)[0]
            flags["init"] = True
            break
    if not all(flags.values()):
        raise RuntimeError(f"v6 Notebook adaptation markers missing: {flags}")
    code = "\n".join(lines) + "\n"

    replacements = [
        ("if config.get('repair_from') != 'shadow-tpu-pilot-v4':", "if config.get('repair_from') != 'shadow-tpu-pilot-v5':", "repair parent"),
        ("    'repair_from': 'shadow-tpu-pilot-v4',", "    'repair_from': 'shadow-tpu-pilot-v5',", "summary parent"),
        ("shadow TPU compatibility pilot v5 failed acceptance gates; do not retry automatically", "shadow TPU compatibility pilot v6 failed acceptance gates; do not retry automatically", "failure marker"),
    ]
    for old, new, label in replacements:
        if old not in code:
            raise RuntimeError(f"v5 generated {label} missing")
        code = code.replace(old, new, 1)

    old_scope = (
        "if repair.get('compatibility_layer') != 'shadow-xla-compat-v1' or "
        "repair.get('spawn_layer') != 'shadow-xla-spawn-safe-v1' or "
        "repair.get('kaggle_tpu_environment_layer') != 'shadow-kaggle-tpu-pjrt-env-v1' or "
        "repair.get('xla_inference_layer') != 'shadow-xla-no-grad-forward-v1' or "
        "repair.get('xla_fixed_forward_context') != 'torch.no_grad' or "
        "repair.get('non_xla_fixed_forward_context') != 'unchanged' or "
        "repair.get('optimizer_step_context_changed') is not False or "
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
        f"repair.get('notebook_builder_repair') != '{BUILDER_REPAIR}' or "
        "repair.get('scientific_protocol_changed') is not False:"
    )
    if old_scope not in code:
        raise RuntimeError("v5 generated repair-scope guard missing")
    code = code.replace(old_scope, new_scope, 1)
    compile(code, NOTEBOOK_NAME, "exec")
    validate_write_text_ast(code)
    return code


def validate_bundle(kernel_dir: Path) -> None:
    expected_files = {"kernel-metadata.json", NOTEBOOK_NAME}
    if {p.name for p in kernel_dir.iterdir() if p.is_file()} != expected_files:
        raise RuntimeError("kernel bundle allowlist changed")
    notebook = json.loads((kernel_dir / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][1]["source"])
    compile(code, NOTEBOOK_NAME, "exec")
    validate_write_text_ast(code)
    for marker in (
        RESEARCH_COMMIT, "shadow-gpt2-byte-5m", "shadow-llama-byte-5m",
        "shadow-xla-compat-v1", "shadow-xla-spawn-safe-v1",
        "shadow-kaggle-tpu-pjrt-env-v1", "shadow-xla-no-grad-forward-v1",
        BUILDER_REPAIR, "install_shadow_xla_inference_compat", "torch.no_grad", SUMMARY_NAME,
    ):
        if marker not in code:
            raise RuntimeError(f"Notebook marker missing: {marker}")
    expected = {
        "id": TARGET, "code_file": NOTEBOOK_NAME, "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": True,
        "enable_internet": False, "machine_shape": MACHINE_SHAPE,
        "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"metadata mismatch: {key}")


def materialize(
    snapshot_root: Path, launcher_v7: Path, wrapper_v6: Path, launcher_v5: Path,
    launcher_v4: Path, launcher_v3: Path, kernel_dir: Path,
) -> None:
    if kernel_dir.exists() and any(kernel_dir.iterdir()):
        raise RuntimeError("kernel directory already exists and is not empty")
    config_path = snapshot_root / V6_CONFIG_PATH
    verify_file(config_path, V6_CONFIG_BLOB, 32768)
    config_text = config_path.read_text(encoding="utf-8")
    config = json.loads(config_text)
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("repair_from") != "shadow-tpu-pilot-v5":
        raise RuntimeError("pilot v6 identity changed")
    runtime = config.get("runtime") or {}
    if runtime.get("machine_shape") != MACHINE_SHAPE or runtime.get("max_steps_per_architecture") != 2 or runtime.get("expected_world_size") != 8:
        raise RuntimeError("pilot runtime contract changed")
    scope = config.get("repair_scope") or {}
    if scope.get("notebook_builder_repair") != BUILDER_REPAIR or scope.get("optimizer_step_context_changed") is not False or scope.get("scientific_protocol_changed") is not False:
        raise RuntimeError("pilot repair scope changed")

    baseline = kernel_dir.parent / (kernel_dir.name + "-v5-baseline")
    shutil.rmtree(baseline, ignore_errors=True)
    build_v5_baseline(
        root=snapshot_root, launcher_v7=launcher_v7, wrapper_v6=wrapper_v6,
        launcher_v5=launcher_v5, launcher_v4=launcher_v4, launcher_v3=launcher_v3,
        output=baseline,
    )
    base_notebook = json.loads((baseline / "shadow-tpu-pilot-v5.ipynb").read_text(encoding="utf-8"))
    code = adapt_v5_code("".join(base_notebook["cells"][1]["source"]), config_text)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type":"markdown","id":"shadow-tpu-v6-intro","metadata":{},"source":["# Controlled shadow TPU compatibility pilot v6\n","Builder-only re-test; scientific/XLA contract unchanged from v5.\n"]},
            {"cell_type":"code","id":"shadow-tpu-v6-run","execution_count":None,"metadata":{},"outputs":[],"source":code.splitlines(keepends=True)},
        ],
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},
        "nbformat":4,"nbformat_minor":5,
    }
    (kernel_dir / NOTEBOOK_NAME).write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "id": TARGET, "title": "Controlled shadow TPU compatibility pilot v6",
        "code_file": NOTEBOOK_NAME, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": True,
        "enable_internet": False, "machine_shape": MACHINE_SHAPE,
        "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.rmtree(baseline, ignore_errors=True)
    validate_bundle(kernel_dir)
    print("SHADOW_TPU_V6_MATERIALIZE PASS snapshots=9 builder_ast=pass private_repo_access=0")


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
    print("SHADOW_TPU_V6_LAUNCH_ACCEPTED target=renta0426/shadow-tpu-pilot-v6 accelerator=tpu retries=0 submissions=0 private_repo_access=0 post_push_getkernel=0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--launcher-v7", type=Path)
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
        print("SHADOW_TPU_V6_STATIC PASS write_calls=1 retries=0 submissions=0 private_repo_access=0")
        return
    if args.materialize:
        required = (args.snapshot_root,args.launcher_v7,args.wrapper_v6,args.launcher_v5,args.launcher_v4,args.launcher_v3,args.kernel_dir)
        if any(v is None for v in required): raise SystemExit("materialize paths required")
        materialize(args.snapshot_root,args.launcher_v7,args.wrapper_v6,args.launcher_v5,args.launcher_v4,args.launcher_v3,args.kernel_dir)
        return
    if args.kaggle_bin is None or args.kernel_dir is None: raise SystemExit("execute paths required")
    execute(args.kaggle_bin,args.kernel_dir)

if __name__ == "__main__":
    main()
