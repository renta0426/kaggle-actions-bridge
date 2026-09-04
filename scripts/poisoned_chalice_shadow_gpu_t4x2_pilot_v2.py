"""Audited wrapper for the controlled-shadow T4 x2 pilot builder.

The pinned v1 launcher is used only for materialization.  This wrapper replaces
one over-broad generated-code marker check, validates the embedded dual-GPU
orchestrator source directly, and owns the sole Kaggle write.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

BASE_LAUNCHER_BLOB = "0bc8342d9831f75e16ea297b486669d4e067ace1"
TARGET = "renta0426/shadow-gpu-t4x2-pilot-v1"
NOTEBOOK_NAME = "shadow-gpu-t4x2-pilot-v1.ipynb"
MACHINE_SHAPE = "NvidiaTeslaT4"
ORCHESTRATOR_PATH = "materialized/poisoned-chalice-shadow-gpu-t4x2-pilot-v1/poisoned_chalice/shadow_training_dual_gpu.py"
ORCHESTRATOR_BLOB = "c75a5963fa930916024cb16b4fbf562ba4afa818"


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def load_base(path: Path):
    data = path.read_bytes()
    if blob_sha(data) != BASE_LAUNCHER_BLOB:
        raise RuntimeError("base GPU launcher blob mismatch")
    spec = importlib.util.spec_from_file_location("shadow_gpu_t4x2_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load base GPU launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corrected_generated_code_validation(code: str, research_commit: str, summary_name: str) -> None:
    tree = ast.parse(code)
    required = (
        research_commit,
        "shadow-gpt2-byte-5m",
        "shadow-llama-byte-5m",
        "expected_gpu_count=2",
        "required_gpu_name_fragment='T4'",
        "distributed_training': False",
        "ddp_used': False",
        "nccl_used': False",
        "os.environ['PYTHONPATH']",
        "checkpoint_reload_atol=1e-5",
        summary_name,
    )
    for marker in required:
        if marker not in code:
            raise RuntimeError(f"generated Notebook marker missing: {marker}")
    for marker in (
        "torch_xla",
        "PJRT_DEVICE",
        "TPU_PROCESS_ADDRESSES",
        "shadow-xla-compat",
        "shadow-xla-spawn-safe",
        "shadow-kaggle-tpu-pjrt-env",
        "torch.distributed",
        "DistributedDataParallel",
        "torchrun",
    ):
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


def validate_orchestrator(snapshot_root: Path) -> None:
    path = snapshot_root / ORCHESTRATOR_PATH
    data = path.read_bytes()
    if len(data) > 65536 or blob_sha(data) != ORCHESTRATOR_BLOB:
        raise RuntimeError("dual-GPU orchestrator identity mismatch")
    source = data.decode("utf-8")
    compile(source, ORCHESTRATOR_PATH, "exec")
    required = (
        'for slot, gpu_index in (("left", 0), ("right", 1)):',
        '"CUDA_VISIBLE_DEVICES": str(gpu_index)',
        "subprocess.Popen(",
        'backend="cuda"',
        'expected_count=1',
        '"WORLD_SIZE"',
        '"PJRT_DEVICE"',
    )
    for marker in required:
        if marker not in source:
            raise RuntimeError(f"dual-GPU orchestrator marker missing: {marker}")
    for forbidden in ("torch.distributed", "DistributedDataParallel", "torchrun"):
        if forbidden in source:
            raise RuntimeError(f"distributed training leaked into orchestrator: {forbidden}")


def materialize(base_launcher: Path, snapshot_root: Path, kernel_dir: Path) -> None:
    validate_orchestrator(snapshot_root)
    base = load_base(base_launcher)
    base.validate_generated_code = lambda code: corrected_generated_code_validation(
        code, base.RESEARCH_COMMIT, base.SUMMARY_NAME
    )
    base.materialize(snapshot_root, kernel_dir)
    validate_bundle(base, kernel_dir)
    print("SHADOW_GPU_T4X2_V2_MATERIALIZE PASS snapshots=4 orchestrator=isolated tpu_layers=0 ddp=0")


def validate_bundle(base, kernel_dir: Path) -> None:
    base.validate_bundle(kernel_dir)
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    expected = {
        "id": TARGET,
        "code_file": NOTEBOOK_NAME,
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
            raise RuntimeError(f"GPU bundle metadata mismatch: {key}")
    notebook = json.loads((kernel_dir / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][1]["source"])
    corrected_generated_code_validation(code, base.RESEARCH_COMMIT, base.SUMMARY_NAME)


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
        raise RuntimeError("wrapper must contain exactly one kernels push")
    for forbidden in (
        ("competitions", "submit"),
        ("datasets", "create"),
        ("models", "create"),
        ("kernels", "delete"),
        ("kernels", "cancel"),
    ):
        if any(pair(command, *forbidden) for command in commands):
            raise RuntimeError(f"forbidden write: {' '.join(forbidden)}")


def execute(base_launcher: Path, kaggle_bin: Path, kernel_dir: Path) -> None:
    base = load_base(base_launcher)
    base.validate_generated_code = lambda code: corrected_generated_code_validation(
        code, base.RESEARCH_COMMIT, base.SUMMARY_NAME
    )
    validate_bundle(base, kernel_dir)
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
    parser.add_argument("--base-launcher", type=Path)
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
        print("SHADOW_GPU_T4X2_V2_STATIC PASS write_calls=1 retries=0 submissions=0 tpu_layers=0")
        return
    if args.base_launcher is None or args.kernel_dir is None:
        raise SystemExit("--base-launcher and --kernel-dir required")
    if args.materialize:
        if args.snapshot_root is None:
            raise SystemExit("--snapshot-root required")
        materialize(args.base_launcher, args.snapshot_root, args.kernel_dir)
        return
    if args.kaggle_bin is None:
        raise SystemExit("--kaggle-bin required")
    execute(args.base_launcher, args.kaggle_bin, args.kernel_dir)


if __name__ == "__main__":
    main()
