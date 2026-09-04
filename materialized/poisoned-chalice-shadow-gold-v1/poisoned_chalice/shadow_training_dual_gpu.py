"""Isolated two-GPU orchestration for controlled shadow training.

The scientific training implementation remains in :mod:`shadow_training`.
This module only assigns one architecture to each physical CUDA device and runs
both single-GPU jobs concurrently.  It deliberately does not use DDP, NCCL,
model sharding, or cross-GPU gradient communication.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

from .shadow_training import ShadowRuntimeConfig, train_shadow_model


SHADOW_DUAL_GPU_RUNTIME_VERSION = "shadow-dual-gpu-isolated-v1"


class ShadowDualGpuError(RuntimeError):
    """Raised when the frozen dual-GPU execution contract is violated."""


@dataclass(frozen=True)
class GpuChildSpec:
    slot: str
    physical_gpu_index: int
    output_directory: str
    command: tuple[str, ...]
    environment_overrides: Mapping[str, str]


def visible_cuda_device_names() -> list[str]:
    import torch

    if not torch.cuda.is_available():
        return []
    return [str(torch.cuda.get_device_name(index)) for index in range(torch.cuda.device_count())]


def validate_gpu_inventory(
    names: list[str],
    *,
    expected_count: int = 2,
    required_name_fragment: str = "T4",
) -> None:
    if len(names) != expected_count:
        raise ShadowDualGpuError(
            f"expected exactly {expected_count} visible CUDA devices, found {len(names)}"
        )
    if required_name_fragment and any(required_name_fragment.casefold() not in name.casefold() for name in names):
        raise ShadowDualGpuError(
            f"visible CUDA devices do not satisfy required name fragment {required_name_fragment!r}: {names}"
        )


def _child_command(
    *,
    protocol_directory: Path,
    output_directory: Path,
    slot: str,
    max_steps: int,
    checkpoint_reload_atol: float,
    parameter_sync_atol: float,
    required_gpu_name_fragment: str,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "poisoned_chalice.shadow_training_dual_gpu",
        "--child",
        "--protocol-directory",
        str(protocol_directory),
        "--output-directory",
        str(output_directory),
        "--slot",
        slot,
        "--max-steps",
        str(max_steps),
        "--checkpoint-reload-atol",
        repr(float(checkpoint_reload_atol)),
        "--parameter-sync-atol",
        repr(float(parameter_sync_atol)),
        "--required-gpu-name-fragment",
        required_gpu_name_fragment,
    )


def build_child_specs(
    protocol_directory: str | Path,
    output_directory: str | Path,
    *,
    max_steps: int = 2,
    checkpoint_reload_atol: float = 1e-5,
    parameter_sync_atol: float = 1e-5,
    required_gpu_name_fragment: str = "T4",
) -> tuple[GpuChildSpec, GpuChildSpec]:
    protocol = Path(protocol_directory).resolve()
    output = Path(output_directory).resolve()
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if checkpoint_reload_atol < 0 or parameter_sync_atol < 0:
        raise ValueError("tolerances must be non-negative")

    specs: list[GpuChildSpec] = []
    for slot, gpu_index in (("left", 0), ("right", 1)):
        child_output = output / slot
        specs.append(
            GpuChildSpec(
                slot=slot,
                physical_gpu_index=gpu_index,
                output_directory=str(child_output),
                command=_child_command(
                    protocol_directory=protocol,
                    output_directory=child_output,
                    slot=slot,
                    max_steps=max_steps,
                    checkpoint_reload_atol=checkpoint_reload_atol,
                    parameter_sync_atol=parameter_sync_atol,
                    required_gpu_name_fragment=required_gpu_name_fragment,
                ),
                environment_overrides={
                    "CUDA_VISIBLE_DEVICES": str(gpu_index),
                    "TOKENIZERS_PARALLELISM": "false",
                },
            )
        )
    return specs[0], specs[1]


def validate_child_manifest(
    manifest: Mapping[str, Any],
    *,
    slot: str,
    max_steps: int,
    checkpoint_reload_atol: float,
) -> None:
    if manifest.get("status") != "complete":
        raise ShadowDualGpuError(f"{slot} child did not complete")
    if manifest.get("architecture_slot") != slot:
        raise ShadowDualGpuError(f"{slot} child architecture slot mismatch")
    if manifest.get("backend") != "cuda":
        raise ShadowDualGpuError(f"{slot} child backend is not cuda")
    if int(manifest.get("world_size", -1)) != 1:
        raise ShadowDualGpuError(f"{slot} child world size is not 1")
    if int(manifest.get("completed_steps", -1)) != max_steps:
        raise ShadowDualGpuError(f"{slot} child completed-step mismatch")
    if manifest.get("checkpoint_reload_passed") is not True:
        raise ShadowDualGpuError(f"{slot} checkpoint reload did not pass")
    reload_difference = float(
        manifest.get("checkpoint_reload_max_absolute_logit_difference", float("inf"))
    )
    if reload_difference > checkpoint_reload_atol:
        raise ShadowDualGpuError(
            f"{slot} checkpoint reload difference {reload_difference} exceeds {checkpoint_reload_atol}"
        )
    if manifest.get("random_initialisation") is not True:
        raise ShadowDualGpuError(f"{slot} random-initialisation assertion missing")
    for key in (
        "pretrained_weights_used",
        "evaluation_inputs_read",
        "evaluation_labels_read",
        "sample_ids_used_as_features",
    ):
        if manifest.get(key) is not False:
            raise ShadowDualGpuError(f"{slot} clean-room guard failed: {key}")
    if int(manifest.get("automatic_compute_retries", -1)) != 0:
        raise ShadowDualGpuError(f"{slot} automatic retry guard failed")
    for key in ("initial_loss", "final_loss", "minimum_loss", "maximum_loss"):
        value = float(manifest.get(key, float("nan")))
        if not (value == value and abs(value) != float("inf")):
            raise ShadowDualGpuError(f"{slot} non-finite metric: {key}")


def _tail(path: Path, maximum: int) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return data[-maximum:].decode("utf-8", errors="replace")


def run_dual_gpu_shadow(
    protocol_directory: str | Path,
    output_directory: str | Path,
    *,
    max_steps: int = 2,
    checkpoint_reload_atol: float = 1e-5,
    parameter_sync_atol: float = 1e-5,
    expected_gpu_count: int = 2,
    required_gpu_name_fragment: str = "T4",
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Run the two architecture slots concurrently on separate physical GPUs."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    names = visible_cuda_device_names()
    validate_gpu_inventory(
        names,
        expected_count=expected_gpu_count,
        required_name_fragment=required_gpu_name_fragment,
    )

    protocol = Path(protocol_directory).resolve()
    output = Path(output_directory).resolve()
    if not protocol.is_dir():
        raise ShadowDualGpuError(f"protocol directory does not exist: {protocol}")
    if output.exists() and any(output.iterdir()):
        raise ShadowDualGpuError("dual-GPU output directory already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "_child_logs"
    logs.mkdir()

    specs = build_child_specs(
        protocol,
        output,
        max_steps=max_steps,
        checkpoint_reload_atol=checkpoint_reload_atol,
        parameter_sync_atol=parameter_sync_atol,
        required_gpu_name_fragment=required_gpu_name_fragment,
    )
    processes: dict[str, tuple[subprocess.Popen[Any], Any, Any, GpuChildSpec]] = {}
    for spec in specs:
        env = os.environ.copy()
        for key in (
            "PJRT_DEVICE",
            "TPU_PROCESS_ADDRESSES",
            "CLOUD_TPU_TASK_ID",
            "XRT_TPU_CONFIG",
            "LOCAL_RANK",
            "RANK",
            "WORLD_SIZE",
        ):
            env.pop(key, None)
        env.update(spec.environment_overrides)
        stdout_path = logs / f"{spec.slot}.stdout"
        stderr_path = logs / f"{spec.slot}.stderr"
        stdout_handle = stdout_path.open("wb")
        stderr_handle = stderr_path.open("wb")
        process = subprocess.Popen(
            list(spec.command),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        processes[spec.slot] = (process, stdout_handle, stderr_handle, spec)

    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            if all(process.poll() is not None for process, _, _, _ in processes.values()):
                break
            if time.monotonic() >= deadline:
                for process, _, _, _ in processes.values():
                    if process.poll() is None:
                        process.kill()
                raise ShadowDualGpuError("dual-GPU child timeout; automatic retry forbidden")
            time.sleep(0.25)
    finally:
        for process, stdout_handle, stderr_handle, _ in processes.values():
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            stdout_handle.close()
            stderr_handle.close()

    failures: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}
    for slot, (process, _, _, spec) in processes.items():
        stdout_path = logs / f"{slot}.stdout"
        stderr_path = logs / f"{slot}.stderr"
        if process.returncode != 0:
            failures.append(
                {
                    "slot": slot,
                    "physical_gpu_index": spec.physical_gpu_index,
                    "returncode": process.returncode,
                    "stdout_tail": _tail(stdout_path, 2000),
                    "stderr_tail": _tail(stderr_path, 5000),
                }
            )
            continue
        manifest_path = Path(spec.output_directory) / "training_manifest.json"
        if not manifest_path.is_file():
            failures.append(
                {
                    "slot": slot,
                    "physical_gpu_index": spec.physical_gpu_index,
                    "returncode": process.returncode,
                    "stdout_tail": _tail(stdout_path, 2000),
                    "stderr_tail": "training_manifest.json missing\n" + _tail(stderr_path, 4000),
                }
            )
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_child_manifest(
            manifest,
            slot=slot,
            max_steps=max_steps,
            checkpoint_reload_atol=checkpoint_reload_atol,
        )
        manifests[slot] = manifest

    if failures:
        raise ShadowDualGpuError(
            "one or more isolated GPU children failed: "
            + json.dumps(failures, sort_keys=True)
        )
    if set(manifests) != {"left", "right"}:
        raise ShadowDualGpuError("dual-GPU manifests are incomplete")

    return {
        "status": "complete",
        "runtime_version": SHADOW_DUAL_GPU_RUNTIME_VERSION,
        "gpu_names": names,
        "expected_gpu_count": expected_gpu_count,
        "required_gpu_name_fragment": required_gpu_name_fragment,
        "distributed_training": False,
        "ddp_used": False,
        "nccl_used": False,
        "architecture_gpu_assignment": {"left": 0, "right": 1},
        "max_steps_per_architecture": max_steps,
        "checkpoint_reload_atol": checkpoint_reload_atol,
        "automatic_compute_retries": 0,
        "left": manifests["left"],
        "right": manifests["right"],
    }


def _run_child(args: argparse.Namespace) -> int:
    import torch

    names = visible_cuda_device_names()
    validate_gpu_inventory(
        names,
        expected_count=1,
        required_name_fragment=args.required_gpu_name_fragment,
    )
    runtime = ShadowRuntimeConfig(
        backend="cuda",
        architecture_slot=args.slot,
        max_steps=args.max_steps,
        checkpoint_reload_atol=args.checkpoint_reload_atol,
        parameter_sync_atol=args.parameter_sync_atol,
        save_optimizer_state=True,
    )
    result = train_shadow_model(
        args.protocol_directory,
        args.output_directory,
        runtime,
    )
    if result is None:
        raise ShadowDualGpuError("CUDA child did not return a training result")
    validate_child_manifest(
        result,
        slot=args.slot,
        max_steps=args.max_steps,
        checkpoint_reload_atol=args.checkpoint_reload_atol,
    )
    if torch.cuda.device_count() != 1:
        raise ShadowDualGpuError("CUDA child visibility changed during execution")
    print(
        json.dumps(
            {
                "status": "complete",
                "slot": args.slot,
                "visible_gpu_name": names[0],
                "runtime_config": asdict(runtime),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--protocol-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--slot", choices=("left", "right"))
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--checkpoint-reload-atol", type=float, default=1e-5)
    parser.add_argument("--parameter-sync-atol", type=float, default=1e-5)
    parser.add_argument("--required-gpu-name-fragment", default="T4")
    args = parser.parse_args()
    if not args.child:
        raise SystemExit("this module CLI is reserved for isolated child execution")
    if args.protocol_directory is None or args.output_directory is None or args.slot is None:
        raise SystemExit("child execution requires protocol directory, output directory, and slot")
    raise SystemExit(_run_child(args))


if __name__ == "__main__":
    main()


__all__ = [
    "GpuChildSpec",
    "SHADOW_DUAL_GPU_RUNTIME_VERSION",
    "ShadowDualGpuError",
    "build_child_specs",
    "run_dual_gpu_shadow",
    "validate_child_manifest",
    "validate_gpu_inventory",
    "visible_cuda_device_names",
]
