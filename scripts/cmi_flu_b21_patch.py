#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASE_REQUEST_ID = "20260903-cmi-flu-b2-004"
BASE_SOURCE_COMMIT = "d6297c36366ab5c3ef49b9077c2357277f82a708"
B21_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"
TARGET_REQUEST_ID = "20260903-cmi-flu-b21-001"
TARGET_STAGE = "b21_taskwise_robust"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("request_id") != TARGET_REQUEST_ID:
        raise SystemExit("B2.1 request identity mismatch")
    if request.get("science_source_commit") != B21_SOURCE_COMMIT:
        raise SystemExit("B2.1 science commit mismatch")

    adapter = args.adapter.read_text(encoding="utf-8")
    compile(adapter, str(args.adapter), "exec")
    adapter_sha = hashlib.sha256(adapter.encode("utf-8")).hexdigest()
    if request.get("runtime_adapter_sha256") != adapter_sha:
        raise SystemExit("B2.1 runtime adapter SHA mismatch")

    text = args.source.read_text(encoding="utf-8")
    text = replace_once(
        text,
        f'REQUEST_ID = "{BASE_REQUEST_ID}"',
        f'REQUEST_ID = "{TARGET_REQUEST_ID}"',
        label="request id",
    )
    text = replace_once(
        text,
        f'SOURCE_COMMIT = "{BASE_SOURCE_COMMIT}"',
        f'SOURCE_COMMIT = "{B21_SOURCE_COMMIT}"',
        label="science source commit",
    )

    marker = "\ndef execute(input_dir: Path, output_dir: Path) -> Mapping[str, Any]:\n"
    injected = (
        f'\nB21_ADAPTER_SHA256 = "{adapter_sha}"\n'
        f'B21_ADAPTER_SOURCE = {adapter!r}\n'
        + marker
    )
    text = replace_once(text, marker, injected, label="adapter insertion")

    old_run = '''        stage = "run_b02"\n        from cmi_flu.runner import run_from_config\n\n        result = run_from_config(\n'''
    new_run = '''        stage = "install_b21_adapter"\n        adapter_namespace: dict[str, Any] = {}\n        exec(compile(B21_ADAPTER_SOURCE, "<b21_runtime_adapter>", "exec"), adapter_namespace, adapter_namespace)\n        install_adapter = adapter_namespace.get("install")\n        if not callable(install_adapter):\n            raise BundleContractError("B2.1 runtime adapter lacks install()")\n        install_adapter()\n\n        stage = "run_b021"\n        from cmi_flu.runner import run_from_config\n\n        result = run_from_config(\n'''
    text = replace_once(text, old_run, new_run, label="B2.1 adapter execution")
    text = replace_once(
        text,
        '"kernel_stage": "b2_taskwise_compact",',
        f'"kernel_stage": "{TARGET_STAGE}",\n            "runtime_adapter_sha256": B21_ADAPTER_SHA256,',
        label="kernel stage",
    )
    compile(text, str(args.output), "exec")
    args.output.write_text(text, encoding="utf-8")
    print(f"CMI_FLU_B21_PATCH PASS adapter_sha256={adapter_sha} source_commit={B21_SOURCE_COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
