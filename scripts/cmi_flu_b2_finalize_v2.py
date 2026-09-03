#!/usr/bin/env python3
"""Retry-safe entrypoint for the immutable CMI-Flu B2 finalizer.

Kaggle's ListKernelSessionOutput RPC can return HTTP 404 while a newly created
private kernel version has metadata but no published session output yet. The
v1 finalizer treated that state as terminal.  This wrapper preserves every v1
verification/submission guard and converts only that exact 404 into an empty,
non-terminal output snapshot so the existing bounded polling loop continues.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace
from typing import Any

import requests

CORE_FILENAME = "cmi_flu_b2_finalize.py"
FINALIZE_REQUEST_ID = "20260903-cmi-flu-b2-finalize-002"
TRANSIENT_OUTPUT_STATUS_CODES = frozenset({404})


def load_core() -> Any:
    core_path = pathlib.Path(__file__).with_name(CORE_FILENAME)
    if not core_path.is_file() or core_path.is_symlink():
        raise SystemExit("pinned B2 finalizer core is absent or unsafe")
    spec = importlib.util.spec_from_file_location("cmi_flu_b2_finalize_core_v1", core_path)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load pinned B2 finalizer core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_transient_output_guard(core: Any) -> None:
    original = core.read_kernel_output
    transient_count = 0

    def guarded_read_kernel_output(api: Any) -> Any:
        nonlocal transient_count
        try:
            return original(api)
        except requests.exceptions.HTTPError as error:
            response = getattr(error, "response", None)
            status_code = getattr(response, "status_code", None)
            if status_code not in TRANSIENT_OUTPUT_STATUS_CODES:
                raise
            transient_count += 1
            if transient_count == 1 or transient_count % 5 == 0:
                print(
                    "CMI_FLU_B2_OUTPUT_API_PENDING "
                    f"status={status_code} transient_count={transient_count}"
                )
            return SimpleNamespace(log="", files=[])

    core.read_kernel_output = guarded_read_kernel_output


def main() -> int:
    core = load_core()
    core.FINALIZE_REQUEST_ID = FINALIZE_REQUEST_ID
    install_transient_output_guard(core)
    return int(core.main())


if __name__ == "__main__":
    raise SystemExit(main())
