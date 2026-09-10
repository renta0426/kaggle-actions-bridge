#!/usr/bin/env python3
"""Validate E11b 004 package transport without requiring network in CI."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import subprocess
from pathlib import Path

WHEEL_FILENAME = "tabpfn-8.5.0-py3-none-any.whl"
WHEEL_BYTES = 771_167
WHEEL_SHA256 = "4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0"
MAX_RUNTIME_BYTES = 900_000


def load_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("e11b_004_transport_contract", path)
    if spec is None or spec.loader is None:
        raise SystemExit("E11b 004 transport test cannot load runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    runtime_path = args.runtime.resolve()
    wheel_path = args.wheel.resolve()
    raw = runtime_path.read_bytes()
    text = raw.decode("utf-8")
    if len(raw) >= MAX_RUNTIME_BYTES:
        raise SystemExit(f"E11b 004 runtime source budget failed:{len(raw)}")
    required = (
        'TABPFN_WHEEL_B64 = ""',
        "TABPFN_WHEEL_BYTES = 771167",
        '"pip","download"',
        '"--no-deps"',
        '"--only-binary=:all:"',
        "import subprocess as _subprocess",
        "e11b_tabpfn_wheel_bytes",
        "e11b_tabpfn_wheel_sha",
        "pypi_exact_hash_verified",
    )
    for token in required:
        if token not in text:
            raise SystemExit(f"E11b 004 runtime transport token missing:{token}")
    if not wheel_path.is_file() or wheel_path.name != WHEEL_FILENAME:
        raise SystemExit("E11b 004 fixture filename mismatch")
    wheel_data = wheel_path.read_bytes()
    if len(wheel_data) != WHEEL_BYTES or hashlib.sha256(wheel_data).hexdigest() != WHEEL_SHA256:
        raise SystemExit("E11b 004 fixture identity mismatch")

    runtime = load_runtime(runtime_path)
    original_run = subprocess.run
    calls: list[list[str]] = []

    def fake_run(command, *pargs, **kwargs):
        cmd = [str(x) for x in command]
        calls.append(cmd)
        if "download" in cmd and "tabpfn==8.5.0" in cmd:
            dest = Path(cmd[cmd.index("--dest") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(wheel_path, dest / WHEEL_FILENAME)
            return subprocess.CompletedProcess(cmd, 0, "fixture-download\n", "")
        return original_run(command, *pargs, **kwargs)

    subprocess.run = fake_run
    try:
        result = runtime.install_tabpfn_wheel()
    finally:
        subprocess.run = original_run
    if result.get("version") != "8.5.0":
        raise SystemExit("E11b 004 installed package version mismatch")
    if result.get("wheel_sha256") != WHEEL_SHA256 or int(result.get("wheel_bytes", -1)) != WHEEL_BYTES:
        raise SystemExit("E11b 004 installed wheel identity mismatch")
    if result.get("transport") != "pypi_exact_hash_verified":
        raise SystemExit("E11b 004 transport marker mismatch")
    download_calls = [c for c in calls if "download" in c]
    install_calls = [c for c in calls if "install" in c]
    if len(download_calls) != 1 or len(install_calls) != 1:
        raise SystemExit("E11b 004 pip call cardinality mismatch")
    if "--no-deps" not in download_calls[0] or "--only-binary=:all:" not in download_calls[0]:
        raise SystemExit("E11b 004 download flags changed")
    if "--no-deps" not in install_calls[0]:
        raise SystemExit("E11b 004 install flags changed")
    print(
        "CMI_FLU_E11B_004_ONLINE_INSTALLER_CONTRACT_PASS "
        f"runtime_bytes={len(raw)} exact_wheel=true download_calls=1 install_calls=1 network_used=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
