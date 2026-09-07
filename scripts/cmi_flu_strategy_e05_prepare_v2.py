#!/usr/bin/env python3
"""Build E05 repair runtime with hash-verified organizer references embedded in code."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-002"
TARGET_KERNEL = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-002"
SCIENCE_COMMIT = "e02a601f38250480b526e548a48cf7f2526c00ca"
OLD_REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-001"
OLD_TARGET_KERNEL = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-001"
OLD_PREPARE = "scripts/cmi_flu_strategy_e05_prepare.py"
REF_SHA256 = {
    "strain_sequences.csv": "63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887",
    "vaccine_strains_per_season.txt": "8f6c7116f37f29df0bb21d6049d82fa28b4e42b2d10ed9394a1ae6f926bd9f35",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_old(root: Path):
    path = root / OLD_PREPARE
    spec = importlib.util.spec_from_file_location("cmi_flu_e05_prepare_v1_for_v2", path)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load E05 v1 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        "REQUEST_ID": OLD_REQUEST_ID,
        "TARGET_KERNEL": OLD_TARGET_KERNEL,
        "SCIENCE_COMMIT": SCIENCE_COMMIT,
        "REF_SHA256": REF_SHA256,
        "E05_BLOB": "78cde9e3a6b6e3b332352218c4b4545f769aa6c4",
        "HAI_TRANSFER_BLOB": "b671d8bf7f10bebbd65aca2a5bad42e267ee78d5",
    }
    for key, value in expected.items():
        if getattr(module, key, None) != value:
            raise SystemExit(f"E05 v1 builder contract changed:{key}")
    return module


def read_references(reference_dir: Path) -> dict[str, bytes]:
    result = {}
    for name, expected in REF_SHA256.items():
        path = reference_dir / name
        data = path.read_bytes()
        if sha256(data) != expected:
            raise SystemExit(f"locked reference hash mismatch:{name}")
        result[name] = data
    return result


def build_v1_runtime(old, root: Path) -> str:
    base = old.load_base(root)
    e01, e01v2, config = base.load_exact_science(root)
    hai, e05 = old.load_science(root)
    with tempfile.TemporaryDirectory(prefix="cmi-e05-v2-base-") as tmp:
        package, adapter = base.extract_frozen_runtime(root, Path(tmp))
        base.REQUEST_ID = OLD_REQUEST_ID
        base.TARGET_KERNEL = OLD_TARGET_KERNEL
        base.SCIENCE_COMMIT = SCIENCE_COMMIT
        runtime = base.build_runtime(package, adapter, e01, e01v2, config)
    return old.patch_runtime(runtime, hai, e05)


def patch_runtime(old, runtime: str, references: dict[str, bytes]) -> str:
    if runtime.count(OLD_REQUEST_ID) < 1 or runtime.count(OLD_TARGET_KERNEL) < 1:
        raise SystemExit("E05 v1 runtime identity anchors missing")
    runtime = runtime.replace(OLD_REQUEST_ID, REQUEST_ID).replace(OLD_TARGET_KERNEL, TARGET_KERNEL)

    marker = f"LOCKED_REFERENCE_SHA256 = {REF_SHA256!r}\n"
    if runtime.count(marker) != 1:
        raise SystemExit("locked reference digest anchor changed")
    encoded = {name: base64.b64encode(data).decode("ascii") for name, data in references.items()}
    injected = marker + f"LOCKED_REFERENCE_B64 = {encoded!r}\n"
    runtime = runtime.replace(marker, injected, 1)

    replacement = '''def locked_reference_bytes(name: str) -> bytes:
    if name not in LOCKED_REFERENCE_SHA256 or name not in LOCKED_REFERENCE_B64:
        raise BridgeContractError("locked_reference_name")
    try:
        data = base64.b64decode(LOCKED_REFERENCE_B64[name].encode("ascii"), validate=True)
    except Exception as exc:
        raise BridgeContractError("locked_reference_base64") from exc
    if hashlib.sha256(data).hexdigest() != LOCKED_REFERENCE_SHA256[name]:
        raise BridgeContractError(f"locked_reference_embedded_hash:{name}")
    return data


def locate_locked_reference(name: str) -> Path:
    data = locked_reference_bytes(name)
    base = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
    root = base / ".e05-locked-references"
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != LOCKED_REFERENCE_SHA256[name]:
        raise BridgeContractError(f"locked_reference_existing_hash:{name}")
    if not path.exists():
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != LOCKED_REFERENCE_SHA256[name]:
        raise BridgeContractError(f"locked_reference_materialized_hash:{name}")
    return path'''
    runtime = old.replace_function(runtime, "def locate_locked_reference(name: str) -> Path:\n", "def validate_result(", replacement)

    self_start = "def self_test() -> int:\n"
    self_end = "def locate_competition_data("
    a = runtime.find(self_start)
    b = runtime.find(self_end, a + len(self_start))
    if a < 0 or b < 0:
        raise SystemExit("self-test anchors changed")
    block = runtime[a:b]
    needle = '    compile(E05_SOURCE, "cmi_flu/strategy_e05.py", "exec")\n'
    if block.count(needle) != 1:
        raise SystemExit("self-test E05 compile anchor changed")
    extra = needle + '    for name, expected in LOCKED_REFERENCE_SHA256.items():\n        if hashlib.sha256(locked_reference_bytes(name)).hexdigest() != expected:\n            raise BridgeContractError(f"locked_reference_self_test:{name}")\n'
    block = block.replace(needle, extra, 1)
    runtime = runtime[:a] + block + runtime[b:]

    if OLD_REQUEST_ID in runtime or OLD_TARGET_KERNEL in runtime:
        raise SystemExit("old E05 identity remained in repaired runtime")
    if "LOCKED_REFERENCE_B64" not in runtime or "locked_reference_bytes" not in runtime:
        raise SystemExit("embedded reference transport missing")
    if "competition_submit" in runtime or "kaggle competitions submit" in runtime:
        raise SystemExit("repaired E05 runtime contains submission path")
    compile(runtime, "generated_e05_v2.py", "exec")
    return runtime


def main():
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    references = read_references(args.reference_dir.expanduser().resolve())
    old = load_old(root)
    runtime = patch_runtime(old, build_v1_runtime(old, root), references)
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E05_V2_BUILD PASS "
        f"science_commit={SCIENCE_COMMIT} request_id={REQUEST_ID} target={TARGET_KERNEL} "
        f"runtime_sha256={sha256(runtime.encode())} references=embedded_verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
