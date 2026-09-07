#!/usr/bin/env python3
"""Build E05 003 runtime with AST-scoped reference transport repair."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-003"
TARGET_KERNEL = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-003"
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


def load_v1(root: Path):
    path = root / OLD_PREPARE
    spec = importlib.util.spec_from_file_location("cmi_flu_e05_prepare_v1_for_v3", path)
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
        data = (reference_dir / name).read_bytes()
        if sha256(data) != expected:
            raise SystemExit(f"locked reference hash mismatch:{name}")
        result[name] = data
    return result


def build_v1_runtime(old, root: Path) -> str:
    base = old.load_base(root)
    e01, e01v2, config = base.load_exact_science(root)
    hai, e05 = old.load_science(root)
    with tempfile.TemporaryDirectory(prefix="cmi-e05-v3-base-") as tmp:
        package, adapter = base.extract_frozen_runtime(root, Path(tmp))
        base.REQUEST_ID = OLD_REQUEST_ID
        base.TARGET_KERNEL = OLD_TARGET_KERNEL
        base.SCIENCE_COMMIT = SCIENCE_COMMIT
        runtime = base.build_runtime(package, adapter, e01, e01v2, config)
    return old.patch_runtime(runtime, hai, e05)


def replace_top_level_function(text: str, name: str, replacement: str) -> str:
    tree = ast.parse(text)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(nodes) != 1:
        raise SystemExit(f"top-level function contract changed:{name}:{len(nodes)}")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1])
    end = sum(len(line) for line in lines[: node.end_lineno])
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


def top_level_functions(text: str) -> list[str]:
    return [node.name for node in ast.parse(text).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def patch_runtime(runtime: str, references: dict[str, bytes]) -> str:
    if OLD_REQUEST_ID not in runtime or OLD_TARGET_KERNEL not in runtime:
        raise SystemExit("E05 v1 identity anchors missing")
    runtime = runtime.replace(OLD_REQUEST_ID, REQUEST_ID).replace(OLD_TARGET_KERNEL, TARGET_KERNEL)

    marker = f"LOCKED_REFERENCE_SHA256 = {REF_SHA256!r}\n"
    if runtime.count(marker) != 1:
        raise SystemExit("locked reference digest anchor changed")
    encoded = {name: base64.b64encode(data).decode("ascii") for name, data in references.items()}
    runtime = runtime.replace(marker, marker + f"LOCKED_REFERENCE_B64 = {encoded!r}\n", 1)

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
    runtime = replace_top_level_function(runtime, "locate_locked_reference", replacement)

    names = top_level_functions(runtime)
    if names.count("json_safe") != 1 or names.count("locate_locked_reference") != 1 or names.count("locked_reference_bytes") != 1:
        raise SystemExit("E05 v3 runtime binding contract failed")
    if OLD_REQUEST_ID in runtime or OLD_TARGET_KERNEL in runtime or "20260907-cmi-flu-strategy-e05-hai-donor-strain-002" in runtime:
        raise SystemExit("prior E05 identity remained in v3 runtime")
    if "competition_submit" in runtime or "kaggle competitions submit" in runtime:
        raise SystemExit("E05 v3 runtime contains submission path")
    compile(runtime, "generated_e05_v3.py", "exec")
    return runtime


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    refs = read_references(args.reference_dir.expanduser().resolve())
    old = load_v1(root)
    runtime = patch_runtime(build_v1_runtime(old, root), refs)
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E05_V3_BUILD PASS "
        f"science_commit={SCIENCE_COMMIT} request_id={REQUEST_ID} target={TARGET_KERNEL} "
        f"runtime_sha256={sha256(runtime.encode())} references=embedded_verified json_safe=preserved"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
