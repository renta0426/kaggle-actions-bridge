#!/usr/bin/env python3
"""E11b 004 repair: remove embedded wheel and retrieve the same frozen wheel at runtime."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PRIOR = Path(__file__).with_name("cmi_flu_strategy_e11b_prepare_v4.py")
PRIOR_BLOB = "44d36b6c7a3b48845159e1b1f0e69da12a1fbeea"
OLD_REQUEST = "20260910-cmi-flu-strategy-e11b-tabpfn3-003"
OLD_TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-003"
NEW_REQUEST = "20260910-cmi-flu-strategy-e11b-tabpfn3-004"
NEW_TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-004"
REQUEST_PATH = "requests/cmi-flu-strategy-e11b-tabpfn3-004.json"
PARENT_REQUEST_PATH = "requests/cmi-flu-strategy-e11b-tabpfn3-003.json"
WHEEL_FILENAME = "tabpfn-8.5.0-py3-none-any.whl"
WHEEL_BYTES = 771_167
WHEEL_SHA256 = "4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0"
MAX_RUNTIME_BYTES = 900_000


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--tabpfn-wheel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def replace_function(text: str, name: str, replacement: str) -> str:
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(nodes) != 1:
        raise SystemExit(f"E11b 004 function binding changed:{name}:{len(nodes)}")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    return "".join(lines[: node.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno :])


def validate_request(root: Path) -> None:
    current = json.loads((root / REQUEST_PATH).read_text())
    parent = json.loads((root / PARENT_REQUEST_PATH).read_text())
    if current.get("request_id") != NEW_REQUEST or current.get("parent_request_id") != OLD_REQUEST:
        raise SystemExit("E11b 004 request identity mismatch")
    if current.get("target") != NEW_TARGET or current.get("enable_internet") is not True:
        raise SystemExit("E11b 004 target/internet contract mismatch")
    for key in (
        "competition", "science_repository", "science_source_commit", "science_transport",
        "strategy_e11b_blob_sha", "strategy_e11b_synthetic_blob_sha", "dependency_blobs",
        "expected_kernel_version", "resource", "api_budget", "model_access_preflight",
        "experiment_contract", "allowed_output_paths", "automatic_compute_retries",
        "competition_submission_attempted", "leaderboard_used_for_selection", "publish_only_sanitized_aggregate",
    ):
        if current.get(key) != parent.get(key):
            raise SystemExit(f"E11b 004 science/runtime drift:{key}")
    cur_ext = dict(current.get("external_model") or {})
    par_ext = dict(parent.get("external_model") or {})
    if cur_ext.pop("wheel_bytes", None) != WHEEL_BYTES or cur_ext != par_ext:
        raise SystemExit("E11b 004 external model drift")
    provenance = current.get("repair_provenance") or {}
    expected_provenance = {
        "failed_request_id": OLD_REQUEST,
        "failed_target": OLD_TARGET,
        "failed_actions_run": 34440743047,
        "failed_actions_job": 102755230624,
        "failure_stage": "save_kernel_source_validation",
        "failure_http_status": 400,
        "failure_response_body_bytes": 119,
        "failure_response_body_sha256": "6cb47ba6e12dfa7ecf97850018d8b9deaad8a92714c5e882d6aef10a5d79ef4c",
        "failure_runtime_sha256": "8df0a934a1eadcce52335f8087db3820649accb492d72aa8ffe66062d4a14449",
        "model_access_succeeded": True,
        "kaggle_compute_started": False,
        "root_cause": "embedded base64 TabPFN wheel expanded runtime source above Kaggle SaveKernel 1 MB source limit",
        "repair_scope": "dependency_transport_only_online_exact_wheel_download_and_hash_verify",
    }
    if provenance != expected_provenance:
        raise SystemExit("E11b 004 repair provenance mismatch")
    transport = current.get("dependency_transport") or {}
    expected_transport = {
        "package_source": "PyPI",
        "package_spec": "tabpfn==8.5.0",
        "download_only_binary": True,
        "download_no_deps": True,
        "verify_filename": True,
        "verify_bytes": WHEEL_BYTES,
        "verify_sha256": WHEEL_SHA256,
        "install_no_deps": True,
        "embedded_wheel_allowed": False,
        "runtime_source_max_bytes": MAX_RUNTIME_BYTES,
        "internet_use_scope": "retrieve the exact frozen tabpfn wheel only; model checkpoint remains the attached official Kaggle Model input",
    }
    if transport != expected_transport:
        raise SystemExit("E11b 004 dependency transport mismatch")
    write_transport = current.get("transport") or {}
    if write_transport != {
        "kernel_write_method": "KaggleApi.kernels_push",
        "cli_kernels_push_used": False,
        "structured_failure_receipt_required": True,
        "raw_server_error_persisted": False,
        "confirm_failed_003_target_absent_before_write": True,
    }:
        raise SystemExit("E11b 004 write transport mismatch")


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    prior_data = PRIOR.read_bytes()
    if git_blob_sha(prior_data) != PRIOR_BLOB:
        raise SystemExit("E11b prepare-v5 ancestry changed")
    validate_request(root)
    wheel = args.tabpfn_wheel.resolve()
    if wheel.name != WHEEL_FILENAME or not wheel.is_file():
        raise SystemExit("E11b 004 wheel path contract failed")
    wheel_data = wheel.read_bytes()
    if len(wheel_data) != WHEEL_BYTES or hashlib.sha256(wheel_data).hexdigest() != WHEEL_SHA256:
        raise SystemExit("E11b 004 wheel identity mismatch")

    with tempfile.TemporaryDirectory(prefix="cmi-flu-e11b-v5-") as tmp:
        prior_runtime = Path(tmp) / "runtime.py"
        subprocess.run(
            [sys.executable, str(PRIOR), "--repository-root", str(root),
             "--reference-dir", str(args.reference_dir.resolve()),
             "--tabpfn-wheel", str(wheel), "--output", str(prior_runtime)],
            check=True,
        )
        runtime = prior_runtime.read_text()

    if runtime.count(OLD_REQUEST) < 1 or runtime.count(OLD_TARGET) < 1:
        raise SystemExit("E11b 004 runtime identity anchors missing")
    runtime = runtime.replace(OLD_REQUEST, NEW_REQUEST).replace(OLD_TARGET, NEW_TARGET)
    if OLD_REQUEST in runtime or OLD_TARGET in runtime:
        raise SystemExit("E11b 004 old identity remained")

    lines = runtime.splitlines(keepends=True)
    wheel_lines = [i for i, line in enumerate(lines) if line.startswith("TABPFN_WHEEL_B64 = ")]
    if len(wheel_lines) != 1:
        raise SystemExit("E11b 004 embedded wheel binding changed")
    idx = wheel_lines[0]
    encoded = ast.literal_eval(lines[idx].split("=", 1)[1].strip())
    decoded = base64.b64decode(encoded.encode("ascii"), validate=True)
    if len(decoded) != WHEEL_BYTES or hashlib.sha256(decoded).hexdigest() != WHEEL_SHA256:
        raise SystemExit("E11b 004 prior embedded wheel identity changed")
    lines[idx] = f'TABPFN_WHEEL_B64 = ""\nTABPFN_WHEEL_BYTES = {WHEEL_BYTES}\n'
    runtime = "".join(lines)

    installer = r'''def install_tabpfn_wheel() -> dict:
    import importlib.metadata as _metadata
    root=Path("/tmp")/"cmi-flu-e11b-tabpfn-site"
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    download=subprocess.run(
        [sys.executable,"-m","pip","download","--disable-pip-version-check","--no-input","--no-deps","--only-binary=:all:","--dest",str(root),f"tabpfn=={TABPFN_PACKAGE_VERSION}"],
        capture_output=True,text=True,timeout=180,check=False,
    )
    if download.returncode!=0:
        code=hashlib.sha256((download.stdout+download.stderr).encode("utf-8",errors="replace")).hexdigest()[:20]
        raise BridgeContractError(f"e11b_tabpfn_download_failed:{code}")
    wheels=[p for p in root.iterdir() if p.is_file() and p.name==TABPFN_WHEEL_FILENAME and not p.is_symlink()]
    if len(wheels)!=1:
        raise BridgeContractError(f"e11b_tabpfn_wheel_count:{len(wheels)}")
    wheel_path=wheels[0]
    wheel_bytes=wheel_path.read_bytes()
    if len(wheel_bytes)!=TABPFN_WHEEL_BYTES:
        raise BridgeContractError("e11b_tabpfn_wheel_bytes")
    if hashlib.sha256(wheel_bytes).hexdigest()!=TABPFN_WHEEL_SHA256:
        raise BridgeContractError("e11b_tabpfn_wheel_sha")
    site=root/"site"; site.mkdir()
    install=subprocess.run(
        [sys.executable,"-m","pip","install","--disable-pip-version-check","--no-input","--no-deps","--target",str(site),str(wheel_path)],
        capture_output=True,text=True,timeout=180,check=False,
    )
    if install.returncode!=0:
        code=hashlib.sha256((install.stdout+install.stderr).encode("utf-8",errors="replace")).hexdigest()[:20]
        raise BridgeContractError(f"e11b_tabpfn_install_failed:{code}")
    sys.path.insert(0,str(site))
    found=_metadata.version("tabpfn")
    if found!=TABPFN_PACKAGE_VERSION:
        raise BridgeContractError("e11b_tabpfn_version")
    return {"version":found,"wheel_sha256":TABPFN_WHEEL_SHA256,"wheel_bytes":TABPFN_WHEEL_BYTES,"site":str(site),"transport":"pypi_exact_hash_verified"}
'''
    runtime = replace_function(runtime, "install_tabpfn_wheel", installer)

    selftest = r'''def self_test() -> int:
    package=package_bytes()
    for source,expected,label in ((E11B_SOURCE,E11B_BLOB,"e11b"),(E11B_SYNTH_SOURCE,E11B_SYNTH_BLOB,"e11b_synthetic")):
        if git_blob_sha(source.encode("utf-8"))!=expected: raise BridgeContractError(f"{label}_blob_mismatch")
        compile(source,f"cmi_flu/{label}.py","exec")
    if TABPFN_WHEEL_B64!="": raise BridgeContractError("e11b_embedded_wheel_present")
    if TABPFN_WHEEL_FILENAME!="tabpfn-8.5.0-py3-none-any.whl": raise BridgeContractError("e11b_wheel_filename_selftest")
    if int(TABPFN_WHEEL_BYTES)!=771167: raise BridgeContractError("e11b_wheel_bytes_selftest")
    if TABPFN_WHEEL_SHA256!="4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0": raise BridgeContractError("e11b_wheel_sha_selftest")
    if not callable(install_tabpfn_wheel): raise BridgeContractError("e11b_installer_selftest")
    if "result['tasks']" in globals().get("__loader_source__",""): raise BridgeContractError("e11b_writer_legacy")
    print(f"CMI_FLU_E11B_RUNTIME_SELF_TEST PASS request_id={REQUEST_ID} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} e11b_blob={E11B_BLOB} wheel_bytes={TABPFN_WHEEL_BYTES} wheel_transport=pypi_exact_hash_verified")
    return 0
'''
    runtime = replace_function(runtime, "self_test", selftest)

    compile(runtime, "generated_e11b_004_runtime.py", "exec")
    raw = runtime.encode("utf-8")
    if len(raw) >= MAX_RUNTIME_BYTES:
        raise SystemExit(f"E11b 004 runtime remains too large:{len(raw)}")
    if encoded[:128] in runtime or len(encoded) < 1_000_000:
        raise SystemExit("E11b 004 embedded wheel payload removal failed")
    if 'TABPFN_WHEEL_B64 = ""' not in runtime or 'pip","download"' not in runtime:
        raise SystemExit("E11b 004 online wheel installer anchors missing")

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime)
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E11B_PREPARE_V5_PASS repair=online_exact_wheel_transport "
        f"request_id={NEW_REQUEST} target={NEW_TARGET} science_change=false "
        f"runtime_bytes={len(raw)} runtime_sha256={hashlib.sha256(raw).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
