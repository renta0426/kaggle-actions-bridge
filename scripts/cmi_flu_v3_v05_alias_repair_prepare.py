#!/usr/bin/env python3
"""Build the fresh V3-05 Task1.3 study-alias repair runtime."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import tempfile

import cmi_flu_v3_v05_prepare as parent
from cmi_flu_v3_batch1_prepare import replace_function

REQUEST_ID = "20260912-cmi-flu-strategy-v3-v05-task13-scale-002"
TARGET = "renta0426/cmi-flu-v3-v05-task13-scale-20260912-002"
PAYLOAD = "payloads/cmi-flu-v3-v05-task13-alias-repair-002/strategy_v3_v05_v2.py"
REPAIR_SCIENCE_COMMIT = "f8641f7489f22478b8caf3430d2d432e617c4a25"
REPAIR_SCIENCE_BLOB = "493a8ffa7caed30796a5fe6711577aee6c58905c"
BASE_SCIENCE_COMMIT = parent.SCIENCE_COMMIT
BASE_SCIENCE_BLOB = parent.SCIENCE_BLOB
FAILURE_CODE = "88315152273aaa70245f"
ALIAS_HASH = "6626de8ebc839dbdac44faa19e8c3280348325d2a214fe940984a9d209fe2985"


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def load_repair_source(root: Path) -> str:
    raw = (root / PAYLOAD).read_bytes()
    if git_blob(raw) != REPAIR_SCIENCE_BLOB:
        raise SystemExit("V3-05 repair exact science blob mismatch")
    source = raw.decode("utf-8")
    compile(source, "cmi_flu/strategy_v3_v05_v2.py", "exec")
    required = (
        'FROZEN_STUDY_ALIASES = {"2024UGA": "2024UGA", "2024_UGA": "2024UGA"}',
        f'OBSERVED_FAILURE_CODE = "{FAILURE_CODE}"',
        f'EXPECTED_ALIAS_HASH = "{ALIAS_HASH}"',
        '"scope": "participant_context.study_group_identity_only"',
        '"candidate_formulas_changed": False',
        '"competition_submission_attempted": False',
    )
    if any(token not in source for token in required):
        raise SystemExit("V3-05 repair source contract token missing")
    lowered = source.casefold()
    if any(token in lowered for token in ("competition_submit(", "kaggle competitions submit", "competitions submit")):
        raise SystemExit("V3-05 repair source contains Competition submit path")
    return source


def build_runtime(root: Path, output: Path) -> str:
    source = load_repair_source(root)
    with tempfile.TemporaryDirectory(prefix="cmi-v305-repair-parent-") as tmp:
        raw = parent.build_runtime(root, Path(tmp) / "parent.py")
    raw = raw.replace(parent.REQUEST_ID, REQUEST_ID).replace(parent.TARGET, TARGET)

    marker = f'V305_SCIENCE_BLOB = {BASE_SCIENCE_BLOB!r}\n'
    if raw.count(marker) != 1:
        raise SystemExit("V3-05 repair provenance insertion anchor changed")
    raw = raw.replace(
        marker,
        marker
        + f'V305_REPAIR_SCIENCE_COMMIT = {REPAIR_SCIENCE_COMMIT!r}\n'
        + f'V305_REPAIR_SCIENCE_BLOB = {REPAIR_SCIENCE_BLOB!r}\n'
        + f'V305_REPAIR_SOURCE = {source!r}\n'
        + f'V305_REPAIR_PARENT_FAILURE_CODE = {FAILURE_CODE!r}\n'
        + f'V305_REPAIR_ALIAS_HASH = {ALIAS_HASH!r}\n',
        1,
    )

    loader = r'''def load_v3_v05_v2_module() -> object:
    import sys, types
    load_v3_v05_module()
    module = types.ModuleType("cmi_flu.strategy_v3_v05_v2")
    module.__file__ = "<cmi_flu.strategy_v3_v05_v2>"
    module.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_v3_v05_v2"] = module
    exec(compile(V305_REPAIR_SOURCE, "cmi_flu/strategy_v3_v05_v2.py", "exec"), module.__dict__, module.__dict__)
    for name in ("run_v3_05_task13", "write_v3_05_outputs", "observed_failure_code", "study_alias_hash"):
        if not callable(getattr(module, name, None)):
            raise BridgeContractError("v305_repair_entry_missing:" + name)
    if module.observed_failure_code() != V305_REPAIR_PARENT_FAILURE_CODE:
        raise BridgeContractError("v305_repair_failure_fingerprint_changed")
    if module.study_alias_hash() != V305_REPAIR_ALIAS_HASH:
        raise BridgeContractError("v305_repair_alias_hash_changed")
    if tuple(getattr(module, "NEW_CONDITIONS", ())) != ("S1", "S2", "S3", "S4") or int(getattr(module, "MAX_FITS", -1)) != 96:
        raise BridgeContractError("v305_repair_science_boundary_changed")
    return module
'''
    anchor = "def execute(input_dir: Path, output_dir: Path) -> int:\n"
    if raw.count(anchor) != 1:
        raise SystemExit("V3-05 repair loader insertion anchor changed")
    raw = raw.replace(anchor, loader + "\n" + anchor, 1)

    execute_nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "execute"]
    if len(execute_nodes) != 1:
        raise SystemExit("V3-05 repair execute binding changed")
    execute_source = ast.get_source_segment(raw, execute_nodes[0])
    old_call = "            v305 = load_v3_v05_module()"
    if execute_source.count(old_call) != 1:
        raise SystemExit("V3-05 repair execute science-loader call changed")
    execute_source = execute_source.replace(old_call, "            v305 = load_v3_v05_v2_module()", 1)
    runtime_anchor = '                "science_blob": V305_SCIENCE_BLOB,\n'
    if execute_source.count(runtime_anchor) != 1:
        raise SystemExit("V3-05 repair runtime provenance anchor changed")
    execute_source = execute_source.replace(
        runtime_anchor,
        runtime_anchor
        + '                "repair_science_commit": V305_REPAIR_SCIENCE_COMMIT,\n'
        + '                "repair_science_blob": V305_REPAIR_SCIENCE_BLOB,\n'
        + '                "repair_parent_failure_code": V305_REPAIR_PARENT_FAILURE_CODE,\n'
        + '                "repair_study_alias_sha256": V305_REPAIR_ALIAS_HASH,\n',
        1,
    )
    raw = replace_function(raw, "execute", execute_source)

    self_test = r'''def self_test() -> int:
    import ast, sys, tempfile
    if git_blob_sha(V305_SOURCE.encode()) != V305_SCIENCE_BLOB:
        raise BridgeContractError("v305_repair_base_blob_mismatch")
    if git_blob_sha(V305_REPAIR_SOURCE.encode()) != V305_REPAIR_SCIENCE_BLOB:
        raise BridgeContractError("v305_repair_blob_mismatch")
    for science in (V305_SOURCE, V305_REPAIR_SOURCE):
        tree = ast.parse(science)
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and "submit" in n.func.attr.casefold() for n in ast.walk(tree)):
            raise BridgeContractError("v305_repair_submit_call_present")
    with tempfile.TemporaryDirectory(prefix="v305-repair-selftest-") as tmp:
        package = Path(tmp) / "cmi_flu_bundle.zip"; package.write_bytes(package_bytes()); sys.path.insert(0, str(package))
        try:
            module = load_v3_v05_v2_module()
            if module.observed_failure_code() != V305_REPAIR_PARENT_FAILURE_CODE:
                raise BridgeContractError("v305_repair_failure_code")
            if module.study_alias_hash() != V305_REPAIR_ALIAS_HASH:
                raise BridgeContractError("v305_repair_alias_digest")
            if module.canonical_study("2024_UGA") != "2024UGA" or module.canonical_study("2024 UGA") != "2024 UGA":
                raise BridgeContractError("v305_repair_alias_scope")
        finally:
            if sys.path and sys.path[0] == str(package): sys.path.pop(0)
    print(
        f"CMI_FLU_V3_V05_REPAIR_SELF_TEST PASS request_id={V305_REQUEST_ID} base_blob={V305_SCIENCE_BLOB} "
        f"repair_commit={V305_REPAIR_SCIENCE_COMMIT} repair_blob={V305_REPAIR_SCIENCE_BLOB} "
        f"parent_failure={V305_REPAIR_PARENT_FAILURE_CODE} alias_hash={V305_REPAIR_ALIAS_HASH} conditions=4 fit_limit=96 submission=false"
    )
    return 0
'''
    raw = replace_function(raw, "self_test", self_test)
    compile(raw, "v3-v05-repair-runtime.py", "exec")
    if len(raw.encode()) >= 1100000:
        raise SystemExit("V3-05 repair runtime exceeds reviewed byte budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(raw, encoding="utf-8")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = build_runtime(args.repository_root.resolve(), args.output.resolve()).encode()
    print(
        f"CMI_FLU_V3_V05_REPAIR_PREPARE PASS bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()} "
        f"repair_blob={REPAIR_SCIENCE_BLOB} parent_failure={FAILURE_CODE} submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
