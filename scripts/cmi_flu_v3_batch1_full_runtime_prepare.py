#!/usr/bin/env python3
"""Build V3 batch1 004; keep science logic unchanged and test the real entry path."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile

import cmi_flu_v3_batch1_task12_audit_repair_prepare as parent
from cmi_flu_v3_batch1_prepare import replace_function

REQUEST = "requests/cmi-flu-v3-batch1-full-runtime-repair-004.json"
REQUEST_ID = "20260912-cmi-flu-strategy-v3-batch1-full-runtime-repair-004"
TARGET = "renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260912-004"
SCIENCE_COMMIT = "9a3b7dd8d736da8d8074ec0898e7a10f1774aba2"
LAZY_BLOB = "5c6725dc757a5ba9dd21289b1c4f09997e1afdb8"
LAZY_PATH = "payloads/cmi-flu-v3-batch1-full-runtime-repair-004/task13_harmonization.py"
FIXTURE_BLOB = "fbffefc766df6ec52e312276be4953df2067f40c"
SOURCE_B_SHA = "0f9df53c3aa8c6e4ac693f6a42dbd2633b4b61d1462df798a9bd767c220be3a5"


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def validate_request(root: Path) -> dict:
    r = json.loads((root / REQUEST).read_text())
    exact = {"request_id": REQUEST_ID, "target": TARGET, "title": TARGET.split("/", 1)[1],
             "operation": "save_kernel_once", "execution_policy": "kaggle_native_capacity_v2",
             "science_source_commit": SCIENCE_COMMIT, "expected_version": 1,
             "automatic_compute_retries": 0, "enable_internet": False,
             "competition_submission_authorized": False, "model_fit_allowed": False,
             "refit_allowed": False, "hpo_allowed": False,
             "scientific_audit_logic_changed": False, "reference_expectations_mutated": False,
             "prediction_logic_changed": False, "synthetic_data_allowed_in_production": False}
    if any(r.get(k) != v for k, v in exact.items()):
        raise ValueError("V3 full-runtime request boundary mismatch")
    b = r["source_B"]
    if (b["source_kernel"], b["expected_current_version"], b["bytes"], b["sha256"]) != (
        "renta0426/cmi-flu-e12c-manual-submission-20260911-002", 1, 5926, SOURCE_B_SHA):
        raise ValueError("V3 full-runtime immutable source B changed")
    failure = r["parent_runtime_failure"]
    code = hashlib.sha256(b"run_no_fit_audit:ModuleNotFoundError:No module named 'cmi_flu.task13_harmonization'").hexdigest()[:20]
    if failure["code"] != code or failure["actions_run"] != 34623685474 or failure["version"] != 1:
        raise ValueError("V3 full-runtime parent failure proof changed")
    if failure["same_target_version_rerun_forbidden"] is not True:
        raise ValueError("V3 full-runtime consumed version must not be reused")
    if r["science_blobs"]["task13_harmonization.py"] != LAZY_BLOB:
        raise ValueError("V3 lazy dependency identity changed")
    return r


def function_source(source: str, name: str) -> str:
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f"generated function binding changed:{name}")
    return ast.get_source_segment(source, nodes[0]) + "\n"


def build_runtime(root: Path, output: Path) -> str:
    request = validate_request(root)
    lazy = (root / LAZY_PATH).read_bytes()
    if git_blob(lazy) != LAZY_BLOB:
        raise ValueError("lazy dependency is not exact science blob")
    if git_blob((root / "tests/v3_batch1_integration_support.py").read_bytes()) != FIXTURE_BLOB:
        raise ValueError("CI-only synthetic fixture relay mismatch")
    with tempfile.TemporaryDirectory(prefix="v3-004-build-") as tmp:
        raw = parent.build_runtime(root, Path(tmp) / "parent.py")
    raw = raw.replace(parent.REQUEST_ID, REQUEST_ID).replace(parent.TARGET, TARGET)
    constants = (f'V3_FULL_RUNTIME_SCIENCE_COMMIT = {SCIENCE_COMMIT!r}\n'
                 f'V3_LAZY_DEPENDENCY_BLOB = {LAZY_BLOB!r}\n'
                 f'V3_LAZY_DEPENDENCY_SOURCE = {lazy.decode()!r}\n'
                 'V3_SOURCE_SEARCH_ROOT = Path("/kaggle/input")\n'
                 'V3_RUNTIME_SCRATCH_ROOT = Path("/tmp/cmi-flu-v3-batch1-runtime")\n')
    closure = function_source(raw, "load_v3_dependency_closure")
    previous = closure.replace("def load_v3_dependency_closure()", "def _load_v3_parent_dependency_closure()", 1)
    wrapper = '''def load_v3_dependency_closure() -> None:
    _load_v3_parent_dependency_closure()
    if git_blob_sha(V3_LAZY_DEPENDENCY_SOURCE.encode()) != V3_LAZY_DEPENDENCY_BLOB:
        raise BridgeContractError("v3_lazy_dependency_blob_mismatch")
    module = _v3_exec_module("task13_harmonization", V3_LAZY_DEPENDENCY_SOURCE)
    if not callable(getattr(module, "sdy272_asc_proxy_mask", None)):
        raise BridgeContractError("v3_lazy_dependency_entry_missing")
'''
    raw = replace_function(raw, "load_v3_dependency_closure", constants + previous + "\n" + wrapper)

    body = function_source(raw, "execute").replace("def execute(", "def _execute_audit_body(", 1)
    old_root = 'Path("/tmp") / "cmi-flu-v3-batch1-runtime"'
    old_search = 'Path("/kaggle/input").rglob("submission.csv")'
    if body.count(old_root) != 1 or body.count(old_search) != 1:
        raise ValueError("generated private scratch/input boundary changed")
    body = body.replace(old_root, "V3_RUNTIME_SCRATCH_ROOT").replace(old_search, 'V3_SOURCE_SEARCH_ROOT.rglob("submission.csv")')
    anchor = '            "model_fit_count": 0,\n'
    if body.count(anchor) != 1:
        raise ValueError("generated receipt binding changed")
    body = body.replace(anchor, anchor + '            "input_sha256": audit.get("input_sha256"),\n'
                        + '            "reference_expectations": audit.get("reference_expectations_not_mutated_to_match_runtime"),\n', 1)
    execution = r'''def execute(input_dir: Path, output_dir: Path) -> int:
    import contextlib, importlib.metadata, io, json, re, shutil, sys, tempfile
    global V3_RUNTIME_SCRATCH_ROOT
    declared = [
        "v3_public_singleton_Task1_1.csv", "v3_public_singleton_Task1_2.csv", "v3_public_singleton_Task1_3.csv",
        "v3_public_singleton_Task1_4.csv", "v3_public_singleton_Task2_1.csv", "v3_public_singleton_Task2_2.csv",
        "teacher_ledger.json", "measurement_contracts.json", "split_support.json", "auxiliary_label_coverage.json",
        "source_alignment_audit.json", "diagnostic_manifest.json", "runtime_receipt.json",
    ]
    published = []
    previous_path = list(sys.path)
    success = False
    stage = "staged_output_boundary"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.is_symlink() or any(output_dir.iterdir()):
            raise BridgeContractError("v3_final_output_must_be_empty")
        with tempfile.TemporaryDirectory(prefix="cmi-v3-004-", dir="/tmp") as tmp:
            tmp_root = Path(tmp)
            V3_RUNTIME_SCRATCH_ROOT = tmp_root / "runtime"
            staged = tmp_root / "staged"
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                code = _execute_audit_body(input_dir, staged)
            if code != 0:
                markers = re.findall(r"CMI_FLU_V3_BATCH1_RUNTIME_FAIL stage=[A-Za-z0-9_]+ type=[A-Za-z0-9_]+ code=[0-9a-f]{20}", captured.getvalue())
                if len(set(markers)) != 1:
                    raise BridgeContractError("v3_bounded_failure_marker_missing")
                print(markers[0], file=sys.stderr)
                return 1
            stage = "validate_staged_outputs"
            if sorted(p.name for p in staged.iterdir()) != sorted(declared):
                raise BridgeContractError("v3_staged_output_allowlist_mismatch")
            if any(p.is_symlink() or not p.is_file() for p in staged.iterdir()):
                raise BridgeContractError("v3_nonregular_staged_output")
            manifest = json.loads((staged / "diagnostic_manifest.json").read_text())
            for item in manifest["files"]:
                p = staged / item["filename"]
                if p.stat().st_size != item["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
                    raise BridgeContractError("v3_staged_csv_hash_mismatch")
            receipt_path = staged / "runtime_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["runtime_repair_science_commit"] = V3_FULL_RUNTIME_SCIENCE_COMMIT
            receipt["lazy_dependency_blob"] = V3_LAZY_DEPENDENCY_BLOB
            receipt["status"] = "diagnostic_files_generated_waiting_for_manual_submissions_and_scores"
            receipt["library_versions"] = {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn")}
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stage = "publish_validated_outputs"
            for name in declared:
                destination = output_dir / name
                with destination.open("xb") as stream:
                    published.append(destination)
                    stream.write((staged / name).read_bytes())
            for item in manifest["files"]:
                p = output_dir / item["filename"]
                if hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
                    raise BridgeContractError("v3_saved_csv_hash_mismatch")
            success = True
        print("CMI_FLU_V3_BATCH1_RUNTIME_PASS model_fit_count=0 competition_submit=false diagnostic_files=6 aggregate_files=5 diagnostic_not_final=true scratch_cleanup=true")
        return 0
    except Exception as exc:
        code = hashlib.sha256(f"{stage}:{type(exc).__name__}:{exc}".encode()).hexdigest()[:20]
        print(f"CMI_FLU_V3_BATCH1_RUNTIME_FAIL stage={stage} type={type(exc).__name__} code={code}", file=sys.stderr)
        return 1
    finally:
        sys.path[:] = previous_path
        if not success:
            for path in reversed(published):
                path.unlink(missing_ok=True)
'''
    raw = replace_function(raw, "execute", body + "\n" + execution)
    entry = r'''def main() -> int:
    import argparse, hashlib, sys
    global V3_SOURCE_SEARCH_ROOT
    parser = argparse.ArgumentParser(description="CMI-Flu V3 no-fit audit and fixed private diagnostics")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    try:
        V3_SOURCE_SEARCH_ROOT = args.input_dir.resolve()
        required = ("participants.tsv", "investigations_260821.tsv", "publicData_cytokine.tsv", "publicData_ex_vivo_flow.tsv", "publicData_serology_260821.tsv", "2025LJI_aim.tsv", "2025LJI_cytokine.tsv", "2025LJI_ex_vivo_flow.tsv", "2025LJI_serology.tsv", "md5sum")
        roots = {p.parent for p in V3_SOURCE_SEARCH_ROOT.rglob("sample_submission_part1.csv") if all((p.parent / n).is_file() for n in required)}
        if len(roots) != 1:
            raise BridgeContractError("v3_exact_competition_root_missing_or_ambiguous")
        return execute(next(iter(roots)), args.output_dir.resolve())
    except Exception as exc:
        stage = "locate_competition_input"
        code = hashlib.sha256(f"{stage}:{type(exc).__name__}:{exc}".encode()).hexdigest()[:20]
        print(f"CMI_FLU_V3_BATCH1_RUNTIME_FAIL stage={stage} type={type(exc).__name__} code={code}", file=sys.stderr)
        return 1
'''
    raw = replace_function(raw, "main", entry)
    compile(raw, "v3-full-runtime-004.py", "exec")
    if "make_workspace" in raw or "SYN_SHARED_SUBJECT" in raw:
        raise ValueError("CI-only fixture leaked into production Notebook")
    if len(raw.encode()) >= 1100000:
        raise ValueError("V3 runtime exceeds reviewed executor byte budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(raw, encoding="utf-8")
    return raw


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    raw = build_runtime(a.repository_root.resolve(), a.output.resolve()).encode()
    print(f"V3_FULL_RUNTIME_PREPARE PASS bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()} lazy_dependency_blob={LAZY_BLOB} model_fit=false competition_submit=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
