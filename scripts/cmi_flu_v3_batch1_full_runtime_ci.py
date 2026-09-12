#!/usr/bin/env python3
"""Credential-free end-to-end tests of the generated V3 script, not stubs."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import cmi_flu_v3_batch1_full_runtime_prepare as prepare


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def argv(values):
    old = sys.argv
    sys.argv = values
    try:
        yield
    finally:
        sys.argv = old


def worker(root: Path, runtime: Path, mode: str) -> None:
    fixture = load(root / "tests/v3_batch1_integration_support.py", "v3_synthetic_fixture")
    rt = load(runtime, "v3_exact_generated_runtime")
    with tempfile.TemporaryDirectory(prefix="v3-full-worker-") as tmp:
        work = Path(tmp)
        data, source = fixture.make_workspace(work / "input")
        original = source.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        output = work / "output"
        if mode == "reproduce_parent":
            package = work / "old-package.zip"
            package.write_bytes(rt.package_bytes())
            sys.path.insert(0, str(package))
            module = rt.load_v3_batch1_module()
            with fixture.forbid_fit_and_submit():
                try:
                    module.run_v3_01_audit(data)
                except ModuleNotFoundError as exc:
                    code = hashlib.sha256(f"run_no_fit_audit:ModuleNotFoundError:{exc}".encode()).hexdigest()[:20]
                    assert code == "38041141c51c6fb42b4c", code
                else:
                    raise AssertionError("exact consumed parent did not reproduce missing lazy dependency")
            print("V3_FULL_RUNTIME_PARENT_REPRODUCTION PASS code=38041141c51c6fb42b4c")
            return
        assert rt.V3_SOURCE_B_BYTES == 5926
        assert rt.V3_SOURCE_B_SHA256 == prepare.SOURCE_B_SHA
        original_loader = rt.load_v3_batch1_module
        audit_calls = []
        partial_files = []
        if mode != "reject_source":
            # Only this isolated CI process accepts synthetic CSV bytes. Neither
            # generated source on disk nor production request constants change.
            rt.V3_SOURCE_B_BYTES = len(original)
            rt.V3_SOURCE_B_SHA256 = digest
            def testing_loader():
                module = original_loader()
                base = sys.modules["cmi_flu.strategy_v3_batch1"]
                base.EXPECTED_E12C_BYTES = len(original)
                base.EXPECTED_E12C_SHA256 = digest
                native_audit = module.run_v3_01_audit
                def checked_audit(*args, **kwargs):
                    result = native_audit(*args, **kwargs)
                    fixture.assert_audit(result)
                    audit_calls.append(result)
                    return result
                module.run_v3_01_audit = checked_audit
                if mode == "cleanup_after_output_failure":
                    native_writer = module.write_aggregate_jsons
                    def fail_after_writing(result, directory):
                        native_writer(result, directory)
                        partial_files.extend(p.name for p in Path(directory).iterdir())
                        raise ValueError("synthetic_failure_after_output_creation")
                    module.write_aggregate_jsons = fail_after_writing
                return module
            rt.load_v3_batch1_module = testing_loader
        captured = io.StringIO()
        old_path = list(sys.path)
        with argv([str(runtime), "--input-dir", str(work / "input"), "--output-dir", str(output)]):
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured), fixture.forbid_fit_and_submit():
                rc = rt.main()
        text = captured.getvalue()
        assert sys.path == old_path
        assert not rt.V3_RUNTIME_SCRATCH_ROOT.exists()
        assert source.read_bytes() == original
        if mode == "reject_source":
            assert rc == 1 and "stage=locate_exact_source_B" in text, text
            assert not list(output.iterdir())
            assert not audit_calls
        elif mode == "cleanup_after_output_failure":
            assert rc == 1 and "stage=write_aggregate_records" in text, text
            assert len([n for n in partial_files if n.endswith(".csv")]) == 6
            assert len([n for n in partial_files if n.endswith(".json")]) == 5
            assert not list(output.iterdir())
        else:
            assert rc == 0, text
            assert len(audit_calls) == 1
            assert text.strip().splitlines()[-1].startswith("CMI_FLU_V3_BATCH1_RUNTIME_PASS "), text
            assert "scratch_cleanup=true" in text
            manifest = json.loads((output / "diagnostic_manifest.json").read_text())
            fixture.assert_diagnostics(source, output, manifest)
            assert len(list(output.iterdir())) == 13
            receipt = json.loads((output / "runtime_receipt.json").read_text())
            assert receipt["lazy_dependency_blob"] == prepare.LAZY_BLOB
            assert receipt["status"] == "diagnostic_files_generated_waiting_for_manual_submissions_and_scores"
            import cmi_flu_v3_batch1_execute as execute
            execute.SOURCE_B_SHA256 = digest
            execute.validate_private_output(output)
            safe = work / "safe"
            safe.mkdir()
            for name in execute.SAFE_NAMES:
                shutil.copyfile(output / name, safe / name)
            sanitizer = load(root / "scripts/cmi_flu_v3_batch1_sanitize.py", "v3_actual_sanitizer")
            sanitizer.SOURCE_B_SHA256 = digest
            emitted = io.StringIO()
            with argv(["sanitize", "--input-dir", str(safe)]), contextlib.redirect_stdout(emitted):
                assert sanitizer.main() == 0
            assert "CMI_FLU_V3_BATCH1_SANITIZE PASS" in emitted.getvalue()
            assert "SYN_" not in emitted.getvalue()
            # No row-level CSV or ID can pass the recovery boundary.
            csv_path = output / manifest["files"][0]["filename"]
            shutil.copyfile(csv_path, safe / csv_path.name)
            with argv(["sanitize", "--input-dir", str(safe)]):
                try:
                    sanitizer.main()
                except SystemExit:
                    pass
                else:
                    raise AssertionError("sanitizer accepted private CSV")
            (safe / csv_path.name).unlink()
            ledger = json.loads((safe / "teacher_ledger.json").read_text())
            ledger["participant_id"] = "SYN_MUST_NOT_ESCAPE"
            (safe / "teacher_ledger.json").write_text(json.dumps(ledger))
            emitted = io.StringIO()
            with argv(["sanitize", "--input-dir", str(safe)]), contextlib.redirect_stdout(emitted):
                try:
                    sanitizer.main()
                except SystemExit:
                    pass
                else:
                    raise AssertionError("sanitizer accepted individual identifier")
            assert not emitted.getvalue()
            saved = csv_path.read_bytes()
            csv_path.write_bytes(saved + b"\n")
            try:
                execute.validate_private_output(output)
            except RuntimeError:
                pass
            else:
                raise AssertionError("persisted CSV hash mutation not rejected")
            csv_path.write_bytes(saved)
            # Never overwrite or remove a preexisting operator file.
            with argv([str(runtime), "--input-dir", str(work / "input"), "--output-dir", str(output)]), contextlib.redirect_stderr(io.StringIO()):
                assert rt.main() == 1
            assert csv_path.read_bytes() == saved
        print(f"V3_FULL_RUNTIME_SCENARIO PASS mode={mode} synthetic_only=true model_fit=0 kaggle_write=0 competition_submit=0")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", required=True, type=Path)
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--runtime", type=Path)
    p.add_argument("--worker", choices=("success", "reject_source", "cleanup_after_output_failure", "reproduce_parent"))
    a = p.parse_args()
    root, work = a.repository_root.resolve(), a.workdir.resolve()
    if a.worker:
        worker(root, a.runtime.resolve(), a.worker)
        return 0
    work.mkdir(parents=True, exist_ok=True)
    runtime, repeated = work / "runtime.py", work / "runtime-repeat.py"
    prepare.build_runtime(root, runtime)
    prepare.build_runtime(root, repeated)
    assert runtime.read_bytes() == repeated.read_bytes(), "materialized runtime is not deterministic"
    for path in (runtime, root / "scripts/cmi_flu_v3_batch1_full_runtime_execute.py"):
        compile(path.read_text(), str(path), "exec")
    old = work / "consumed-parent.py"
    prepare.parent.build_runtime(root, old)
    commands = [(old, "reproduce_parent"), (runtime, "success"), (runtime, "reject_source"), (runtime, "cleanup_after_output_failure")]
    for path, mode in commands:
        subprocess.run([sys.executable, __file__, "--repository-root", str(root), "--workdir", str(work), "--runtime", str(path), "--worker", mode], check=True, timeout=180)
    result = subprocess.run([sys.executable, str(runtime), "--self-test"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SELF_TEST PASS" in result.stdout
    raw = runtime.read_bytes()
    print(f"V3_FULL_RUNTIME_CI PASS sha256={hashlib.sha256(raw).hexdigest()} bytes={len(raw)} real_libraries=true library_stubs=false full_generated_main=true sanitizer=true failure_cleanup=true synthetic_only=true model_fit=0 kaggle_write=0 compute=0 competition_submit=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
