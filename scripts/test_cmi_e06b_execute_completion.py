#!/usr/bin/env python3
"""Run the actual generated E06b ``execute`` final path through completion/cleanup."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd


def load_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("e06b_execute_completion_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Checksum:
    verified = ("synthetic-md5",)


class _Inputs:
    checksum_report = _Checksum()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    args = p.parse_args()
    runtime = load_runtime(args.runtime.resolve())
    result = json.loads(args.metrics.read_text(encoding="utf-8"))
    runtime.validate_result(result)

    with tempfile.TemporaryDirectory(prefix="e06b-execute-completion-") as tmp:
        root = Path(tmp)
        package = root / "cmi_flu_bundle.zip"
        package.write_bytes(runtime.package_bytes())
        sys.path.insert(0, str(package))
        try:
            from cmi_flu import runner
            original_load_inputs = runner.load_inputs
            runner.load_inputs = lambda _config: _Inputs()

            runtime.derive_reference_files = lambda _input, _external: (3, 12)
            ref_root = root / "refs"
            ref_root.mkdir()
            for name in runtime.LOCKED_REFERENCE_SHA256:
                (ref_root / name).write_bytes(runtime.locked_reference_bytes(name))
            runtime.locate_locked_reference = lambda name: ref_root / name

            fake_hai = types.SimpleNamespace(
                load_vaccine_strain_reference=lambda path: pd.DataFrame(
                    {"season": ["synthetic"], "virus_strain": ["H1"]}
                )
            )
            fake_run = lambda *_args, **_kwargs: result
            runtime.load_e06_module = lambda: (fake_run, fake_hai, types.SimpleNamespace())

            input_dir = root / "input"
            input_dir.mkdir()
            output_dir = root / "output"
            stdout = io.StringIO()
            old_cwd = Path.cwd()
            os.chdir(root)
            try:
                with contextlib.redirect_stdout(stdout):
                    rc = runtime.execute(input_dir, output_dir)
            finally:
                os.chdir(old_cwd)
            assert rc == 0, stdout.getvalue()
            text = stdout.getvalue()
            assert "CMI_FLU_E06B_COMPLETE" in text
            assert "conditions=2" in text
            assert "tasks=" not in text
            assert sorted(path.name for path in output_dir.iterdir() if path.is_file()) == [
                "bridge-result.json",
                "metrics.json",
                "summary.md",
            ]
            assert not (output_dir / "e01-runtime").exists()
            bridge = json.loads((output_dir / "bridge-result.json").read_text(encoding="utf-8"))
            assert bridge["request_id"] == runtime.REQUEST_ID
            assert bridge["strategy_e06b_blob_sha"] == runtime.E06B_BLOB
            assert bridge["strategy_e06b_v2_blob_sha"] == runtime.E06B_V2_BLOB
            assert bridge["competition_submission_attempted"] is False
            assert bridge["contains_participant_identifiers"] is False
            assert bridge["contains_row_level_predictions"] is False
        finally:
            try:
                runner.load_inputs = original_load_inputs
            except Exception:
                pass
            sys.path.remove(str(package))

    print(
        "CMI_FLU_E06B_EXECUTE_COMPLETION_PASS execute=true validate=true final_files=true "
        "completion_conditions=true cleanup=true tasks_key=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
