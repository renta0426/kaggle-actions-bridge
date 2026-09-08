#!/usr/bin/env python3
"""Standard-library-only output-reader safety tests; no SDK/network credentials."""
from __future__ import annotations
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import kaggle_current_output_read as reader
import kaggle_exact_identity as identity


class OutputContracts(unittest.TestCase):
    def exercise(self, scenario):
        called = []
        metadata = types.SimpleNamespace(ref="owner/notebook", is_private=True, current_version_number=1)
        api = types.SimpleNamespace(authenticate=lambda: None, kernels_status=lambda k: types.SimpleNamespace(status="COMPLETE"), kernels_list=lambda **kw: self.fail("must not search"))
        stub = types.ModuleType("kaggle.api.kaggle_api_extended"); stub.KaggleApi = lambda: api
        def cli(args, **kwargs):
            self.assertEqual(args[1:3], ["kernels", "output"])
            self.assertTrue(kwargs["capture_output"])
            self.assertFalse(kwargs.get("shell", False))
            self.assertEqual(kwargs["timeout"], 180)
            folder = Path(args[-1]); called.append(folder)
            (folder / "metrics.json").write_text('{"synthetic":1}')
            (folder / "kernel.log").write_text("synthetic-log")
            if scenario == "unexpected":
                (folder / "private-canary.txt").write_text("not permitted")
            elif scenario == "duplicate":
                (folder / "nested").mkdir(); (folder / "nested/metrics.json").write_text('{}')
            elif scenario == "symlink":
                (folder / "bad-link").symlink_to(folder / "metrics.json")
            elif scenario == "missing":
                (folder / "metrics.json").unlink()
            elif scenario == "version_changed":
                metadata.current_version_number = 2
            return subprocess.CompletedProcess(args, 1 if scenario == "cli_failed" else 0, "private-canary" if scenario == "cli_failed" else "", "")
        modules = {"kaggle": types.ModuleType("kaggle"), "kaggle.api": types.ModuleType("kaggle.api"), "kaggle.api.kaggle_api_extended": stub}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), patch.object(identity, "exact_metadata", return_value=metadata) as exact_read, patch.object(reader.shutil, "which", return_value="/synthetic/kaggle"), patch.object(reader.subprocess, "run", side_effect=cli):
            out = Path(tmp) / "out"
            if scenario == "ok":
                with redirect_stdout(io.StringIO()):
                    reader.read_current_output(kernel="owner/notebook", expected_version=1, allow={"metrics.json": 100}, output_dir=out)
                self.assertEqual([p.name for p in out.iterdir()], ["metrics.json"])
            else:
                with self.assertRaises((RuntimeError, identity.IdentityError)) as failure:
                    reader.read_current_output(kernel="owner/notebook", expected_version=1, allow={"metrics.json": 1 if scenario == "oversize" else 100}, output_dir=out)
                self.assertNotIn("private-canary", str(failure.exception))
                self.assertFalse(out.exists())
        self.assertEqual(exact_read.call_count, 1 if scenario == "cli_failed" else 2)
        self.assertEqual(len(called), 1)
        self.assertFalse(called[0].exists(), "temporary broad download must be cleaned")

    def test_success_and_transport_log_cleanup(self):
        self.exercise("ok")

    def test_reject_unexpected_duplicate_symlink_missing_or_oversize(self):
        for scenario in ("unexpected", "duplicate", "symlink", "missing", "oversize"):
            with self.subTest(scenario=scenario):
                self.exercise(scenario)

    def test_reject_version_change_during_download(self):
        self.exercise("version_changed")

    def test_cli_failure_is_captured_and_not_retried(self):
        self.exercise("cli_failed")

    def test_allowlist_limits(self):
        for values in ([], ["../private:1"], ["ok:0"], ["ok:67108865"], ["ok:1", "ok:2"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                reader._parse_allow(values)

    def test_safe_underscore_and_path_rejection(self):
        self.assertEqual(reader._parse_allow(["full_predictions.parquet:678787", "__huggingface_repos__.json:428"]), {"full_predictions.parquet": 678787, "__huggingface_repos__.json": 428})
        for name in ("../secret", "/tmp/x", "a/b", "a\\b", "..", "a..b", "_..secret", ".hidden", "A" * 129):
            with self.subTest(name=name), self.assertRaises(ValueError):
                reader._parse_allow([f"{name}:1"])

    def test_single_bounded_cli_and_no_search(self):
        source = Path(reader.__file__).read_text(); tree = ast.parse(source)
        runs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "subprocess" and n.func.attr == "run"]
        self.assertEqual(len(runs), 1)
        self.assertNotIn("kernels_list", source)
        self.assertNotIn("shell=True", source)
        self.assertNotIn("os.system(", source)
        self.assertNotIn("subprocess.Popen", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
