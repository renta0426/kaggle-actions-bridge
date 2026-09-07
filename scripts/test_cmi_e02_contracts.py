#!/usr/bin/env python3
"""Credential-free regression and synthetic integration tests for E02 recovery."""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from enum import Enum
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import kaggle_exact_identity as identity
from build_cmi_e02_release import replace_top_level_function
from cmi_flu_e02_recover import project, cli_read


class Contracts(unittest.TestCase):
    def metadata(self, **changes):
        values = dict(ref="owner/notebook", current_version_number=1, is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=False)
        values.update(changes)
        return types.SimpleNamespace(**values)

    def test_exact_identity_does_not_use_search(self):
        api = types.SimpleNamespace(kernels_status=lambda ref: types.SimpleNamespace(status="COMPLETE"), kernels_list=lambda **kw: self.fail("search must not be called"))
        with patch.object(identity, "exact_metadata", return_value=self.metadata()):
            self.assertEqual(identity.verify_current(api, "owner/notebook", 1), "COMPLETE")

    def test_identity_rejects_wrong_ref_version_privacy(self):
        for change in ({"ref": "owner/other"}, {"current_version_number": 2}, {"is_private": False}, {"is_private": None}):
            with self.subTest(change=change), self.assertRaises(identity.IdentityError):
                identity.validate_metadata(self.metadata(**change), "owner/notebook", 1)

    def test_accelerator_must_be_explicit(self):
        for value in (True, None):
            with self.assertRaises(identity.IdentityError):
                identity.validate_metadata(self.metadata(enable_gpu=value), "owner/notebook", 1, cpu=True)

    def test_status_enum_and_unknown(self):
        class Status(Enum):
            COMPLETE = 1
        self.assertEqual(identity.status_name(Status.COMPLETE), "COMPLETE")
        for value in (None, 1, "NOT_COMPLETE", "FAILURE_COMPLETE", ""):
            with self.assertRaises(identity.IdentityError):
                identity.status_name(value)

    def test_failed_read_requires_explicit_recovery(self):
        api = types.SimpleNamespace(kernels_status=lambda ref: types.SimpleNamespace(status="ERROR"))
        with patch.object(identity, "exact_metadata", return_value=self.metadata()):
            with self.assertRaises(identity.IdentityError):
                identity.verify_current(api, "owner/notebook", 1)
            self.assertEqual(identity.verify_current(api, "owner/notebook", 1, allow_failed=True), "ERROR")

    def test_error_diagnostics_do_not_echo_private_text(self):
        self.assertNotIn("private-canary", json.dumps(identity.safe_exception(RuntimeError("private-canary"))))

    def test_ast_edit_preserves_neighbor_and_embedded_text(self):
        original = 'PAYLOAD = "def execute(): fake"\ndef execute():\n    return 1\ndef main():\n    return execute()\n'
        result = replace_top_level_function(original, "execute", lambda s: s.replace("return 1", "return 2"))
        self.assertIn('PAYLOAD = "def execute(): fake"', result)
        self.assertIn("def main():\n    return execute()", result)
        with self.assertRaises(ValueError):
            replace_top_level_function(original, "missing", lambda s: s)

    def test_recovery_cannot_invoke_write_verb(self):
        with self.assertRaises(ValueError):
            cli_read("push", Path("/unused"))


def sink_test(path: Path, expected_success: bool) -> None:
    """Run actual execute through serialization; only data/model input is fake."""
    n = runpy.run_path(str(path), run_name="synthetic_e02_sink")
    result = {"schema_version": 1, "experiment": "strategy_v2_e02_task11_structured_logfc", "random_seed": 20260907,
              "comparison_contract": "paired_subject_purged_v2", "task": "Task1.1",
              "cohort": {"train_rows": 127, "train_studies": 4, "challenge_rows": 40},
              "fixed_conditions": {"b21_model": "pls_2", "anchor_residual_model": "pls_1", "anchor_residual_lambda": 0.25, "structured_ridge_alpha": 10.0},
              "conditions": {k: {"metrics": {"rows": 127}} for k in ("anchor", "b21", "anchor_residual", "structured_ridge")},
              "negative_controls": {"availability_only": {}, "structured_subject_block_label_shuffle": {}},
              "repeat_baseline_extension": {"audit": {"status": "data_limited"}},
              "contains_participant_identifiers": False, "contains_row_level_predictions": False,
              "leaderboard_used_for_selection": False, "competition_submission_attempted": False}
    @dataclass
    class Config:
        source_path: object = None
        raw: object = None
        baseline: str = "b02_taskwise_compact"
        verify_md5: bool = True
        def section(self, key):
            return {"policy": "robust_v1"}
    runner = types.ModuleType("cmi_flu.runner")
    runner.load_inputs = lambda c: types.SimpleNamespace(checksum_report=types.SimpleNamespace(verified=list(range(28))))
    runner.run_compact_task = lambda *a, **k: None
    runner.run_hai_compact_for_panels = lambda *a, **k: None
    config_module = types.ModuleType("cmi_flu.configuration")
    config_module.load_baseline_config = lambda *a, **k: Config(raw={})
    evaluation = types.ModuleType("cmi_flu.evaluation")
    package = types.ModuleType("cmi_flu")
    package.__path__ = []
    package.runner = runner; package.evaluation = evaluation; package.configuration = config_module
    modules = {"cmi_flu": package, "cmi_flu.runner": runner, "cmi_flu.evaluation": evaluation, "cmi_flu.configuration": config_module}
    g = n["execute"].__globals__
    g.update(package_bytes=lambda: b"synthetic-package", derive_reference_files=lambda *a: (3, 12), B21_ADAPTER_SOURCE="def install():\n    pass\n", load_e02_module=lambda: (lambda c, i: result))
    before_path = list(sys.path)
    try:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            root = Path(tmp); inp = root / "inputs"; inp.mkdir(); out = root / "outputs"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = n["execute"](inp, out)
            assert (out / "metrics.json").is_file()
            assert (out / "summary.md").is_file()
            if expected_success:
                assert rc == 0, "repaired output sink did not complete"
                bridge = json.loads((out / "bridge-result.json").read_text())
                assert bridge["task"] == "Task1.1" and bridge["strategy_e02_blob_sha"] == n["E02_BLOB"]
                assert sorted(p.name for p in out.iterdir()) == ["bridge-result.json", "metrics.json", "summary.md"]
            else:
                assert rc == 2 and not (out / "bridge-result.json").exists(), "original failure was not reproduced"
    finally:
        sys.path[:] = before_path


def science_smoke(path: Path) -> None:
    """Exercise the exact frozen package + adapter + all fixed E02 models.

    Only raw data loading is substituted by synthetic TaskDataset rows. This is
    not a scientific result, baseline reproduction or Kaggle image equivalence.
    """
    import numpy as np
    import pandas as pd
    from dataclasses import replace
    n = runpy.run_path(str(path), run_name="e02_actual_science_smoke")
    before_path = list(sys.path)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); package = root / "package.zip"; package.write_bytes(n["package_bytes"]())
        sys.path.insert(0, str(package))
        try:
            adapter = {}; exec(n["B21_ADAPTER_SOURCE"], adapter); adapter["install"]()
            tree = ast.parse(path.read_text())
            execute = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "execute")
            body = next(x for x in execute.body if isinstance(x, ast.Try)).body
            start = next(i for i, x in enumerate(body) if isinstance(x, ast.Import) and any(a.asname == "_e02_evaluation" for a in x.names))
            end = next(i for i in range(start, len(body)) if isinstance(body[i], ast.Assign) and isinstance(body[i].value, ast.Constant) and body[i].value.value == "load_inputs")
            shim = ast.Module(body=body[start:end], type_ignores=[])
            scope = {"BridgeContractError": n["BridgeContractError"]}
            exec(compile(ast.fix_missing_locations(shim), "actual_frozen_shim", "exec"), scope)
            entry = n["load_e02_module"]()
            from cmi_flu.configuration import load_baseline_config
            from cmi_flu.datasets import TaskDataset
            import cmi_flu.strategy_e02 as e02
            cfgpath = root / "config.yaml"
            cfgpath.write_text(n["CONFIG_TEXT"].replace("baseline: b021_taskwise_robust", "baseline: b02_taskwise_compact"))
            config = load_baseline_config(cfgpath, repository_root=root)
            config = replace(config, baseline="b021_taskwise_robust", raw={**dict(config.raw), "baseline": "b021_taskwise_robust"})
            rng = np.random.default_rng(20260907)
            def rows(study, size, offset):
                b = rng.uniform(1, 15, size); c = rng.uniform(2, 18, size)
                return pd.DataFrame({"participant_id": [f"synthetic-{study}-{j}" for j in range(size)],
                    "subject_group": [f"synthetic-subject-{study}-{j}" for j in range(size)], "study_group": study,
                    "cytokine_raw__CXCL10": b, "cytokine_log1p__CXCL10": np.log1p(b),
                    "cytokine_rank__CXCL10": pd.Series(b).rank(pct=True),
                    "cytokine_log1p__IL6": np.log1p(c), "cytokine_raw__IL6": c,
                    "target_value": np.exp(-0.3 * np.log(b) + 0.1 * np.log(c) + offset + rng.normal(0, 0.1, size))})
            train = pd.concat([rows(s, count, 0.1 * i) for i, (s, count) in enumerate((("SDY180", 34), ("SDY515", 16), ("SDY519", 17), ("SDY56", 60)))], ignore_index=True)
            challenge = rows("SYNTHETIC_CHALLENGE", 40, 0).drop(columns="target_value")
            dataset = TaskDataset(task="Task1.1", train=train, challenge=challenge, target_column="target_value")
            empty = pd.DataFrame(columns=["participant_id", "study_accession", "timepoint", "analyte", "value"])
            inputs = types.SimpleNamespace(tables={"public_cytokine": empty, "challenge_cytokine": empty})
            with patch.object(e02, "build_b02_datasets", return_value={"Task1.1": dataset}):
                result = entry(config, inputs)
            n["validate_result"](result)
            safe = project(result)
            assert set(safe["conditions"]) == {"anchor", "b21", "anchor_residual", "structured_ridge"}
            assert result["repeat_baseline_extension"]["condition_executed"] is False
            assert result["cohort"]["train_rows"] == 127
            print("E02_SYNTHETIC_SCIENCE PASS rows=127 studies=4 challenge=40 real_frozen_models=true private_data=false")
        finally:
            sys.path[:] = before_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    a = p.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful():
        return 1
    sink_test(a.original, False)
    print("E02_ORIGINAL_SERIALIZATION_FAILURE REPRODUCED")
    sink_test(a.release, True)
    print("E02_RELEASE_FULL_OUTPUT_LIFECYCLE PASS")
    science_smoke(a.release)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
