#!/usr/bin/env python3
"""Synthetic I/O rehearsal of the full recovery path; no live API calls."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cmi_flu_e02_recover as recovery

ORIGINAL: Path


def synthetic_metrics() -> dict:
    folds = [{"study": s, "n": n, "spearman": 0.0, "rank_rmse": 0.5, "constant_prediction": False, "tie_fraction": 0.0} for s, n in (("SDY180", 34), ("SDY515", 16), ("SDY519", 17), ("SDY56", 60))]
    metric = {"rows": 127, "studies": 4, "undefined_fold_count": 0, "constant_fold_count": 0,
              "study_equal_weight_spearman_mean_strict": 0.0, "folds": folds,
              "pooled_within_study_rank_spearman": {"n": 127, "value": 0.0}, "rank_rmse": 0.5, "raw_scale_rmse": None}
    return {
        "schema_version": 1, "experiment": "strategy_v2_e02_task11_structured_logfc",
        "random_seed": 20260907, "comparison_contract": "paired_subject_purged_v2", "task": "Task1.1",
        "cohort": {"train_rows": 127, "train_studies": 4, "challenge_rows": 40},
        "fixed_conditions": {"b21_model": "pls_2", "anchor_residual_model": "pls_1", "anchor_residual_lambda": 0.25, "structured_ridge_alpha": 10.0},
        "conditions": {k: {"metrics": metric} for k in ("anchor", "b21", "anchor_residual", "structured_ridge")},
        "comparisons": {}, "sensitivity_28": {},
        "negative_controls": {k: {"metrics": metric} for k in ("availability_only", "structured_subject_block_label_shuffle")},
        "repeat_baseline_extension": {"audit": {"status": "data_limited", "challenge_paired_subjects": 40, "full_outer_contract_gate_passed": False}, "condition_executed": False, "candidate": None},
        "decision": {"structured_ridge_candidate_over_anchor": False, "repeat_extension_candidate_over_anchor": None, "candidate_delta_threshold": 0.02, "large_study_decline_review_threshold": -0.10},
        "contains_participant_identifiers": False, "contains_row_level_predictions": False,
        "leaderboard_used_for_selection": False, "competition_submission_attempted": False,
    }


class RecoveryIO(unittest.TestCase):
    def exercise(self, scenario: str) -> None:
        calls = []
        metadata = types.SimpleNamespace(ref=recovery.TARGET, is_private=True, current_version_number=1)
        api = types.SimpleNamespace(authenticate=lambda: None)
        sdk = types.ModuleType("kaggle.api.kaggle_api_extended"); sdk.KaggleApi = lambda: api
        executor = types.ModuleType("cmi_flu_strategy_e02_execute"); executor.live_rules = lambda a: None
        modules = {"kaggle": types.ModuleType("kaggle"), "kaggle.api": types.ModuleType("kaggle.api"), "kaggle.api.kaggle_api_extended": sdk, "cmi_flu_strategy_e02_execute": executor}
        def read(verb, directory):
            calls.append(verb)
            if verb == "pull":
                data = ORIGINAL.read_bytes()
                (directory / "script.py").write_bytes(data + (b"\n# changed" if scenario == "wrong_code" else b""))
                return
            self.assertEqual(verb, "output")
            metrics = synthetic_metrics()
            if scenario == "private_row":
                metrics["participant_id"] = "private-canary"
            path = directory / "metrics.json"
            path.write_text(json.dumps(metrics))
            (directory / "summary.md").write_text("Synthetic summary; not a scientific result.\n")
            if scenario == "original_success":
                bridge = {"request_id": recovery.ORIGINAL_REQUEST, "science_commit": recovery.SCIENCE, "metrics_sha256": recovery.digest(path)}
                (directory / "bridge-result.json").write_text(json.dumps(bridge))
            signature = hashlib.sha256(b"KeyError:'frozen_incumbent'").hexdigest()[:20]
            if scenario == "unknown_failure":
                signature = "0" * 20
            (directory / "kernel.log").write_text("private-canary must never be published\nCMI_FLU_E02_FAILED stage=write_outputs exception_type=KeyError error_code=" + signature + "\n")
            if scenario == "version_changed":
                metadata.current_version_number = 2
        should_pass = scenario in {"original_success", "known_sink_failure"}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), patch.object(recovery, "verify_current", return_value="COMPLETE" if scenario == "original_success" else "ERROR"), patch.object(recovery, "exact_metadata", return_value=metadata), patch.object(recovery, "cli_read", side_effect=read):
            output = Path(tmp) / "recovered"
            log = io.StringIO()
            with redirect_stdout(log):
                if should_pass:
                    recovery.recover(ORIGINAL, output)
                else:
                    with self.assertRaises(Exception):
                        recovery.recover(ORIGINAL, output)
            self.assertNotIn("private-canary", log.getvalue())
            self.assertNotIn("push", calls)
            if should_pass:
                self.assertEqual(calls, ["pull", "output"])
                self.assertEqual(sorted(p.name for p in output.iterdir()), ["aggregate.json", "recovery.json"])
                receipt = json.loads((output / "recovery.json").read_text())
                self.assertTrue(receipt["scientific_metrics_validated"])
                self.assertFalse(receipt["new_write_attempted"])
                self.assertEqual(receipt["original_notebook_success"], scenario == "original_success")
            else:
                self.assertFalse(output.exists())
                self.assertNotIn("E02_AGGREGATES_BEGIN", log.getvalue())

    def test_original_success_read(self):
        self.exercise("original_success")

    def test_known_serialization_failure_salvages_without_forging_receipt(self):
        self.exercise("known_sink_failure")

    def test_unknown_failure_cannot_salvage(self):
        self.exercise("unknown_failure")

    def test_wrong_code_rejected_before_output_read(self):
        self.exercise("wrong_code")

    def test_version_changed_rejected_before_export(self):
        self.exercise("version_changed")

    def test_private_row_rejected_before_export(self):
        self.exercise("private_row")


def main() -> int:
    global ORIGINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    args = parser.parse_args()
    ORIGINAL = args.original.resolve()
    assert recovery.digest(ORIGINAL) == recovery.RUNTIME_SHA
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryIO))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
