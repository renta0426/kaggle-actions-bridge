#!/usr/bin/env python3
"""Execute exact frozen E06b science on synthetic Challenge-shaped HAI data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def load_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("e06b_frozen_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_frame(day: int) -> pd.DataFrame:
    panel = ("H1", "H3", "B")
    rows: list[dict[str, object]] = []
    for study_index, study in enumerate(("SYN_A", "SYN_B", "SYN_C", "SYN_D", "SYN_ZERO")):
        strains = ("X",) if study == "SYN_ZERO" else panel
        for donor in range(6):
            for strain_index, strain in enumerate(strains):
                pre = 24.0 + 3.0 * donor + 1.5 * strain_index + study_index
                day_effect = 0.62 if day == 28 else 0.31
                fold = 1.0 + day_effect + 0.035 * donor + 0.012 * strain_index + 0.008 * study_index
                rows.append(
                    {
                        "participant_id": f"{study}-p{donor}",
                        "subject_group": f"{study}-s{donor}",
                        "study_group": study,
                        "virus_strain": strain,
                        "log2_pre_hai": float(np.log2(pre)),
                        "age": 24.0 + donor + study_index,
                        "post_hai": float(pre * fold),
                        "target_log2_fold": float(np.log2(fold)),
                    }
                )
    return pd.DataFrame(rows)


def synthetic_challenge() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for donor in range(40):
        for strain_index, strain in enumerate(("H1", "H3", "B")):
            pre = 26.0 + 0.8 * donor + 1.2 * strain_index
            rows.append(
                {
                    "participant_id": f"challenge-p{donor}",
                    "subject_group": f"challenge-s{donor}",
                    "study_group": "CHALLENGE",
                    "virus_strain": strain,
                    "log2_pre_hai": float(np.log2(pre)),
                    "age": 28.0 + donor % 20,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runtime_path = args.runtime.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtime = load_runtime(runtime_path)

    with tempfile.TemporaryDirectory(prefix="e06b-synthetic-package-") as tmp:
        package_path = Path(tmp) / "cmi_flu_bundle.zip"
        package_path.write_bytes(runtime.package_bytes())
        sys.path.insert(0, str(package_path))
        try:
            run, _hai, e06bv2 = runtime.load_e06_module()
            assert callable(run)
            base = sys.modules["cmi_flu.strategy_e06b"]
            assert base._fit_two_head_ridge is e06bv2._fit_two_head_ridge
            from cmi_flu.datasets import HAIModelDataset
            from cmi_flu.models import ModelSpec

            panel = ("H1", "H3", "B")
            challenge = synthetic_challenge()
            d28 = HAIModelDataset(
                day=28,
                train=synthetic_frame(28),
                challenge=challenge,
                target_representation="residual",
                metadata={"challenge_panel_strains": list(panel)},
            )
            d365 = HAIModelDataset(
                day=365,
                train=synthetic_frame(365),
                challenge=challenge,
                target_representation="residual",
                metadata={"challenge_panel_strains": list(panel)},
            )
            d28.validate(); d365.validate()
            spec = ModelSpec(
                name="ridge_exact_a100",
                family="ridge",
                params={"alpha": 100.0},
                target_transform="identity",
            )
            condition_a = base._evaluate_condition(
                "b21_reference",
                d28,
                d365,
                spec=spec,
                panel_strains=panel,
                expected_challenge_donors=40,
            )
            condition_b = base._evaluate_condition(
                "phase_a_target_domain",
                d28,
                d365,
                spec=spec,
                panel_strains=panel,
                expected_challenge_donors=40,
            )
            for condition in (condition_a, condition_b):
                historical = condition["historical_study_out"]
                assert historical["outer_candidate_split_count"] == 5
                assert historical["outer_split_count"] == 4
                assert historical["skipped_outer_split_count"] == 1
                assert historical["skipped_outer_folds"][0]["reason"] == "zero_fixed_panel_overlap"
                assert all(
                    fold["observed_d28_used_as_d365_feature"] is False
                    and fold["compatibility_patch"] == "ci_caught_day28_variable_typo_only"
                    for fold in historical["fold_training_contract"]
                )
                assert condition["challenge_shared_vs_independent_rank"]["donor_count"] == 40

            result = {
                "experiment": "strategy_v2_e06b_task23_d28_d365_two_head",
                "task": "Task2.3",
                "primary_day": 365,
                "auxiliary_day": 28,
                "conditions": {
                    "b21_reference": condition_a,
                    "phase_a_target_domain": condition_b,
                },
                "selected_promoted_condition": None,
                "promotion_thresholds": {
                    "minimum_equal_study_mean_rmse_reduction": 0.02,
                    "minimum_median_study_rmse_reduction_strictly_greater_than": 0.0,
                    "maximum_worst_study_rmse_ratio": 1.05,
                    "minimum_equal_study_spearman_delta": -0.01,
                    "large_study_n": 10,
                    "maximum_large_study_spearman_decline": -0.10,
                },
                "historical_panel_proxy_coverage": {
                    "requested_strains": 3,
                    "historically_observed_requested_strains": 3,
                    "complete_requested_panel_donors": 24,
                },
                "model_contract": {
                    "family": "ridge",
                    "alpha": 100.0,
                    "shared_design": "[Z, day_code*Z, day_code]",
                    "day_code_d28": -0.5,
                    "day_code_d365": 0.5,
                    "preprocessing_fit": "outer_training_rows_only_across_D28_and_D365",
                    "held_study_and_subjects_purged_from_both_days": True,
                    "observed_d28_used_as_d365_feature": False,
                    "d28_role": "auxiliary_training_label_only",
                    "d365_role": "primary_Task2.3_target",
                    "two_head_fit_adapter": "strategy_e06b_v2_ci_typo_fix",
                },
                "historical_target_contract": "synthetic fixed-panel proxy",
                "negative_result_scope": "synthetic",
                "competition_submission_attempted": False,
                "leaderboard_used_for_selection": False,
                "output_policy": "aggregate_only_public_study_names_no_participant_ids_or_row_predictions",
            }
            safe = runtime.json_safe(result)
            runtime.validate_result(safe)
            summary = runtime.render_summary(safe)
        finally:
            sys.path.remove(str(package_path))

    metrics_path = out / "metrics.json"
    summary_path = out / "summary.md"
    bridge_path = out / "bridge-result.json"
    metrics_path.write_text(json.dumps(safe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")
    bridge = {
        "schema_version": 1,
        "request_id": runtime.REQUEST_ID,
        "competition": runtime.COMPETITION,
        "target_kernel": runtime.TARGET_KERNEL,
        "science_commit": runtime.SCIENCE_COMMIT,
        "strategy_e06b_blob_sha": runtime.E06B_BLOB,
        "strategy_e06b_v2_blob_sha": runtime.E06B_V2_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
        "metrics_sha256": file_sha(metrics_path),
        "summary_sha256": file_sha(summary_path),
    }
    bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    receipt_dir = out / "receipt"
    receipt_dir.mkdir()
    receipt = {
        "status": "PASS",
        "model_executed": True,
        "two_head_executed": True,
        "result_validated": True,
        "all_final_files_saved": True,
        "completion_schema_checked": True,
        "cleanup_completed": True,
        "aggregate_only": True,
    }
    (receipt_dir / "release-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert sorted(path.name for path in out.iterdir() if path.is_file()) == [
        "bridge-result.json",
        "metrics.json",
        "summary.md",
    ]
    assert not any(path.name == "e01-runtime" for path in out.iterdir())
    print(
        "CMI_FLU_E06B_SYNTHETIC PASS model=true two_head=true validate=true "
        "final_files=true receipt=true cleanup=true d28_feature=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
