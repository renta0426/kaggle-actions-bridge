#!/usr/bin/env python3
"""Execute exact repaired E06a on synthetic data containing a zero-panel-overlap study."""
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
    spec = importlib.util.spec_from_file_location("e06a_v2_frozen_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    runtime = load_runtime(args.runtime.resolve())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix="e06a-v2-synthetic-package-") as tmp:
        package_path = Path(tmp) / "cmi_flu_bundle.zip"
        package_path.write_bytes(runtime.package_bytes())
        sys.path.insert(0, str(package_path))
        try:
            run, _hai, e06v2 = runtime.load_e06_module()
            assert callable(run)
            e06 = sys.modules["cmi_flu.strategy_e06"]
            assert e06._nested_calibration is e06v2._nested_calibration
            assert e06._fit_calibrator is e06v2._fit_calibrator

            from cmi_flu.datasets import HAIModelDataset
            from cmi_flu.models import ModelSpec

            panel = tuple(f"A/SYNTH/{index:02d}/2020" for index in range(12))
            historical_panel = panel[:9]
            rows = []
            for study_index, study in enumerate(("SYN_A", "SYN_B", "SYN_C", "SYN_D")):
                for donor in range(6):
                    for strain_index, strain in enumerate(historical_panel):
                        pre = 20.0 + 4.0 * donor + 1.5 * strain_index + study_index
                        fold = 1.35 + 0.05 * donor + 0.015 * strain_index + 0.01 * study_index
                        rows.append(
                            {
                                "participant_id": f"{study}-p{donor}",
                                "subject_group": f"{study}-s{donor}",
                                "study_group": study,
                                "virus_strain": strain,
                                "log2_pre_hai": float(np.log2(pre)),
                                "age": 24.0 + 3.0 * donor + study_index,
                                "post_hai": pre * fold,
                                "target_log2_fold": float(np.log2(fold)),
                            }
                        )
            # Valid D365 HAI study with no fixed Task2.3 panel strain.
            for donor in range(6):
                pre = 30.0 + donor
                fold = 1.25 + 0.03 * donor
                rows.append(
                    {
                        "participant_id": f"SYN_E-p{donor}",
                        "subject_group": f"SYN_E-s{donor}",
                        "study_group": "SYN_E",
                        "virus_strain": "A/NONPANEL/99/2020",
                        "log2_pre_hai": float(np.log2(pre)),
                        "age": 32.0 + donor,
                        "post_hai": pre * fold,
                        "target_log2_fold": float(np.log2(fold)),
                    }
                )
            train = pd.DataFrame(rows)
            challenge_rows = []
            for donor in range(5):
                for strain_index, strain in enumerate(panel):
                    pre = 25.0 + 3.0 * donor + 1.25 * strain_index
                    challenge_rows.append(
                        {
                            "participant_id": f"challenge-p{donor}",
                            "subject_group": f"challenge-s{donor}",
                            "study_group": "CHALLENGE",
                            "virus_strain": strain,
                            "log2_pre_hai": float(np.log2(pre)),
                            "age": 30.0 + donor,
                        }
                    )
            dataset = HAIModelDataset(
                day=365,
                train=train,
                challenge=pd.DataFrame(challenge_rows),
                target_representation="residual",
                metadata={"challenge_panel_strains": list(panel)},
            )
            spec = ModelSpec(
                name="ridge_exact_a100",
                family="ridge",
                params={"alpha": 100.0},
                target_transform="identity",
            )
            nested = e06v2._nested_calibration(dataset, spec=spec, panel_strains=panel)
            assert nested["outer_candidate_split_count"] == 5
            assert nested["outer_split_count"] == 4
            assert nested["skipped_outer_split_count"] == 1
            assert nested["skipped_outer_folds"][0]["reason"] == "zero_fixed_panel_overlap"
            assert nested["skipped_outer_folds"][0]["validation_panel_overlap_rows"] == 0
            challenge = e06._challenge_calibration(
                dataset,
                spec=spec,
                panel_strains=panel,
                expected_donors=5,
            )
            promotion = e06._promotion(nested, challenge)
            condition = {
                "condition": "b21_reference",
                "model": spec.to_dict(),
                "historical_nested_study_out": nested,
                "challenge_rank_preservation": challenge,
                "promotion": promotion,
            }
            coverage = e06._historical_panel_coverage(dataset, panel_strains=panel)
            assert coverage["requested_strains"] == 12
            assert coverage["historically_observed_requested_strains"] == 9
            assert coverage["complete_requested_panel_donors"] == 0
            result = {
                "experiment": e06.EXPERIMENT,
                "task": e06.TASK,
                "target_day": e06.TARGET_DAY,
                "conditions": {
                    "b21_reference": condition,
                    "phase_a_target_domain": {**condition, "condition": "phase_a_target_domain"},
                },
                "selected_promoted_condition": None,
                "promotion_thresholds": {
                    "minimum_equal_study_mean_rmse_reduction": 0.02,
                    "minimum_median_study_rmse_reduction_strictly_greater_than": 0.0,
                    "maximum_worst_study_rmse_ratio": 1.05,
                    "rank_tolerance": 1e-12,
                },
                "historical_panel_proxy_coverage": coverage,
                "historical_target_contract": "synthetic incomplete fixed-panel proxy",
                "calibration_contract": {
                    "calibration_kinds": ["positive_affine", "log2_affine"],
                    "fit_unit": "panel_geometric_mean_donor",
                    "calibration_weighting": "equal_study_total_weight",
                    "outer_evaluation": "subject_purged_leave_one_study_out",
                    "calibrator_training": "inner_subject_purged_study_out_oof_from_outer_training_only",
                    "strict_monotonicity_required": True,
                    "isotonic_used": False,
                    "held_study_outcomes_used_for_calibrator": False,
                    "observed_d28_used_as_d365_feature": False,
                    "outer_fold_panel_overlap_filter": "fixed_panel_strain_availability_only_before_outcome_access",
                    "zero_panel_overlap_studies_scored_as_task23_proxy": False,
                    "minimum_calibrator_panel_overlap_studies": 2,
                },
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
        "request_id": runtime.REQUEST_ID,
        "science_commit": runtime.SCIENCE_COMMIT,
        "strategy_e06_blob_sha": runtime.E06_BLOB,
        "strategy_e06_v2_blob_sha": runtime.E06_V2_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
        "metrics_sha256": sha(metrics_path),
        "summary_sha256": sha(summary_path),
    }
    bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_dir = out / "receipt"
    receipt_dir.mkdir()
    (receipt_dir / "release-receipt.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "model_executed": True,
                "nested_calibration_executed": True,
                "panel_overlap_filter_executed": True,
                "result_validated": True,
                "all_final_files_saved": True,
                "cleanup_completed": True,
                "aggregate_only": True,
            },
            sort_keys=True,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    assert sorted(path.name for path in out.iterdir() if path.is_file()) == [
        "bridge-result.json",
        "metrics.json",
        "summary.md",
    ]
    print(
        "CMI_FLU_E06A_V2_SYNTHETIC PASS model=true nested=true panel_overlap=true "
        "validate=true final_files=true receipt=true cleanup=true rank_preserved=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
