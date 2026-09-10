#!/usr/bin/env python3
"""Exercise the full non-synthetic E11b validator without Competition data."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def load_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("e11b_runtime_contract", path)
    if spec is None or spec.loader is None:
        raise SystemExit("E11b validator test cannot load runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    runtime_path = p.parse_args().runtime.resolve()
    text = runtime_path.read_text(encoding="utf-8")
    required = (
        "strategy_v2_e11b_task11_tabpfn3",
        "tabpfn-v3-regressor-v3_default.ckpt",
        "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301",
        "4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0",
        "prior-labsai/tabpfn-3/pytorch/default/1",
        "locate_tabpfn_checkpoint",
        "install_tabpfn_wheel",
    )
    for token in required:
        if token not in text:
            raise SystemExit(f"E11b validator runtime token missing:{token}")
    if "result['tasks']" in text:
        raise SystemExit("E11b validator runtime retains stale tasks writer access")
    if "kaggle competitions submit" in text or "competition_submit(" in text:
        raise SystemExit("E11b validator runtime contains submission path")

    runtime = load_runtime(runtime_path)
    run_e11b, run_synthetic, e11b = runtime.load_e11b_modules()
    synthetic = __import__("cmi_flu.strategy_e11b_synthetic", fromlist=["*"])
    dataset = synthetic.make_dataset()
    config = synthetic.make_config()
    task = e11b.evaluate_task_e11b(
        dataset,
        config=config,
        checkpoint_path=None,
        estimator_factory=synthetic.synthetic_estimator_factory,
    )

    checkpoint = {
        "filename": "tabpfn-v3-regressor-v3_default.ckpt",
        "bytes": 233289807,
        "sha256": "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301",
        "kaggle_model_source": "prior-labsai/tabpfn-3/pytorch/default/1",
        "verified": True,
    }
    audits = [fold["fit_audit"] for fold in task["folds"]]
    audits.append(task["final_fit_audit"])
    for audit in audits:
        audit["backend"] = "tabpfn"
        audit["package_verified"] = True
        audit["package_version_expected"] = "8.5.0"
        audit["package_source_commit"] = "9ed44abd5882140b88c9f2816c5791987ce059b9"
        audit["wheel_filename"] = "tabpfn-8.5.0-py3-none-any.whl"
        audit["wheel_sha256"] = "4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0"
        audit["model_source"] = "prior-labsai/tabpfn-3/pytorch/default/1"
        audit["checkpoint"] = dict(checkpoint)

    result = {
        "schema_version": 1,
        "experiment": "strategy_v2_e11b_task11_tabpfn3",
        "comparison_contract": "paired_subject_purged_v2",
        "frozen_conditions": {
            "task": "Task1.1",
            "base_model": "pls_2",
            "target_transform": "log",
            "feature_space": "unchanged_compact_b21_task11",
            "max_raw_features": 200,
            "tabpfn_package_version": "8.5.0",
            "tabpfn_source_commit": "9ed44abd5882140b88c9f2816c5791987ce059b9",
            "tabpfn_wheel_filename": "tabpfn-8.5.0-py3-none-any.whl",
            "tabpfn_wheel_sha256": "4c076a019cfa5520e9c41405cecda845bdd09d909ede4a60c43839bbc83bf7a0",
            "kaggle_model_source": "prior-labsai/tabpfn-3/pytorch/default/1",
            "checkpoint_filename": "tabpfn-v3-regressor-v3_default.ckpt",
            "checkpoint_bytes": 233289807,
            "checkpoint_sha256": "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301",
            "tabpfn_params": {
                "n_estimators": 8,
                "device": "cpu",
                "fit_mode": "fit_preprocessors",
                "memory_saving_mode": "auto",
                "random_state": 20260910,
                "n_preprocessing_jobs": 1,
                "show_progress_bar": False,
                "ignore_pretraining_limits": False,
            },
            "promotion_mean_delta": 0.02,
            "promotion_minimum_study_delta": -0.10,
            "promotion_requires_strict_majority_wins": True,
        },
        "external_model_license": {
            "model_license": "tabpfn-3-license-v1.0",
            "permitted_scope": "non-commercial data-science competition and academic research",
            "weight_redistribution_by_project": False,
            "official_kaggle_model_input": True,
        },
        "task": task,
        "cross_study_target_scale_redefined": False,
        "held_outcomes_used_for_fit": False,
        "held_outcomes_used_for_scoring": False,
        "automatic_compute_retries": 0,
        "leaderboard_used_for_selection": False,
        "public_probe_authorized": False,
        "competition_submission_attempted": False,
        "incumbent_changed": False,
    }
    runtime.validate_result(result, synthetic=False)
    print(
        "CMI_FLU_E11B_FULL_VALIDATOR_PASS competition_data=false "
        "package_verified_fixture=true checkpoint_verified_fixture=true submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
