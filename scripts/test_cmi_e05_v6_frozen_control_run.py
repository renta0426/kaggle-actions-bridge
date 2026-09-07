#!/usr/bin/env python3
"""Run the exact E05 control/condition readout against the frozen B2 package.

This test exists because E05 006 passed low-level Ridge/reference tests but the
remote run failed when newer E05 code accessed fold-diagnostic attributes that
do not exist in the frozen incumbent evaluator.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def load_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("e05_v6_frozen_control_runtime", path)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load generated E05 v6 runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_frames():
    strains = (
        "H1N1 A/TestOne/1/2020_MDCK",
        "H3N2 A/TestTwo/2/2020_MDCK",
        "B/Victoria B/TestThree/3/2020_MDCK",
    )
    rows = []
    for study_index, study in enumerate(("SDY_X", "SDY_Y")):
        for donor_index in range(5):
            participant = f"{study}-p{donor_index}"
            subject = f"{study}-s{donor_index}"
            for strain_index, strain in enumerate(strains):
                log2_pre = 2.0 + 0.35 * donor_index + 0.15 * strain_index + 0.1 * study_index
                log2_post = 3.0 + 0.55 * donor_index + 0.25 * strain_index + 0.2 * study_index
                rows.append(
                    {
                        "participant_id": participant,
                        "virus_strain": strain,
                        "subject_group": subject,
                        "study_group": study,
                        "log2_pre_hai": log2_pre,
                        "post_hai": float(np.exp2(log2_post)),
                        "target_log2_post": log2_post,
                    }
                )
    train = pd.DataFrame(rows)
    challenge = train.loc[train["study_group"].eq("SDY_Y"), [
        "participant_id", "virus_strain", "subject_group", "study_group", "log2_pre_hai"
    ]].copy()
    challenge["participant_id"] = "challenge-" + challenge["participant_id"].astype(str)
    challenge["subject_group"] = "challenge-" + challenge["subject_group"].astype(str)
    challenge["study_group"] = "CHALLENGE"
    return train, challenge, strains


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    args = p.parse_args()
    runtime = load_runtime(args.runtime.resolve())

    with tempfile.TemporaryDirectory(prefix="e05-v6-frozen-control-") as tmp:
        package = Path(tmp) / "cmi_flu_bundle.zip"
        package.write_bytes(runtime.package_bytes())
        sys.path.insert(0, str(package))
        try:
            from cmi_flu.cv import purged_leave_one_study_out
            from cmi_flu.datasets import HAIModelDataset
            from cmi_flu.evaluation import HAICandidateEvaluation
            from cmi_flu.models import ModelSpec

            legacy_fields = {field.name for field in dataclasses.fields(HAICandidateEvaluation)}
            assert "panel_proxy_fold_metrics" not in legacy_fields
            assert "panel_proxy_fold_summary" not in legacy_fields

            run, _hai = runtime.load_e05_module()
            assert callable(run)
            root = sys.modules["cmi_flu.strategy_e05"]
            compat = sys.modules["cmi_flu.strategy_e05_v4"]
            assert root._control_run is compat._control_run_compat
            assert root._condition_metrics is compat._condition_metrics_compat

            train, challenge, panel = dataset_frames()
            dataset = HAIModelDataset(
                day=28,
                train=train,
                challenge=challenge,
                target_representation="direct",
            )
            dataset.validate()
            splits = purged_leave_one_study_out(
                train["study_group"].astype(str),
                train["subject_group"].astype(str),
            )
            assert len(splits) == 2
            spec = ModelSpec(name="frozen_compat_ridge", family="ridge", params={"alpha": 1.0})

            control = root._control_run(
                dataset,
                spec=spec,
                splits=splits,
                panel_strains=panel,
                name="frozen_compat_control",
                fit_challenge=False,
            )
            fold_metrics = control.metrics.get("panel_proxy_fold_metrics") or []
            fold_summary = control.metrics.get("panel_proxy_fold_summary") or {}
            assert len(fold_metrics) == 2
            assert int(fold_summary.get("count", -1)) == 2
            assert all(str(item.get("split", "")).startswith("study=") for item in fold_metrics)

            condition = root._condition_metrics(control.oof, panel_strains=panel)
            assert len(condition.get("panel_proxy_fold_metrics") or []) == 2
            assert int((condition.get("panel_proxy_fold_summary") or {}).get("count", -1)) == 2
        finally:
            sys.path.remove(str(package))

    print(
        "CMI_FLU_E05_V6_FROZEN_CONTROL_RUN_PASS "
        "legacy_eval=true missing_fold_attributes=true control_run=true condition_metrics=true fold_recompute=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
