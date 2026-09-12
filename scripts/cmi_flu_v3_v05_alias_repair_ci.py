#!/usr/bin/env python3
"""Credential-free regression for the V3-05 real-data study-alias failure repair."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import pandas as pd

import cmi_flu_v3_v05_alias_repair_prepare as prepare
import cmi_flu_v3_v05_ci as parent_ci


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def real_mismatch_synthetic_tables() -> dict[str, pd.DataFrame]:
    """Reproduce the real organizer/context mismatch that escaped the 001 CI."""
    tables = parent_ci.synthetic_tables()
    participants = tables["participants"].copy()
    investigations = tables["investigations"].copy()
    source_participant = participants["study_accession"].eq("2024UGA")
    source_investigation = investigations["study_accession"].eq("2024UGA")
    if int(source_participant.sum()) != 23 or int(source_investigation.sum()) != 1:
        raise AssertionError("synthetic source fixture identity changed")
    participants.loc[source_participant, "study_accession"] = "2024_UGA"
    investigations.loc[source_investigation, "study_accession"] = "2024_UGA"
    tables["participants"] = participants
    tables["investigations"] = investigations
    # public_flow intentionally remains 2024UGA, matching the real V3-01 audit
    # where measurement metadata is canonical while participant context is raw.
    assert set(tables["public_flow"]["study_accession"].astype(str)) == {"2024UGA"}
    return tables


def full_repair_synthetic(runtime: Path) -> None:
    generated = load(runtime, "v305_repair_generated_runtime")
    with tempfile.TemporaryDirectory(prefix="v305-repair-ci-package-") as td:
        package = Path(td) / "bundle.zip"
        package.write_bytes(generated.package_bytes())
        sys.path.insert(0, str(package))
        try:
            repair = generated.load_v3_v05_v2_module()
            from cmi_flu.contracts import DataContractError
            from cmi_flu.models import ModelSpec

            config = parent_ci.FakeConfig({
                "task_13": [ModelSpec("pls_1", "pls", {"n_components": 1}, target_transform="log1p", clip_min=0.0)]
            })
            inputs = SimpleNamespace(tables=real_mismatch_synthetic_tables())

            # Prove the synthetic fixture reproduces the exact consumed-001 science
            # failure before exercising the repair.
            try:
                repair._base.run_v3_05_task13(config, inputs)
            except DataContractError as exc:
                assert str(exc) == "V3-05 source cohort is not the frozen single 2024UGA domain"
            else:
                raise AssertionError("consumed-001 study-label failure was not reproduced")

            aggregate, oof, challenge = repair.run_v3_05_task13(config, inputs)
            assert aggregate["new_candidate_conditions"] == ["S1", "S2", "S3", "S4"]
            assert aggregate["new_candidate_condition_count"] == 4
            assert aggregate["Task1.2"]["state"] == "not_executed_data_limited"
            assert aggregate["Task1.3"]["measurement"]["source_subjects"] == 23
            assert aggregate["Task1.3"]["measurement"]["challenge_subjects"] == 40
            assert aggregate["fit_count"] == 80 and aggregate["fit_count"] <= 96
            assert aggregate["raw_unit_conversion_performed"] is False
            assert aggregate["pseudocount_added"] is False
            assert aggregate["public_leaderboard_used"] is False
            assert aggregate["competition_submission_attempted"] is False
            assert len(oof) == 23 * 3
            assert len(challenge) == 40
            item = aggregate["runtime_repair"]
            assert item["parent_failure_code"] == prepare.FAILURE_CODE
            assert item["study_alias_sha256"] == prepare.ALIAS_HASH
            assert item["scope"] == "participant_context.study_group_identity_only"
            for key in (
                "teacher_changed", "target_changed", "measurement_values_changed", "features_changed",
                "splits_changed", "candidate_formulas_changed", "fit_budget_changed",
                "unit_conversion_performed", "public_leaderboard_used", "competition_submission_attempted",
            ):
                assert item[key] is False
            assert repair.canonical_study("2024_UGA") == "2024UGA"
            assert repair.canonical_study("2024 UGA") == "2024 UGA"

            with tempfile.TemporaryDirectory(prefix="v305-repair-ci-output-") as out:
                manifest = repair.write_v3_05_outputs(aggregate, oof, challenge, out)
                assert [x["filename"] for x in manifest["files"]] == [
                    "v3_v05_task13_oof_bank.csv", "v3_v05_task13_challenge_bank.csv"
                ]
                safe = (Path(out) / "v3_v05_task13_summary.json").read_text() + (
                    Path(out) / "v3_v05_task13_bank_manifest.json"
                ).read_text()
                assert "UGA00" not in safe and "LJI00" not in safe
            print(
                f"V305_ALIAS_REPAIR_FULL_SYNTHETIC PASS reproduced_parent_failure={prepare.FAILURE_CODE} "
                f"fit_count={aggregate['fit_count']} oof_rows={len(oof)} challenge_rows={len(challenge)} conditions=4"
            )
        finally:
            if sys.path and sys.path[0] == str(package):
                sys.path.pop(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    one, two = work / "runtime.py", work / "runtime2.py"
    prepare.build_runtime(root, one)
    prepare.build_runtime(root, two)
    assert one.read_bytes() == two.read_bytes(), "V3-05 repair runtime materialization not deterministic"
    compile(one.read_text(), str(one), "exec")
    module = load(one, "v305_repair_selftest_runtime")
    assert module.self_test() == 0
    full_repair_synthetic(one)
    raw = one.read_bytes()
    print(
        f"V305_ALIAS_REPAIR_CI PASS bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()} "
        "real_libraries=true real_mismatch_reproduced=true kaggle_write=0 competition_submit=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
