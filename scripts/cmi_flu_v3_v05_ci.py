#!/usr/bin/env python3
"""Credential-free real-library regression for V3-05 generated runtime/science."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import numpy as np
import pandas as pd

import cmi_flu_v3_v05_prepare as prepare


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeConfig:
    specs: dict
    def model_specs(self, name: str):
        return list(self.specs[name])


def synthetic_tables() -> dict[str, pd.DataFrame]:
    participants = []
    public = []
    challenge = []
    gate = "CD19+ CD20- CD27+ CD38++"
    for i in range(23):
        pid = f"UGA{i:02d}"
        participants.append({
            "participant_id": pid, "subject": f"U{i:02d}", "study_accession": "2024UGA",
            "arm_id": "A", "age": 20 + i, "biological_sex": "female" if i % 2 else "male",
            "race": "synthetic",
        })
        baseline = 0.05 + 0.018 * i + 0.002 * (i % 3)
        target = baseline * np.exp(0.35 + 0.004 * (20 + i - 31)) + 0.002 * (i % 2)
        common = {
            "participant_id": pid, "subject": f"U{i:02d}", "study_accession": "2024UGA",
            "name": "Antibody-secreting cells (ASC)", "population_definition": gate,
            "parent_population": "PBMC", "unit": "percentage", "material": "PBMC", "comments": "",
        }
        public.append({**common, "timepoint": "Pre-vacc", "value": baseline})
        public.append({**common, "timepoint": "7", "value": target})
    for i in range(40):
        pid = f"LJI{i:02d}"
        participants.append({
            "participant_id": pid, "subject": f"L{i:02d}", "study_accession": "2025LJI",
            "arm_id": "B", "age": 24 + (i % 35), "biological_sex": "female" if i % 2 else "male",
            "race": "synthetic",
        })
        baseline = 0.045 + 0.011 * i + 0.001 * (i % 4)
        challenge.append({
            "participant_id": pid, "subject": f"L{i:02d}", "study_accession": "2025LJI",
            "timepoint": "Pre-vacc", "name": "Antibody-secreting cells (ASC)", "value": baseline,
            "population_definition": gate, "parent_population": "PBMC", "unit": "percentage",
            "material": "PBMCs", "comments": "",
        })
    investigations = pd.DataFrame([
        {"study_accession": "2024UGA", "arm_id": "A", "vaccine_season": "2023-24", "vaccine": "V1", "year": 2024},
        {"study_accession": "2025LJI", "arm_id": "B", "vaccine_season": "2024-25", "vaccine": "V2", "year": 2025},
    ])
    return {
        "participants": pd.DataFrame(participants),
        "investigations": investigations,
        "public_flow": pd.DataFrame(public),
        "challenge_flow": pd.DataFrame(challenge),
    }


def full_science_synthetic(runtime: Path) -> None:
    generated = load(runtime, "v305_generated_runtime")
    with tempfile.TemporaryDirectory(prefix="v305-ci-package-") as td:
        package = Path(td) / "bundle.zip"
        package.write_bytes(generated.package_bytes())
        sys.path.insert(0, str(package))
        try:
            rt = generated.load_v3_v05_module()
            from cmi_flu.models import ModelSpec
            config = FakeConfig({
                "task_13": [
                    ModelSpec("pls_1", "pls", {"n_components": 1}, target_transform="log1p", clip_min=0.0)
                ]
            })
            inputs = SimpleNamespace(tables=synthetic_tables())
            aggregate, oof, challenge = rt.run_v3_05_task13(config, inputs)
            assert aggregate["new_candidate_conditions"] == ["S1", "S2", "S3", "S4"]
            assert aggregate["new_candidate_condition_count"] == 4
            assert aggregate["Task1.2"]["state"] == "not_executed_data_limited"
            assert aggregate["Task1.3"]["measurement"]["source_subjects"] == 23
            assert aggregate["Task1.3"]["measurement"]["challenge_subjects"] == 40
            assert aggregate["Task1.3"]["measurement"]["legacy_b21_builder_target_exact_match"] is True
            assert aggregate["fit_count"] == 80
            assert aggregate["fit_count"] <= 96
            assert aggregate["raw_unit_conversion_performed"] is False
            assert aggregate["pseudocount_added"] is False
            assert aggregate["competition_submission_attempted"] is False
            assert len(oof) == 23 * 3
            assert len(challenge) == 40
            for name in ("S2_vs_raw_baseline", "S3_vs_raw_baseline"):
                contract = aggregate["Task1.3"]["challenge_rank_contracts"][name]
                assert contract["same_rank_vector"] is True
                assert contract["same_tie_equivalence"] is True
            assert aggregate["Task1.3"]["conditions"]["S4"]["state"] == "evaluated"
            with tempfile.TemporaryDirectory(prefix="v305-ci-output-") as out:
                manifest = rt.write_v3_05_outputs(aggregate, oof, challenge, out)
                assert [x["filename"] for x in manifest["files"]] == [
                    "v3_v05_task13_oof_bank.csv", "v3_v05_task13_challenge_bank.csv"
                ]
                safe = (Path(out) / "v3_v05_task13_summary.json").read_text() + (
                    Path(out) / "v3_v05_task13_bank_manifest.json"
                ).read_text()
                assert "UGA00" not in safe and "LJI00" not in safe
            print(
                f"V305_FULL_SCIENCE_SYNTHETIC PASS fit_count={aggregate['fit_count']} "
                f"oof_rows={len(oof)} challenge_rows={len(challenge)} conditions=4"
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
    assert one.read_bytes() == two.read_bytes(), "V3-05 runtime materialization not deterministic"
    compile(one.read_text(), str(one), "exec")
    module = load(one, "v305_selftest_runtime")
    assert module.self_test() == 0
    full_science_synthetic(one)
    raw = one.read_bytes()
    print(
        f"V305_CI PASS bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()} "
        "real_libraries=true synthetic_only=true kaggle_write=0 competition_submit=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
