#!/usr/bin/env python3
"""Credential-free real-library regression for V3-02 source and generated runtime."""
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

import cmi_flu_v3_v02_prepare as prepare


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeConfig:
    specs: dict
    baseline: str = "b021_taskwise_robust"
    random_state: int = 42
    def section(self, name: str):
        if name == "selection": return {"policy": "robust_v1"}
        return {}
    def model_specs(self, name: str): return list(self.specs[name])


def compact_dataset(rt, task: str, studies: list[tuple[str, int]], challenge_ids: list[str]):
    rows = []
    anchor = {"Task1.1": "cytokine_rank__CXCL10", "Task1.2": "flow_rank__Classical_monocytes", "Task1.3": "flow_rank__Antibody-secreting_cells_(ASC)"}[task]
    for sidx, (study, n) in enumerate(studies):
        for i in range(n):
            base = (i + 1) / (n + 1)
            rows.append({"participant_id": f"{study}_P{i}", "subject_group": f"{study}_U{i}", "study_group": study,
                         "x1": float(i + sidx + 1), "x2": float((i % 5) + 1), anchor: base,
                         "target_value": float((1.0 + base + .03 * sidx) if task == "Task1.1" else (.05 + .4 * base + .01 * sidx))})
    challenge = []
    for i, pid in enumerate(challenge_ids):
        base = (i + 1) / 41
        challenge.append({"participant_id": pid, "subject_group": f"CU{i}", "study_group": "2025LJI",
                          "x1": float(i + 1), "x2": float((i % 5) + 1), anchor: base})
    return rt.TaskDataset(task=task, train=pd.DataFrame(rows), challenge=pd.DataFrame(challenge), target_column="target_value")


def hai_dataset(rt, day: int, studies: list[tuple[str, int]], challenge_ids: list[str], strains: tuple[str, ...]):
    rows = []
    for sidx, (study, n) in enumerate(studies):
        for i in range(n):
            for k, strain in enumerate(strains):
                pre = 3.0 + .05 * i + .1 * k
                fold = .2 + .01 * sidx + .005 * i + .01 * k
                rows.append({"participant_id": f"{study}_P{i}", "subject_group": f"{study}_U{i}", "study_group": study,
                             "virus_strain": strain, "strain_subtype": "H1" if k == 0 else "H3",
                             "log2_pre_hai": pre, "target_log2_fold": fold, "post_hai": float(2 ** (pre + fold)),
                             "age": 20 + i % 40})
    challenge = []
    for i, pid in enumerate(challenge_ids):
        for k, strain in enumerate(strains):
            challenge.append({"participant_id": pid, "subject_group": f"CU{i}", "study_group": "2025LJI",
                              "virus_strain": strain, "strain_subtype": "H1" if k == 0 else "H3",
                              "log2_pre_hai": 3.2 + .02 * i + .1 * k, "age": 20 + i % 40})
    return rt.HAIModelDataset(day=day, train=pd.DataFrame(rows), challenge=pd.DataFrame(challenge), target_representation="residual")


def full_science_synthetic(runtime: Path, root: Path) -> None:
    generated = load(runtime, "v302_generated_runtime")
    with tempfile.TemporaryDirectory(prefix="v302-ci-package-") as td:
        package = Path(td) / "bundle.zip"; package.write_bytes(generated.package_bytes()); sys.path.insert(0, str(package))
        try:
            rt = generated.load_v3_v02_module()
            from cmi_flu.models import ModelSpec
            ids = [f"C{i:02d}" for i in range(40)]
            datasets = {
                "Task1.1": compact_dataset(rt, "Task1.1", [("A", 12), ("B", 12), ("C", 12), ("D", 40)], ids),
                "Task1.2": compact_dataset(rt, "Task1.2", [("E", 12), ("F", 40), ("G", 12)], ids),
                "Task1.3": compact_dataset(rt, "Task1.3", [("H", 30)], ids),
            }
            strains = ("H1N1 A/Victoria/4897/2022", "Vic B/Austria/1359417/2021", "H3N2 A/Tasmania/503/2020")
            datasets["HAI_D28"] = hai_dataset(rt, 28, [("Y1", 12), ("Y2", 12), ("Y3", 12), ("Y4", 12)], ids, strains)
            datasets["HAI_D365"] = hai_dataset(rt, 365, [("Z1", 12), ("Z2", 12), ("Z3", 12)], ids, strains)
            specs = {
                "task_11": [ModelSpec("pls_2", "pls", {"n_components": 2}, target_transform="log", clip_min=0.0)],
                "task_12": [ModelSpec("enet_a0.001_l0.5", "elastic_net", {"alpha": .001, "l1_ratio": .5}, target_transform="log1p", clip_min=0.0),
                            ModelSpec("et_d5_l5_sqrt", "extra_trees", {"n_estimators": 30, "max_depth": 5, "min_samples_leaf": 5, "max_features": "sqrt"}, target_transform="log1p", clip_min=0.0)],
                "task_13": [ModelSpec("pls_1", "pls", {"n_components": 1}, target_transform="log1p", clip_min=0.0)],
                "hai": [ModelSpec("et_subtype_d3_l5", "extra_trees", {"n_estimators": 30, "max_depth": 3, "min_samples_leaf": 5, "max_features": "sqrt"}, drop_columns=("virus_strain",)),
                        ModelSpec("et_subtype_d5_l10", "extra_trees", {"n_estimators": 30, "max_depth": 5, "min_samples_leaf": 10, "max_features": "sqrt"}, drop_columns=("virus_strain",)),
                        ModelSpec("ridge_exact_a100", "ridge", {"alpha": 100.0})],
            }
            config = FakeConfig(specs)
            inputs = SimpleNamespace(vaccine_strains=strains[:2], challenge_strains=strains)
            native_builder = rt.build_b02_datasets; rt.build_b02_datasets = lambda config, inputs: datasets
            try:
                source = pd.DataFrame({"participant_id": ids})
                for j, task in enumerate(rt.TASK_UNITS): source[task] = np.linspace(0.01 + j, 1.0 + j, 40)
                aggregate, oof, challenge = rt.run_v3_02(config, inputs, source_b_csv=source)
            finally:
                rt.build_b02_datasets = native_builder
            assert aggregate["new_candidate_conditions"] == 0
            assert 0 < aggregate["fit_count"] <= 256
            assert aggregate["bank"]["oof_rows"] == len(oof)
            assert aggregate["bank"]["challenge_rows"] == len(challenge)
            assert set(aggregate["pseudo40_12_28"]) == {"Task1.1", "Task1.2"}
            assert all(aggregate["hai_complete_local_panel_p1"][task] for task in ("Task2.1", "Task2.2", "Task2.3"))
            with tempfile.TemporaryDirectory(prefix="v302-ci-output-") as out:
                manifest = rt.write_v3_02_outputs(aggregate, oof, challenge, out)
                assert len(manifest["files"]) == 2
                assert all(x["private_row_level"] for x in manifest["files"])
                safe = (Path(out) / "v3_v02_summary.json").read_text() + (Path(out) / "v3_v02_bank_manifest.json").read_text()
                assert "C00" not in safe
            print(f"V302_FULL_SCIENCE_SYNTHETIC PASS fit_count={aggregate['fit_count']} oof_rows={len(oof)} challenge_rows={len(challenge)}")
        finally:
            if sys.path and sys.path[0] == str(package): sys.path.pop(0)


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--repository-root", required=True, type=Path); p.add_argument("--workdir", required=True, type=Path); a = p.parse_args()
    root, work = a.repository_root.resolve(), a.workdir.resolve(); work.mkdir(parents=True, exist_ok=True)
    one, two = work / "runtime.py", work / "runtime2.py"
    prepare.build_runtime(root, one); prepare.build_runtime(root, two)
    assert one.read_bytes() == two.read_bytes(), "V3-02 runtime materialization not deterministic"
    compile(one.read_text(), str(one), "exec")
    mod = load(one, "v302_selftest_runtime"); assert mod.self_test() == 0
    full_science_synthetic(one, root)
    raw = one.read_bytes(); print(f"V302_CI PASS bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()} real_libraries=true synthetic_only=true kaggle_write=0 competition_submit=0")
    return 0


if __name__ == "__main__": raise SystemExit(main())
