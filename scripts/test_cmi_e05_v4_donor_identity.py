#!/usr/bin/env python3
"""Execute E05 004 donor-rank correction on repeated-subject Challenge-shaped rows."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def load_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("e05_v4_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def frame() -> pd.DataFrame:
    rows = []
    strains = ("H1", "H3", "B")
    units = (
        ("P1", "shared", 3.0, 0.40, 0.25, 30.0),
        ("P2", "shared", 4.0, 0.50, 0.75, 35.0),
        ("P3", "other", 3.5, 0.45, 0.50, 40.0),
    )
    for participant, subject, mean, std, breadth, age in units:
        for j, strain in enumerate(strains):
            rows.append(
                {
                    "participant_id": participant,
                    "study_group": "SDY_X",
                    "subject_group": subject,
                    "virus_strain": strain,
                    "log2_pre_hai": 2.5 + 0.25 * j + 0.1 * (participant == "P2"),
                    "hai_log2_mean": mean,
                    "hai_log2_std": std,
                    "hai_breadth_ge_40": breadth,
                    "age": age,
                    "strain_subtype": ("H1N1", "H3N2", "B")[j],
                    "strain_substrate": "MDCK",
                    "ontology_official_component_vaccine": "yes" if j == 0 else "no",
                    "ontology_seq_distance_to_study_vaccine": 0.1 + 0.2 * j,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    a = p.parse_args()
    runtime = load_runtime(a.runtime.resolve())
    with tempfile.TemporaryDirectory(prefix="e05-v4-donor-") as tmp:
        package = Path(tmp) / "cmi_flu_bundle.zip"
        package.write_bytes(runtime.package_bytes())
        sys.path.insert(0, str(package))
        try:
            run, _hai = runtime.load_e05_module()
            assert callable(run)
            base = sys.modules["cmi_flu.strategy_e05"]
            shim = sys.modules["cmi_flu.strategy_e05_v2"]
            assert base._subject_level_rank is shim._donor_level_rank
            source = frame()
            design = base._e05_design(source, interactions=True)
            assert len(design) == len(source)
            joined = source[["participant_id", "subject_group"]].reset_index(drop=True).copy()
            joined["rank"] = design["z_hai_mean_rank"].to_numpy(dtype=float)
            by_participant = joined.groupby("participant_id")["rank"].agg(["nunique", "first"])
            assert int(by_participant["nunique"].max()) == 1
            assert by_participant.loc["P1", "first"] != by_participant.loc["P2", "first"]
            assert source.loc[source["participant_id"].isin(["P1", "P2"]), "subject_group"].nunique() == 1
            assert np.isfinite(design.select_dtypes(include=["number"]).to_numpy(dtype=float)).all()
        finally:
            sys.path.remove(str(package))
    print("CMI_FLU_E05_V4_DONOR_IDENTITY_PASS repeated_subject=true participant_unit_ranks=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
