#!/usr/bin/env python3
"""Standalone CMI-Flu B1 anchor runner for one private Kaggle CPU kernel.

The implementation mirrors the B1 contracts at CMI-Flu commit
1e075cfa565698f708e872d22be629f97704a24f but intentionally depends only on
the Competition mount and Kaggle's preinstalled scientific Python stack.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Iterable, Sequence

COMPETITION = "cmi-flu-first-prediction-challenge"
REQUEST_ID = "20260902-cmi-flu-b1-002"
CMI_COMMIT = "1e075cfa565698f708e872d22be629f97704a24f"
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_SUBMISSION = OUTPUT_DIR / "submission.csv"
OUTPUT_METADATA = OUTPUT_DIR / "bridge-result.json"
OUTPUT_SUMMARY = OUTPUT_DIR / "summary.md"
OUTPUT_FAILURE = OUTPUT_DIR / "bridge-failure.json"
TASK_COLUMNS = (
    "Task1.1",
    "Task1.2",
    "Task1.3",
    "Task1.4",
    "Task2.1",
    "Task2.2",
    "Task2.3",
)
PUBLIC_TASKS = TASK_COLUMNS[:-1]
EXPECTED_AUDITS = {
    "Task1.1": (127, 127, 4),
    "Task1.2": (81, 81, 3),
    "Task1.3": (29, 29, 1),
    "HAI_D28_challenge_overlap": (10_617, 2_689, 33),
    "HAI_D28_vaccine_overlap": (1_141, 903, 5),
    "HAI_D365_challenge_overlap": (4_878, 914, 8),
}
SOURCE_BLOBS = {
    "aliases.py": "5b9930c8e26f2b462ee8ce64fdfbdb76df2d1af3",
    "anchors.py": "5696fb9e4997bc95360b0c2edb7b15d11bcd4a29",
    "audit.py": "93bdd0df48f0d002bd144fb4c5547565763b31e6",
    "contracts.py": "6e2791e526255d9c534e71c9c0886f0c300df7bd",
    "metrics.py": "8eae3c881e1359d720dfb39303bbedc66728ea7d",
    "submission.py": "78a9c8918ae94ca94760178b24eba65682672142",
    "targets.py": "154681d9a51e36abd635b596c6d8f7cee22c2d96",
}


class ContractError(ValueError):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


def fail(code: str, message: str = "") -> None:
    raise ContractError(code, message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text(value: object) -> str:
    import pandas as pd

    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def canonical_timepoint(value: object) -> str:
    raw = text(value)
    if not raw:
        return raw
    compact = raw.casefold().replace("_", "-").replace(" ", "")
    if compact in {"pre-vacc", "prevacc", "pre-vax", "prevax", "baseline"}:
        return "Pre-vacc"
    if compact.startswith("day"):
        compact = compact[3:]
    if compact.startswith("d") and compact[1:].lstrip("-").replace(".", "", 1).isdigit():
        compact = compact[1:]
    try:
        numeric = float(compact)
    except ValueError:
        return raw
    return str(int(numeric)) if numeric.is_integer() else format(numeric, "g")


def canonical_cytokine(value: object) -> str:
    raw = text(value)
    mapping = {
        "IP10": "CXCL10",
        "IP-10": "CXCL10",
        "CXCL10": "CXCL10",
        "IL8": "CXCL8",
        "IL-8": "CXCL8",
        "CXCL8": "CXCL8",
        "VEGF": "VEGFA",
        "VEGF-A": "VEGFA",
        "VEGFA": "VEGFA",
        "IL1RA": "IL1RN",
        "IL-1RA": "IL1RN",
        "IL1RN": "IL1RN",
        "IL18": "IL18",
        "IL-18": "IL18",
    }
    return mapping.get(raw.upper(), raw)


def canonical_flow(value: object) -> str:
    raw = text(value)
    if not raw:
        return raw
    normalized = re.sub(r"[\s/]+", "_", raw)
    normalized = re.sub(r"_+", "_", normalized)
    mapping = {
        "CLASSICAL_MONOCYTES": "Classical_monocytes",
        "CLASSICAL_MONOCYTE": "Classical_monocytes",
        "ANTIBODY-SECRETING_CELLS_(ASC)": "Antibody-secreting_cells_(ASC)",
        "ANTIBODY_SECRETING_CELLS_(ASC)": "Antibody-secreting_cells_(ASC)",
        "ASC": "Antibody-secreting_cells_(ASC)",
    }
    return mapping.get(normalized.upper(), normalized)


def canonical_aim(value: object) -> str:
    raw = text(value)
    mapping = {"CON": "Conserved", "CONSERVED": "Conserved", "CONSERVED POOL": "Conserved"}
    return mapping.get(raw.upper(), raw)


def canonical_strain(value: object) -> str:
    raw = text(value)
    if not raw:
        return raw
    raw = re.sub(r"(?i)_cell$", "_MDCK", raw)
    raw = re.sub(r"(?i)\s+cell$", "_MDCK", raw)
    return raw


def require_columns(frame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        fail("missing_columns", f"{name}: {missing}")


def present_ids(frame) -> list[str]:
    columns = [column for column in ("participant_id", "subject", "study_accession") if column in frame]
    if "participant_id" not in columns:
        fail("missing_participant_id")
    return columns


def numeric(series, name: str, *, allow_missing: bool = True):
    import pandas as pd

    converted = pd.to_numeric(series, errors="coerce")
    newly_missing = series.notna() & converted.isna()
    if newly_missing.any():
        fail("non_numeric_value", name)
    if not allow_missing and converted.isna().any():
        fail("missing_numeric_value", name)
    return converted


def aggregate_mean(frame, groups: Sequence[str], value_column: str, output_column: str):
    grouped = (
        frame.groupby(list(groups), dropna=False, observed=True)[value_column]
        .mean()
        .rename(output_column)
        .reset_index()
    )
    if grouped.duplicated(list(groups)).any():
        fail("duplicate_aggregate_key", output_column)
    return grouped


def select_pre_vacc(frame, groups: Sequence[str], *, value_column: str = "value", output: str = "pre_vacc"):
    import pandas as pd

    require_columns(frame, [*groups, "timepoint", value_column], "baseline source")
    working = frame.copy()
    working["timepoint"] = working["timepoint"].map(canonical_timepoint)
    working[value_column] = numeric(working[value_column], value_column, allow_missing=True)
    working = working.dropna(subset=[value_column])

    literal = aggregate_mean(
        working.loc[working["timepoint"].eq("Pre-vacc")].copy(), groups, value_column, output
    )
    numeric_timepoint = pd.to_numeric(working["timepoint"], errors="coerce")
    fallback = aggregate_mean(
        working.loc[numeric_timepoint.notna() & numeric_timepoint.le(0)].copy(),
        groups,
        value_column,
        output,
    )
    if literal.empty:
        result = fallback
    elif fallback.empty:
        result = literal
    else:
        literal_keys = pd.MultiIndex.from_frame(literal[list(groups)])
        fallback_keys = pd.MultiIndex.from_frame(fallback[list(groups)])
        result = pd.concat([literal, fallback.loc[~fallback_keys.isin(literal_keys)]], ignore_index=True)
    if result.empty:
        fail("no_pre_vacc_values")
    if result.duplicated(list(groups)).any():
        fail("duplicate_pre_vacc_key")
    return result


def select_timepoint(frame, day: int, groups: Sequence[str], *, output: str):
    working = frame.copy()
    working["timepoint"] = working["timepoint"].map(canonical_timepoint)
    selected = working.loc[working["timepoint"].eq(str(day))].copy()
    if selected.empty:
        fail("missing_target_timepoint", str(day))
    selected["value"] = numeric(selected["value"], "value", allow_missing=True)
    selected = selected.dropna(subset=["value"])
    return aggregate_mean(selected, groups, "value", output)


def positive(values, name: str) -> None:
    import numpy as np

    array = np.asarray(values, dtype=float)
    if ((~np.isfinite(array)) | (array <= 0)).any():
        fail("non_positive_value", name)


def build_task11_target(cytokine):
    import numpy as np

    require_columns(cytokine, ["participant_id", "timepoint", "analyte", "value"], "cytokine")
    working = cytokine.copy()
    working["analyte"] = working["analyte"].map(canonical_cytokine)
    working = working.loc[working["analyte"].eq("CXCL10")].copy()
    ids = present_ids(working)
    groups = [*ids, "analyte"]
    pre = select_pre_vacc(working, groups, output="pre_vacc")
    day1 = select_timepoint(working, 1, groups, output="day1")
    target = pre.merge(day1, on=groups, how="inner", validate="one_to_one")
    positive(target["pre_vacc"], "Task1.1 pre")
    positive(target["day1"], "Task1.1 day1")
    target["target"] = target["day1"] / target["pre_vacc"]
    target["target_log"] = np.log(target["target"])
    return target


def build_flow_target(flow, population: str, day: int):
    require_columns(flow, ["participant_id", "timepoint", "name", "value"], "flow")
    working = flow.copy()
    working["name"] = working["name"].map(canonical_flow)
    working = working.loc[working["name"].eq(canonical_flow(population))].copy()
    ids = present_ids(working)
    return select_timepoint(working, day, [*ids, "name"], output="target")


def build_task14_anchor(aim):
    require_columns(aim, ["participant_id", "timepoint", "stimulation", "value"], "AIM")
    working = aim.copy()
    working["stimulation"] = working["stimulation"].map(canonical_aim)
    working = working.loc[working["stimulation"].eq("Conserved")].copy()
    ids = present_ids(working)
    optional = ["name"] if "name" in working else []
    baseline = select_pre_vacc(working, [*ids, "stimulation", *optional], output="anchor")
    if optional:
        baseline = (
            baseline.groupby(ids, dropna=False, observed=True)["anchor"]
            .mean()
            .rename("anchor")
            .reset_index()
        )
    return baseline[[*ids, "anchor"]]


def single_readout_anchor(frame, feature_column: str, feature_value: str, canonicalizer, *, strict_flow=False):
    require_columns(frame, ["participant_id", "timepoint", feature_column, "value"], "anchor source")
    working = frame.copy()
    working[feature_column] = working[feature_column].map(canonicalizer)
    working = working.loc[working[feature_column].eq(canonicalizer(feature_value))].copy()
    if strict_flow:
        require_columns(working, ["unit", "material"], "strict flow anchor")
        unit = working["unit"].fillna("").astype(str).str.strip().str.casefold()
        material = working["material"].fillna("").astype(str).str.strip().str.casefold()
        working = working.loc[
            unit.isin({"percentage", "% of parent", "percent", "%"}) & material.str.startswith("pbmc")
        ].copy()
    ids = present_ids(working)
    baseline = select_pre_vacc(working, [*ids, feature_column], output="anchor")
    if baseline.empty:
        fail("empty_anchor", feature_value)
    result = baseline[[*ids, "anchor"]]
    if result.duplicated(["participant_id"]).any():
        fail("duplicate_anchor_participant", feature_value)
    return result


def safe_spearman(true, pred):
    import numpy as np
    from scipy.stats import spearmanr

    a = np.asarray(true, dtype=float)
    b = np.asarray(pred, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return None
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else None


def infer_task11_direction(public_cytokine) -> int:
    import numpy as np

    target = build_task11_target(public_cytokine)
    study_values: list[float] = []
    if "study_accession" in target:
        for _, group in target.groupby("study_accession", dropna=False, observed=True):
            metric = safe_spearman(group["pre_vacc"], group["target"])
            if metric is not None and metric != 0:
                study_values.append(metric)
    if study_values:
        median = float(np.median(study_values))
        if median != 0:
            return 1 if median > 0 else -1
        sign_sum = int(np.sign(study_values).sum())
        if sign_sum != 0:
            return 1 if sign_sum > 0 else -1
    pooled = safe_spearman(target["pre_vacc"], target["target"])
    if pooled is not None and pooled != 0:
        return 1 if pooled > 0 else -1
    return -1


def percentile_rank(values, direction: int):
    import numpy as np
    from scipy.stats import rankdata

    array = np.asarray(values, dtype=float) * direction
    if not np.isfinite(array).all() or array.size == 0:
        fail("invalid_rank_input")
    if array.size == 1:
        return np.array([0.5], dtype=float)
    ranks = rankdata(array, method="average")
    return (ranks - 1.0) / (array.size - 1.0)


def prepare_hai(serology):
    require_columns(serology, ["participant_id", "timepoint", "virus_strain", "assay", "value"], "serology")
    working = serology.copy()
    assay = working["assay"].fillna("").astype(str).str.strip().str.casefold()
    working = working.loc[assay.eq("hai")].copy()
    working["virus_strain"] = working["virus_strain"].map(canonical_strain)
    working["value"] = numeric(working["value"], "HAI value", allow_missing=True)
    working = working.dropna(subset=["value"])
    positive(working["value"], "HAI value")
    return working


def geometric_mean(values: Iterable[float]) -> float:
    import numpy as np

    array = np.asarray(list(values), dtype=float)
    positive(array, "geometric mean")
    return float(np.exp(np.mean(np.log(array))))


def build_hai_anchor(serology, panel_strains: Sequence[str]):
    working = prepare_hai(serology)
    ids = present_ids(working)
    baseline = select_pre_vacc(working, [*ids, "virus_strain"], output="pre_hai")
    panel = tuple(canonical_strain(value) for value in panel_strains)
    if len(panel) == 0 or len(set(panel)) != len(panel):
        fail("invalid_hai_panel")
    selected = baseline.loc[baseline["virus_strain"].isin(panel)].copy()
    if selected.duplicated([*ids, "virus_strain"]).any():
        fail("duplicate_hai_panel_strain")
    counts = selected.groupby(ids, dropna=False, observed=True)["virus_strain"].nunique()
    if (counts != len(panel)).any():
        fail("incomplete_challenge_hai_panel")
    return (
        selected.groupby(ids, dropna=False, observed=True)["pre_hai"]
        .apply(geometric_mean)
        .rename("anchor")
        .reset_index()
    )


def derive_panels(challenge_serology):
    require_columns(challenge_serology, ["assay", "virus_strain", "virus_in_vaccine"], "challenge serology")
    assay = challenge_serology["assay"].fillna("").astype(str).str.strip().str.casefold()
    hai = challenge_serology.loc[assay.eq("hai")].copy()
    raw_challenge = tuple(dict.fromkeys(hai["virus_strain"].dropna().astype(str).str.strip()))
    if len(raw_challenge) != 12:
        fail("unexpected_challenge_hai_strain_count", str(len(raw_challenge)))
    numeric_flag = __import__("pandas").to_numeric(hai["virus_in_vaccine"], errors="coerce")
    string_flag = hai["virus_in_vaccine"].fillna("").astype(str).str.strip().str.casefold()
    flag = numeric_flag.eq(1) | string_flag.isin({"true", "yes", "y"})
    raw_vaccine = tuple(dict.fromkeys(hai.loc[flag, "virus_strain"].dropna().astype(str).str.strip()))
    if len(raw_vaccine) != 3:
        fail("unexpected_vaccine_hai_strain_count", str(len(raw_vaccine)))
    challenge = tuple(dict.fromkeys(canonical_strain(value) for value in raw_challenge))
    vaccine = tuple(dict.fromkeys(canonical_strain(value) for value in raw_vaccine))
    if len(challenge) != 12 or len(vaccine) != 3 or not set(vaccine).issubset(set(challenge)):
        fail("canonical_hai_panel_contract")
    return raw_challenge, raw_vaccine, challenge, vaccine


def audit_frame(name: str, frame):
    require_columns(frame, ["participant_id", "study_accession"], name)
    actual = (
        int(len(frame)),
        int(frame["participant_id"].nunique()),
        int(frame["study_accession"].nunique()),
    )
    if actual != EXPECTED_AUDITS[name]:
        fail("aggregate_audit_mismatch", name)
    return {
        "rows": actual[0],
        "participants": actual[1],
        "studies": actual[2],
        "passed": True,
    }


def hai_audit_rows(public_serology, day: int, raw_panel: Sequence[str]):
    require_columns(
        public_serology,
        ["participant_id", "study_accession", "timepoint", "virus_strain", "assay"],
        "public serology",
    )
    working = public_serology.copy()
    assay = working["assay"].fillna("").astype(str).str.strip().str.casefold()
    working = working.loc[assay.eq("hai")].copy()
    working["timepoint"] = working["timepoint"].map(canonical_timepoint)
    raw = working["virus_strain"].fillna("").astype(str).str.strip()
    return working.loc[working["timepoint"].eq(str(day)) & raw.isin(set(raw_panel))].copy()


def run_audits(public_cytokine, public_flow, public_serology, raw_challenge, raw_vaccine):
    checks = {
        "Task1.1": audit_frame("Task1.1", build_task11_target(public_cytokine)),
        "Task1.2": audit_frame(
            "Task1.2", build_flow_target(public_flow, "Classical_monocytes", 1)
        ),
        "Task1.3": audit_frame(
            "Task1.3", build_flow_target(public_flow, "Antibody-secreting_cells_(ASC)", 7)
        ),
        "HAI_D28_challenge_overlap": audit_frame(
            "HAI_D28_challenge_overlap", hai_audit_rows(public_serology, 28, raw_challenge)
        ),
        "HAI_D28_vaccine_overlap": audit_frame(
            "HAI_D28_vaccine_overlap", hai_audit_rows(public_serology, 28, raw_vaccine)
        ),
        "HAI_D365_challenge_overlap": audit_frame(
            "HAI_D365_challenge_overlap", hai_audit_rows(public_serology, 365, raw_challenge)
        ),
    }
    return checks


def build_predictions(public_cytokine, challenge_cytokine, challenge_flow, challenge_aim, challenge_serology, vaccine, challenge):
    direction11 = infer_task11_direction(public_cytokine)
    anchors = {
        "Task1.1": single_readout_anchor(
            challenge_cytokine, "analyte", "CXCL10", canonical_cytokine
        ),
        "Task1.2": single_readout_anchor(
            challenge_flow,
            "name",
            "Classical_monocytes",
            canonical_flow,
            strict_flow=True,
        ),
        "Task1.3": single_readout_anchor(
            challenge_flow,
            "name",
            "Antibody-secreting_cells_(ASC)",
            canonical_flow,
            strict_flow=True,
        ),
        "Task1.4": build_task14_anchor(challenge_aim),
        "Task2.1": build_hai_anchor(challenge_serology, vaccine),
        "Task2.2": build_hai_anchor(challenge_serology, challenge),
        "Task2.3": build_hai_anchor(challenge_serology, challenge),
    }
    directions = {
        "Task1.1": direction11,
        "Task1.2": 1,
        "Task1.3": 1,
        "Task1.4": 1,
        "Task2.1": 1,
        "Task2.2": 1,
        "Task2.3": 1,
    }
    predictions = {}
    diagnostics = {"directions": directions, "tasks": {}}
    for task, anchor in anchors.items():
        if anchor.duplicated(["participant_id"]).any():
            fail("duplicate_anchor_participant", task)
        output = anchor[["participant_id"]].copy()
        output["prediction"] = percentile_rank(anchor["anchor"], directions[task])
        predictions[task] = output
        diagnostics["tasks"][task] = {
            "rows": int(len(output)),
            "anchor_unique": int(anchor["anchor"].nunique(dropna=False)),
            "prediction_unique": int(output["prediction"].nunique(dropna=False)),
        }
    return predictions, diagnostics


def build_submission(sample, predictions):
    import numpy as np
    import pandas as pd

    expected_columns = ("participant_id", *TASK_COLUMNS)
    if tuple(sample.columns) != expected_columns or len(sample) != 40:
        fail("sample_submission_contract")
    if sample["participant_id"].isna().any() or sample["participant_id"].duplicated().any():
        fail("sample_participant_contract")
    result = sample.copy()
    ids = result["participant_id"]
    unique_counts = {}
    for task in TASK_COLUMNS:
        prediction = predictions[task]
        if prediction["participant_id"].duplicated().any():
            fail("duplicate_prediction_participant", task)
        mapped = ids.map(prediction.set_index("participant_id")["prediction"])
        if mapped.isna().any():
            fail("missing_prediction_participant", task)
        values = pd.to_numeric(mapped, errors="raise").astype(float)
        if not np.isfinite(values.to_numpy()).all() or (values == -99).any():
            fail("invalid_submission_value", task)
        result[task] = values
        unique_counts[task] = int(values.nunique(dropna=False))
    if any(unique_counts[task] < 2 for task in PUBLIC_TASKS):
        fail("constant_public_task")
    return result, unique_counts


def parse_md5_manifest(path: Path):
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^([0-9a-fA-F]{32})\s+\*?(.+)$", stripped)
        if not match:
            fail("invalid_md5_manifest_line")
        entries.append((match.group(1).lower(), match.group(2).strip()))
    if not entries:
        fail("empty_md5_manifest")
    return entries


def verify_md5(data_dir: Path):
    manifest = data_dir / "md5sum"
    if not manifest.is_file():
        fail("missing_md5_manifest")
    verified = 0
    skipped_self = 0
    for expected, relative in parse_md5_manifest(manifest):
        normalized = relative.lstrip("./")
        if Path(normalized).name == "md5sum":
            skipped_self += 1
            continue
        target = data_dir / normalized
        if not target.is_file():
            fail("md5_target_missing", normalized)
        if md5(target) != expected:
            fail("md5_mismatch", normalized)
        verified += 1
    if verified != 28 or skipped_self != 1:
        fail("md5_manifest_count_contract")
    return {"verified": verified, "skipped_self": skipped_self}


def find_competition_input() -> Path:
    root = Path("/kaggle/input")
    matches = list(dict.fromkeys(path.parent for path in root.rglob("sample_submission_part1.csv")))
    if len(matches) != 1:
        fail("competition_mount_count", str(len(matches)))
    return matches[0]


def read_table(path: Path):
    import pandas as pd

    sep = "\t" if path.suffix.casefold() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def package_versions():
    import importlib.metadata

    names = ("numpy", "pandas", "scipy")
    return {name: importlib.metadata.version(name) for name in names}


def write_failure(stage: str, error: BaseException) -> None:
    code = error.code if isinstance(error, ContractError) else "unexpected_exception"
    payload = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": COMPETITION,
        "cmi_commit": CMI_COMMIT,
        "stage": stage,
        "exception_type": type(error).__name__,
        "error_code": code,
    }
    OUTPUT_FAILURE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    import pandas as pd

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stage = "runtime_packages"
    try:
        versions = package_versions()
        stage = "locate_competition_data"
        data_dir = find_competition_input()
        stage = "verify_md5"
        checksum = verify_md5(data_dir)
        stage = "load_required_tables"
        files = {
            "public_cytokine": "publicData_cytokine.tsv",
            "public_flow": "publicData_ex_vivo_flow.tsv",
            "public_serology": "publicData_serology_260821.tsv",
            "challenge_aim": "2025LJI_aim.tsv",
            "challenge_cytokine": "2025LJI_cytokine.tsv",
            "challenge_flow": "2025LJI_ex_vivo_flow.tsv",
            "challenge_serology": "2025LJI_serology.tsv",
            "sample": "sample_submission_part1.csv",
        }
        tables = {}
        for key, filename in files.items():
            path = data_dir / filename
            if not path.is_file():
                fail("required_competition_file_missing", filename)
            tables[key] = read_table(path)
        stage = "derive_hai_panels"
        raw_challenge, raw_vaccine, challenge, vaccine = derive_panels(tables["challenge_serology"])
        stage = "aggregate_audits"
        audits = run_audits(
            tables["public_cytokine"],
            tables["public_flow"],
            tables["public_serology"],
            raw_challenge,
            raw_vaccine,
        )
        stage = "build_anchor_predictions"
        predictions, diagnostics = build_predictions(
            tables["public_cytokine"],
            tables["challenge_cytokine"],
            tables["challenge_flow"],
            tables["challenge_aim"],
            tables["challenge_serology"],
            vaccine,
            challenge,
        )
        stage = "build_submission"
        submission, unique_counts = build_submission(tables["sample"], predictions)
        submission.to_csv(OUTPUT_SUBMISSION, index=False)
        stage = "finalize_outputs"
        run_id = "b01-anchor-standalone-002"
        payload = {
            "schema_version": 2,
            "request_id": REQUEST_ID,
            "competition": COMPETITION,
            "kernel_stage": "b1_anchor",
            "implementation": "standalone_equivalent",
            "cmi_commit": CMI_COMMIT,
            "source_blobs": SOURCE_BLOBS,
            "python_version": sys.version.split()[0],
            "package_versions": versions,
            "md5_verified": checksum,
            "audit_passed": True,
            "audit_checks": audits,
            "run_id": run_id,
            "submission_rows": int(len(submission)),
            "submission_sha256": sha256(OUTPUT_SUBMISSION),
            "task_unique_counts": unique_counts,
            "minus99_tasks": [],
            "task_directions": diagnostics["directions"],
            "derived_panel_sizes": {"vaccine": len(vaccine), "challenge": len(challenge)},
        }
        OUTPUT_METADATA.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lines = [
            "# CMI-Flu B1 standalone run",
            "",
            f"- Request: `{REQUEST_ID}`",
            f"- CMI source contract: `{CMI_COMMIT}`",
            f"- MD5 files verified: {checksum['verified']}",
            "- Aggregate audits: PASS",
            f"- Submission rows: {len(submission)}",
            f"- Task 1.1 direction: {diagnostics['directions']['Task1.1']}",
        ]
        for name in sorted(audits):
            item = audits[name]
            lines.append(
                f"- {name}: rows={item['rows']}, participants={item['participants']}, studies={item['studies']}"
            )
        OUTPUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            "CMI_FLU_B1_COMPLETE "
            f"request_id={REQUEST_ID} rows={len(submission)} audit_passed=true"
        )
        return 0
    except BaseException as error:
        write_failure(stage, error)
        print(
            "CMI_FLU_B1_FAILED "
            f"request_id={REQUEST_ID} stage={stage} "
            f"exception={type(error).__name__} "
            f"code={getattr(error, 'code', 'unexpected_exception')}",
            file=sys.stderr,
        )
        traceback.print_exc(limit=8)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
