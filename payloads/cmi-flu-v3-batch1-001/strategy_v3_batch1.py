"""Strategy v3 batch 1: no-fit teacher audit and fixed Public diagnostics.

The public API in this module never fits a model and never submits to a Kaggle
competition. Private execution may inspect row-level input, but only aggregate
records returned by :func:`run_v3_01_audit` are exportable.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .aliases import canonicalize_flow_population, canonicalize_strain, canonicalize_timepoint
from .contracts import (
    DataContractError,
    PUBLIC_LEADERBOARD_TASKS,
    SUBMISSION_COLUMNS,
    TASK_COLUMNS,
    validate_sample_submission,
    validate_submission,
)
from .cv import purged_leave_one_study_out
from .datasets import build_task_11_dataset, build_task_12_dataset, build_task_13_dataset
from .features.metadata import build_participant_context
from .io import file_digest, resolve_competition_paths, verify_md5_manifest
from .strategy_e04_contract import audit_measurements
from .strategy_e04b import apply_material_plural_ontology
from .strategy_e09a import PUBLIC_RNA_FILES, _scan_rna
from .targets import build_hai_long_target, build_task_11_target, build_task_12_target, build_task_13_target

SCHEMA_VERSION = 1
EXPERIMENT = "strategy_v3_batch1_v3_00_v3_01_v3_03"
EXPECTED_E12C_BYTES = 5926
EXPECTED_E12C_SHA256 = "0f9df53c3aa8c6e4ac693f6a42dbd2633b4b61d1462df798a9bd767c220be3a5"
E12B_BYTES = 5933
E12B_SHA256 = "983aaf097d04477c4ccf7bf817fdf66e552937e69bceaa48cfb260cb84413f1b"
DIAGNOSTIC_TASKS = tuple(PUBLIC_LEADERBOARD_TASKS)
STUDY_ALIASES: Mapping[str, str] = {"2024UGA": "2024UGA", "2024_UGA": "2024UGA"}
FLOW_FIELDS = ("population_definition", "parent_population", "unit", "material")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return sha256_bytes(raw)


def study_alias_hash() -> str:
    return stable_json_sha256(dict(STUDY_ALIASES))


def canonical_study(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    return STUDY_ALIASES.get(text, text)


def _subject_key(frame: pd.DataFrame) -> pd.Series:
    if "subject_group" in frame:
        subject = frame["subject_group"].astype("string")
    elif "subject" in frame:
        subject = frame["subject"].astype("string")
    else:
        subject = pd.Series(pd.NA, index=frame.index, dtype="string")
    pid = frame["participant_id"].astype("string")
    return subject.fillna("participant:" + pid).astype(str)


def _study_key(frame: pd.DataFrame) -> pd.Series:
    if "study_group" in frame:
        study = frame["study_group"].astype("string")
    elif "study_accession" in frame:
        study = frame["study_accession"].astype("string")
    elif "study" in frame:
        study = frame["study"].astype("string")
    else:
        raise DataContractError("study key is unavailable")
    return study.map(canonical_study).astype(str)


def _by_study(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    return {str(k): int(v) for k, v in _study_key(frame).value_counts().sort_index().items()}


def _unique_subjects(frame: pd.DataFrame) -> int:
    return int(_subject_key(frame).nunique()) if len(frame) else 0


def _split_support(frame: pd.DataFrame, task: str) -> dict[str, Any]:
    work = frame.reset_index(drop=True)
    studies, subjects = _study_key(work), _subject_key(work)
    try:
        splits = purged_leave_one_study_out(studies, subjects)
    except DataContractError as error:
        return {"task": task, "status": "data_limited", "reason": str(error), "folds": [], "metric_rows_covered": 0}
    folds, covered = [], set()
    for split in splits:
        held = str(split.held_out_group)
        before = int((studies != held).sum())
        after = int(len(split.train_indices))
        folds.append({
            "split_id": split.name,
            "held_out_study": held,
            "n_train_before_purge": before,
            "n_train_after_purge": after,
            "n_valid_after_purge": int(len(split.validation_indices)),
            "n_rows_purged": before - after,
            "n_subjects_purged": len(split.purged_subjects),
        })
        covered.update(int(i) for i in split.validation_indices)
    return {"task": task, "status": "ok", "folds": folds, "metric_rows_covered": len(covered)}


def _input_checksums(root: Path) -> dict[str, str]:
    names = [
        "participants.tsv", "investigations_260821.tsv", "publicData_cytokine.tsv",
        "publicData_ex_vivo_flow.tsv", "publicData_serology_260821.tsv", "2025LJI_aim.tsv",
        "2025LJI_cytokine.tsv", "2025LJI_ex_vivo_flow.tsv", "2025LJI_serology.tsv",
        "sample_submission_part1.csv", "md5sum", *PUBLIC_RNA_FILES.values(),
        "2025LJI_rnaseq.tsv", "publicData_bulkBCR.tsv",
    ]
    return {name: file_digest(root / name) for name in names if (root / name).is_file()}


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _flow_signature_rows(frame: pd.DataFrame, population: str, timepoint: Any) -> pd.DataFrame:
    required = ["participant_id", "study_accession", "timepoint", "name", "value", *FLOW_FIELDS]
    missing = [c for c in required if c not in frame]
    if missing:
        raise DataContractError(f"flow measurement audit missing columns: {missing}")
    work = frame[required].copy()
    work["_name"] = work["name"].map(canonicalize_flow_population)
    work["_time"] = work["timepoint"].map(canonicalize_timepoint)
    work = work.loc[
        work["_name"].eq(canonicalize_flow_population(population))
        & work["_time"].eq(canonicalize_timepoint(timepoint))
    ].copy()
    work["_value"] = pd.to_numeric(work["value"], errors="coerce")
    work["_study"] = work["study_accession"].map(canonical_study)
    for col in FLOW_FIELDS:
        work[f"_{col}"] = work[col].map(_safe_text)
    work["metadata_complete"] = True
    for col in FLOW_FIELDS:
        work["metadata_complete"] &= work[f"_{col}"].ne("")
    work["signature"] = work.apply(
        lambda r: stable_json_sha256({c: r[f"_{c}"] for c in FLOW_FIELDS})
        if bool(r["metadata_complete"]) else "unknown", axis=1,
    )
    rows = []
    for (pid, study), group in work.groupby(["participant_id", "_study"], sort=True):
        signatures = sorted(set(group["signature"]))
        finite = np.isfinite(group["_value"].to_numpy(float))
        rows.append({
            "participant_id": str(pid), "study_accession": str(study),
            "signature": signatures[0] if len(signatures) == 1 else "conflict",
            "metadata_complete": bool(group["metadata_complete"].all()),
            "finite_nonnegative": bool(finite.all() and group["_value"].ge(0).all()),
        })
    return pd.DataFrame(rows)


def _flow_compatibility(public: pd.DataFrame, challenge: pd.DataFrame, population: str, day: int) -> dict[str, Any]:
    source = _flow_signature_rows(public, population, day)
    reference = _flow_signature_rows(challenge, population, day)
    known = reference.loc[
        reference["metadata_complete"] & reference["finite_nonnegative"]
        & ~reference["signature"].isin(["unknown", "conflict"])
    ]
    signatures = sorted(set(known["signature"]))
    unknown = int((~source["metadata_complete"] | source["signature"].isin(["unknown", "conflict"])).sum())
    if len(signatures) != 1:
        return {"status": "unresolved", "challenge_signature_count": len(signatures), "compatible_participant_years": 0, "unknown_or_conflict_participant_years": unknown}
    sig = signatures[0]
    compatible = source.loc[source["metadata_complete"] & source["finite_nonnegative"] & source["signature"].eq(sig)]
    return {
        "status": "ok", "challenge_signature_count": 1, "challenge_signature_sha256": sig,
        "public_target_participant_years": int(len(source)),
        "compatible_participant_years": int(len(compatible)),
        "compatible_by_study": _by_study(compatible),
        "unknown_or_conflict_participant_years": unknown,
        "unknown_unit_parent_gate_policy": "incompatible_not_matched",
    }


def _task14_audit(root: Path, challenge_aim: pd.DataFrame) -> dict[str, Any]:
    work = challenge_aim.copy()
    stim = work["stimulation"].astype("string").str.strip().str.casefold()
    work = work.loc[stim.str.contains("conserved", na=False) | stim.eq("con")].copy()
    work["_time"] = work["timepoint"].map(canonicalize_timepoint)
    work["_value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.loc[np.isfinite(work["_value"])]
    literal = work.loc[work["_time"].eq("Pre-vacc")].groupby("participant_id")["_value"].mean()
    numeric = pd.to_numeric(work["_time"], errors="coerce")
    alt = work.loc[numeric.isin([-14.0, 0.0])].groupby("participant_id")["_value"].mean()
    common = literal.index.intersection(alt.index)
    diff = literal.loc[common] - alt.loc[common] if len(common) else pd.Series(dtype=float)
    result = {
        "literal_prevacc_subjects": int(literal.size), "minus14_or_0_mean_subjects": int(alt.size),
        "common_subjects": int(len(common)),
        "different_subjects": int((~np.isclose(diff.to_numpy(float), 0, atol=1e-12, rtol=0)).sum()) if len(diff) else 0,
        "mean_absolute_difference": float(np.mean(np.abs(diff))) if len(diff) else None,
        "max_absolute_difference": float(np.max(np.abs(diff))) if len(diff) else None,
    }
    public_aim_path = root / "publicData_aim.tsv"
    if public_aim_path.is_file():
        aim = pd.read_csv(public_aim_path, sep="\t", low_memory=False)
        aim["_time"] = aim["timepoint"].map(canonicalize_timepoint)
        d7 = aim.loc[aim["_time"].eq("7")]
        result.update(public_AIM_file_present=True, public_D7_rows=int(len(d7)), public_D7_participant_years=int(d7["participant_id"].nunique()))
    else:
        result.update(public_AIM_file_present=False, public_D7_rows=0, public_D7_participant_years=0)
    result["direct_public_D7_AIM_teacher_count"] = result["public_D7_participant_years"]
    return result


def _task13_rna_alignment(root: Path, public_flow: pd.DataFrame, challenge_flow: pd.DataFrame) -> dict[str, Any]:
    transformed, ontology = apply_material_plural_ontology({"public_flow": public_flow, "challenge_flow": challenge_flow})
    audit = audit_measurements(transformed["public_flow"], transformed["challenge_flow"])
    eligible = audit.eligibility
    strict = eligible.loc[
        eligible["study"].map(canonical_study).eq("2024UGA")
        & eligible["within_gate_compatible"].fillna(False).astype(bool)
        & eligible["challenge_gate_compatible"].fillna(False).astype(bool)
        & ~eligible["proxy_allowed"].fillna(False).astype(bool)
    ]
    teacher = set(strict["participant_id"].astype(str))
    scan = _scan_rna(root / PUBLIC_RNA_FILES["2024_UGA"], expected_study="2024_UGA", value_column="tpm")
    day0 = set(scan.day0_participants["2024_UGA"])
    challenge_scan = _scan_rna(root / "2025LJI_rnaseq.tsv", expected_study="2025LJI", value_column="tpm")
    challenge_material = set().union(*challenge_scan.day0_materials_by_participant["2025LJI"].values())
    challenge_material.discard("not_reported")
    materials = scan.day0_materials_by_participant["2024_UGA"]
    compatible = {pid for pid in day0 if set(materials.get(pid, set())) & challenge_material}
    return {
        "raw_study_label": "2024_UGA", "teacher_study_label": "2024UGA", "canonical_study_label": "2024UGA",
        "alias_map_sha256": study_alias_hash(), "strict_teacher_subjects": len(teacher),
        "rna_day0_subjects": len(day0), "rna_material_compatible_day0_subjects": len(compatible),
        "strict_teacher_with_day0_rna": len(teacher & day0),
        "strict_teacher_with_measurement_compatible_day0_rna": len(teacher & compatible),
        "ontology_version": ontology.get("ontology_version"), "legacy_zero_join_is_valid": False,
    }


def _truthy(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.casefold()
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.eq(1) | text.isin({"true", "yes", "y", "vaccine", "in_vaccine"})


def _hai_panels(public: pd.DataFrame, challenge: pd.DataFrame) -> dict[str, Any]:
    c = challenge.loc[challenge["assay"].fillna("").astype(str).str.strip().str.casefold().eq("hai")].copy()
    c["_time"] = c["timepoint"].map(canonicalize_timepoint)
    c["_strain"] = c["virus_strain"].map(canonicalize_strain)
    time = pd.to_numeric(c["_time"], errors="coerce")
    base = c.loc[c["_time"].eq("Pre-vacc") | time.le(0).fillna(False)]
    challenge_panel = tuple(sorted(set(base["_strain"].dropna().astype(str))))
    vaccine_panel: tuple[str, ...] = ()
    if "virus_in_vaccine" in base:
        vaccine_panel = tuple(sorted(set(base.loc[_truthy(base["virus_in_vaccine"]), "_strain"].dropna().astype(str))))
    p = public.loc[public["assay"].fillna("").astype(str).str.strip().str.casefold().eq("hai")].copy()
    p["_time"] = p["timepoint"].map(canonicalize_timepoint)
    p["_strain"] = p["virus_strain"].map(canonicalize_strain)
    if not vaccine_panel and "virus_in_vaccine" in p:
        vaccine_panel = tuple(sorted(set(p.loc[_truthy(p["virus_in_vaccine"]), "_strain"].dropna().astype(str)) & set(challenge_panel)))
    any_overlap = set(p["_strain"].dropna().astype(str)) & set(challenge_panel)
    d28 = set(p.loc[p["_time"].eq("28"), "_strain"].dropna().astype(str)) & set(challenge_panel)
    d365 = set(p.loc[p["_time"].eq("365"), "_strain"].dropna().astype(str)) & set(challenge_panel)
    return {"challenge_panel": challenge_panel, "vaccine_panel": vaccine_panel, "any": tuple(sorted(any_overlap)), "D28": tuple(sorted(d28)), "D365": tuple(sorted(d365))}


def _hai_proxy(target: pd.DataFrame, panel: Sequence[str]) -> pd.DataFrame:
    out = target.copy()
    out["virus_strain"] = out["virus_strain"].map(canonicalize_strain)
    return out.loc[out["virus_strain"].isin(set(panel))].copy()


def _complete_panel_candidates(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    work = frame.copy(); work["_study"] = _study_key(work); work["_strain"] = work["virus_strain"].map(canonicalize_strain)
    rows = []
    for study, group in work.groupby("_study", sort=True):
        sets = group.groupby("participant_id")["_strain"].apply(set)
        if sets.empty:
            continue
        counts = Counter(tuple(sorted(x)) for x in sets)
        panel, support = sorted(counts.items(), key=lambda kv: (-len(kv[0]), -kv[1], kv[0]))[0]
        rows.append({"study": str(study), "local_panel_size": len(panel), "complete_subjects": int(support), "panel_sha256": sha256_bytes("\n".join(panel).encode())})
    return sorted(rows, key=lambda x: (-x["local_panel_size"], -x["complete_subjects"], x["study"]))[:3]


def _task_record(task: str, raw_rows: int, target: pd.DataFrame, train: pd.DataFrame, compatibility: Mapping[str, Any] | None = None) -> dict[str, Any]:
    split = _split_support(train, task)
    return {
        "task": task, "raw_assay_rows": int(raw_rows), "target_rows": int(len(target)),
        "target_bearing_participant_years": int(target["participant_id"].nunique()),
        "unique_subjects": _unique_subjects(target), "target_by_study": _by_study(target),
        "baseline_join_rows": int(len(train)), "model_train_rows": int(len(train)),
        "model_train_unique_subjects": _unique_subjects(train),
        "raw_scale_compatible": None if compatibility is None else int(compatibility.get("compatible_participant_years", 0)),
        "measurement_compatibility": None if compatibility is None else dict(compatibility),
        "metric_subjects": int(split.get("metric_rows_covered", 0)), "split_support": split,
    }


def _hai_record(task: str, raw_rows: int, full: pd.DataFrame, proxy: pd.DataFrame) -> dict[str, Any]:
    split = _split_support(full, task)
    counts = proxy.groupby("participant_id")["virus_strain"].nunique() if len(proxy) else pd.Series(dtype=int)
    return {
        "task": task, "raw_assay_rows": int(raw_rows), "raw_strain_long_learning_rows": int(len(full)),
        "learning_participant_years": int(full["participant_id"].nunique()), "learning_unique_subjects": _unique_subjects(full),
        "learning_studies": int(_study_key(full).nunique()), "model_train_rows": int(len(full)),
        "evaluation_proxy_rows": int(len(proxy)), "evaluation_proxy_participant_years": int(proxy["participant_id"].nunique()),
        "evaluation_proxy_unique_subjects": _unique_subjects(proxy), "evaluation_proxy_studies": int(_study_key(proxy).nunique()) if len(proxy) else 0,
        "evaluation_proxy_by_study": _by_study(proxy), "metric_subjects": int(proxy["participant_id"].nunique()),
        "proxy_strain_count_min": int(counts.min()) if len(counts) else 0, "proxy_strain_count_max": int(counts.max()) if len(counts) else 0,
        "split_support": split,
    }


def _auxiliary(root: Path, cytokine: pd.DataFrame, t11: pd.DataFrame, t28: pd.DataFrame, t365: pd.DataFrame) -> dict[str, Any]:
    cyt = cytokine.copy(); cyt["_time"] = cyt["timepoint"].map(canonicalize_timepoint)
    d1 = cyt.loc[cyt["_time"].eq("1")].copy()
    names = d1["analyte"].astype("string").str.strip().str.casefold()
    aux = d1.loc[~names.isin({"ip10", "ip-10", "cxcl10"})]
    t11_ids, aux_ids = set(t11["participant_id"].astype(str)), set(aux["participant_id"].astype(str))
    a, b = t28.copy(), t365.copy(); a["_subject"] = _subject_key(a); b["_subject"] = _subject_key(b)
    a["_strain"] = a["virus_strain"].map(canonicalize_strain); b["_strain"] = b["virus_strain"].map(canonicalize_strain)
    paired = a[["_subject", "_strain"]].drop_duplicates().merge(b[["_subject", "_strain"]].drop_duplicates(), on=["_subject", "_strain"], how="inner")
    rna = {}
    for study, name in PUBLIC_RNA_FILES.items():
        scan = _scan_rna(root / name, expected_study=study, value_column="tpm")
        rna[canonical_study(study)] = len(scan.day0_participants[study])
    return {
        "task11_teacher_with_D1_auxiliary_cytokine": len(t11_ids & aux_ids),
        "task11_teacher_without_D1_auxiliary_cytokine": len(t11_ids - aux_ids),
        "D1_auxiliary_cytokine_participant_years": len(aux_ids), "D1_auxiliary_cytokine_analytes": int(aux["analyte"].nunique()),
        "D28_D365_same_subject_same_strain_pairs": int(len(paired)), "D28_D365_same_subjects": int(paired["_subject"].nunique()) if len(paired) else 0,
        "baseline_RNA_by_study_counts": dict(sorted(rna.items())),
    }


def run_v3_01_audit(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir).expanduser().resolve(); paths = resolve_competition_paths(root); md5 = verify_md5_manifest(root)
    participants = pd.read_csv(paths["participants"], sep="\t", low_memory=False)
    investigations = pd.read_csv(paths["investigations"], sep="\t", low_memory=False)
    pc = pd.read_csv(paths["public_cytokine"], sep="\t", low_memory=False)
    pf = pd.read_csv(paths["public_flow"], sep="\t", low_memory=False)
    ps = pd.read_csv(paths["public_serology"], sep="\t", low_memory=False)
    cc = pd.read_csv(paths["challenge_cytokine"], sep="\t", low_memory=False)
    cf = pd.read_csv(paths["challenge_flow"], sep="\t", low_memory=False)
    cs = pd.read_csv(paths["challenge_serology"], sep="\t", low_memory=False)
    ca = pd.read_csv(paths["challenge_aim"], sep="\t", low_memory=False)
    context = build_participant_context(participants, investigations)[["participant_id", "subject_group", "study_group"]]
    t11 = build_task_11_target(pc).merge(context, on="participant_id", how="left", validate="one_to_one")
    t12 = build_task_12_target(pf).merge(context, on="participant_id", how="left", validate="one_to_one")
    t13 = build_task_13_target(pf).merge(context, on="participant_id", how="left", validate="one_to_one")
    d11 = build_task_11_dataset(pc, cc, participants, investigations)
    d12 = build_task_12_dataset(pf, cf, participants, investigations, mode="strict")
    d13 = build_task_13_dataset(pf, cf, participants, investigations, mode="strict")
    m12 = _flow_compatibility(pf, cf, "Classical_monocytes", 1)
    transformed, _ = apply_material_plural_ontology({"public_flow": pf, "challenge_flow": cf})
    asc = audit_measurements(transformed["public_flow"], transformed["challenge_flow"])
    strict13 = asc.eligibility.loc[
        asc.eligibility["study"].map(canonical_study).eq("2024UGA")
        & asc.eligibility["within_gate_compatible"].fillna(False).astype(bool)
        & asc.eligibility["challenge_gate_compatible"].fillna(False).astype(bool)
        & ~asc.eligibility["proxy_allowed"].fillna(False).astype(bool)
    ]
    m13 = {
        "status": "ok", "compatible_participant_years": int(len(strict13)), "compatible_by_study": _by_study(strict13),
        "measurement_contract_version": asc.aggregate.get("measurement_contract_version"),
        "unknown_or_conflict_participant_years": int(asc.aggregate.get("day7_conflict_keys", 0)),
        "unknown_unit_parent_gate_policy": "incompatible_not_matched",
    }
    panels = _hai_panels(ps, cs); t28 = build_hai_long_target(ps, day=28); t365 = build_hai_long_target(ps, day=365)
    p21, p22, p23 = _hai_proxy(t28, panels["vaccine_panel"]), _hai_proxy(t28, panels["challenge_panel"]), _hai_proxy(t365, panels["challenge_panel"])
    ledger = {
        "Task1.1": _task_record("Task1.1", len(pc), t11, d11.train),
        "Task1.2": _task_record("Task1.2", len(pf), t12, d12.train, m12),
        "Task1.3": _task_record("Task1.3", len(pf), t13, d13.train, m13),
        "Task1.4": {"task": "Task1.4", "raw_assay_rows": int(len(ca)), "target_bearing_participant_years": 0, "unique_subjects": 0, "baseline_join_rows": 0, "raw_scale_compatible": 0, "model_train_rows": 0, "metric_subjects": 0, "split_support": {"task": "Task1.4", "status": "data_limited", "reason": "public D7 AIM teacher checked separately", "folds": [], "metric_rows_covered": 0}},
        "Task2.1": _hai_record("Task2.1", len(ps), t28, p21),
        "Task2.2": _hai_record("Task2.2", len(ps), t28, p22),
        "Task2.3": _hai_record("Task2.3", len(ps), t365, p23),
    }
    pseudo40 = {}
    for task, target in (("Task1.1", t11), ("Task1.2", t12), ("Task1.3", t13)):
        f = target.copy(); f["_subject"] = _subject_key(f); f["_study"] = _study_key(f)
        pseudo40[task] = [{"study": str(s), "unique_subjects": int(n), "supports_nonreplacement_40_to_12_28": bool(n >= 40)} for s, n in f.groupby("_study")["_subject"].nunique().sort_index().items()]
    task14 = _task14_audit(root, ca)
    ledger["Task1.4"]["target_bearing_participant_years"] = int(task14["direct_public_D7_AIM_teacher_count"])
    hashes = _input_checksums(root); dataset_version = stable_json_sha256({"input_sha256": hashes, "alias_map_sha256": study_alias_hash()})
    d365_runtime = len(panels["D365"]); profile = len(panels["any"])
    return {
        "schema_version": SCHEMA_VERSION, "experiment": EXPERIMENT, "audit_type": "no_fit_teacher_measurement_split_support",
        "model_fit_count": 0, "competition_submission_attempted": False, "dataset_version": dataset_version,
        "alias_map_sha256": study_alias_hash(), "input_sha256": hashes, "md5_manifest_verified_count": len(md5.verified),
        "teacher_ledger": ledger,
        "measurement_contracts": {"Task1.2": m12, "Task1.3": m13, "unknown_unit_parent_gate_policy": "incompatible_not_matched"},
        "split_support": {t: r["split_support"] for t, r in ledger.items()},
        "auxiliary_label_coverage": _auxiliary(root, pc, t11, t28, t365),
        "source_alignment_audit": {
            "Task1.3_RNA": _task13_rna_alignment(root, pf, cf), "Task1.4_baseline_definition": task14,
            "HAI_panels": {"challenge_panel_size": len(panels["challenge_panel"]), "vaccine_panel_size": len(panels["vaccine_panel"]), "exact_historical_any_timepoint_count": profile, "exact_historical_D28_count": len(panels["D28"]), "exact_historical_D365_count": d365_runtime},
            "Task2.3_profile_vs_runtime": {"profile_exact_historical_any_timepoint_strains": profile, "runtime_D365_exact_strains": d365_runtime, "difference": profile - d365_runtime, "attributed_to_data_version": "unresolved", "attributed_to_alias": "unresolved", "attributed_to_timepoint_availability": bool(profile != d365_runtime), "baseline_pairing_effect": d365_runtime - int(p23["virus_strain"].nunique()), "note": "availability-only; target values did not select a panel"},
            "pseudo40_support": pseudo40,
            "complete_local_panel_candidates": {"Task2.1": _complete_panel_candidates(p21), "Task2.2": _complete_panel_candidates(p22), "Task2.3": _complete_panel_candidates(p23)},
        },
        "reference_expectations_not_mutated_to_match_runtime": {
            "Task1.1_target_by_study": {"SDY180": 34, "SDY515": 16, "SDY519": 17, "SDY56": 60},
            "Task1.2_target_by_study": {"SDY296": 36, "SDY301": 40, "SDY416": 5},
            "Task1.3_target_participant_years": 29, "Task1.3_baseline_join_rows": 23,
            "Task2.1_proxy_rows": 1141, "Task2.1_proxy_participant_years": 903,
            "Task2.2_proxy_rows": 10617, "Task2.2_proxy_participant_years": 2689,
            "Task2.3_proxy_rows": 4878, "Task2.3_proxy_participant_years": 914,
            "mismatch_policy": "record_cause_do_not_change_expected_value_to_force_PASS",
        },
        "privacy": {"contains_individual_ids": False, "contains_teacher_values": False, "contains_predictions": False, "aggregate_only": True},
    }


def validate_e12c_source_csv(path: str | Path, sample_path: str | Path) -> pd.DataFrame:
    path = Path(path); raw = path.read_bytes(); digest = sha256_bytes(raw)
    if len(raw) == E12B_BYTES and digest == E12B_SHA256:
        raise DataContractError("E12b CSV rejected: V3-03 requires exact E12c-v2 source identity B")
    if len(raw) != EXPECTED_E12C_BYTES or digest != EXPECTED_E12C_SHA256:
        raise DataContractError(f"E12c-v2 source identity mismatch: bytes={len(raw)}, sha256={digest}")
    source = pd.read_csv(path, float_precision="round_trip"); sample = pd.read_csv(sample_path, float_precision="round_trip")
    validate_sample_submission(sample); validate_submission(source, sample)
    if tuple(source.columns) != SUBMISSION_COLUMNS:
        raise DataContractError("E12c-v2 source columns/order mismatch")
    return source


def _csv_tokens(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or tuple(rows[0]) != SUBMISSION_COLUMNS or len(rows[1:]) != 40 or any(len(r) != len(rows[0]) for r in rows[1:]):
        raise DataContractError("source CSV token shape/header mismatch")
    return rows[0], rows[1:]


def generate_singleton_diagnostics(source_path: str | Path, sample_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source_path, sample_path = Path(source_path), Path(sample_path)
    source = validate_e12c_source_csv(source_path, sample_path); sample = pd.read_csv(sample_path, float_precision="round_trip")
    header, body = _csv_tokens(source_path); indexes = {t: header.index(t) for t in TASK_COLUMNS}
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); staged = []
    for task in DIAGNOSTIC_TASKS:
        path = output / f"v3_public_singleton_{task.replace('.', '_')}.csv"; target_i = indexes[task]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n"); writer.writerow(header)
            for row in body:
                new = list(row)
                for other in TASK_COLUMNS:
                    if other != task: new[indexes[other]] = "-99"
                if new[target_i] != row[target_i]: raise AssertionError("target numeric token mutated")
                writer.writerow(new)
        staged.append((task, path))
    files = []
    for task, path in staged:
        frame = pd.read_csv(path, float_precision="round_trip")
        report = validate_submission(frame, sample, allowed_minus99_tasks=[t for t in TASK_COLUMNS if t != task], require_nonconstant_public_tasks=True)
        if not np.array_equal(frame[task].to_numpy(float), source[task].to_numpy(float)):
            raise DataContractError(f"{task} float round-trip changed target values")
        for other in TASK_COLUMNS:
            if other != task and not bool((frame[other] == -99).all()): raise DataContractError(f"{task}: {other} is not all -99")
        files.append({"task": task, "filename": path.name, "bytes": path.stat().st_size, "sha256": file_digest(path), "source_csv_sha256": EXPECTED_E12C_SHA256, "rows": report.rows, "target_unique_count": report.task_unique_counts[task]})
    return {"schema_version": 1, "diagnostic_not_final": True, "source_identity": {"target": "renta0426/cmi-flu-e12c-manual-submission-20260911-002", "version": 1, "bytes": EXPECTED_E12C_BYTES, "sha256": EXPECTED_E12C_SHA256}, "files": files, "all_generated_before_scoring": True, "competition_submission_attempted": False}


def singleton_score_interval(display_value: float, decimals: int = 3) -> tuple[float, float]:
    half = 0.5 * 10 ** (-decimals); return (6 * (display_value - half), 6 * (display_value + half))


def summarize_singleton_public_scores(displayed: Mapping[str, float], *, macro_display: float = 0.205, decimals: int = 3) -> dict[str, Any]:
    missing = [t for t in DIAGNOSTIC_TASKS if t not in displayed]
    if missing: raise DataContractError(f"singleton scores missing tasks: {missing}")
    tasks = {}
    for task in DIAGNOSTIC_TASKS:
        value = float(displayed[task]); low, high = singleton_score_interval(value, decimals)
        tasks[task] = {"singleton_display": value, "approx_task_public_spearman": 6 * value, "display_rounding_interval": [low, high], "interval_is_statistical_confidence_interval": False, "measurement_type": "direct_singleton_display_transformation"}
    total = sum(float(displayed[t]) for t in DIAGNOSTIC_TASKS); half = 0.5 * 10 ** (-decimals)
    macro_interval, sum_interval = [macro_display - half, macro_display + half], [total - 6 * half, total + 6 * half]
    return {"schema_version": 1, "macro_display": macro_display, "macro_identity": "E12c-v2 user_report_until_independently_retrieved", "tasks": tasks, "sum_singleton_displays": total, "sum_rounding_interval": sum_interval, "macro_rounding_interval": macro_interval, "sum_consistent_with_macro_display": not (sum_interval[1] < macro_interval[0] or sum_interval[0] > macro_interval[1]), "task13_old_pls_difference_estimate": {"status": "indirect_only", "do_not_label_as_direct_absolute_score": True}, "Task2.3_public_value_inferred": False, "RMSE_inferred": False}


def write_aggregate_jsons(result: Mapping[str, Any], output_dir: str | Path) -> list[str]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    names = ["teacher_ledger", "measurement_contracts", "split_support", "auxiliary_label_coverage", "source_alignment_audit"]
    for name in names:
        (output / f"{name}.json").write_text(json.dumps(result[name], indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return [f"{name}.json" for name in names]
