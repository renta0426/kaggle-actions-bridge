"""Strategy v3 V3-06: actual D28 ET incumbent log2-affine calibration.

Exactly one new scale condition is evaluated per task:

* Task2.1: current ``et_subtype_d3_l5`` incumbent;
* Task2.2: current ``et_subtype_d5_l10`` incumbent;
* calibration: ``log2(y_cal) = a + b * log2(y_ET)``, ``b > 0``.

The map is fit only from inner OOF predictions produced inside each outer
subject-purged held-study fold.  V3 freezes three inner study folds to keep the
calibration validation finite while retaining study separation.  The historical
E06c ridge calibration remains comparator context only; no affine/power grid is
reopened and no Public-LB value is consumed.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .aliases import canonicalize_strain
from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns
from .cv import NamedSplit, assert_no_subject_leakage, purged_leave_one_study_out
from .datasets import HAIModelDataset, build_hai_model_dataset
from .evaluation import aggregate_hai_task_predictions, evaluate_hai_spec
from .metrics import safe_spearman
from .models import ModelSpec, fit_final_model
from . import strategy_e06 as _e06
from . import strategy_e06_v2 as _e06v2

EXPERIMENT = "strategy_v3_v06_actual_d28_et_log2_affine_calibration"
SCHEMA_VERSION = 1
TARGET_DAY = 28
TASKS = ("Task2.1", "Task2.2")
MODEL_BY_TASK = {
    "Task2.1": "et_subtype_d3_l5",
    "Task2.2": "et_subtype_d5_l10",
}
CONDITION_BY_TASK = {
    "Task2.1": "task21_et_log_affine",
    "Task2.2": "task22_et_log_affine",
}
INNER_STUDY_FOLDS = 3
CALIBRATION_KIND = "log2_affine"
MAX_FITS = 256
RANDOM_SEED = 20260911


def _json_safe(value: Any) -> Any:
    return _e06._json_safe(value)


def _find_model(config: BaselineConfig, name: str) -> ModelSpec:
    matches = [spec for spec in config.model_specs("hai") if spec.name == name]
    if len(matches) != 1:
        raise DataContractError(f"V3-06 expected exactly one hai/{name}; found {len(matches)}")
    spec = matches[0]
    if spec.family != "extra_trees":
        raise DataContractError(f"V3-06 incumbent {name} is no longer ExtraTrees")
    return spec


def _panel_overlap_studies(frame: pd.DataFrame, panel: Sequence[str]) -> list[str]:
    require_columns(frame, ["study_group", "virus_strain"], table_name="V3-06 overlap frame")
    wanted = set(_e06._canonical_panel(panel))
    strain = frame["virus_strain"].map(canonicalize_strain)
    return sorted(set(frame.loc[strain.isin(wanted), "study_group"].astype(str)))


def _purged_inner_study_kfold(frame: pd.DataFrame, *, panel_strains: Sequence[str], n_splits: int = INNER_STUDY_FOLDS) -> list[NamedSplit]:
    require_columns(frame, ["study_group", "subject_group", "virus_strain"])
    studies = frame["study_group"].astype(str).to_numpy()
    subjects = frame["subject_group"].astype(str).to_numpy()
    eligible = _panel_overlap_studies(frame, panel_strains)
    if len(eligible) < n_splits:
        raise DataContractError(f"V3-06 requires >={n_splits} panel-overlap studies for inner calibration; found {len(eligible)}")
    counts = {study: int(np.sum(studies == study)) for study in eligible}
    rng = np.random.default_rng(RANDOM_SEED)
    order = list(np.asarray(eligible, dtype=object)[rng.permutation(len(eligible))])
    order.sort(key=lambda study: counts[str(study)], reverse=True)
    fold_sizes = np.zeros(n_splits, dtype=int)
    fold_studies: list[list[str]] = [[] for _ in range(n_splits)]
    for raw_study in order:
        study = str(raw_study)
        slot = int(np.argmin(fold_sizes))
        fold_studies[slot].append(study)
        fold_sizes[slot] += counts[study]
    splits: list[NamedSplit] = []
    for fold, held_studies in enumerate(fold_studies):
        validation_mask = np.isin(studies, held_studies)
        held_subjects = np.unique(subjects[validation_mask])
        train_mask = (~validation_mask) & (~np.isin(subjects, held_subjects))
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        if len(train_indices) < 2 or len(validation_indices) < 2:
            raise DataContractError(f"V3-06 inner fold {fold} has an empty/undersized partition")
        purged = tuple(sorted(set(subjects[(~validation_mask) & np.isin(subjects, held_subjects)])))
        split = NamedSplit(name=f"inner_study_fold={fold}", train_indices=train_indices, validation_indices=validation_indices, held_out_group=",".join(sorted(held_studies)), purged_subjects=purged)
        assert_no_subject_leakage(split, subjects)
        splits.append(split)
    return splits


def _eligible_outer_splits(dataset: HAIModelDataset, *, panel_strains: Sequence[str]) -> tuple[list[NamedSplit], list[dict[str, Any]], int]:
    train = dataset.train.reset_index(drop=True)
    candidates = purged_leave_one_study_out(train["study_group"].astype(str), train["subject_group"].astype(str))
    eligible: list[NamedSplit] = []
    skipped: list[dict[str, Any]] = []
    for split in candidates:
        outer_train = train.iloc[split.train_indices].reset_index(drop=True)
        outer_valid = train.iloc[split.validation_indices].reset_index(drop=True)
        valid_overlap = len(_panel_overlap_studies(outer_valid, panel_strains))
        train_overlap_studies = len(_panel_overlap_studies(outer_train, panel_strains))
        if valid_overlap == 0:
            skipped.append({"held_out_group": str(split.held_out_group), "reason": "zero_fixed_panel_overlap", "outer_training_panel_overlap_studies": train_overlap_studies})
            continue
        if train_overlap_studies < INNER_STUDY_FOLDS:
            skipped.append({"held_out_group": str(split.held_out_group), "reason": "fewer_than_three_inner_panel_overlap_studies", "outer_training_panel_overlap_studies": train_overlap_studies})
            continue
        eligible.append(split)
    if len(eligible) < 3:
        raise DataContractError("V3-06 requires at least three evaluable outer studies")
    return eligible, skipped, len(candidates)


def _rank_contract(raw_values: Sequence[float], calibrated_values: Sequence[float]) -> dict[str, Any]:
    raw = np.asarray(raw_values, dtype=float)
    calibrated = np.asarray(calibrated_values, dtype=float)
    if raw.shape != calibrated.shape or not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
        raise DataContractError("V3-06 rank contract shape/finite check failed")
    raw_rank = rankdata(raw, method="average")
    calibrated_rank = rankdata(calibrated, method="average")
    same_ties = np.array_equal(raw[:, None] == raw[None, :], calibrated[:, None] == calibrated[None, :])
    rho = safe_spearman(raw, calibrated)
    return {"same_rank_vector": bool(np.array_equal(raw_rank, calibrated_rank)), "same_tie_equivalence": bool(same_ties), "changed_rank_count": int(np.sum(raw_rank != calibrated_rank)), "spearman": None if rho.status != "ok" else float(rho.value), "spearman_status": rho.status}


def _nested_task(dataset: HAIModelDataset, *, spec: ModelSpec, panel_strains: Sequence[str], task: str) -> tuple[dict[str, Any], pd.DataFrame, int]:
    train = dataset.train.reset_index(drop=True)
    outer_splits, skipped, candidate_count = _eligible_outer_splits(dataset, panel_strains=panel_strains)
    raw_parts: list[pd.DataFrame] = []
    calibrated_parts: list[pd.DataFrame] = []
    parameters: list[dict[str, Any]] = []
    fit_count = 0
    for outer in outer_splits:
        outer_train = train.iloc[outer.train_indices].reset_index(drop=True)
        outer_valid = train.iloc[outer.validation_indices].reset_index(drop=True)
        inner_splits = _purged_inner_study_kfold(outer_train, panel_strains=panel_strains)
        inner_dataset = replace(dataset, train=outer_train)
        inner_eval = evaluate_hai_spec(inner_dataset, spec=spec, splits=inner_splits, panel_strains=panel_strains)
        fit_count += len(inner_splits)
        calibration_train = _e06._panel_frame_from_oof(inner_eval, panel_strains=panel_strains)
        if int(calibration_train["study_group"].astype(str).nunique()) < 2:
            raise DataContractError("V3-06 inner OOF calibration has <2 panel-overlap studies")
        params = _e06v2._fit_calibrator(calibration_train, kind=CALIBRATION_KIND)
        _, target_prediction = fit_final_model(outer_train, outer_valid, target_column=dataset.target_column, spec=spec, excluded_columns=dataset.excluded_columns)
        fit_count += 1
        post_prediction = dataset.target_prediction_to_post_hai(target_prediction, frame=outer_valid)
        raw_panel = _e06._panel_frame(outer_valid, post_prediction, panel_strains=panel_strains, split=str(outer.name))
        calibrated = raw_panel.copy()
        calibrated["prediction"] = _e06._apply_calibrator(calibrated["prediction"].to_numpy(dtype=float), params)
        raw_parts.append(raw_panel)
        calibrated_parts.append(calibrated)
        parameters.append({"outer_fold": str(outer.name), "held_out_group": str(outer.held_out_group), "inner_split_count": len(inner_splits), **dict(params)})
    raw = pd.concat(raw_parts, ignore_index=True)
    calibrated = pd.concat(calibrated_parts, ignore_index=True)
    raw_metrics = _e06._metric_bundle(raw)
    calibrated_metrics = _e06._metric_bundle(calibrated)
    rank = _e06._rank_preservation(raw, calibrated)
    rmse = _e06._rmse_diagnostics(raw_metrics, calibrated_metrics)
    keys = ["split", "study_group", "participant_id", "subject_group"]
    bank = raw[keys + ["target", "prediction", "panel_size"]].rename(columns={"prediction": "raw_et_prediction"}).merge(calibrated[keys + ["prediction"]].rename(columns={"prediction": "calibrated_prediction"}), on=keys, how="inner", validate="one_to_one")
    bank.insert(0, "task", task)
    raw_mean = float(raw_metrics["equal_study_summary"]["rmse_mean"])
    cal_mean = float(calibrated_metrics["equal_study_summary"]["rmse_mean"])
    raw_pooled = float(raw_metrics["pooled"]["rmse"]["value"])
    cal_pooled = float(calibrated_metrics["pooled"]["rmse"]["value"])
    median_reduction = float(rmse["median_study_relative_reduction"])
    role_decision = "retain_scale_candidate_for_v3_11" if cal_mean < raw_mean and cal_pooled < raw_pooled and median_reduction > 0 else "reject_global_log2_affine_map"
    summary = _json_safe({"task": task, "model": spec.to_dict(), "condition": CONDITION_BY_TASK[task], "panel_size_requested": len(_e06._canonical_panel(panel_strains)), "historical_panel_proxy_coverage": _e06._historical_panel_coverage(dataset, panel_strains=panel_strains), "outer_candidate_split_count": candidate_count, "outer_split_count": len(outer_splits), "skipped_outer_split_count": len(skipped), "skipped_outer_folds": skipped, "raw_metrics": raw_metrics, "calibrated_metrics": calibrated_metrics, "rmse_diagnostics": rmse, "rank_preservation": rank, "outer_fold_calibrators": parameters, "role_decision": role_decision})
    return summary, bank, fit_count


def _challenge_task(dataset: HAIModelDataset, *, spec: ModelSpec, panel_strains: Sequence[str], task: str) -> tuple[dict[str, Any], pd.DataFrame, int]:
    train = dataset.train.reset_index(drop=True)
    inner_splits = _purged_inner_study_kfold(train, panel_strains=panel_strains)
    evaluation = evaluate_hai_spec(dataset, spec=spec, splits=inner_splits, panel_strains=panel_strains)
    fit_count = len(inner_splits)
    calibration_train = _e06._panel_frame_from_oof(evaluation, panel_strains=panel_strains)
    params = _e06v2._fit_calibrator(calibration_train, kind=CALIBRATION_KIND)
    _, target_prediction = fit_final_model(dataset.train, dataset.challenge, target_column=dataset.target_column, spec=spec, excluded_columns=dataset.excluded_columns)
    fit_count += 1
    post_prediction = dataset.target_prediction_to_post_hai(target_prediction, frame=dataset.challenge)
    strain_prediction = dataset.challenge[["participant_id", "virus_strain"]].copy()
    strain_prediction["prediction"] = post_prediction
    raw_task = aggregate_hai_task_predictions(strain_prediction, panel_strains=panel_strains, task=task)
    if len(raw_task) != 40:
        raise DataContractError(f"V3-06 {task} expected 40 Challenge donors; found {len(raw_task)}")
    raw_values = raw_task["prediction"].to_numpy(dtype=float)
    calibrated_values = _e06._apply_calibrator(raw_values, params)
    bank = raw_task[["participant_id", "prediction"]].rename(columns={"prediction": "raw_et_prediction"})
    bank["calibrated_prediction"] = calibrated_values
    bank.insert(0, "task", task)
    summary = _json_safe({"full_calibrator": params, "rank_contract": _rank_contract(raw_values, calibrated_values), "donors": len(bank)})
    return summary, bank, fit_count


def run_v3_06(config: BaselineConfig, inputs: Any) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if str(config.section("hai").get("target_representation", "residual")) != "residual":
        raise DataContractError("V3-06 requires residual HAI target representation")
    dataset = build_hai_model_dataset(inputs.tables["public_serology"], inputs.tables["challenge_serology"], inputs.tables["participants"], inputs.tables["investigations"], day=TARGET_DAY, target_representation="residual", challenge_panel_strains=inputs.challenge_strains)
    panels = {"Task2.1": inputs.vaccine_strains, "Task2.2": inputs.challenge_strains}
    task_results: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    challenge_parts: list[pd.DataFrame] = []
    fit_count = 0
    for task in TASKS:
        spec = _find_model(config, MODEL_BY_TASK[task])
        nested, oof, nested_fits = _nested_task(dataset, spec=spec, panel_strains=panels[task], task=task)
        challenge, challenge_bank, challenge_fits = _challenge_task(dataset, spec=spec, panel_strains=panels[task], task=task)
        fit_count += nested_fits + challenge_fits
        if fit_count > MAX_FITS:
            raise DataContractError(f"V3-06 fit budget exceeded:{fit_count}>{MAX_FITS}")
        task_results[task] = {**nested, "challenge": challenge}
        oof_parts.append(oof)
        challenge_parts.append(challenge_bank)
    aggregate = _json_safe({"schema_version": SCHEMA_VERSION, "experiment": EXPERIMENT, "stage": "V3-06", "target_day": TARGET_DAY, "tasks": task_results, "new_candidate_conditions": [CONDITION_BY_TASK[t] for t in TASKS], "new_candidate_condition_count": len(TASKS), "fit_count": fit_count, "fit_limit": MAX_FITS, "calibration_contract": {"kind": CALIBRATION_KIND, "formula": "log2(yhat_cal)=a+b*log2(yhat_ET), b>0", "base_models": dict(MODEL_BY_TASK), "outer_evaluation": "subject_purged_leave_one_study_out", "inner_calibration": "three_subject_purged_study_folds_from_outer_training_only", "inner_study_folds": INNER_STUDY_FOLDS, "calibration_weighting": "equal_study_total_weight", "held_study_outcomes_used_for_calibrator": False, "strict_monotonicity_required": True, "positive_predictions_required": True, "inverse_transform_interpretation": "exp2 of weighted least-squares prediction in log2 target space; without a smearing correction this is geometric/median-like on the original scale, not an arithmetic-mean estimator", "historical_e06c_ridge_is_comparator_only": True, "affine_or_power_grid_reopened": False}, "public_leaderboard_used": False, "competition_submission_attempted": False, "final_submission_selection_attempted": False, "automatic_compute_retries": 0, "private_bank_contains_row_level_values": True, "aggregate_contains_row_level_values": False})
    return aggregate, pd.concat(oof_parts, ignore_index=True), pd.concat(challenge_parts, ignore_index=True)


def write_v3_06_outputs(aggregate: Mapping[str, Any], oof_bank: pd.DataFrame, challenge_bank: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for name, frame in (("v3_v06_oof_bank.csv", oof_bank), ("v3_v06_challenge_bank.csv", challenge_bank)):
        path = out / name
        if path.exists():
            raise DataContractError(f"V3-06 refuses to overwrite:{name}")
        frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        raw = path.read_bytes()
        files.append({"filename": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "rows": int(len(frame)), "private_row_level": True})
    summary_path = out / "v3_v06_summary.json"
    manifest_path = out / "v3_v06_bank_manifest.json"
    summary_path.write_text(json.dumps(_json_safe(aggregate), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "experiment": EXPERIMENT, "files": files, "row_level_contents_must_not_be_publicly_emitted": True, "competition_submission_attempted": False}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest
