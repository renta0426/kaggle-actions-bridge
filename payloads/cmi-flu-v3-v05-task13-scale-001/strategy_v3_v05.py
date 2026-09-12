"""Strategy v3 V3-05: Task1.3 absolute-value heads on strict compatible teachers.

Task1.2 is intentionally absent: V3-01 found zero strict raw-unit/parent
compatible supervised baseline support.  Task1.3 is evaluated only on the
predeclared 23-subject, single-domain strict ASC cohort after the outcome-
independent E04b PBMCs->PBMC ontology repair.

New conditions are exactly S1-S4 from the frozen Strategy-v3 plan.  No HPO,
Public-LB selection, Competition submission, unit conversion, or synthetic
teacher construction occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

from .contracts import DataContractError, require_columns, require_finite
from .cv import NamedSplit, repeated_group_kfold
from .datasets import build_task_13_dataset
from .features.metadata import build_participant_context
from .metrics import evaluate_predictions, root_mean_squared_error, safe_spearman
from .models import evaluate_model_spec, fit_final_model
from .runner import InputBundle
from .strategy_e01 import _find_spec
from .strategy_e04_contract import ASC, audit_measurements
from .strategy_e04b import (
    REAL_EXPECTED_CHALLENGE_COMPATIBLE_2025LJI,
    REAL_EXPECTED_SUPERVISED_STRICT_SOURCE,
    apply_material_plural_ontology,
)

EXPERIMENT = "strategy_v3_v05_task13_absolute_scale_heads"
SCHEMA_VERSION = 1
RANDOM_SEED = 20260911
TASK = "Task1.3"
SOURCE_STUDY = "2024UGA"
CHALLENGE_STUDY = "2025LJI"
BASELINE_COLUMN = "strict_ASC_raw_baseline"
TARGET_COLUMN = "strict_ASC_D7_target"
NEW_CONDITIONS = ("S1", "S2", "S3", "S4")
OUTER_SPLITS = 5
OUTER_REPEATS = 3
SMALL_SUBSET_N = 12
SMALL_SUBSET_REPETITIONS = 200
RIDGE_ALPHA = 10.0
MAX_FITS = 96


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class _FitCounter:
    count: int = 0

    def add(self, n: int, *, label: str) -> None:
        if type(n) is not int or n < 0:
            raise DataContractError(f"V3-05 invalid fit count:{label}")
        self.count += n
        if self.count > MAX_FITS:
            raise DataContractError(
                f"V3-05 fit budget exceeded:{self.count}>{MAX_FITS}:{label}"
            )


def _sha256_frame(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    require_columns(frame, list(columns), table_name="V3-05 fingerprint frame")
    raw = (
        frame[list(columns)]
        .sort_values(list(columns[:-1]), kind="mergesort")
        .to_csv(index=False, float_format="%.17g", lineterminator="\n")
        .encode()
    )
    return hashlib.sha256(raw).hexdigest()


def _empirical_score(reference: Sequence[float], values: Sequence[float]) -> np.ndarray:
    """Training-only empirical percentile score with deterministic tie handling."""
    ref = np.asarray(reference, dtype=float)
    val = np.asarray(values, dtype=float)
    require_finite(ref, name="V3-05 empirical reference")
    require_finite(val, name="V3-05 empirical values")
    if ref.ndim != 1 or val.ndim != 1 or not len(ref):
        raise DataContractError("V3-05 empirical score requires 1-D nonempty reference")
    result = np.empty(len(val), dtype=float)
    for i, value in enumerate(val):
        less = int(np.sum(ref < value))
        equal = int(np.sum(ref == value))
        result[i] = (less + 0.5 * equal) / len(ref)
    return result


def _leave_one_out_rank_score(values: Sequence[float]) -> np.ndarray:
    """OOF score for a no-fit monotone rank expert; each row excludes itself."""
    x = np.asarray(values, dtype=float)
    require_finite(x, name="V3-05 LOO rank input")
    if len(x) < 3:
        raise DataContractError("V3-05 S3 requires at least three training subjects")
    out = np.empty(len(x), dtype=float)
    for i in range(len(x)):
        ref = np.delete(x, i)
        out[i] = _empirical_score(ref, [x[i]])[0]
    return out


def _rank_ties(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    require_finite(x, name="V3-05 rank-tie input")
    return rankdata(x, method="average")


def _rank_order_contract(reference: Sequence[float], candidate: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(reference, dtype=float)
    b = np.asarray(candidate, dtype=float)
    require_finite(a, name="V3-05 rank reference")
    require_finite(b, name="V3-05 rank candidate")
    if a.shape != b.shape:
        raise DataContractError("V3-05 rank contract shape mismatch")
    ra, rb = _rank_ties(a), _rank_ties(b)
    same_ties = bool(
        np.array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :])
    )
    rho = safe_spearman(a, b)
    return {
        "same_rank_vector": bool(np.array_equal(ra, rb)),
        "same_tie_equivalence": same_ties,
        "spearman": None if rho.status != "ok" else float(rho.value),
        "spearman_status": rho.status,
        "changed_rank_count": int(np.sum(ra != rb)),
    }


def _task13_frames(inputs: InputBundle) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Any]:
    """Build the exact strict-compatible 2024UGA teacher and 2025LJI baseline frames."""
    tables, ontology = apply_material_plural_ontology(dict(inputs.tables))
    audit = audit_measurements(tables["public_flow"], tables["challenge_flow"])

    eligible = audit.eligibility.copy()
    require_columns(
        eligible,
        ["participant_id", "study", "name", "within_gate_compatible"],
        table_name="V3-05 Task1.3 eligibility",
    )
    source_ids = (
        eligible.loc[
            eligible["study"].eq(SOURCE_STUDY)
            & eligible["name"].eq(ASC)
            & eligible["within_gate_compatible"].eq(True),
            ["participant_id"],
        ]
        .drop_duplicates()
        .copy()
    )
    if len(source_ids) != REAL_EXPECTED_SUPERVISED_STRICT_SOURCE:
        raise DataContractError(
            f"V3-05 expected {REAL_EXPECTED_SUPERVISED_STRICT_SOURCE} strict Task1.3 teachers; "
            f"found {len(source_ids)}"
        )

    baseline = audit.baseline.copy()
    day7 = audit.day7.copy()
    require_columns(
        baseline, ["participant_id", "study", "name", "valid", "value", "signature"],
        table_name="V3-05 baseline audit",
    )
    require_columns(
        day7, ["participant_id", "study", "name", "valid", "value", "signature"],
        table_name="V3-05 D7 audit",
    )

    source_baseline = baseline.loc[
        baseline["study"].eq(SOURCE_STUDY)
        & baseline["name"].eq(ASC)
        & baseline["valid"].eq(True),
        ["participant_id", "value", "signature"],
    ].rename(columns={"value": BASELINE_COLUMN, "signature": "baseline_signature"})
    source_target = day7.loc[
        day7["study"].eq(SOURCE_STUDY)
        & day7["name"].eq(ASC)
        & day7["valid"].eq(True),
        ["participant_id", "value", "signature"],
    ].rename(columns={"value": TARGET_COLUMN, "signature": "target_signature"})
    train = (
        source_ids.merge(source_baseline, on="participant_id", how="inner", validate="one_to_one")
        .merge(source_target, on="participant_id", how="inner", validate="one_to_one")
    )
    if len(train) != REAL_EXPECTED_SUPERVISED_STRICT_SOURCE:
        raise DataContractError("V3-05 strict Task1.3 source merge lost teachers")
    if not train["baseline_signature"].eq(train["target_signature"]).all():
        raise DataContractError("V3-05 strict Task1.3 baseline/D7 signatures differ")

    challenge = baseline.loc[
        baseline["study"].eq(CHALLENGE_STUDY)
        & baseline["name"].eq(ASC)
        & baseline["valid"].eq(True),
        ["participant_id", "value", "signature"],
    ].rename(columns={"value": BASELINE_COLUMN, "signature": "baseline_signature"})
    if len(challenge) != REAL_EXPECTED_CHALLENGE_COMPATIBLE_2025LJI:
        raise DataContractError(
            f"V3-05 expected {REAL_EXPECTED_CHALLENGE_COMPATIBLE_2025LJI} Challenge ASC baselines; "
            f"found {len(challenge)}"
        )

    context = build_participant_context(tables["participants"], tables["investigations"])
    context_cols = ["participant_id", "subject_group", "study_group", "age"]
    train = train.merge(context[context_cols], on="participant_id", how="left", validate="one_to_one")
    challenge = challenge.merge(
        context[context_cols], on="participant_id", how="left", validate="one_to_one"
    )
    if train[["subject_group", "study_group"]].isna().any().any():
        raise DataContractError("V3-05 source context is incomplete")
    if challenge[["subject_group", "study_group"]].isna().any().any():
        raise DataContractError("V3-05 Challenge context is incomplete")
    if set(train["study_group"].astype(str)) != {SOURCE_STUDY}:
        raise DataContractError("V3-05 source cohort is not the frozen single 2024UGA domain")
    if set(challenge["study_group"].astype(str)) != {CHALLENGE_STUDY}:
        raise DataContractError("V3-05 Challenge cohort identity changed")

    for column in (BASELINE_COLUMN, TARGET_COLUMN):
        if column in train:
            train[column] = pd.to_numeric(train[column], errors="coerce")
    challenge[BASELINE_COLUMN] = pd.to_numeric(challenge[BASELINE_COLUMN], errors="coerce")
    require_finite(
        train[[BASELINE_COLUMN, TARGET_COLUMN]].to_numpy(dtype=float),
        name="V3-05 strict train values",
    )
    require_finite(
        challenge[[BASELINE_COLUMN]].to_numpy(dtype=float),
        name="V3-05 Challenge baseline values",
    )
    if (train[[BASELINE_COLUMN, TARGET_COLUMN]] < 0).any().any():
        raise DataContractError("V3-05 strict raw flow values must be nonnegative")
    if (challenge[BASELINE_COLUMN] < 0).any():
        raise DataContractError("V3-05 Challenge raw flow baseline must be nonnegative")

    strict_dataset = build_task_13_dataset(
        tables["public_flow"],
        tables["challenge_flow"],
        tables["participants"],
        tables["investigations"],
        mode="strict",
        include_sdy272_asc_proxy=False,
    )
    train_ids = train["participant_id"].astype(str).tolist()
    challenge_ids = challenge["participant_id"].astype(str).tolist()
    if set(strict_dataset.train["participant_id"].astype(str)) != set(train_ids):
        raise DataContractError("V3-05 strict B2.1 training cohort differs from measurement-compatible teachers")
    if set(strict_dataset.challenge["participant_id"].astype(str)) != set(challenge_ids):
        raise DataContractError("V3-05 strict B2.1 Challenge cohort differs from measurement-compatible ASC baselines")
    strict_dataset.train = (
        strict_dataset.train.set_index(strict_dataset.train["participant_id"].astype(str))
        .loc[train_ids]
        .reset_index(drop=True)
    )
    strict_dataset.challenge = (
        strict_dataset.challenge.set_index(strict_dataset.challenge["participant_id"].astype(str))
        .loc[challenge_ids]
        .reset_index(drop=True)
    )
    legacy_target = pd.to_numeric(
        strict_dataset.train[strict_dataset.target_column], errors="coerce"
    ).to_numpy(dtype=float)
    audited_target = train[TARGET_COLUMN].to_numpy(dtype=float)
    target_difference = legacy_target - audited_target
    legacy_target_match = bool(
        np.isfinite(legacy_target).all()
        and np.allclose(legacy_target, audited_target, rtol=0.0, atol=1e-12)
    )
    legacy_target_max_abs_difference = (
        float(np.max(np.abs(target_difference))) if len(target_difference) else 0.0
    )
    strict_dataset.train[strict_dataset.target_column] = audited_target

    aggregate_audit = {
        "ontology_version": ontology.get("ontology_version"),
        "ontology_changed_rows": ontology.get("changed_rows"),
        "source_subjects": int(len(train)),
        "source_unique_subjects": int(train["subject_group"].astype(str).nunique()),
        "source_studies": int(train["study_group"].astype(str).nunique()),
        "challenge_subjects": int(len(challenge)),
        "challenge_unique_subjects": int(challenge["subject_group"].astype(str).nunique()),
        "source_baseline_zero_count": int(train[BASELINE_COLUMN].eq(0).sum()),
        "source_target_zero_count": int(train[TARGET_COLUMN].eq(0).sum()),
        "challenge_baseline_zero_count": int(challenge[BASELINE_COLUMN].eq(0).sum()),
        "source_age_missing": int(pd.to_numeric(train["age"], errors="coerce").isna().sum()),
        "challenge_age_missing": int(
            pd.to_numeric(challenge["age"], errors="coerce").isna().sum()
        ),
        "single_domain_conditional": True,
        "raw_unit_conversion_performed": False,
        "synthetic_teacher_rows": 0,
        "legacy_b21_builder_target_exact_match": legacy_target_match,
        "legacy_b21_builder_target_max_abs_difference": legacy_target_max_abs_difference,
        "b21_reference_target_rebound_to_audited_compatible_teacher": True,
    }
    return train.reset_index(drop=True), challenge.reset_index(drop=True), aggregate_audit, strict_dataset


def _candidate_metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    metrics = evaluate_predictions(frame[TARGET_COLUMN], frame[column])
    return _json_safe(metrics)


def _aggregate_repeats(raw: pd.DataFrame, *, prediction_columns: Sequence[str]) -> pd.DataFrame:
    require_columns(
        raw,
        ["row_index", "repeat", TARGET_COLUMN, *prediction_columns],
        table_name="V3-05 raw OOF",
    )
    expected_repeats = set(range(OUTER_REPEATS))
    if set(raw["repeat"].astype(int).unique()) != expected_repeats:
        raise DataContractError("V3-05 OOF repeat set changed")
    counts = raw.groupby(["repeat", "row_index"], observed=True).size()
    if counts.empty or int(counts.max()) != 1 or int(counts.min()) != 1:
        raise DataContractError("V3-05 one prediction per repeat/row contract failed")
    spread = raw.groupby("row_index", observed=True)[TARGET_COLUMN].agg(["min", "max"])
    if not np.allclose(spread["min"], spread["max"], rtol=0, atol=1e-12):
        raise DataContractError("V3-05 repeated OOF targets disagree")
    agg_map: dict[str, tuple[str, str]] = {TARGET_COLUMN: (TARGET_COLUMN, "first")}
    for column in prediction_columns:
        agg_map[column] = (column, "mean")
    result = (
        raw.groupby("row_index", as_index=False, observed=True)
        .agg(**agg_map)
        .sort_values("row_index")
        .reset_index(drop=True)
    )
    return result


def _repeat_metrics(raw: pd.DataFrame, prediction_columns: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat, group in raw.groupby("repeat", observed=True, sort=True):
        item: dict[str, Any] = {"repeat": int(repeat), "n": int(len(group))}
        for column in prediction_columns:
            metric = evaluate_predictions(group[TARGET_COLUMN], group[column])
            item[column] = _json_safe(metric)
        rows.append(item)
    return rows


def _small_subset_sensitivity(
    top: pd.DataFrame,
    *,
    candidate_columns: Sequence[str],
) -> dict[str, Any]:
    n = len(top)
    subset_n = min(SMALL_SUBSET_N, n)
    if subset_n < 3:
        return {"status": "insufficient_n", "n": n}
    rng = np.random.default_rng(RANDOM_SEED)
    references = ("raw_baseline", "S1")
    store: dict[str, list[float]] = {
        f"{candidate}_minus_{reference}": []
        for candidate in candidate_columns
        for reference in references
        if candidate != reference
    }
    for _ in range(SMALL_SUBSET_REPETITIONS):
        idx = rng.choice(np.arange(n), size=subset_n, replace=False)
        y = top.iloc[idx][TARGET_COLUMN].to_numpy(dtype=float)
        rmse = {
            column: float(
                root_mean_squared_error(y, top.iloc[idx][column].to_numpy(dtype=float)).value
            )
            for column in set(candidate_columns).union(references)
        }
        for key in store:
            candidate, reference = key.split("_minus_", 1)
            store[key].append(rmse[candidate] - rmse[reference])
    summary: dict[str, Any] = {}
    for key, values in store.items():
        a = np.asarray(values, dtype=float)
        summary[key] = {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "q10": float(np.quantile(a, 0.10)),
            "q90": float(np.quantile(a, 0.90)),
            "improvement_fraction": float(np.mean(a < 0)),
        }
    return {
        "status": "ok",
        "seed": RANDOM_SEED,
        "source_n": n,
        "subset_n": subset_n,
        "repetitions": SMALL_SUBSET_REPETITIONS,
        "without_replacement": True,
        "delta_definition": "candidate_rmse_minus_reference_rmse_negative_is_better",
        "comparisons": summary,
    }


def _fit_s2(train: pd.DataFrame, prediction: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    x = train[BASELINE_COLUMN].to_numpy(dtype=float)
    y = train[TARGET_COLUMN].to_numpy(dtype=float)
    denom = float(np.dot(x, x))
    if denom <= 0:
        c = 0.0
    else:
        c = max(0.0, float(np.dot(x, y) / denom))
    pred = c * prediction[BASELINE_COLUMN].to_numpy(dtype=float)
    return pred, {"c": c, "rank_preserving_parameter": bool(c > 0)}


def _fit_s3(train: pd.DataFrame, prediction: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    train_score = _leave_one_out_rank_score(train[BASELINE_COLUMN].to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(train_score)), train_score])
    coefficients, _ = nnls(design, train[TARGET_COLUMN].to_numpy(dtype=float))
    a, b = map(float, coefficients)
    pred_score = _empirical_score(
        train[BASELINE_COLUMN].to_numpy(dtype=float),
        prediction[BASELINE_COLUMN].to_numpy(dtype=float),
    )
    pred = a + b * pred_score
    return pred, {
        "a": a,
        "b": b,
        "rank_score_source": "leave_one_subject_out_empirical_baseline_rank",
        "rank_preserving_parameter": bool(b > 0),
        "constant_degenerate": bool(b == 0),
    }


def _fit_s4(
    train: pd.DataFrame,
    prediction: pd.DataFrame,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    x = train[BASELINE_COLUMN].to_numpy(dtype=float)
    y = train[TARGET_COLUMN].to_numpy(dtype=float)
    p = prediction[BASELINE_COLUMN].to_numpy(dtype=float)
    age_train = pd.to_numeric(train["age"], errors="coerce").to_numpy(dtype=float)
    age_pred = pd.to_numeric(prediction["age"], errors="coerce").to_numpy(dtype=float)
    if (
        np.any(x <= 0)
        or np.any(y <= 0)
        or np.any(p <= 0)
        or not np.isfinite(age_train).all()
        or not np.isfinite(age_pred).all()
    ):
        return None, {
            "state": "data_limited_nonpositive_or_missing_age",
            "reason": "S4 requires strictly positive baseline/target and finite age under the frozen formula; no pseudocount or imputation is authorized",
        }
    mean = float(np.mean(age_train))
    scale = float(np.std(age_train))
    if not np.isfinite(scale) or scale <= 0:
        return None, {
            "state": "data_limited_constant_age",
            "reason": "S4 age standardization is undefined for constant training age",
        }
    z_train = ((age_train - mean) / scale).reshape(-1, 1)
    z_pred = ((age_pred - mean) / scale).reshape(-1, 1)
    residual = np.log(y) - np.log(x)
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(z_train, residual)
    correction = model.predict(z_pred)
    pred = p * np.exp(np.clip(correction, -50.0, 50.0))
    require_finite(pred, name="V3-05 S4 prediction")
    return pred, {
        "state": "evaluated",
        "alpha": RIDGE_ALPHA,
        "age_mean_train": mean,
        "age_std_train": scale,
        "b0": float(model.intercept_),
        "b1": float(np.asarray(model.coef_).reshape(-1)[0]),
        "pseudocount_used": False,
        "age_imputation_used": False,
    }


def run_v3_05_task13(config: Any, inputs: InputBundle) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate the four frozen Task1.3 raw-scale heads and build private banks."""
    train, challenge, measurement, strict_dataset = _task13_frames(inputs)
    subjects = train["subject_group"].astype(str)
    if subjects.nunique() != len(train):
        raise DataContractError("V3-05 Task1.3 requires one strict row per independent subject")
    splits = repeated_group_kfold(
        subjects, n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=42
    )
    counter = _FitCounter()
    b21_spec = _find_spec(config, "task_13", "pls_1")
    s4_global_evaluable = bool(
        train[BASELINE_COLUMN].gt(0).all()
        and train[TARGET_COLUMN].gt(0).all()
        and challenge[BASELINE_COLUMN].gt(0).all()
        and pd.to_numeric(train["age"], errors="coerce").notna().all()
        and pd.to_numeric(challenge["age"], errors="coerce").notna().all()
    )
    s4_global_reason = None if s4_global_evaluable else (
        "S4 requires strictly positive source target/baseline, positive Challenge baseline, "
        "and finite age; no pseudocount or age imputation is authorized"
    )

    b21_eval = evaluate_model_spec(
        strict_dataset.train,
        target_column=strict_dataset.target_column,
        splits=splits,
        spec=b21_spec,
        excluded_columns=strict_dataset.excluded_columns,
        aggregate_repeats=False,
    )
    counter.add(len(splits), label="b21_pls1_outer_reference")
    b21_by = b21_eval.raw_oof_predictions[
        ["row_index", "split", "target", "prediction"]
    ].copy()
    b21_by["repeat"] = (
        b21_by["split"].astype(str).str.extract(r"^repeat=(\d+)/fold=\d+$", expand=False).astype(int)
    )
    b21_by = b21_by.rename(columns={"prediction": "b21_compatible_pls1", "target": "__b21_target"})

    parts: list[pd.DataFrame] = []
    fit_receipts: list[dict[str, Any]] = []
    for split in splits:
        repeat_text = str(split.name).split("/", 1)[0]
        repeat = int(repeat_text.split("=", 1)[1])
        outer_train = train.iloc[split.train_indices].copy()
        validation = train.iloc[split.validation_indices].copy()
        target = validation[TARGET_COLUMN].to_numpy(dtype=float)
        part = pd.DataFrame(
            {
                "row_index": split.validation_indices,
                "repeat": repeat,
                "split": split.name,
                TARGET_COLUMN: target,
                "raw_baseline": validation[BASELINE_COLUMN].to_numpy(dtype=float),
            }
        )
        s1 = float(np.mean(outer_train[TARGET_COLUMN].to_numpy(dtype=float)))
        part["S1"] = s1
        counter.add(1, label="S1_outer")
        part["S2"], s2_info = _fit_s2(outer_train, validation)
        counter.add(1, label="S2_outer")
        part["S3"], s3_info = _fit_s3(outer_train, validation)
        counter.add(1, label="S3_outer")
        if s4_global_evaluable:
            s4, s4_info = _fit_s4(outer_train, validation)
            if s4 is None:
                raise DataContractError("V3-05 S4 passed global precheck but failed an outer fold")
            part["S4"] = s4
            counter.add(1, label="S4_outer")
        else:
            s4_info = {"state": "data_limited_global_precheck", "reason": s4_global_reason}
            part["S4"] = np.nan
        fit_receipts.append(
            {
                "split": split.name,
                "train_n": int(len(outer_train)),
                "validation_n": int(len(validation)),
                "S1": {"mean_target": s1},
                "S2": s2_info,
                "S3": s3_info,
                "S4": s4_info,
            }
        )
        parts.append(part)

    raw = pd.concat(parts, ignore_index=True)
    raw = raw.merge(
        b21_by[["row_index", "repeat", "b21_compatible_pls1", "__b21_target"]],
        on=["row_index", "repeat"],
        how="left",
        validate="one_to_one",
    )
    if not np.allclose(raw[TARGET_COLUMN], raw["__b21_target"], rtol=0, atol=1e-12):
        raise DataContractError("V3-05 regenerated B2.1 OOF target alignment failed")
    raw = raw.drop(columns="__b21_target")

    prediction_columns = ["raw_baseline", "b21_compatible_pls1", "S1", "S2", "S3"]
    s4_evaluated = bool(raw["S4"].notna().all())
    if raw["S4"].notna().any() and not s4_evaluated:
        raise DataContractError("V3-05 S4 must be evaluable on all outer folds or none")
    if s4_evaluated:
        prediction_columns.append("S4")

    top = _aggregate_repeats(raw, prediction_columns=prediction_columns)
    if len(top) != len(train) or set(top["row_index"]) != set(range(len(train))):
        raise DataContractError("V3-05 aggregated OOF does not cover every strict teacher once")

    strict_rank = _rank_ties(train[BASELINE_COLUMN].to_numpy(dtype=float))
    target_rank_metric = safe_spearman(train[TARGET_COLUMN], strict_rank)
    metrics = {column: _candidate_metrics(top, column) for column in prediction_columns}
    rmse_deltas: dict[str, Any] = {}
    for candidate in ("S1", "S2", "S3", "S4"):
        if candidate not in metrics:
            rmse_deltas[candidate] = {"state": "data_limited"}
            continue
        value = float(metrics[candidate]["rmse"]["value"])
        rmse_deltas[candidate] = {
            "candidate_rmse": value,
            "minus_raw_baseline": value - float(metrics["raw_baseline"]["rmse"]["value"]),
            "minus_S1": value - float(metrics["S1"]["rmse"]["value"]),
            "improves_raw_baseline": bool(value < float(metrics["raw_baseline"]["rmse"]["value"])),
            "improves_S1": bool(value < float(metrics["S1"]["rmse"]["value"])),
        }

    challenge_bank = challenge[
        ["participant_id", "subject_group", "study_group", BASELINE_COLUMN]
    ].copy()
    challenge_bank["raw_baseline"] = challenge[BASELINE_COLUMN].to_numpy(dtype=float)
    _, b21_challenge = fit_final_model(
        strict_dataset.train,
        strict_dataset.challenge,
        target_column=strict_dataset.target_column,
        spec=b21_spec,
        excluded_columns=strict_dataset.excluded_columns,
    )
    counter.add(1, label="b21_pls1_full_reference")
    challenge_bank["b21_compatible_pls1"] = b21_challenge
    s1_full = float(np.mean(train[TARGET_COLUMN].to_numpy(dtype=float)))
    challenge_bank["S1"] = s1_full
    counter.add(1, label="S1_full")
    challenge_bank["S2"], s2_full = _fit_s2(train, challenge)
    counter.add(1, label="S2_full")
    challenge_bank["S3"], s3_full = _fit_s3(train, challenge)
    counter.add(1, label="S3_full")
    if s4_global_evaluable:
        s4_full, s4_full_info = _fit_s4(train, challenge)
        if s4_full is None:
            raise DataContractError("V3-05 S4 passed global precheck but failed full fit")
        challenge_bank["S4"] = s4_full
        counter.add(1, label="S4_full")
    else:
        s4_full_info = {"state": "data_limited_global_precheck", "reason": s4_global_reason}

    rank_contracts = {
        "S2_vs_raw_baseline": _rank_order_contract(
            challenge_bank["raw_baseline"], challenge_bank["S2"]
        ),
        "S3_vs_raw_baseline": _rank_order_contract(
            challenge_bank["raw_baseline"], challenge_bank["S3"]
        ),
    }
    if "S4" in challenge_bank:
        rank_contracts["S4_vs_raw_baseline"] = _rank_order_contract(
            challenge_bank["raw_baseline"], challenge_bank["S4"]
        )

    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "stage": "V3-05",
        "task": TASK,
        "new_candidate_conditions": list(NEW_CONDITIONS),
        "new_candidate_condition_count": len(NEW_CONDITIONS),
        "Task1.2": {
            "state": "not_executed_data_limited",
            "reason": "V3-01 found zero strict raw-unit/parent compatible supervised baseline support",
        },
        "Task1.3": {
            "state": "single_domain_conditional",
            "measurement": measurement,
            "outer_cv": {
                "split_family": "repeated_group_kfold",
                "n_splits": OUTER_SPLITS,
                "n_repeats": OUTER_REPEATS,
                "subject_purged": True,
                "raw_oof_rows": int(len(raw)),
                "top_line_rows": int(len(top)),
            },
            "current_rank_expert": {
                "name": "strict_ASC_same_readout",
                "spearman": None if target_rank_metric.status != "ok" else float(target_rank_metric.value),
                "spearman_status": target_rank_metric.status,
                "rmse_assigned": False,
            },
            "references": {
                "raw_baseline": metrics["raw_baseline"],
                "old_raw_b21_compatible_pls1": metrics["b21_compatible_pls1"],
            },
            "conditions": {
                name: (
                    {"state": "evaluated", "metrics": metrics[name], "paired_rmse": rmse_deltas[name]}
                    if name in metrics
                    else {"state": "data_limited", "paired_rmse": rmse_deltas[name]}
                )
                for name in NEW_CONDITIONS
            },
            "repeat_metrics": _repeat_metrics(raw, prediction_columns),
            "small_subset_sensitivity": _small_subset_sensitivity(
                top, candidate_columns=[name for name in NEW_CONDITIONS if name in metrics]
            ),
            "challenge_rank_contracts": rank_contracts,
            "full_fit_parameters": {
                "S1": {"mean_target": s1_full},
                "S2": s2_full,
                "S3": s3_full,
                "S4": s4_full_info,
            },
        },
        "fit_count": counter.count,
        "fit_limit": MAX_FITS,
        "raw_unit_conversion_performed": False,
        "pseudocount_added": False,
        "public_leaderboard_used": False,
        "competition_submission_attempted": False,
        "final_submission_selection_attempted": False,
        "private_bank_contains_row_level_values": True,
        "aggregate_contains_row_level_values": False,
    }

    oof_bank = raw.copy()
    source = train[
        ["participant_id", "subject_group", "study_group", BASELINE_COLUMN, TARGET_COLUMN]
    ].reset_index().rename(columns={"index": "row_index"})
    oof_bank = oof_bank.merge(
        source[["row_index", "participant_id", "subject_group", "study_group"]],
        on="row_index",
        how="left",
        validate="many_to_one",
    )
    oof_bank.insert(0, "task", TASK)
    return _json_safe(aggregate), oof_bank, challenge_bank


def write_v3_05_outputs(
    aggregate: Mapping[str, Any],
    oof_bank: pd.DataFrame,
    challenge_bank: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Persist private row-level banks plus aggregate-safe summary/manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for name, frame in (
        ("v3_v05_task13_oof_bank.csv", oof_bank),
        ("v3_v05_task13_challenge_bank.csv", challenge_bank),
    ):
        path = out / name
        if path.exists():
            raise DataContractError(f"V3-05 refuses to overwrite:{name}")
        frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        raw = path.read_bytes()
        files.append(
            {
                "filename": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "rows": int(len(frame)),
                "private_row_level": True,
            }
        )
    summary = out / "v3_v05_task13_summary.json"
    manifest = out / "v3_v05_task13_bank_manifest.json"
    if summary.exists() or manifest.exists():
        raise DataContractError("V3-05 aggregate output already exists")
    summary.write_text(
        json.dumps(_json_safe(aggregate), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    safe_manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "files": files,
        "row_level_contents_must_not_be_publicly_emitted": True,
        "competition_submission_attempted": False,
    }
    manifest.write_text(
        json.dumps(safe_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return safe_manifest
