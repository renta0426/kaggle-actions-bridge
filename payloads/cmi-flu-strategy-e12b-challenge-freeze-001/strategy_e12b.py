"""Strategy-v2 E12b: freeze the deterministic 40-donor final portfolio prediction.

E12b is not a model-search stage.  It regenerates the already-reconciled E12a-v2
portfolio with fixed model identities, validates the Competition submission shape,
and exports only aggregate fingerprints and diagnostics.  The row-level candidate
submission exists only transiently inside the authorized private data environment.

A later Competition submission must be separately authorized and must regenerate a
candidate whose semantic fingerprint exactly matches the frozen E12b fingerprint.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import struct
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .configuration import BaselineConfig
from .contracts import (
    PARTICIPANT_ID_COLUMN,
    TASK_COLUMNS,
    DataContractError,
    require_columns,
    require_finite,
    validate_submission,
)
from .datasets import TaskDataset, HAIModelDataset
from .evaluation import aggregate_hai_task_predictions
from .metrics import percentile_rank, safe_spearman
from .models import ModelSpec, fit_final_model
from .runner import InputBundle, build_b02_datasets
from .strategy_e12a_v2 import (
    EXPECTED_FINAL_PORTFOLIO,
    TASK13_ANCHOR_COLUMN,
    TASK13_FINAL_INCUMBENT,
    run_strict_task13_anchor_audit,
)
from .submission import build_submission
from .targets import build_task_14_anchor


EXPERIMENT = "strategy_v2_e12b_challenge_prediction_freeze"
EXPECTED_CHALLENGE_ROWS = 40
EXPECTED_TASK13_CHALLENGE_UNIQUE = 36

B21_FIXED_MODELS = {
    "Task1.1": ("task_11", "pls_2"),
    "Task1.2": ("task_12", "enet_a0.001_l0.5"),
    "Task1.3": ("task_13", "pls_1"),
    "Task2.1": ("hai", "et_subtype_d3_l5"),
    "Task2.2": ("hai", "et_subtype_d5_l10"),
    "Task2.3": ("hai", "ridge_exact_a100"),
}
B21_PORTFOLIO = {
    "Task1.1": "b21_pls_2",
    "Task1.2": "b21_enet_a0.001_l0.5",
    "Task1.3": "b21_pls_1",
    "Task1.4": "raw_pre_vacc_conserved_anchor",
    "Task2.1": "b21_et_subtype_d3_l5",
    "Task2.2": "b21_et_subtype_d5_l10",
    "Task2.3": "b21_ridge_exact_a100",
}
TASK12_ONLY_PORTFOLIO = {
    **B21_PORTFOLIO,
    "Task1.2": EXPECTED_FINAL_PORTFOLIO["Task1.2"],
}
EXPECTED_CHANGED_FROM_B21 = ("Task1.2", "Task1.3")
EXPECTED_CHANGED_FROM_TASK12_ONLY = ("Task1.3",)
CSV_FLOAT_FORMAT = "%.17g"
SEMANTIC_HASH_VERSION = "cmi-flu-e12b-semantic-v1"
TASK12_ANCHOR_COLUMN = "flow_rank__Classical_monocytes"
TASK12_MODEL_NAME = "et_d5_l5_sqrt"
TASK12_LAMBDA = 0.5
TASK12_RESIDUAL_TARGET_COLUMN = "__e12b_task12_anchor_residual"

_BANNED_AGGREGATE_KEYS = {
    "participant_id",
    "subject_group",
    "row_index",
    "challenge_predictions",
    "submission_rows",
    "prediction_vector",
    "oof_predictions",
}


def _assert_frozen_config(config: BaselineConfig) -> None:
    if config.baseline != "b021_taskwise_robust":
        raise DataContractError("E12b requires b021_taskwise_robust config")
    if str(config.section("selection").get("policy", "")) != "robust_v1":
        raise DataContractError("E12b requires robust_v1 selection policy")
    flow = config.section("flow")
    if (
        str(flow.get("task_12_mode")) != "broad"
        or str(flow.get("task_13_mode")) != "broad"
        or bool(flow.get("include_sdy272_asc_proxy"))
    ):
        raise DataContractError("E12b requires frozen broad-flow/no-proxy B2.1 contract")


def _find_spec(config: BaselineConfig, model_set: str, name: str) -> ModelSpec:
    matches = [spec for spec in config.model_specs(model_set) if spec.name == name]
    if len(matches) != 1:
        raise DataContractError(
            f"E12b expected one fixed model spec {model_set}/{name}; found {len(matches)}"
        )
    return matches[0]


def _prediction_frame(
    participant_id: pd.Series,
    prediction: np.ndarray | pd.Series,
    *,
    task: str,
) -> pd.DataFrame:
    ids = participant_id.astype(str).reset_index(drop=True)
    values = pd.to_numeric(pd.Series(prediction), errors="coerce").to_numpy(dtype=float)
    if len(ids) != EXPECTED_CHALLENGE_ROWS or values.shape != (EXPECTED_CHALLENGE_ROWS,):
        raise DataContractError(f"E12b {task} challenge row count changed")
    if ids.duplicated().any():
        raise DataContractError(f"E12b {task} duplicate challenge participant")
    require_finite(values, name=f"E12b.{task}.challenge_prediction")
    return pd.DataFrame({PARTICIPANT_ID_COLUMN: ids, "prediction": values})


def _fixed_compact_prediction(
    dataset: TaskDataset,
    *,
    config: BaselineConfig,
    task: str,
) -> pd.DataFrame:
    model_set, model_name = B21_FIXED_MODELS[task]
    spec = _find_spec(config, model_set, model_name)
    _, prediction = fit_final_model(
        dataset.train,
        dataset.challenge,
        target_column=dataset.target_column,
        spec=spec,
        excluded_columns=dataset.excluded_columns,
    )
    values = np.asarray(prediction, dtype=float)
    require_finite(values, name=f"E12b.{task}.fixed_compact")
    if task == "Task1.1":
        historical = pd.to_numeric(
            dataset.train[dataset.target_column], errors="coerce"
        ).to_numpy(dtype=float)
        require_finite(historical, name="E12b.Task1.1.historical_target")
        historical_max = float(np.max(historical))
        maximum = float(np.max(values))
        ratio = maximum / historical_max if historical_max > 0 else np.inf
        if float(np.min(values)) < 0.0 or not np.isfinite(ratio) or ratio > 10.0:
            raise DataContractError("E12b Task1.1 fixed prediction plausibility changed")
    return _prediction_frame(
        dataset.challenge[PARTICIPANT_ID_COLUMN],
        values,
        task=task,
    )


def _fixed_hai_prediction(
    dataset: HAIModelDataset,
    *,
    config: BaselineConfig,
    task: str,
    panel_strains: tuple[str, ...],
) -> pd.DataFrame:
    _, model_name = B21_FIXED_MODELS[task]
    spec = _find_spec(config, "hai", model_name)
    _, target_prediction = fit_final_model(
        dataset.train,
        dataset.challenge,
        target_column=dataset.target_column,
        spec=spec,
        excluded_columns=dataset.excluded_columns,
    )
    post_prediction = dataset.target_prediction_to_post_hai(
        np.asarray(target_prediction, dtype=float),
        frame=dataset.challenge,
    )
    require_finite(post_prediction, name=f"E12b.{task}.post_hai")
    strain = dataset.challenge[[PARTICIPANT_ID_COLUMN, "virus_strain"]].copy()
    strain["prediction"] = np.asarray(post_prediction, dtype=float)
    result = aggregate_hai_task_predictions(
        strain,
        panel_strains=panel_strains,
        task=task,
    )
    if len(result) != EXPECTED_CHALLENGE_ROWS:
        raise DataContractError(f"E12b {task} aggregated challenge row count changed")
    require_finite(result["prediction"], name=f"E12b.{task}.panel_prediction")
    return result


def _identity_spec(spec: ModelSpec) -> ModelSpec:
    return replace(spec, target_transform="identity", clip_min=None)


def _task12_anchor_residual_prediction(
    dataset: TaskDataset,
    *,
    config: BaselineConfig,
) -> pd.DataFrame:
    """Reproduce the already-promoted Task1.2 anchor-residual prediction exactly."""
    require_columns(
        dataset.train,
        [
            PARTICIPANT_ID_COLUMN,
            "study_group",
            dataset.target_column,
            TASK12_ANCHOR_COLUMN,
        ],
        table_name="E12b Task1.2 training",
    )
    require_columns(
        dataset.challenge,
        [PARTICIPANT_ID_COLUMN, TASK12_ANCHOR_COLUMN],
        table_name="E12b Task1.2 Challenge",
    )
    train = dataset.train.copy()
    target_rank = train.groupby(
        "study_group", dropna=False, observed=True
    )[dataset.target_column].transform(
        lambda values: percentile_rank(
            pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
        )
    )
    train_anchor = pd.to_numeric(
        train[TASK12_ANCHOR_COLUMN], errors="raise"
    ).to_numpy(dtype=float)
    challenge_anchor = pd.to_numeric(
        dataset.challenge[TASK12_ANCHOR_COLUMN], errors="raise"
    ).to_numpy(dtype=float)
    require_finite(train_anchor, name="E12b.Task1.2.train_anchor")
    require_finite(challenge_anchor, name="E12b.Task1.2.challenge_anchor")
    train[TASK12_RESIDUAL_TARGET_COLUMN] = (
        np.asarray(target_rank, dtype=float) - train_anchor
    )
    spec = _identity_spec(_find_spec(config, "task_12", TASK12_MODEL_NAME))
    excluded = tuple(
        dict.fromkeys([*dataset.excluded_columns, dataset.target_column])
    )
    _, correction = fit_final_model(
        train,
        dataset.challenge,
        target_column=TASK12_RESIDUAL_TARGET_COLUMN,
        spec=spec,
        excluded_columns=excluded,
    )
    correction = np.asarray(correction, dtype=float)
    require_finite(correction, name="E12b.Task1.2.residual_correction")
    score = challenge_anchor + TASK12_LAMBDA * correction
    require_finite(score, name="E12b.Task1.2.combined_score")
    return _prediction_frame(
        dataset.challenge[PARTICIPANT_ID_COLUMN],
        score,
        task="Task1.2",
    )


def _strict_task13_prediction(
    config: BaselineConfig,
    inputs: InputBundle,
    dataset: TaskDataset,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    audit = run_strict_task13_anchor_audit(config, inputs)
    if audit.get("reproduced") is not True:
        raise DataContractError("E12b requires reproduced strict Task1.3 anchor")
    if audit.get("incumbent") != TASK13_FINAL_INCUMBENT:
        raise DataContractError("E12b strict Task1.3 identity changed")
    if int(audit.get("challenge_rows", -1)) != EXPECTED_CHALLENGE_ROWS:
        raise DataContractError("E12b strict Task1.3 challenge rows changed")
    if int(audit.get("challenge_anchor_unique_values", -1)) != EXPECTED_TASK13_CHALLENGE_UNIQUE:
        raise DataContractError("E12b strict Task1.3 Challenge uniqueness changed")
    require_columns(
        dataset.challenge,
        [PARTICIPANT_ID_COLUMN, TASK13_ANCHOR_COLUMN],
        table_name="E12b Task1.3 strict Challenge",
    )
    prediction = _prediction_frame(
        dataset.challenge[PARTICIPANT_ID_COLUMN],
        dataset.challenge[TASK13_ANCHOR_COLUMN],
        task="Task1.3",
    )
    return prediction, audit


def _task14_prediction(inputs: InputBundle) -> pd.DataFrame:
    anchor = build_task_14_anchor(inputs.tables["challenge_aim"])[
        [PARTICIPANT_ID_COLUMN, "anchor"]
    ].rename(columns={"anchor": "prediction"})
    if len(anchor) != EXPECTED_CHALLENGE_ROWS:
        raise DataContractError("E12b Task1.4 challenge row count changed")
    require_finite(anchor["prediction"], name="E12b.Task1.4.anchor")
    return anchor


def _changed_tasks(left: pd.DataFrame, right: pd.DataFrame) -> tuple[str, ...]:
    if not left[PARTICIPANT_ID_COLUMN].astype(str).equals(
        right[PARTICIPANT_ID_COLUMN].astype(str)
    ):
        raise DataContractError("E12b submission participant order mismatch")
    changed = []
    for task in TASK_COLUMNS:
        a = pd.to_numeric(left[task], errors="raise").to_numpy(dtype=float)
        b = pd.to_numeric(right[task], errors="raise").to_numpy(dtype=float)
        if not np.array_equal(a, b, equal_nan=False):
            changed.append(task)
    return tuple(changed)


def semantic_submission_sha256(submission: pd.DataFrame) -> str:
    """Hash row order, participant identity, task names, and IEEE754 prediction bits."""
    digest = hashlib.sha256()
    digest.update((SEMANTIC_HASH_VERSION + "\n").encode("utf-8"))
    for task in TASK_COLUMNS:
        digest.update(task.encode("utf-8") + b"\0")
    for _, row in submission.iterrows():
        participant = str(row[PARTICIPANT_ID_COLUMN]).encode("utf-8")
        digest.update(struct.pack(">I", len(participant)))
        digest.update(participant)
        for task in TASK_COLUMNS:
            value = float(row[task])
            if not np.isfinite(value):
                raise DataContractError(f"E12b semantic hash saw nonfinite {task}")
            digest.update(struct.pack(">d", value))
    return digest.hexdigest()


def task_prediction_sha256(submission: pd.DataFrame, task: str) -> str:
    digest = hashlib.sha256()
    digest.update((SEMANTIC_HASH_VERSION + ":" + task + "\n").encode("utf-8"))
    for _, row in submission.iterrows():
        participant = str(row[PARTICIPANT_ID_COLUMN]).encode("utf-8")
        digest.update(struct.pack(">I", len(participant)))
        digest.update(participant)
        digest.update(struct.pack(">d", float(row[task])))
    return digest.hexdigest()


def canonical_csv_bytes(submission: pd.DataFrame) -> bytes:
    return submission.to_csv(
        index=False,
        lineterminator="\n",
        float_format=CSV_FLOAT_FORMAT,
    ).encode("utf-8")


def _task_summary(submission: pd.DataFrame, task: str) -> Mapping[str, Any]:
    values = pd.to_numeric(submission[task], errors="raise").to_numpy(dtype=float)
    require_finite(values, name=f"E12b.{task}.summary")
    unique = int(len(np.unique(values)))
    return {
        "task": task,
        "rows": int(len(values)),
        "prediction_min": float(np.min(values)),
        "prediction_max": float(np.max(values)),
        "prediction_unique": unique,
        "tie_fraction": float(1.0 - unique / len(values)),
        "prediction_sha256": task_prediction_sha256(submission, task),
    }


def _rank_comparison(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    task: str,
) -> Mapping[str, Any]:
    c = pd.to_numeric(candidate[task], errors="raise").to_numpy(dtype=float)
    r = pd.to_numeric(reference[task], errors="raise").to_numpy(dtype=float)
    require_finite(c, name=f"E12b.{task}.candidate_rank")
    require_finite(r, name=f"E12b.{task}.reference_rank")
    cr = percentile_rank(c)
    rr = percentile_rank(r)
    metric = safe_spearman(c, r)
    return {
        "task": task,
        "rank_spearman": metric.value,
        "rank_spearman_status": metric.status,
        "changed_rank_count": int(np.sum(cr != rr)),
        "mean_absolute_percentile_shift": float(np.mean(np.abs(cr - rr))),
        "maximum_absolute_percentile_shift": float(np.max(np.abs(cr - rr))),
    }


def _assert_aggregate_only(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _BANNED_AGGREGATE_KEYS:
                raise DataContractError(f"E12b aggregate serialization leaked key:{key}")
            _assert_aggregate_only(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_aggregate_only(item)


def summarize_candidate_frames(
    sample: pd.DataFrame,
    b21_submission: pd.DataFrame,
    task12_only_submission: pd.DataFrame,
    final_submission: pd.DataFrame,
    *,
    task13_audit: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    for label, frame in (
        ("b21", b21_submission),
        ("task12_only", task12_only_submission),
        ("final", final_submission),
    ):
        report = validate_submission(
            frame,
            sample,
            require_nonconstant_public_tasks=True,
        )
        if int(report.rows) != EXPECTED_CHALLENGE_ROWS:
            raise DataContractError(f"E12b {label} validation row count changed")

    from_b21 = _changed_tasks(final_submission, b21_submission)
    from_task12 = _changed_tasks(final_submission, task12_only_submission)
    if from_b21 != EXPECTED_CHANGED_FROM_B21:
        raise DataContractError(
            f"E12b final-vs-B2.1 changed task contract changed:{from_b21}"
        )
    if from_task12 != EXPECTED_CHANGED_FROM_TASK12_ONLY:
        raise DataContractError(
            f"E12b final-vs-Task1.2-only changed task contract changed:{from_task12}"
        )

    task_summaries = {
        task: _task_summary(final_submission, task)
        for task in TASK_COLUMNS
    }
    if task_summaries["Task1.3"]["prediction_unique"] != EXPECTED_TASK13_CHALLENGE_UNIQUE:
        raise DataContractError("E12b final Task1.3 unique-value contract changed")

    canonical = canonical_csv_bytes(final_submission)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "final_portfolio": dict(EXPECTED_FINAL_PORTFOLIO),
        "portfolio_task_count": len(EXPECTED_FINAL_PORTFOLIO),
        "challenge_rows": EXPECTED_CHALLENGE_ROWS,
        "submission_validation": asdict(
            validate_submission(
                final_submission,
                sample,
                require_nonconstant_public_tasks=True,
            )
        ),
        "fingerprint_contract": {
            "semantic_hash_version": SEMANTIC_HASH_VERSION,
            "semantic_submission_sha256": semantic_submission_sha256(final_submission),
            "canonical_csv_float_format": CSV_FLOAT_FORMAT,
            "canonical_csv_lineterminator": "LF",
            "canonical_csv_sha256": hashlib.sha256(canonical).hexdigest(),
            "canonical_csv_bytes": len(canonical),
        },
        "task_prediction_summaries": task_summaries,
        "structural_controls": {
            "b21_portfolio": dict(B21_PORTFOLIO),
            "task12_only_portfolio": dict(TASK12_ONLY_PORTFOLIO),
            "final_changed_tasks_vs_regenerated_b21": list(from_b21),
            "final_changed_tasks_vs_regenerated_task12_only": list(from_task12),
            "regenerated_b21_semantic_sha256": semantic_submission_sha256(b21_submission),
            "regenerated_task12_only_semantic_sha256": semantic_submission_sha256(
                task12_only_submission
            ),
            "historical_public_0_218_is_not_assumed_byte_identical": True,
            "task12_final_vs_b21": _rank_comparison(
                final_submission, b21_submission, "Task1.2"
            ),
            "task13_final_vs_b21": _rank_comparison(
                final_submission, b21_submission, "Task1.3"
            ),
            "task13_final_vs_task12_only": _rank_comparison(
                final_submission, task12_only_submission, "Task1.3"
            ),
        },
        "task13_reconciliation": {
            "final_incumbent": TASK13_FINAL_INCUMBENT,
            "predictor_feature": TASK13_ANCHOR_COLUMN,
            "challenge_unique_values": EXPECTED_TASK13_CHALLENGE_UNIQUE,
            "e12a_v2_reproduced": (
                bool(task13_audit.get("reproduced"))
                if task13_audit is not None
                else None
            ),
        },
        "new_model_selection_performed": False,
        "leaderboard_used_for_selection": False,
        "public_probe_performed": False,
        "competition_submission_attempted": False,
        "competition_submission_authorized": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
        "row_level_candidate_persisted": False,
        "next_step": "ready_for_separate_submission_authorization",
    }
    _assert_aggregate_only(result)
    return result


def run_strategy_e12b(
    config: BaselineConfig,
    inputs: InputBundle,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Return the transient final candidate and aggregate-only freeze record."""
    _assert_frozen_config(config)
    datasets = build_b02_datasets(config, inputs)

    compact_b21: dict[str, pd.DataFrame] = {}
    for task in ("Task1.1", "Task1.2", "Task1.3"):
        dataset = datasets[task]
        if not isinstance(dataset, TaskDataset):
            raise DataContractError(f"E12b expected TaskDataset:{task}")
        compact_b21[task] = _fixed_compact_prediction(
            dataset,
            config=config,
            task=task,
        )

    hai_b21: dict[str, pd.DataFrame] = {}
    definitions = {
        "Task2.1": (datasets["HAI_D28"], inputs.vaccine_strains),
        "Task2.2": (datasets["HAI_D28"], inputs.challenge_strains),
        "Task2.3": (datasets["HAI_D365"], inputs.challenge_strains),
    }
    for task, (dataset, panel) in definitions.items():
        if not isinstance(dataset, HAIModelDataset):
            raise DataContractError(f"E12b expected HAIModelDataset:{task}")
        hai_b21[task] = _fixed_hai_prediction(
            dataset,
            config=config,
            task=task,
            panel_strains=tuple(panel),
        )

    task14 = _task14_prediction(inputs)
    b21_predictions = {
        **compact_b21,
        "Task1.4": task14,
        **hai_b21,
    }
    sample = inputs.tables["sample_submission"]
    b21_submission, _ = build_submission(
        sample,
        b21_predictions,
        require_nonconstant_public_tasks=True,
    )

    task12_dataset = datasets["Task1.2"]
    if not isinstance(task12_dataset, TaskDataset):
        raise DataContractError("E12b expected TaskDataset:Task1.2")
    task12_final = _task12_anchor_residual_prediction(
        task12_dataset,
        config=config,
    )
    task13_dataset = datasets["Task1.3"]
    if not isinstance(task13_dataset, TaskDataset):
        raise DataContractError("E12b expected TaskDataset:Task1.3")
    task13_final, task13_audit = _strict_task13_prediction(
        config,
        inputs,
        task13_dataset,
    )

    task12_only_predictions = dict(b21_predictions)
    task12_only_predictions["Task1.2"] = task12_final
    task12_only_submission, _ = build_submission(
        sample,
        task12_only_predictions,
        require_nonconstant_public_tasks=True,
    )

    final_predictions = dict(task12_only_predictions)
    final_predictions["Task1.3"] = task13_final
    final_submission, _ = build_submission(
        sample,
        final_predictions,
        require_nonconstant_public_tasks=True,
    )

    aggregate = summarize_candidate_frames(
        sample,
        b21_submission,
        task12_only_submission,
        final_submission,
        task13_audit=task13_audit,
    )
    return final_submission, aggregate
