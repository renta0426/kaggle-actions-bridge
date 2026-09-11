"""Strategy-v2 E12c-v2: assemble the manual submission from the exact Public-0.218 backbone.

E12c-v1 attempted to refit all seven task components and then require bit-identical
reproduction of the E12b full-candidate fingerprint. The first authorized Kaggle
run completed model construction and structural validation but failed the exact
frozen-candidate fingerprint gate. E12c-v2 removes unnecessary refits from the
manual-submission path: the exact historical Task1.2-only Public-0.218 CSV is the
immutable backbone, and only the already-selected strict Task1.3 ASC anchor is
regenerated from Competition data.

This preserves the intended competition intervention exactly: every column except
Task1.3 is numeric content from the known 0.218 candidate, while Task1.3 is the
frozen strict ASC predictor. No leaderboard result is used for selection.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
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
from .datasets import build_task_13_dataset
from .runner import InputBundle
from .strategy_e12a_v2 import (
    TASK13_ANCHOR_COLUMN,
    TASK13_FINAL_INCUMBENT,
    run_strict_task13_anchor_audit,
)
from .strategy_e12b import (
    canonical_csv_bytes,
    semantic_submission_sha256,
    task_prediction_sha256,
)
from .strategy_e12c import (
    EXPECTED_HISTORICAL_TASK12_ONLY_SHA256,
    locate_historical_task12_only,
    sha256_path,
)

EXPERIMENT = "strategy_v2_e12c_v2_historical_backbone_manual_submission"
EXPECTED_ROWS = 40
EXPECTED_TASK13_UNIQUE = 36
EXPECTED_TASK13_SHA256 = "de8bc3b6bbd3e3aad4099b83eb63a1ebbd81c4f4eeb60f77808858f4674be5da"
EXPECTED_CHANGED_TASKS = ("Task1.3",)


def _read_historical_backbone(
    path: str | Path,
    *,
    sample_submission: pd.DataFrame,
) -> tuple[Path, pd.DataFrame]:
    source = locate_historical_task12_only(
        path,
        expected_sha256=EXPECTED_HISTORICAL_TASK12_ONLY_SHA256,
    )
    frame = pd.read_csv(
        source,
        dtype={PARTICIPANT_ID_COLUMN: str},
        float_precision="round_trip",
    )
    report = validate_submission(
        frame,
        sample_submission,
        require_nonconstant_public_tasks=True,
    )
    if int(report.rows) != EXPECTED_ROWS:
        raise DataContractError("E12c-v2 historical backbone row count changed")
    return source, frame


def _strict_task13_frame(
    config: BaselineConfig,
    inputs: InputBundle,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    audit = run_strict_task13_anchor_audit(config, inputs)
    if audit.get("reproduced") is not True:
        raise DataContractError("E12c-v2 strict Task1.3 anchor did not reproduce")
    if audit.get("incumbent") != TASK13_FINAL_INCUMBENT:
        raise DataContractError("E12c-v2 strict Task1.3 incumbent identity changed")
    if audit.get("predictor_feature") != TASK13_ANCHOR_COLUMN:
        raise DataContractError("E12c-v2 strict Task1.3 feature identity changed")
    if int(audit.get("challenge_rows", -1)) != EXPECTED_ROWS:
        raise DataContractError("E12c-v2 strict Task1.3 Challenge row count changed")
    if int(audit.get("challenge_anchor_unique_values", -1)) != EXPECTED_TASK13_UNIQUE:
        raise DataContractError("E12c-v2 strict Task1.3 Challenge uniqueness changed")

    tables = inputs.tables
    required = ("public_flow", "challenge_flow", "participants", "investigations")
    if any(name not in tables for name in required):
        raise DataContractError("E12c-v2 strict Task1.3 required table missing")
    dataset = build_task_13_dataset(
        tables["public_flow"],
        tables["challenge_flow"],
        tables["participants"],
        tables["investigations"],
        mode="broad",
        include_sdy272_asc_proxy=False,
    )
    challenge = dataset.challenge
    require_columns(
        challenge,
        [PARTICIPANT_ID_COLUMN, TASK13_ANCHOR_COLUMN],
        table_name="E12c-v2 strict Task1.3 Challenge",
    )
    result = challenge[[PARTICIPANT_ID_COLUMN, TASK13_ANCHOR_COLUMN]].copy()
    result[PARTICIPANT_ID_COLUMN] = result[PARTICIPANT_ID_COLUMN].astype(str)
    result[TASK13_ANCHOR_COLUMN] = pd.to_numeric(
        result[TASK13_ANCHOR_COLUMN], errors="raise"
    ).astype(float)
    if len(result) != EXPECTED_ROWS or result[PARTICIPANT_ID_COLUMN].duplicated().any():
        raise DataContractError("E12c-v2 strict Task1.3 Challenge identity contract changed")
    require_finite(
        result[TASK13_ANCHOR_COLUMN],
        name="E12c-v2.strict_task13.challenge_anchor",
    )
    if int(result[TASK13_ANCHOR_COLUMN].nunique(dropna=False)) != EXPECTED_TASK13_UNIQUE:
        raise DataContractError("E12c-v2 strict Task1.3 Challenge values changed")
    return result, audit


def assemble_from_backbone(
    historical_backbone: pd.DataFrame,
    strict_task13: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Replace exactly Task1.3 in an already-validated historical submission."""
    validate_submission(
        historical_backbone,
        sample_submission,
        require_nonconstant_public_tasks=True,
    )
    require_columns(
        strict_task13,
        [PARTICIPANT_ID_COLUMN, TASK13_ANCHOR_COLUMN],
        table_name="E12c-v2 strict Task1.3 frame",
    )
    if len(strict_task13) != EXPECTED_ROWS:
        raise DataContractError("E12c-v2 strict Task1.3 frame row count changed")
    if strict_task13[PARTICIPANT_ID_COLUMN].astype(str).duplicated().any():
        raise DataContractError("E12c-v2 strict Task1.3 frame participant duplicate")

    strict = strict_task13.copy()
    strict[PARTICIPANT_ID_COLUMN] = strict[PARTICIPANT_ID_COLUMN].astype(str)
    strict[TASK13_ANCHOR_COLUMN] = pd.to_numeric(
        strict[TASK13_ANCHOR_COLUMN], errors="raise"
    ).astype(float)
    require_finite(strict[TASK13_ANCHOR_COLUMN], name="E12c-v2.strict_task13")

    aligned = sample_submission[[PARTICIPANT_ID_COLUMN]].copy()
    aligned[PARTICIPANT_ID_COLUMN] = aligned[PARTICIPANT_ID_COLUMN].astype(str)
    aligned = aligned.merge(
        strict,
        on=PARTICIPANT_ID_COLUMN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if aligned[TASK13_ANCHOR_COLUMN].isna().any():
        raise DataContractError("E12c-v2 strict Task1.3 participant alignment incomplete")
    if not aligned[PARTICIPANT_ID_COLUMN].equals(
        sample_submission[PARTICIPANT_ID_COLUMN].astype(str)
    ):
        raise DataContractError("E12c-v2 strict Task1.3 participant order changed")

    final = historical_backbone.copy(deep=True)
    final["Task1.3"] = aligned[TASK13_ANCHOR_COLUMN].to_numpy(dtype=float)
    report = validate_submission(
        final,
        sample_submission,
        require_nonconstant_public_tasks=True,
    )
    if int(report.rows) != EXPECTED_ROWS:
        raise DataContractError("E12c-v2 final row count changed")

    changed: list[str] = []
    for task in TASK_COLUMNS:
        left = pd.to_numeric(final[task], errors="raise").to_numpy(dtype=float)
        right = pd.to_numeric(historical_backbone[task], errors="raise").to_numpy(dtype=float)
        if not np.array_equal(left, right, equal_nan=False):
            changed.append(task)
    if tuple(changed) != EXPECTED_CHANGED_TASKS:
        raise DataContractError(
            f"E12c-v2 historical-backbone changed-task contract failed:{tuple(changed)}"
        )

    task13_sha = task_prediction_sha256(final, "Task1.3")
    if task13_sha != EXPECTED_TASK13_SHA256:
        raise DataContractError(
            f"E12c-v2 strict Task1.3 fingerprint mismatch:{task13_sha}"
        )
    canonical = canonical_csv_bytes(final)
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "submission_rows": int(report.rows),
        "submission_columns": list(report.columns),
        "historical_backbone_sha256": None,
        "historical_public_score": 0.218,
        "changed_tasks_vs_historical_0_218": list(changed),
        "task13_final_incumbent": TASK13_FINAL_INCUMBENT,
        "task13_predictor_feature": TASK13_ANCHOR_COLUMN,
        "task13_unique_values": int(final["Task1.3"].nunique(dropna=False)),
        "task13_prediction_sha256": task13_sha,
        "semantic_submission_sha256": semantic_submission_sha256(final),
        "canonical_csv_sha256": hashlib.sha256(canonical).hexdigest(),
        "canonical_csv_bytes": len(canonical),
        "manual_submission_ready": True,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
    }
    return final, result


def run_e12c_v2(
    config: BaselineConfig,
    inputs: InputBundle,
    historical_task12_only: str | Path,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Assemble the manual candidate without refitting unchanged task models."""
    sample = inputs.tables["sample_submission"]
    source, historical = _read_historical_backbone(
        historical_task12_only,
        sample_submission=sample,
    )
    strict, audit = _strict_task13_frame(config, inputs)
    final, aggregate = assemble_from_backbone(historical, strict, sample)
    result = dict(aggregate)
    result["historical_backbone_sha256"] = sha256_path(source)
    result["task13_historical_reproduced"] = bool(audit.get("reproduced"))
    result["task13_historical_study_equal_spearman"] = float(
        audit["study_equal_spearman"]
    )
    result["task13_challenge_anchor_complete"] = bool(
        audit["challenge_anchor_complete"]
    )
    return final, result
