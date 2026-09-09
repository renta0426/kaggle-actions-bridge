"""Data and submission contracts for the CMI-Flu baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PARTICIPANT_ID_COLUMN = "participant_id"
TASK_COLUMNS: tuple[str, ...] = (
    "Task1.1",
    "Task1.2",
    "Task1.3",
    "Task1.4",
    "Task2.1",
    "Task2.2",
    "Task2.3",
)
PUBLIC_LEADERBOARD_TASKS: tuple[str, ...] = TASK_COLUMNS[:-1]
SUBMISSION_COLUMNS: tuple[str, ...] = (PARTICIPANT_ID_COLUMN, *TASK_COLUMNS)

DEFAULT_SENTINELS: frozenset[str] = frozenset(
    {"", "na", "n/a", "nan", "none", "null", "unknown"}
)

REQUIRED_COMPETITION_FILES: tuple[str, ...] = (
    "participants.tsv",
    "investigations_260821.tsv",
    "publicData_cytokine.tsv",
    "publicData_ex_vivo_flow.tsv",
    "publicData_serology_260821.tsv",
    "2025LJI_aim.tsv",
    "2025LJI_cytokine.tsv",
    "2025LJI_ex_vivo_flow.tsv",
    "2025LJI_serology.tsv",
    "sample_submission_part1.csv",
    "md5sum",
)


class DataContractError(ValueError):
    """Raised when an input or generated artifact violates a hard contract."""


@dataclass(frozen=True)
class SubmissionValidationReport:
    """Summary returned after a submission passes all hard checks."""

    rows: int
    columns: tuple[str, ...]
    task_unique_counts: Mapping[str, int]
    minus99_tasks: tuple[str, ...]


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    table_name: str = "table",
) -> None:
    """Require a set of columns and report all missing names at once."""

    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataContractError(f"{table_name} is missing required columns: {missing}")


def require_unique(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    table_name: str = "table",
) -> None:
    """Require a candidate key to be unique."""

    require_columns(frame, columns, table_name=table_name)
    duplicate_mask = frame.duplicated(list(columns), keep=False)
    if duplicate_mask.any():
        count = int(duplicate_mask.sum())
        raise DataContractError(
            f"{table_name} has {count} rows participating in duplicate key {list(columns)}"
        )


def normalize_sentinel_strings(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    sentinels: Iterable[str] = DEFAULT_SENTINELS,
) -> pd.DataFrame:
    """Convert configured literal sentinels to missing values without touching numbers."""

    result = frame.copy()
    normalized_sentinels = {str(value).strip().casefold() for value in sentinels}
    selected = list(columns) if columns is not None else list(result.columns)
    require_columns(result, selected)

    for column in selected:
        series = result[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        stripped = series.astype("string").str.strip()
        mask = stripped.str.casefold().isin(normalized_sentinels)
        result[column] = stripped.mask(mask, pd.NA)
    return result


def coerce_numeric(
    series: pd.Series,
    *,
    name: str | None = None,
    allow_missing: bool = False,
) -> pd.Series:
    """Coerce values to numeric and fail when non-sentinel values cannot be parsed."""

    label = name or str(series.name)
    converted = pd.to_numeric(series, errors="coerce")
    newly_missing = series.notna() & converted.isna()
    if newly_missing.any():
        examples = series.loc[newly_missing].astype(str).drop_duplicates().head(5).tolist()
        raise DataContractError(f"{label} contains non-numeric values: {examples}")
    if not allow_missing and converted.isna().any():
        raise DataContractError(f"{label} contains {int(converted.isna().sum())} missing values")
    return converted


def require_finite(
    values: pd.Series | np.ndarray,
    *,
    name: str,
    allow_missing: bool = False,
) -> None:
    """Require finite numeric values."""

    array = np.asarray(values, dtype=float)
    if allow_missing:
        invalid = ~(np.isfinite(array) | np.isnan(array))
    else:
        invalid = ~np.isfinite(array)
    if invalid.any():
        raise DataContractError(f"{name} contains {int(invalid.sum())} non-finite values")


def require_positive(values: pd.Series | np.ndarray, *, name: str) -> None:
    """Require strictly positive finite values, as needed for geometric means and logs."""

    array = np.asarray(values, dtype=float)
    invalid = (~np.isfinite(array)) | (array <= 0)
    if invalid.any():
        raise DataContractError(f"{name} contains {int(invalid.sum())} non-positive values")


def validate_sample_submission(sample: pd.DataFrame) -> None:
    """Validate the immutable schema source used for all generated submissions."""

    actual_columns = tuple(sample.columns)
    if actual_columns != SUBMISSION_COLUMNS:
        raise DataContractError(
            "sample submission columns differ from the expected contract: "
            f"expected={SUBMISSION_COLUMNS}, actual={actual_columns}"
        )
    if len(sample) != 40:
        raise DataContractError(f"sample submission must contain 40 rows, found {len(sample)}")
    if sample[PARTICIPANT_ID_COLUMN].isna().any():
        raise DataContractError("sample submission contains a missing participant_id")
    require_unique(sample, [PARTICIPANT_ID_COLUMN], table_name="sample submission")


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    allowed_minus99_tasks: Iterable[str] = (),
    require_nonconstant_public_tasks: bool = True,
) -> SubmissionValidationReport:
    """Validate a generated submission against the organizer-provided template."""

    validate_sample_submission(sample)
    allowed_minus99 = frozenset(allowed_minus99_tasks)
    unknown_allowed = allowed_minus99.difference(TASK_COLUMNS)
    if unknown_allowed:
        raise DataContractError(f"unknown tasks in allowed_minus99_tasks: {sorted(unknown_allowed)}")

    actual_columns = tuple(submission.columns)
    if actual_columns != tuple(sample.columns):
        raise DataContractError(
            "submission columns or order differ from sample submission: "
            f"expected={tuple(sample.columns)}, actual={actual_columns}"
        )
    if len(submission) != len(sample):
        raise DataContractError(
            f"submission row count differs from sample: expected={len(sample)}, actual={len(submission)}"
        )
    if not submission[PARTICIPANT_ID_COLUMN].equals(sample[PARTICIPANT_ID_COLUMN]):
        raise DataContractError("submission participant_id values or order differ from sample")
    require_unique(submission, [PARTICIPANT_ID_COLUMN], table_name="submission")

    unique_counts: dict[str, int] = {}
    minus99_tasks: list[str] = []
    for task in TASK_COLUMNS:
        numeric = coerce_numeric(submission[task], name=f"submission.{task}")
        require_finite(numeric, name=f"submission.{task}")
        is_minus99 = bool((numeric == -99).all())
        has_partial_minus99 = bool((numeric == -99).any() and not is_minus99)
        if has_partial_minus99:
            raise DataContractError(f"{task} mixes -99 sentinels with predictions")
        if is_minus99:
            minus99_tasks.append(task)
            if task not in allowed_minus99:
                raise DataContractError(f"{task} is entirely -99 but was not explicitly allowed")
        elif (numeric == -99).any():
            raise DataContractError(f"{task} contains an unexpected -99 sentinel")
        unique_counts[task] = int(numeric.nunique(dropna=False))

    if require_nonconstant_public_tasks:
        constant = [
            task
            for task in PUBLIC_LEADERBOARD_TASKS
            if task not in allowed_minus99 and unique_counts[task] < 2
        ]
        if constant:
            raise DataContractError(f"public leaderboard tasks are constant: {constant}")

    return SubmissionValidationReport(
        rows=len(submission),
        columns=actual_columns,
        task_unique_counts=unique_counts,
        minus99_tasks=tuple(minus99_tasks),
    )
