"""Strategy-v2 E12c: generate the exact E12b candidate for manual submission.

This module performs no Kaggle network operation. It regenerates the frozen E12b
candidate, verifies its exact fingerprints, verifies the historical Task1.2-only
0.218 control by its frozen byte hash, proves that Task1.3 is the only changed task,
and only then persists a canonical ``submission.csv`` for manual upload.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .configuration import load_baseline_config
from .contracts import PARTICIPANT_ID_COLUMN, TASK_COLUMNS, DataContractError, validate_submission
from .runner import load_inputs
from .strategy_e12b import (
    canonical_csv_bytes,
    run_strategy_e12b,
    semantic_submission_sha256,
)

EXPERIMENT = "strategy_v2_e12c_manual_submission_generator"
EXPECTED_E12B_SEMANTIC_SHA256 = (
    "5faf93bee4b68aba23c53e09adbb251bdd07d287dd48fb6d94d6fce53ef9a60f"
)
EXPECTED_E12B_CANONICAL_CSV_SHA256 = (
    "983aaf097d04477c4ccf7bf817fdf66e552937e69bceaa48cfb260cb84413f1b"
)
EXPECTED_E12B_CANONICAL_CSV_BYTES = 5933
EXPECTED_HISTORICAL_TASK12_ONLY_SHA256 = (
    "365607d59cd530656b929a1c1c57412cc6d375265a8d1ba10d304c64e012f387"
)
EXPECTED_E12B_SOURCE_BLOB = "af5df92ca81abc34a7f7046dba5bc284c98302d4"
EXPECTED_E12A_V2_SOURCE_BLOB = "0c4c970c8bacfed61bfbb9587e0a0bfdec7903d9"
EXPECTED_B21_CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
EXPECTED_CHANGED_TASKS_VS_HISTORICAL = ("Task1.3",)
EXPECTED_ROWS = 40


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha_path(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def validate_frozen_source_identity(repository_root: str | Path) -> Mapping[str, str]:
    root = Path(repository_root).expanduser().resolve()
    expected = {
        "src/cmi_flu/strategy_e12b.py": EXPECTED_E12B_SOURCE_BLOB,
        "src/cmi_flu/strategy_e12a_v2.py": EXPECTED_E12A_V2_SOURCE_BLOB,
        "configs/baseline_b021_robust.yaml": EXPECTED_B21_CONFIG_BLOB,
    }
    observed: dict[str, str] = {}
    for relative, wanted in expected.items():
        path = root / relative
        if not path.is_file():
            raise DataContractError(f"E12c frozen source missing: {relative}")
        found = git_blob_sha_path(path)
        if found != wanted:
            raise DataContractError(
                f"E12c frozen source identity changed: {relative}: {found} != {wanted}"
            )
        observed[relative] = found
    return observed


def locate_historical_task12_only(
    source: str | Path,
    *,
    expected_sha256: str = EXPECTED_HISTORICAL_TASK12_ONLY_SHA256,
) -> Path:
    candidate = Path(source).expanduser().resolve()
    if candidate.is_file():
        found = sha256_path(candidate)
        if found != expected_sha256:
            raise DataContractError(
                f"E12c historical Task1.2-only hash mismatch: {found} != {expected_sha256}"
            )
        return candidate
    if not candidate.is_dir():
        raise DataContractError(f"E12c historical control path does not exist: {candidate}")

    matches: list[Path] = []
    for path in candidate.rglob("*.csv"):
        if path.is_file() and sha256_path(path) == expected_sha256:
            matches.append(path)
    if len(matches) != 1:
        raise DataContractError(
            "E12c historical Task1.2-only discovery must find exactly one exact-hash CSV; "
            f"found {len(matches)}"
        )
    return matches[0]


def changed_tasks_exact(left: pd.DataFrame, right: pd.DataFrame) -> tuple[str, ...]:
    if tuple(left.columns) != tuple(right.columns):
        raise DataContractError("E12c comparison columns/order differ")
    if not left[PARTICIPANT_ID_COLUMN].astype(str).equals(
        right[PARTICIPANT_ID_COLUMN].astype(str)
    ):
        raise DataContractError("E12c comparison participant order/content differs")

    changed: list[str] = []
    for task in TASK_COLUMNS:
        left_values = pd.to_numeric(left[task], errors="raise").to_numpy(dtype=float)
        right_values = pd.to_numeric(right[task], errors="raise").to_numpy(dtype=float)
        if not np.array_equal(left_values, right_values, equal_nan=False):
            changed.append(task)
    return tuple(changed)


def validate_historical_task12_only(
    historical_path: str | Path,
    *,
    final_submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    expected_sha256: str = EXPECTED_HISTORICAL_TASK12_ONLY_SHA256,
) -> Mapping[str, Any]:
    path = locate_historical_task12_only(
        historical_path,
        expected_sha256=expected_sha256,
    )
    # round_trip is required here: this gate compares the numeric content of an
    # already-submitted CSV against in-memory IEEE754 values and must not introduce
    # parser rounding of its own.
    historical = pd.read_csv(
        path,
        dtype={PARTICIPANT_ID_COLUMN: str},
        float_precision="round_trip",
    )
    validate_submission(
        historical,
        sample_submission,
        require_nonconstant_public_tasks=True,
    )
    changed = changed_tasks_exact(final_submission, historical)
    if changed != EXPECTED_CHANGED_TASKS_VS_HISTORICAL:
        raise DataContractError(
            "E12c final candidate is not a Task1.3-only intervention versus the exact "
            f"historical 0.218 control: changed={changed}"
        )
    return {
        "historical_task12_only_sha256": sha256_path(path),
        "historical_rows": int(len(historical)),
        "changed_tasks_vs_historical_task12_only": list(changed),
        "task13_only_intervention_proven": True,
    }


def verify_frozen_candidate(
    submission: pd.DataFrame,
    aggregate: Mapping[str, Any],
    sample_submission: pd.DataFrame,
) -> Mapping[str, Any]:
    validation = validate_submission(
        submission,
        sample_submission,
        require_nonconstant_public_tasks=True,
    )
    if int(validation.rows) != EXPECTED_ROWS:
        raise DataContractError(
            f"E12c submission row count changed: {validation.rows} != {EXPECTED_ROWS}"
        )

    semantic = semantic_submission_sha256(submission)
    canonical = canonical_csv_bytes(submission)
    canonical_sha = sha256_bytes(canonical)
    if semantic != EXPECTED_E12B_SEMANTIC_SHA256:
        raise DataContractError(
            f"E12c semantic fingerprint mismatch: {semantic} != {EXPECTED_E12B_SEMANTIC_SHA256}"
        )
    if canonical_sha != EXPECTED_E12B_CANONICAL_CSV_SHA256:
        raise DataContractError(
            "E12c canonical CSV fingerprint mismatch: "
            f"{canonical_sha} != {EXPECTED_E12B_CANONICAL_CSV_SHA256}"
        )
    if len(canonical) != EXPECTED_E12B_CANONICAL_CSV_BYTES:
        raise DataContractError(
            "E12c canonical CSV byte count mismatch: "
            f"{len(canonical)} != {EXPECTED_E12B_CANONICAL_CSV_BYTES}"
        )

    fingerprints = aggregate.get("fingerprint_contract") or {}
    if fingerprints.get("semantic_submission_sha256") != semantic:
        raise DataContractError("E12c aggregate semantic fingerprint does not match candidate")
    if fingerprints.get("canonical_csv_sha256") != canonical_sha:
        raise DataContractError("E12c aggregate canonical fingerprint does not match candidate")
    if int(aggregate.get("challenge_rows", -1)) != EXPECTED_ROWS:
        raise DataContractError("E12c E12b aggregate Challenge row contract changed")
    controls = aggregate.get("structural_controls") or {}
    if controls.get("final_changed_tasks_vs_regenerated_task12_only") != ["Task1.3"]:
        raise DataContractError("E12c E12b Task1.2-only structural control changed")
    if aggregate.get("competition_submission_attempted") is not False:
        raise DataContractError("E12c parent E12b unexpectedly attempted a submission")

    return {
        "semantic_submission_sha256": semantic,
        "canonical_csv_sha256": canonical_sha,
        "canonical_csv_bytes": len(canonical),
        "submission_rows": int(validation.rows),
        "submission_columns": list(validation.columns),
        "minus99_tasks": list(validation.minus99_tasks),
    }


def generate_manual_submission(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    historical_task12_only: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    """Generate and persist the exact frozen candidate only after all E12c gates pass."""
    root = Path(repository_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise DataContractError(f"E12c output already exists: {output}")

    source_identity = validate_frozen_source_identity(root)
    config = load_baseline_config(config_path, repository_root=root)
    inputs = load_inputs(config)
    submission, aggregate = run_strategy_e12b(config, inputs)
    sample = inputs.tables["sample_submission"]

    frozen = verify_frozen_candidate(submission, aggregate, sample)
    historical = validate_historical_task12_only(
        historical_task12_only,
        final_submission=submission,
        sample_submission=sample,
    )

    canonical = canonical_csv_bytes(submission)
    # No row-level file is persisted before every frozen-candidate and historical-control
    # gate above has passed.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        temporary.write_bytes(canonical)
        if sha256_path(temporary) != EXPECTED_E12B_CANONICAL_CSV_SHA256:
            raise DataContractError("E12c persisted temporary CSV hash mismatch")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    if sha256_path(output) != EXPECTED_E12B_CANONICAL_CSV_SHA256:
        output.unlink(missing_ok=True)
        raise DataContractError("E12c final persisted CSV hash mismatch")

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "output_path": str(output),
        **frozen,
        **historical,
        "frozen_source_git_blobs": dict(source_identity),
        "manual_submission_ready": True,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
    }
    return result


def render_safe_result(result: Mapping[str, Any]) -> str:
    safe = {
        key: result[key]
        for key in (
            "experiment",
            "output_path",
            "semantic_submission_sha256",
            "canonical_csv_sha256",
            "canonical_csv_bytes",
            "submission_rows",
            "historical_task12_only_sha256",
            "changed_tasks_vs_historical_task12_only",
            "task13_only_intervention_proven",
            "manual_submission_ready",
            "competition_submission_attempted",
        )
    }
    return json.dumps(safe, sort_keys=True, ensure_ascii=False)
