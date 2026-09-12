#!/usr/bin/env python3
"""Execute exactly one fresh V3-05 study-alias repair Notebook."""
from __future__ import annotations

from pathlib import Path

import cmi_flu_v3_v05_execute as parent

REQUEST_ID = "20260912-cmi-flu-strategy-v3-v05-task13-scale-002"
TARGET = "renta0426/cmi-flu-v3-v05-task13-scale-20260912-002"
TITLE = "cmi-flu-v3-v05-task13-scale-20260912-002"
REPAIR_SCIENCE_COMMIT = "f8641f7489f22478b8caf3430d2d432e617c4a25"
REPAIR_SCIENCE_BLOB = "493a8ffa7caed30796a5fe6711577aee6c58905c"
BASE_SCIENCE_BLOB = "136d92de6a5d4f23fb4bb66d4c339aa8f49e79f0"
FAILURE_CODE = "88315152273aaa70245f"
ALIAS_HASH = "6626de8ebc839dbdac44faa19e8c3280348325d2a214fe940984a9d209fe2985"

_NATIVE_VALIDATE_PRIVATE_OUTPUT = parent.validate_private_output


def validate_private_output(private_dir: Path):
    summary, manifest = _NATIVE_VALIDATE_PRIVATE_OUTPUT(private_dir)
    runtime = summary.get("runtime") or {}
    exact_runtime = {
        "repair_science_commit": REPAIR_SCIENCE_COMMIT,
        "repair_science_blob": REPAIR_SCIENCE_BLOB,
        "repair_parent_failure_code": FAILURE_CODE,
        "repair_study_alias_sha256": ALIAS_HASH,
    }
    for key, expected in exact_runtime.items():
        if runtime.get(key) != expected:
            raise RuntimeError(f"V3-05 repair runtime provenance changed:{key}")
    repair = summary.get("runtime_repair") or {}
    if repair.get("repair_version") != "strategy_v3_v05_v2_context_study_alias_repair":
        raise RuntimeError("V3-05 repair version changed")
    if repair.get("parent_failure_code") != FAILURE_CODE or repair.get("study_alias_sha256") != ALIAS_HASH:
        raise RuntimeError("V3-05 repair failure/alias identity changed")
    if repair.get("scope") != "participant_context.study_group_identity_only":
        raise RuntimeError("V3-05 repair scope changed")
    for key in (
        "teacher_changed", "target_changed", "measurement_values_changed", "features_changed",
        "splits_changed", "candidate_formulas_changed", "fit_budget_changed",
        "unit_conversion_performed", "public_leaderboard_used", "competition_submission_attempted",
    ):
        if repair.get(key) is not False:
            raise RuntimeError(f"V3-05 repair boundary changed:{key}")
    return summary, manifest


def main() -> int:
    # Reuse the reviewed one-write/current-output executor while rebinding only
    # the fresh target/request and adding repair-provenance validation.
    parent.REQUEST_ID = REQUEST_ID
    parent.TARGET = TARGET
    parent.TITLE = TITLE
    parent.EXPECTED_VERSION = 1
    parent.SCIENCE_COMMIT = REPAIR_SCIENCE_COMMIT
    parent.SCIENCE_BLOB = BASE_SCIENCE_BLOB
    parent.validate_private_output = validate_private_output
    return parent.main()


if __name__ == "__main__":
    raise SystemExit(main())
