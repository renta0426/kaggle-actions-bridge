#!/usr/bin/env python3
"""Prove request 003 changes only identity plus the established MD5 staging repair."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

OLD = "requests/cmi-flu-strategy-e03-task11-optional-view-fusion-002.json"
NEW = "requests/cmi-flu-strategy-e03-task11-optional-view-fusion-003.json"
OLD_REQUEST = "20260908-cmi-flu-strategy-e03-task11-optional-view-fusion-002"
NEW_REQUEST = "20260908-cmi-flu-strategy-e03-task11-optional-view-fusion-003"
OLD_TARGET = "renta0426/cmi-flu-e03-task11-optview-20260908-002"
NEW_TARGET = "renta0426/cmi-flu-e03-task11-optview-20260908-003"


def normalized(payload: dict) -> dict:
    value = copy.deepcopy(payload)
    value.pop("request_id", None)
    value.pop("target", None)
    value.pop("repair_provenance", None)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    old = json.loads((root / OLD).read_text(encoding="utf-8"))
    new = json.loads((root / NEW).read_text(encoding="utf-8"))

    if old.get("request_id") != OLD_REQUEST or new.get("request_id") != NEW_REQUEST:
        raise SystemExit("E03 request identity lineage mismatch")
    if old.get("target") != OLD_TARGET or new.get("target") != NEW_TARGET:
        raise SystemExit("E03 target lineage mismatch")
    if normalized(old) != normalized(new):
        raise SystemExit("E03 request 003 changed science/resource/output contract")

    repair = new.get("repair_provenance") or {}
    expected = {
        "repair_of_request_id": OLD_REQUEST,
        "prior_workflow_run_id": 34199556112,
        "prior_target": OLD_TARGET,
        "prior_failure_class": "kaggle_runtime_md5_manifest_entry_missing_on_staged_tree",
        "prior_terminal_status": "KERNELWORKERSTATUS.ERROR",
        "observed_missing_manifest_entry": "2025LJI_bulkBCR.tsv",
        "single_repair": "stage_full_competition_input_directory_for_md5_contract",
        "scientific_contract_changed": False,
        "resource_contract_changed": False,
    }
    if repair != expected:
        raise SystemExit("E03 request 003 repair provenance mismatch")
    if new.get("competition_submission_attempted") is not False:
        raise SystemExit("E03 request 003 unexpectedly enables submission")
    if new.get("automatic_compute_retries") != 0:
        raise SystemExit("E03 request 003 retry contract changed")

    print(
        "CMI_FLU_E03_MD5_STAGE_REPAIR_REQUEST PASS "
        "science_resource_output_unchanged=true single_repair=full_input_symlink"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
