#!/usr/bin/env python3
"""Manual E11c-002 successor builder after the pre-write CPU admission block.

Science, model identity, CV, promotion gates, source-attachment normalization,
and dependency transport are byte/field identical to E11c-001.  Only the
request/target identity changes, with explicit provenance that 001 stopped
before SaveKernel and produced no scientific result.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cmi_flu_strategy_e11c_prepare as base
import cmi_flu_strategy_e11c_prepare_v2 as relay

REQUEST_ID = "20260910-cmi-flu-strategy-e11c-robust-tabpfn-002"
TARGET = "renta0426/cmi-flu-e11c-robust-tabpfn-20260910-002"
REQUEST_PATH = "requests/cmi-flu-strategy-e11c-robust-tabpfn-002.json"
PARENT_REQUEST_PATH = "requests/cmi-flu-strategy-e11c-robust-tabpfn-001.json"
PARENT_REQUEST_ID = "20260910-cmi-flu-strategy-e11c-robust-tabpfn-001"
PARENT_TARGET = "renta0426/cmi-flu-e11c-robust-tabpfn-20260910-001"

_ORIGINAL = {
    "REQUEST_ID": base.REQUEST_ID,
    "TARGET_KERNEL": base.TARGET_KERNEL,
    "REQUEST_PATH": base.REQUEST_PATH,
    "validate_request": base.validate_request,
}
_ORIGINAL_VALIDATE = base.validate_request


def _validate_successor(root: Path) -> None:
    request = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    parent = json.loads((root / PARENT_REQUEST_PATH).read_text(encoding="utf-8"))

    if parent.get("request_id") != PARENT_REQUEST_ID or parent.get("target") != PARENT_TARGET:
        raise SystemExit("E11c-002 parent request identity changed")
    if request.get("request_id") != REQUEST_ID or request.get("target") != TARGET:
        raise SystemExit("E11c-002 request/target identity changed")
    if request.get("parent_request_id") != PARENT_REQUEST_ID:
        raise SystemExit("E11c-002 parent linkage changed")
    if request.get("science_transport") != relay.NEW_TRANSPORT:
        raise SystemExit("E11c-002 science transport changed")
    if request.get("science_payloads") != relay.PAYLOADS:
        raise SystemExit("E11c-002 science payload map changed")
    if request.get("rules_preflight") != {
        "required_before_any_kernel_write": True,
        "method": "inherited_base.live_rules_authenticated_read",
        "failure_write": False,
        "record_in_run_log": True,
    }:
        raise SystemExit("E11c-002 Rules preflight changed")

    provenance = request.get("manual_successor_provenance") or {}
    if provenance != {
        "prior_request_id": PARENT_REQUEST_ID,
        "prior_actions_run": 34474957373,
        "prior_actions_job": 102863653370,
        "failure_classification": "resource_concurrency_admission_prewrite",
        "model_access_preflight_succeeded": True,
        "fresh_target_absence_confirmed": True,
        "savekernel_attempted": False,
        "kernel_version_consumed": False,
        "compute_started": False,
        "scientific_result_produced": False,
        "automatic_retry": False,
        "science_changed": False,
    }:
        raise SystemExit("E11c-002 admission-failure provenance changed")

    # Prove the successor is field-identical to 001 except for the new
    # request lineage/target and the explicit no-write failure provenance.
    normalized = dict(request)
    normalized.pop("manual_successor_provenance", None)
    normalized["request_id"] = parent["request_id"]
    normalized["parent_request_id"] = parent["parent_request_id"]
    normalized["target"] = parent["target"]
    if normalized != parent:
        changed = sorted(
            key for key in set(normalized) | set(parent)
            if normalized.get(key) != parent.get(key)
        )
        raise SystemExit("E11c-002 changed frozen parent fields:" + ",".join(changed))

    # Reuse the already-audited E11c-001 validator after adapting only the
    # lineage field that is intentionally new in this manual successor.
    adapted = dict(request)
    adapted.pop("manual_successor_provenance", None)
    adapted["parent_request_id"] = base.OLD_REQUEST
    adapted["science_transport"] = relay.OLD_TRANSPORT
    adapted.pop("science_payloads", None)
    adapted.pop("rules_preflight", None)
    with tempfile.TemporaryDirectory(prefix="cmi-flu-e11c-002-request-") as tmp:
        temp_root = Path(tmp)
        target = temp_root / REQUEST_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(adapted, indent=2) + "\n", encoding="utf-8")
        _ORIGINAL_VALIDATE(temp_root)


def main() -> int:
    base.REQUEST_ID = REQUEST_ID
    base.TARGET_KERNEL = TARGET
    base.REQUEST_PATH = REQUEST_PATH
    base.validate_request = _validate_successor
    try:
        return base.main()
    finally:
        base.REQUEST_ID = _ORIGINAL["REQUEST_ID"]
        base.TARGET_KERNEL = _ORIGINAL["TARGET_KERNEL"]
        base.REQUEST_PATH = _ORIGINAL["REQUEST_PATH"]
        base.validate_request = _ORIGINAL["validate_request"]


if __name__ == "__main__":
    raise SystemExit(main())
