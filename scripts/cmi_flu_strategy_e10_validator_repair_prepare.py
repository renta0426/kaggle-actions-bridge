#!/usr/bin/env python3
"""Fresh E10-003 builder: repair aggregate-validator runtime imports only."""
from __future__ import annotations

import json
from pathlib import Path

import cmi_flu_strategy_e10_repair_prepare as prior

base = prior.base

REQUEST_ID = "20260910-cmi-flu-strategy-e10-pooled-domain-correction-003"
TARGET_KERNEL = "renta0426/cmi-flu-e10-pooled-domain-correction-20260910-003"
SCIENCE_COMMIT = "3a8728879179115487f5ce0ba4a4a0467e1910bb"
E10_BLOB = "638b09860f85689714cb0b354f4f377a405deeac"
E10_SYNTH_BLOB = "f1f33f3909e64e69fb859d98b4163cdb4860d64d"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e10-pooled-domain-correction-002"
REQUEST_PATH = "requests/cmi-flu-strategy-e10-pooled-domain-correction-003.json"
PARENT_REQUEST_PATH = "requests/cmi-flu-strategy-e10-pooled-domain-correction-002.json"
MIN_SOURCE_SUBJECTS = 5

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260909-cmi-flu-strategy-e10-pooled-domain-correction-002",
    "TARGET_KERNEL": "renta0426/cmi-flu-e10-pooled-domain-correction-20260909-002",
    "SCIENCE_COMMIT": SCIENCE_COMMIT,
    "E10_BLOB": E10_BLOB,
    "E10_SYNTH_BLOB": E10_SYNTH_BLOB,
    "PAYLOAD_ROOT": PAYLOAD_ROOT,
    "MIN_SOURCE_SUBJECTS": MIN_SOURCE_SUBJECTS,
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(prior, key, None) != value:
        raise SystemExit(f"E10 validator-repair builder ancestry changed:{key}")

_inherited_patch_runtime = base.patch_runtime

# Retarget the v1/v2 builder globals to a fresh target while reusing the exact
# already-reviewed E10-002 science payload. The science blobs themselves are unchanged.
base.REQUEST_ID = REQUEST_ID
base.TARGET_KERNEL = TARGET_KERNEL
base.SCIENCE_COMMIT = SCIENCE_COMMIT
base.E10_BLOB = E10_BLOB
base.E10_SYNTH_BLOB = E10_SYNTH_BLOB
base.PAYLOAD_ROOT = PAYLOAD_ROOT
base.E10_PATH = f"{PAYLOAD_ROOT}/strategy_e10.py"
base.E10_SYNTH_PATH = f"{PAYLOAD_ROOT}/strategy_e10_synthetic.py"
base.REQUEST_PATH = REQUEST_PATH


def validate_request(root: Path) -> None:
    request = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    parent = json.loads((root / PARENT_REQUEST_PATH).read_text(encoding="utf-8"))

    exact = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "parent_request_id": parent["request_id"],
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_repository": "renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": parent["science_transport"],
        "strategy_e10_blob_sha": E10_BLOB,
        "strategy_e10_synthetic_blob_sha": E10_SYNTH_BLOB,
        "expected_kernel_version": 1,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
    }
    for key, value in exact.items():
        if request.get(key) != value:
            raise SystemExit(f"E10 validator-repair request mismatch:{key}")

    for key in ("dependency_blobs", "resource", "api_budget", "allowed_output_paths", "experiment_contract"):
        if request.get(key) != parent.get(key):
            raise SystemExit(f"E10 validator-repair scientific/runtime contract drift:{key}")
    if request.get("side_effects") != [
        "create exactly one fresh private CPU Notebook version and read only its current aggregate outputs"
    ]:
        raise SystemExit("E10 validator-repair side-effect contract changed")
    provenance = request.get("repair_provenance") or {}
    expected_provenance = {
        "consumed_failed_target": parent["target"],
        "consumed_failed_version": 1,
        "failed_actions_run": 34364922051,
        "failed_actions_job": 102511311607,
        "failure_stage": "validate_e10",
        "failure_exception_type": "NameError",
        "failure_error_code": "05e03c78697d5df8e699",
        "root_cause": "generated aggregate validator referenced numpy as np without a guaranteed runtime import",
        "repair_scope": "bridge_validator_runtime_imports_only",
    }
    if provenance != expected_provenance:
        raise SystemExit("E10 validator-repair provenance mismatch")


def patch_runtime(v3, runtime: str, e10: str, synth: str, anchor: str, rank_transfer: str, study_similarity: str) -> str:
    runtime = _inherited_patch_runtime(
        v3,
        runtime,
        e10,
        synth,
        anchor,
        rank_transfer,
        study_similarity,
    )

    # Demonstrated E10-002 failure: the full aggregate validator referenced
    # `np` without a guaranteed module-level binding. It also references math.
    # Explicitly provide both runtime imports; no science source is changed.
    import_anchor = "import hashlib\n"
    if runtime.count(import_anchor) != 1:
        raise SystemExit("E10 validator-repair import anchor changed")
    runtime = runtime.replace(import_anchor, import_anchor + "import math\nimport numpy as np\n", 1)

    if "import numpy as np\n" not in runtime or "import math\n" not in runtime:
        raise SystemExit("E10 validator-repair imports missing")
    if '"minimum_source_subjects": 5' not in runtime:
        raise SystemExit("E10 validator-repair five-subject contract missing")
    if "kaggle competitions submit" in runtime or "competition_submit" in runtime:
        raise SystemExit("E10 validator-repair generated runtime contains submission path")
    compile(runtime, "generated_e10_validator_repair_runtime.py", "exec")
    return runtime


base.validate_request = validate_request
base.patch_runtime = patch_runtime


def main() -> int:
    return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
