#!/usr/bin/env python3
"""E10 repair builder: exact five-subject science relay on proven E10-v2 ancestry."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import cmi_flu_strategy_e10_prepare as prior

base = prior.base

REQUEST_ID = "20260909-cmi-flu-strategy-e10-pooled-domain-correction-002"
TARGET_KERNEL = "renta0426/cmi-flu-e10-pooled-domain-correction-20260909-002"
SCIENCE_COMMIT = "3a8728879179115487f5ce0ba4a4a0467e1910bb"
E10_BLOB = "638b09860f85689714cb0b354f4f377a405deeac"
E10_SYNTH_BLOB = "f1f33f3909e64e69fb859d98b4163cdb4860d64d"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e10-pooled-domain-correction-002"
REQUEST_PATH = "requests/cmi-flu-strategy-e10-pooled-domain-correction-002.json"
MIN_SOURCE_SUBJECTS = 5

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260909-cmi-flu-strategy-e10-pooled-domain-correction-001",
    "TARGET_KERNEL": "renta0426/cmi-flu-e10-pooled-domain-correction-20260909-001",
    "SCIENCE_COMMIT": "9e75a6d547d38e369620cdf12bbc6af5e74d1ba8",
    "E10_BLOB": "f7b6b05b8b8f0061a5a378256b729f230960fe1c",
    "E10_SYNTH_BLOB": "5859fd21dfa2b9788090ca8a5a2076f4059cd1eb",
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(base, key, None) != value:
        raise SystemExit(f"E10 repair builder ancestry changed:{key}")

_original_validate_request = base.validate_request
_original_patch_runtime = base.patch_runtime

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
    """Reuse the old request validator after isolating the one frozen repair field."""
    request = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    contract = request.get("experiment_contract") or {}
    if contract.get("minimum_source_subjects") != MIN_SOURCE_SUBJECTS:
        raise SystemExit("E10 repair request minimum_source_subjects mismatch")
    if request.get("automatic_compute_retries") != 0:
        raise SystemExit("E10 repair automatic retry boundary changed")
    if request.get("competition_submission_attempted") is not False:
        raise SystemExit("E10 repair submission boundary changed")
    if request.get("leaderboard_used_for_selection") is not False:
        raise SystemExit("E10 repair leaderboard-selection boundary changed")

    compatibility = copy.deepcopy(request)
    compatibility["experiment_contract"]["minimum_source_subjects"] = 8
    with tempfile.TemporaryDirectory(prefix="cmi-flu-e10-repair-request-") as tmp:
        tmp_root = Path(tmp)
        target = tmp_root / REQUEST_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(compatibility, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _original_validate_request(tmp_root)


def patch_runtime(v3, runtime: str, e10: str, synth: str, anchor: str, rank_transfer: str, study_similarity: str) -> str:
    """Patch only the implementation guardrail mirrored by the frozen science repair."""
    runtime = _original_patch_runtime(
        v3,
        runtime,
        e10,
        synth,
        anchor,
        rank_transfer,
        study_similarity,
    )
    frozen_old = '        "minimum_source_subjects": 8,\n'
    frozen_new = '        "minimum_source_subjects": 5,\n'
    if runtime.count(frozen_old) != 1:
        raise SystemExit("E10 repair frozen-condition runtime anchor changed")
    runtime = runtime.replace(frozen_old, frozen_new, 1)

    support_old = "min(int(v) for v in counts.values()) < 8"
    support_new = "min(int(v) for v in counts.values()) < 5"
    if runtime.count(support_old) != 1:
        raise SystemExit("E10 repair source-support runtime anchor changed")
    runtime = runtime.replace(support_old, support_new, 1)

    synthetic_old = '''        if result.get("held_outcomes_used") is not False:
            raise BridgeContractError("e10_synthetic_xonly_boundary")
'''
    synthetic_new = '''        if result.get("held_outcomes_used") is not False:
            raise BridgeContractError("e10_synthetic_xonly_boundary")
        if result.get("minimum_source_counts") != {"A": 5, "B": 8} or result.get("minimum_source_contract_supported") is not True:
            raise BridgeContractError("e10_synthetic_five_subject_contract")
'''
    if runtime.count(synthetic_old) != 1:
        raise SystemExit("E10 repair synthetic runtime anchor changed")
    runtime = runtime.replace(synthetic_old, synthetic_new, 1)

    compile(runtime, "generated_e10_repair_runtime.py", "exec")
    return runtime


base.validate_request = validate_request
base.patch_runtime = patch_runtime


def main() -> int:
    return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
