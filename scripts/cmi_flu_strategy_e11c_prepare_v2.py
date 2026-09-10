#!/usr/bin/env python3
"""E11c prepare wrapper for connector-verified private-science Git-blob relay."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cmi_flu_strategy_e11c_prepare as prior

NEW_TRANSPORT = "connector_verified_private_science_git_blob_relay_and_runtime_embedding"
OLD_TRANSPORT = "exact_commit_archive_plus_git_blob_verification_and_runtime_embedding"
PAYLOADS = {
    "src/cmi_flu/strategy_e11c.py": "payloads/cmi-flu-strategy-e11c-robust-tabpfn-001/strategy_e11c.py",
    "src/cmi_flu/strategy_e11c_synthetic.py": "payloads/cmi-flu-strategy-e11c-robust-tabpfn-001/strategy_e11c_synthetic.py",
    "src/cmi_flu/strategy_e11b.py": "payloads/cmi-flu-strategy-e11b-tabpfn3-001/strategy_e11b.py",
    "src/cmi_flu/strategy_e11b_synthetic.py": "payloads/cmi-flu-strategy-e11b-tabpfn3-001/strategy_e11b_synthetic.py",
    "configs/baseline_b021_robust.yaml": "payloads/cmi-flu-strategy-e11c-robust-tabpfn-001/baseline_b021_robust.yaml",
    "configs/strategy_e11c_task11_robust_tabpfn.json": "payloads/cmi-flu-strategy-e11c-robust-tabpfn-001/strategy_e11c_task11_robust_tabpfn.json",
}
_ORIGINAL_VALIDATE = prior.validate_request


def validate_relay_request(root: Path) -> None:
    request_path = root / prior.REQUEST_PATH
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("science_transport") != NEW_TRANSPORT:
        raise SystemExit("E11c relay science transport mismatch")
    if request.get("science_payloads") != PAYLOADS:
        raise SystemExit("E11c relay payload map mismatch")
    preflight = request.get("rules_preflight") or {}
    if preflight != {
        "required_before_any_kernel_write": True,
        "method": "inherited_base.live_rules_authenticated_read",
        "failure_write": False,
        "record_in_run_log": True,
    }:
        raise SystemExit("E11c live Rules preflight contract mismatch")
    adapted = dict(request)
    adapted["science_transport"] = OLD_TRANSPORT
    adapted.pop("science_payloads", None)
    adapted.pop("rules_preflight", None)
    with tempfile.TemporaryDirectory(prefix="cmi-flu-e11c-v1-request-") as tmp:
        temp_root = Path(tmp)
        target = temp_root / prior.REQUEST_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(adapted, indent=2) + "\n", encoding="utf-8")
        _ORIGINAL_VALIDATE(temp_root)


def main() -> int:
    prior.validate_request = validate_relay_request
    try:
        return prior.main()
    finally:
        prior.validate_request = _ORIGINAL_VALIDATE


if __name__ == "__main__":
    raise SystemExit(main())
