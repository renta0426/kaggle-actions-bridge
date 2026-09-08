#!/usr/bin/env python3
"""Regression checks for the generated CMI-Flu E03 runtime and output boundary."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

EXPECTED_REQUEST = "20260908-cmi-flu-strategy-e03-task11-optional-view-fusion-001"
EXPECTED_SCIENCE = "2eaf8fac9a632c67f32c4a0ed37b64eeac571387"
EXPECTED_PACKAGE = "87c317789b3b8fcbd5fce2ff8f74663d23c19d0fcdce0e05ba1e809ea7728cb2"
EXPECTED_CONDITIONS = (
    "b21",
    "hai_raw_fusion_w0.25",
    "hai_rank_fusion_w0.25",
    "hai_crossfit_residual_w0.25",
    "innate_flow_rank_fusion_w0.25",
    "innate_flow_crossfit_residual_w0.25",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    args = parser.parse_args()
    runtime = args.runtime.expanduser().resolve()
    text = runtime.read_text(encoding="utf-8")
    compile(text, str(runtime), "exec")
    if "kaggle competitions submit" in text or "competition_submit(" in text:
        raise SystemExit("generated E03 runtime contains a competition submission path")
    subprocess.run([sys.executable, str(runtime), "--self-test"], check=True)

    spec = importlib.util.spec_from_file_location("cmi_e03_runtime_contract", runtime)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to import generated E03 runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.REQUEST_ID != EXPECTED_REQUEST:
        raise SystemExit("E03 runtime request identity mismatch")
    if module.SCIENCE_COMMIT != EXPECTED_SCIENCE:
        raise SystemExit("E03 runtime science commit mismatch")
    if module.PACKAGE_SHA256 != EXPECTED_PACKAGE:
        raise SystemExit("E03 frozen B2.1 package mismatch")
    if tuple(module.CONDITIONS) != EXPECTED_CONDITIONS:
        raise SystemExit("E03 condition ordering mismatch")
    if tuple(module.ALLOWED_OUTPUTS) != ("bridge-result.json", "metrics.json", "summary.md"):
        raise SystemExit("E03 output allowlist mismatch")

    module.scan_output({"safe": {"held_study": "S1", "value": 0.5}})
    for payload in (
        {"participant_id": "forbidden"},
        {"nested": {"row_predictions": [0.1, 0.2]}},
        {"predicted_values": [1.0]},
    ):
        try:
            module.scan_output(payload)
        except module.BridgeContractError:
            pass
        else:
            raise SystemExit(f"E03 sanitizer accepted forbidden payload: {payload}")

    print(
        "CMI_FLU_E03_RUNTIME_CONTRACT PASS "
        f"request_id={EXPECTED_REQUEST} science_commit={EXPECTED_SCIENCE} "
        f"package_sha256={EXPECTED_PACKAGE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
