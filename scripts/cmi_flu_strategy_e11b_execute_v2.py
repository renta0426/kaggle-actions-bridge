#!/usr/bin/env python3
"""Fresh E11b 002 executor; repair only Kaggle CLI JSON output flag."""
from __future__ import annotations

import hashlib
import json
import subprocess

import cmi_flu_strategy_e11b_execute as prior

REQUEST_ID = "20260910-cmi-flu-strategy-e11b-tabpfn3-002"
TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-002"
TITLE = "CMI Flu E11b TabPFN3 20260910 002"

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260910-cmi-flu-strategy-e11b-tabpfn3-001",
    "TARGET": "renta0426/cmi-flu-e11b-tabpfn3-20260910-001",
    "TITLE": "CMI Flu E11b TabPFN3 20260910 001",
    "MODEL_SOURCE": "prior-labsai/tabpfn-3/pytorch/default/1",
    "CHECKPOINT_FILENAME": "tabpfn-v3-regressor-v3_default.ckpt",
    "CHECKPOINT_BYTES": 233_289_807,
    "EXPECTED_VERSION": 1,
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(prior, key, None) != value:
        raise SystemExit(f"E11b executor-v2 ancestry changed:{key}")

# Retarget the inherited, already-reviewed transport. No science/model setting changes.
prior.REQUEST_ID = REQUEST_ID
prior.TARGET = TARGET
prior.TARGET_SLUG = TARGET.split("/", 1)[1]
prior.TITLE = TITLE


def model_access_preflight() -> None:
    """Read-only exact model-version file listing; CLI 2.2.4 uses --format json."""
    completed = subprocess.run(
        [
            "kaggle", "models", "instances", "versions", "files",
            prior.MODEL_SOURCE,
            "--format", "json",
            "--page-size", "20",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    stdout, stderr = completed.stdout, completed.stderr
    marker = {
        "request_id": REQUEST_ID,
        "operation": "model_instance_version_files",
        "model_source_sha256": hashlib.sha256(prior.MODEL_SOURCE.encode()).hexdigest(),
        "return_code": int(completed.returncode),
        "stdout_bytes": len(stdout.encode("utf-8", errors="replace")),
        "stdout_sha256": prior._digest_text(stdout),
        "stderr_bytes": len(stderr.encode("utf-8", errors="replace")),
        "stderr_sha256": prior._digest_text(stderr),
        "write_attempted": False,
        "cli_output_format": "--format json",
    }
    print("CMI_FLU_E11B_MODEL_ACCESS_RECEIPT " + json.dumps(marker, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("E11b model access preflight failed; no kernel write attempted")
    try:
        payload = json.loads(stdout)
    except Exception as exc:
        raise RuntimeError("E11b model access preflight returned non-JSON") from exc
    serialized = json.dumps(payload, sort_keys=True)
    if prior.CHECKPOINT_FILENAME not in serialized:
        raise RuntimeError("E11b model access preflight checkpoint not visible")
    if str(prior.CHECKPOINT_BYTES) not in serialized and f"{prior.CHECKPOINT_BYTES}.0" not in serialized:
        raise RuntimeError("E11b model access preflight checkpoint byte identity changed")
    print(
        "CMI_FLU_E11B_MODEL_ACCESS_PASS "
        f"model_source_sha256={hashlib.sha256(prior.MODEL_SOURCE.encode()).hexdigest()} "
        f"checkpoint={prior.CHECKPOINT_FILENAME} bytes={prior.CHECKPOINT_BYTES} write=false compute=false"
    )


def prewrite_guard(api) -> None:
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != prior.TARGET_SLUG or prior.TITLE != TITLE:
        raise RuntimeError("E11b v2 target identity contract changed")
    try:
        discovered = api.kernels_list(user=owner, search=prior.TARGET_SLUG, page_size=20) or []
    except Exception as exc:
        raise RuntimeError("E11b v2 duplicate sentinel unavailable") from exc
    if any(str(getattr(item, "ref", "")) == TARGET for item in discovered):
        raise RuntimeError("E11b v2 duplicate sentinel found exact target; write refused")


prior.model_access_preflight = model_access_preflight
prior.prewrite_guard = prewrite_guard


def main() -> int:
    return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
