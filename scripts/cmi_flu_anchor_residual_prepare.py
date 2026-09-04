#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import py_compile
import subprocess
import sys
from typing import Any

REQUEST_ID = "20260904-cmi-flu-anchor-residual-001"
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_COMMIT = "23ab4ff53d65eeb8b8e5582f5442081f245f03b3"
ANCHOR_RESIDUAL_BLOB = "9a7814ecdf70e0b38b2740ade21e9db588869379"
RANK_TRANSFER_BLOB = "d5e07cdd09d2eabdc935eb1733ec238e26ab4c17"
RANK_TRANSFER_BASE_REQUEST_ID = "20260904-cmi-flu-rank-transfer-003"
B21_BASE_REQUEST_ID = "20260903-cmi-flu-b21-001"
TARGET_COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-anchor-residual-20260904-001"
TARGET_STAGE = "phase_a_anchor_residual_task11_task12"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    request_path = root / "requests/cmi-flu-anchor-residual-001.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    allowed = {
        "schema_version",
        "request_id",
        "competition",
        "operation",
        "target",
        "purpose",
        "science_repository",
        "science_source_commit",
        "science_transport",
        "anchor_residual_blob_sha",
        "rank_transfer_base_request_id",
        "rank_transfer_blob_sha",
        "b21_base_request_id",
        "resource",
        "api_budget",
        "side_effects",
        "competition_submission_attempted",
        "automatic_compute_retries",
        "enable_internet",
        "rules_checked_at_utc",
    }
    if set(request) != allowed:
        raise SystemExit(f"manifest fields mismatch: {sorted(set(request) ^ allowed)}")
    if request["schema_version"] != 1 or request["request_id"] != REQUEST_ID:
        raise SystemExit("request identity mismatch")
    if (
        request["competition"] != TARGET_COMPETITION
        or request["target"] != TARGET_KERNEL
        or request["operation"] != "kernel_run"
    ):
        raise SystemExit("target/operation mismatch")
    if (
        request["science_repository"] != SCIENCE_REPOSITORY
        or request["science_source_commit"] != SCIENCE_COMMIT
        or request["science_transport"] != "agent_relay_exact_blob"
    ):
        raise SystemExit("science provenance mismatch")
    if request["anchor_residual_blob_sha"] != ANCHOR_RESIDUAL_BLOB:
        raise SystemExit("anchor-residual source provenance mismatch")
    if request["rank_transfer_base_request_id"] != RANK_TRANSFER_BASE_REQUEST_ID:
        raise SystemExit("rank-transfer runtime lineage mismatch")
    if request["rank_transfer_blob_sha"] != RANK_TRANSFER_BLOB:
        raise SystemExit("rank-transfer dependency provenance mismatch")
    if request["b21_base_request_id"] != B21_BASE_REQUEST_ID:
        raise SystemExit("B2.1 lineage mismatch")
    if (
        request["competition_submission_attempted"] is not False
        or request["automatic_compute_retries"] != 0
        or request["enable_internet"] is not False
    ):
        raise SystemExit("safety contract mismatch")
    if request["resource"] != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 90,
        "hard_timeout_minutes": 180,
        "max_active_runs": 1,
    }:
        raise SystemExit("resource contract mismatch")
    if request["api_budget"] != {
        "max_calls": 20,
        "poll_interval_seconds": 900,
        "max_pages": 2,
    }:
        raise SystemExit("API budget mismatch")
    if request["side_effects"] != [
        "create one private Notebook version and start one CPU run"
    ]:
        raise SystemExit("side-effect contract mismatch")

    anchor_path = root / "payloads/cmi-flu-anchor-residual-001/anchor_residual.py"
    anchor_data = anchor_path.read_bytes()
    if git_blob_sha(anchor_data) != ANCHOR_RESIDUAL_BLOB:
        raise SystemExit("agent-relayed anchor-residual blob mismatch")
    compile(anchor_data.decode("utf-8"), "cmi_flu/anchor_residual.py", "exec")

    rank_path = root / "payloads/cmi-flu-rank-transfer-001/rank_transfer.py"
    rank_data = rank_path.read_bytes()
    if git_blob_sha(rank_data) != RANK_TRANSFER_BLOB:
        raise SystemExit("rank-transfer dependency blob mismatch")
    compile(rank_data.decode("utf-8"), "cmi_flu/rank_transfer.py", "exec")

    work = output.parent
    rank_runtime = work / "rank-runtime.py"
    anchor_v1 = work / "anchor-v1.py"
    run(
        sys.executable,
        str(root / "scripts/cmi_flu_rank_transfer_prepare_v3.py"),
        "--repository-root",
        str(root),
        "--output",
        str(rank_runtime),
    )
    run(
        sys.executable,
        str(root / "scripts/cmi_flu_anchor_residual_patch.py"),
        "--source",
        str(rank_runtime),
        "--anchor-source",
        str(anchor_path),
        "--request",
        str(request_path),
        "--output",
        str(anchor_v1),
    )
    run(
        sys.executable,
        str(root / "scripts/cmi_flu_anchor_residual_patch_v2.py"),
        "--source",
        str(anchor_v1),
        "--output",
        str(output),
    )

    for path in (
        root / "scripts/cmi_flu_anchor_residual_patch.py",
        root / "scripts/cmi_flu_anchor_residual_patch_v2.py",
        output,
    ):
        py_compile.compile(str(path), doraise=True)

    text = output.read_text(encoding="utf-8")
    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'SOURCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'TARGET_STAGE = "{TARGET_STAGE}"',
        f'ANCHOR_RESIDUAL_SOURCE_BLOB_SHA = "{ANCHOR_RESIDUAL_BLOB}"',
        f'RANK_TRANSFER_SOURCE_BLOB_SHA = "{RANK_TRANSFER_BLOB}"',
        "models.summarize_metric_frame = metric_summary",
        "CMI_FLU_ANCHOR_RESIDUAL_COMPLETE",
        "def json_safe(value: Any) -> Any:",
        'if config.baseline != "b02_taskwise_compact":',
        'raw_compat["baseline"] = "b021_taskwise_robust"',
        'object.__setattr__(config, "baseline", "b021_taskwise_robust")',
        'if str(config.section("selection").get("policy", "")) != "robust_v1":',
        'weights != (0.25, 0.5, 1.0)',
        '"competition_submission_attempted": False',
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"generated runtime missing tokens: {missing}")
    forbidden = (
        'TARGET_STAGE = "phase_a_rank_transfer_task11_task12"',
        "kaggle competitions submit",
        "api.competition_submit",
    )
    present = [token for token in forbidden if token in text]
    if present:
        raise SystemExit(f"generated runtime contains forbidden tokens: {present}")

    namespace: dict[str, Any] = {"__name__": "anchor_residual_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("generated runtime request identity mismatch")
    if namespace.get("SOURCE_COMMIT") != SCIENCE_COMMIT:
        raise SystemExit("generated runtime science commit mismatch")
    if namespace.get("TARGET_STAGE") != TARGET_STAGE:
        raise SystemExit("generated runtime stage identity mismatch")
    if namespace.get("ANCHOR_RESIDUAL_SOURCE_BLOB_SHA") != ANCHOR_RESIDUAL_BLOB:
        raise SystemExit("generated runtime relayed anchor blob mismatch")
    if namespace.get("RANK_TRANSFER_SOURCE_BLOB_SHA") != RANK_TRANSFER_BLOB:
        raise SystemExit("generated runtime rank dependency mismatch")
    config_text = namespace.get("CONFIG_TEXT")
    if not isinstance(config_text, str) or "baseline: b02_taskwise_compact" not in config_text:
        raise SystemExit("embedded base config no longer has legacy loader-compatible baseline")
    helper = namespace.get("json_safe")
    if not callable(helper):
        raise SystemExit("generated runtime json_safe missing")
    probe = helper({"nan": float("nan"), "values": (1.5, 2)})
    if probe != {"nan": None, "values": [1.5, 2]}:
        raise SystemExit(f"json_safe smoke mismatch: {probe!r}")

    run(sys.executable, str(output), "--self-test")
    print(
        "CMI_FLU_ANCHOR_RESIDUAL_001_PREPARE PASS "
        f"science_commit={SCIENCE_COMMIT} anchor_blob={ANCHOR_RESIDUAL_BLOB} "
        f"rank_blob={RANK_TRANSFER_BLOB} loader_compat=legacy_b02_then_promote_b021"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
