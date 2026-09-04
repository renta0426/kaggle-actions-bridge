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

REQUEST_ID = "20260904-cmi-flu-rank-transfer-003"
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_COMMIT = "9e9922806209241121274f9242a6cf09bf140d75"
RANK_TRANSFER_BLOB = "d5e07cdd09d2eabdc935eb1733ec238e26ab4c17"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
BASE_SHA256 = "7894a9232e7372950f70a36d35de41bcc8bda90ec4d0320187e82c3b8c76f2db"
BASE_SIZE = 93363
TARGET_COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-rank-transfer-20260904-003"
TARGET_STAGE = "phase_a_rank_transfer_task11_task12"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


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

    request_path = root / "requests/cmi-flu-rank-transfer-003.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    allowed = {
        "schema_version", "request_id", "competition", "operation", "target", "purpose",
        "science_repository", "science_source_commit", "science_transport",
        "rank_transfer_blob_sha", "b21_base_request_id", "resource", "api_budget",
        "side_effects", "competition_submission_attempted", "automatic_compute_retries",
        "enable_internet", "rules_checked_at_utc",
    }
    if set(request) != allowed:
        raise SystemExit(f"manifest fields mismatch: {sorted(set(request) ^ allowed)}")
    if request["schema_version"] != 1 or request["request_id"] != REQUEST_ID:
        raise SystemExit("request identity mismatch")
    if request["competition"] != TARGET_COMPETITION or request["target"] != TARGET_KERNEL or request["operation"] != "kernel_run":
        raise SystemExit("target/operation mismatch")
    if request["science_repository"] != SCIENCE_REPOSITORY or request["science_source_commit"] != SCIENCE_COMMIT:
        raise SystemExit("science provenance mismatch")
    if request["science_transport"] != "agent_relay_exact_blob":
        raise SystemExit("science transport must be agent_relay_exact_blob")
    if request["rank_transfer_blob_sha"] != RANK_TRANSFER_BLOB:
        raise SystemExit("rank-transfer source provenance mismatch")
    if request["b21_base_request_id"] != "20260903-cmi-flu-b21-001":
        raise SystemExit("B2.1 lineage mismatch")
    if request["competition_submission_attempted"] is not False or request["automatic_compute_retries"] != 0 or request["enable_internet"] is not False:
        raise SystemExit("safety contract mismatch")
    if request["resource"] != {"accelerator": "cpu", "expected_runtime_minutes": 60, "hard_timeout_minutes": 180, "max_active_runs": 1}:
        raise SystemExit("resource contract mismatch")
    if request["api_budget"] != {"max_calls": 20, "poll_interval_seconds": 900, "max_pages": 2}:
        raise SystemExit("API budget mismatch")

    rank_path = root / "payloads/cmi-flu-rank-transfer-001/rank_transfer.py"
    rank_data = rank_path.read_bytes()
    if git_blob_sha(rank_data) != RANK_TRANSFER_BLOB:
        raise SystemExit("agent-relayed rank-transfer blob mismatch")
    compile(rank_data.decode("utf-8"), "cmi_flu/rank_transfer.py", "exec")

    adapter_path = root / "scripts/cmi_flu_b21_runtime_adapter.py"
    adapter = adapter_path.read_bytes()
    if git_blob_sha(adapter) != B21_ADAPTER_BLOB:
        raise SystemExit("B2.1 adapter blob mismatch")

    base_dir = root / "payloads/cmi-flu-b2-broad-004"
    base = b"".join((base_dir / f"part-{index:02d}").read_bytes() for index in range(20))
    if len(base) != BASE_SIZE or hashlib.sha256(base).hexdigest() != BASE_SHA256:
        raise SystemExit("B2 004 base payload mismatch")

    work = output.parent
    base_source = work / "base-source.py"
    base004 = work / "base004.py"
    b21 = work / "b21.py"
    base_source.write_bytes(base)
    run(sys.executable, str(root / "scripts/cmi_flu_b2_patch_v4.py"), "--source", str(base_source), "--request", str(root / "requests/cmi-flu-b2-launch-v4.json"), "--output", str(base004))
    run(sys.executable, str(root / "scripts/cmi_flu_b21_patch.py"), "--source", str(base004), "--adapter", str(adapter_path), "--request", str(root / "requests/cmi-flu-b21-launch-v1.json"), "--output", str(b21))
    run(sys.executable, str(root / "scripts/cmi_flu_rank_transfer_patch_v3.py"), "--source", str(b21), "--rank-source", str(rank_path), "--request", str(request_path), "--output", str(output))

    for path in (
        root / "scripts/cmi_flu_rank_transfer_patch.py",
        root / "scripts/cmi_flu_rank_transfer_patch_v2.py",
        root / "scripts/cmi_flu_rank_transfer_patch_v3.py",
        output,
    ):
        py_compile.compile(str(path), doraise=True)

    text = output.read_text(encoding="utf-8")
    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'SOURCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'TARGET_STAGE = "{TARGET_STAGE}"',
        f'RANK_TRANSFER_SOURCE_BLOB_SHA = "{RANK_TRANSFER_BLOB}"',
        "models.summarize_metric_frame = metric_summary",
        "CMI_FLU_RANK_TRANSFER_COMPLETE",
        "def json_safe(value: Any) -> Any:",
        'if config.baseline != "b02_taskwise_compact":',
        'raw_compat["baseline"] = "b021_taskwise_robust"',
        'object.__setattr__(config, "baseline", "b021_taskwise_robust")',
        'if str(config.section("selection").get("policy", "")) != "robust_v1":',
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"generated runtime missing tokens: {missing}")
    forbidden = '''config_text = config_text.replace(\n            "baseline: b02_taskwise_compact",\n            "baseline: b021_taskwise_robust",'''
    if forbidden in text:
        raise SystemExit("generated runtime still rewrites baseline before legacy loader")

    namespace: dict[str, Any] = {"__name__": "rank_transfer_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("generated runtime request identity mismatch")
    if namespace.get("SOURCE_COMMIT") != SCIENCE_COMMIT:
        raise SystemExit("generated runtime science commit mismatch")
    if namespace.get("RANK_TRANSFER_SOURCE_BLOB_SHA") != RANK_TRANSFER_BLOB:
        raise SystemExit("generated runtime relayed blob mismatch")
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
        "CMI_FLU_RANK_TRANSFER_003_PREPARE PASS "
        f"science_commit={SCIENCE_COMMIT} rank_blob={RANK_TRANSFER_BLOB} loader_compat=legacy_b02_then_promote_b021"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
