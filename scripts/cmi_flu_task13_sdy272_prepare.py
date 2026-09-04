#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import py_compile
import ssl
import subprocess
import sys
import urllib.request
from typing import Any

REQUEST_ID = "20260904-cmi-flu-task13-sdy272-001"
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_COMMIT = "baae9fb40329057a316935d0d1285adc64c948b0"
SCIENCE_PATH = "src/cmi_flu/task13_harmonization.py"
TASK13_BLOB = "0f1e728fe2e5ea0f3713c1442c3beeba21b8d347"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
BASE_SHA256 = "7894a9232e7372950f70a36d35de41bcc8bda90ec4d0320187e82c3b8c76f2db"
BASE_SIZE = 93363
TARGET_COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-task13-sdy272-20260904-001"
TARGET_STAGE = "phase_a_task13_sdy272_harmonization"
EXPECTED_KERNEL_VERSION = 1
ALLOWED_OUTPUT_PATHS = [
    "cmi-flu-task13-sdy272/bridge-result.json",
    "cmi-flu-task13-sdy272/metrics.json",
    "cmi-flu-task13-sdy272/summary.md",
]


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def fetch_science_source() -> bytes:
    url = (
        "https://raw.githubusercontent.com/"
        f"{SCIENCE_REPOSITORY}/{SCIENCE_COMMIT}/{SCIENCE_PATH}"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "kaggle-actions-bridge/1"},
    )
    with urllib.request.urlopen(
        request,
        timeout=45,
        context=ssl.create_default_context(),
    ) as response:
        data = response.read(65537)
    if len(data) > 65536:
        raise SystemExit("Task1.3 science source exceeds byte budget")
    if git_blob_sha(data) != TASK13_BLOB:
        raise SystemExit("pinned Task1.3 science source blob mismatch")
    compile(data.decode("utf-8"), SCIENCE_PATH, "exec")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    request_path = root / "requests/cmi-flu-task13-sdy272-001.json"
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
        "science_source_path",
        "science_transport",
        "task13_harmonization_blob_sha",
        "b21_base_request_id",
        "b21_runtime_adapter_blob_sha",
        "expected_kernel_version",
        "allowed_output_paths",
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
        or request["operation"] != "kernel_run_and_current_output_read"
    ):
        raise SystemExit("target/operation mismatch")
    if (
        request["science_repository"] != SCIENCE_REPOSITORY
        or request["science_source_commit"] != SCIENCE_COMMIT
        or request["science_source_path"] != SCIENCE_PATH
        or request["science_transport"] != "pinned_public_github_blob"
    ):
        raise SystemExit("science provenance/transport mismatch")
    if request["task13_harmonization_blob_sha"] != TASK13_BLOB:
        raise SystemExit("Task1.3 source provenance mismatch")
    if request["b21_base_request_id"] != "20260903-cmi-flu-b21-001":
        raise SystemExit("B2.1 lineage mismatch")
    if request["b21_runtime_adapter_blob_sha"] != B21_ADAPTER_BLOB:
        raise SystemExit("B2.1 adapter provenance mismatch")
    if request["expected_kernel_version"] != EXPECTED_KERNEL_VERSION:
        raise SystemExit("expected kernel version mismatch")
    if request["allowed_output_paths"] != ALLOWED_OUTPUT_PATHS:
        raise SystemExit("allowed output path contract mismatch")
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
        "max_calls": 30,
        "poll_interval_seconds": 900,
        "max_pages": 2,
    }:
        raise SystemExit("API budget mismatch")
    if request["side_effects"] != [
        "create one private Notebook version and start one CPU run",
        "after successful completion read only current version 1 aggregate outputs bridge-result.json metrics.json summary.md",
    ]:
        raise SystemExit("side-effect contract mismatch")

    task13_data = fetch_science_source()
    task13_path = output.parent / "task13_harmonization.py"
    task13_path.write_bytes(task13_data)

    adapter_path = root / "scripts/cmi_flu_b21_runtime_adapter.py"
    adapter = adapter_path.read_bytes()
    if git_blob_sha(adapter) != B21_ADAPTER_BLOB:
        raise SystemExit("B2.1 adapter blob mismatch")

    base_dir = root / "payloads/cmi-flu-b2-broad-004"
    base = b"".join(
        (base_dir / f"part-{index:02d}").read_bytes()
        for index in range(20)
    )
    if len(base) != BASE_SIZE or hashlib.sha256(base).hexdigest() != BASE_SHA256:
        raise SystemExit("B2 004 base payload mismatch")

    work = output.parent
    base_source = work / "base-source.py"
    base004 = work / "base004.py"
    b21 = work / "b21.py"
    base_source.write_bytes(base)
    run(
        sys.executable,
        str(root / "scripts/cmi_flu_b2_patch_v4.py"),
        "--source",
        str(base_source),
        "--request",
        str(root / "requests/cmi-flu-b2-launch-v4.json"),
        "--output",
        str(base004),
    )
    run(
        sys.executable,
        str(root / "scripts/cmi_flu_b21_patch.py"),
        "--source",
        str(base004),
        "--adapter",
        str(adapter_path),
        "--request",
        str(root / "requests/cmi-flu-b21-launch-v1.json"),
        "--output",
        str(b21),
    )
    run(
        sys.executable,
        str(root / "scripts/cmi_flu_task13_sdy272_patch.py"),
        "--source",
        str(b21),
        "--task13-source",
        str(task13_path),
        "--request",
        str(request_path),
        "--output",
        str(output),
    )

    for path in (
        root / "scripts/cmi_flu_task13_sdy272_patch.py",
        output,
    ):
        py_compile.compile(str(path), doraise=True)

    text = output.read_text(encoding="utf-8")
    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'SOURCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'TARGET_STAGE = "{TARGET_STAGE}"',
        f'TASK13_HARMONIZATION_SOURCE_BLOB_SHA = "{TASK13_BLOB}"',
        f'B21_ADAPTER_BLOB_SHA = "{B21_ADAPTER_BLOB}"',
        'default=Path("/kaggle/working/cmi-flu-task13-sdy272")',
        'if config.baseline != "b02_taskwise_compact":',
        'object.__setattr__(config, "baseline", "b021_taskwise_robust")',
        'if str(config.section("selection").get("policy", "")) != "robust_v1":',
        'if str(config.section("flow").get("task_13_mode", "")) != "broad":',
        "run_task13_sdy272_harmonization_experiment",
        "CMI_FLU_TASK13_SDY272_COMPLETE",
        '"competition_submission_attempted": False',
        "shutil.rmtree(runtime_root, ignore_errors=True)",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"generated runtime missing tokens: {missing}")
    forbidden = (
        "kaggle competitions submit",
        "api.competition_submit",
        'default=Path("/kaggle/working/cmi-flu-b2")',
    )
    present = [token for token in forbidden if token in text]
    if present:
        raise SystemExit(f"generated runtime contains forbidden tokens: {present}")
    if text.count("shutil.rmtree(runtime_root, ignore_errors=True)") != 2:
        raise SystemExit("runtime scratch cleanup must cover success and failure paths")

    namespace: dict[str, Any] = {"__name__": "task13_sdy272_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("generated runtime request identity mismatch")
    if namespace.get("SOURCE_COMMIT") != SCIENCE_COMMIT:
        raise SystemExit("generated runtime science commit mismatch")
    if namespace.get("TARGET_STAGE") != TARGET_STAGE:
        raise SystemExit("generated runtime stage identity mismatch")
    if namespace.get("TASK13_HARMONIZATION_SOURCE_BLOB_SHA") != TASK13_BLOB:
        raise SystemExit("generated runtime Task1.3 source mismatch")
    helper = namespace.get("task13_json_safe")
    if not callable(helper):
        raise SystemExit("generated runtime JSON safety helper missing")
    probe = helper({"nan": float("nan"), "values": (1.5, 2)})
    if probe != {"nan": None, "values": [1.5, 2]}:
        raise SystemExit(f"task13_json_safe smoke mismatch: {probe!r}")

    run(sys.executable, str(output), "--self-test")
    print(
        "CMI_FLU_TASK13_SDY272_001_PREPARE PASS "
        f"science_commit={SCIENCE_COMMIT} task13_blob={TASK13_BLOB} "
        f"b21_adapter_blob={B21_ADAPTER_BLOB} expected_version=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
