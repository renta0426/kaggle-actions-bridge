#!/usr/bin/env python3
"""Build E03 from relayed exact science blobs and the proven frozen B2.1 release."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REQUEST_ID = "20260908-cmi-flu-strategy-e03-task11-optional-view-fusion-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-e03-task11-optional-view-fusion-20260908-001"
SCIENCE_COMMIT = "2eaf8fac9a632c67f32c4a0ed37b64eeac571387"
SCIENCE_TRANSPORT = "agent_relay_exact_blobs"
E03_BLOB = "e38fe01606dc5653ad3d1542f722726c1374b278"
TASK11_PRIOR_BLOB = "50d9a43604d2b75479b8f873a86a8daf9d5bd7a9"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
FROZEN_B21_PACKAGE_SHA256 = "87c317789b3b8fcbd5fce2ff8f74663d23c19d0fcdce0e05ba1e809ea7728cb2"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e03-task11-optional-view-001"
E03_PARTS = tuple(f"{PAYLOAD_ROOT}/strategy_e03.part{i:02d}" for i in range(6))
TASK11_PRIOR_PATH = "payloads/cmi-flu-task11-prior-immunity-001/task11_prior_immunity.py"
CONFIG_PATH = "payloads/cmi-flu-strategy-e01-paired-evaluation-001/baseline_b021_robust.yaml"
REQUEST_PATH = "requests/cmi-flu-strategy-e03-task11-optional-view-fusion-001.json"
CONDITIONS = (
    "b21",
    "hai_raw_fusion_w0.25",
    "hai_rank_fusion_w0.25",
    "hai_crossfit_residual_w0.25",
    "innate_flow_rank_fusion_w0.25",
    "innate_flow_crossfit_residual_w0.25",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_blob(data: bytes, expected: str, label: str) -> str:
    found = git_blob_sha(data)
    if found != expected:
        raise SystemExit(f"{label} relay blob mismatch: expected={expected} found={found}")
    text = data.decode("utf-8")
    compile(text, label, "exec")
    return text


def load_relays(root: Path) -> tuple[str, str, str]:
    e03 = require_blob(
        b"".join((root / rel).read_bytes() for rel in E03_PARTS),
        E03_BLOB,
        "cmi_flu/strategy_e03.py",
    )
    prior = require_blob(
        (root / TASK11_PRIOR_PATH).read_bytes(),
        TASK11_PRIOR_BLOB,
        "cmi_flu/task11_prior_immunity.py",
    )
    config = require_blob((root / CONFIG_PATH).read_bytes(), CONFIG_BLOB, "baseline_b021_robust.yaml")
    required = (
        'EXPERIMENT = "strategy_v2_e03_task11_optional_view_fusion"',
        'B21_MODEL_NAME = "pls_2"',
        'FUSION_WEIGHT = 0.25',
        'VIEW_ALPHA = 10.0',
        'RANDOM_SEED = 20260907',
        'MIN_FLOW_TOTAL_SUBJECTS = 24',
        'MIN_FLOW_TOTAL_STUDIES = 2',
        'MIN_VIEW_SOURCE_SUBJECTS = 8',
        'MAX_FLOW_FEATURES = 8',
        '"competition_submission_attempted": False',
    )
    if any(token not in e03 for token in required):
        raise SystemExit("E03 relayed source contract token missing")
    if "kaggle competitions submit" in e03 or "competition_submit" in e03:
        raise SystemExit("E03 relayed source contains submission path")
    return e03, prior, config


def validate_request(root: Path) -> None:
    request = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": COMPETITION,
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_repository": "renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": SCIENCE_TRANSPORT,
        "strategy_e03_blob_sha": E03_BLOB,
        "task11_prior_immunity_blob_sha": TASK11_PRIOR_BLOB,
        "task11_prior_immunity_relay_path": TASK11_PRIOR_PATH,
        "config_blob_sha": CONFIG_BLOB,
        "config_relay_path": CONFIG_PATH,
        "frozen_b21_package_sha256": FROZEN_B21_PACKAGE_SHA256,
        "frozen_b21_transport": "existing_task11_prior_runtime_release",
        "expected_kernel_version": 1,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"E03 request mismatch: {key}")
    if tuple(request.get("strategy_e03_relay_parts", ())) != E03_PARTS:
        raise SystemExit("E03 relay part manifest mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 45,
        "hard_timeout_minutes": 75,
        "max_active_runs": 1,
    }:
        raise SystemExit("E03 resource contract mismatch")
    if tuple(request.get("allowed_output_paths", ())) != (
        "bridge-result.json", "metrics.json", "summary.md"
    ):
        raise SystemExit("E03 output allowlist mismatch")
    contract = request.get("experiment_contract") or {}
    if tuple(contract.get("conditions", ())) != CONDITIONS:
        raise SystemExit("E03 condition contract mismatch")


def extract_frozen_b21(root: Path, work: Path) -> tuple[bytes, str]:
    e01_builder = load_module(root / "scripts/cmi_flu_strategy_e01_prepare.py", "cmi_e03_e01_builder")
    package, adapter = e01_builder.extract_frozen_runtime(root, work)
    found = hashlib.sha256(package).hexdigest()
    if found != FROZEN_B21_PACKAGE_SHA256:
        raise SystemExit(
            f"frozen B2.1 release mismatch: expected={FROZEN_B21_PACKAGE_SHA256} found={found}"
        )
    if not isinstance(adapter, str) or not adapter:
        raise SystemExit("B2.1 adapter source unavailable")
    compile(adapter, "cmi_flu_b21_runtime_adapter.py", "exec")
    return package, adapter


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_runtime(runtime: str, adapter: str) -> str:
    runtime = replace_once(
        runtime,
        "import argparse\nimport base64\n",
        "import argparse\nimport base64\nfrom dataclasses import replace\n",
        "runtime dataclass import",
    )
    runtime = replace_once(
        runtime,
        "PACKAGE_B64 = \"\"\"",
        f"B21_ADAPTER_SOURCE = {adapter!r}\nPACKAGE_B64 = \"\"\"",
        "runtime adapter constant",
    )
    runtime = replace_once(
        runtime,
        '    import cmi_flu  # noqa: F401\n    prior = types.ModuleType("cmi_flu.task11_prior_immunity")',
        '    import cmi_flu  # noqa: F401\n'
        '    adapter_ns = {}\n'
        '    exec(compile(B21_ADAPTER_SOURCE, "<b21_runtime_adapter>", "exec"), adapter_ns, adapter_ns)\n'
        '    adapter_install = adapter_ns.get("install")\n'
        '    if not callable(adapter_install):\n'
        '        raise BridgeContractError("b21_adapter_install_missing")\n'
        '    adapter_install()\n'
        '    prior = types.ModuleType("cmi_flu.task11_prior_immunity")',
        "runtime adapter install",
    )
    runtime = replace_once(
        runtime,
        '    (config_dir / "baseline_b021_robust.yaml").write_text(CONFIG_TEXT, encoding="utf-8")\n'
        '    (work / "artifacts").mkdir(exist_ok=True)\n',
        '    canonical = config_dir / "baseline_b021_robust.yaml"\n'
        '    canonical.write_text(CONFIG_TEXT, encoding="utf-8")\n'
        '    if git_blob_sha(canonical.read_bytes()) != CONFIG_BLOB:\n'
        '        raise BridgeContractError("runtime_config_blob_mismatch")\n'
        '    if CONFIG_TEXT.count("baseline: b021_taskwise_robust") != 1:\n'
        '        raise BridgeContractError("b021_config_anchor_changed")\n'
        '    compat = config_dir / "baseline_b02_transport_compat.yaml"\n'
        '    compat.write_text(CONFIG_TEXT.replace("baseline: b021_taskwise_robust", "baseline: b02_taskwise_compact", 1), encoding="utf-8")\n'
        '    (work / "artifacts").mkdir(exist_ok=True)\n',
        "runtime config compatibility",
    )
    runtime = replace_once(
        runtime,
        '        from cmi_flu.configuration import load_baseline_config\n'
        '        from cmi_flu.runner import load_inputs\n'
        '        config = load_baseline_config(work / "configs/baseline_b021_robust.yaml", repository_root=work)\n'
        '        inputs = load_inputs(config)\n',
        '        from cmi_flu.configuration import load_baseline_config\n'
        '        from cmi_flu.runner import load_inputs\n'
        '        canonical_config = work / "configs/baseline_b021_robust.yaml"\n'
        '        compat_config = work / "configs/baseline_b02_transport_compat.yaml"\n'
        '        config = load_baseline_config(compat_config, repository_root=work)\n'
        '        raw = dict(config.raw); raw["baseline"] = "b021_taskwise_robust"\n'
        '        config = replace(config, source_path=canonical_config, raw=raw, baseline="b021_taskwise_robust")\n'
        '        if not config.verify_md5 or str(config.section("selection").get("policy", "")) != "robust_v1":\n'
        '            raise BridgeContractError("runtime_config_contract_mismatch")\n'
        '        inputs = load_inputs(config)\n'
        '        if inputs.checksum_report is None:\n'
        '            raise BridgeContractError("md5_verification_missing")\n',
        "runtime robust config load",
    )
    runtime = replace_once(
        runtime,
        '    compile(E03_SOURCE, "cmi_flu/strategy_e03.py", "exec")\n',
        '    compile(B21_ADAPTER_SOURCE, "cmi_flu_b21_runtime_adapter.py", "exec")\n'
        '    compile(E03_SOURCE, "cmi_flu/strategy_e03.py", "exec")\n',
        "runtime adapter self-test",
    )
    compile(runtime, "generated_cmi_e03_runtime_v2.py", "exec")
    return runtime


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validate_request(root)
    e03, prior, config = load_relays(root)
    v1 = load_module(root / "scripts/cmi_flu_strategy_e03_prepare.py", "cmi_e03_prepare_v1")
    with tempfile.TemporaryDirectory(prefix="cmi-e03-relay-build-") as tmp:
        package, adapter = extract_frozen_b21(root, Path(tmp))
        runtime = v1.build_runtime(package, e03, prior, config)
        runtime = patch_runtime(runtime, adapter)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_E03_BUILD_V2 PASS "
        f"request_id={REQUEST_ID} science_commit={SCIENCE_COMMIT} science_transport={SCIENCE_TRANSPORT} "
        f"e03_blob={E03_BLOB} package_sha256={FROZEN_B21_PACKAGE_SHA256} "
        f"runtime_sha256={hashlib.sha256(runtime.encode('utf-8')).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
