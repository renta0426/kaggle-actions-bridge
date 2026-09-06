#!/usr/bin/env python3
"""Build the one-shot aggregate-only Kaggle runtime for CMI-Flu strategy E01."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REQUEST_ID = "20260907-cmi-flu-strategy-e01-paired-evaluation-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-e01-paired-eval-20260907-001"
SCIENCE_COMMIT = "0b2ecb47eaa09f22450424c9c06dc88cf44bc1fb"
E01_BLOB = "dd27aea0cf97d41bad3cec64819c4c4269d94cbd"
E01_V2_BLOB = "8cc64dc5ab9483d5957cfada18d445188566c56c"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
TASK11_PRIOR_BLOB = "50d9a43604d2b75479b8f873a86a8daf9d5bd7a9"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e01-paired-evaluation-001"
E01_PARTS = tuple(f"{PAYLOAD_ROOT}/strategy_e01.part{i:02d}" for i in range(6))
E01_V2_PATH = f"{PAYLOAD_ROOT}/strategy_e01_v2.py"
CONFIG_PATH = f"{PAYLOAD_ROOT}/baseline_b021_robust.yaml"
TASK11_PRIOR_PATH = "payloads/cmi-flu-task11-prior-immunity-001/task11_prior_immunity.py"
REQUEST_PATH = "requests/cmi-flu-strategy-e01-paired-evaluation-001.json"

REFERENCE_PLACEHOLDERS = {
    "cytokine_name_map.csv": "source,target\n",
    "flow_name_revised.csv": "source,target\n",
    "hai_map.csv": "source,target\n",
    "strain_sequences.csv": "virus_strain,sequence\n",
    "vaccine_strains_per_season.txt": "# Unused by E01 paired evaluation.\n",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def require_blob(data: bytes, expected: str, *, label: str) -> str:
    found = git_blob_sha(data)
    if found != expected:
        raise SystemExit(f"{label} relay blob mismatch: {found}")
    return data.decode("utf-8")


def load_exact_science(root: Path) -> tuple[str, str, str]:
    e01_data = b"".join((root / rel).read_bytes() for rel in E01_PARTS)
    e01 = require_blob(e01_data, E01_BLOB, label="strategy E01")
    e01_v2 = require_blob((root / E01_V2_PATH).read_bytes(), E01_V2_BLOB, label="strategy E01 v2")
    config = require_blob((root / CONFIG_PATH).read_bytes(), CONFIG_BLOB, label="E01 config")
    compile(e01, "cmi_flu/strategy_e01.py", "exec")
    compile(e01_v2, "cmi_flu/strategy_e01_v2.py", "exec")
    required = (
        'EXPERIMENT = "strategy_v2_e01_paired_evaluation"',
        'RANDOM_SEED = 20260907',
        'SUBSET_N = 28',
        'SUBSET_REPETITIONS = 200',
        'TASK12_AR_LAMBDA = 0.5',
        '"competition_submission_attempted": False',
    )
    if any(token not in e01 for token in required):
        raise SystemExit("strategy E01 source contract token missing")
    if "kaggle competitions submit" in e01 or "competition_submit" in e01:
        raise SystemExit("strategy E01 source contains submission path")
    if "study_group/subject_group" not in e01_v2:
        raise SystemExit("strategy E01 v2 identity correction missing")
    return e01, e01_v2, config


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
        "science_transport": "agent_relay_exact_blobs",
        "strategy_e01_blob_sha": E01_BLOB,
        "strategy_e01_v2_blob_sha": E01_V2_BLOB,
        "config_blob_sha": CONFIG_BLOB,
        "expected_kernel_version": 1,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"E01 request mismatch: {key}")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 30,
        "hard_timeout_minutes": 60,
        "max_active_runs": 1,
    }:
        raise SystemExit("E01 resource contract mismatch")
    if request.get("allowed_output_paths") != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E01 output allowlist mismatch")


def extract_frozen_runtime(root: Path, work: Path) -> tuple[bytes, str]:
    prior = root / TASK11_PRIOR_PATH
    require_blob(prior.read_bytes(), TASK11_PRIOR_BLOB, label="Task1.1 prior-immunity")
    generated = work / "task11-frozen-runtime.py"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/cmi_flu_task11_prior_immunity_prepare_v2.py"),
            "--repository-root",
            str(root),
            "--science-source",
            str(prior),
            "--output",
            str(generated),
        ],
        check=True,
    )
    text = generated.read_text(encoding="utf-8")
    namespace: dict[str, object] = {"__name__": "cmi_flu_e01_frozen_source"}
    exec(compile(text, str(generated), "exec"), namespace, namespace)
    package_fn = namespace.get("package_bytes")
    adapter_source = namespace.get("B21_ADAPTER_SOURCE")
    if not callable(package_fn) or not isinstance(adapter_source, str):
        raise SystemExit("frozen B2.1 runtime contract unavailable")
    package = package_fn()
    if not isinstance(package, bytes) or not package:
        raise SystemExit("frozen B2.1 package invalid")
    compile(adapter_source, "cmi_flu_b21_runtime_adapter.py", "exec")
    return package, adapter_source


def chunk64(data: bytes, width: int = 96) -> str:
    text = base64.b64encode(data).decode("ascii")
    return "\n".join(text[i:i + width] for i in range(0, len(text), width))


def build_runtime(package: bytes, adapter_source: str, e01: str, e01_v2: str, config: str) -> str:
    template = r'''#!/usr/bin/env python3
"""CMI-Flu strategy E01 paired evaluation. Aggregate outputs only; no submission."""
from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import hashlib
import io
import json
import math
import platform
import re
import shutil
import sys
import types
import zipfile
from pathlib import Path

REQUEST_ID = "__REQUEST_ID__"
COMPETITION = "__COMPETITION__"
SCIENCE_COMMIT = "__SCIENCE_COMMIT__"
TARGET_KERNEL = "__TARGET_KERNEL__"
E01_BLOB = "__E01_BLOB__"
E01_V2_BLOB = "__E01_V2_BLOB__"
CONFIG_BLOB = "__CONFIG_BLOB__"
PACKAGE_SHA256 = "__PACKAGE_SHA256__"
PACKAGE_B64 = __PACKAGE_B64__
B21_ADAPTER_SOURCE = __B21_ADAPTER_SOURCE__
E01_SOURCE = __E01_SOURCE__
E01_V2_SOURCE = __E01_V2_SOURCE__
CONFIG_TEXT = __CONFIG_TEXT__
REFERENCE_PLACEHOLDERS = __REFERENCE_PLACEHOLDERS__

CORE_FILES = (
    "participants.tsv", "investigations_260821.tsv",
    "publicData_cytokine.tsv", "publicData_ex_vivo_flow.tsv",
    "publicData_serology_260821.tsv", "2025LJI_aim.tsv",
    "2025LJI_cytokine.tsv", "2025LJI_ex_vivo_flow.tsv",
    "2025LJI_serology.tsv", "sample_submission_part1.csv", "md5sum",
)
EXPECTED_INCUMBENT = {
    "Task1.1": "b21_pls_2",
    "Task1.2": "task12_anchor_residual_et_d5_l5_sqrt_lambda0.5",
    "Task1.3": "b21_pls_1",
    "Task2.1": "b21_et_subtype_d3_l5",
    "Task2.2": "b21_et_subtype_d5_l10",
    "Task2.3": "b21_ridge_exact_a100",
}

class BridgeContractError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def package_bytes() -> bytes:
    data = base64.b64decode("".join(PACKAGE_B64.split()), validate=True)
    if hashlib.sha256(data).hexdigest() != PACKAGE_SHA256:
        raise BridgeContractError("package_sha_mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        if zf.testzip() is not None:
            raise BridgeContractError("package_corrupt")
        names = set(zf.namelist())
    required = {
        "cmi_flu/runner.py", "cmi_flu/evaluation.py", "cmi_flu/models.py",
        "cmi_flu/datasets.py", "cmi_flu/cv.py", "cmi_flu/metrics.py",
        "cmi_flu/targets.py", "cmi_flu/configuration.py",
    }
    if required - names:
        raise BridgeContractError("package_missing_required_modules")
    return data


def self_test() -> int:
    package = package_bytes()
    if git_blob_sha(E01_SOURCE.encode("utf-8")) != E01_BLOB:
        raise BridgeContractError("e01_blob_mismatch")
    if git_blob_sha(E01_V2_SOURCE.encode("utf-8")) != E01_V2_BLOB:
        raise BridgeContractError("e01_v2_blob_mismatch")
    if git_blob_sha(CONFIG_TEXT.encode("utf-8")) != CONFIG_BLOB:
        raise BridgeContractError("config_blob_mismatch")
    compile(B21_ADAPTER_SOURCE, "cmi_flu_b21_runtime_adapter.py", "exec")
    compile(E01_SOURCE, "cmi_flu/strategy_e01.py", "exec")
    compile(E01_V2_SOURCE, "cmi_flu/strategy_e01_v2.py", "exec")
    print(
        "CMI_FLU_E01_RUNTIME_SELF_TEST PASS "
        f"request_id={REQUEST_ID} package_bytes={len(package)} "
        f"package_sha256={PACKAGE_SHA256} science_commit={SCIENCE_COMMIT}"
    )
    return 0


def locate_competition_data(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    exact = Path("/kaggle/input") / COMPETITION
    candidates.append(exact)
    root = Path("/kaggle/input")
    if root.is_dir():
        try:
            candidates.extend(sorted(p.parent for p in root.rglob("sample_submission_part1.csv")))
        except OSError:
            pass
    valid = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        if all((resolved / name).is_file() for name in CORE_FILES):
            valid.append(resolved)
    if not valid:
        raise BridgeContractError("competition_mount_not_found")
    try:
        exact_resolved = exact.resolve()
    except OSError:
        exact_resolved = exact
    if exact_resolved in valid:
        return exact_resolved
    if explicit is not None and explicit.expanduser().resolve() in valid:
        return explicit.expanduser().resolve()
    valid = list(dict.fromkeys(valid))
    if len(valid) != 1:
        raise BridgeContractError("competition_mount_ambiguous")
    return valid[0]


def canonical_strain(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    raw = re.sub(r"(?i)_cell$", "_MDCK", raw)
    raw = re.sub(r"(?i)\s+cell$", "_MDCK", raw)
    return raw


def vaccine_flag(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "1.0", "true", "yes", "y"}


def derive_reference_files(input_dir: Path, external_root: Path) -> tuple[int, int]:
    import pandas as pd
    ref = external_root / "google-drive" / "challenge-resources" / "reference_files"
    ref.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(
        input_dir / "2025LJI_serology.tsv", sep="\t",
        usecols=["assay", "virus_strain", "virus_in_vaccine"], low_memory=False,
    )
    hai = source.loc[source["assay"].fillna("").astype(str).str.strip().str.casefold().eq("hai")].copy()
    hai["virus_strain"] = hai["virus_strain"].map(canonical_strain)
    challenge = tuple(sorted(v for v in hai["virus_strain"].dropna().unique() if v))
    vaccine = tuple(sorted(v for v in hai.loc[hai["virus_in_vaccine"].map(vaccine_flag), "virus_strain"].dropna().unique() if v))
    if len(challenge) != 12 or len(vaccine) != 3 or not set(vaccine).issubset(challenge):
        raise BridgeContractError("derived_hai_panel_contract_changed")
    (ref / "all_challenge_virus_strains.txt").write_text("\n".join(challenge) + "\n", encoding="utf-8")
    (ref / "vaccine_strains_2025.txt").write_text("\n".join(vaccine) + "\n", encoding="utf-8")
    for name, content in REFERENCE_PLACEHOLDERS.items():
        (ref / name).write_text(content, encoding="utf-8")
    return len(vaccine), len(challenge)


def load_e01_module() -> object:
    base = types.ModuleType("cmi_flu.strategy_e01")
    base.__file__ = "<cmi_flu.strategy_e01>"
    base.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_e01"] = base
    exec(compile(E01_SOURCE, "cmi_flu/strategy_e01.py", "exec"), base.__dict__, base.__dict__)
    v2 = types.ModuleType("cmi_flu.strategy_e01_v2")
    v2.__file__ = "<cmi_flu.strategy_e01_v2>"
    v2.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_e01_v2"] = v2
    exec(compile(E01_V2_SOURCE, "cmi_flu/strategy_e01_v2.py", "exec"), v2.__dict__, v2.__dict__)
    run = getattr(v2, "run_strategy_e01", None)
    if not callable(run):
        raise BridgeContractError("e01_entry_missing")
    return run


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except Exception:
            pass
    return str(value)


def validate_result(result: dict) -> None:
    if result.get("experiment") != "strategy_v2_e01_paired_evaluation":
        raise BridgeContractError("experiment_identity_mismatch")
    if result.get("comparison_contract") != "paired_subject_purged_v2":
        raise BridgeContractError("comparison_contract_mismatch")
    if result.get("random_seed") != 20260907:
        raise BridgeContractError("random_seed_mismatch")
    if result.get("subset_sensitivity") != {"n": 28, "repetitions": 200, "without_replacement": True}:
        raise BridgeContractError("subset_sensitivity_contract_mismatch")
    if result.get("frozen_incumbent") != EXPECTED_INCUMBENT:
        raise BridgeContractError("frozen_incumbent_mismatch")
    if set((result.get("tasks") or {})) != set(EXPECTED_INCUMBENT):
        raise BridgeContractError("task_set_mismatch")
    if result.get("task14_excluded") is not True:
        raise BridgeContractError("task14_contract_mismatch")
    if result.get("contains_participant_identifiers") is not False or result.get("contains_row_level_predictions") is not False:
        raise BridgeContractError("row_privacy_contract_mismatch")
    if result.get("leaderboard_used_for_selection") is not False or result.get("competition_submission_attempted") is not False:
        raise BridgeContractError("leaderboard_submission_contract_mismatch")
    heur = result.get("decision_heuristics") or {}
    if float(heur.get("study_mean_delta_candidate_threshold", 99)) != 0.02:
        raise BridgeContractError("delta_threshold_mismatch")
    if float(heur.get("large_study_decline_review_threshold", 99)) != -0.10:
        raise BridgeContractError("decline_threshold_mismatch")
    for task, payload in result["tasks"].items():
        incumbent = payload.get("incumbent") or {}
        if incumbent.get("name") != EXPECTED_INCUMBENT[task]:
            raise BridgeContractError(f"incumbent_name_mismatch_{task}")
        metrics = incumbent.get("metrics") or {}
        if int(metrics.get("rows", 0)) <= 0 or int(metrics.get("studies", 0)) <= 0:
            raise BridgeContractError(f"incumbent_metrics_missing_{task}")
        if not isinstance(payload.get("comparisons"), dict) or not isinstance(payload.get("sensitivity_28"), dict):
            raise BridgeContractError(f"comparison_payload_missing_{task}")
        if not isinstance(payload.get("availability_only_negative_control"), dict):
            raise BridgeContractError(f"availability_control_missing_{task}")
        if not isinstance(payload.get("subject_block_label_shuffle_negative_control"), dict):
            raise BridgeContractError(f"shuffle_control_missing_{task}")
    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"', '"subject_group"', '"row_index"', '"oof_predictions"', '"challenge_predictions"')
    if any(token in serialized for token in banned):
        raise BridgeContractError("aggregate_output_contains_row_fields")


def metric_value(task_payload: dict) -> str:
    metrics = (task_payload.get("incumbent") or {}).get("metrics") or {}
    strict = metrics.get("study_equal_weight_spearman_mean_strict")
    finite = metrics.get("study_equal_weight_spearman_mean_finite_diagnostic")
    pooled = (metrics.get("pooled_within_study_rank_spearman") or {}).get("value")
    return f"strict={strict} finite={finite} pooled_within_study_rank={pooled} undefined={metrics.get('undefined_fold_count')} constant={metrics.get('constant_fold_count')}"


def render_summary(result: dict) -> str:
    lines = [
        "# CMI-Flu strategy E01 paired evaluation", "",
        "Aggregate-only fixed-condition evaluation; no Competition submission was attempted.", "",
        f"- science commit: `{SCIENCE_COMMIT}`",
        "- comparison contract: `paired_subject_purged_v2`",
        "- 28-subject sensitivity: `200` without-replacement samples, seed `20260907`",
        "",
        "## Frozen incumbents", "",
    ]
    for task in EXPECTED_INCUMBENT:
        payload = result["tasks"][task]
        lines.append(f"- {task} / {payload['incumbent']['name']}: {metric_value(payload)}")
        for name, comp in (payload.get("comparisons") or {}).items():
            lines.append(
                f"  - {name}: study_mean_delta={comp.get('study_mean_delta_strict')}, "
                f"candidate_heuristic={comp.get('passes_candidate_delta_heuristic')}, "
                f"large_study_review={len(comp.get('large_studies_requiring_review') or [])}"
            )
    lines.extend(["", "HAI RMSE values are historical fixed-panel proxies, not exact 2025 target CV.", ""])
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def execute(input_dir: Path, output_dir: Path) -> int:
    runtime_root = output_dir / "e01-runtime"
    stage = "initialize"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        runtime_root.mkdir()
        stage = "materialize_package"
        package = package_bytes()
        package_path = runtime_root / "cmi_flu_bundle.zip"
        package_path.write_bytes(package)
        sys.path.insert(0, str(package_path))
        stage = "prepare_tree"
        data_parent = runtime_root / "data"
        data_parent.mkdir(parents=True)
        (data_parent / "raw").symlink_to(input_dir, target_is_directory=True)
        vaccine_n, challenge_n = derive_reference_files(input_dir, runtime_root / "external")
        if (vaccine_n, challenge_n) != (3, 12):
            raise BridgeContractError("panel_count_mismatch")
        config_dir = runtime_root / "configs"
        config_dir.mkdir()
        canonical_config = config_dir / "baseline_b021_robust.yaml"
        canonical_config.write_text(CONFIG_TEXT, encoding="utf-8")
        if git_blob_sha(canonical_config.read_bytes()) != CONFIG_BLOB:
            raise BridgeContractError("runtime_config_blob_mismatch")
        compat = config_dir / "baseline_b02_transport_compat.yaml"
        if CONFIG_TEXT.count("baseline: b021_taskwise_robust") != 1:
            raise BridgeContractError("b021_config_anchor_changed")
        compat.write_text(CONFIG_TEXT.replace("baseline: b021_taskwise_robust", "baseline: b02_taskwise_compact", 1), encoding="utf-8")
        stage = "install_b21_adapter"
        adapter_ns = {}
        exec(compile(B21_ADAPTER_SOURCE, "<b21_runtime_adapter>", "exec"), adapter_ns, adapter_ns)
        install = adapter_ns.get("install")
        if not callable(install):
            raise BridgeContractError("b21_adapter_install_missing")
        install()
        stage = "load_inputs"
        from cmi_flu.configuration import load_baseline_config
        from cmi_flu.runner import load_inputs
        config = load_baseline_config(compat, repository_root=runtime_root)
        raw = dict(config.raw); raw["baseline"] = "b021_taskwise_robust"
        config = replace(config, source_path=canonical_config, raw=raw, baseline="b021_taskwise_robust")
        if not config.verify_md5 or str(config.section("selection").get("policy", "")) != "robust_v1":
            raise BridgeContractError("runtime_config_contract_mismatch")
        inputs = load_inputs(config)
        if inputs.checksum_report is None:
            raise BridgeContractError("md5_verification_missing")
        stage = "load_e01"
        run_e01 = load_e01_module()
        stage = "run_e01"
        result = json_safe(dict(run_e01(config, inputs)))
        stage = "validate_e01"
        validate_result(result)
        stage = "write_outputs"
        metrics_path = output_dir / "metrics.json"
        summary_path = output_dir / "summary.md"
        bridge_path = output_dir / "bridge-result.json"
        metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        summary_path.write_text(render_summary(result), encoding="utf-8")
        bridge = {
            "schema_version": 1,
            "request_id": REQUEST_ID,
            "competition": COMPETITION,
            "target_kernel": TARGET_KERNEL,
            "science_commit": SCIENCE_COMMIT,
            "strategy_e01_blob_sha": E01_BLOB,
            "strategy_e01_v2_blob_sha": E01_V2_BLOB,
            "config_blob_sha": CONFIG_BLOB,
            "package_sha256": PACKAGE_SHA256,
            "python_version": platform.python_version(),
            "md5_verified_count": len(inputs.checksum_report.verified),
            "metrics_sha256": sha256_file(metrics_path),
            "summary_sha256": sha256_file(summary_path),
            "frozen_incumbent": result["frozen_incumbent"],
            "competition_submission_attempted": False,
            "leaderboard_used_for_selection": False,
            "contains_participant_identifiers": False,
            "contains_row_level_predictions": False,
        }
        bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.rmtree(runtime_root, ignore_errors=True)
        print(
            "CMI_FLU_E01_COMPLETE "
            f"request_id={REQUEST_ID} tasks={len(result['tasks'])} md5_verified={len(inputs.checksum_report.verified)} "
            "submission=false leaderboard_selection=false"
        )
        return 0
    except Exception as exc:
        shutil.rmtree(runtime_root, ignore_errors=True)
        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode("utf-8", errors="replace")).hexdigest()[:20]
        print(f"CMI_FLU_E01_FAILED stage={stage} exception_type={type(exc).__name__} error_code={code}", file=sys.stderr)
        return 2


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    try:
        input_dir = locate_competition_data(args.input_dir)
    except Exception as exc:
        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode()).hexdigest()[:20]
        print(f"CMI_FLU_E01_FAILED stage=locate_competition_data exception_type={type(exc).__name__} error_code={code}", file=sys.stderr)
        return 2
    return execute(input_dir, args.output_dir)

if __name__ == "__main__":
    raise SystemExit(main())
'''
    replacements = {
        "__REQUEST_ID__": REQUEST_ID,
        "__COMPETITION__": COMPETITION,
        "__SCIENCE_COMMIT__": SCIENCE_COMMIT,
        "__TARGET_KERNEL__": TARGET_KERNEL,
        "__E01_BLOB__": E01_BLOB,
        "__E01_V2_BLOB__": E01_V2_BLOB,
        "__CONFIG_BLOB__": CONFIG_BLOB,
        "__PACKAGE_SHA256__": sha256(package),
        "__PACKAGE_B64__": repr(chunk64(package)),
        "__B21_ADAPTER_SOURCE__": repr(adapter_source),
        "__E01_SOURCE__": repr(e01),
        "__E01_V2_SOURCE__": repr(e01_v2),
        "__CONFIG_TEXT__": repr(config),
        "__REFERENCE_PLACEHOLDERS__": repr(REFERENCE_PLACEHOLDERS),
    }
    for old, new in replacements.items():
        if old not in template:
            raise SystemExit(f"runtime placeholder missing: {old}")
        template = template.replace(old, new)
    compile(template, "generated_e01.py", "exec")
    return template


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validate_request(root)
    e01, e01_v2, config = load_exact_science(root)
    with tempfile.TemporaryDirectory(prefix="cmi-e01-build-") as tmp:
        package, adapter = extract_frozen_runtime(root, Path(tmp))
        runtime = build_runtime(package, adapter, e01, e01_v2, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_E01_BUILD PASS "
        f"science_commit={SCIENCE_COMMIT} e01_blob={E01_BLOB} e01_v2_blob={E01_V2_BLOB} "
        f"config_blob={CONFIG_BLOB} package_sha256={sha256(package)} runtime_sha256={sha256(runtime.encode())}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
