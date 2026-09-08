#!/usr/bin/env python3
"""Build the immutable aggregate-only CMI-Flu strategy-v2 E03 Kaggle runtime."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import re
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

REQUEST_ID = "20260908-cmi-flu-strategy-e03-task11-optional-view-fusion-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-e03-task11-optional-view-fusion-20260908-001"
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_COMMIT = "2eaf8fac9a632c67f32c4a0ed37b64eeac571387"
SCIENCE_TRANSPORT = "pinned_public_github_snapshot_presecret"
E03_BLOB = "e38fe01606dc5653ad3d1542f722726c1374b278"
TASK11_PRIOR_BLOB = "50d9a43604d2b75479b8f873a86a8daf9d5bd7a9"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
FROZEN_B21_PACKAGE_SHA256 = "87c317789b3b8fcbd5fce2ff8f74663d23c19d0fcdce0e05ba1e809ea7728cb2"
REQUEST_PATH = "requests/cmi-flu-strategy-e03-task11-optional-view-fusion-001.json"
SCIENCE_ARCHIVE_URL = f"https://api.github.com/repos/{SCIENCE_REPOSITORY}/tarball/{SCIENCE_COMMIT}"
MAX_ARCHIVE_BYTES = 25_000_000
CONDITIONS = (
    "b21",
    "hai_raw_fusion_w0.25",
    "hai_rank_fusion_w0.25",
    "hai_crossfit_residual_w0.25",
    "innate_flow_rank_fusion_w0.25",
    "innate_flow_crossfit_residual_w0.25",
)
ALLOWED_OUTPUTS = ("bridge-result.json", "metrics.json", "summary.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-science-tests", action="store_true")
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def require_blob(path: Path, expected: str, label: str) -> str:
    data = path.read_bytes()
    found = git_blob_sha(data)
    if found != expected:
        raise SystemExit(f"{label} blob mismatch: expected={expected} found={found}")
    text = data.decode("utf-8")
    compile(text, str(path), "exec")
    return text


def validate_request(root: Path) -> None:
    request = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": COMPETITION,
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_repository": SCIENCE_REPOSITORY,
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": SCIENCE_TRANSPORT,
        "strategy_e03_blob_sha": E03_BLOB,
        "task11_prior_immunity_blob_sha": TASK11_PRIOR_BLOB,
        "config_blob_sha": CONFIG_BLOB,
        "frozen_b21_package_sha256": FROZEN_B21_PACKAGE_SHA256,
        "expected_kernel_version": 1,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"E03 request mismatch: {key}")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 45,
        "hard_timeout_minutes": 75,
        "max_active_runs": 1,
    }:
        raise SystemExit("E03 resource contract mismatch")
    if tuple(request.get("allowed_output_paths", ())) != ALLOWED_OUTPUTS:
        raise SystemExit("E03 output allowlist mismatch")
    contract = request.get("experiment_contract") or {}
    checks = {
        "task": "Task1.1",
        "conditions": list(CONDITIONS),
        "b21_model": "pls_2",
        "fusion_weight": 0.25,
        "view_ridge_alpha": 10.0,
        "random_seed": 20260907,
        "missing_view_policy": "exact_fallback_to_frozen_b21",
        "residual_teacher": "training_internal_subject_purged_b21_oof_on_within_study_rank_scale",
        "flow_selection_outcome_independent": True,
        "flow_min_total_subjects": 24,
        "flow_min_total_studies": 2,
        "flow_min_source_subjects_per_fold": 8,
        "flow_max_features": 8,
        "promotion_mean_paired_delta": 0.02,
        "large_study_n": 10,
        "large_study_decline_review": -0.10,
    }
    for key, value in checks.items():
        if contract.get(key) != value:
            raise SystemExit(f"E03 experiment contract mismatch: {key}")


def materialize_science_snapshot(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        SCIENCE_ARCHIVE_URL,
        headers={"User-Agent": "kaggle-actions-bridge-cmi-e03/1"},
    )
    with urllib.request.urlopen(req, timeout=45, context=ssl.create_default_context()) as response:
        data = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(data) > MAX_ARCHIVE_BYTES:
        raise SystemExit("science snapshot exceeds byte budget")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        prefixes = {member.name.split("/", 1)[0] for member in files}
        if len(prefixes) != 1:
            raise SystemExit("science archive layout invalid")
        prefix = next(iter(prefixes)) + "/"
        wanted_prefixes = ("src/cmi_flu/", "scripts/", "configs/", "tests/")
        for member in files:
            if not member.name.startswith(prefix):
                raise SystemExit("science archive path invalid")
            rel = PurePosixPath(member.name[len(prefix):])
            if not rel.parts or ".." in rel.parts:
                raise SystemExit("unsafe science archive path")
            rel_text = rel.as_posix()
            if not any(rel_text.startswith(item) for item in wanted_prefixes):
                continue
            payload = archive.extractfile(member).read()
            out = target.joinpath(*rel.parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(payload)
    required = (
        "src/cmi_flu/strategy_e03.py",
        "src/cmi_flu/task11_prior_immunity.py",
        "configs/baseline_b021_robust.yaml",
        "scripts/build_b21_kaggle_bundle.py",
        "scripts/build_b2_kaggle_bundle.py",
        "scripts/run_b21_kaggle_bundle.py",
        "tests/test_strategy_e03.py",
    )
    missing = [name for name in required if not (target / name).is_file()]
    if missing:
        raise SystemExit(f"science snapshot incomplete: {missing}")
    return target


def frozen_b21_package(science: Path) -> bytes:
    builder_path = science / "scripts/build_b21_kaggle_bundle.py"
    spec = importlib.util.spec_from_file_location("cmi_e03_b21_builder", builder_path)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to import pinned B2.1 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = module.deterministic_b21_package_zip(science / "src")
    found = sha256(package)
    if found != FROZEN_B21_PACKAGE_SHA256:
        raise SystemExit(
            f"frozen B2.1 package mismatch: expected={FROZEN_B21_PACKAGE_SHA256} found={found}"
        )
    runner_text = (science / "scripts/run_b21_kaggle_bundle.py").read_text(encoding="utf-8")
    marker = re.search(r'^PACKAGE_ZIP_SHA256 = "([0-9a-f]{64})"$', runner_text, re.MULTILINE)
    if marker is None or marker.group(1) != found:
        raise SystemExit("checked-in B2.1 runner does not attest the rebuilt package")
    return package


def run_science_tests(science: Path) -> None:
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(science / "src")
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(science / "tests/test_strategy_e03.py")],
        env=env,
        check=True,
    )


def chunk64(data: bytes, width: int = 96) -> str:
    text = base64.b64encode(data).decode("ascii")
    return "\n".join(text[i : i + width] for i in range(0, len(text), width))


def build_runtime(package: bytes, e03: str, prior: str, config: str) -> str:
    template = r'''#!/usr/bin/env python3
"""CMI-Flu E03 Task1.1 optional-view fusion; aggregate outputs only, no submission."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import types
import zipfile

REQUEST_ID = "__REQUEST_ID__"
COMPETITION = "__COMPETITION__"
SCIENCE_COMMIT = "__SCIENCE_COMMIT__"
TARGET_KERNEL = "__TARGET_KERNEL__"
E03_BLOB = "__E03_BLOB__"
TASK11_PRIOR_BLOB = "__TASK11_PRIOR_BLOB__"
CONFIG_BLOB = "__CONFIG_BLOB__"
PACKAGE_SHA256 = "__PACKAGE_SHA256__"
PACKAGE_B64 = """__PACKAGE_B64__"""
E03_SOURCE = __E03_SOURCE__
TASK11_PRIOR_SOURCE = __TASK11_PRIOR_SOURCE__
CONFIG_TEXT = __CONFIG_TEXT__
CONDITIONS = __CONDITIONS__
ALLOWED_OUTPUTS = __ALLOWED_OUTPUTS__

CORE_FILES = (
    "participants.tsv", "investigations_260821.tsv",
    "publicData_cytokine.tsv", "publicData_ex_vivo_flow.tsv",
    "publicData_serology_260821.tsv", "2025LJI_aim.tsv",
    "2025LJI_cytokine.tsv", "2025LJI_ex_vivo_flow.tsv",
    "2025LJI_serology.tsv", "sample_submission_part1.csv", "md5sum",
)
REFERENCE_PLACEHOLDERS = {
    "cytokine_name_map.csv": "source,target\n",
    "flow_name_revised.csv": "source,target\n",
    "hai_map.csv": "source,target\n",
    "strain_sequences.csv": "virus_strain,sequence\n",
    "vaccine_strains_per_season.txt": "# Unused by E03 Task1.1.\n",
}
FORBIDDEN_OUTPUT_KEYS = {
    "participant_id", "participant_ids", "row_prediction", "row_predictions",
    "prediction_rows_raw", "predicted_values", "submission", "submission_rows_raw",
}

class BridgeContractError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def package_bytes() -> bytes:
    data = base64.b64decode("".join(PACKAGE_B64.split()), validate=True)
    if sha256(data) != PACKAGE_SHA256:
        raise BridgeContractError("frozen_b21_package_sha_mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise BridgeContractError("frozen_b21_package_corrupt")
        names = set(archive.namelist())
    required = {
        "cmi_flu/runner.py", "cmi_flu/models.py", "cmi_flu/cv.py",
        "cmi_flu/metrics.py", "cmi_flu/configuration.py", "cmi_flu/targets.py",
        "cmi_flu/contracts.py", "cmi_flu/aliases.py",
    }
    if required - names:
        raise BridgeContractError("frozen_b21_package_missing_modules")
    forbidden = {"cmi_flu/strategy_e03.py", "cmi_flu/task11_prior_immunity.py"}
    if names.intersection(forbidden):
        raise BridgeContractError("postfreeze_module_inside_frozen_b21_package")
    return data


def validate_science_contract() -> None:
    if git_blob_sha(E03_SOURCE.encode("utf-8")) != E03_BLOB:
        raise BridgeContractError("e03_blob_mismatch")
    if git_blob_sha(TASK11_PRIOR_SOURCE.encode("utf-8")) != TASK11_PRIOR_BLOB:
        raise BridgeContractError("task11_prior_blob_mismatch")
    if git_blob_sha(CONFIG_TEXT.encode("utf-8")) != CONFIG_BLOB:
        raise BridgeContractError("config_blob_mismatch")
    compile(E03_SOURCE, "cmi_flu/strategy_e03.py", "exec")
    compile(TASK11_PRIOR_SOURCE, "cmi_flu/task11_prior_immunity.py", "exec")
    required = (
        'EXPERIMENT = "strategy_v2_e03_task11_optional_view_fusion"',
        'B21_MODEL_NAME = "pls_2"', 'FUSION_WEIGHT = 0.25', 'VIEW_ALPHA = 10.0',
        'RANDOM_SEED = 20260907', 'MIN_FLOW_TOTAL_SUBJECTS = 24',
        'MIN_FLOW_TOTAL_STUDIES = 2', 'MIN_VIEW_SOURCE_SUBJECTS = 8',
        'MAX_FLOW_FEATURES = 8', '"competition_submission_attempted": False',
    )
    if any(token not in E03_SOURCE for token in required):
        raise BridgeContractError("e03_science_contract_token_missing")
    if "kaggle competitions submit" in E03_SOURCE or "competition_submit" in E03_SOURCE:
        raise BridgeContractError("e03_source_contains_submission_path")


def install_modules(work: Path):
    package = package_bytes()
    archive_path = work / "frozen-b21.zip"
    archive_path.write_bytes(package)
    sys.path.insert(0, str(archive_path))
    import cmi_flu  # noqa: F401
    prior = types.ModuleType("cmi_flu.task11_prior_immunity")
    prior.__file__ = "<cmi_flu.task11_prior_immunity>"
    prior.__package__ = "cmi_flu"
    sys.modules[prior.__name__] = prior
    exec(compile(TASK11_PRIOR_SOURCE, prior.__file__, "exec"), prior.__dict__, prior.__dict__)
    e03 = types.ModuleType("cmi_flu.strategy_e03")
    e03.__file__ = "<cmi_flu.strategy_e03>"
    e03.__package__ = "cmi_flu"
    sys.modules[e03.__name__] = e03
    exec(compile(E03_SOURCE, e03.__file__, "exec"), e03.__dict__, e03.__dict__)
    return e03


def self_test() -> int:
    validate_science_contract()
    with tempfile.TemporaryDirectory(prefix="cmi-e03-selftest-") as tmp:
        e03 = install_modules(Path(tmp))
        if tuple(e03.CONDITION_NAMES) != tuple(CONDITIONS):
            raise BridgeContractError("e03_condition_contract_mismatch")
        base = __import__("numpy").array([0.1, 0.4, 0.9], dtype=float)
        expert = __import__("numpy").array([math.nan, 0.2, 0.8], dtype=float)
        blended = e03._blend(base, expert)
        corrected = e03._residual_correct(base, expert)
        if blended[0] != base[0] or corrected[0] != base[0]:
            raise BridgeContractError("missing_view_fallback_not_exact")
    print(
        "CMI_FLU_E03_RUNTIME_SELF_TEST PASS "
        f"request_id={REQUEST_ID} science_commit={SCIENCE_COMMIT} "
        f"package_sha256={PACKAGE_SHA256} conditions={len(CONDITIONS)}"
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
            candidates.extend(sorted(path.parent for path in root.rglob("sample_submission_part1.csv")))
        except OSError:
            pass
    valid, seen = [], set()
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
    exact_resolved = exact.resolve() if exact.exists() else exact
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


def stage_workspace(input_dir: Path, work: Path) -> Path:
    import pandas as pd
    data_dir = work / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in CORE_FILES:
        source = input_dir / name
        target = data_dir / name
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)
    ref = work / "external" / "google-drive" / "challenge-resources" / "reference_files"
    ref.mkdir(parents=True, exist_ok=True)
    serology = pd.read_csv(
        input_dir / "2025LJI_serology.tsv", sep="\t",
        usecols=["assay", "virus_strain", "virus_in_vaccine"], low_memory=False,
    )
    hai = serology.loc[
        serology["assay"].fillna("").astype(str).str.strip().str.casefold().eq("hai")
    ].copy()
    hai["virus_strain"] = hai["virus_strain"].map(canonical_strain)
    challenge = tuple(sorted(value for value in hai["virus_strain"].dropna().unique() if value))
    vaccine = tuple(sorted(value for value in hai.loc[hai["virus_in_vaccine"].map(vaccine_flag), "virus_strain"].dropna().unique() if value))
    if len(challenge) != 12 or len(vaccine) != 3 or not set(vaccine).issubset(challenge):
        raise BridgeContractError("derived_hai_panel_contract_changed")
    (ref / "all_challenge_virus_strains.txt").write_text("\n".join(challenge) + "\n", encoding="utf-8")
    (ref / "vaccine_strains_2025.txt").write_text("\n".join(vaccine) + "\n", encoding="utf-8")
    for name, content in REFERENCE_PLACEHOLDERS.items():
        (ref / name).write_text(content, encoding="utf-8")
    config_dir = work / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "baseline_b021_robust.yaml").write_text(CONFIG_TEXT, encoding="utf-8")
    (work / "artifacts").mkdir(exist_ok=True)
    return work


def scan_output(value, path="root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().casefold()
            if normalized in FORBIDDEN_OUTPUT_KEYS:
                raise BridgeContractError(f"forbidden_output_key:{path}.{key}")
            scan_output(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            scan_output(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise BridgeContractError(f"nonfinite_output:{path}")


def validate_result(result: dict) -> None:
    if result.get("schema_version") != 1:
        raise BridgeContractError("e03_result_schema_version")
    if result.get("experiment") != "strategy_v2_e03_task11_optional_view_fusion":
        raise BridgeContractError("e03_result_experiment")
    if result.get("task") != "Task1.1" or tuple(result.get("conditions", ())) != tuple(CONDITIONS):
        raise BridgeContractError("e03_result_condition_contract")
    if result.get("competition_submission_attempted") is not False:
        raise BridgeContractError("e03_result_submission_flag")
    if result.get("leaderboard_used_for_selection") is not False:
        raise BridgeContractError("e03_result_leaderboard_flag")
    if result.get("output_policy") != "aggregate_only_no_participant_ids_or_row_level_predictions":
        raise BridgeContractError("e03_result_output_policy")
    if int((result.get("challenge") or {}).get("rows", -1)) != 40:
        raise BridgeContractError("e03_result_challenge_row_count")
    folds = result.get("folds") or []
    if len(folds) != 4:
        raise BridgeContractError("e03_result_fold_count")
    scan_output(result)


def render_summary(result: dict) -> str:
    lines = [
        "# CMI-Flu strategy-v2 E03 Task1.1 optional-view fusion",
        "",
        f"- science commit: `{SCIENCE_COMMIT}`",
        "- resource: CPU",
        "- competition submission attempted: false",
        "- leaderboard used for selection: false",
        f"- flow panel state: `{(result.get('flow_panel') or {}).get('state')}`",
        f"- selected promoted condition: `{result.get('selected_promoted_condition')}`",
        "",
        "| condition | equal-study Spearman | paired delta vs B2.1 | guardrail |",
        "|---|---:|---:|---|",
    ]
    summary = result.get("summary") or {}
    paired = result.get("paired_vs_b21") or {}
    for name in CONDITIONS:
        score = (summary.get(name) or {}).get("study_equal_weight_spearman_mean_strict")
        delta = None if name == "b21" else (paired.get(name) or {}).get("study_mean_delta_strict")
        guard = "reference" if name == "b21" else str((paired.get(name) or {}).get("passes_guardrail"))
        score_text = "NA" if score is None else f"{float(score):.9f}"
        delta_text = "NA" if delta is None else f"{float(delta):+.9f}"
        lines.append(f"| `{name}` | {score_text} | {delta_text} | {guard} |")
    lines += ["", "## Flow support", ""]
    panel = result.get("flow_panel") or {}
    lines.append(
        f"selected_features={panel.get('selected_count')} selected_subjects={panel.get('selected_subjects')} "
        f"selected_studies={panel.get('selected_studies')} broad_claim={panel.get('broad_cross_study_replacement_claim_permitted')}"
    )
    return "\n".join(lines) + "\n"


def write_outputs(result: dict, output_dir: Path) -> None:
    validate_result(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.iterdir() if path.is_file() or path.is_symlink()}
    unexpected = existing.difference(ALLOWED_OUTPUTS)
    if unexpected:
        raise BridgeContractError(f"working_directory_not_clean:{sorted(unexpected)}")
    metrics_text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    summary_text = render_summary(result)
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    metrics_path.write_text(metrics_text, encoding="utf-8")
    summary_path.write_text(summary_text, encoding="utf-8")
    bridge = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e03_blob_sha": E03_BLOB,
        "task11_prior_immunity_blob_sha": TASK11_PRIOR_BLOB,
        "config_blob_sha": CONFIG_BLOB,
        "frozen_b21_package_sha256": PACKAGE_SHA256,
        "status": "complete",
        "selected_promoted_condition": result.get("selected_promoted_condition"),
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "metrics_sha256": sha256(metrics_text.encode("utf-8")),
        "summary_sha256": sha256(summary_text.encode("utf-8")),
    }
    scan_output(bridge)
    (output_dir / "bridge-result.json").write_text(
        json.dumps(bridge, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    final = {path.name for path in output_dir.iterdir() if path.is_file() or path.is_symlink()}
    if final != set(ALLOWED_OUTPUTS) or any((output_dir / name).is_symlink() for name in ALLOWED_OUTPUTS):
        raise BridgeContractError(f"output_allowlist_violation:{sorted(final)}")


def compact_log(result: dict) -> dict:
    return {
        "experiment": result.get("experiment"),
        "selected_promoted_condition": result.get("selected_promoted_condition"),
        "flow_panel": {
            key: (result.get("flow_panel") or {}).get(key)
            for key in ("state", "selected_count", "selected_subjects", "selected_studies", "broad_cross_study_replacement_claim_permitted")
        },
        "summary": {
            name: (result.get("summary") or {}).get(name, {}).get("study_equal_weight_spearman_mean_strict")
            for name in CONDITIONS
        },
        "paired_delta": {
            name: (result.get("paired_vs_b21") or {}).get(name, {}).get("study_mean_delta_strict")
            for name in CONDITIONS if name != "b21"
        },
        "review_studies": {
            name: (result.get("paired_vs_b21") or {}).get(name, {}).get("large_studies_requiring_review")
            for name in CONDITIONS if name != "b21"
        },
        "challenge": {
            "rows": (result.get("challenge") or {}).get("rows"),
            "hai_rows": (result.get("challenge") or {}).get("hai_rows"),
            "flow_rows": (result.get("challenge") or {}).get("flow_rows"),
        },
    }


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    validate_science_contract()
    input_dir = locate_competition_data(args.input_dir)
    output_dir = args.output_dir.expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="cmi-e03-runtime-", dir="/tmp") as tmp:
        work = stage_workspace(input_dir, Path(tmp))
        e03 = install_modules(work)
        from cmi_flu.configuration import load_baseline_config
        from cmi_flu.runner import load_inputs
        config = load_baseline_config(work / "configs/baseline_b021_robust.yaml", repository_root=work)
        inputs = load_inputs(config)
        result = dict(e03.run_strategy_e03(config, inputs))
        validate_result(result)
        write_outputs(result, output_dir)
        print("E03_AGGREGATE=" + json.dumps(compact_log(result), sort_keys=True, separators=(",", ":")))
        print(
            "CMI_FLU_E03_COMPLETE "
            f"request_id={REQUEST_ID} selected={result.get('selected_promoted_condition')} "
            "submission=false leaderboard_selection=false"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    replacements = {
        "__REQUEST_ID__": REQUEST_ID,
        "__COMPETITION__": COMPETITION,
        "__SCIENCE_COMMIT__": SCIENCE_COMMIT,
        "__TARGET_KERNEL__": TARGET_KERNEL,
        "__E03_BLOB__": E03_BLOB,
        "__TASK11_PRIOR_BLOB__": TASK11_PRIOR_BLOB,
        "__CONFIG_BLOB__": CONFIG_BLOB,
        "__PACKAGE_SHA256__": sha256(package),
        "__PACKAGE_B64__": chunk64(package),
        "__E03_SOURCE__": repr(e03),
        "__TASK11_PRIOR_SOURCE__": repr(prior),
        "__CONFIG_TEXT__": repr(config),
        "__CONDITIONS__": repr(CONDITIONS),
        "__ALLOWED_OUTPUTS__": repr(ALLOWED_OUTPUTS),
    }
    runtime = template
    for marker, value in replacements.items():
        if runtime.count(marker) != 1:
            raise SystemExit(f"runtime marker count changed: {marker}")
        runtime = runtime.replace(marker, value, 1)
    compile(runtime, "generated_cmi_e03_runtime.py", "exec")
    return runtime


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validate_request(root)
    with tempfile.TemporaryDirectory(prefix="cmi-e03-build-") as tmp:
        science = materialize_science_snapshot(Path(tmp) / "science")
        e03 = require_blob(science / "src/cmi_flu/strategy_e03.py", E03_BLOB, "strategy E03")
        prior = require_blob(
            science / "src/cmi_flu/task11_prior_immunity.py", TASK11_PRIOR_BLOB, "Task1.1 prior"
        )
        config_data = (science / "configs/baseline_b021_robust.yaml").read_bytes()
        if git_blob_sha(config_data) != CONFIG_BLOB:
            raise SystemExit("E03 config blob mismatch")
        config = config_data.decode("utf-8")
        package = frozen_b21_package(science)
        if args.run_science_tests:
            run_science_tests(science)
        runtime = build_runtime(package, e03, prior, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_E03_BUILD PASS "
        f"request_id={REQUEST_ID} science_commit={SCIENCE_COMMIT} e03_blob={E03_BLOB} "
        f"task11_prior_blob={TASK11_PRIOR_BLOB} config_blob={CONFIG_BLOB} "
        f"frozen_b21_package_sha256={FROZEN_B21_PACKAGE_SHA256} runtime_sha256={sha256(runtime.encode('utf-8'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
