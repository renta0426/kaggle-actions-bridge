#!/usr/bin/env python3
"""Build the one-shot aggregate-only CMI-Flu E12b Challenge prediction freeze runtime."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

REQUEST_ID = "20260911-cmi-flu-strategy-e12b-challenge-freeze-001"
TARGET = "renta0426/cmi-flu-e12b-challenge-freeze-20260911-001"
SCIENCE_COMMIT = "a411bf85a4a79fec2e9a2b9c2bc8cbe186adee5f"
E12B_BLOB = "af5df92ca81abc34a7f7046dba5bc284c98302d4"
E12B_SYNTH_BLOB = "6b36d94299ee3656f9faed623a2ac030a97a714d"
E12B_CONTRACT_BLOB = "7e844b2d56bd6739793888bf6306946a0028d966"
E12A_V2_BLOB = "0c4c970c8bacfed61bfbb9587e0a0bfdec7903d9"
E01_BLOB = "dd27aea0cf97d41bad3cec64819c4c4269d94cbd"
E01_V2_BLOB = "8cc64dc5ab9483d5957cfada18d445188566c56c"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e12b-challenge-freeze-001"
E12B_PATH = f"{PAYLOAD_ROOT}/strategy_e12b.py"
E12B_SYNTH_PATH = f"{PAYLOAD_ROOT}/strategy_e12b_synthetic.py"
E12B_CONTRACT_PATH = f"{PAYLOAD_ROOT}/strategy_e12b_challenge_prediction_freeze.json"
REQUEST_PATH = "requests/cmi-flu-strategy-e12b-challenge-freeze-001.json"
BASE_BUILDER = "scripts/cmi_flu_strategy_e12a_v2_prepare_v2.py"
BASE_BUILDER_BLOB = "32f0d40689db1013bc1d42de2286e4ff26c91c44"
E12A_REQUEST = "20260911-cmi-flu-strategy-e12a-v2-portfolio-reconciliation-001"
E12A_TARGET = "renta0426/cmi-flu-e12a-v2-portfolio-reconcile-20260911-001"
E12A_SCIENCE = "e6e05e578f172ce710bc0f1da55acdf290e83dd9"
SUMMARY_HEADING = "# CMI-Flu Strategy-v2 E12b Challenge prediction freeze"
EXPECTED_PORTFOLIO = {
    "Task1.1": "b21_pls_2",
    "Task1.2": "task12_anchor_residual_et_d5_l5_sqrt_lambda0.5",
    "Task1.3": "strict_asc_anchor",
    "Task1.4": "raw_pre_vacc_conserved_anchor",
    "Task2.1": "b21_et_subtype_d3_l5",
    "Task2.2": "b21_et_subtype_d5_l10",
    "Task2.3": "b21_ridge_exact_a100",
}
TASKS = tuple(EXPECTED_PORTFOLIO)


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def require_blob(path: Path, expected: str, *, label: str) -> bytes:
    data = path.read_bytes()
    found = git_blob(data)
    if found != expected:
        raise SystemExit(f"{label} relay blob mismatch:{found}")
    return data


def replace_function(text: str, name: str, replacement: str) -> str:
    tree = ast.parse(text)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(nodes) != 1:
        raise SystemExit(f"E12b runtime function binding changed:{name}:{len(nodes)}")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    return (
        "".join(lines[: node.lineno - 1])
        + replacement.rstrip()
        + "\n"
        + "".join(lines[node.end_lineno :])
    )


def validate_request(root: Path) -> dict:
    request = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    exact = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET,
        "science_repository": "renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": "connector_verified_exact_blob_relay",
        "strategy_e12b_blob_sha": E12B_BLOB,
        "strategy_e12b_synthetic_blob_sha": E12B_SYNTH_BLOB,
        "strategy_e12b_contract_blob_sha": E12B_CONTRACT_BLOB,
        "strategy_e12a_v2_blob_sha": E12A_V2_BLOB,
        "strategy_e01_blob_sha": E01_BLOB,
        "strategy_e01_v2_blob_sha": E01_V2_BLOB,
        "config_blob_sha": CONFIG_BLOB,
        "expected_kernel_version": 1,
        "enable_internet": False,
        "automatic_compute_retries": 0,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "publish_only_sanitized_aggregate": True,
    }
    for key, value in exact.items():
        if request.get(key) != value:
            raise SystemExit(f"E12b request mismatch:{key}")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 30,
        "hard_timeout_minutes": 60,
        "max_active_runs": 1,
    }:
        raise SystemExit("E12b resource contract mismatch")
    if request.get("api_budget") != {
        "max_calls": 60,
        "poll_interval_seconds": 120,
        "max_polls": 35,
        "max_pages": 2,
    }:
        raise SystemExit("E12b API budget mismatch")
    parent = request.get("parent_e12a_v2") or {}
    if parent != {
        "request_id": E12A_REQUEST,
        "target": E12A_TARGET,
        "version": 1,
        "actions_run": 34547852282,
        "actions_job": 103104288165,
        "savekernel_succeeded": True,
        "kernel_terminal_state": "COMPLETE",
        "kernel_version_consumed": True,
        "accepted_output_zip_sha256": "7dca4546caa96f3c8fedeba3e4e758c677cb9cd7e7627ab67a79570d66dc0b64",
        "all_supervised_tasks_reproduced": True,
        "same_version_rerun_forbidden": True,
        "next_step": "proceed_to_E12b_challenge_prediction_freeze",
        "runtime_incident_classification": "bridge_runtime_compatibility_output_recovery_path",
    }:
        raise SystemExit("E12b parent E12a-v2 provenance mismatch")
    experiment = request.get("experiment_contract") or {}
    if experiment != {
        "experiment": "strategy_v2_e12b_challenge_prediction_freeze",
        "portfolio_task_count": 7,
        "challenge_rows": 40,
        "task13_final_incumbent": "strict_asc_anchor",
        "task13_anchor_feature": "flow_rank__Antibody-secreting_cells_(ASC)",
        "task13_expected_challenge_unique_values": 36,
        "changed_tasks_vs_regenerated_b21": ["Task1.2", "Task1.3"],
        "changed_tasks_vs_regenerated_task12_only": ["Task1.3"],
        "semantic_hash_version": "cmi-flu-e12b-semantic-v1",
        "canonical_csv_float_format": "%.17g",
        "new_model_selection_allowed": False,
        "row_level_candidate_persisted": False,
        "public_probe_authorized": False,
        "competition_submission_authorized": False,
        "if_pass": "ready_for_separate_submission_authorization",
        "if_fail": "investigate_prediction_freeze_contract_no_model_selection",
    }:
        raise SystemExit("E12b experiment contract mismatch")
    if request.get("output_reader_path_repair") != {
        "parent_error": "official Kaggle CLI is not on PATH",
        "locked_client": "kaggle==2.2.4",
        "required_path_source": "directory containing protected-job Python executable",
        "credential_free_ci_self_test_required": True,
        "science_changed_by_repair": False,
    }:
        raise SystemExit("E12b output-reader PATH repair contract mismatch")
    if request.get("allowed_output_paths") != [
        "bridge-result.json",
        "metrics.json",
        "summary.md",
    ]:
        raise SystemExit("E12b output allowlist mismatch")
    return request


def load_science(root: Path) -> tuple[str, str, str]:
    source = require_blob(root / E12B_PATH, E12B_BLOB, label="E12b science").decode("utf-8")
    synthetic = require_blob(
        root / E12B_SYNTH_PATH, E12B_SYNTH_BLOB, label="E12b synthetic"
    ).decode("utf-8")
    contract_text = require_blob(
        root / E12B_CONTRACT_PATH, E12B_CONTRACT_BLOB, label="E12b contract"
    ).decode("utf-8")
    compile(source, "cmi_flu/strategy_e12b.py", "exec")
    compile(synthetic, "cmi_flu/strategy_e12b_synthetic.py", "exec")
    contract = json.loads(contract_text)
    if contract.get("experiment") != "strategy_v2_e12b_challenge_prediction_freeze":
        raise SystemExit("E12b contract identity changed")
    if contract.get("final_portfolio") != EXPECTED_PORTFOLIO:
        raise SystemExit("E12b frozen portfolio changed")
    challenge = contract.get("challenge_contract") or {}
    if challenge.get("rows") != 40 or challenge.get("task13_expected_unique_values") != 36:
        raise SystemExit("E12b Challenge contract changed")
    if challenge.get("changed_tasks_vs_regenerated_b21") != ["Task1.2", "Task1.3"]:
        raise SystemExit("E12b B2.1 structural control changed")
    if challenge.get("changed_tasks_vs_regenerated_task12_only") != ["Task1.3"]:
        raise SystemExit("E12b Task1.2-only structural control changed")
    execution = contract.get("execution") or {}
    if execution.get("new_model_selection") is not False or execution.get("competition_submission") is not False:
        raise SystemExit("E12b execution boundary changed")
    if execution.get("row_level_candidate_persisted") is not False:
        raise SystemExit("E12b row-level persistence boundary changed")
    required = (
        'EXPERIMENT = "strategy_v2_e12b_challenge_prediction_freeze"',
        'TASK12_MODEL_NAME = "et_d5_l5_sqrt"',
        'TASK12_ANCHOR_COLUMN = "flow_rank__Classical_monocytes"',
        "TASK12_LAMBDA = 0.5",
        'TASK13_FINAL_INCUMBENT',
        'EXPECTED_CHANGED_FROM_B21 = ("Task1.2", "Task1.3")',
        'EXPECTED_CHANGED_FROM_TASK12_ONLY = ("Task1.3",)',
        'SEMANTIC_HASH_VERSION = "cmi-flu-e12b-semantic-v1"',
        '"row_level_candidate_persisted": False',
        '"competition_submission_attempted": False',
        "def run_strategy_e12b(",
    )
    if any(token not in source for token in required):
        raise SystemExit("E12b science contract token missing")
    if "from .public_probes" in source:
        raise SystemExit("E12b science unexpectedly depends on historical Public probe module")
    lowered = source.casefold()
    if "kaggle competitions submit" in lowered or "competition_submit(" in lowered:
        raise SystemExit("E12b science source contains submission path")
    return source, synthetic, contract_text


def build_runtime(root: Path, output: Path, source: str) -> str:
    base_path = root / BASE_BUILDER
    if git_blob(base_path.read_bytes()) != BASE_BUILDER_BLOB:
        raise SystemExit("E12b base E12a-v2 builder changed")
    with tempfile.TemporaryDirectory(prefix="cmi-e12b-base-") as tmp:
        generated = Path(tmp) / "e12a-v2.py"
        subprocess.run(
            [
                sys.executable,
                str(base_path),
                "--repository-root",
                str(root),
                "--output",
                str(generated),
            ],
            check=True,
        )
        runtime = generated.read_text(encoding="utf-8")

    for old, new, label in (
        (E12A_REQUEST, REQUEST_ID, "request"),
        (E12A_TARGET, TARGET, "target"),
        (E12A_SCIENCE, SCIENCE_COMMIT, "science"),
    ):
        if old not in runtime:
            raise SystemExit(f"E12b base runtime {label} anchor missing")
        runtime = runtime.replace(old, new)
        if old in runtime:
            raise SystemExit(f"E12b base runtime retained old {label}")

    marker = f'E12A_V2_CONTRACT_BLOB = "6b40ac35a3b43c3bafcdd08fcc49fc9fe22960c1"\n'
    if runtime.count(marker) != 1:
        raise SystemExit("E12b E12a-v2 constant anchor changed")
    runtime = runtime.replace(
        marker,
        marker
        + f'E12B_BLOB = "{E12B_BLOB}"\n'
        + f'E12B_CONTRACT_BLOB = "{E12B_CONTRACT_BLOB}"\n'
        + f'E12B_SOURCE = {source!r}\n',
        1,
    )

    loader = r'''def load_e12b_module() -> object:
    load_e01_module()
    load_e12a_v2_functions()
    module = types.ModuleType("cmi_flu.strategy_e12b")
    module.__file__ = "<cmi_flu.strategy_e12b>"
    module.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_e12b"] = module
    exec(compile(E12B_SOURCE, "cmi_flu/strategy_e12b.py", "exec"), module.__dict__, module.__dict__)
    run = getattr(module, "run_strategy_e12b", None)
    if not callable(run):
        raise BridgeContractError("e12b_entry_missing")
    return run
'''
    execute_marker = "def execute(input_dir: Path, output_dir: Path) -> int:\n"
    if runtime.count(execute_marker) != 1:
        raise SystemExit("E12b execute insertion anchor changed")
    runtime = runtime.replace(execute_marker, loader + "\n" + execute_marker, 1)

    validator = r'''def validate_result(result: dict) -> None:
    expected_portfolio = {
        "Task1.1": "b21_pls_2",
        "Task1.2": "task12_anchor_residual_et_d5_l5_sqrt_lambda0.5",
        "Task1.3": "strict_asc_anchor",
        "Task1.4": "raw_pre_vacc_conserved_anchor",
        "Task2.1": "b21_et_subtype_d3_l5",
        "Task2.2": "b21_et_subtype_d5_l10",
        "Task2.3": "b21_ridge_exact_a100",
    }
    tasks = tuple(expected_portfolio)
    hex64 = re.compile(r"^[0-9a-f]{64}$")
    if result.get("experiment") != "strategy_v2_e12b_challenge_prediction_freeze":
        raise BridgeContractError("e12b_identity")
    if result.get("final_portfolio") != expected_portfolio or int(result.get("portfolio_task_count", -1)) != 7:
        raise BridgeContractError("e12b_portfolio")
    if int(result.get("challenge_rows", -1)) != 40:
        raise BridgeContractError("e12b_challenge_rows")
    for key in (
        "new_model_selection_performed", "leaderboard_used_for_selection",
        "public_probe_performed", "competition_submission_attempted",
        "competition_submission_authorized", "contains_participant_identifiers",
        "contains_row_level_predictions", "row_level_candidate_persisted",
    ):
        if result.get(key) is not False:
            raise BridgeContractError(f"e12b_boundary:{key}")
    if result.get("next_step") != "ready_for_separate_submission_authorization":
        raise BridgeContractError("e12b_next_step")

    validation = result.get("submission_validation") or {}
    if int(validation.get("rows", -1)) != 40:
        raise BridgeContractError("e12b_submission_rows")
    expected_columns = ["participant_id", *tasks]
    if validation.get("columns") != expected_columns:
        raise BridgeContractError("e12b_submission_columns")
    if validation.get("minus99_tasks") != []:
        raise BridgeContractError("e12b_minus99")
    unique_counts = validation.get("task_unique_counts") or {}
    if set(unique_counts) != set(tasks):
        raise BridgeContractError("e12b_submission_task_set")
    for task in tasks[:-1]:
        if int(unique_counts.get(task, 0)) < 2:
            raise BridgeContractError(f"e12b_public_task_constant:{task}")

    fingerprints = result.get("fingerprint_contract") or {}
    if fingerprints.get("semantic_hash_version") != "cmi-flu-e12b-semantic-v1":
        raise BridgeContractError("e12b_semantic_hash_version")
    if fingerprints.get("canonical_csv_float_format") != "%.17g" or fingerprints.get("canonical_csv_lineterminator") != "LF":
        raise BridgeContractError("e12b_csv_contract")
    for key in ("semantic_submission_sha256", "canonical_csv_sha256"):
        if not isinstance(fingerprints.get(key), str) or not hex64.fullmatch(fingerprints[key]):
            raise BridgeContractError(f"e12b_hash:{key}")
    if int(fingerprints.get("canonical_csv_bytes", 0)) <= 0:
        raise BridgeContractError("e12b_csv_size")

    summaries = result.get("task_prediction_summaries") or {}
    if set(summaries) != set(tasks):
        raise BridgeContractError("e12b_task_summary_set")
    for task in tasks:
        row = summaries[task]
        if row.get("task") != task or int(row.get("rows", -1)) != 40:
            raise BridgeContractError(f"e12b_task_summary_identity:{task}")
        minimum = float(row.get("prediction_min"))
        maximum = float(row.get("prediction_max"))
        unique = int(row.get("prediction_unique", 0))
        tie = float(row.get("tie_fraction"))
        if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum < minimum:
            raise BridgeContractError(f"e12b_task_summary_range:{task}")
        if unique < 1 or unique > 40 or not math.isfinite(tie) or abs(tie - (1.0 - unique / 40.0)) > 1e-12:
            raise BridgeContractError(f"e12b_task_summary_unique:{task}")
        if not isinstance(row.get("prediction_sha256"), str) or not hex64.fullmatch(row["prediction_sha256"]):
            raise BridgeContractError(f"e12b_task_hash:{task}")
    if int(summaries["Task1.3"].get("prediction_unique", 0)) != 36:
        raise BridgeContractError("e12b_task13_unique")

    controls = result.get("structural_controls") or {}
    if controls.get("final_changed_tasks_vs_regenerated_b21") != ["Task1.2", "Task1.3"]:
        raise BridgeContractError("e12b_changed_vs_b21")
    if controls.get("final_changed_tasks_vs_regenerated_task12_only") != ["Task1.3"]:
        raise BridgeContractError("e12b_changed_vs_task12")
    if controls.get("historical_public_0_218_is_not_assumed_byte_identical") is not True:
        raise BridgeContractError("e12b_public_history_boundary")
    for key in ("regenerated_b21_semantic_sha256", "regenerated_task12_only_semantic_sha256"):
        if not isinstance(controls.get(key), str) or not hex64.fullmatch(controls[key]):
            raise BridgeContractError(f"e12b_control_hash:{key}")
    for key, expected_task in (
        ("task12_final_vs_b21", "Task1.2"),
        ("task13_final_vs_b21", "Task1.3"),
        ("task13_final_vs_task12_only", "Task1.3"),
    ):
        comp = controls.get(key) or {}
        if comp.get("task") != expected_task:
            raise BridgeContractError(f"e12b_rank_comparison_identity:{key}")
        value = comp.get("rank_spearman")
        if value is not None and not math.isfinite(float(value)):
            raise BridgeContractError(f"e12b_rank_comparison_nonfinite:{key}")
        changed = int(comp.get("changed_rank_count", -1))
        mean_shift = float(comp.get("mean_absolute_percentile_shift", -1))
        max_shift = float(comp.get("maximum_absolute_percentile_shift", -1))
        if changed < 0 or changed > 40 or not math.isfinite(mean_shift) or not math.isfinite(max_shift) or mean_shift < 0 or max_shift < 0:
            raise BridgeContractError(f"e12b_rank_comparison_range:{key}")

    task13 = result.get("task13_reconciliation") or {}
    if task13.get("final_incumbent") != "strict_asc_anchor":
        raise BridgeContractError("e12b_task13_incumbent")
    if task13.get("predictor_feature") != "flow_rank__Antibody-secreting_cells_(ASC)":
        raise BridgeContractError("e12b_task13_feature")
    if int(task13.get("challenge_unique_values", 0)) != 36 or task13.get("e12a_v2_reproduced") is not True:
        raise BridgeContractError("e12b_task13_parent_gate")

    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"', '"subject_group"', '"row_index"', '"challenge_predictions"', '"submission_rows"', '"prediction_vector"', '"oof_predictions"')
    if any(token in serialized for token in banned):
        raise BridgeContractError("e12b_aggregate_contains_row_level_key")
'''
    runtime = replace_function(runtime, "validate_result", validator)

    summary = r'''def render_summary(result: dict) -> str:
    validate_result(result)
    fp = result["fingerprint_contract"]
    controls = result["structural_controls"]
    lines = [
        "# CMI-Flu Strategy-v2 E12b Challenge prediction freeze", "",
        "Aggregate-only fingerprint audit of the fixed 40-donor seven-task candidate. The row-level candidate was not persisted and no Competition submission or Public query was performed.", "",
        f"- science commit: `{SCIENCE_COMMIT}`",
        f"- semantic submission SHA-256: `{fp['semantic_submission_sha256']}`",
        f"- canonical CSV SHA-256: `{fp['canonical_csv_sha256']}`",
        f"- canonical CSV bytes: `{fp['canonical_csv_bytes']}`",
        f"- next step: `{result['next_step']}`", "",
        "## Frozen portfolio", "",
    ]
    for task, name in result["final_portfolio"].items():
        lines.append(f"- {task}: `{name}`")
    lines += ["", "## Structural controls", ""]
    lines.append(f"- changed vs regenerated B2.1: `{','.join(controls['final_changed_tasks_vs_regenerated_b21'])}`")
    lines.append(f"- changed vs regenerated Task1.2-only: `{','.join(controls['final_changed_tasks_vs_regenerated_task12_only'])}`")
    for key in ("task12_final_vs_b21", "task13_final_vs_b21"):
        comp = controls[key]
        lines.append(
            f"- {key}: rank_spearman={comp.get('rank_spearman')}, changed_ranks={comp.get('changed_rank_count')}, "
            f"mean_abs_percentile_shift={comp.get('mean_absolute_percentile_shift')}, max_abs_percentile_shift={comp.get('maximum_absolute_percentile_shift')}"
        )
    lines += ["", "## Per-task aggregate fingerprints", ""]
    for task in TASKS:
        row = result["task_prediction_summaries"][task]
        lines.append(
            f"- {task}: unique={row['prediction_unique']}, min={row['prediction_min']:.12g}, max={row['prediction_max']:.12g}, sha256=`{row['prediction_sha256']}`"
        )
    text = "\n".join(lines) + "\n"
    if "None" in text or not text.startswith("# CMI-Flu Strategy-v2 E12b Challenge prediction freeze\n"):
        raise BridgeContractError("e12b_summary_contract")
    return text
'''
    runtime = replace_function(runtime, "render_summary", summary)

    terminal = r'''def terminal_success_line(result: dict, md5_verified: int) -> str:
    validate_result(result)
    return (
        "CMI_FLU_E12B_COMPLETE "
        f"request_id={REQUEST_ID} challenge_rows={result['challenge_rows']} portfolio_tasks={result['portfolio_task_count']} "
        f"md5_verified={int(md5_verified)} row_persisted=false submission=false leaderboard_selection=false"
    )
'''
    runtime = replace_function(runtime, "terminal_success_line", terminal)

    self_test = r'''def self_test() -> int:
    package = package_bytes()
    if git_blob_sha(E01_SOURCE.encode("utf-8")) != E01_BLOB:
        raise BridgeContractError("e01_blob_mismatch")
    if git_blob_sha(E01_V2_SOURCE.encode("utf-8")) != E01_V2_BLOB:
        raise BridgeContractError("e01_v2_blob_mismatch")
    if git_blob_sha(CONFIG_TEXT.encode("utf-8")) != CONFIG_BLOB:
        raise BridgeContractError("config_blob_mismatch")
    if git_blob_sha(E12A_V2_SOURCE.encode("utf-8")) != E12A_V2_BLOB:
        raise BridgeContractError("e12a_v2_blob_mismatch")
    if git_blob_sha(E12B_SOURCE.encode("utf-8")) != E12B_BLOB:
        raise BridgeContractError("e12b_blob_mismatch")
    compile(B21_ADAPTER_SOURCE, "cmi_flu_b21_runtime_adapter.py", "exec")
    compile(E01_SOURCE, "cmi_flu/strategy_e01.py", "exec")
    compile(E01_V2_SOURCE, "cmi_flu/strategy_e01_v2.py", "exec")
    compile(E12A_V2_SOURCE, "cmi_flu/strategy_e12a_v2.py", "exec")
    compile(E12B_SOURCE, "cmi_flu/strategy_e12b.py", "exec")
    h = "0" * 64
    columns = ["participant_id", "Task1.1", "Task1.2", "Task1.3", "Task1.4", "Task2.1", "Task2.2", "Task2.3"]
    portfolio = {
        "Task1.1": "b21_pls_2", "Task1.2": "task12_anchor_residual_et_d5_l5_sqrt_lambda0.5",
        "Task1.3": "strict_asc_anchor", "Task1.4": "raw_pre_vacc_conserved_anchor",
        "Task2.1": "b21_et_subtype_d3_l5", "Task2.2": "b21_et_subtype_d5_l10", "Task2.3": "b21_ridge_exact_a100",
    }
    summaries = {}
    for task in portfolio:
        unique = 36 if task == "Task1.3" else 40
        summaries[task] = {
            "task": task, "rows": 40, "prediction_min": 0.1, "prediction_max": 1.0,
            "prediction_unique": unique, "tie_fraction": 1.0 - unique / 40.0,
            "prediction_sha256": h,
        }
    comp12 = {"task": "Task1.2", "rank_spearman": 0.9, "rank_spearman_status": "ok", "changed_rank_count": 20, "mean_absolute_percentile_shift": 0.05, "maximum_absolute_percentile_shift": 0.2}
    comp13 = {"task": "Task1.3", "rank_spearman": 0.8, "rank_spearman_status": "ok", "changed_rank_count": 30, "mean_absolute_percentile_shift": 0.1, "maximum_absolute_percentile_shift": 0.4}
    fixture = {
        "schema_version": 1, "experiment": "strategy_v2_e12b_challenge_prediction_freeze",
        "final_portfolio": portfolio, "portfolio_task_count": 7, "challenge_rows": 40,
        "submission_validation": {"rows": 40, "columns": columns, "task_unique_counts": {task: summaries[task]["prediction_unique"] for task in portfolio}, "minus99_tasks": []},
        "fingerprint_contract": {"semantic_hash_version": "cmi-flu-e12b-semantic-v1", "semantic_submission_sha256": h, "canonical_csv_float_format": "%.17g", "canonical_csv_lineterminator": "LF", "canonical_csv_sha256": h, "canonical_csv_bytes": 4096},
        "task_prediction_summaries": summaries,
        "structural_controls": {
            "b21_portfolio": {}, "task12_only_portfolio": {},
            "final_changed_tasks_vs_regenerated_b21": ["Task1.2", "Task1.3"],
            "final_changed_tasks_vs_regenerated_task12_only": ["Task1.3"],
            "regenerated_b21_semantic_sha256": h, "regenerated_task12_only_semantic_sha256": h,
            "historical_public_0_218_is_not_assumed_byte_identical": True,
            "task12_final_vs_b21": comp12, "task13_final_vs_b21": comp13, "task13_final_vs_task12_only": comp13,
        },
        "task13_reconciliation": {"final_incumbent": "strict_asc_anchor", "predictor_feature": "flow_rank__Antibody-secreting_cells_(ASC)", "challenge_unique_values": 36, "e12a_v2_reproduced": True},
        "new_model_selection_performed": False, "leaderboard_used_for_selection": False,
        "public_probe_performed": False, "competition_submission_attempted": False,
        "competition_submission_authorized": False, "contains_participant_identifiers": False,
        "contains_row_level_predictions": False, "row_level_candidate_persisted": False,
        "next_step": "ready_for_separate_submission_authorization",
    }
    validate_result(fixture)
    summary = render_summary(fixture)
    line = terminal_success_line(fixture, 28)
    if not summary.startswith("# CMI-Flu Strategy-v2 E12b Challenge prediction freeze\n") or "challenge_rows=40" not in line:
        raise BridgeContractError("e12b_terminal_summary_self_test")
    print(
        "CMI_FLU_E12B_RUNTIME_SELF_TEST PASS "
        f"request_id={REQUEST_ID} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} e12b_blob={E12B_BLOB}"
    )
    return 0
'''
    runtime = replace_function(runtime, "self_test", self_test)

    execute = r'''def execute(input_dir: Path, output_dir: Path) -> int:
    runtime_root = Path("/tmp") / "cmi-flu-e12b-runtime"
    stage = "initialize"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        runtime_root.mkdir(parents=True)
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
        stage = "load_e12b"
        run_e12b = load_e12b_module()
        stage = "run_e12b"
        candidate, aggregate = run_e12b(config, inputs)
        result = json_safe(dict(aggregate))
        del candidate, aggregate
        stage = "validate_e12b"
        validate_result(result)
        shutil.rmtree(runtime_root, ignore_errors=True)
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
            "strategy_e12b_blob_sha": E12B_BLOB,
            "strategy_e12b_contract_blob_sha": E12B_CONTRACT_BLOB,
            "strategy_e12a_v2_blob_sha": E12A_V2_BLOB,
            "config_blob_sha": CONFIG_BLOB,
            "package_sha256": PACKAGE_SHA256,
            "python_version": platform.python_version(),
            "md5_verified_count": len(inputs.checksum_report.verified),
            "metrics_sha256": sha256_file(metrics_path),
            "summary_sha256": sha256_file(summary_path),
            "frozen_portfolio": result["final_portfolio"],
            "semantic_submission_sha256": result["fingerprint_contract"]["semantic_submission_sha256"],
            "canonical_csv_sha256": result["fingerprint_contract"]["canonical_csv_sha256"],
            "next_step": result["next_step"],
            "row_level_candidate_persisted": False,
            "competition_submission_attempted": False,
            "leaderboard_used_for_selection": False,
            "contains_participant_identifiers": False,
            "contains_row_level_predictions": False,
        }
        bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(terminal_success_line(result, len(inputs.checksum_report.verified)))
        return 0
    except Exception as exc:
        shutil.rmtree(runtime_root, ignore_errors=True)
        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode("utf-8", errors="replace")).hexdigest()[:20]
        print(f"CMI_FLU_E12B_FAILED stage={stage} exception_type={type(exc).__name__} error_code={code}", file=sys.stderr)
        return 2
'''
    runtime = replace_function(runtime, "execute", execute)
    runtime = runtime.replace("CMI_FLU_E12A_V2_FAILED", "CMI_FLU_E12B_FAILED")

    required_runtime = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'TARGET_KERNEL = "{TARGET}"',
        f'SCIENCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'E12B_BLOB = "{E12B_BLOB}"',
        f'E12B_CONTRACT_BLOB = "{E12B_CONTRACT_BLOB}"',
        "def load_e12b_module(",
        "def terminal_success_line(",
        "CMI_FLU_E12B_COMPLETE",
        '"Task1.3": "strict_asc_anchor"',
        '"row_level_candidate_persisted": False',
    )
    if any(token not in runtime for token in required_runtime):
        raise SystemExit("E12b generated runtime contract incomplete")
    if E12A_TARGET in runtime or E12A_REQUEST in runtime:
        raise SystemExit("E12b runtime references consumed E12a-v2 identity")
    lowered = runtime.casefold()
    if "kaggle competitions submit" in lowered or "competition_submit(" in lowered:
        raise SystemExit("E12b generated runtime contains submission path")
    raw = runtime.encode("utf-8")
    if len(raw) >= 900000:
        raise SystemExit(f"E12b runtime too large:{len(raw)}")
    compile(runtime, "generated_e12b.py", "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    return runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validate_request(root)
    source, _, _ = load_science(root)
    runtime = build_runtime(root, output, source)
    encoded = runtime.encode("utf-8")
    print(
        "CMI_FLU_E12B_PREPARE PASS "
        f"request_id={REQUEST_ID} science_commit={SCIENCE_COMMIT} e12b_blob={E12B_BLOB} "
        f"runtime_bytes={len(encoded)} runtime_sha256={hashlib.sha256(encoded).hexdigest()} "
        "row_persisted=false submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
