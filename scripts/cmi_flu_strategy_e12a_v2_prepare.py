#!/usr/bin/env python3
"""Build E12a-v2 from the proven E01-repair runtime plus exact reconciled science."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REQUEST_ID = "20260911-cmi-flu-strategy-e12a-v2-portfolio-reconciliation-001"
TARGET = "renta0426/cmi-flu-e12a-v2-portfolio-reconcile-20260911-001"
SCIENCE_COMMIT = "e6e05e578f172ce710bc0f1da55acdf290e83dd9"
E12A_V2_BLOB = "0c4c970c8bacfed61bfbb9587e0a0bfdec7903d9"
E12A_V2_SYNTH_BLOB = "6477076d9c5aba7d37d816147e4443b5c94ad2d4"
E12A_V2_CONTRACT_BLOB = "6b40ac35a3b43c3bafcdd08fcc49fc9fe22960c1"
E01_BLOB = "dd27aea0cf97d41bad3cec64819c4c4269d94cbd"
E01_V2_BLOB = "8cc64dc5ab9483d5957cfada18d445188566c56c"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e12a-v2-portfolio-reconciliation-001"
E12A_V2_PATH = f"{PAYLOAD_ROOT}/strategy_e12a_v2.py"
E12A_V2_SYNTH_PATH = f"{PAYLOAD_ROOT}/strategy_e12a_v2_synthetic.py"
E12A_V2_CONTRACT_PATH = f"{PAYLOAD_ROOT}/strategy_e12a_v2_portfolio_reconciliation.json"
REQUEST_PATH = "requests/cmi-flu-strategy-e12a-v2-portfolio-reconciliation-001.json"
BASE_BUILDER = "scripts/cmi_flu_strategy_e01_prepare_v2.py"
BASE_REQUEST = "20260907-cmi-flu-strategy-e01-paired-evaluation-repair-002"
BASE_TARGET = "renta0426/cmi-flu-e01-paired-eval-repair-20260907-002"
BASE_SCIENCE = "0b2ecb47eaa09f22450424c9c06dc88cf44bc1fb"
OLD_E12A_TARGET = "renta0426/cmi-flu-e12a-final-reproduction-20260910-001"
SUMMARY_HEADING = "# CMI-Flu Strategy-v2 E12a-v2 final portfolio reconciliation"


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
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    ]
    if len(nodes) != 1:
        raise SystemExit(f"E12a-v2 runtime function binding changed:{name}:{len(nodes)}")
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
        "strategy_e12a_v2_blob_sha": E12A_V2_BLOB,
        "strategy_e12a_v2_synthetic_blob_sha": E12A_V2_SYNTH_BLOB,
        "strategy_e12a_v2_contract_blob_sha": E12A_V2_CONTRACT_BLOB,
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
            raise SystemExit(f"E12a-v2 request mismatch:{key}")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 30,
        "hard_timeout_minutes": 60,
        "max_active_runs": 1,
    }:
        raise SystemExit("E12a-v2 resource contract mismatch")
    if request.get("api_budget") != {
        "max_calls": 60,
        "poll_interval_seconds": 120,
        "max_polls": 35,
        "max_pages": 2,
    }:
        raise SystemExit("E12a-v2 API budget mismatch")
    parent = request.get("parent_e12a_v1") or {}
    if parent != {
        "request_id": "20260910-cmi-flu-strategy-e12a-final-reproduction-001",
        "target": OLD_E12A_TARGET,
        "version": 1,
        "actions_run": 34536723038,
        "actions_job": 103070014918,
        "savekernel_succeeded": True,
        "kernel_version_consumed": True,
        "scientific_outputs_recovered": True,
        "same_version_rerun_forbidden": True,
        "portfolio_disposition": "reproduction_harness_passed_final_portfolio_contract_invalid",
        "runtime_incident_classification": "bridge_runtime_post_artifact_terminal_log_defect",
    }:
        raise SystemExit("E12a-v2 parent incident provenance mismatch")
    exp = request.get("experiment_contract") or {}
    expected_exp = {
        "experiment": "strategy_v2_e12a_v2_portfolio_reconciliation",
        "comparison_contract": "paired_subject_purged_v2",
        "reproduction_absolute_tolerance": 0.000002,
        "e11_task11_candidates_carried": [],
        "portfolio_task_count": 7,
        "corrected_task": "Task1.3",
        "stale_task13_e01_control": "b21_pls_1",
        "final_task13_incumbent": "strict_asc_anchor",
        "task13_anchor_feature": "flow_rank__Antibody-secreting_cells_(ASC)",
        "task13_expected_train_rows": 23,
        "task13_expected_challenge_rows": 40,
        "task13_expected_study_equal_spearman": 0.38158872734833993,
        "new_model_selection_allowed": False,
        "task14_supervised_cv_available": False,
        "public_probe_authorized": False,
        "competition_submission_authorized": False,
        "if_pass": "proceed_to_E12b_challenge_prediction_freeze",
        "if_fail": "investigate_reconciliation_contract_no_model_selection",
    }
    if exp != expected_exp:
        raise SystemExit("E12a-v2 experiment contract mismatch")
    if request.get("summary_contract") != {
        "experiment_heading": SUMMARY_HEADING,
        "must_not_contain_none_placeholder": True,
        "must_match_metrics": True,
        "must_show_task13_strict_anchor": True,
        "must_not_show_b21_pls_1_as_final_task13": True,
    }:
        raise SystemExit("E12a-v2 summary contract mismatch")
    if request.get("allowed_output_paths") != [
        "bridge-result.json",
        "metrics.json",
        "summary.md",
    ]:
        raise SystemExit("E12a-v2 output allowlist mismatch")
    return request


def load_science(root: Path) -> tuple[str, str, str]:
    source = require_blob(root / E12A_V2_PATH, E12A_V2_BLOB, label="E12a-v2 science").decode("utf-8")
    synthetic = require_blob(
        root / E12A_V2_SYNTH_PATH, E12A_V2_SYNTH_BLOB, label="E12a-v2 synthetic"
    ).decode("utf-8")
    contract_text = require_blob(
        root / E12A_V2_CONTRACT_PATH,
        E12A_V2_CONTRACT_BLOB,
        label="E12a-v2 frozen contract",
    ).decode("utf-8")
    compile(source, "cmi_flu/strategy_e12a_v2.py", "exec")
    compile(synthetic, "cmi_flu/strategy_e12a_v2_synthetic.py", "exec")
    contract = json.loads(contract_text)
    if contract.get("expected_final_portfolio", {}).get("Task1.3") != "strict_asc_anchor":
        raise SystemExit("E12a-v2 contract does not retain strict Task1.3 anchor")
    if contract.get("task13_reconciliation", {}).get("e01_control_identity") != "b21_pls_1":
        raise SystemExit("E12a-v2 historical Task1.3 control changed")
    required = (
        'EXPERIMENT = "strategy_v2_e12a_v2_portfolio_reconciliation"',
        'TASK13_FINAL_INCUMBENT = "strict_asc_anchor"',
        'TASK13_ANCHOR_COLUMN = "flow_rank__Antibody-secreting_cells_(ASC)"',
        "TASK13_EXPECTED_TRAIN_ROWS = 23",
        "TASK13_EXPECTED_CHALLENGE_ROWS = 40",
        "TASK13_EXPECTED_STUDY_EQUAL_SPEARMAN = 0.38158872734833993",
        'TASK13_E01_CONTROL_INCUMBENT = "b21_pls_1"',
        '"final_portfolio_member": False',
        '"new_model_selection_performed": False',
        '"competition_submission_attempted": False',
    )
    if any(token not in source for token in required):
        raise SystemExit("E12a-v2 science contract token missing")
    if "kaggle competitions submit" in source.casefold() or "competition_submit" in source.casefold():
        raise SystemExit("E12a-v2 science source contains submission path")
    return source, synthetic, contract_text


def build_runtime(root: Path, output: Path, source: str) -> str:
    with tempfile.TemporaryDirectory(prefix="cmi-e12a-v2-base-") as tmp:
        base_path = Path(tmp) / "e01.py"
        subprocess.run(
            [
                sys.executable,
                str(root / BASE_BUILDER),
                "--repository-root",
                str(root),
                "--output",
                str(base_path),
            ],
            check=True,
        )
        runtime = base_path.read_text(encoding="utf-8")

    for old, new in (
        (BASE_REQUEST, REQUEST_ID),
        (BASE_TARGET, TARGET),
        (BASE_SCIENCE, SCIENCE_COMMIT),
    ):
        if old not in runtime:
            raise SystemExit(f"E12a-v2 base runtime identity anchor missing:{old}")
        runtime = runtime.replace(old, new)

    target_anchor = f'TARGET_KERNEL = "{TARGET}"\n'
    if runtime.count(target_anchor) != 1:
        raise SystemExit("E12a-v2 target runtime anchor changed")
    runtime = runtime.replace(
        target_anchor,
        target_anchor
        + f'E12A_V2_BLOB = "{E12A_V2_BLOB}"\n'
        + f'E12A_V2_CONTRACT_BLOB = "{E12A_V2_CONTRACT_BLOB}"\n'
        + f'E12A_V2_SOURCE = {source!r}\n',
        1,
    )

    loader = r'''def load_e12a_v2_functions() -> tuple[object, object]:
    module = types.ModuleType("cmi_flu.strategy_e12a_v2")
    module.__file__ = "<cmi_flu.strategy_e12a_v2>"
    module.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_e12a_v2"] = module
    exec(compile(E12A_V2_SOURCE, "cmi_flu/strategy_e12a_v2.py", "exec"), module.__dict__, module.__dict__)
    task13 = getattr(module, "run_strict_task13_anchor_audit", None)
    evaluate = getattr(module, "evaluate_e12a_v2", None)
    if not callable(task13) or not callable(evaluate):
        raise BridgeContractError("e12a_v2_entry_missing")
    return task13, evaluate
'''
    marker = "def load_e01_module() -> object:\n"
    if runtime.count(marker) != 1:
        raise SystemExit("E12a-v2 E01 loader anchor changed")
    runtime = runtime.replace(marker, loader + "\n" + marker, 1)

    old_run = '''        stage = "load_e01"\n        run_e01 = load_e01_module()\n        stage = "run_e01"\n        result = json_safe(dict(run_e01(config, inputs)))\n        stage = "validate_e01"\n        validate_result(result)\n'''
    new_run = '''        stage = "load_e01"\n        run_e01 = load_e01_module()\n        stage = "run_e01"\n        e01_result = json_safe(dict(run_e01(config, inputs)))\n        stage = "load_e12a_v2"\n        run_task13, evaluate_e12a_v2 = load_e12a_v2_functions()\n        stage = "run_task13_strict_anchor"\n        task13_anchor = json_safe(dict(run_task13(config, inputs)))\n        stage = "run_e12a_v2"\n        result = json_safe(dict(evaluate_e12a_v2(e01_result, task13_anchor)))\n        stage = "validate_e12a_v2"\n        validate_result(result)\n'''
    if runtime.count(old_run) != 1:
        raise SystemExit("E12a-v2 execution patch anchor changed")
    runtime = runtime.replace(old_run, new_run, 1)

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
    expected_scores = {
        "Task1.1": 0.09970718035376519,
        "Task1.2": 0.5271033295423541,
        "Task1.3": 0.38158872734833993,
        "Task2.1": 0.623449034547404,
        "Task2.2": 0.576505972340535,
        "Task2.3": 0.6957305642219944,
    }
    if result.get("experiment") != "strategy_v2_e12a_v2_portfolio_reconciliation":
        raise BridgeContractError("e12a_v2_identity")
    if result.get("comparison_contract") != "paired_subject_purged_v2":
        raise BridgeContractError("e12a_v2_contract")
    if result.get("portfolio_manifest_corrected") is not True or result.get("corrected_task") != "Task1.3":
        raise BridgeContractError("e12a_v2_reconciliation_marker")
    if result.get("frozen_portfolio") != expected_portfolio or result.get("portfolio_task_count") != 7:
        raise BridgeContractError("e12a_v2_final_portfolio")
    stale = result.get("stale_task13_e01_control") or {}
    if stale.get("incumbent") != "b21_pls_1" or stale.get("final_portfolio_member") is not False:
        raise BridgeContractError("e12a_v2_stale_task13_boundary")
    stale_score = float(stale.get("study_equal_spearman"))
    if not math.isfinite(stale_score) or abs(stale_score - 0.10630410834652516) > 0.000002:
        raise BridgeContractError("e12a_v2_stale_task13_reproduction")
    gain = float(result.get("task13_strict_anchor_gain_vs_e01_control"))
    if not math.isfinite(gain) or gain <= 0.27:
        raise BridgeContractError("e12a_v2_task13_gain_contract")
    if result.get("e11_task11_candidates_carried") != []:
        raise BridgeContractError("e12a_v2_e11_candidate_boundary")
    for key in (
        "new_model_selection_performed", "competition_incumbent_changed",
        "public_probe_authorized", "competition_submission_authorized",
        "leaderboard_used_for_selection", "competition_submission_attempted",
        "contains_participant_identifiers", "contains_row_level_predictions",
    ):
        if result.get(key) is not False:
            raise BridgeContractError(f"e12a_v2_boundary:{key}")
    rows = result.get("supervised_reproduction") or []
    if len(rows) != 6 or {row.get("task") for row in rows} != set(expected_scores):
        raise BridgeContractError("e12a_v2_supervised_set")
    by = {row["task"]: row for row in rows}
    for task, expected_score in expected_scores.items():
        row = by[task]
        if row.get("incumbent") != expected_portfolio[task]:
            raise BridgeContractError(f"e12a_v2_incumbent:{task}")
        exp = float(row.get("expected_study_equal_spearman"))
        obs = float(row.get("observed_study_equal_spearman"))
        dev = float(row.get("deviation"))
        absolute = float(row.get("absolute_deviation"))
        if not all(math.isfinite(value) for value in (exp, obs, dev, absolute)):
            raise BridgeContractError("e12a_v2_nonfinite")
        if abs(exp - expected_score) > 1e-12 or abs((obs - exp) - dev) > 1e-12 or abs(abs(dev) - absolute) > 1e-12:
            raise BridgeContractError("e12a_v2_reproduction_arithmetic")
        if bool(row.get("reproduced")) is not (absolute <= 0.000002):
            raise BridgeContractError("e12a_v2_tolerance_arithmetic")
    task13 = by["Task1.3"]
    if task13.get("predictor_feature") != "flow_rank__Antibody-secreting_cells_(ASC)":
        raise BridgeContractError("e12a_v2_task13_feature")
    if int(task13.get("challenge_rows", -1)) != 40 or task13.get("challenge_anchor_complete") is not True:
        raise BridgeContractError("e12a_v2_task13_challenge_contract")
    if int(task13.get("challenge_anchor_unique_values", 0)) < 2:
        raise BridgeContractError("e12a_v2_task13_challenge_constant")
    all_pass = all(bool(by[task].get("reproduced")) for task in expected_scores)
    if result.get("all_supervised_tasks_reproduced") is not all_pass:
        raise BridgeContractError("e12a_v2_all_pass")
    expected_next = "proceed_to_E12b_challenge_prediction_freeze" if all_pass else "investigate_reconciliation_contract_no_model_selection"
    if result.get("next_step") != expected_next:
        raise BridgeContractError("e12a_v2_next_step")
    task14 = result.get("task14_contract") or {}
    if task14.get("incumbent") != "raw_pre_vacc_conserved_anchor" or task14.get("supervised_cv_available") is not False or task14.get("outcomes_accessed") is not False or int(task14.get("challenge_subjects_expected", -1)) != 40:
        raise BridgeContractError("e12a_v2_task14_contract")
'''
    runtime = replace_function(runtime, "validate_result", validator)

    summary = r'''def render_summary(result: dict) -> str:
    validate_result(result)
    lines = [
        "# CMI-Flu Strategy-v2 E12a-v2 final portfolio reconciliation", "",
        "Aggregate-only reconciliation of the already-selected competition portfolio; no new model selection, Public probe, or Competition submission was performed.", "",
        f"- science commit: `{SCIENCE_COMMIT}`",
        f"- corrected task: `{result['corrected_task']}`",
        f"- all supervised final components reproduced: `{str(bool(result['all_supervised_tasks_reproduced'])).lower()}`",
        f"- next step: `{result['next_step']}`", "",
        "## Supervised final-component reproduction", "",
    ]
    for row in result["supervised_reproduction"]:
        lines.append(
            f"- {row['task']}: incumbent `{row['incumbent']}`, expected={row['expected_study_equal_spearman']:.12f}, "
            f"observed={row['observed_study_equal_spearman']:.12f}, abs_delta={row['absolute_deviation']:.9g}, "
            f"reproduced={str(bool(row['reproduced'])).lower()}"
        )
    stale = result["stale_task13_e01_control"]
    lines += [
        "", "## Task1.3 reconciliation", "",
        f"- final incumbent: `{result['frozen_portfolio']['Task1.3']}`",
        "- final predictor feature: `flow_rank__Antibody-secreting_cells_(ASC)`",
        f"- stale E01 control: `{stale['incumbent']}`; final portfolio member=`{str(bool(stale['final_portfolio_member'])).lower()}`",
        f"- strict-anchor gain versus stale E01 control: `{result['task13_strict_anchor_gain_vs_e01_control']:.12f}`",
        "", "## Task1.4", "",
        f"- incumbent: `{result['task14_contract']['incumbent']}`",
        f"- supervised CV available: `{str(bool(result['task14_contract']['supervised_cv_available'])).lower()}`",
        "", "## Frozen final portfolio", "",
    ]
    for task, name in result["frozen_portfolio"].items():
        lines.append(f"- {task}: `{name}`")
    text = "\n".join(lines) + "\n"
    if "None" in text or not text.startswith("# CMI-Flu Strategy-v2 E12a-v2 final portfolio reconciliation\n"):
        raise BridgeContractError("e12a_v2_summary_contract")
    return text
'''
    runtime = replace_function(runtime, "render_summary", summary)

    terminal_helper = r'''def terminal_success_line(result: dict, md5_verified: int) -> str:
    if result.get("experiment") != "strategy_v2_e12a_v2_portfolio_reconciliation":
        raise BridgeContractError("e12a_v2_terminal_identity")
    if int(result.get("portfolio_task_count", -1)) != 7:
        raise BridgeContractError("e12a_v2_terminal_portfolio_count")
    if type(result.get("all_supervised_tasks_reproduced")) is not bool:
        raise BridgeContractError("e12a_v2_terminal_reproduction_flag")
    if not isinstance(result.get("next_step"), str) or not result["next_step"]:
        raise BridgeContractError("e12a_v2_terminal_next_step")
    return (
        "CMI_FLU_E12A_V2_COMPLETE "
        f"request_id={REQUEST_ID} portfolio_tasks={result['portfolio_task_count']} "
        f"all_reproduced={str(result['all_supervised_tasks_reproduced']).lower()} "
        f"md5_verified={int(md5_verified)} submission=false leaderboard_selection=false"
    )
'''
    execute_marker = "def execute(input_dir: Path, output_dir: Path) -> int:\n"
    if runtime.count(execute_marker) != 1:
        raise SystemExit("E12a-v2 execute anchor changed")
    runtime = runtime.replace(execute_marker, terminal_helper + "\n" + execute_marker, 1)

    old_bridge = '            "frozen_incumbent": result["frozen_incumbent"],\n'
    if runtime.count(old_bridge) != 1:
        raise SystemExit("E12a-v2 bridge-result E01 field anchor changed")
    runtime = runtime.replace(
        old_bridge,
        '            "frozen_portfolio": result["frozen_portfolio"],\n'
        '            "all_supervised_tasks_reproduced": result["all_supervised_tasks_reproduced"],\n'
        '            "portfolio_manifest_corrected": result["portfolio_manifest_corrected"],\n'
        '            "corrected_task": result["corrected_task"],\n'
        '            "next_step": result["next_step"],\n'
        '            "strategy_e12a_v2_blob_sha": E12A_V2_BLOB,\n'
        '            "strategy_e12a_v2_contract_blob_sha": E12A_V2_CONTRACT_BLOB,\n',
        1,
    )

    old_success = '''        print(\n            "CMI_FLU_E01_COMPLETE "\n            f"request_id={REQUEST_ID} tasks={len(result['tasks'])} md5_verified={len(inputs.checksum_report.verified)} "\n            "submission=false leaderboard_selection=false"\n        )\n        return 0\n'''
    new_success = '''        print(terminal_success_line(result, len(inputs.checksum_report.verified)))\n        return 0\n'''
    if runtime.count(old_success) != 1:
        raise SystemExit("E12a-v2 terminal E01 success anchor changed")
    runtime = runtime.replace(old_success, new_success, 1)
    if "len(result['tasks'])" in runtime:
        raise SystemExit("E12a-v2 inherited terminal Task1.3/schema defect remains")
    runtime = runtime.replace("CMI_FLU_E01_FAILED", "CMI_FLU_E12A_V2_FAILED")

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
    compile(B21_ADAPTER_SOURCE, "cmi_flu_b21_runtime_adapter.py", "exec")
    compile(E01_SOURCE, "cmi_flu/strategy_e01.py", "exec")
    compile(E01_V2_SOURCE, "cmi_flu/strategy_e01_v2.py", "exec")
    compile(E12A_V2_SOURCE, "cmi_flu/strategy_e12a_v2.py", "exec")
    fixture = {
        "experiment": "strategy_v2_e12a_v2_portfolio_reconciliation",
        "portfolio_task_count": 7,
        "all_supervised_tasks_reproduced": True,
        "next_step": "proceed_to_E12b_challenge_prediction_freeze",
    }
    terminal = terminal_success_line(fixture, 28)
    if "portfolio_tasks=7" not in terminal or "all_reproduced=true" not in terminal:
        raise BridgeContractError("e12a_v2_terminal_self_test")
    if "tasks=" in terminal:
        raise BridgeContractError("e12a_v2_terminal_inherited_schema")
    print(
        "CMI_FLU_E12A_V2_RUNTIME_SELF_TEST PASS "
        f"request_id={REQUEST_ID} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} "
        f"e12a_v2_blob={E12A_V2_BLOB} terminal_path=true"
    )
    return 0
'''
    runtime = replace_function(runtime, "self_test", self_test)

    raw = runtime.encode("utf-8")
    if len(raw) >= 900000:
        raise SystemExit(f"E12a-v2 runtime too large:{len(raw)}")
    required_runtime = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'TARGET_KERNEL = "{TARGET}"',
        f'SCIENCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'E12A_V2_BLOB = "{E12A_V2_BLOB}"',
        '"Task1.3": "strict_asc_anchor"',
        '"incumbent": "b21_pls_1"',
        '"final_portfolio_member": False',
        "def terminal_success_line(",
        "CMI_FLU_E12A_V2_COMPLETE",
    )
    if any(token not in runtime for token in required_runtime):
        raise SystemExit("E12a-v2 generated runtime contract incomplete")
    if OLD_E12A_TARGET in runtime:
        raise SystemExit("E12a-v2 generated runtime references consumed E12a-v1 target")
    if "kaggle competitions submit" in runtime.casefold() or "competition_submit" in runtime.casefold():
        raise SystemExit("E12a-v2 runtime contains submission path")
    compile(runtime, "generated_e12a_v2.py", "exec")
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
    print(
        "CMI_FLU_E12A_V2_PREPARE PASS "
        f"request_id={REQUEST_ID} science_commit={SCIENCE_COMMIT} e12a_v2_blob={E12A_V2_BLOB} "
        f"runtime_bytes={len(runtime.encode('utf-8'))} runtime_sha256={hashlib.sha256(runtime.encode('utf-8')).hexdigest()} "
        "terminal_success_path_tested=true submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
