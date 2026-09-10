#!/usr/bin/env python3
"""Build E12a from the proven E01-repair runtime plus exact E12a science."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REQUEST_ID = "20260910-cmi-flu-strategy-e12a-final-reproduction-001"
TARGET = "renta0426/cmi-flu-e12a-final-reproduction-20260910-001"
SCIENCE_COMMIT = "f227694aee240a05de1b4e318a8c17acc2db5651"
E12A_BLOB = "84c9368d520b2b7f88f189d9d9a92a3306e54b0e"
E01_BLOB = "dd27aea0cf97d41bad3cec64819c4c4269d94cbd"
E01_V2_BLOB = "8cc64dc5ab9483d5957cfada18d445188566c56c"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
PAYLOAD = "payloads/cmi-flu-strategy-e12a-final-reproduction-001/strategy_e12a.py"
REQUEST_PATH = "requests/cmi-flu-strategy-e12a-final-reproduction-001.json"
BASE_BUILDER = "scripts/cmi_flu_strategy_e01_prepare_v2.py"
BASE_REQUEST = "20260907-cmi-flu-strategy-e01-paired-evaluation-repair-002"
BASE_TARGET = "renta0426/cmi-flu-e01-paired-eval-repair-20260907-002"
BASE_SCIENCE = "0b2ecb47eaa09f22450424c9c06dc88cf44bc1fb"
SUMMARY_HEADING = "# CMI-Flu Strategy-v2 E12a final-system reproduction audit"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def replace_function(text: str, name: str, replacement: str) -> str:
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(nodes) != 1:
        raise SystemExit(f"E12a runtime function binding changed:{name}:{len(nodes)}")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno:])


def validate_request(root: Path) -> dict:
    request = json.loads((root / REQUEST_PATH).read_text())
    exact = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET,
        "science_repository": "renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": "connector_verified_exact_blob_relay",
        "strategy_e12a_blob_sha": E12A_BLOB,
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
            raise SystemExit(f"E12a request mismatch:{key}")
    if request.get("resource") != {"accelerator":"cpu","expected_runtime_minutes":30,"hard_timeout_minutes":60,"max_active_runs":1}:
        raise SystemExit("E12a resource contract mismatch")
    exp = request.get("experiment_contract") or {}
    expected_exp = {
        "experiment":"strategy_v2_e12a_final_system_reproduction",
        "comparison_contract":"paired_subject_purged_v2",
        "reproduction_absolute_tolerance":0.000002,
        "e11_task11_candidates_carried":[],
        "portfolio_task_count":7,
        "new_model_selection_allowed":False,
        "task14_supervised_cv_available":False,
        "public_probe_authorized":False,
        "competition_submission_authorized":False,
        "if_pass":"proceed_to_E12b_challenge_prediction_freeze",
        "if_fail":"investigate_reproduction_contract_no_model_selection",
    }
    if exp != expected_exp:
        raise SystemExit("E12a experiment contract mismatch")
    if request.get("summary_contract") != {"experiment_heading":SUMMARY_HEADING,"must_not_contain_none_placeholder":True,"must_match_metrics":True}:
        raise SystemExit("E12a summary contract mismatch")
    if request.get("allowed_output_paths") != ["bridge-result.json","metrics.json","summary.md"]:
        raise SystemExit("E12a output allowlist mismatch")
    return request


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.repository_root.resolve()
    validate_request(root)
    source_bytes = (root / PAYLOAD).read_bytes()
    if git_blob(source_bytes) != E12A_BLOB:
        raise SystemExit("E12a science relay blob mismatch")
    e12a_source = source_bytes.decode()
    compile(e12a_source, "cmi_flu/strategy_e12a.py", "exec")
    if "kaggle competitions submit" in e12a_source.casefold() or "competition_submit" in e12a_source.casefold():
        raise SystemExit("E12a science source contains submission path")

    with tempfile.TemporaryDirectory(prefix="cmi-e12a-base-") as tmp:
        runtime_path = Path(tmp) / "e01.py"
        subprocess.run([sys.executable, str(root / BASE_BUILDER), "--repository-root", str(root), "--output", str(runtime_path)], check=True)
        runtime = runtime_path.read_text()

    for old, new in ((BASE_REQUEST, REQUEST_ID), (BASE_TARGET, TARGET), (BASE_SCIENCE, SCIENCE_COMMIT)):
        if old not in runtime:
            raise SystemExit(f"E12a base runtime identity anchor missing:{old}")
        runtime = runtime.replace(old, new)

    target_anchor = f'TARGET_KERNEL = "{TARGET}"\n'
    if runtime.count(target_anchor) != 1:
        raise SystemExit("E12a target runtime anchor changed")
    runtime = runtime.replace(target_anchor, target_anchor + f'E12A_BLOB = "{E12A_BLOB}"\nE12A_SOURCE = {e12a_source!r}\n', 1)

    loader = r'''def load_e12a_evaluator() -> object:
    module = types.ModuleType("cmi_flu.strategy_e12a")
    module.__file__ = "<cmi_flu.strategy_e12a>"
    module.__package__ = "cmi_flu"
    sys.modules["cmi_flu.strategy_e12a"] = module
    exec(compile(E12A_SOURCE, "cmi_flu/strategy_e12a.py", "exec"), module.__dict__, module.__dict__)
    fn = getattr(module, "evaluate_e12a_from_e01_result", None)
    if not callable(fn):
        raise BridgeContractError("e12a_evaluator_missing")
    return fn
'''
    marker = "def load_e01_module() -> object:\n"
    if runtime.count(marker) != 1:
        raise SystemExit("E12a E01 loader anchor changed")
    runtime = runtime.replace(marker, loader + "\n" + marker, 1)

    old_run = '''        stage = "load_e01"\n        run_e01 = load_e01_module()\n        stage = "run_e01"\n        result = json_safe(dict(run_e01(config, inputs)))\n        stage = "validate_e01"\n        validate_result(result)\n'''
    new_run = '''        stage = "load_e01"\n        run_e01 = load_e01_module()\n        stage = "run_e01"\n        e01_result = json_safe(dict(run_e01(config, inputs)))\n        stage = "load_e12a"\n        evaluate_e12a = load_e12a_evaluator()\n        stage = "run_e12a"\n        result = json_safe(dict(evaluate_e12a(e01_result)))\n        stage = "validate_e12a"\n        validate_result(result)\n'''
    if runtime.count(old_run) != 1:
        raise SystemExit("E12a execution patch anchor changed")
    runtime = runtime.replace(old_run, new_run, 1)

    validator = r'''def validate_result(result: dict) -> None:
    if result.get("experiment") != "strategy_v2_e12a_final_system_reproduction": raise BridgeContractError("e12a_identity")
    if result.get("comparison_contract") != "paired_subject_purged_v2": raise BridgeContractError("e12a_contract")
    if result.get("portfolio_task_count") != 7: raise BridgeContractError("e12a_portfolio_count")
    if result.get("e11_task11_candidates_carried") != []: raise BridgeContractError("e12a_e11_candidate_boundary")
    if result.get("new_model_selection_performed") is not False or result.get("incumbent_changed") is not False: raise BridgeContractError("e12a_selection_boundary")
    for key in ("public_probe_authorized","competition_submission_authorized","leaderboard_used_for_selection","competition_submission_attempted","contains_participant_identifiers","contains_row_level_predictions"):
        if result.get(key) is not False: raise BridgeContractError(f"e12a_boundary:{key}")
    rows=result.get("supervised_reproduction") or []
    if len(rows)!=6 or {r.get("task") for r in rows}!={"Task1.1","Task1.2","Task1.3","Task2.1","Task2.2","Task2.3"}: raise BridgeContractError("e12a_supervised_set")
    all_pass=all(bool(r.get("reproduced")) for r in rows)
    if result.get("all_supervised_tasks_reproduced") is not all_pass: raise BridgeContractError("e12a_pass_arithmetic")
    expected_next="proceed_to_E12b_challenge_prediction_freeze" if all_pass else "investigate_reproduction_contract_no_model_selection"
    if result.get("next_step")!=expected_next: raise BridgeContractError("e12a_next_step")
    for r in rows:
        exp=float(r.get("expected_study_equal_spearman")); obs=float(r.get("observed_study_equal_spearman")); dev=float(r.get("deviation")); absolute=float(r.get("absolute_deviation"))
        if not all(math.isfinite(x) for x in (exp,obs,dev,absolute)): raise BridgeContractError("e12a_nonfinite")
        if abs((obs-exp)-dev)>1e-12 or abs(abs(dev)-absolute)>1e-12: raise BridgeContractError("e12a_deviation_arithmetic")
        if bool(r.get("reproduced")) is not (absolute<=0.000002): raise BridgeContractError("e12a_tolerance_arithmetic")
    t14=result.get("task14_contract") or {}
    if t14.get("incumbent")!="raw_pre_vacc_conserved_anchor" or t14.get("supervised_cv_available") is not False or t14.get("outcomes_accessed") is not False: raise BridgeContractError("e12a_task14_contract")
'''
    runtime = replace_function(runtime, "validate_result", validator)

    summary = r'''def render_summary(result: dict) -> str:
    validate_result(result)
    lines=["# CMI-Flu Strategy-v2 E12a final-system reproduction audit","","Aggregate-only reproduction audit; no model selection, Public probe, or Competition submission was performed.","",f"- science commit: `{SCIENCE_COMMIT}`",f"- all supervised tasks reproduced: `{str(bool(result['all_supervised_tasks_reproduced'])).lower()}`",f"- next step: `{result['next_step']}`","","## Supervised reproduction",""]
    for row in result["supervised_reproduction"]:
        lines.append(f"- {row['task']}: incumbent `{row['incumbent']}`, expected={row['expected_study_equal_spearman']:.9f}, observed={row['observed_study_equal_spearman']:.9f}, abs_delta={row['absolute_deviation']:.9g}, reproduced={str(bool(row['reproduced'])).lower()}")
    lines += ["","## Task1.4","",f"- incumbent: `{result['task14_contract']['incumbent']}`",f"- supervised CV available: `{str(bool(result['task14_contract']['supervised_cv_available'])).lower()}`","","## Frozen portfolio",""]
    for task,name in result["frozen_portfolio"].items(): lines.append(f"- {task}: `{name}`")
    text="\n".join(lines)+"\n"
    if "None" in text or not text.startswith("# CMI-Flu Strategy-v2 E12a final-system reproduction audit\n"): raise BridgeContractError("e12a_summary_contract")
    return text
'''
    runtime = replace_function(runtime, "render_summary", summary)

    old_bridge = '            "frozen_incumbent": result["frozen_incumbent"],\n'
    if runtime.count(old_bridge) != 1:
        raise SystemExit("E12a bridge-result E01 field anchor changed")
    runtime = runtime.replace(old_bridge, '            "frozen_portfolio": result["frozen_portfolio"],\n            "all_supervised_tasks_reproduced": result["all_supervised_tasks_reproduced"],\n            "next_step": result["next_step"],\n            "strategy_e12a_blob_sha": E12A_BLOB,\n', 1)
    runtime = runtime.replace("CMI_FLU_E01_COMPLETE", "CMI_FLU_E12A_COMPLETE")
    runtime = runtime.replace("# CMI-Flu strategy E01", "# CMI-Flu Strategy-v2 E12a")

    raw = runtime.encode()
    if len(raw) >= 900000:
        raise SystemExit(f"E12a runtime too large:{len(raw)}")
    if runtime.count("E12A_SOURCE = ") != 1 or E12A_BLOB not in runtime:
        raise SystemExit("E12a runtime science binding missing")
    if "kaggle competitions submit" in runtime.casefold() or "competition_submit" in runtime.casefold():
        raise SystemExit("E12a runtime contains submission path")
    compile(runtime, "generated_e12a.py", "exec")
    args.output.resolve().write_text(runtime)
    subprocess.run([sys.executable, str(args.output.resolve()), "--self-test"], check=True)
    print(f"CMI_FLU_E12A_PREPARE PASS science_commit={SCIENCE_COMMIT} e12a_blob={E12A_BLOB} runtime_bytes={len(raw)} runtime_sha256={hashlib.sha256(raw).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
