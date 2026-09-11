#!/usr/bin/env python3
"""Credential-free end-to-end CI for the frozen E12a-v2 bridge request."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

REQUEST = "requests/cmi-flu-strategy-e12a-v2-portfolio-reconciliation-001.json"
PREPARE = "scripts/cmi_flu_strategy_e12a_v2_prepare_v2.py"
EXECUTE = "scripts/cmi_flu_strategy_e12a_v2_execute.py"
SANITIZE = "scripts/cmi_flu_strategy_e12a_v2_sanitize.py"
TARGET = "renta0426/cmi-flu-e12a-v2-portfolio-reconcile-20260911-001"
SCIENCE_COMMIT = "e6e05e578f172ce710bc0f1da55acdf290e83dd9"
E12A_V2_BLOB = "0c4c970c8bacfed61bfbb9587e0a0bfdec7903d9"
E12A_V2_SYNTH_BLOB = "6477076d9c5aba7d37d816147e4443b5c94ad2d4"
E12A_V2_CONTRACT_BLOB = "6b40ac35a3b43c3bafcdd08fcc49fc9fe22960c1"
CONSUMED_PARENT = "renta0426/cmi-flu-e12a-final-reproduction-20260910-001"

FROZEN_BLOBS = {
    REQUEST: "a5b799f059fce856fb4d23277f41c6e29a55b57d",
    "payloads/cmi-flu-strategy-e12a-v2-portfolio-reconciliation-001/strategy_e12a_v2.py": E12A_V2_BLOB,
    "payloads/cmi-flu-strategy-e12a-v2-portfolio-reconciliation-001/strategy_e12a_v2_synthetic.py": E12A_V2_SYNTH_BLOB,
    "payloads/cmi-flu-strategy-e12a-v2-portfolio-reconciliation-001/strategy_e12a_v2_portfolio_reconciliation.json": E12A_V2_CONTRACT_BLOB,
    "scripts/cmi_flu_strategy_e12a_v2_prepare.py": "e9909b877ec9d2d46cc287174e21a5a865a69463",
    PREPARE: "32f0d40689db1013bc1d42de2286e4ff26c91c44",
    EXECUTE: "8dbf44a1fbb574eacc0ca99801fb1359b30066d9",
    SANITIZE: "7056a580d1c9e157936b896dbdc829a87a11f2cb",
    "scripts/cmi_flu_strategy_e12a_execute.py": "46bfb2c29002775c57abcef024f239276470fd9e",
    "scripts/cmi_flu_strategy_e01_prepare_v2.py": "7a0a3484778afd0f7372f8182dd65c76b0b0e071",
    "scripts/cmi_flu_strategy_e01_prepare.py": "692cf1f521f5a377eebb7a85f8f6aae3821e5007",
    "scripts/cmi_flu_task11_prior_immunity_prepare_v2.py": "12fb52988e365b9c4e215c884b5ca724cb32ae5b",
    "scripts/kaggle_current_output_read.py": "37caa4dbbd42f2970b5d950ab6adbc08c6a6097d",
    "scripts/kaggle_exact_identity.py": "d4b9ab1199aa21977bbc14541a6b7456b5984331",
    "requirements/kaggle-2.2.4.lock": "1b85d07980c34d32b63de4da61f0b94e89ca2474",
}

PORTFOLIO = {
    "Task1.1": "b21_pls_2",
    "Task1.2": "task12_anchor_residual_et_d5_l5_sqrt_lambda0.5",
    "Task1.3": "strict_asc_anchor",
    "Task1.4": "raw_pre_vacc_conserved_anchor",
    "Task2.1": "b21_et_subtype_d3_l5",
    "Task2.2": "b21_et_subtype_d5_l10",
    "Task2.3": "b21_ridge_exact_a100",
}
SCORES = {
    "Task1.1": 0.09970718035376519,
    "Task1.2": 0.5271033295423541,
    "Task1.3": 0.38158872734833993,
    "Task2.1": 0.623449034547404,
    "Task2.2": 0.576505972340535,
    "Task2.3": 0.6957305642219944,
}


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen(root: Path) -> None:
    for rel, expected in FROZEN_BLOBS.items():
        found = git_blob((root / rel).read_bytes())
        if found != expected:
            raise SystemExit(f"E12a-v2 frozen dependency changed:{rel}:{found}")
    request = json.loads((root / REQUEST).read_text(encoding="utf-8"))
    if request.get("target") != TARGET or request.get("science_source_commit") != SCIENCE_COMMIT:
        raise SystemExit("E12a-v2 request identity mismatch")
    if request.get("strategy_e12a_v2_blob_sha") != E12A_V2_BLOB:
        raise SystemExit("E12a-v2 request science blob mismatch")
    if request.get("strategy_e12a_v2_synthetic_blob_sha") != E12A_V2_SYNTH_BLOB:
        raise SystemExit("E12a-v2 request synthetic blob mismatch")
    if request.get("strategy_e12a_v2_contract_blob_sha") != E12A_V2_CONTRACT_BLOB:
        raise SystemExit("E12a-v2 request contract blob mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 30,
        "hard_timeout_minutes": 60,
        "max_active_runs": 1,
    }:
        raise SystemExit("E12a-v2 request resource mismatch")
    if request.get("automatic_compute_retries") != 0:
        raise SystemExit("E12a-v2 automatic retry changed")
    if request.get("competition_submission_attempted") is not False:
        raise SystemExit("E12a-v2 submission boundary changed")
    if request.get("leaderboard_used_for_selection") is not False:
        raise SystemExit("E12a-v2 leaderboard boundary changed")
    parent = request.get("parent_e12a_v1") or {}
    if parent.get("target") != CONSUMED_PARENT:
        raise SystemExit("E12a-v2 consumed parent identity mismatch")
    if parent.get("kernel_version_consumed") is not True or parent.get("same_version_rerun_forbidden") is not True:
        raise SystemExit("E12a-v2 consumed parent replay guard changed")
    contract = request.get("experiment_contract") or {}
    if contract.get("final_task13_incumbent") != "strict_asc_anchor":
        raise SystemExit("E12a-v2 final Task1.3 identity mismatch")
    if contract.get("stale_task13_e01_control") != "b21_pls_1":
        raise SystemExit("E12a-v2 stale Task1.3 control mismatch")
    base_tree = ast.parse((root / "scripts/cmi_flu_strategy_e12a_execute.py").read_text(encoding="utf-8"))
    base_calls = [
        node.func.attr
        for node in ast.walk(base_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    wrapper_tree = ast.parse((root / EXECUTE).read_text(encoding="utf-8"))
    wrapper_calls = [
        node.func.attr
        for node in ast.walk(wrapper_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    if base_calls.count("kernels_push") != 1 or wrapper_calls.count("kernels_push") != 0:
        raise SystemExit("E12a-v2 write-call contract changed")
    if "competition_submit" in base_calls + wrapper_calls or "competition_submit_new" in base_calls + wrapper_calls:
        raise SystemExit("E12a-v2 submission call detected")


def build_fixture() -> dict:
    rows = []
    for task, score in SCORES.items():
        row = {
            "task": task,
            "incumbent": PORTFOLIO[task],
            "expected_study_equal_spearman": score,
            "observed_study_equal_spearman": score,
            "deviation": 0.0,
            "absolute_deviation": 0.0,
            "reproduced": True,
        }
        if task == "Task1.3":
            row.update(
                {
                    "source": "E04/E04b retained strict ASC anchor",
                    "predictor_feature": "flow_rank__Antibody-secreting_cells_(ASC)",
                    "challenge_rows": 40,
                    "challenge_anchor_complete": True,
                    "challenge_anchor_unique_values": 17,
                }
            )
        else:
            row.update(
                {
                    "source": "accepted_e01_reproduction_harness",
                    "undefined_fold_count": 0,
                    "constant_fold_count": 0,
                }
            )
        rows.append(row)
    return {
        "schema_version": 1,
        "experiment": "strategy_v2_e12a_v2_portfolio_reconciliation",
        "comparison_contract": "paired_subject_purged_v2",
        "reproduction_absolute_tolerance": 2e-6,
        "source_e12a_v1_experiment": "strategy_v2_e12a_final_system_reproduction",
        "source_e12a_v1_disposition": "reproduction_harness_passed_final_portfolio_invalid_stale_task13_identity",
        "source_e01_experiment": "strategy_v2_e01_paired_evaluation",
        "portfolio_manifest_corrected": True,
        "corrected_task": "Task1.3",
        "stale_task13_e01_control": {
            "incumbent": "b21_pls_1",
            "study_equal_spearman": 0.10630410834652516,
            "final_portfolio_member": False,
        },
        "task13_strict_anchor_gain_vs_e01_control": 0.38158872734833993 - 0.10630410834652516,
        "supervised_reproduction": sorted(rows, key=lambda row: row["task"]),
        "all_supervised_tasks_reproduced": True,
        "task14_contract": {
            "task": "Task1.4",
            "incumbent": "raw_pre_vacc_conserved_anchor",
            "source_experiment": "strategy_v2_e07_task14_formal_closure",
            "supervised_cv_available": False,
            "outcomes_accessed": False,
            "challenge_subjects_expected": 40,
            "incumbent_changed": False,
        },
        "e11_task11_candidates_carried": [],
        "frozen_portfolio": dict(PORTFOLIO),
        "portfolio_provenance": {},
        "portfolio_task_count": 7,
        "new_model_selection_performed": False,
        "competition_incumbent_changed": False,
        "public_probe_authorized": False,
        "competition_submission_authorized": False,
        "leaderboard_used_for_selection": False,
        "competition_submission_attempted": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
        "next_step": "proceed_to_E12b_challenge_prediction_freeze",
    }


def import_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("e12a_v2_generated_runtime", path)
    if spec is None or spec.loader is None:
        raise SystemExit("E12a-v2 generated runtime import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_generated(root: Path, workdir: Path) -> None:
    runtime = workdir / "runtime.py"
    subprocess.run(
        [
            sys.executable,
            str(root / PREPARE),
            "--repository-root",
            str(root),
            "--output",
            str(runtime),
        ],
        check=True,
    )
    subprocess.run([sys.executable, str(runtime), "--self-test"], check=True)
    source = runtime.read_text(encoding="utf-8")
    required = (
        f'REQUEST_ID = "20260911-cmi-flu-strategy-e12a-v2-portfolio-reconciliation-001"',
        f'TARGET_KERNEL = "{TARGET}"',
        f'SCIENCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'E12A_V2_BLOB = "{E12A_V2_BLOB}"',
        '"Task1.3": "strict_asc_anchor"',
        '"final_portfolio_member": False',
        "def terminal_success_line(",
        "CMI_FLU_E12A_V2_COMPLETE",
    )
    if any(token not in source for token in required):
        raise SystemExit("E12a-v2 generated runtime contract incomplete")
    if "len(result['tasks'])" in source:
        raise SystemExit("E12a-v2 inherited E01 terminal defect remains")
    if CONSUMED_PARENT in source:
        raise SystemExit("E12a-v2 generated runtime references consumed parent target")
    if "kaggle competitions submit" in source.casefold() or "competition_submit" in source.casefold():
        raise SystemExit("E12a-v2 generated runtime contains submission path")

    module = import_runtime(runtime)
    metrics = build_fixture()
    module.validate_result(metrics)
    terminal = module.terminal_success_line(metrics, 28)
    if "portfolio_tasks=7" not in terminal or "all_reproduced=true" not in terminal:
        raise SystemExit("E12a-v2 terminal success fixture mismatch")
    if " tasks=" in terminal:
        raise SystemExit("E12a-v2 inherited terminal tasks field remains")
    summary = module.render_summary(metrics)
    bad = copy.deepcopy(metrics)
    bad["frozen_portfolio"]["Task1.3"] = "b21_pls_1"
    try:
        module.validate_result(bad)
    except module.BridgeContractError:
        pass
    else:
        raise SystemExit("E12a-v2 stale Task1.3 final portfolio unexpectedly accepted")

    fixture = workdir / "fixture"
    fixture.mkdir()
    (fixture / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (fixture / "summary.md").write_text(summary, encoding="utf-8")
    bridge = {
        "schema_version": 1,
        "request_id": "20260911-cmi-flu-strategy-e12a-v2-portfolio-reconciliation-001",
        "competition": "cmi-flu-first-prediction-challenge",
        "target_kernel": TARGET,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e01_blob_sha": "dd27aea0cf97d41bad3cec64819c4c4269d94cbd",
        "strategy_e01_v2_blob_sha": "8cc64dc5ab9483d5957cfada18d445188566c56c",
        "config_blob_sha": "170d3211e2795c0730e481056c7bb068accf97c9",
        "package_sha256": "synthetic-fixture",
        "python_version": sys.version.split()[0],
        "md5_verified_count": 28,
        "metrics_sha256": sha256(fixture / "metrics.json"),
        "summary_sha256": sha256(fixture / "summary.md"),
        "frozen_portfolio": dict(PORTFOLIO),
        "all_supervised_tasks_reproduced": True,
        "portfolio_manifest_corrected": True,
        "corrected_task": "Task1.3",
        "next_step": "proceed_to_E12b_challenge_prediction_freeze",
        "strategy_e12a_v2_blob_sha": E12A_V2_BLOB,
        "strategy_e12a_v2_contract_blob_sha": E12A_V2_CONTRACT_BLOB,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }
    (fixture / "bridge-result.json").write_text(
        json.dumps(bridge, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    subprocess.run(
        [sys.executable, str(root / SANITIZE), "--input-dir", str(fixture)],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    workdir = args.workdir.expanduser().resolve()
    if workdir.exists():
        raise SystemExit("E12a-v2 CI workdir must be fresh")
    workdir.mkdir(parents=True)
    verify_frozen(root)
    for rel in (PREPARE, EXECUTE, SANITIZE):
        compile((root / rel).read_text(encoding="utf-8"), rel, "exec")
    validate_generated(root, workdir)
    print(
        "CMI_FLU_E12A_V2_SECRET_FREE_CI PASS "
        "exact_relay=true terminal_success_path=true stale_task13_rejected=true "
        "write=false compute=false submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
