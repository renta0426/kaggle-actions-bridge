#!/usr/bin/env python3
"""One approved read-only recovery of E02 version 1. No push or execution path.

Downloaded Notebook code is hashed but NEVER imported or executed. The validator
is reconstructed from the approved public bridge commit in a secret-free step.
Only a typed projection of aggregate metrics reaches the public log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import tempfile

from kaggle_exact_identity import exact_metadata, validate_metadata, verify_current, safe_exception

REQUEST = "20260907-cmi-flu-e02-current-recovery-001"
ORIGINAL_REQUEST = "20260907-cmi-flu-strategy-e02-task11-structured-logfc-001"
TARGET = "renta0426/cmi-flu-e02-task11-structured-logfc-20260907-001"
VERSION = 1
RUNTIME_SHA = "ad36802efe2af9d4678b5d216b4afbb904a74746ec971818e75fdb2692709f37"
SCIENCE = "0d399f89f35145e99fafd10d3061d9295cd749bb"
STUDIES = {"SDY180", "SDY515", "SDY519", "SDY56"}
CONDITIONS = ("anchor", "b21", "anchor_residual", "structured_ridge", "structured_ridge_plus_repeat")
LIMITS = {"metrics.json": 8388608, "summary.md": 1048576, "bridge-result.json": 1048576}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cli_read(verb: str, folder: Path) -> None:
    if verb not in {"pull", "output"}:
        raise ValueError("read_verb_not_allowed")
    cli = shutil.which("kaggle")
    if not cli:
        raise RuntimeError("kaggle_cli_missing")
    result = subprocess.run([cli, "kernels", verb, TARGET, "-p", str(folder)], capture_output=True, timeout=180, check=False)
    if result.returncode:
        info = {"stage": verb, "return_code": result.returncode,
                "stdout_bytes": len(result.stdout), "stderr_bytes": len(result.stderr),
                "diagnostic_sha256": hashlib.sha256(result.stdout + result.stderr).hexdigest()}
        print("E02_RECOVERY_CLI " + json.dumps(info, sort_keys=True))
        raise RuntimeError("read_cli_failed")


def numbered(source: dict, keys: tuple[str, ...]) -> dict:
    result = {}
    for key in keys:
        value = source.get(key)
        if value is not None and (type(value) not in (int, float, bool) or not math.isfinite(float(value))):
            raise ValueError("non_numeric_aggregate")
        result[key] = value
    return result


def scalar_metric(source: dict, key: str):
    value = source.get(key)
    if isinstance(value, dict):
        return numbered(value, ("n", "value"))
    return numbered({key: value}, (key,))[key]


def study_rows(rows: list, keys: tuple[str, ...]) -> list:
    if not isinstance(rows, list) or len(rows) > 4:
        raise ValueError("study_aggregate_size")
    result = []
    for row in rows:
        if row.get("study") not in STUDIES:
            raise ValueError("unexpected_study_in_aggregate")
        result.append({"study": row["study"], **numbered(row, keys)})
    return result


def project(metrics: dict) -> dict:
    result = {"conditions": {}, "comparisons": {}, "sensitivity_28": {}, "negative_controls": {}}
    for name in CONDITIONS:
        item = metrics["conditions"].get(name)
        if item is None:
            continue
        m = item["metrics"]
        out = numbered(m, ("rows", "studies", "undefined_fold_count", "constant_fold_count", "study_equal_weight_spearman_mean_strict"))
        for key in ("pooled_within_study_rank_spearman", "rank_rmse", "raw_scale_rmse"):
            out[key] = scalar_metric(m, key)
        out["folds"] = study_rows(m["folds"], ("n", "spearman", "rank_rmse", "constant_prediction", "tie_fraction"))
        result["conditions"][name] = out
    compare_names = [f"{c}_vs_{r}" for c in CONDITIONS for r in CONDITIONS if c != r]
    for name in compare_names:
        item = metrics.get("comparisons", {}).get(name)
        if item is not None:
            out = numbered(item, ("rows", "all_folds_valid", "study_mean_delta_strict", "passes_candidate_delta_heuristic"))
            out["folds"] = study_rows(item["folds"], ("n", "candidate_spearman", "reference_spearman", "delta"))
            result["comparisons"][name] = out
        item = metrics.get("sensitivity_28", {}).get(name)
        if item is not None:
            result["sensitivity_28"][name] = {
                **numbered(item, ("seed", "subset_n", "repetitions", "without_replacement")),
                "studies": study_rows(item["studies"], ("n", "valid_repetitions", "invalid_repetitions", "candidate_better_fraction", "spearman_delta_mean", "spearman_delta_q10", "spearman_delta_q90")),
            }
    for key in ("availability_only", "structured_subject_block_label_shuffle"):
        m = metrics["negative_controls"][key]["metrics"]
        result["negative_controls"][key] = numbered(m, ("study_equal_weight_spearman_mean_strict", "undefined_fold_count", "constant_fold_count"))
    repeat = metrics["repeat_baseline_extension"]
    audit = repeat["audit"]
    if audit["status"] not in {"ready", "data_limited"}:
        raise ValueError("repeat_status")
    result["repeat"] = {"status": audit["status"], **numbered(repeat, ("condition_executed", "candidate")), **numbered(audit, ("challenge_paired_subjects", "full_outer_contract_gate_passed"))}
    result["decision"] = numbered(metrics["decision"], ("structured_ridge_candidate_over_anchor", "repeat_extension_candidate_over_anchor", "candidate_delta_threshold", "large_study_decline_review_threshold"))
    return result


def recover(runtime: Path, output: Path) -> None:
    if output.exists() or digest(runtime) != RUNTIME_SHA:
        raise ValueError("recovery_local_contract")
    trusted = runpy.run_path(str(runtime), run_name="approved_e02_validator")
    if trusted.get("SCIENCE_COMMIT") != SCIENCE or trusted.get("REQUEST_ID") != ORIGINAL_REQUEST:
        raise ValueError("trusted_runtime_identity")
    from kaggle.api.kaggle_api_extended import KaggleApi
    from cmi_flu_strategy_e02_execute import live_rules
    api = KaggleApi()
    api.authenticate()
    live_rules(api)
    state = verify_current(api, TARGET, VERSION, allow_failed=True)
    receipt = {"request_id": REQUEST, "original_request_id": ORIGINAL_REQUEST,
               "target_sha256": hashlib.sha256(TARGET.encode()).hexdigest(),
               "version": VERSION, "remote_status": state, "new_write_attempted": False,
               "new_compute_requested": False, "source_sha256": RUNTIME_SHA, "science_commit": SCIENCE}
    with tempfile.TemporaryDirectory(prefix="cmi-e02-recovery-") as tmp:
        root = Path(tmp)
        code_dir = root / "code"; code_dir.mkdir()
        cli_read("pull", code_dir)
        code_files = [p for p in code_dir.rglob("*") if p.is_file()]
        if len(code_files) != 1 or any(p.is_symlink() for p in code_dir.rglob("*")):
            raise RuntimeError("current_code_file_contract")
        if code_files[0].stat().st_size > 2097152 or digest(code_files[0]) != RUNTIME_SHA:
            raise RuntimeError("current_code_hash_mismatch")
        validate_metadata(exact_metadata(api, TARGET), TARGET, VERSION)
        download = root / "outputs"; download.mkdir()
        cli_read("output", download)
        validate_metadata(exact_metadata(api, TARGET), TARGET, VERSION)
        paths = list(download.rglob("*"))
        if any(p.is_symlink() for p in paths):
            raise RuntimeError("output_symlink")
        files = [p for p in paths if p.is_file()]
        if sum(p.stat().st_size for p in files) > 33554432:
            raise RuntimeError("recovery_download_size")
        logs = [p for p in files if p.parent == download and p.suffix == ".log"]
        if len(logs) > 1:
            raise RuntimeError("transport_log_count")
        selected = {}
        for p in files:
            if p in logs:
                continue
            if p.name not in LIMITS or p.name in selected or not 0 < p.stat().st_size <= LIMITS[p.name]:
                raise RuntimeError("unexpected_or_oversize_output")
            selected[p.name] = p
        signatures = []
        if logs:
            text = logs[0].read_text(encoding="utf-8", errors="replace")
            for stage, kind, code in re.findall(r"CMI_FLU_E02_FAILED stage=([a-z_]{1,64}) exception_type=([A-Za-z]{1,64}) error_code=([0-9a-f]{20})", text):
                if stage in {"initialize", "materialize_package", "prepare_tree", "install_b21_adapter", "install_e02_api_compat", "load_inputs", "load_e01", "run_e01", "validate_e01", "write_outputs", "locate_competition_data"} and kind in {"KeyError", "TypeError", "ValueError", "RuntimeError", "DataContractError", "BridgeContractError", "ImportError", "AttributeError"}:
                    signatures.append({"stage": stage, "exception_type": kind, "error_code": code})
        receipt["failure_signatures"] = signatures[:2]
        receipt["original_bridge_result_present"] = "bridge-result.json" in selected
        if "metrics.json" not in selected or "summary.md" not in selected:
            print("E02_RECOVERY_NO_METRICS " + json.dumps(receipt, sort_keys=True))
            raise RuntimeError("no_complete_scientific_metrics_to_salvage")
        metrics = json.loads(selected["metrics.json"].read_text())
        trusted["validate_result"](metrics)
        safe = project(metrics)
        if "bridge-result.json" in selected:
            original = json.loads(selected["bridge-result.json"].read_text())
            if original.get("request_id") != ORIGINAL_REQUEST or original.get("science_commit") != SCIENCE or original.get("metrics_sha256") != digest(selected["metrics.json"]):
                raise RuntimeError("original_provenance_mismatch")
        elif not any(s["stage"] == "write_outputs" and s["exception_type"] == "KeyError" and s["error_code"] == hashlib.sha256(b"KeyError:'frozen_incumbent'").hexdigest()[:20] for s in signatures):
            raise RuntimeError("missing_bridge_without_expected_serialization_failure")
        receipt["metrics_sha256"] = digest(selected["metrics.json"])
        receipt["summary_sha256"] = digest(selected["summary.md"])
        receipt["scientific_metrics_validated"] = True
        receipt["original_notebook_success"] = state == "COMPLETE"
        # This is a recovery receipt, never a fabricated original bridge-result.
        output.mkdir(parents=True)
        (output / "recovery.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        (output / "aggregate.json").write_text(json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print("E02_RECOVERY_VERIFIED " + json.dumps(receipt, sort_keys=True))
        print("E02_AGGREGATES_BEGIN\n" + json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\nE02_AGGREGATES_END")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        recover(a.runtime, a.output)
    except Exception as exc:
        print("E02_RECOVERY_FAILED " + json.dumps({"request_id": REQUEST, "new_write_attempted": False, **safe_exception(exc)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
