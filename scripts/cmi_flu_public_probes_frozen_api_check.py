#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import io
import pathlib
import zipfile
from typing import Any

TEMPLATE_HASH_CALL = "sha256_bytes(backbone_bytes)"


def functions(source: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(source)
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = node.args
    return {item.arg for item in [*args.posonlyargs, *args.args, *args.kwonlyargs]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=pathlib.Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.expanduser().resolve()
    text = runtime.read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": "public_probes_frozen_api_check"}
    exec(compile(text, str(runtime), "exec"), namespace, namespace)
    package_bytes = namespace.get("package_bytes")
    if not callable(package_bytes):
        raise SystemExit("generated runtime lacks package_bytes()")
    if text.count(TEMPLATE_HASH_CALL) != 1:
        raise SystemExit("generated template backbone-hash repair anchor mismatch")
    if not callable(namespace.get("sha256_file")):
        raise SystemExit("generated runtime lacks sha256_file() required for probe hashes")
    bundle = package_bytes()
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        required = {
            "cmi_flu/aliases.py",
            "cmi_flu/contracts.py",
            "cmi_flu/datasets.py",
            "cmi_flu/metrics.py",
            "cmi_flu/models.py",
            "cmi_flu/runner.py",
        }
        if not required.issubset(set(archive.namelist())):
            raise SystemExit("frozen B2.1 package missing controlled-probe API modules")
        sources = {name: archive.read(name).decode("utf-8") for name in required}

    model = functions(sources["cmi_flu/models.py"]).get("fit_final_model")
    if model is None or not {
        "train_frame", "prediction_frame", "target_column", "spec", "excluded_columns"
    }.issubset(params(model)):
        raise SystemExit("frozen fit_final_model signature mismatch")
    dataset_functions = functions(sources["cmi_flu/datasets.py"])
    if "build_task_13_dataset" not in dataset_functions:
        raise SystemExit("frozen build_task_13_dataset missing")
    task13 = dataset_functions["build_task_13_dataset"]
    if not {"mode", "include_sdy272_asc_proxy"}.issubset(params(task13)):
        raise SystemExit("frozen build_task_13_dataset signature mismatch")
    runner_functions = functions(sources["cmi_flu/runner.py"])
    if "build_b02_datasets" not in runner_functions or "load_inputs" not in runner_functions:
        raise SystemExit("frozen runner dataset/input APIs missing")
    if "canonicalize_flow_population" not in functions(sources["cmi_flu/aliases.py"]):
        raise SystemExit("frozen canonicalize_flow_population missing")
    contracts = sources["cmi_flu/contracts.py"]
    if "TASK_COLUMNS" not in contracts or "def validate_submission" not in contracts:
        raise SystemExit("frozen submission contract APIs missing")
    metric_functions = functions(sources["cmi_flu/metrics.py"])
    for name in ("percentile_rank", "safe_spearman"):
        if name not in metric_functions:
            raise SystemExit(f"frozen metric API missing: {name}")

    science = namespace.get("PUBLIC_PROBES_SOURCE")
    if not isinstance(science, str) or not science:
        raise SystemExit("generated runtime lacks Public-probe source")
    tree = ast.parse(science)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.update(alias.name for alias in node.names)
    expected_imports = {
        "canonicalize_flow_population", "DataContractError", "TASK_COLUMNS",
        "validate_submission", "TaskDataset", "build_task_13_dataset",
        "percentile_rank", "safe_spearman", "ModelSpec", "fit_final_model",
        "InputBundle", "build_b02_datasets",
    }
    if not expected_imports.issubset(imports):
        raise SystemExit(f"controlled-probe source import contract mismatch: {sorted(expected_imports - imports)}")
    print(
        "CMI_FLU_PUBLIC_PROBES_FROZEN_API PASS "
        "fit_final_model=true task13_builder=true b02_builder=true submission_contract=true "
        "backbone_hash_repair_anchor=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
