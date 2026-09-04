#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import io
import pathlib
import zipfile
from typing import Any


def parse_functions(source: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(source)
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def function_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = node.args
    return {
        item.arg
        for item in [*args.posonlyargs, *args.args, *args.kwonlyargs]
    }


def class_methods(source: str, class_name: str) -> set[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise SystemExit(f"frozen package class missing: {class_name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=pathlib.Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.expanduser().resolve()
    text = runtime.read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": "study_similarity_frozen_api_check"}
    exec(compile(text, str(runtime), "exec"), namespace, namespace)

    package_bytes = namespace.get("package_bytes")
    if not callable(package_bytes):
        raise SystemExit("generated runtime lacks package_bytes()")
    bundle = package_bytes()
    if not isinstance(bundle, bytes) or not bundle:
        raise SystemExit("generated runtime package bundle invalid")

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        required_files = {
            "cmi_flu/models.py",
            "cmi_flu/cv.py",
            "cmi_flu/datasets.py",
            "cmi_flu/metrics.py",
        }
        if not required_files.issubset(set(archive.namelist())):
            raise SystemExit("frozen package missing required API modules")
        sources = {
            name: archive.read(name).decode("utf-8")
            for name in required_files
        }

    model_functions = parse_functions(sources["cmi_flu/models.py"])
    fit = model_functions.get("fit_final_model")
    if fit is None:
        raise SystemExit("frozen models.py lacks fit_final_model")
    expected_fit = {
        "train_frame",
        "prediction_frame",
        "target_column",
        "spec",
        "excluded_columns",
    }
    if not expected_fit.issubset(function_parameters(fit)):
        raise SystemExit(
            f"frozen fit_final_model parameters mismatch: {sorted(function_parameters(fit))}"
        )

    cv_functions = parse_functions(sources["cmi_flu/cv.py"])
    split = cv_functions.get("purged_leave_one_study_out")
    if split is None or not {"studies", "subjects"}.issubset(function_parameters(split)):
        raise SystemExit("frozen purged_leave_one_study_out signature mismatch")

    dataset_functions = parse_functions(sources["cmi_flu/datasets.py"])
    for name in ("build_task_11_dataset", "build_task_12_dataset"):
        if name not in dataset_functions:
            raise SystemExit(f"frozen dataset API missing: {name}")
    methods = class_methods(sources["cmi_flu/datasets.py"], "TaskDataset")
    if "feature_columns" not in methods or "validate" not in methods:
        raise SystemExit("frozen TaskDataset method contract mismatch")
    if "excluded_columns" not in sources["cmi_flu/datasets.py"]:
        raise SystemExit("frozen TaskDataset excluded_columns field missing")

    metric_functions = parse_functions(sources["cmi_flu/metrics.py"])
    for name in ("percentile_rank", "root_mean_squared_error", "safe_spearman"):
        if name not in metric_functions:
            raise SystemExit(f"frozen metrics API missing: {name}")

    science = namespace.get("STUDY_SIMILARITY_SOURCE")
    if not isinstance(science, str) or not science:
        raise SystemExit("generated runtime lacks study-similarity source")
    science_tree = ast.parse(science)
    imported_names = set()
    for node in ast.walk(science_tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_names.update(alias.name for alias in node.names)
    required_imports = {
        "fit_final_model",
        "purged_leave_one_study_out",
        "build_task_11_dataset",
        "build_task_12_dataset",
        "percentile_rank",
        "root_mean_squared_error",
        "safe_spearman",
    }
    if not required_imports.issubset(imported_names):
        raise SystemExit(
            f"science source frozen-API imports mismatch: missing={sorted(required_imports - imported_names)}"
        )

    print(
        "CMI_FLU_STUDY_SIMILARITY_FROZEN_API PASS "
        f"fit_parameters={','.join(sorted(expected_fit))} taskdataset_feature_columns=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
