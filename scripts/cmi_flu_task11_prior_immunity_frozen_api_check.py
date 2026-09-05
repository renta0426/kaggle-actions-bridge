#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import io
import pathlib
import zipfile
from typing import Any

EXPECTED_BASE_CONDITIONS = ("b1", "b21", "anchor_residual")
EXPECTED_FUSION_CONDITIONS = (
    "b1_plus_prior_w0.25",
    "b1_plus_prior_w0.5",
    "b21_plus_prior_w0.25",
    "b21_plus_prior_w0.5",
    "anchor_residual_plus_prior_w0.25",
    "anchor_residual_plus_prior_w0.5",
)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=pathlib.Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.expanduser().resolve()
    text = runtime.read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": "task11_prior_immunity_frozen_api_check"}
    exec(compile(text, str(runtime), "exec"), namespace, namespace)

    if tuple(namespace.get("BASE_CONDITIONS", ())) != EXPECTED_BASE_CONDITIONS:
        raise SystemExit("generated runtime base-condition globals mismatch")
    if tuple(namespace.get("FUSION_CONDITIONS", ())) != EXPECTED_FUSION_CONDITIONS:
        raise SystemExit("generated runtime fusion-condition globals mismatch")

    package_bytes = namespace.get("package_bytes")
    if not callable(package_bytes):
        raise SystemExit("generated runtime lacks package_bytes()")
    bundle = package_bytes()
    if not isinstance(bundle, bytes) or not bundle:
        raise SystemExit("generated runtime package bundle invalid")

    required_files = {
        "cmi_flu/models.py",
        "cmi_flu/evaluation.py",
        "cmi_flu/features/serology.py",
        "cmi_flu/metrics.py",
        "cmi_flu/runner.py",
    }
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        if not required_files.issubset(set(archive.namelist())):
            missing = sorted(required_files - set(archive.namelist()))
            raise SystemExit(f"frozen package missing required API modules: {missing}")
        sources = {name: archive.read(name).decode("utf-8") for name in required_files}

    model_functions = parse_functions(sources["cmi_flu/models.py"])
    fit = model_functions.get("fit_final_model")
    expected_fit = {
        "train_frame",
        "prediction_frame",
        "target_column",
        "spec",
        "excluded_columns",
    }
    if fit is None or not expected_fit.issubset(function_parameters(fit)):
        found = [] if fit is None else sorted(function_parameters(fit))
        raise SystemExit(f"frozen fit_final_model parameters mismatch: {found}")

    evaluation_functions = parse_functions(sources["cmi_flu/evaluation.py"])
    split = evaluation_functions.get("default_splits_for_task")
    if split is None or "dataset" not in function_parameters(split):
        raise SystemExit("frozen default_splits_for_task signature mismatch")

    serology_functions = parse_functions(sources["cmi_flu/features/serology.py"])
    baseline = serology_functions.get("build_hai_baseline_long")
    if baseline is None or "serology" not in function_parameters(baseline):
        raise SystemExit("frozen build_hai_baseline_long signature mismatch")
    panel = serology_functions.get("build_hai_panel_summaries")
    if panel is None:
        raise SystemExit("frozen build_hai_panel_summaries missing")
    panel_params = function_parameters(panel)
    if not {"baseline_long", "group_columns"}.issubset(panel_params):
        raise SystemExit(
            f"frozen build_hai_panel_summaries parameters mismatch: {sorted(panel_params)}"
        )

    metric_functions = parse_functions(sources["cmi_flu/metrics.py"])
    for name in ("percentile_rank", "safe_spearman"):
        if name not in metric_functions:
            raise SystemExit(f"frozen metrics API missing: {name}")

    runner_functions = parse_functions(sources["cmi_flu/runner.py"])
    if "build_b02_datasets" not in runner_functions:
        raise SystemExit("frozen runner API missing build_b02_datasets")

    science = namespace.get("TASK11_PRIOR_IMMUNITY_SOURCE")
    if not isinstance(science, str) or not science:
        raise SystemExit("generated runtime lacks Task1.1 prior-immunity source")
    science_tree = ast.parse(science)
    imported_names: set[str] = set()
    for node in ast.walk(science_tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_names.update(alias.name for alias in node.names)
    required_imports = {
        "default_splits_for_task",
        "build_hai_baseline_long",
        "build_hai_panel_summaries",
        "percentile_rank",
        "safe_spearman",
        "ModelSpec",
        "fit_final_model",
        "build_b02_datasets",
    }
    if not required_imports.issubset(imported_names):
        raise SystemExit(
            "science source frozen-API imports mismatch: "
            f"missing={sorted(required_imports - imported_names)}"
        )

    print(
        "CMI_FLU_TASK11_PRIOR_IMMUNITY_FROZEN_API PASS "
        f"fit_parameters={','.join(sorted(expected_fit))} "
        "hai_group_columns=true runtime_condition_globals=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
