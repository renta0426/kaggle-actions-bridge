#!/usr/bin/env python3
"""Build E02 from exact blobs with robust frozen-runtime stanza patching."""
from __future__ import annotations

import importlib.util
from pathlib import Path

BASE = Path(__file__).with_name("cmi_flu_strategy_e02_prepare.py")


def main() -> int:
    spec = importlib.util.spec_from_file_location("cmi_flu_strategy_e02_prepare_base_v4", BASE)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load E02 base builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "E02_BLOB", None) != "e6aaba7450a527de1e75ad6e7c0d9ef7e0031d7e":
        raise SystemExit("E02 base builder contract changed")
    module.E02_PARTS = (f"{module.PAYLOAD_ROOT}/strategy_e02.py",)

    def robust_patch_runtime(runtime: str, e02: str, e02_v2: str) -> str:
        anchor = '        install()\n        stage = "load_inputs"\n'
        if runtime.count(anchor) != 1:
            raise SystemExit("E02 runtime install anchor changed")
        compat = '''        install()\n        stage = "install_e02_api_compat"\n        import cmi_flu.evaluation as _e02_evaluation\n        import cmi_flu.runner as _e02_runner\n        _e02_robust_compact = _e02_runner.run_compact_task\n        _e02_robust_hai = _e02_runner.run_hai_compact_for_panels\n\n        def _e02_compact_compat(dataset, *, specs, splits=None, random_state=42, selection_policy="robust_v1"):\n            if selection_policy != "robust_v1":\n                raise BridgeContractError("e02_compact_selection_policy_not_robust_v1")\n            return _e02_robust_compact(dataset, specs=specs, splits=splits, random_state=random_state)\n\n        def _e02_hai_compat(dataset, *, specs, selection_panels, splits=None, selection_policy="robust_v1"):\n            if selection_policy != "robust_v1":\n                raise BridgeContractError("e02_hai_selection_policy_not_robust_v1")\n            return _e02_robust_hai(dataset, specs=specs, selection_panels=selection_panels, splits=splits)\n\n        _e02_evaluation.run_compact_task = _e02_compact_compat\n        _e02_evaluation.run_hai_compact_for_panels = _e02_hai_compat\n        stage = "load_inputs"\n'''
        runtime = runtime.replace(anchor, compat, 1)

        marker = "REFERENCE_PLACEHOLDERS = "
        pos = runtime.find(marker)
        if pos < 0:
            raise SystemExit("E02 source injection anchor missing")
        injected = (
            f'E02_BLOB = "{module.E02_BLOB}"\n'
            f'E02_V2_BLOB = "{module.E02_V2_BLOB}"\n'
            f'E02_SOURCE = {e02!r}\n'
            f'E02_V2_SOURCE = {e02_v2!r}\n'
            f'COMPAT_MARKER = "{module.COMPAT_MARKER}"\n'
        )
        runtime = runtime[:pos] + injected + runtime[pos:]

        runtime = module.replace_function(
            runtime,
            "def self_test() -> int:\n",
            "def locate_competition_data(",
            '''def self_test() -> int:\n    package = package_bytes()\n    checks = ((E01_SOURCE, E01_BLOB, "e01"), (E01_V2_SOURCE, E01_V2_BLOB, "e01_v2"), (E02_SOURCE, E02_BLOB, "e02"), (E02_V2_SOURCE, E02_V2_BLOB, "e02_v2"), (CONFIG_TEXT, CONFIG_BLOB, "config"))\n    for source, expected, label in checks:\n        if git_blob_sha(source.encode("utf-8")) != expected:\n            raise BridgeContractError(f"{label}_blob_mismatch")\n    compile(B21_ADAPTER_SOURCE, "cmi_flu_b21_runtime_adapter.py", "exec")\n    compile(E01_SOURCE, "cmi_flu/strategy_e01.py", "exec")\n    compile(E01_V2_SOURCE, "cmi_flu/strategy_e01_v2.py", "exec")\n    compile(E02_SOURCE, "cmi_flu/strategy_e02.py", "exec")\n    compile(E02_V2_SOURCE, "cmi_flu/strategy_e02_v2.py", "exec")\n    print(f"CMI_FLU_E02_RUNTIME_SELF_TEST PASS request_id={REQUEST_ID} package_bytes={len(package)} package_sha256={PACKAGE_SHA256} science_commit={SCIENCE_COMMIT} compat={COMPAT_MARKER}")\n    return 0''',
        )

        runtime = module.replace_function(
            runtime,
            "def load_e01_module() -> object:\n",
            "def json_safe(value):\n",
            '''def load_e02_module() -> object:\n    base = types.ModuleType("cmi_flu.strategy_e01")\n    base.__file__ = "<cmi_flu.strategy_e01>"\n    base.__package__ = "cmi_flu"\n    sys.modules["cmi_flu.strategy_e01"] = base\n    exec(compile(E01_SOURCE, "cmi_flu/strategy_e01.py", "exec"), base.__dict__, base.__dict__)\n    e01v2 = types.ModuleType("cmi_flu.strategy_e01_v2")\n    e01v2.__file__ = "<cmi_flu.strategy_e01_v2>"\n    e01v2.__package__ = "cmi_flu"\n    sys.modules["cmi_flu.strategy_e01_v2"] = e01v2\n    exec(compile(E01_V2_SOURCE, "cmi_flu/strategy_e01_v2.py", "exec"), e01v2.__dict__, e01v2.__dict__)\n    e02 = types.ModuleType("cmi_flu.strategy_e02")\n    e02.__file__ = "<cmi_flu.strategy_e02>"\n    e02.__package__ = "cmi_flu"\n    sys.modules["cmi_flu.strategy_e02"] = e02\n    exec(compile(E02_SOURCE, "cmi_flu/strategy_e02.py", "exec"), e02.__dict__, e02.__dict__)\n    e02v2 = types.ModuleType("cmi_flu.strategy_e02_v2")\n    e02v2.__file__ = "<cmi_flu.strategy_e02_v2>"\n    e02v2.__package__ = "cmi_flu"\n    sys.modules["cmi_flu.strategy_e02_v2"] = e02v2\n    exec(compile(E02_V2_SOURCE, "cmi_flu/strategy_e02_v2.py", "exec"), e02v2.__dict__, e02v2.__dict__)\n    run = getattr(e02v2, "run_strategy_e02", None)\n    if not callable(run):\n        raise BridgeContractError("e02_entry_missing")\n    return run''',
        )

        runtime = module.replace_function(
            runtime,
            "def validate_result(result: dict) -> None:\n",
            "def render_summary(result: dict) -> str:\n",
            '''def validate_result(result: dict) -> None:\n    if result.get("schema_version") != 1 or result.get("experiment") != "strategy_v2_e02_task11_structured_logfc":\n        raise BridgeContractError("e02_experiment_identity_mismatch")\n    if result.get("comparison_contract") != "paired_subject_purged_v2" or result.get("random_seed") != 20260907 or result.get("task") != "Task1.1":\n        raise BridgeContractError("e02_evaluation_contract_mismatch")\n    cohort = result.get("cohort") or {}\n    if cohort.get("train_rows") != 127 or cohort.get("train_studies") != 4 or cohort.get("challenge_rows") != 40:\n        raise BridgeContractError("e02_cohort_contract_mismatch")\n    fixed = result.get("fixed_conditions") or {}\n    if fixed.get("b21_model") != "pls_2" or fixed.get("anchor_residual_model") != "pls_1" or fixed.get("anchor_residual_lambda") != 0.25 or fixed.get("structured_ridge_alpha") != 10.0:\n        raise BridgeContractError("e02_fixed_condition_mismatch")\n    conditions = result.get("conditions") or {}\n    for key in ("anchor", "b21", "anchor_residual", "structured_ridge"):\n        metrics = (conditions.get(key) or {}).get("metrics") or {}\n        if metrics.get("rows") != 127:\n            raise BridgeContractError(f"e02_condition_rows_mismatch:{key}")\n    controls = result.get("negative_controls") or {}\n    if "availability_only" not in controls or "structured_subject_block_label_shuffle" not in controls:\n        raise BridgeContractError("e02_negative_controls_missing")\n    repeat = result.get("repeat_baseline_extension") or {}\n    audit = repeat.get("audit") or {}\n    if audit.get("status") not in {"ready", "data_limited"}:\n        raise BridgeContractError("e02_repeat_gate_status_invalid")\n    if result.get("contains_participant_identifiers") is not False or result.get("contains_row_level_predictions") is not False:\n        raise BridgeContractError("e02_row_level_output_flag")\n    if result.get("leaderboard_used_for_selection") is not False or result.get("competition_submission_attempted") is not False:\n        raise BridgeContractError("e02_selection_or_submission_flag")\n    forbidden = {"participant_id", "subject_group", "row_index", "oof_predictions", "challenge_predictions", "predictions"}\n    def walk(value):\n        if isinstance(value, dict):\n            for key, child in value.items():\n                if key in forbidden:\n                    raise BridgeContractError(f"e02_forbidden_output_key:{key}")\n                walk(child)\n        elif isinstance(value, list):\n            for child in value:\n                walk(child)\n    walk(result)''',
        )

        runtime = module.replace_function(
            runtime,
            "def render_summary(result: dict) -> str:\n",
            "def main() -> int:\n",
            '''def render_summary(result: dict) -> str:\n    lines = ["# CMI-Flu strategy E02 Task1.1 structured logFC", "", "Aggregate-only fixed-condition evaluation; no Competition submission was attempted.", "", f"- science commit: `{SCIENCE_COMMIT}`", "- task: `Task1.1`", "- primary reference: `same_readout_oriented_anchor`", ""]\n    conditions = result.get("conditions") or {}\n    for key in ("anchor", "b21", "anchor_residual", "structured_ridge", "structured_ridge_plus_repeat"):\n        item = conditions.get(key)\n        if not item:\n            continue\n        metrics = item.get("metrics") or {}\n        lines.append(f"- {key}: strict={metrics.get('study_equal_weight_spearman_mean_strict')} pooled_rank={(metrics.get('pooled_within_study_rank_spearman') or {}).get('value')} undefined={metrics.get('undefined_fold_count')}")\n    comp = (result.get("comparisons") or {}).get("structured_ridge_vs_anchor") or {}\n    lines.extend(["", f"- structured vs anchor strict delta: {comp.get('study_mean_delta_strict')}", f"- large-study review: {comp.get('large_studies_requiring_review')}", f"- repeat branch: {(result.get('repeat_baseline_extension') or {}).get('audit', {}).get('status')}", f"- structured candidate: {(result.get('decision') or {}).get('structured_ridge_candidate_over_anchor')}", ""])\n    return "\\n".join(lines)''',
        )

        replacements = (
            ('stage = "load_e01"', 'stage = "load_e02"'),
            ('run_e01 = load_e01_module()', 'run_e02 = load_e02_module()'),
            ('stage = "run_e01"', 'stage = "run_e02"'),
            ('result = json_safe(dict(run_e01(config, inputs)))', 'result = json_safe(dict(run_e02(config, inputs)))'),
            ('stage = "validate_e01"', 'stage = "validate_e02"'),
        )
        for old, new in replacements:
            if runtime.count(old) != 1:
                raise SystemExit(f"E02 runtime execution token count changed: {old} count={runtime.count(old)}")
            runtime = runtime.replace(old, new, 1)

        runtime = runtime.replace(
            "CMI-Flu strategy E01 paired evaluation. Aggregate outputs only; no submission.",
            "CMI-Flu strategy E02 Task1.1 structured logFC. Aggregate outputs only; no submission.",
            1,
        )
        runtime = runtime.replace("CMI_FLU_E01_FAILED", "CMI_FLU_E02_FAILED")
        if "kaggle competitions submit" in runtime or "competition_submit" in runtime:
            raise SystemExit("generated E02 runtime contains submission path")
        for token in (
            "load_e02_module",
            "run_strategy_e02",
            "_e02_evaluation.run_compact_task = _e02_compact_compat",
            f'COMPAT_MARKER = "{module.COMPAT_MARKER}"',
        ):
            if token not in runtime:
                raise SystemExit(f"generated E02 runtime missing token: {token}")
        compile(runtime, "generated_e02.py", "exec")
        return runtime

    module.patch_runtime = robust_patch_runtime
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
