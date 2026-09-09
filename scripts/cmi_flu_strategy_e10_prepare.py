#!/usr/bin/env python3
"""Build exact E10 pooled domain-correction runtime from proven E05-v6 ancestry."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cmi_flu_strategy_e09b_prepare as e09b_parent

REQUEST_ID = "20260909-cmi-flu-strategy-e10-pooled-domain-correction-001"
TARGET_KERNEL = "renta0426/cmi-flu-e10-pooled-domain-correction-20260909-001"
SCIENCE_COMMIT = "9e75a6d547d38e369620cdf12bbc6af5e74d1ba8"
E10_BLOB = "f7b6b05b8b8f0061a5a378256b729f230960fe1c"
E10_SYNTH_BLOB = "5859fd21dfa2b9788090ca8a5a2076f4059cd1eb"
ANCHOR_BLOB = "9a7814ecdf70e0b38b2740ade21e9db588869379"
RANK_TRANSFER_BLOB = "d5e07cdd09d2eabdc935eb1733ec238e26ab4c17"
STUDY_SIMILARITY_BLOB = "27351df3d9187899c4bce2ff1a24b06efc160185"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e10-pooled-domain-correction-001"
E10_PATH = f"{PAYLOAD_ROOT}/strategy_e10.py"
E10_SYNTH_PATH = f"{PAYLOAD_ROOT}/strategy_e10_synthetic.py"
ANCHOR_PATH = "payloads/cmi-flu-anchor-residual-001/anchor_residual.py"
RANK_TRANSFER_PATH = "payloads/cmi-flu-rank-transfer-001/rank_transfer.py"
STUDY_SIMILARITY_PATH = "payloads/cmi-flu-study-similarity-001/study_similarity.py"
REQUEST_PATH = "requests/cmi-flu-strategy-e10-pooled-domain-correction-001.json"
OLD_REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-007"
OLD_TARGET_KERNEL = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-007"
OLD_SCIENCE_COMMIT = "92fac34e52485f6f73ea1b7aedb983d8e8e0ee27"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_text(root: Path, relative: str, expected: str, label: str) -> str:
    data = (root / relative).read_bytes()
    found = git_blob_sha(data)
    if found != expected:
        raise SystemExit(f"E10 exact relay mismatch:{label}:{found}")
    text = data.decode("utf-8")
    if relative.endswith(".py"):
        compile(text, f"cmi_flu/{Path(relative).name}", "exec")
    return text


def verify_parent_contract() -> None:
    expected = {
        "REQUEST_ID": "20260909-cmi-flu-strategy-e09b-hipc9-rna-residual-001",
        "TARGET_KERNEL": "renta0426/cmi-flu-e09b-hipc9-rna-residual-20260909-001",
        "SCIENCE_COMMIT": "8f36ba0fa0d5fae6809ff09e0bf15bb132dd1ff5",
        "E09B_BLOB": "6a58b1f0a09ac6b2b07473627af37847d9ac41c4",
    }
    for key, value in expected.items():
        if getattr(e09b_parent, key, None) != value:
            raise SystemExit(f"E10 E09b-builder ancestry changed:{key}")
    prior = e09b_parent.prior
    expected_prior = {
        "REQUEST_ID": OLD_REQUEST_ID,
        "TARGET_KERNEL": OLD_TARGET_KERNEL,
        "SCIENCE_COMMIT": OLD_SCIENCE_COMMIT,
        "CONFIG_BLOB": CONFIG_BLOB,
        "E05_V4_BLOB": "ab93726fe0810299892091b7e0e8f63dcacec4f3",
    }
    for key, value in expected_prior.items():
        if getattr(prior, key, None) != value:
            raise SystemExit(f"E10 E05-v6 ancestry changed:{key}")


def load_exact_sources(root: Path) -> tuple[str, str, str, str, str]:
    e10 = require_text(root, E10_PATH, E10_BLOB, "e10")
    synth = require_text(root, E10_SYNTH_PATH, E10_SYNTH_BLOB, "e10_synthetic")
    anchor = require_text(root, ANCHOR_PATH, ANCHOR_BLOB, "anchor_residual")
    rank_transfer = require_text(root, RANK_TRANSFER_PATH, RANK_TRANSFER_BLOB, "rank_transfer")
    study_similarity = require_text(root, STUDY_SIMILARITY_PATH, STUDY_SIMILARITY_BLOB, "study_similarity")
    required = (
        'EXPERIMENT = "strategy_v2_e10_pooled_domain_conditioned_residual"',
        'RIDGE_ALPHA = 10.0',
        'CORRECTION_SHRINKAGE = 0.25',
        'MAX_ABSOLUTE_RANK_CORRECTION = 0.05',
        'SOURCE_WEIGHT_MIN = 0.5',
        'SOURCE_WEIGHT_MAX = 2.0',
        'DEVIATION_SCALE = 0.25',
        'PROMOTION_MEAN_DELTA = 0.02',
        'PROMOTION_MIN_STUDY_DELTA = -0.10',
        '"held_target_outcomes_used_for_domain_weights": False',
        '"isolated_source_study_models_allowed": False',
        '"competition_submission_attempted": False',
    )
    if any(token not in e10 for token in required):
        raise SystemExit("E10 frozen science token missing")
    for source in (e10, synth, anchor, rank_transfer, study_similarity):
        if "kaggle competitions submit" in source or "competition_submit" in source:
            raise SystemExit("E10 relayed source contains submission path")
    return e10, synth, anchor, rank_transfer, study_similarity


def validate_request(root: Path) -> None:
    request = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_repository": "renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": "agent_relay_exact_blobs_plus_proven_e05_v6_runtime",
        "strategy_e10_blob_sha": E10_BLOB,
        "strategy_e10_synthetic_blob_sha": E10_SYNTH_BLOB,
        "expected_kernel_version": 1,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"E10 request mismatch:{key}")
    if request.get("dependency_blobs") != {
        "anchor_residual.py": ANCHOR_BLOB,
        "rank_transfer.py": RANK_TRANSFER_BLOB,
        "study_similarity.py": STUDY_SIMILARITY_BLOB,
        "baseline_b021_robust.yaml": CONFIG_BLOB,
    }:
        raise SystemExit("E10 dependency blob contract mismatch")
    if request.get("resource") != {
        "accelerator": "cpu", "expected_runtime_minutes": 45,
        "hard_timeout_minutes": 120, "max_active_runs": 1,
    }:
        raise SystemExit("E10 resource contract mismatch")
    if request.get("api_budget") != {
        "max_calls": 100, "poll_interval_seconds": 120, "max_polls": 65, "max_pages": 2,
    }:
        raise SystemExit("E10 API budget mismatch")
    if request.get("allowed_output_paths") != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E10 output allowlist mismatch")
    contract = request.get("experiment_contract") or {}
    locked = {
        "experiment": "strategy_v2_e10_pooled_domain_conditioned_residual",
        "tasks": ["Task1.1", "Task1.2"],
        "task11_base_kind": "b21",
        "task11_base_model": "pls_2",
        "task12_base_kind": "anchor_residual",
        "task12_anchor_column": "flow_rank__Classical_monocytes",
        "task12_residual_model": "et_d5_l5_sqrt",
        "task12_residual_lambda": 0.5,
        "candidates": ["xonly_weighted_shared", "shrunk_study_deviation"],
        "ridge_alpha": 10.0,
        "correction_shrinkage": 0.25,
        "absolute_rank_correction_cap": 0.05,
        "source_weight_clip": [0.5, 2.0],
        "source_row_weight_mean": 1.0,
        "deviation_scale": 0.25,
        "effective_deviation_penalty_multiplier": 16.0,
        "minimum_source_subjects": 8,
        "inner_loso_crossfit_residual": True,
        "held_outcomes_used_for_domain_weights": False,
        "isolated_source_study_models_allowed": False,
        "promotion_mean_delta": 0.02,
        "promotion_minimum_study_delta": -0.10,
        "promotion_requires_strict_majority_wins": True,
        "public_probe_authorized": False,
        "automatic_incumbent_change_authorized": False,
    }
    for key, value in locked.items():
        if contract.get(key) != value:
            raise SystemExit(f"E10 experiment contract mismatch:{key}")


def replace_block(text: str, start: str, end: str, replacement: str) -> str:
    a = text.find(start)
    b = text.find(end, a + len(start))
    if a < 0 or b < 0 or b <= a:
        raise SystemExit(f"E10 runtime block anchors changed:{start!r}->{end!r}")
    return text[:a] + replacement.rstrip() + "\n" + text[b:]


def build_parent_runtime(root: Path, reference_dir: Path):
    verify_parent_contract()
    return e09b_parent.build_parent_runtime(root, reference_dir)


def patch_runtime(v3, runtime: str, e10: str, synth: str, anchor: str, rank_transfer: str, study_similarity: str) -> str:
    for old, new in (
        (OLD_REQUEST_ID, REQUEST_ID),
        (OLD_TARGET_KERNEL, TARGET_KERNEL),
        (OLD_SCIENCE_COMMIT, SCIENCE_COMMIT),
    ):
        if runtime.count(old) < 1:
            raise SystemExit(f"E10 identity anchor missing:{old}")
        runtime = runtime.replace(old, new)

    marker = 'E05_V4_BLOB = "ab93726fe0810299892091b7e0e8f63dcacec4f3"\n'
    if runtime.count(marker) != 1:
        raise SystemExit("E10 source injection anchor changed")
    injected = (
        f'E10_BLOB = "{E10_BLOB}"\n'
        f'E10_SYNTH_BLOB = "{E10_SYNTH_BLOB}"\n'
        f'ANCHOR_BLOB = "{ANCHOR_BLOB}"\n'
        f'RANK_TRANSFER_BLOB = "{RANK_TRANSFER_BLOB}"\n'
        f'STUDY_SIMILARITY_BLOB = "{STUDY_SIMILARITY_BLOB}"\n'
        f'E10_SOURCE = {e10!r}\n'
        f'E10_SYNTH_SOURCE = {synth!r}\n'
        f'ANCHOR_SOURCE = {anchor!r}\n'
        f'RANK_TRANSFER_SOURCE = {rank_transfer!r}\n'
        f'STUDY_SIMILARITY_SOURCE = {study_similarity!r}\n'
    )
    runtime = runtime.replace(marker, marker + injected, 1)

    parse_replacement = '''def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    return p.parse_args()'''
    runtime = v3.replace_top_level_function(runtime, "parse_args", parse_replacement)

    load_replacement = '''def load_e10_modules() -> tuple[object, object, object]:
    load_e05_module()
    def install(name: str, source: str) -> object:
        module = types.ModuleType(name)
        module.__file__ = f"<{name}>"; module.__package__ = "cmi_flu"; sys.modules[name] = module
        exec(compile(source, name.replace(".", "/") + ".py", "exec"), module.__dict__, module.__dict__)
        return module
    install("cmi_flu.rank_transfer", RANK_TRANSFER_SOURCE)
    install("cmi_flu.anchor_residual", ANCHOR_SOURCE)
    install("cmi_flu.study_similarity", STUDY_SIMILARITY_SOURCE)
    e10 = install("cmi_flu.strategy_e10", E10_SOURCE)
    synthetic = install("cmi_flu.strategy_e10_synthetic", E10_SYNTH_SOURCE)
    run = getattr(e10, "run_e10", None); run_synthetic = getattr(synthetic, "run_synthetic", None)
    if not callable(run) or not callable(run_synthetic):
        raise BridgeContractError("e10_entry_missing")
    return run, run_synthetic, e10'''
    runtime = runtime.replace(
        "def validate_result(result: dict) -> None:\n",
        load_replacement + "\n\ndef validate_result(result: dict) -> None:\n",
        1,
    )

    self_test_replacement = '''def self_test() -> int:
    package = package_bytes()
    for source, expected, label in (
        (E10_SOURCE,E10_BLOB,"e10"),
        (E10_SYNTH_SOURCE,E10_SYNTH_BLOB,"e10_synthetic"),
        (ANCHOR_SOURCE,ANCHOR_BLOB,"anchor_residual"),
        (RANK_TRANSFER_SOURCE,RANK_TRANSFER_BLOB,"rank_transfer"),
        (STUDY_SIMILARITY_SOURCE,STUDY_SIMILARITY_BLOB,"study_similarity"),
    ):
        if git_blob_sha(source.encode("utf-8")) != expected:
            raise BridgeContractError(f"{label}_blob_mismatch")
        compile(source, f"cmi_flu/{label}.py", "exec")
    print(f"CMI_FLU_E10_RUNTIME_SELF_TEST PASS request_id={REQUEST_ID} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} e10_blob={E10_BLOB}")
    return 0'''
    runtime = v3.replace_top_level_function(runtime, "self_test", self_test_replacement)

    validate_replacement = '''def validate_result(result: dict, *, synthetic: bool = False) -> None:
    if synthetic:
        if int(result.get("schema_version", -1)) != 1 or result.get("experiment") != "synthetic_e10_contract":
            raise BridgeContractError("e10_synthetic_identity_mismatch")
        if result.get("held_outcomes_used") is not False:
            raise BridgeContractError("e10_synthetic_xonly_boundary")
        if not (0.5 - 1e-12 <= float(result.get("source_weight_min", -1)) <= float(result.get("source_weight_max", 99)) <= 2.0 + 1e-12):
            raise BridgeContractError("e10_synthetic_weight_clip")
        if abs(float(result.get("row_weight_mean", 99)) - 1.0) > 1e-8:
            raise BridgeContractError("e10_synthetic_weight_mean")
        if not (0.0 < float(result.get("row_weight_ess_fraction", 0)) <= 1.0 + 1e-12):
            raise BridgeContractError("e10_synthetic_ess")
        for key in ("weighted_finite","hierarchical_finite","candidate_finite","promotion_passed"):
            if result.get(key) is not True:
                raise BridgeContractError(f"e10_synthetic_boolean:{key}")
        if result.get("weighted_pooled_model_count") != 1 or result.get("weighted_isolated_source_models") != 0:
            raise BridgeContractError("e10_synthetic_weighted_pooling")
        if result.get("hierarchical_pooled_model_count") != 1 or result.get("hierarchical_isolated_source_models") != 0 or result.get("hierarchical_unseen_deviation") != "all_zero":
            raise BridgeContractError("e10_synthetic_hierarchical_pooling")
        if float(result.get("movement_cap", 99)) > 0.050000000001:
            raise BridgeContractError("e10_synthetic_movement_cap")
        if result.get("competition_submission_attempted") is not False or result.get("leaderboard_used_for_selection") is not False:
            raise BridgeContractError("e10_synthetic_execution_boundary")
        return

    if int(result.get("schema_version", -1)) != 1 or result.get("experiment") != "strategy_v2_e10_pooled_domain_conditioned_residual":
        raise BridgeContractError("e10_experiment_identity_mismatch")
    if result.get("comparison_contract") != "paired_subject_purged_v2" or set((result.get("tasks") or {})) != {"Task1.1","Task1.2"}:
        raise BridgeContractError("e10_task_or_comparison_contract")
    frozen = result.get("frozen_conditions") or {}
    expected_frozen = {
        "ridge_alpha": 10.0,
        "correction_shrinkage": 0.25,
        "absolute_rank_correction_cap": 0.05,
        "source_weight_clip": [0.5,2.0],
        "deviation_scale": 0.25,
        "minimum_source_subjects": 8,
        "promotion_mean_delta": 0.02,
        "promotion_minimum_study_delta": -0.10,
        "task_contract": {
            "Task1.1": {"model_set":"task_11","base_model":"pls_2","base_kind":"b21","assay_prefix":"cytokine_","anchor_column":None,"anchor_lambda":None},
            "Task1.2": {"model_set":"task_12","base_model":"et_d5_l5_sqrt","base_kind":"anchor_residual","assay_prefix":"flow_","anchor_column":"flow_rank__Classical_monocytes","anchor_lambda":0.5},
        },
    }
    if frozen != expected_frozen:
        raise BridgeContractError("e10_frozen_condition_mismatch")
    if result.get("held_target_outcomes_used_for_domain_weights") is not False or result.get("isolated_source_study_models_allowed") is not False:
        raise BridgeContractError("e10_domain_boundary_mismatch")
    if result.get("leaderboard_used_for_selection") is not False or result.get("competition_submission_attempted") is not False or result.get("incumbent_changed") is not False or result.get("public_probe_authorized") is not False or int(result.get("automatic_compute_retries", -1)) != 0:
        raise BridgeContractError("e10_execution_boundary_mismatch")

    expected_bases = {
        "Task1.1": {"kind":"b21","model":"pls_2","anchor_column":None,"anchor_lambda":None,"current_competition_incumbent":True},
        "Task1.2": {"kind":"anchor_residual","model":"et_d5_l5_sqrt","anchor_column":"flow_rank__Classical_monocytes","anchor_lambda":0.5,"current_competition_incumbent":True},
    }
    expected_folds = {"Task1.1":4,"Task1.2":3}
    for task, payload in result["tasks"].items():
        if payload.get("task") != task or payload.get("base_contract") != expected_bases[task]:
            raise BridgeContractError(f"e10_base_contract:{task}")
        folds = payload.get("folds") or []
        if len(folds) != expected_folds[task]:
            raise BridgeContractError(f"e10_fold_count:{task}:{len(folds)}")
        if set((payload.get("candidates") or {})) != {"xonly_weighted_shared","shrunk_study_deviation"}:
            raise BridgeContractError(f"e10_candidate_set:{task}")
        if int(payload.get("challenge_rows", -1)) != 40:
            raise BridgeContractError(f"e10_challenge_rows:{task}")
        for fold in folds:
            if int(fold.get("n", 0)) < 3 or int(fold.get("training_studies", 0)) < 2:
                raise BridgeContractError(f"e10_fold_support:{task}")
            for metric in ("base_spearman","xonly_weighted_shared_spearman","shrunk_study_deviation_spearman","xonly_weighted_shared_delta","shrunk_study_deviation_delta"):
                if not math.isfinite(float(fold.get(metric, float("nan")))):
                    raise BridgeContractError(f"e10_nonfinite_fold:{task}:{metric}")
            audit = fold.get("xonly_weight_audit") or {}
            weights = audit.get("source_weights") or {}
            counts = audit.get("source_counts") or {}
            if set(weights) != set(counts) or len(weights) < 2 or min(int(v) for v in counts.values()) < 8:
                raise BridgeContractError(f"e10_source_support:{task}")
            if min(float(v) for v in weights.values()) < 0.5 - 1e-12 or max(float(v) for v in weights.values()) > 2.0 + 1e-12:
                raise BridgeContractError(f"e10_weight_clip:{task}")
            if abs(float(audit.get("row_weight_mean", 99)) - 1.0) > 1e-8 or not (0.0 < float(audit.get("row_weight_ess_fraction", 0)) <= 1.0 + 1e-12) or audit.get("held_outcomes_used") is not False:
                raise BridgeContractError(f"e10_weight_audit:{task}")
            if float((fold.get("xonly_weighted_movement") or {}).get("max_absolute_rank_correction", 99)) > 0.050000000001 or float((fold.get("shrunk_study_deviation_movement") or {}).get("max_absolute_rank_correction", 99)) > 0.050000000001:
                raise BridgeContractError(f"e10_fold_movement_cap:{task}")
            wf = fold.get("xonly_weighted_fit") or {}; hf = fold.get("shrunk_study_deviation_fit") or {}
            if wf.get("pooled_model_count") != 1 or wf.get("isolated_source_models_fit") != 0:
                raise BridgeContractError(f"e10_weighted_pooling:{task}")
            if hf.get("pooled_model_count") != 1 or hf.get("isolated_source_models_fit") != 0 or hf.get("unseen_target_deviation_columns") != "all_zero":
                raise BridgeContractError(f"e10_hierarchical_pooling:{task}")
        for name, candidate in payload["candidates"].items():
            promotion = candidate.get("promotion") or {}
            deltas = [float(fold[f"{name}_delta"]) for fold in folds]
            wins = sum(value > 0 for value in deltas); required = len(deltas)//2 + 1
            expected_pass = bool(np.mean(deltas) >= 0.02 and min(deltas) >= -0.10 and wins >= required)
            if promotion.get("passed") is not expected_pass or int(promotion.get("wins", -1)) != wins or int(promotion.get("required_wins", -1)) != required:
                raise BridgeContractError(f"e10_promotion_boolean:{task}:{name}")
            agreement = candidate.get("challenge_agreement_vs_base") or {}
            if not math.isfinite(float((agreement.get("rank_spearman") or {}).get("value", float("nan")))) or not (0 <= int(agreement.get("changed_rank_count", -1)) <= 40):
                raise BridgeContractError(f"e10_challenge_agreement:{task}:{name}")
            if float((candidate.get("challenge_movement") or {}).get("max_absolute_rank_correction", 99)) > 0.050000000001:
                raise BridgeContractError(f"e10_challenge_movement_cap:{task}:{name}")
        passing = [name for name,candidate in payload["candidates"].items() if candidate["promotion"]["passed"]]
        selected = payload.get("selected_local_candidate")
        if (selected is None) != (len(passing) == 0) or (selected is not None and selected not in passing):
            raise BridgeContractError(f"e10_local_selection:{task}")
        if payload.get("competition_candidate") is not False or payload.get("public_probe_authorized") is not False:
            raise BridgeContractError(f"e10_competition_boundary:{task}")
        ca = payload.get("challenge_source_weight_audit") or {}; cw = ca.get("source_weights") or {}
        if len(cw) < 2 or min(float(v) for v in cw.values()) < 0.5-1e-12 or max(float(v) for v in cw.values()) > 2.0+1e-12 or abs(float(ca.get("row_weight_mean",99))-1.0)>1e-8 or not (0.0 < float(ca.get("row_weight_ess_fraction",0)) <= 1.0+1e-12) or ca.get("held_outcomes_used") is not False:
            raise BridgeContractError(f"e10_challenge_weight_audit:{task}")
    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"','ROW_','SUB_')
    if any(token in serialized for token in banned):
        raise BridgeContractError("e10_aggregate_privacy_contract")'''
    # numpy is already imported by the parent runtime as np.
    runtime = v3.replace_top_level_function(runtime, "validate_result", validate_replacement)

    summary_replacement = '''def render_summary(result: dict) -> str:
    lines = [
        "# CMI-Flu strategy E10 pooled domain-conditioned correction", "",
        "One-pooled-model X-only/domain-conditioned residual audit; aggregate-only; no Competition submission or Public probe.", "",
        f"- science commit: `{SCIENCE_COMMIT}`", f"- E10 blob: `{E10_BLOB}`", "",
    ]
    for task,payload in (result.get("tasks") or {}).items():
        lines.append(f"## {task}")
        base = payload.get("base_contract") or {}
        lines.append(f"- base: `{base.get('kind')}/{base.get('model')}`")
        for name,candidate in (payload.get("candidates") or {}).items():
            p = candidate.get("promotion") or {}; a = candidate.get("challenge_agreement_vs_base") or {}
            lines.append(f"- {name}: pass=`{p.get('passed')}` mean_delta=`{p.get('mean_delta')}` min_delta=`{p.get('minimum_delta')}` wins=`{p.get('wins')}/{p.get('required_wins')}` Challenge_rank_agreement=`{(a.get('rank_spearman') or {}).get('value')}` changed=`{a.get('changed_rank_count')}`")
        lines.append(f"- selected local candidate: `{payload.get('selected_local_candidate')}`; competition candidate=`False`")
        lines.append("")
    return "\\n".join(lines) + "\\n"'''
    runtime = v3.replace_top_level_function(runtime, "render_summary", summary_replacement)

    runtime = replace_block(
        runtime,
        '        stage = "load_e05"\n',
        '        stage = "write_outputs"\n',
        '''        stage = "load_e10"
        run_e10, run_synthetic, e10_module = load_e10_modules()
        stage = "run_e10"
        result = json_safe(dict(run_e10(config, inputs)))
        stage = "validate_e10"
        validate_result(result, synthetic=False)''',
    )

    provenance = '            "strategy_e05_v4_blob_sha": E05_V4_BLOB,\n'
    if runtime.count(provenance) != 1:
        raise SystemExit("E10 bridge provenance anchor changed")
    runtime = runtime.replace(
        provenance,
        provenance
        + '            "strategy_e10_blob_sha": E10_BLOB,\n'
        + '            "strategy_e10_synthetic_blob_sha": E10_SYNTH_BLOB,\n'
        + '            "anchor_residual_blob_sha": ANCHOR_BLOB,\n'
        + '            "rank_transfer_blob_sha": RANK_TRANSFER_BLOB,\n'
        + '            "study_similarity_blob_sha": STUDY_SIMILARITY_BLOB,\n',
        1,
    )

    main_replacement = '''def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.synthetic:
        output_dir = args.output_dir.expanduser().resolve(); output_dir.mkdir(parents=True, exist_ok=False)
        runtime_root = Path("/tmp") / "cmi-flu-e10-synthetic-runtime"
        if runtime_root.exists(): shutil.rmtree(runtime_root)
        runtime_root.mkdir(parents=True)
        package_path = runtime_root / "cmi_flu_bundle.zip"; package_path.write_bytes(package_bytes()); sys.path.insert(0, str(package_path))
        try:
            run_e10, run_synthetic, e10 = load_e10_modules()
            result = json_safe(dict(run_synthetic()))
            validate_result(result, synthetic=True)
            metrics_path=output_dir/"metrics.json"; summary_path=output_dir/"summary.md"; bridge_path=output_dir/"bridge-result.json"
            metrics_path.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\\n",encoding="utf-8")
            summary_path.write_text("# CMI-Flu E10 synthetic contract\\n\\nSynthetic-only validation; no Competition submission.\\n",encoding="utf-8")
            bridge={"schema_version":1,"request_id":REQUEST_ID,"competition":COMPETITION,"target_kernel":TARGET_KERNEL,"science_commit":SCIENCE_COMMIT,"strategy_e10_blob_sha":E10_BLOB,"strategy_e10_synthetic_blob_sha":E10_SYNTH_BLOB,"anchor_residual_blob_sha":ANCHOR_BLOB,"rank_transfer_blob_sha":RANK_TRANSFER_BLOB,"study_similarity_blob_sha":STUDY_SIMILARITY_BLOB,"metrics_sha256":sha256_file(metrics_path),"summary_sha256":sha256_file(summary_path),"competition_submission_attempted":False,"leaderboard_used_for_selection":False,"contains_participant_identifiers":False,"contains_row_level_predictions":False,"synthetic":True}
            bridge_path.write_text(json.dumps(bridge,indent=2,sort_keys=True)+"\\n",encoding="utf-8")
            print("CMI_FLU_E10_COMPLETE synthetic=true submission=false")
            return 0
        finally:
            shutil.rmtree(runtime_root,ignore_errors=True)
    try:
        input_dir=locate_competition_data(args.input_dir)
    except Exception as exc:
        code=hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode()).hexdigest()[:20]
        print(f"CMI_FLU_E10_FAILED stage=locate_competition_data exception_type={type(exc).__name__} error_code={code}",file=sys.stderr)
        return 2
    return execute(input_dir,args.output_dir)'''
    runtime = v3.replace_top_level_function(runtime, "main", main_replacement)

    runtime = runtime.replace(
        "CMI-Flu strategy E05 HAI donor x strain. Aggregate outputs only; no submission.",
        "CMI-Flu strategy E10 pooled domain correction. Aggregate outputs only; no submission.",
    )
    runtime = runtime.replace("CMI_FLU_E05_FAILED","CMI_FLU_E10_FAILED").replace("CMI_FLU_E05_COMPLETE","CMI_FLU_E10_COMPLETE")
    names = v3.top_level_functions(runtime)
    for name in ("parse_args","self_test","load_e05_module","load_e10_modules","validate_result","render_summary","execute","main"):
        if names.count(name) != 1:
            raise SystemExit(f"E10 top-level binding contract failed:{name}:{names.count(name)}")
    for old in (OLD_REQUEST_ID,OLD_TARGET_KERNEL,OLD_SCIENCE_COMMIT):
        if old in runtime:
            raise SystemExit(f"E10 prior identity remained:{old}")
    if "kaggle competitions submit" in runtime or "competition_submit" in runtime:
        raise SystemExit("E10 generated runtime contains submission path")
    compile(runtime,"generated_e10_runtime.py","exec")
    return runtime


def main() -> int:
    args=parse_args(); root=args.repository_root.expanduser().resolve(); reference_dir=args.reference_dir.expanduser().resolve()
    validate_request(root)
    e10,synth,anchor,rank_transfer,study_similarity=load_exact_sources(root)
    v3,parent_runtime=build_parent_runtime(root,reference_dir)
    runtime=patch_runtime(v3,parent_runtime,e10,synth,anchor,rank_transfer,study_similarity)
    out=args.output.expanduser().resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(runtime,encoding="utf-8")
    subprocess.run([sys.executable,str(out),"--self-test"],check=True)
    print(f"CMI_FLU_E10_PREPARE_PASS science_commit={SCIENCE_COMMIT} request_id={REQUEST_ID} target={TARGET_KERNEL} e10_blob={E10_BLOB} runtime_sha256={sha256(runtime.encode())} e05_v6_ancestry=true submission=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
