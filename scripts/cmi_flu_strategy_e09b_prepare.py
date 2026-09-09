#!/usr/bin/env python3
"""Build exact E09b HIPC9 RNA residual runtime from proven E05 v6 ancestry."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cmi_flu_strategy_e05_prepare_v6 as prior

REQUEST_ID = "20260909-cmi-flu-strategy-e09b-hipc9-rna-residual-001"
TARGET_KERNEL = "renta0426/cmi-flu-e09b-hipc9-rna-residual-20260909-001"
SCIENCE_COMMIT = "8f36ba0fa0d5fae6809ff09e0bf15bb132dd1ff5"
E09B_BLOB = "6a58b1f0a09ac6b2b07473627af37847d9ac41c4"
E09B_SYNTH_BLOB = "3c5a08f5bd209ae094e4fcc1d66f65abe949c018"
MAPPING_BLOB = "ae42cab3dfffb242dd695a6b1df0357fc90cf2d0"
E09A_BLOB = "37270a0a8aafc070d91b5f43fbea3ba169cd61b0"
E09B_PATH = "payloads/cmi-flu-strategy-e09b-hipc9-rna-residual-001/strategy_e09b.py"
E09B_SYNTH_PATH = "payloads/cmi-flu-strategy-e09b-hipc9-rna-residual-001/strategy_e09b_synthetic.py"
MAPPING_PATH = "payloads/cmi-flu-strategy-e09b-hipc9-rna-residual-001/strategy_e09b_hipc9.json"
E09A_PATH = "payloads/cmi-flu-strategy-e09a-module-teacher-audit-001/strategy_e09a.py"
REQUEST_PATH = "requests/cmi-flu-strategy-e09b-hipc9-rna-residual-001.json"
OLD_REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-007"
OLD_TARGET_KERNEL = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-007"
OLD_SCIENCE_COMMIT = "92fac34e52485f6f73ea1b7aedb983d8e8e0ee27"
EXPECTED_GENES = ["RAB24", "GRB2", "DPP3", "ACTB", "MVP", "DPP7", "ARPC4", "PLEKHB2", "ARRB1"]
EXPECTED_ENSEMBL = [
    "ENSG00000169228", "ENSG00000177885", "ENSG00000254986",
    "ENSG00000075624", "ENSG00000013364", "ENSG00000176978",
    "ENSG00000241553", "ENSG00000115762", "ENSG00000137486",
]


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
        raise SystemExit(f"E09b exact relay mismatch:{label}:{found}")
    text = data.decode("utf-8")
    if relative.endswith(".py"):
        compile(text, f"cmi_flu/{Path(relative).name}", "exec")
    return text


def load_exact_sources(root: Path) -> tuple[str, str, str, str]:
    e09b = require_text(root, E09B_PATH, E09B_BLOB, "e09b")
    synth = require_text(root, E09B_SYNTH_PATH, E09B_SYNTH_BLOB, "e09b_synthetic")
    mapping = require_text(root, MAPPING_PATH, MAPPING_BLOB, "hipc9_mapping")
    e09a = require_text(root, E09A_PATH, E09A_BLOB, "e09a")
    required = (
        'EXPERIMENT = "strategy_v2_e09b_hipc9_rna_residual"',
        'SOURCE_DOI = "10.1126/sciimmunol.aal4656"',
        'QUALIFYING_STUDIES = ("2020_UGA", "2024_UGA")',
        'RIDGE_ALPHA = 10.0',
        'RESIDUAL_SHRINKAGE = 0.25',
        'MAX_SCORE_CORRECTION = 0.10',
        'MIN_GENE_COVERAGE = 7',
        'PROMOTION_MEAN_DELTA = 0.02',
        'PROMOTION_MIN_STUDY_DELTA = -0.05',
        '"competition_submission_attempted": False',
        '"public_leaderboard_used_for_selection": False',
    )
    if any(token not in e09b for token in required):
        raise SystemExit("E09b frozen science token missing")
    if "kaggle competitions submit" in e09b or "competition_submit" in e09b:
        raise SystemExit("E09b source contains submission path")
    parsed = json.loads(mapping)
    genes = [row["symbol"] for row in parsed.get("gene_mapping", [])]
    ensembl = [row["ensembl_gene_id"] for row in parsed.get("gene_mapping", [])]
    if genes != EXPECTED_GENES or ensembl != EXPECTED_ENSEMBL:
        raise SystemExit("E09b mapping file contract changed")
    if parsed.get("outcome_based_gene_selection") is not False or parsed.get("real_data_inspected_before_freeze") is not False:
        raise SystemExit("E09b mapping freeze provenance changed")
    return e09b, synth, mapping, e09a


def verify_prior_contract() -> None:
    expected = {
        "REQUEST_ID": OLD_REQUEST_ID,
        "TARGET_KERNEL": OLD_TARGET_KERNEL,
        "SCIENCE_COMMIT": OLD_SCIENCE_COMMIT,
        "E05_BLOB": "78cde9e3a6b6e3b332352218c4b4545f769aa6c4",
        "E05_V2_BLOB": "50951807d28418bb50f4e9dba656fa2b3e4d86ff",
        "E05_V3_BLOB": "129a4a5713a758d58adbb02489105e17a1c13922",
        "E05_V4_BLOB": "ab93726fe0810299892091b7e0e8f63dcacec4f3",
        "HAI_TRANSFER_BLOB": "b671d8bf7f10bebbd65aca2a5bad42e267ee78d5",
        "HAI_TRANSFER_V2_BLOB": "ebf4110df4ab49d726d80af03c1203f93e9c89c0",
        "CONFIG_BLOB": "170d3211e2795c0730e481056c7bb068accf97c9",
    }
    for key, value in expected.items():
        if getattr(prior, key, None) != value:
            raise SystemExit(f"E09b E05-v6 ancestry changed:{key}")


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
        "strategy_e09b_blob_sha": E09B_BLOB,
        "strategy_e09b_synthetic_blob_sha": E09B_SYNTH_BLOB,
        "strategy_e09b_mapping_blob_sha": MAPPING_BLOB,
        "strategy_e09a_blob_sha": E09A_BLOB,
        "expected_kernel_version": 1,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"E09b request mismatch:{key}")
    if request.get("resource") != {"accelerator": "cpu", "expected_runtime_minutes": 45, "hard_timeout_minutes": 120, "max_active_runs": 1}:
        raise SystemExit("E09b resource contract mismatch")
    if request.get("allowed_output_paths") != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E09b output allowlist mismatch")
    contract = request.get("experiment_contract") or {}
    locked = {
        "experiment": "strategy_v2_e09b_hipc9_rna_residual",
        "tasks": ["Task2.1", "Task2.2"],
        "qualifying_studies": ["2020_UGA", "2024_UGA"],
        "literature_doi": "10.1126/sciimmunol.aal4656",
        "gene_symbols": EXPECTED_GENES,
        "raw_day0_tpm_only": True,
        "within_study_gene_rank": True,
        "minimum_gene_coverage": 7,
        "ridge_alpha": 10.0,
        "residual_shrinkage": 0.25,
        "max_score_correction": 0.1,
        "minimum_fold_subjects": 24,
        "promotion_mean_delta": 0.02,
        "promotion_min_study_delta": -0.05,
        "shuffle_seed": 20260909,
        "outcome_based_gene_selection": False,
        "batch_corrected_challenge_expression_used": False,
        "historical_panels_are_incomplete_proxies": True,
        "official_complete_2025_panel_cv_claimed": False,
        "challenge_missing_rna_policy": "exact_frozen_b21_fallback",
        "public_probe_authorized": False,
        "automatic_incumbent_change_authorized": False,
    }
    for key, value in locked.items():
        if contract.get(key) != value:
            raise SystemExit(f"E09b experiment contract mismatch:{key}")


def replace_block(text: str, start: str, end: str, replacement: str) -> str:
    a = text.find(start)
    b = text.find(end, a + len(start))
    if a < 0 or b < 0 or b <= a:
        raise SystemExit(f"E09b runtime block anchors changed:{start!r}->{end!r}")
    return text[:a] + replacement.rstrip() + "\n" + text[b:]


def build_parent_runtime(root: Path, reference_dir: Path):
    verify_prior_contract()
    v5 = prior.load_v5(root)
    e05_v4 = prior.load_exact_source(root)
    v3, base_runtime = prior.build_v5_runtime(v5, root, reference_dir)
    return v3, prior.patch_runtime(v3, base_runtime, e05_v4)


def patch_runtime(v3, runtime: str, e09b: str, synth: str, mapping: str, e09a: str) -> str:
    for old, new in (
        (OLD_REQUEST_ID, REQUEST_ID),
        (OLD_TARGET_KERNEL, TARGET_KERNEL),
        (OLD_SCIENCE_COMMIT, SCIENCE_COMMIT),
    ):
        if runtime.count(old) < 1:
            raise SystemExit(f"E09b identity anchor missing:{old}")
        runtime = runtime.replace(old, new)

    marker = 'E05_V4_BLOB = "ab93726fe0810299892091b7e0e8f63dcacec4f3"\n'
    if runtime.count(marker) != 1:
        raise SystemExit("E09b source injection anchor changed")
    injected = (
        f'E09A_BLOB = "{E09A_BLOB}"\n'
        f'E09B_BLOB = "{E09B_BLOB}"\n'
        f'E09B_SYNTH_BLOB = "{E09B_SYNTH_BLOB}"\n'
        f'E09B_MAPPING_BLOB = "{MAPPING_BLOB}"\n'
        f'E09A_SOURCE = {e09a!r}\n'
        f'E09B_SOURCE = {e09b!r}\n'
        f'E09B_SYNTH_SOURCE = {synth!r}\n'
        f'E09B_MAPPING_TEXT = {mapping!r}\n'
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

    load_replacement = '''def load_e09b_modules() -> tuple[object, object, object]:
    # First install the proven exact E05-v6 ancestry and its compatibility patches.
    load_e05_module()
    # E09b imports only `_material_category` from E09a. Execute the exact E09a
    # blob while fail-closing its unrelated E04 dependencies if accidentally used.
    def unavailable(*args, **kwargs):
        raise BridgeContractError("e09a_unrelated_dependency_called_from_e09b")
    e04c = types.ModuleType("cmi_flu.strategy_e04_contract")
    e04c.__file__ = "<cmi_flu.strategy_e04_contract:e09b-guard>"; e04c.__package__ = "cmi_flu"
    e04c.audit_measurements = unavailable; sys.modules["cmi_flu.strategy_e04_contract"] = e04c
    e04b = types.ModuleType("cmi_flu.strategy_e04b")
    e04b.__file__ = "<cmi_flu.strategy_e04b:e09b-guard>"; e04b.__package__ = "cmi_flu"
    e04b.apply_material_plural_ontology = unavailable; sys.modules["cmi_flu.strategy_e04b"] = e04b
    e09a = types.ModuleType("cmi_flu.strategy_e09a")
    e09a.__file__ = "<cmi_flu.strategy_e09a>"; e09a.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e09a"] = e09a
    exec(compile(E09A_SOURCE, "cmi_flu/strategy_e09a.py", "exec"), e09a.__dict__, e09a.__dict__)
    if getattr(e09a, "_material_category", lambda _x: None)("PBMCs") != "PBMC":
        raise BridgeContractError("e09a_material_category_contract_mismatch")
    e09b = types.ModuleType("cmi_flu.strategy_e09b")
    e09b.__file__ = "<cmi_flu.strategy_e09b>"; e09b.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e09b"] = e09b
    exec(compile(E09B_SOURCE, "cmi_flu/strategy_e09b.py", "exec"), e09b.__dict__, e09b.__dict__)
    synthetic = types.ModuleType("cmi_flu.strategy_e09b_synthetic")
    synthetic.__file__ = "<cmi_flu.strategy_e09b_synthetic>"; synthetic.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e09b_synthetic"] = synthetic
    exec(compile(E09B_SYNTH_SOURCE, "cmi_flu/strategy_e09b_synthetic.py", "exec"), synthetic.__dict__, synthetic.__dict__)
    run = getattr(e09b, "run_strategy_e09b", None)
    run_synthetic = getattr(synthetic, "run_synthetic", None)
    if not callable(run) or not callable(run_synthetic):
        raise BridgeContractError("e09b_entry_missing")
    return run, run_synthetic, e09b'''
    runtime = runtime.replace("def validate_result(result: dict) -> None:\n", load_replacement + "\n\ndef validate_result(result: dict) -> None:\n", 1)

    self_test_replacement = '''def self_test() -> int:
    package = package_bytes()
    for source, expected, label in (
        (E09A_SOURCE,E09A_BLOB,"e09a"),
        (E09B_SOURCE,E09B_BLOB,"e09b"),
        (E09B_SYNTH_SOURCE,E09B_SYNTH_BLOB,"e09b_synthetic"),
        (E09B_MAPPING_TEXT,E09B_MAPPING_BLOB,"e09b_mapping"),
    ):
        if git_blob_sha(source.encode("utf-8")) != expected:
            raise BridgeContractError(f"{label}_blob_mismatch")
    compile(E09A_SOURCE, "cmi_flu/strategy_e09a.py", "exec")
    compile(E09B_SOURCE, "cmi_flu/strategy_e09b.py", "exec")
    compile(E09B_SYNTH_SOURCE, "cmi_flu/strategy_e09b_synthetic.py", "exec")
    mapping = json.loads(E09B_MAPPING_TEXT)
    if [row.get("symbol") for row in mapping.get("gene_mapping", [])] != EXPECTED_GENES:
        raise BridgeContractError("e09b_mapping_symbols_mismatch")
    print(f"CMI_FLU_E09B_RUNTIME_SELF_TEST PASS request_id={REQUEST_ID} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} e09b_blob={E09B_BLOB}")
    return 0'''.replace("EXPECTED_GENES", repr(EXPECTED_GENES))
    runtime = v3.replace_top_level_function(runtime, "self_test", self_test_replacement)

    validate_replacement = '''def validate_result(result: dict, *, synthetic: bool = False) -> None:
    if int(result.get("schema_version", -1)) != 1 or result.get("experiment") != "strategy_v2_e09b_hipc9_rna_residual":
        raise BridgeContractError("e09b_experiment_identity_mismatch")
    if set((result.get("tasks") or {})) != {"Task2.1", "Task2.2"}:
        raise BridgeContractError("e09b_task_set_mismatch")
    feature = result.get("feature_contract") or {}
    expected_feature = {
        "source_doi": "10.1126/sciimmunol.aal4656",
        "source_pmid": "28842433",
        "gene_symbols": ["RAB24","GRB2","DPP3","ACTB","MVP","DPP7","ARPC4","PLEKHB2","ARRB1"],
        "ensembl_ids": ["ENSG00000169228","ENSG00000177885","ENSG00000254986","ENSG00000075624","ENSG00000013364","ENSG00000176978","ENSG00000241553","ENSG00000115762","ENSG00000137486"],
        "validation_subset": ["GRB2","ACTB","MVP","DPP7","ARPC4","PLEKHB2","ARRB1"],
        "day0_only": True,
        "raw_tpm_only": True,
        "within_study_gene_rank": True,
        "minimum_gene_coverage": 7,
        "ridge_alpha": 10.0,
        "residual_shrinkage": 0.25,
        "max_score_correction": 0.1,
        "minimum_fold_subjects": 24,
        "outcome_based_gene_selection": False,
        "batch_corrected_challenge_expression_used": False,
    }
    for key, value in expected_feature.items():
        if feature.get(key) != value:
            raise BridgeContractError(f"e09b_feature_contract_mismatch:{key}")
    evaluation = result.get("evaluation_contract") or {}
    expected_evaluation = {
        "qualifying_studies": ["2020_UGA", "2024_UGA"],
        "historical_panels_are_incomplete_proxies": True,
        "official_complete_2025_panel_cv_claimed": False,
        "promotion_mean_delta": 0.02,
        "promotion_min_study_delta": -0.05,
        "shuffle_seed": 20260909,
        "challenge_missing_rna_policy": "exact_frozen_b21_fallback",
    }
    for key, value in expected_evaluation.items():
        if evaluation.get(key) != value:
            raise BridgeContractError(f"e09b_evaluation_contract_mismatch:{key}")
    if result.get("competition_submission_attempted") is not False or result.get("public_leaderboard_used_for_selection") is not False or int(result.get("automatic_compute_retries", -1)) != 0 or result.get("incumbent_changed") is not False:
        raise BridgeContractError("e09b_execution_boundary_mismatch")
    for task, payload in result["tasks"].items():
        if payload.get("task") != task or int(payload.get("historical_joined_subjects", 0)) < 48:
            raise BridgeContractError(f"e09b_historical_support_mismatch:{task}")
        studies = payload.get("study_metrics") or []
        if [row.get("study") for row in studies] != ["2020_UGA", "2024_UGA"]:
            raise BridgeContractError(f"e09b_study_set_mismatch:{task}")
        for row in studies:
            if int(row.get("n", 0)) < 24:
                raise BridgeContractError(f"e09b_study_n_mismatch:{task}:{row.get('study')}")
            for key in ("base_spearman","candidate_spearman","hipc7_zero_shot_spearman","shuffle_spearman","delta_vs_base"):
                if not math.isfinite(float(row.get(key, float("nan")))):
                    raise BridgeContractError(f"e09b_nonfinite_study_metric:{task}:{key}")
        for key in ("study_equal_base_spearman","study_equal_candidate_spearman","study_equal_shuffle_spearman","study_equal_delta_vs_base","minimum_study_delta_vs_base"):
            if not math.isfinite(float(payload.get(key, float("nan")))):
                raise BridgeContractError(f"e09b_nonfinite_summary:{task}:{key}")
        movement = payload.get("challenge_movement") or {}
        donors = int(movement.get("donors", -1)); available = int(movement.get("rna_available_donors", -1)); fallback = int(movement.get("frozen_base_fallback_donors", -1)); changed = int(movement.get("changed_rank_count_vs_base", -1))
        expected_donors = 20 if synthetic else 40
        if donors != expected_donors or not (0 <= available <= donors) or fallback != donors - available or not (0 <= changed <= donors):
            raise BridgeContractError(f"e09b_challenge_movement_accounting:{task}")
        if not synthetic and not (35 <= available <= 40):
            raise BridgeContractError(f"e09b_real_rna_coverage_unexpected:{task}:{available}")
        if not math.isfinite(float(movement.get("rank_spearman_vs_base", float("nan")))):
            raise BridgeContractError(f"e09b_challenge_rank_agreement_missing:{task}")
        if float(movement.get("max_absolute_score_correction", 999)) > 0.100000000001:
            raise BridgeContractError(f"e09b_correction_cap_broken:{task}")
        decision = payload.get("decision") or {}
        mean_delta = float(payload["study_equal_delta_vs_base"]); min_delta = float(payload["minimum_study_delta_vs_base"]); candidate = float(payload["study_equal_candidate_spearman"]); shuffle = float(payload["study_equal_shuffle_spearman"])
        expected_local = bool(mean_delta >= 0.02 and min_delta >= -0.05 and candidate >= shuffle)
        expected_competition = bool(expected_local and changed > 0)
        if decision.get("local_promotion_gate_passed") is not expected_local or decision.get("competition_candidate") is not expected_competition:
            raise BridgeContractError(f"e09b_decision_boolean_mismatch:{task}")
        if decision.get("public_probe_authorized") is not False or decision.get("incumbent_changed") is not False:
            raise BridgeContractError(f"e09b_probe_incumbent_boundary:{task}")
    if not synthetic:
        rna = result.get("rna_audit") or {}; public = rna.get("public") or {}; challenge = rna.get("challenge") or {}
        if set(public) != {"2020_UGA", "2024_UGA"}:
            raise BridgeContractError("e09b_public_rna_audit_set")
        for study, audit in public.items():
            if audit.get("study") != study or audit.get("raw_tpm_only") is not True or audit.get("day0_only") is not True or audit.get("material_category") != "PBMC" or int(audit.get("eligible_participants", 0)) < 24:
                raise BridgeContractError(f"e09b_public_rna_audit:{study}")
        if challenge.get("study") != "2025LJI" or challenge.get("raw_tpm_only") is not True or challenge.get("day0_only") is not True or int(challenge.get("eligible_participants", 0)) < 35:
            raise BridgeContractError("e09b_challenge_rna_audit")
    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"','SYN_E09B_')
    if any(token in serialized for token in banned):
        raise BridgeContractError("e09b_aggregate_privacy_contract")'''
    runtime = v3.replace_top_level_function(runtime, "validate_result", validate_replacement)

    summary_replacement = '''def render_summary(result: dict) -> str:
    lines = [
        "# CMI-Flu strategy E09b HIPC9 RNA residual", "",
        "Literature-frozen Day-0 RNA residual against B2.1; aggregate-only; no Competition submission or Public probe.", "",
        f"- science commit: `{SCIENCE_COMMIT}`",
        f"- E09b blob: `{E09B_BLOB}`",
        "",
    ]
    for task, payload in (result.get("tasks") or {}).items():
        movement = payload.get("challenge_movement") or {}; decision = payload.get("decision") or {}
        lines.extend([
            f"## {task}",
            f"- study-equal base: `{payload.get('study_equal_base_spearman')}`",
            f"- study-equal candidate: `{payload.get('study_equal_candidate_spearman')}`",
            f"- delta: `{payload.get('study_equal_delta_vs_base')}`",
            f"- minimum study delta: `{payload.get('minimum_study_delta_vs_base')}`",
            f"- shuffle: `{payload.get('study_equal_shuffle_spearman')}`",
            f"- Challenge RNA/fallback: `{movement.get('rna_available_donors')}/{movement.get('frozen_base_fallback_donors')}`",
            f"- Challenge rank agreement: `{movement.get('rank_spearman_vs_base')}`; changed ranks=`{movement.get('changed_rank_count_vs_base')}`",
            f"- local gate: `{decision.get('local_promotion_gate_passed')}`; competition candidate: `{decision.get('competition_candidate')}`",
            "",
        ])
    lines.append("Historical HAI panels remain incomplete proxies rather than official complete-2025-panel CV.")
    return "\\n".join(lines) + "\\n"'''
    runtime = v3.replace_top_level_function(runtime, "render_summary", summary_replacement)

    runtime = replace_block(
        runtime,
        '        stage = "load_e05"\n',
        '        stage = "write_outputs"\n',
        '''        stage = "load_e09b"
        run_e09b, run_synthetic, e09b_module = load_e09b_modules()
        stage = "run_e09b"
        result = json_safe(dict(run_e09b(config, inputs, data_dir=input_dir)))
        stage = "validate_e09b"
        validate_result(result, synthetic=False)''',
    )

    provenance = '            "strategy_e05_v4_blob_sha": E05_V4_BLOB,\n'
    if runtime.count(provenance) != 1:
        raise SystemExit("E09b bridge provenance anchor changed")
    runtime = runtime.replace(
        provenance,
        provenance
        + '            "strategy_e09a_blob_sha": E09A_BLOB,\n'
        + '            "strategy_e09b_blob_sha": E09B_BLOB,\n'
        + '            "strategy_e09b_synthetic_blob_sha": E09B_SYNTH_BLOB,\n'
        + '            "strategy_e09b_mapping_blob_sha": E09B_MAPPING_BLOB,\n',
        1,
    )

    main_replacement = '''def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.synthetic:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        runtime_root = Path("/tmp") / "cmi-flu-e09b-synthetic-runtime"
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        runtime_root.mkdir(parents=True)
        package_path = runtime_root / "cmi_flu_bundle.zip"
        package_path.write_bytes(package_bytes())
        sys.path.insert(0, str(package_path))
        try:
            run_e09b, run_synthetic, e09b = load_e09b_modules()
            tasks = {task: json_safe(dict(run_synthetic(task))) for task in ("Task2.1", "Task2.2")}
            result = {
                "schema_version": 1,
                "experiment": "strategy_v2_e09b_hipc9_rna_residual",
                "tasks": tasks,
                "feature_contract": {
                    "source_doi": e09b.SOURCE_DOI, "source_pmid": e09b.SOURCE_PMID,
                    "gene_symbols": [symbol for symbol, _ in e09b.HIPC9],
                    "ensembl_ids": [ensembl for _, ensembl in e09b.HIPC9],
                    "validation_subset": list(e09b.HIPC7_VALIDATION),
                    "day0_only": True, "raw_tpm_only": True, "within_study_gene_rank": True,
                    "minimum_gene_coverage": e09b.MIN_GENE_COVERAGE, "ridge_alpha": e09b.RIDGE_ALPHA,
                    "residual_shrinkage": e09b.RESIDUAL_SHRINKAGE, "max_score_correction": e09b.MAX_SCORE_CORRECTION,
                    "minimum_fold_subjects": e09b.MIN_FOLD_SUBJECTS, "outcome_based_gene_selection": False,
                    "batch_corrected_challenge_expression_used": False,
                },
                "rna_audit": {"synthetic": True},
                "evaluation_contract": {
                    "qualifying_studies": list(e09b.QUALIFYING_STUDIES),
                    "historical_panels_are_incomplete_proxies": True,
                    "official_complete_2025_panel_cv_claimed": False,
                    "promotion_mean_delta": e09b.PROMOTION_MEAN_DELTA,
                    "promotion_min_study_delta": e09b.PROMOTION_MIN_STUDY_DELTA,
                    "shuffle_seed": e09b.SHUFFLE_SEED,
                    "challenge_missing_rna_policy": "exact_frozen_b21_fallback",
                },
                "competition_submission_attempted": False,
                "public_leaderboard_used_for_selection": False,
                "automatic_compute_retries": 0,
                "incumbent_changed": False,
            }
            result = json_safe(result)
            validate_result(result, synthetic=True)
            metrics_path = output_dir / "metrics.json"; summary_path = output_dir / "summary.md"; bridge_path = output_dir / "bridge-result.json"
            metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\\n", encoding="utf-8")
            summary_path.write_text(render_summary(result), encoding="utf-8")
            bridge = {
                "schema_version": 1, "request_id": REQUEST_ID, "competition": COMPETITION,
                "target_kernel": TARGET_KERNEL, "science_commit": SCIENCE_COMMIT,
                "strategy_e09a_blob_sha": E09A_BLOB, "strategy_e09b_blob_sha": E09B_BLOB,
                "strategy_e09b_synthetic_blob_sha": E09B_SYNTH_BLOB, "strategy_e09b_mapping_blob_sha": E09B_MAPPING_BLOB,
                "metrics_sha256": sha256_file(metrics_path), "summary_sha256": sha256_file(summary_path),
                "competition_submission_attempted": False, "leaderboard_used_for_selection": False,
                "contains_participant_identifiers": False, "contains_row_level_predictions": False, "synthetic": True,
            }
            bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
            print("CMI_FLU_E09B_COMPLETE synthetic=true tasks=2 submission=false")
            return 0
        finally:
            shutil.rmtree(runtime_root, ignore_errors=True)
    try:
        input_dir = locate_competition_data(args.input_dir)
    except Exception as exc:
        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode()).hexdigest()[:20]
        print(f"CMI_FLU_E09B_FAILED stage=locate_competition_data exception_type={type(exc).__name__} error_code={code}", file=sys.stderr)
        return 2
    return execute(input_dir, args.output_dir)'''
    runtime = v3.replace_top_level_function(runtime, "main", main_replacement)

    runtime = runtime.replace("CMI-Flu strategy E05 HAI donor x strain. Aggregate outputs only; no submission.", "CMI-Flu strategy E09b HIPC9 RNA residual. Aggregate outputs only; no submission.")
    runtime = runtime.replace("CMI_FLU_E05_FAILED", "CMI_FLU_E09B_FAILED").replace("CMI_FLU_E05_COMPLETE", "CMI_FLU_E09B_COMPLETE")
    names = v3.top_level_functions(runtime)
    for name in ("parse_args", "self_test", "load_e05_module", "load_e09b_modules", "validate_result", "render_summary", "execute", "main"):
        if names.count(name) != 1:
            raise SystemExit(f"E09b top-level binding contract failed:{name}:{names.count(name)}")
    for old in (OLD_REQUEST_ID, OLD_TARGET_KERNEL, OLD_SCIENCE_COMMIT):
        if old in runtime:
            raise SystemExit(f"E09b prior identity remained:{old}")
    if "kaggle competitions submit" in runtime or "competition_submit" in runtime:
        raise SystemExit("E09b generated runtime contains submission path")
    compile(runtime, "generated_e09b_runtime.py", "exec")
    return runtime


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    validate_request(root)
    e09b, synth, mapping, e09a = load_exact_sources(root)
    v3, parent_runtime = build_parent_runtime(root, reference_dir)
    runtime = patch_runtime(v3, parent_runtime, e09b, synth, mapping, e09a)
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E09B_PREPARE_PASS "
        f"science_commit={SCIENCE_COMMIT} request_id={REQUEST_ID} target={TARGET_KERNEL} "
        f"e09b_blob={E09B_BLOB} runtime_sha256={sha256(runtime.encode())} "
        "e05_v6_ancestry=true hipc9_frozen=true submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
