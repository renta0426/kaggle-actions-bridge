#!/usr/bin/env python3
"""Build the exact aggregate-only E09a module/teacher applicability runtime."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REQUEST_ID = "20260909-cmi-flu-strategy-e09a-module-teacher-audit-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-e09a-module-teacher-audit-20260909-001"
SCIENCE_COMMIT = "537cc7fbcf668d8729010c8be304c4b1bb9b63a5"
REQUEST_PATH = "requests/cmi-flu-strategy-e09a-module-teacher-audit-001.json"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e09a-module-teacher-audit-001"
E09A_PATH = f"{PAYLOAD_ROOT}/strategy_e09a.py"
E09A_SYNTH_PATH = f"{PAYLOAD_ROOT}/strategy_e09a_synthetic.py"
E09A_BLOB = "37270a0a8aafc070d91b5f43fbea3ba169cd61b0"
E09A_SYNTH_BLOB = "231a3739fd6015c61d389cce12ffa1a7aefb3f38"
ALIASES_PATH = "payloads/cmi-flu-strategy-e07-task14-formal-closure-001/aliases.py"
CONTRACTS_PATH = "payloads/cmi-flu-strategy-e07-task14-formal-closure-001/contracts.py"
ALIASES_BLOB = "5b9930c8e26f2b462ee8ce64fdfbdb76df2d1af3"
CONTRACTS_BLOB = "6e2791e526255d9c534e71c9c0886f0c300df7bd"
E04B_PATH = "payloads/cmi-flu-strategy-e04b-task13-material-bridge-001/strategy_e04b.py"
E04_PATH = "payloads/cmi-flu-strategy-e04-task13-rescue-001/strategy_e04.py"
E04_CONTRACT_PATH = "payloads/cmi-flu-strategy-e04-task13-rescue-001/strategy_e04_contract.py"
TASK13_PATH = "payloads/cmi-flu-strategy-e04-task13-rescue-001/task13_harmonization.py"
E04B_BLOB = "72f936a7248dfa17338932077d17fb43123cb056"
E04_BLOB = "73c112a8bb3b1ee7a6dd5bfbd5645a945bea6ebb"
E04_CONTRACT_BLOB = "3982541febfb4641fbf895438ae1a465bdbb3d5e"
TASK13_BLOB = "5c6725dc757a5ba9dd21289b1c4f09997e1afdb8"
BASE_PREPARE = "scripts/cmi_flu_strategy_e04b_prepare_v2.py"
OLD_REQUEST_ID = "20260909-cmi-flu-strategy-e04b-task13-material-bridge-001"
OLD_TARGET = "renta0426/cmi-flu-e04b-task13-material-bridge-20260909-001"
OLD_SCIENCE = "cc51692a66055ed1716fd6d66ac574c4a22f0dcc"

PINNED = {
    "aliases.py": ALIASES_BLOB,
    "contracts.py": CONTRACTS_BLOB,
    "strategy_e04.py": E04_BLOB,
    "strategy_e04_contract.py": E04_CONTRACT_BLOB,
    "strategy_e04b.py": E04B_BLOB,
    "task13_harmonization.py": TASK13_BLOB,
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_blob(path: Path, expected: str, label: str) -> str:
    data = path.read_bytes()
    found = git_blob_sha(data)
    if found != expected:
        raise SystemExit(f"E09a exact relay mismatch:{label}:{found}")
    text = data.decode("utf-8")
    compile(text, label, "exec")
    return text


def validate_request(root: Path) -> None:
    request = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": COMPETITION,
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_repository": "renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": "agent_relay_exact_blobs_plus_frozen_e04b_runtime",
        "strategy_e09a_blob_sha": E09A_BLOB,
        "strategy_e09a_synthetic_blob_sha": E09A_SYNTH_BLOB,
        "expected_kernel_version": 1,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"E09a request mismatch:{key}")
    if request.get("pinned_dependency_blobs") != PINNED:
        raise SystemExit("E09a dependency blob contract mismatch")
    if request.get("resource") != {"accelerator": "cpu", "expected_runtime_minutes": 90, "hard_timeout_minutes": 150, "max_active_runs": 1}:
        raise SystemExit("E09a resource contract mismatch")
    if request.get("api_budget") != {"max_calls": 100, "poll_interval_seconds": 120, "max_polls": 80, "max_pages": 2}:
        raise SystemExit("E09a API budget mismatch")
    if request.get("allowed_output_paths") != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E09a output allowlist mismatch")
    contract = request.get("experiment_contract") or {}
    locked = {
        "experiment": "strategy_v2_e09a_module_teacher_applicability_audit",
        "audit_type": "outcome_independent_presence_schema_and_measurement_compatibility",
        "direct_tasks": ["Task1.3", "Task2.1", "Task2.2", "Task2.3"],
        "minimum_direct_studies": 2,
        "minimum_subjects_per_study": 8,
        "minimum_total_subjects": 24,
        "minimum_common_genes": 1000,
        "challenge_expected_donors": 40,
        "historical_serology_outcome_values_read": False,
        "model_fitting_authorized": False,
        "module_gene_sets_frozen": False,
        "batch_corrected_challenge_only_form_selected": False,
        "missing_view_policy": "fallback_to_frozen_base",
        "public_probe_authorized": False,
    }
    for key, value in locked.items():
        if contract.get(key) != value:
            raise SystemExit(f"E09a experiment contract mismatch:{key}")


def replace_top_level_function(text: str, name: str, replacement: str) -> str:
    tree = ast.parse(text)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(nodes) != 1:
        raise SystemExit(f"E09a top-level function contract changed:{name}:{len(nodes)}")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1])
    end = sum(len(line) for line in lines[: node.end_lineno])
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


def top_level_functions(text: str) -> list[str]:
    return [node.name for node in ast.parse(text).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def build_base_runtime(root: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="cmi-e09a-base-") as tmp:
        target = Path(tmp) / "e04b-runtime.py"
        subprocess.run(
            [sys.executable, str(root / BASE_PREPARE), "--repository-root", str(root), "--output", str(target)],
            check=True,
        )
        return target.read_text(encoding="utf-8")


def patch_runtime(
    runtime: str,
    *,
    e09a: str,
    e09a_synth: str,
    aliases: str,
    contracts: str,
) -> str:
    for old, new in ((OLD_REQUEST_ID, REQUEST_ID), (OLD_TARGET, TARGET_KERNEL), (OLD_SCIENCE, SCIENCE_COMMIT)):
        if runtime.count(old) < 1:
            raise SystemExit(f"E09a identity anchor missing:{old}")
        runtime = runtime.replace(old, new)

    marker = "E04B_SYNTH_BLOB = '6f3c081ea47a0562150398ab307ab1a6b4d4c000'\n"
    if runtime.count(marker) != 1:
        raise SystemExit("E09a source injection anchor changed")
    injection = (
        f"E09A_BLOB = {E09A_BLOB!r}\n"
        f"E09A_SYNTH_BLOB = {E09A_SYNTH_BLOB!r}\n"
        f"ALIASES_BLOB = {ALIASES_BLOB!r}\n"
        f"CONTRACTS_BLOB = {CONTRACTS_BLOB!r}\n"
        f"E09A_SOURCE = {e09a!r}\n"
        f"E09A_SYNTH_SOURCE = {e09a_synth!r}\n"
        f"ALIASES_SOURCE = {aliases!r}\n"
        f"CONTRACTS_SOURCE = {contracts!r}\n"
    )
    runtime = runtime.replace(marker, marker + injection, 1)

    load_replacement = '''def load_e09a_modules() -> tuple[object, object]:
    def install(name: str, source: str) -> object:
        module = types.ModuleType(name)
        module.__file__ = f"<{name}>"
        module.__package__ = "cmi_flu"
        sys.modules[name] = module
        exec(compile(source, name.replace(".", "/") + ".py", "exec"), module.__dict__, module.__dict__)
        return module
    install("cmi_flu.contracts", CONTRACTS_SOURCE)
    install("cmi_flu.aliases", ALIASES_SOURCE)
    install("cmi_flu.task13_harmonization", TASK13_SOURCE)
    install("cmi_flu.strategy_e04_contract", CONTRACT_SOURCE)
    install("cmi_flu.strategy_e04", E04_SOURCE)
    install("cmi_flu.strategy_e04_synthetic", SYNTHETIC_SOURCE)
    install("cmi_flu.strategy_e04b", E04B_SOURCE)
    e09a = install("cmi_flu.strategy_e09a", E09A_SOURCE)
    synthetic = install("cmi_flu.strategy_e09a_synthetic", E09A_SYNTH_SOURCE)
    run_real = getattr(e09a, "run_e09a_from_paths", None)
    run_synthetic = getattr(synthetic, "run_synthetic", None)
    if not callable(run_real) or not callable(run_synthetic):
        raise BridgeContractError("e09a_entry_missing")
    return run_real, run_synthetic'''
    runtime = replace_top_level_function(runtime, "load_e04_modules", load_replacement)

    self_test_replacement = '''def self_test() -> int:
    package = package_bytes()
    sources = (
        (E09A_SOURCE, E09A_BLOB, "e09a"),
        (E09A_SYNTH_SOURCE, E09A_SYNTH_BLOB, "e09a_synthetic"),
        (ALIASES_SOURCE, ALIASES_BLOB, "aliases"),
        (CONTRACTS_SOURCE, CONTRACTS_BLOB, "contracts"),
        (E04B_SOURCE, E04B_BLOB, "e04b"),
        (E04_SOURCE, E04_BLOB, "e04"),
        (CONTRACT_SOURCE, CONTRACT_BLOB, "e04_contract"),
        (TASK13_SOURCE, TASK13_BLOB, "task13"),
    )
    for source, expected, label in sources:
        if git_blob_sha(source.encode("utf-8")) != expected:
            raise BridgeContractError(f"{label}_blob_mismatch")
        compile(source, f"cmi_flu/{label}.py", "exec")
    print(
        "CMI_FLU_E09A_RUNTIME_SELF_TEST PASS "
        f"request_id={REQUEST_ID} package_bytes={len(package)} package_sha256={PACKAGE_SHA256} "
        f"science_commit={SCIENCE_COMMIT} e09a_blob={E09A_BLOB}"
    )
    return 0'''
    runtime = replace_top_level_function(runtime, "self_test", self_test_replacement)

    validate_replacement = '''def validate_result(result: dict, *, synthetic: bool) -> None:
    if result.get("schema_version") != 1 or result.get("experiment") != "strategy_v2_e09a_module_teacher_applicability_audit":
        raise BridgeContractError("e09a_experiment_identity_mismatch")
    if result.get("audit_type") != "outcome_independent_presence_schema_and_measurement_compatibility":
        raise BridgeContractError("e09a_audit_type_mismatch")
    if result.get("outcome_values_accessed") is not False or result.get("public_leaderboard_used_for_selection") is not False or result.get("competition_submission_attempted") is not False:
        raise BridgeContractError("e09a_outcome_selection_submission_boundary")
    decision = result.get("decision") or {}
    expected_decision = {
        "module_gene_sets_frozen": False,
        "model_fitting_authorized_by_e09a": False,
        "batch_corrected_challenge_only_form_selected": False,
        "missing_view_policy": "fallback_to_frozen_base",
    }
    for key, value in expected_decision.items():
        if decision.get(key) != value:
            raise BridgeContractError(f"e09a_decision_boundary:{key}")
    if result.get("manifest_entries_present_for_audited_files") is not True:
        raise BridgeContractError("e09a_manifest_coverage_missing")
    challenge = result.get("challenge_rna") or {}
    raw = challenge.get("raw") or {}; corrected = challenge.get("batch_corrected") or {}
    if int(challenge.get("expected_donors", -1)) != 40:
        raise BridgeContractError("e09a_expected_donor_contract")
    if not synthetic and int(raw.get("all_participants", -1)) != 40:
        raise BridgeContractError("e09a_challenge_raw_participant_count")
    if raw.get("value_column") != "tpm" or int(raw.get("negative_values", -1)) != 0 or int(raw.get("numeric_parse_failures", -1)) != 0:
        raise BridgeContractError("e09a_raw_rna_value_contract")
    if corrected.get("value_column") != "batch_corrected_expression" or int(corrected.get("numeric_parse_failures", -1)) != 0:
        raise BridgeContractError("e09a_corrected_rna_value_contract")
    if challenge.get("batch_corrected_has_negative_values") is not True or challenge.get("public_batch_corrected_counterpart_available") is not False or challenge.get("batch_corrected_form_admissible_for_cross_study_module_training") is not False:
        raise BridgeContractError("e09a_batch_corrected_boundary")
    if int(challenge.get("raw_day0_missing_donors_require_frozen_base_fallback", -1)) != 40 - int(challenge.get("raw_day0_donors", -999)):
        raise BridgeContractError("e09a_missing_view_accounting")
    tasks = result.get("task_readiness") or {}
    if set(tasks) != {"Task1.3", "Task2.1", "Task2.2", "Task2.3"}:
        raise BridgeContractError("e09a_task_set_mismatch")
    for task, payload in tasks.items():
        if payload.get("task") != task or not isinstance(payload.get("direct_rna_teacher_data_ready"), bool):
            raise BridgeContractError(f"e09a_task_readiness_type:{task}")
        if int(payload.get("studies_meeting_per_study_gate", -1)) < 0 or int(payload.get("total_material_compatible_paired_subjects", -1)) < 0:
            raise BridgeContractError(f"e09a_task_readiness_count:{task}")
        studies = payload.get("source_studies") or []
        if len(studies) != (3 if synthetic else 6):
            raise BridgeContractError(f"e09a_task_source_study_count:{task}")
        for item in studies:
            if not isinstance(item.get("meets_per_study_data_gate"), bool):
                raise BridgeContractError(f"e09a_study_gate_type:{task}")
        if task == "Task1.3":
            if payload.get("official_strict_gate_teacher_required") is not True or payload.get("proxy_teacher_promoted_to_strict") is not False:
                raise BridgeContractError("e09a_task13_teacher_boundary")
        else:
            rule = payload.get("readiness_rule") or {}
            if rule != {"minimum_studies": 2, "minimum_subjects_per_study": 8, "minimum_total_subjects": 24, "minimum_common_genes": 1000}:
                raise BridgeContractError(f"e09a_readiness_rule:{task}")
            if payload.get("official_complete_panel_cv_claimed") is not False:
                raise BridgeContractError(f"e09a_official_cv_claim:{task}")
    repertoire = result.get("repertoire_applicability") or {}
    if repertoire.get("single_cell_vdj_external_teacher_available") is not False or repertoire.get("generic_clonality_promoted_to_antigen_specific_teacher") is not False:
        raise BridgeContractError("e09a_repertoire_boundary")
    if synthetic:
        if decision.get("e09b_hai_module_data_gate_ready") is not True or decision.get("task13_rna_module_data_gate_ready") is not False:
            raise BridgeContractError("e09a_synthetic_decision_mismatch")
    else:
        public_rna = result.get("public_rna") or {}
        if set(public_rna) != {"2019_UGA", "2020_UGA", "2024_UGA", "SDY224", "SDY2867", "SDY2941"}:
            raise BridgeContractError("e09a_public_rna_study_set")
        bridge = result.get("task13_measurement_bridge") or {}
        if bridge.get("ontology_version") != "e04b_task13_material_plural_v1" or bridge.get("strict_gate_reconstructed") is not False:
            raise BridgeContractError("e09a_task13_measurement_bridge")
    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"', '"subject_group"', '"row_index"', '"oof_predictions"', '"challenge_predictions"', '"cdr3"', '"sequence_id"')
    if any(token in serialized for token in banned):
        raise BridgeContractError("e09a_aggregate_privacy_contract")'''
    runtime = replace_top_level_function(runtime, "validate_result", validate_replacement)

    summary_replacement = '''def render_summary(result: dict) -> str:
    decision = result.get("decision") or {}
    challenge = result.get("challenge_rna") or {}
    lines = [
        "# CMI-Flu strategy E09a module/teacher applicability audit", "",
        "Outcome-independent aggregate-only data gate; no model fitting or Competition submission was attempted.", "",
        f"- science commit: `{SCIENCE_COMMIT}`",
        f"- raw Challenge Day0 donors: `{challenge.get('raw_day0_donors')}` / `{challenge.get('expected_donors')}`",
        f"- raw Challenge negative-baseline donors: `{challenge.get('raw_negative_baseline_donors')}`",
        f"- corrected Day0 donors: `{challenge.get('batch_corrected_day0_donors')}`",
        f"- E09b HAI data gate ready: `{decision.get('e09b_hai_module_data_gate_ready')}`",
        f"- Task1.3 RNA data gate ready: `{decision.get('task13_rna_module_data_gate_ready')}`", "",
    ]
    for task, payload in (result.get("task_readiness") or {}).items():
        lines.append(
            f"- {task}: ready={payload.get('direct_rna_teacher_data_ready')} "
            f"ready_studies={payload.get('studies_meeting_per_study_gate')} "
            f"paired={payload.get('total_material_compatible_paired_subjects')}"
        )
    rep = result.get("repertoire_applicability") or {}
    lines.append(f"- repertoire direct-teacher ready: `{rep.get('cross_study_direct_repertoire_teacher_ready')}`")
    return "\\n".join(lines) + "\\n"'''
    runtime = replace_top_level_function(runtime, "render_summary", summary_replacement)

    execute_replacement = '''def execute(input_dir: Path | None, output_dir: Path, *, synthetic: bool = False) -> int:
    runtime_root = (Path("/tmp") / "cmi-flu-e09a-runtime") if Path("/tmp").is_dir() else (output_dir / "e09a-runtime")
    stage = "initialize"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("bridge-result.json", "metrics.json", "summary.md"):
            if (output_dir / name).exists():
                raise BridgeContractError(f"e09a_preexisting_output:{name}")
        if runtime_root.exists():
            raise BridgeContractError("e09a_runtime_scratch_preexists")
        runtime_root.mkdir(parents=True)
        stage = "materialize_package"
        package_path = runtime_root / "cmi_flu_bundle.zip"
        package_path.write_bytes(package_bytes())
        sys.path.insert(0, str(package_path))
        stage = "load_e09a"
        run_real, run_synthetic = load_e09a_modules()
        if synthetic:
            stage = "run_synthetic"
            result = run_synthetic()
        else:
            if input_dir is None:
                raise BridgeContractError("e09a_production_input_missing")
            stage = "derive_panels"
            external = runtime_root / "external"
            vaccine_n, challenge_n = derive_reference_files(input_dir, external)
            if (vaccine_n, challenge_n) != (3, 12):
                raise BridgeContractError("e09a_panel_count_mismatch")
            ref = external / "google-drive" / "challenge-resources" / "reference_files"
            vaccine_strains = tuple(line.strip() for line in (ref / "vaccine_strains_2025.txt").read_text(encoding="utf-8").splitlines() if line.strip())
            challenge_strains = tuple(line.strip() for line in (ref / "all_challenge_virus_strains.txt").read_text(encoding="utf-8").splitlines() if line.strip())
            stage = "run_e09a"
            result = run_real(input_dir, vaccine_strains=vaccine_strains, challenge_strains=challenge_strains)
        result = json_safe(dict(result))
        stage = "validate_e09a"
        validate_result(result, synthetic=synthetic)
        stage = "write_outputs"
        metrics_path = output_dir / "metrics.json"
        summary_path = output_dir / "summary.md"
        bridge_path = output_dir / "bridge-result.json"
        metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\\n", encoding="utf-8")
        summary_path.write_text(render_summary(result), encoding="utf-8")
        bridge = {
            "schema_version": 1,
            "request_id": REQUEST_ID,
            "competition": COMPETITION,
            "target_kernel": TARGET_KERNEL,
            "science_commit": SCIENCE_COMMIT,
            "strategy_e09a_blob_sha": E09A_BLOB,
            "strategy_e09a_synthetic_blob_sha": E09A_SYNTH_BLOB,
            "aliases_blob_sha": ALIASES_BLOB,
            "contracts_blob_sha": CONTRACTS_BLOB,
            "strategy_e04b_blob_sha": E04B_BLOB,
            "strategy_e04_blob_sha": E04_BLOB,
            "strategy_e04_contract_blob_sha": CONTRACT_BLOB,
            "task13_harmonization_blob_sha": TASK13_BLOB,
            "package_sha256": PACKAGE_SHA256,
            "python_version": platform.python_version(),
            "metrics_sha256": sha256_file(metrics_path),
            "summary_sha256": sha256_file(summary_path),
            "competition_submission_attempted": False,
            "leaderboard_used_for_selection": False,
            "contains_participant_identifiers": False,
            "contains_row_level_predictions": False,
            "synthetic": bool(synthetic),
        }
        bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        shutil.rmtree(runtime_root, ignore_errors=True)
        print(
            "CMI_FLU_E09A_COMPLETE "
            f"synthetic={synthetic} hai_ready={(result.get('decision') or {}).get('e09b_hai_module_data_gate_ready')} "
            "outcomes=false submission=false"
        )
        return 0
    except Exception as exc:
        shutil.rmtree(runtime_root, ignore_errors=True)
        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode("utf-8", errors="replace")).hexdigest()[:20]
        detail = f" detail={str(exc)}" if synthetic else ""
        print(f"CMI_FLU_E09A_FAILED stage={stage} exception_type={type(exc).__name__} error_code={code}{detail}", file=sys.stderr)
        return 2'''
    runtime = replace_top_level_function(runtime, "execute", execute_replacement)

    runtime = runtime.replace("CMI_FLU_E04B_FAILED", "CMI_FLU_E09A_FAILED")
    runtime = runtime.replace("CMI_FLU_E04B_COMPLETE", "CMI_FLU_E09A_COMPLETE")
    runtime = runtime.replace("CMI_FLU_E04B_RUNTIME_SELF_TEST", "CMI_FLU_E09A_RUNTIME_SELF_TEST")

    names = top_level_functions(runtime)
    required = ("parse_args", "package_bytes", "self_test", "locate_competition_data", "derive_reference_files", "load_e09a_modules", "validate_result", "render_summary", "execute", "main")
    for name in required:
        if names.count(name) != 1:
            raise SystemExit(f"E09a final top-level contract:{name}:{names.count(name)}")
    if "load_e04_modules" in names:
        raise SystemExit("E09a retained obsolete E04 loader")
    for old in (OLD_REQUEST_ID, OLD_TARGET, OLD_SCIENCE):
        if old in runtime:
            raise SystemExit(f"E09a old identity remained:{old}")
    if "kaggle competitions submit" in runtime or "competition_submit(" in runtime:
        raise SystemExit("E09a runtime contains submission path")
    if "/kaggle/working/e09a-runtime" in runtime:
        raise SystemExit("E09a runtime scratch would be saved as output")
    compile(runtime, "generated_e09a_runtime.py", "exec")
    return runtime


def main() -> int:
    a = args()
    root = a.repository_root.expanduser().resolve()
    output = a.output.expanduser().resolve()
    validate_request(root)
    e09a = require_blob(root / E09A_PATH, E09A_BLOB, "strategy_e09a.py")
    e09a_synth = require_blob(root / E09A_SYNTH_PATH, E09A_SYNTH_BLOB, "strategy_e09a_synthetic.py")
    aliases = require_blob(root / ALIASES_PATH, ALIASES_BLOB, "aliases.py")
    contracts = require_blob(root / CONTRACTS_PATH, CONTRACTS_BLOB, "contracts.py")
    require_blob(root / E04B_PATH, E04B_BLOB, "strategy_e04b.py")
    require_blob(root / E04_PATH, E04_BLOB, "strategy_e04.py")
    require_blob(root / E04_CONTRACT_PATH, E04_CONTRACT_BLOB, "strategy_e04_contract.py")
    require_blob(root / TASK13_PATH, TASK13_BLOB, "task13_harmonization.py")
    runtime = patch_runtime(build_base_runtime(root), e09a=e09a, e09a_synth=e09a_synth, aliases=aliases, contracts=contracts)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_E09A_PREPARE_PASS "
        f"science_commit={SCIENCE_COMMIT} e09a_blob={E09A_BLOB} runtime_sha256={sha256(runtime.encode())} "
        "scratch=/tmp outcomes=false submission=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
