#!/usr/bin/env python3
"""Build E04b from the proven E04 v2 runtime plus exact E04b wrapper blobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REQUEST_ID = "20260909-cmi-flu-strategy-e04b-task13-material-bridge-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-e04b-task13-material-bridge-20260909-001"
SCIENCE_COMMIT = "cc51692a66055ed1716fd6d66ac574c4a22f0dcc"
REQUEST_PATH = "requests/cmi-flu-strategy-e04b-task13-material-bridge-001.json"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e04b-task13-material-bridge-001"
E04B_PATH = f"{PAYLOAD_ROOT}/strategy_e04b.py"
E04B_SYNTH_PATH = f"{PAYLOAD_ROOT}/strategy_e04b_synthetic.py"
E04B_BLOB = "72f936a7248dfa17338932077d17fb43123cb056"
E04B_SYNTH_BLOB = "6f3c081ea47a0562150398ab307ab1a6b4d4c000"
E04_BLOB = "73c112a8bb3b1ee7a6dd5bfbd5645a945bea6ebb"
CONTRACT_BLOB = "3982541febfb4641fbf895438ae1a465bdbb3d5e"
E04_SYNTHETIC_BLOB = "436b6a971622915cc5b335060f9a850feb6108fd"
TASK13_BLOB = "5c6725dc757a5ba9dd21289b1c4f09997e1afdb8"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
BASE_PREPARE = "scripts/cmi_flu_strategy_e04_prepare_v2.py"

PINNED_BLOBS = {
    "src/cmi_flu/strategy_e04b.py": E04B_BLOB,
    "src/cmi_flu/strategy_e04b_synthetic.py": E04B_SYNTH_BLOB,
    "src/cmi_flu/strategy_e04.py": E04_BLOB,
    "src/cmi_flu/strategy_e04_contract.py": CONTRACT_BLOB,
    "src/cmi_flu/strategy_e04_synthetic.py": E04_SYNTHETIC_BLOB,
    "src/cmi_flu/task13_harmonization.py": TASK13_BLOB,
    "configs/baseline_b021_robust.yaml": CONFIG_BLOB,
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
        raise SystemExit(f"{label} relay blob mismatch:{found}")
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
        "science_transport": "agent_relay_exact_blobs_plus_frozen_e04_runtime",
        "expected_kernel_version": 1,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"E04b request mismatch:{key}")
    if request.get("pinned_git_blobs") != PINNED_BLOBS:
        raise SystemExit("E04b pinned blob contract mismatch")
    if request.get("resource") != {"accelerator": "cpu", "expected_runtime_minutes": 30, "hard_timeout_minutes": 60, "max_active_runs": 1}:
        raise SystemExit("E04b resource contract mismatch")
    if request.get("allowed_output_paths") != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E04b output allowlist mismatch")
    contract = request.get("experiment_contract") or {}
    locked = {
        "experiment": "strategy_v2_e04b_task13_material_plural_bridge",
        "parent_experiment": "strategy_v2_e04_task13_anchor_preserving_rescue",
        "ontology_version": "e04b_task13_material_plural_v1",
        "only_scientific_change": "PBMCs_to_PBMC_on_named_strict_ASC_2024UGA_2025LJI_only",
        "expected_baseline_compatible_2024UGA": 33,
        "expected_challenge_compatible_2025LJI": 40,
        "expected_supervised_strict_source_subjects": 23,
        "expected_challenge_correction_rows": 40,
        "residual_alpha": 10.0,
        "residual_shrinkage": 0.25,
        "residual_score_cap": 0.05,
        "expected_strict_delta_vs_anchor": -0.01026600470536132,
        "public_probe_authorized": False,
    }
    for key, value in locked.items():
        if contract.get(key) != value:
            raise SystemExit(f"E04b experiment contract mismatch:{key}")


def replace_block(text: str, start: str, end: str, replacement: str) -> str:
    i = text.find(start)
    if i < 0:
        raise SystemExit(f"E04b runtime patch start missing:{start[:50]}")
    j = text.find(end, i + len(start))
    if j < 0:
        raise SystemExit(f"E04b runtime patch end missing:{end[:50]}")
    return text[:i] + replacement.rstrip() + "\n\n" + text[j:]


def build_runtime(root: Path, e04b: str, e04b_synth: str) -> tuple[str, str]:
    with tempfile.TemporaryDirectory(prefix="cmi-e04b-base-") as tmp:
        base_runtime = Path(tmp) / "e04_runtime.py"
        subprocess.run(
            [sys.executable, str(root / BASE_PREPARE), "--repository-root", str(root), "--output", str(base_runtime)],
            check=True,
        )
        runtime = base_runtime.read_text(encoding="utf-8")

    replacements = {
        'REQUEST_ID = "20260908-cmi-flu-strategy-e04-task13-rescue-001"': f'REQUEST_ID = "{REQUEST_ID}"',
        'SCIENCE_COMMIT = "6e4d786cddc7b2d02dc4671d1c005db551d19ad6"': f'SCIENCE_COMMIT = "{SCIENCE_COMMIT}"',
        'TARGET_KERNEL = "renta0426/cmi-flu-e04-task13-rescue-20260908-001"': f'TARGET_KERNEL = "{TARGET_KERNEL}"',
    }
    for old, new in replacements.items():
        if runtime.count(old) != 1:
            raise SystemExit(f"E04b base runtime anchor count changed:{old[:50]}")
        runtime = runtime.replace(old, new, 1)

    inject_anchor = f'E04_BLOB = {E04_BLOB!r}\n'
    if runtime.count(inject_anchor) != 1:
        raise SystemExit("E04b source injection anchor changed")
    injected = (
        f'E04B_BLOB = {E04B_BLOB!r}\n'
        f'E04B_SYNTH_BLOB = {E04B_SYNTH_BLOB!r}\n'
        f'E04B_SOURCE = {e04b!r}\n'
        f'E04B_SYNTH_SOURCE = {e04b_synth!r}\n'
    )
    runtime = runtime.replace(inject_anchor, injected + inject_anchor, 1)

    selftest_anchor = '        (E04_SOURCE, E04_BLOB, "e04"),\n'
    if runtime.count(selftest_anchor) != 1:
        raise SystemExit("E04b self-test source anchor changed")
    runtime = runtime.replace(
        selftest_anchor,
        '        (E04B_SOURCE, E04B_BLOB, "e04b"),\n'
        '        (E04B_SYNTH_SOURCE, E04B_SYNTH_BLOB, "e04b_synthetic"),\n' + selftest_anchor,
        1,
    )

    runtime = replace_block(runtime, "def load_e04_modules() -> tuple[object, object]:\n", "def json_safe(value):\n", '''def load_e04_modules() -> tuple[object, object]:
    def install(name: str, source: str) -> object:
        module = types.ModuleType(name)
        module.__file__ = f"<{name}>"
        module.__package__ = "cmi_flu"
        sys.modules[name] = module
        exec(compile(source, name.replace(".", "/") + ".py", "exec"), module.__dict__, module.__dict__)
        return module
    install("cmi_flu.task13_harmonization", TASK13_SOURCE)
    install("cmi_flu.strategy_e04_contract", CONTRACT_SOURCE)
    install("cmi_flu.strategy_e04", E04_SOURCE)
    install("cmi_flu.strategy_e04_synthetic", SYNTHETIC_SOURCE)
    e04b = install("cmi_flu.strategy_e04b", E04B_SOURCE)
    synthetic = install("cmi_flu.strategy_e04b_synthetic", E04B_SYNTH_SOURCE)
    run_e04b = getattr(e04b, "run_e04b", None)
    run_synthetic = getattr(synthetic, "run_synthetic", None)
    if not callable(run_e04b) or not callable(run_synthetic):
        raise BridgeContractError("e04b_entry_missing")
    return run_e04b, run_synthetic
''')

    runtime = replace_block(runtime, "def validate_result(result: dict, *, synthetic: bool) -> None:\n", "def metric_value(", '''def validate_result(result: dict, *, synthetic: bool) -> None:
    if result.get("schema_version") != 1 or result.get("experiment") != "strategy_v2_e04b_task13_material_plural_bridge" or result.get("task") != "Task1.3":
        raise BridgeContractError("e04b_experiment_identity_mismatch")
    if result.get("parent_experiment") != "strategy_v2_e04_task13_anchor_preserving_rescue" or result.get("ontology_version") != "e04b_task13_material_plural_v1":
        raise BridgeContractError("e04b_parent_or_ontology_mismatch")
    if result.get("competition_submission_attempted") is not False or result.get("leaderboard_used_for_selection") is not False or result.get("automatic_compute_retries") != 0 or result.get("incumbent_changed") is not False:
        raise BridgeContractError("e04b_execution_boundary_mismatch")
    ontology = result.get("ontology_change") or {}
    if ontology.get("scope") != "named_strict_ASC_2024UGA_2025LJI_only" or ontology.get("source_literal") != "pbmcs" or ontology.get("canonical_literal") != "pbmc":
        raise BridgeContractError("e04b_ontology_scope_mismatch")
    if ontology.get("outcomes_inspected") is not False or ontology.get("non_material_columns_unchanged") is not True or ontology.get("non_asc_rows_changed") != 0:
        raise BridgeContractError("e04b_ontology_boundary_mismatch")
    expected_frozen = {
        "strict_b21": "pls_1",
        "anchor": "flow_rank__Antibody-secreting_cells_(ASC)",
        "historical_rank": "enet_a0.1_l0.5",
        "residual_alpha": 10.0,
        "residual_shrinkage": 0.25,
        "residual_score_cap": 0.05,
        "proxy_weights": [0.25, 1.0],
    }
    if result.get("frozen_conditions") != expected_frozen:
        raise BridgeContractError("e04b_frozen_condition_mismatch")
    audit = result.get("measurement_audit") or {}
    if audit.get("strict_gate_reconstructed") is not False or audit.get("marker_not_reported_is_negative") is not False or audit.get("challenge_D7_observed") is not False:
        raise BridgeContractError("e04b_measurement_boundary_mismatch")
    for group in ("strict_cv", "historical_loso", "strict_cv_plus_proxy"):
        if set((result.get(group) or {})) != {"proxy_weight_0.25", "proxy_weight_1"}:
            raise BridgeContractError(f"e04b_condition_set_mismatch_{group}")
    decision = result.get("e04b_decision") or {}
    transfer = result.get("e04b_transfer_audit") or {}
    if synthetic:
        if transfer.get("measurement_bridge_recovered") is not True or transfer.get("real_preconditions") is not None:
            raise BridgeContractError("e04b_synthetic_transfer_contract_mismatch")
    else:
        pre = transfer.get("real_preconditions") or {}
        if result.get("status") != "complete" or pre.get("all_pass") is not True:
            raise BridgeContractError("e04b_real_precondition_mismatch")
        if transfer.get("baseline_compatible_2024UGA_subjects") != 33 or transfer.get("baseline_compatible_2025LJI_subjects") != 40:
            raise BridgeContractError("e04b_real_baseline_bridge_mismatch")
        if transfer.get("supervised_strict_source_subjects") != [23, 23] or transfer.get("challenge_correction_rows") != [40, 40] or transfer.get("fit_states") != ["evaluated", "evaluated"]:
            raise BridgeContractError("e04b_real_supervised_bridge_mismatch")
        local = transfer.get("strict_local_reproduction") or {}
        if local.get("all_pass") is not True:
            raise BridgeContractError("e04b_local_reproduction_mismatch")
        if decision.get("measurement_bridge_recovered") is not True:
            raise BridgeContractError("e04b_bridge_not_recovered")
    if decision.get("public_probe_authorized") is not False or decision.get("competition_candidate") is not False or decision.get("decision") != "no_promotion":
        raise BridgeContractError("e04b_promotion_boundary_mismatch")
    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"', '"subject_group"', '"row_index"', '"oof_predictions"', '"challenge_predictions"')
    if any(token in serialized for token in banned):
        raise BridgeContractError("e04b_aggregate_output_contains_row_fields")
''')

    runtime = replace_block(runtime, "def render_summary(result: dict) -> str:\n", "def sha256_file(", '''def render_summary(result: dict) -> str:
    transfer = result.get("e04b_transfer_audit") or {}
    decision = result.get("e04b_decision") or {}
    local = transfer.get("strict_local_reproduction") or {}
    lines = [
        "# CMI-Flu strategy E04b Task1.3 material bridge", "",
        "Aggregate-only fixed-condition evaluation; no Competition submission was attempted.", "",
        f"- science commit: `{SCIENCE_COMMIT}`",
        f"- status: `{result.get('status')}`",
        f"- ontology: `{result.get('ontology_version')}`",
        f"- baseline-compatible 2024UGA: `{transfer.get('baseline_compatible_2024UGA_subjects')}`",
        f"- baseline-compatible 2025LJI: `{transfer.get('baseline_compatible_2025LJI_subjects')}`",
        f"- supervised strict source subjects: `{transfer.get('supervised_strict_source_subjects')}`",
        f"- Challenge correction rows: `{transfer.get('challenge_correction_rows')}`",
        f"- strict local candidate: `{(local.get('candidate_values') or [None])[0]}`",
        f"- strict local anchor: `{local.get('anchor_value')}`",
        f"- strict local delta vs anchor: `{(local.get('delta_values') or [None])[0]}`",
        f"- measurement bridge recovered: `{decision.get('measurement_bridge_recovered')}`",
        f"- decision: `{decision.get('decision')}`",
        f"- Public probe authorized: `{decision.get('public_probe_authorized')}`",
        "",
    ]
    for name in ("proxy_weight_0.25", "proxy_weight_1"):
        item = (result.get("challenge") or {}).get(name) or {}
        fit = item.get("fit") or {}
        move = item.get("vs_anchor") or {}
        lines.append(f"- challenge/{name}: state={fit.get('state')} sources={item.get('strict_bridge_source_subjects')} corrected={fit.get('correction_applied')} rank_spearman_vs_anchor={(move.get('rank_spearman') or {}).get('value')} changed_rank_count={move.get('changed_rank_count')}")
    return "\\n".join(lines) + "\\n"
''')

    bridge_anchor = '            "strategy_e04_blob_sha": E04_BLOB,\n'
    if runtime.count(bridge_anchor) != 1:
        raise SystemExit("E04b bridge metadata anchor changed")
    runtime = runtime.replace(
        bridge_anchor,
        '            "strategy_e04b_blob_sha": E04B_BLOB,\n'
        '            "strategy_e04b_synthetic_blob_sha": E04B_SYNTH_BLOB,\n' + bridge_anchor,
        1,
    )

    # Rename stable diagnostic markers after all structural patches.
    runtime = runtime.replace("CMI_FLU_E04_COMPLETE", "CMI_FLU_E04B_COMPLETE")
    runtime = runtime.replace("CMI_FLU_E04_FAILED", "CMI_FLU_E04B_FAILED")
    runtime = runtime.replace("CMI_FLU_E04_RUNTIME_SELF_TEST", "CMI_FLU_E04B_RUNTIME_SELF_TEST")
    # Final generated-source escape normalization, matching the proven E04 v2 pattern.
    runtime = runtime.replace('"\n"', '"\\n"')
    compile(runtime, "generated_e04b_runtime.py", "exec")
    return runtime, sha256(runtime.encode("utf-8"))


def main() -> int:
    a = args()
    root = a.repository_root.expanduser().resolve()
    output = a.output.expanduser().resolve()
    validate_request(root)
    e04b = require_blob(root / E04B_PATH, E04B_BLOB, "cmi_flu/strategy_e04b.py")
    e04b_synth = require_blob(root / E04B_SYNTH_PATH, E04B_SYNTH_BLOB, "cmi_flu/strategy_e04b_synthetic.py")
    runtime, runtime_sha = build_runtime(root, e04b, e04b_synth)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_E04B_PREPARE_PASS "
        f"science_commit={SCIENCE_COMMIT} e04b_blob={E04B_BLOB} e04b_synthetic_blob={E04B_SYNTH_BLOB} runtime_sha256={runtime_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
