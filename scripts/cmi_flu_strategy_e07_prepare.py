#!/usr/bin/env python3
"""Build a self-contained E07 runtime from exact science blobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REQUEST_ID = "20260909-cmi-flu-strategy-e07-task14-formal-closure-001"
SCIENCE_COMMIT = "a6cdc7c43bc487d88e9c15f2c63565ff8c702dac"
TARGET = "renta0426/cmi-flu-e07-task14-formal-closure-20260909-001"
REQUEST_PATH = "requests/cmi-flu-strategy-e07-task14-formal-closure-001.json"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e07-task14-formal-closure-001"
BLOBS = {
    "strategy_e07.py": "44ebdd2a1dc9ceeb771eaeb295c74545eb3cc1a4",
    "strategy_e07_synthetic.py": "b8971d05d0d1f094b03f861ec3438c6f17c8587a",
    "aliases.py": "5b9930c8e26f2b462ee8ce64fdfbdb76df2d1af3",
    "contracts.py": "6e2791e526255d9c534e71c9c0886f0c300df7bd",
    "targets.py": "154681d9a51e36abd635b596c6d8f7cee22c2d96",
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def load_sources(root: Path) -> dict[str, str]:
    sources = {}
    for name, expected in BLOBS.items():
        path = root / PAYLOAD_ROOT / name
        data = path.read_bytes()
        found = git_blob_sha(data)
        if found != expected:
            raise SystemExit(f"E07 relay mismatch:{name}:{found}")
        text = data.decode("utf-8")
        compile(text, name, "exec")
        sources[name] = text
    return sources


def validate_request(root: Path) -> None:
    request = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": "cmi-flu-first-prediction-challenge",
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET,
        "science_repository": "renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": "agent_relay_exact_blobs",
        "expected_kernel_version": 1,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"E07 request mismatch:{key}")
    pinned = {
        "src/cmi_flu/strategy_e07.py": BLOBS["strategy_e07.py"],
        "src/cmi_flu/strategy_e07_synthetic.py": BLOBS["strategy_e07_synthetic.py"],
        "src/cmi_flu/aliases.py": BLOBS["aliases.py"],
        "src/cmi_flu/contracts.py": BLOBS["contracts.py"],
        "src/cmi_flu/targets.py": BLOBS["targets.py"],
    }
    if request.get("pinned_git_blobs") != pinned:
        raise SystemExit("E07 pinned blob contract mismatch")
    if request.get("resource") != {"accelerator": "cpu", "expected_runtime_minutes": 10, "hard_timeout_minutes": 30, "max_active_runs": 1}:
        raise SystemExit("E07 resource contract mismatch")
    if request.get("required_competition_files") != ["2025LJI_aim.tsv", "2025LJI_vdj.tsv", "participants.tsv", "participant_hla.tsv"]:
        raise SystemExit("E07 required file contract mismatch")
    if request.get("allowed_output_paths") != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E07 output allowlist mismatch")
    contract = request.get("experiment_contract") or {}
    locked = {
        "experiment": "strategy_v2_e07_task14_formal_closure",
        "task": "Task1.4",
        "contract_version": "e07_task14_zero_shot_audit_v1",
        "public_training_aim_rows": 0,
        "expected_challenge_aim_rows": 600,
        "expected_challenge_subjects": 40,
        "expected_stimulation_categories": 5,
        "prevacc_arithmetic_tolerance": 1e-8,
        "minimum_external_teacher_coverage": 28,
        "epitope_hla_map_available": False,
        "generic_tcr_clonality_teacher_forbidden": True,
        "background_contrast_sensitivity_only": True,
        "recommended_default_predictor": "raw_pre_vacc_conserved_anchor",
        "public_probe_authorized": False,
        "no_supervised_cv_without_compatible_aim_outcomes": True,
    }
    for key, value in locked.items():
        if contract.get(key) != value:
            raise SystemExit(f"E07 experiment contract mismatch:{key}")


def build_runtime(sources: dict[str, str]) -> str:
    return f'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, types
from pathlib import Path
import numpy as np
import pandas as pd

REQUEST_ID = {REQUEST_ID!r}
SCIENCE_COMMIT = {SCIENCE_COMMIT!r}
TARGET = {TARGET!r}
BLOBS = {BLOBS!r}
CONTRACT_SOURCE = {sources["contracts.py"]!r}
ALIASES_SOURCE = {sources["aliases.py"]!r}
TARGETS_SOURCE = {sources["targets.py"]!r}
E07_SOURCE = {sources["strategy_e07.py"]!r}
E07_SYNTH_SOURCE = {sources["strategy_e07_synthetic.py"]!r}
REQUIRED_FILES = ("2025LJI_aim.tsv", "2025LJI_vdj.tsv", "participants.tsv", "participant_hla.tsv")


def git_blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\\0" + data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def install_modules():
    pkg = types.ModuleType("cmi_flu")
    pkg.__path__ = []
    sys.modules["cmi_flu"] = pkg
    def install(name, source):
        module = types.ModuleType(name)
        module.__package__ = "cmi_flu"
        module.__file__ = "<" + name + ">"
        sys.modules[name] = module
        exec(compile(source, name.replace(".", "/") + ".py", "exec"), module.__dict__, module.__dict__)
        return module
    install("cmi_flu.contracts", CONTRACT_SOURCE)
    install("cmi_flu.aliases", ALIASES_SOURCE)
    install("cmi_flu.targets", TARGETS_SOURCE)
    e07 = install("cmi_flu.strategy_e07", E07_SOURCE)
    synth = install("cmi_flu.strategy_e07_synthetic", E07_SYNTH_SOURCE)
    return e07, synth


def self_test():
    checks = ((CONTRACT_SOURCE, "contracts.py"), (ALIASES_SOURCE, "aliases.py"), (TARGETS_SOURCE, "targets.py"), (E07_SOURCE, "strategy_e07.py"), (E07_SYNTH_SOURCE, "strategy_e07_synthetic.py"))
    for source, name in checks:
        if git_blob_sha(source.encode("utf-8")) != BLOBS[name]:
            raise SystemExit("E07 runtime blob mismatch:" + name)
        compile(source, name, "exec")
    e07, synth = install_modules()
    if not callable(getattr(e07, "run_e07", None)) or not callable(getattr(synth, "run_synthetic", None)):
        raise SystemExit("E07 entrypoint missing")
    print("CMI_FLU_E07_RUNTIME_SELF_TEST PASS science_commit=" + SCIENCE_COMMIT + " e07_blob=" + BLOBS["strategy_e07.py"])


def locate_data_root():
    root = Path("/kaggle/input")
    candidates = []
    for aim in root.rglob("2025LJI_aim.tsv"):
        parent = aim.parent
        if all((parent / name).is_file() for name in REQUIRED_FILES) and (parent / "md5sum").is_file():
            candidates.append(parent)
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError("E07 competition input root is not unique")
    return unique[0]


def verify_required_md5(root):
    entries = {{}}
    for raw in (root / "md5sum").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError("E07 malformed md5 manifest")
        digest, name = parts
        key = Path(name.strip().lstrip("*")).name
        if key in entries and entries[key] != digest.casefold():
            raise RuntimeError("E07 duplicate md5 basename conflict")
        entries[key] = digest.casefold()
    verified = 0
    for name in REQUIRED_FILES:
        expected = entries.get(name)
        if not expected:
            raise RuntimeError("E07 required md5 entry missing")
        if md5_file(root / name) != expected:
            raise RuntimeError("E07 required md5 mismatch")
        verified += 1
    return verified


def validate_result(result, synthetic):
    if result.get("schema_version") != 1 or result.get("experiment") != "strategy_v2_e07_task14_formal_closure" or result.get("task") != "Task1.4":
        raise RuntimeError("E07 experiment identity mismatch")
    if result.get("contract_version") != "e07_task14_zero_shot_audit_v1":
        raise RuntimeError("E07 contract version mismatch")
    if result.get("outcomes_accessed") is not False or result.get("competition_submission_attempted") is not False or result.get("leaderboard_used_for_selection") is not False or result.get("automatic_compute_retries") != 0:
        raise RuntimeError("E07 execution boundary mismatch")
    if result.get("public_training_aim_rows") != 0 or result.get("public_training_aim_studies") != 0:
        raise RuntimeError("E07 fabricated AIM teacher")
    decision = result.get("decision") or {{}}
    if decision.get("recommended_task14_predictor") != "raw_pre_vacc_conserved_anchor" or decision.get("incumbent_changed") is not False or decision.get("public_probe_authorized") is not False or decision.get("supervised_cv_available") is not False:
        raise RuntimeError("E07 decision boundary mismatch")
    if not synthetic:
        real = result.get("real_contract") or {{}}
        if real.get("aim_rows") != 600 or real.get("challenge_subjects") != 40 or real.get("anchor_subjects") != 40 or real.get("stimulation_count") != 5:
            raise RuntimeError("E07 real structural counts changed")
    payload = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"', '"subject"', '"barcode"', '"contig_id"', '"cdr3"', 'SYN_ONLY_E07_')
    if any(token in payload for token in banned):
        raise RuntimeError("E07 unsafe aggregate payload")


def write_outputs(result, output_dir, synthetic, md5_verified):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics = output_dir / "metrics.json"
    summary = output_dir / "summary.md"
    metrics.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\\n", encoding="utf-8")
    aim = result.get("aim_audit") or {{}}
    negative = aim.get("negative_control") or {{}}
    vdj = result.get("vdj_applicability") or {{}}
    hla = result.get("hla_applicability") or {{}}
    decision = result.get("decision") or {{}}
    lines = [
        "# CMI-Flu strategy E07 Task1.4 formal closure", "",
        "Aggregate-only challenge baseline audit; no Task1.4 outcomes or Competition submission were used.", "",
        f"- science commit: `{{SCIENCE_COMMIT}}`",
        f"- status: `{{result.get('status')}}`",
        f"- AIM rows/subjects/stimulations: `{{aim.get('rows')}} / {{aim.get('subjects')}} / {{aim.get('stimulation_count')}}`",
        f"- stimuli: `{{aim.get('stimulations')}}`",
        f"- Pre-vacc arithmetic check: `{{(aim.get('prevacc_arithmetic_mean_check') or {{}}).get('all_within_tolerance')}}`",
        f"- conserved -14/day0 repeat Spearman: `{{((aim.get('conserved_repeat_reliability') or {{}}).get('spearman') or {{}}).get('value')}}`",
        f"- negative-control candidates: `{{negative.get('candidate_names')}}`",
        f"- background contrast sensitivity available: `{{negative.get('sensitivity_available')}}`",
        f"- VDJ participant coverage: `{{vdj.get('participant_coverage')}}`",
        f"- external TCR teacher applicable: `{{vdj.get('external_tcr_teacher_applicable')}}`",
        f"- HLA participant coverage: `{{hla.get('participant_coverage')}}`",
        f"- HLA correction applicable: `{{hla.get('hla_correction_applicable')}}`",
        f"- decision: `{{decision.get('decision')}}`",
        f"- recommended predictor: `{{decision.get('recommended_task14_predictor')}}`",
    ]
    summary.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    bridge = {{
        "request_id": REQUEST_ID,
        "science_commit": SCIENCE_COMMIT,
        "strategy_e07_blob_sha": BLOBS["strategy_e07.py"],
        "strategy_e07_synthetic_blob_sha": BLOBS["strategy_e07_synthetic.py"],
        "aliases_blob_sha": BLOBS["aliases.py"],
        "contracts_blob_sha": BLOBS["contracts.py"],
        "targets_blob_sha": BLOBS["targets.py"],
        "metrics_sha256": sha256_file(metrics),
        "summary_sha256": sha256_file(summary),
        "md5_verified_count": md5_verified,
        "synthetic": bool(synthetic),
        "outcomes_accessed": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "contains_participant_identifiers": False,
        "contains_row_level_predictions": False,
    }}
    (output_dir / "bridge-result.json").write_text(json.dumps(bridge, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--output-dir", type=Path)
    a = p.parse_args()
    if a.self_test:
        self_test(); return 0
    e07, synth = install_modules()
    if a.synthetic:
        result, _ = synth.run_synthetic()
        validate_result(result, True)
        if a.output_dir is None:
            raise SystemExit("synthetic output dir required")
        write_outputs(result, a.output_dir, True, None)
        print("CMI_FLU_E07_SYNTHETIC PASS")
        return 0
    data = locate_data_root()
    verified = verify_required_md5(data)
    read = lambda name: pd.read_csv(data / name, sep="\\t", low_memory=False)
    result = e07.run_e07(
        challenge_aim=read("2025LJI_aim.tsv"),
        challenge_vdj=read("2025LJI_vdj.tsv"),
        participants=read("participants.tsv"),
        participant_hla=read("participant_hla.tsv"),
        expected_real_counts=True,
        epitope_hla_map_available=False,
    )
    validate_result(result, False)
    write_outputs(result, Path("/kaggle/working"), False, verified)
    print("CMI_FLU_E07_RUNTIME PASS status=" + str(result.get("status")) + " decision=" + str((result.get("decision") or {{}}).get("decision")))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> int:
    a = args()
    root = a.repository_root.resolve()
    validate_request(root)
    sources = load_sources(root)
    runtime = build_runtime(sources)
    compile(runtime, "generated_e07_runtime.py", "exec")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(a.output), "--self-test"], check=True)
    print(
        "CMI_FLU_E07_PREPARE PASS "
        f"science_commit={SCIENCE_COMMIT} e07_blob={BLOBS['strategy_e07.py']} "
        f"runtime_sha256={hashlib.sha256(runtime.encode()).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
