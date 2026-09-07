#!/usr/bin/env python3
"""Build the exact aggregate-only Kaggle runtime for CMI-Flu strategy E05."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, subprocess, sys, tempfile
from pathlib import Path

REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-001"
SCIENCE_COMMIT = "e02a601f38250480b526e548a48cf7f2526c00ca"
E01_BLOB = "dd27aea0cf97d41bad3cec64819c4c4269d94cbd"
E01_V2_BLOB = "8cc64dc5ab9483d5957cfada18d445188566c56c"
HAI_TRANSFER_BLOB = "b671d8bf7f10bebbd65aca2a5bad42e267ee78d5"
E05_BLOB = "78cde9e3a6b6e3b332352218c4b4545f769aa6c4"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
BASE_BUILDER = "scripts/cmi_flu_strategy_e01_prepare.py"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e05-hai-donor-strain-001"
HAI_PATH = f"{PAYLOAD_ROOT}/hai_transfer.py"
E05_PATH = f"{PAYLOAD_ROOT}/strategy_e05.py"
REQUEST_PATH = "requests/cmi-flu-strategy-e05-hai-donor-strain-001.json"
REF_SHA256 = {
    "strain_sequences.csv": "63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887",
    "vaccine_strains_per_season.txt": "8f6c7116f37f29df0bb21d6049d82fa28b4e42b2d10ed9394a1ae6f926bd9f35",
}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

def load_base(root: Path):
    spec = importlib.util.spec_from_file_location("cmi_flu_e01_prepare_for_e05", root / BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load frozen E01 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        "SCIENCE_COMMIT": "0b2ecb47eaa09f22450424c9c06dc88cf44bc1fb",
        "E01_BLOB": E01_BLOB,
        "E01_V2_BLOB": E01_V2_BLOB,
        "CONFIG_BLOB": CONFIG_BLOB,
    }
    for key, value in expected.items():
        if getattr(module, key, None) != value:
            raise SystemExit(f"frozen E01 builder contract changed: {key}")
    return module

def load_science(root: Path):
    hai_data = (root / HAI_PATH).read_bytes()
    e05_data = (root / E05_PATH).read_bytes()
    if git_blob_sha(hai_data) != HAI_TRANSFER_BLOB:
        raise SystemExit("HAI transfer exact blob mismatch")
    if git_blob_sha(e05_data) != E05_BLOB:
        raise SystemExit("E05 exact blob mismatch")
    hai, e05 = hai_data.decode(), e05_data.decode()
    compile(hai, "cmi_flu/hai_transfer.py", "exec")
    compile(e05, "cmi_flu/strategy_e05.py", "exec")
    for token in (
        'EXPERIMENT = "strategy_v2_e05_hai_donor_strain"',
        'RIDGE_ALPHA = 10.0',
        'MAX_INTERACTIONS = 8',
        '"competition_submission_attempted": False',
    ):
        if token not in e05:
            raise SystemExit(f"E05 frozen token missing: {token}")
    if "competition_submit" in e05 or "kaggle competitions submit" in e05:
        raise SystemExit("E05 source contains submission path")
    return hai, e05

def validate_request(root: Path):
    request = json.loads((root / REQUEST_PATH).read_text())
    expected = {
        "schema_version": 1, "request_id": REQUEST_ID, "competition": COMPETITION,
        "operation": "kernel_run_and_current_output_read", "target": TARGET_KERNEL,
        "science_repository": "renta0426/CMI-Flu-Invited-Prediction-Challenge",
        "science_source_commit": SCIENCE_COMMIT, "science_transport": "agent_relay_exact_blobs",
        "strategy_e01_blob_sha": E01_BLOB, "strategy_e01_v2_blob_sha": E01_V2_BLOB,
        "hai_transfer_blob_sha": HAI_TRANSFER_BLOB, "strategy_e05_blob_sha": E05_BLOB,
        "config_blob_sha": CONFIG_BLOB, "expected_kernel_version": 1,
        "enable_internet": False, "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False, "automatic_compute_retries": 0,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"E05 request mismatch: {key}")
    if request.get("locked_reference_sha256") != REF_SHA256:
        raise SystemExit("E05 reference hash contract mismatch")
    if request.get("resource") != {"accelerator":"cpu","expected_runtime_minutes":30,"hard_timeout_minutes":60,"max_active_runs":1}:
        raise SystemExit("E05 resource contract mismatch")
    if request.get("allowed_output_paths") != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E05 output allowlist mismatch")

def replace_function(text: str, start: str, end: str, replacement: str) -> str:
    a = text.find(start)
    b = text.find(end, a + len(start))
    if a < 0 or b < 0 or b <= a:
        raise SystemExit(f"runtime anchors changed: {start!r} -> {end!r}")
    return text[:a] + replacement.rstrip() + "\n\n" + text[b:]

def patch_runtime(runtime: str, hai: str, e05: str) -> str:
    marker = "REFERENCE_PLACEHOLDERS = "
    pos = runtime.find(marker)
    if pos < 0:
        raise SystemExit("E05 injection anchor missing")
    injected = (
        f'HAI_TRANSFER_BLOB = "{HAI_TRANSFER_BLOB}"\n'
        f'E05_BLOB = "{E05_BLOB}"\n'
        f'HAI_TRANSFER_SOURCE = {hai!r}\n'
        f'E05_SOURCE = {e05!r}\n'
        f'LOCKED_REFERENCE_SHA256 = {REF_SHA256!r}\n'
    )
    runtime = runtime[:pos] + injected + runtime[pos:]
    runtime = replace_function(runtime, "def self_test() -> int:\n", "def locate_competition_data(", '''def self_test() -> int:
    package = package_bytes()
    for source, expected, label in ((E01_SOURCE,E01_BLOB,"e01"),(E01_V2_SOURCE,E01_V2_BLOB,"e01_v2"),(HAI_TRANSFER_SOURCE,HAI_TRANSFER_BLOB,"hai_transfer"),(E05_SOURCE,E05_BLOB,"e05"),(CONFIG_TEXT,CONFIG_BLOB,"config")):
        if git_blob_sha(source.encode("utf-8")) != expected:
            raise BridgeContractError(f"{label}_blob_mismatch")
    compile(HAI_TRANSFER_SOURCE, "cmi_flu/hai_transfer.py", "exec")
    compile(E05_SOURCE, "cmi_flu/strategy_e05.py", "exec")
    print(f"CMI_FLU_E05_RUNTIME_SELF_TEST PASS request_id={REQUEST_ID} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} e05_blob={E05_BLOB}")
    return 0''')
    runtime = replace_function(runtime, "def load_e01_module() -> object:\n", "def json_safe(value):\n", '''def load_e05_module() -> object:
    base = types.ModuleType("cmi_flu.strategy_e01")
    base.__file__ = "<cmi_flu.strategy_e01>"; base.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e01"] = base
    exec(compile(E01_SOURCE, "cmi_flu/strategy_e01.py", "exec"), base.__dict__, base.__dict__)
    e01v2 = types.ModuleType("cmi_flu.strategy_e01_v2")
    e01v2.__file__ = "<cmi_flu.strategy_e01_v2>"; e01v2.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e01_v2"] = e01v2
    exec(compile(E01_V2_SOURCE, "cmi_flu/strategy_e01_v2.py", "exec"), e01v2.__dict__, e01v2.__dict__)
    hai = types.ModuleType("cmi_flu.hai_transfer")
    hai.__file__ = "<cmi_flu.hai_transfer>"; hai.__package__ = "cmi_flu"; sys.modules["cmi_flu.hai_transfer"] = hai
    exec(compile(HAI_TRANSFER_SOURCE, "cmi_flu/hai_transfer.py", "exec"), hai.__dict__, hai.__dict__)
    e05 = types.ModuleType("cmi_flu.strategy_e05")
    e05.__file__ = "<cmi_flu.strategy_e05>"; e05.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e05"] = e05
    exec(compile(E05_SOURCE, "cmi_flu/strategy_e05.py", "exec"), e05.__dict__, e05.__dict__)
    run = getattr(e05, "run_strategy_e05", None)
    if not callable(run):
        raise BridgeContractError("e05_entry_missing")
    return run, hai

def locate_locked_reference(name: str) -> Path:
    expected = LOCKED_REFERENCE_SHA256[name]
    roots = [Path(__file__).resolve().parent, Path.cwd(), Path("/kaggle/working"), Path("/kaggle/src"), Path("/kaggle/input")]
    found = []
    for root in roots:
        if not root.exists():
            continue
        candidates = []
        direct = root / name
        if direct.is_file():
            candidates.append(direct)
        if str(root).startswith("/kaggle/"):
            try:
                candidates.extend(root.rglob(name))
            except OSError:
                pass
        for path in candidates:
            try:
                if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
                    found.append(path.resolve())
            except OSError:
                pass
    unique = list(dict.fromkeys(found))
    if len(unique) != 1:
        raise BridgeContractError(f"locked_reference_not_unique:{name}:{len(unique)}")
    return unique[0]''')
    runtime = replace_function(runtime, "def validate_result(result: dict) -> None:\n", "def metric_value(", '''def validate_result(result: dict) -> None:
    if result.get("experiment") != "strategy_v2_e05_hai_donor_strain":
        raise BridgeContractError("e05_experiment_identity_mismatch")
    if set((result.get("tasks") or {}).keys()) != {"Task2.1", "Task2.2"}:
        raise BridgeContractError("e05_task_set_mismatch")
    feature = result.get("feature_contract") or {}; split = result.get("split_contract") or {}; weight = result.get("weight_contract") or {}
    if feature.get("ridge_alpha") != 10.0 or feature.get("interaction_count") != 8:
        raise BridgeContractError("e05_feature_contract_mismatch")
    if feature.get("free_participant_embedding") is not False or feature.get("raw_strain_id_one_hot") is not False:
        raise BridgeContractError("e05_identity_feature_contract_mismatch")
    if split.get("fixed_subject_strain_simultaneous_holdouts") != 4 or split.get("simultaneous_selection_uses_outcomes") is not False:
        raise BridgeContractError("e05_split_contract_mismatch")
    if weight.get("study_total_weight") != "equal" or weight.get("subject_total_weight_within_study") != "equal" or weight.get("strain_total_weight_within_subject") != "equal":
        raise BridgeContractError("e05_weight_contract_mismatch")
    if result.get("leaderboard_used_for_selection") is not False or result.get("competition_submission_attempted") is not False:
        raise BridgeContractError("e05_selection_submission_contract_mismatch")
    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"','"predictions"')
    if any(token in serialized for token in banned):
        raise BridgeContractError("e05_aggregate_privacy_contract")

def metric_value(task_payload: dict) -> str:
    route = task_payload.get("routing") or {}
    return f"route={route.get('routing')} positive={route.get('positive_e05_signal')} delta={route.get('historical_proxy_study_mean_delta_vs_anchor')} rank_changed={route.get('challenge_rank_changed_by_at_least_one_position')}"''')
    runtime = replace_function(runtime, "def render_summary(result: dict) -> str:\n", "def sha256_file(", '''def render_summary(result: dict) -> str:
    lines = ["# CMI-Flu strategy E05 HAI donor x strain", "", "Aggregate-only fixed-condition evaluation; no Competition submission was attempted.", "", f"- science commit: `{SCIENCE_COMMIT}`", f"- E05 blob: `{E05_BLOB}`", ""]
    for task, payload in (result.get("tasks") or {}).items():
        lines.extend([f"## {task}", f"- {metric_value(payload)}", ""])
    return "\\n".join(lines)''')
    runtime = replace_function(runtime, '        stage = "load_e01"\n', '        stage = "write_outputs"\n', '''        stage = "load_e05"
        run_e05, hai_module = load_e05_module()
        stage = "load_locked_references"
        import pandas as pd
        sequence_reference = pd.read_csv(locate_locked_reference("strain_sequences.csv"))
        vaccine_reference = hai_module.load_vaccine_strain_reference(locate_locked_reference("vaccine_strains_per_season.txt"))
        stage = "run_e05"
        result = json_safe(dict(run_e05(config, inputs, sequence_reference=sequence_reference, vaccine_reference=vaccine_reference)))
        stage = "validate_e05"
        validate_result(result)''')
    old_bridge = '            "frozen_incumbent": result["frozen_incumbent"],\n'
    if runtime.count(old_bridge) != 1:
        raise SystemExit("E05 bridge provenance anchor changed")
    runtime = runtime.replace(old_bridge, '            "strategy_e05_blob_sha": E05_BLOB,\n            "hai_transfer_blob_sha": HAI_TRANSFER_BLOB,\n', 1)
    runtime = runtime.replace("CMI-Flu strategy E01 paired evaluation. Aggregate outputs only; no submission.", "CMI-Flu strategy E05 HAI donor x strain. Aggregate outputs only; no submission.", 1)
    runtime = runtime.replace("CMI_FLU_E01_FAILED", "CMI_FLU_E05_FAILED").replace("CMI_FLU_E01_COMPLETE", "CMI_FLU_E05_COMPLETE")
    runtime = runtime.replace("tasks={len(result['tasks'])}", "tasks={len(result['tasks'])}")
    if "competition_submit" in runtime or "kaggle competitions submit" in runtime:
        raise SystemExit("generated E05 runtime contains submission path")
    compile(runtime, "generated_e05.py", "exec")
    return runtime

def main():
    args = parse_args(); root = args.repository_root.expanduser().resolve(); validate_request(root)
    base = load_base(root); e01, e01v2, config = base.load_exact_science(root); hai, e05 = load_science(root)
    with tempfile.TemporaryDirectory(prefix="cmi-e05-build-") as tmp:
        package, adapter = base.extract_frozen_runtime(root, Path(tmp))
        base.REQUEST_ID = REQUEST_ID; base.TARGET_KERNEL = TARGET_KERNEL; base.SCIENCE_COMMIT = SCIENCE_COMMIT
        runtime = base.build_runtime(package, adapter, e01, e01v2, config)
    runtime = patch_runtime(runtime, hai, e05)
    out = args.output.expanduser().resolve(); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(f"CMI_FLU_E05_BUILD PASS science_commit={SCIENCE_COMMIT} e05_blob={E05_BLOB} runtime_sha256={sha256(runtime.encode())}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
