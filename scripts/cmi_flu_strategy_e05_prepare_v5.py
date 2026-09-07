#!/usr/bin/env python3
"""Build E05 006 runtime with exact organizer-reference and donor-rank adapters."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-006"
TARGET_KERNEL = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-006"
SCIENCE_COMMIT = "8209ee237dace5ee383d7983d3d56f2e845dfdca"
OLD_REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-004"
OLD_TARGET_KERNEL = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-004"
OLD_SCIENCE_COMMIT = "3bcddeaf4b028795363a6e12f8af06dd23b3bbdf"
E05_BLOB = "78cde9e3a6b6e3b332352218c4b4545f769aa6c4"
E05_V2_BLOB = "50951807d28418bb50f4e9dba656fa2b3e4d86ff"
E05_V3_BLOB = "129a4a5713a758d58adbb02489105e17a1c13922"
HAI_TRANSFER_BLOB = "b671d8bf7f10bebbd65aca2a5bad42e267ee78d5"
HAI_TRANSFER_V2_BLOB = "ebf4110df4ab49d726d80af03c1203f93e9c89c0"
E01_BLOB = "dd27aea0cf97d41bad3cec64819c4c4269d94cbd"
E01_V2_BLOB = "8cc64dc5ab9483d5957cfada18d445188566c56c"
CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
V4_PREPARE = "scripts/cmi_flu_strategy_e05_prepare_v4.py"
PAYLOAD_ROOT = "payloads/cmi-flu-strategy-e05-hai-donor-strain-006"
HAI_V2_PATH = f"{PAYLOAD_ROOT}/hai_transfer_v2.py"
E05_V3_PATH = f"{PAYLOAD_ROOT}/strategy_e05_v3.py"
REF_SHA256 = {
    "strain_sequences.csv": "63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887",
    "vaccine_strains_per_season.txt": "8f6c7116f37f29df0bb21d6049d82fa28b4e42b2d10ed9394a1ae6f926bd9f35",
}
EXPECTED_NATIVE_COLUMNS = ("Virus", "Sequence", "Status_of_sequence")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_v4(root: Path):
    path = root / V4_PREPARE
    spec = importlib.util.spec_from_file_location("cmi_flu_e05_prepare_v4_for_v5", path)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load E05 v4 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        "REQUEST_ID": OLD_REQUEST_ID,
        "TARGET_KERNEL": OLD_TARGET_KERNEL,
        "SCIENCE_COMMIT": OLD_SCIENCE_COMMIT,
        "E05_BLOB": E05_BLOB,
        "E05_V2_BLOB": E05_V2_BLOB,
        "HAI_TRANSFER_BLOB": HAI_TRANSFER_BLOB,
        "E01_BLOB": E01_BLOB,
        "E01_V2_BLOB": E01_V2_BLOB,
        "CONFIG_BLOB": CONFIG_BLOB,
        "REF_SHA256": REF_SHA256,
    }
    for key, value in expected.items():
        if getattr(module, key, None) != value:
            raise SystemExit(f"E05 v4 builder contract changed:{key}")
    return module


def load_exact_source(root: Path, relative: str, expected_blob: str, label: str) -> str:
    data = (root / relative).read_bytes()
    if git_blob_sha(data) != expected_blob:
        raise SystemExit(f"{label} exact blob mismatch")
    source = data.decode("utf-8")
    compile(source, f"cmi_flu/{Path(relative).name}", "exec")
    return source


def build_v4_runtime(v4, root: Path, reference_dir: Path) -> tuple[object, str]:
    v3 = v4.load_v3(root)
    shim = v4.load_shim(root)
    runtime = v4.patch_runtime(v3, v4.build_v3_runtime(v3, root, reference_dir), shim)
    return v3, runtime


def patch_runtime(v3, runtime: str, hai_v2: str, e05_v3: str) -> str:
    for old, new in (
        (OLD_REQUEST_ID, REQUEST_ID),
        (OLD_TARGET_KERNEL, TARGET_KERNEL),
        (OLD_SCIENCE_COMMIT, SCIENCE_COMMIT),
    ):
        if runtime.count(old) < 1:
            raise SystemExit(f"E05 v5 identity anchor missing:{old}")
        runtime = runtime.replace(old, new)

    marker = f'E05_V2_BLOB = "{E05_V2_BLOB}"\n'
    if runtime.count(marker) != 1:
        raise SystemExit("E05 v5 source injection anchor changed")
    runtime = runtime.replace(
        marker,
        marker
        + f'HAI_TRANSFER_V2_BLOB = "{HAI_TRANSFER_V2_BLOB}"\n'
        + f'E05_V3_BLOB = "{E05_V3_BLOB}"\n'
        + f'HAI_TRANSFER_V2_SOURCE = {hai_v2!r}\n'
        + f'E05_V3_SOURCE = {e05_v3!r}\n',
        1,
    )

    load_replacement = '''def load_e05_module() -> object:
    base = types.ModuleType("cmi_flu.strategy_e01")
    base.__file__ = "<cmi_flu.strategy_e01>"; base.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e01"] = base
    exec(compile(E01_SOURCE, "cmi_flu/strategy_e01.py", "exec"), base.__dict__, base.__dict__)
    e01v2 = types.ModuleType("cmi_flu.strategy_e01_v2")
    e01v2.__file__ = "<cmi_flu.strategy_e01_v2>"; e01v2.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e01_v2"] = e01v2
    exec(compile(E01_V2_SOURCE, "cmi_flu/strategy_e01_v2.py", "exec"), e01v2.__dict__, e01v2.__dict__)
    hai = types.ModuleType("cmi_flu.hai_transfer")
    hai.__file__ = "<cmi_flu.hai_transfer>"; hai.__package__ = "cmi_flu"; sys.modules["cmi_flu.hai_transfer"] = hai
    exec(compile(HAI_TRANSFER_SOURCE, "cmi_flu/hai_transfer.py", "exec"), hai.__dict__, hai.__dict__)
    hai_v2 = types.ModuleType("cmi_flu.hai_transfer_v2")
    hai_v2.__file__ = "<cmi_flu.hai_transfer_v2>"; hai_v2.__package__ = "cmi_flu"; sys.modules["cmi_flu.hai_transfer_v2"] = hai_v2
    exec(compile(HAI_TRANSFER_V2_SOURCE, "cmi_flu/hai_transfer_v2.py", "exec"), hai_v2.__dict__, hai_v2.__dict__)
    e05 = types.ModuleType("cmi_flu.strategy_e05")
    e05.__file__ = "<cmi_flu.strategy_e05>"; e05.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e05"] = e05
    exec(compile(E05_SOURCE, "cmi_flu/strategy_e05.py", "exec"), e05.__dict__, e05.__dict__)
    e05v2 = types.ModuleType("cmi_flu.strategy_e05_v2")
    e05v2.__file__ = "<cmi_flu.strategy_e05_v2>"; e05v2.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e05_v2"] = e05v2
    exec(compile(E05_V2_SOURCE, "cmi_flu/strategy_e05_v2.py", "exec"), e05v2.__dict__, e05v2.__dict__)
    e05v3 = types.ModuleType("cmi_flu.strategy_e05_v3")
    e05v3.__file__ = "<cmi_flu.strategy_e05_v3>"; e05v3.__package__ = "cmi_flu"; sys.modules["cmi_flu.strategy_e05_v3"] = e05v3
    exec(compile(E05_V3_SOURCE, "cmi_flu/strategy_e05_v3.py", "exec"), e05v3.__dict__, e05v3.__dict__)
    run = getattr(e05v3, "run_strategy_e05", None)
    if not callable(run):
        raise BridgeContractError("e05_v3_entry_missing")
    if getattr(e05, "_subject_level_rank", None) is not getattr(e05v2, "_donor_level_rank", None):
        raise BridgeContractError("e05_donor_rank_patch_not_installed")
    if getattr(hai, "build_sequence_lookup", None) is not getattr(hai_v2, "build_sequence_lookup", None):
        raise BridgeContractError("e05_sequence_schema_patch_not_installed")
    return run, hai'''
    runtime = v3.replace_top_level_function(runtime, "load_e05_module", load_replacement)

    self_test_replacement = '''def self_test() -> int:
    package = package_bytes()
    for source, expected, label in ((E01_SOURCE,E01_BLOB,"e01"),(E01_V2_SOURCE,E01_V2_BLOB,"e01_v2"),(HAI_TRANSFER_SOURCE,HAI_TRANSFER_BLOB,"hai_transfer"),(HAI_TRANSFER_V2_SOURCE,HAI_TRANSFER_V2_BLOB,"hai_transfer_v2"),(E05_SOURCE,E05_BLOB,"e05"),(E05_V2_SOURCE,E05_V2_BLOB,"e05_v2"),(E05_V3_SOURCE,E05_V3_BLOB,"e05_v3"),(CONFIG_TEXT,CONFIG_BLOB,"config")):
        if git_blob_sha(source.encode("utf-8")) != expected:
            raise BridgeContractError(f"{label}_blob_mismatch")
    compile(HAI_TRANSFER_SOURCE, "cmi_flu/hai_transfer.py", "exec")
    compile(HAI_TRANSFER_V2_SOURCE, "cmi_flu/hai_transfer_v2.py", "exec")
    compile(E05_SOURCE, "cmi_flu/strategy_e05.py", "exec")
    compile(E05_V2_SOURCE, "cmi_flu/strategy_e05_v2.py", "exec")
    compile(E05_V3_SOURCE, "cmi_flu/strategy_e05_v3.py", "exec")
    print(f"CMI_FLU_E05_V5_RUNTIME_SELF_TEST PASS request_id={REQUEST_ID} package_bytes={len(package)} science_commit={SCIENCE_COMMIT} e05_v3_blob={E05_V3_BLOB} hai_v2_blob={HAI_TRANSFER_V2_BLOB}")
    return 0'''
    runtime = v3.replace_top_level_function(runtime, "self_test", self_test_replacement)

    validate_replacement = '''def validate_result(result: dict) -> None:
    if result.get("experiment") != "strategy_v2_e05_hai_donor_strain":
        raise BridgeContractError("e05_experiment_identity_mismatch")
    if set((result.get("tasks") or {}).keys()) != {"Task2.1", "Task2.2"}:
        raise BridgeContractError("e05_task_set_mismatch")
    feature = result.get("feature_contract") or {}; split = result.get("split_contract") or {}; weight = result.get("weight_contract") or {}
    if feature.get("ridge_alpha") != 10.0 or feature.get("interaction_count") != 8:
        raise BridgeContractError("e05_feature_contract_mismatch")
    if feature.get("free_participant_embedding") is not False or feature.get("raw_strain_id_one_hot") is not False:
        raise BridgeContractError("e05_identity_feature_contract_mismatch")
    if feature.get("donor_rank_unit") != "participant_id_within_study" or feature.get("subject_group_role") != "leakage_purge_and_hierarchical_weighting_only":
        raise BridgeContractError("e05_donor_rank_identity_contract_mismatch")
    if feature.get("sequence_reference_schema") != "organizer_native_or_canonical_explicit_adapter":
        raise BridgeContractError("e05_sequence_reference_schema_contract_mismatch")
    if feature.get("organizer_sequence_reference_columns") != ["Virus","Sequence","Status_of_sequence"]:
        raise BridgeContractError("e05_sequence_reference_columns_contract_mismatch")
    if split.get("fixed_subject_strain_simultaneous_holdouts") != 4 or split.get("simultaneous_selection_uses_outcomes") is not False:
        raise BridgeContractError("e05_split_contract_mismatch")
    if weight.get("study_total_weight") != "equal" or weight.get("subject_total_weight_within_study") != "equal" or weight.get("strain_total_weight_within_subject") != "equal":
        raise BridgeContractError("e05_weight_contract_mismatch")
    if result.get("leaderboard_used_for_selection") is not False or result.get("competition_submission_attempted") is not False:
        raise BridgeContractError("e05_selection_submission_contract_mismatch")
    serialized = json.dumps(result, sort_keys=True, ensure_ascii=False)
    banned = ('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"','"predictions"')
    if any(token in serialized for token in banned):
        raise BridgeContractError("e05_aggregate_privacy_contract")'''
    runtime = v3.replace_top_level_function(runtime, "validate_result", validate_replacement)

    provenance = '            "strategy_e05_v2_blob_sha": E05_V2_BLOB,\n'
    if runtime.count(provenance) != 1:
        raise SystemExit("E05 v5 provenance anchor changed")
    runtime = runtime.replace(
        provenance,
        provenance
        + '            "strategy_e05_v3_blob_sha": E05_V3_BLOB,\n'
        + '            "hai_transfer_v2_blob_sha": HAI_TRANSFER_V2_BLOB,\n',
        1,
    )

    names = v3.top_level_functions(runtime)
    for name in ("json_safe", "load_e05_module", "validate_result", "locked_reference_bytes", "locate_locked_reference"):
        if names.count(name) != 1:
            raise SystemExit(f"E05 v5 top-level binding contract failed:{name}:{names.count(name)}")
    for prior in (OLD_REQUEST_ID, OLD_TARGET_KERNEL, OLD_SCIENCE_COMMIT):
        if prior in runtime:
            raise SystemExit(f"prior E05 identity remained in v5 runtime:{prior}")
    if "competition_submit" in runtime or "kaggle competitions submit" in runtime:
        raise SystemExit("E05 v5 runtime contains submission path")
    compile(runtime, "generated_e05_v5.py", "exec")
    return runtime


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    v4 = load_v4(root)
    hai_v2 = load_exact_source(root, HAI_V2_PATH, HAI_TRANSFER_V2_BLOB, "HAI transfer v2")
    e05_v3 = load_exact_source(root, E05_V3_PATH, E05_V3_BLOB, "E05 v3")
    v3, base_runtime = build_v4_runtime(v4, root, reference_dir)
    runtime = patch_runtime(v3, base_runtime, hai_v2, e05_v3)
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E05_V5_BUILD PASS "
        f"science_commit={SCIENCE_COMMIT} request_id={REQUEST_ID} target={TARGET_KERNEL} "
        f"runtime_sha256={sha256(runtime.encode())} e05_v3_blob={E05_V3_BLOB} "
        f"hai_v2_blob={HAI_TRANSFER_V2_BLOB} real_reference_schema=guarded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
