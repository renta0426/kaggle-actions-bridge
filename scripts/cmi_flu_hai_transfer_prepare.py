#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import py_compile
import subprocess
import sys
from typing import Any

REQUEST_ID = "20260904-cmi-flu-hai-transfer-001"
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_COMMIT = "d1ebf13dc0dc7e5d5a2798b29c288265cbf56618"
SCIENCE_PATH = "src/cmi_flu/hai_transfer.py"
SCIENCE_BLOB = "b671d8bf7f10bebbd65aca2a5bad42e267ee78d5"
SCIENCE_TRANSPORT = "agent_relay_exact_blob_base64_parts"
B21_BASE_REQUEST_ID = "20260903-cmi-flu-b21-001"
B21_BASE_SOURCE_COMMIT = "33030746bc7bad02ad2c1e670ac319cc943c524d"
B21_ADAPTER_BLOB = "de71dea0e335bdcd79325c0de926bf8848d0979f"
BASE_SHA256 = "7894a9232e7372950f70a36d35de41bcc8bda90ec4d0320187e82c3b8c76f2db"
BASE_SIZE = 93363
SEQUENCE_SHA256 = "63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887"
VACCINE_SHA256 = "8f6c7116f37f29df0bb21d6049d82fa28b4e42b2d10ed9394a1ae6f926bd9f35"
SEQUENCE_ROWS = 352
VACCINE_ROWS = 72
TARGET_COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-phase-a-hai-strain-transfer-20260904-001"
TARGET_STAGE = "phase_a_hai_strain_transfer"
EXPECTED_KERNEL_VERSION = 1
CONDITIONS = (
    "b21_reference",
    "ontology_metadata",
    "ontology_sequence_local",
    "ontology_sequence_target_domain",
)
TASKS = ("Task2.1", "Task2.2", "Task2.3")
ALLOWED_OUTPUT_PATHS = [
    "cmi-flu-hai-transfer/bridge-result.json",
    "cmi-flu-hai-transfer/metrics.json",
    "cmi-flu-hai-transfer/summary.md",
]


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def load_science(root: pathlib.Path) -> tuple[str, str]:
    payload_dir = root / "payloads/cmi-flu-hai-transfer-001"
    parts = [payload_dir / f"hai_transfer.b64.part-{index:02d}" for index in range(1, 5)]
    missing = [str(path) for path in parts if not path.is_file()]
    if missing:
        raise SystemExit(f"missing HAI science payload parts: {missing}")
    encoded = "".join("".join(path.read_text(encoding="utf-8").split()) for path in parts)
    try:
        data = base64.b64decode(encoded, validate=True)
    except Exception as error:
        raise SystemExit(f"HAI science payload is not valid base64: {type(error).__name__}") from error
    found_blob = git_blob_sha(data)
    if found_blob != SCIENCE_BLOB:
        raise SystemExit(f"HAI science source blob mismatch: {found_blob}")
    source = data.decode("utf-8")
    compile(source, SCIENCE_PATH, "exec")
    required = (
        '"b21_reference"',
        '"ontology_metadata"',
        '"ontology_sequence_local"',
        '"ontology_sequence_target_domain"',
        "purged_leave_one_group_out",
        "run_hai_strain_transfer_experiment",
        '"leaderboard_used_for_selection": False',
        '"competition_submission_attempted": False',
    )
    missing_tokens = [token for token in required if token not in source]
    if missing_tokens:
        raise SystemExit(f"HAI science source contract tokens missing: {missing_tokens}")
    return source, sha256_bytes(data)


def validate_request(root: pathlib.Path) -> dict[str, Any]:
    request = json.loads((root / "requests/cmi-flu-hai-transfer-001.json").read_text(encoding="utf-8"))
    if request.get("schema_version") != 1 or request.get("request_id") != REQUEST_ID:
        raise SystemExit("HAI request identity mismatch")
    if request.get("competition") != TARGET_COMPETITION or request.get("target") != TARGET_KERNEL:
        raise SystemExit("HAI target mismatch")
    if request.get("operation") != "kernel_run_and_current_output_read":
        raise SystemExit("HAI operation mismatch")
    if request.get("science_repository") != SCIENCE_REPOSITORY:
        raise SystemExit("HAI science repository mismatch")
    if request.get("science_source_commit") != SCIENCE_COMMIT:
        raise SystemExit("HAI science commit mismatch")
    if request.get("science_source_path") != SCIENCE_PATH:
        raise SystemExit("HAI science path mismatch")
    if request.get("science_transport") != SCIENCE_TRANSPORT:
        raise SystemExit("HAI science transport mismatch")
    if request.get("hai_transfer_blob_sha") != SCIENCE_BLOB:
        raise SystemExit("HAI science blob mismatch")
    if request.get("b21_base_request_id") != B21_BASE_REQUEST_ID:
        raise SystemExit("B2.1 lineage mismatch")
    if request.get("b21_runtime_adapter_blob_sha") != B21_ADAPTER_BLOB:
        raise SystemExit("B2.1 adapter provenance mismatch")
    if request.get("sequence_reference_sha256") != SEQUENCE_SHA256:
        raise SystemExit("sequence-reference manifest mismatch")
    if request.get("vaccine_reference_sha256") != VACCINE_SHA256:
        raise SystemExit("vaccine-reference manifest mismatch")
    if request.get("expected_kernel_version") != EXPECTED_KERNEL_VERSION:
        raise SystemExit("expected kernel version mismatch")
    if request.get("allowed_output_paths") != ALLOWED_OUTPUT_PATHS:
        raise SystemExit("allowed output path contract mismatch")
    if request.get("resource") != {
        "accelerator": "cpu",
        "expected_runtime_minutes": 120,
        "hard_timeout_minutes": 210,
        "max_active_runs": 1,
    }:
        raise SystemExit("HAI resource contract mismatch")
    if request.get("api_budget") != {
        "max_calls": 30,
        "poll_interval_seconds": 900,
        "max_pages": 2,
    }:
        raise SystemExit("HAI API budget mismatch")
    if request.get("competition_submission_attempted") is not False:
        raise SystemExit("HAI must not attempt Competition submission")
    if request.get("automatic_compute_retries") != 0 or request.get("enable_internet") is not False:
        raise SystemExit("HAI safety contract mismatch")
    return request


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def injected_helpers(
    science_source: str,
    science_sha256: str,
    sequence_bytes: bytes,
    vaccine_bytes: bytes,
) -> str:
    sequence_b64 = base64.b64encode(sequence_bytes).decode("ascii")
    vaccine_b64 = base64.b64encode(vaccine_bytes).decode("ascii")
    return f'''\nTARGET_STAGE = "{TARGET_STAGE}"\nHAI_TRANSFER_SOURCE_BLOB_SHA = "{SCIENCE_BLOB}"\nHAI_TRANSFER_SOURCE_SHA256 = "{science_sha256}"\nHAI_TRANSFER_SOURCE = {science_source!r}\nSEQUENCE_REFERENCE_SHA256 = "{SEQUENCE_SHA256}"\nVACCINE_REFERENCE_SHA256 = "{VACCINE_SHA256}"\nSEQUENCE_REFERENCE_B64 = "{sequence_b64}"\nVACCINE_REFERENCE_B64 = "{vaccine_b64}"\n\ndef hai_json_safe(value: Any) -> Any:\n    if value is None or isinstance(value, (str, bool, int)):\n        return value\n    if isinstance(value, float):\n        return value if math.isfinite(value) else None\n    if isinstance(value, Mapping):\n        return {{str(key): hai_json_safe(item) for key, item in value.items()}}\n    if isinstance(value, (list, tuple)):\n        return [hai_json_safe(item) for item in value]\n    item = getattr(value, "item", None)\n    if callable(item):\n        try:\n            return hai_json_safe(item())\n        except (ValueError, TypeError):\n            pass\n    return str(value)\n\ndef materialize_locked_reference(encoded: str, expected_sha256: str, path: Path) -> None:\n    data = base64.b64decode(encoded.encode("ascii"), validate=True)\n    if hashlib.sha256(data).hexdigest() != expected_sha256:\n        raise BundleContractError(f"locked organizer reference SHA-256 mismatch: {{path.name}}")\n    path.write_bytes(data)\n'''


def execute_source() -> str:
    return r'''
def execute(input_dir: Path, output_dir: Path) -> Mapping[str, Any]:
    stage = "initialize"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = output_dir / "runtime"
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True)

    try:
        stage = "materialize_package"
        bundle = package_bytes()
        package_path = runtime_root / "cmi_flu_bundle.zip"
        package_path.write_bytes(bundle)
        sys.path.insert(0, str(package_path))

        stage = "prepare_runtime_tree"
        data_parent = runtime_root / "data"
        data_parent.mkdir(parents=True, exist_ok=True)
        (data_parent / "raw").symlink_to(input_dir, target_is_directory=True)
        reference_dir = (
            runtime_root
            / "external"
            / "google-drive"
            / "challenge-resources"
            / "reference_files"
        )
        vaccine_panel, challenge_panel = derive_reference_files(input_dir, reference_dir)
        sequence_path = reference_dir / "strain_sequences.csv"
        vaccine_path = reference_dir / "vaccine_strains_per_season.txt"
        materialize_locked_reference(SEQUENCE_REFERENCE_B64, SEQUENCE_REFERENCE_SHA256, sequence_path)
        materialize_locked_reference(VACCINE_REFERENCE_B64, VACCINE_REFERENCE_SHA256, vaccine_path)

        config_text = CONFIG_TEXT
        if config_text.count("baseline: b02_taskwise_compact") != 1:
            raise BundleContractError("B2 base config baseline anchor mismatch")
        if config_text.count("filename: b02_taskwise_compact.csv") == 1:
            config_text = config_text.replace(
                "filename: b02_taskwise_compact.csv",
                "filename: b021_taskwise_robust.csv",
                1,
            )
        if "\nselection:\n" in config_text:
            raise BundleContractError("B2 base config unexpectedly already has selection section")
        config_text = config_text.rstrip() + "\nselection:\n  policy: robust_v1\n"
        config_dir = runtime_root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "baseline_b021_robust.yaml"
        config_path.write_text(config_text, encoding="utf-8")

        stage = "dependency_preflight"
        import joblib
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        import yaml
        _ = (
            joblib.__version__, np.__version__, pd.__version__, scipy.__version__,
            sklearn.__version__, yaml.__version__,
        )

        stage = "install_b21_adapter"
        adapter_namespace: dict[str, Any] = {}
        exec(
            compile(B21_ADAPTER_SOURCE, "<b21_runtime_adapter>", "exec"),
            adapter_namespace,
            adapter_namespace,
        )
        install_adapter = adapter_namespace.get("install")
        if not callable(install_adapter):
            raise BundleContractError("B2.1 runtime adapter lacks install()")
        install_adapter()

        stage = "load_hai_transfer"
        import types
        hai_module = types.ModuleType("cmi_flu.hai_transfer")
        hai_module.__file__ = "<cmi_flu.hai_transfer>"
        hai_module.__package__ = "cmi_flu"
        sys.modules["cmi_flu.hai_transfer"] = hai_module
        exec(
            compile(HAI_TRANSFER_SOURCE, "cmi_flu/hai_transfer.py", "exec"),
            hai_module.__dict__,
            hai_module.__dict__,
        )
        run_experiment = getattr(hai_module, "run_hai_strain_transfer_experiment", None)
        load_vaccine_reference = getattr(hai_module, "load_vaccine_strain_reference", None)
        if not callable(run_experiment) or not callable(load_vaccine_reference):
            raise BundleContractError("HAI transfer science entry points missing")
        if tuple(getattr(hai_module, "CONDITIONS", ())) != (
            "b21_reference",
            "ontology_metadata",
            "ontology_sequence_local",
            "ontology_sequence_target_domain",
        ):
            raise BundleContractError("HAI transfer condition contract mismatch")

        stage = "load_inputs"
        from cmi_flu.configuration import load_baseline_config
        from cmi_flu.runner import load_inputs
        config = load_baseline_config(config_path, repository_root=runtime_root)
        if config.baseline != "b02_taskwise_compact":
            raise BundleContractError("legacy embedded loader returned unexpected baseline")
        if str(config.section("selection").get("policy", "")) != "robust_v1":
            raise BundleContractError("runtime robust_v1 selection contract missing")
        raw_compat = dict(config.raw)
        raw_compat["baseline"] = "b021_taskwise_robust"
        object.__setattr__(config, "raw", raw_compat)
        object.__setattr__(config, "baseline", "b021_taskwise_robust")
        inputs = load_inputs(config)

        stage = "load_locked_references"
        sequence_reference = pd.read_csv(sequence_path)
        vaccine_reference = load_vaccine_reference(vaccine_path)
        if len(sequence_reference) != 352:
            raise BundleContractError(f"strain sequence row contract mismatch: {len(sequence_reference)}")
        if len(vaccine_reference) != 72:
            raise BundleContractError(f"vaccine reference row contract mismatch: {len(vaccine_reference)}")

        stage = "run_hai_transfer"
        metrics = dict(
            run_experiment(
                config,
                inputs,
                sequence_reference=sequence_reference,
                vaccine_reference=vaccine_reference,
            )
        )
        metrics.update(
            {
                "run_id": REQUEST_ID,
                "sequence_reference_sha256": SEQUENCE_REFERENCE_SHA256,
                "vaccine_reference_sha256": VACCINE_REFERENCE_SHA256,
                "checksum": (
                    {
                        "verified_count": len(inputs.checksum_report.verified),
                        "skipped": list(inputs.checksum_report.skipped),
                    }
                    if inputs.checksum_report is not None
                    else None
                ),
            }
        )
        metrics = hai_json_safe(metrics)

        stage = "validate_hai_transfer"
        if metrics.get("experiment") != TARGET_STAGE:
            raise BundleContractError("HAI experiment identity mismatch")
        if metrics.get("leaderboard_used_for_selection") is not False:
            raise BundleContractError("HAI unexpectedly used leaderboard")
        if metrics.get("competition_submission_attempted") is not False:
            raise BundleContractError("HAI unexpectedly attempted Competition submission")
        expected_policy = "aggregate_only_no_participant_ids_or_row_level_predictions_public_strain_names_allowed"
        if metrics.get("output_policy") != expected_policy:
            raise BundleContractError("HAI output-policy mismatch")
        if metrics.get("sequence_reference_rows") != 352:
            raise BundleContractError("HAI sequence row count mismatch")
        if metrics.get("vaccine_reference_rows") != 72:
            raise BundleContractError("HAI vaccine row count mismatch")

        conditions = metrics.get("conditions")
        expected_conditions = {
            "b21_reference",
            "ontology_metadata",
            "ontology_sequence_local",
            "ontology_sequence_target_domain",
        }
        if not isinstance(conditions, dict) or set(conditions) != expected_conditions:
            raise BundleContractError("HAI condition set mismatch")
        for condition, task_items in conditions.items():
            if not isinstance(task_items, dict) or set(task_items) != {"Task2.1", "Task2.2", "Task2.3"}:
                raise BundleContractError(f"HAI task set mismatch: {condition}")
            for task, item in task_items.items():
                selected = item.get("selected_model") if isinstance(item, dict) else None
                if not isinstance(selected, dict) or not str(selected.get("name", "")):
                    raise BundleContractError(f"HAI selected model missing: {condition}/{task}")
                task_metrics = item.get("metrics") or {}
                held = task_metrics.get("panel_proxy_fold_summary") or {}
                if int(held.get("count", 0)) <= 0:
                    raise BundleContractError(f"HAI held-study summary empty: {condition}/{task}")
                stress = item.get("stress")
                if not isinstance(stress, dict):
                    raise BundleContractError(f"HAI stress results missing: {condition}/{task}")
                if "purged_leave_one_strain_out" not in stress or "purged_leave_one_vaccine_season_out" not in stress:
                    raise BundleContractError(f"HAI stress keys missing: {condition}/{task}")

        agreements = metrics.get("challenge_rank_agreements")
        if not isinstance(agreements, dict) or set(agreements) != expected_conditions - {"b21_reference"}:
            raise BundleContractError("HAI challenge agreement condition set mismatch")
        for condition, task_items in agreements.items():
            if not isinstance(task_items, dict) or set(task_items) != {"Task2.1", "Task2.2", "Task2.3"}:
                raise BundleContractError(f"HAI challenge agreement task set mismatch: {condition}")
            for task, item in task_items.items():
                if int(item.get("n", -1)) != 40:
                    raise BundleContractError(f"HAI challenge donor count mismatch: {condition}/{task}")
                value = (item.get("rank_spearman") or {}).get("value")
                if value is None or not math.isfinite(float(value)):
                    raise BundleContractError(f"HAI challenge rank agreement non-finite: {condition}/{task}")

        promotion = metrics.get("promotion")
        if not isinstance(promotion, dict) or set(promotion) != {"Task2.1", "Task2.2", "Task2.3"}:
            raise BundleContractError("HAI promotion task set mismatch")
        for task, tracks in promotion.items():
            if not isinstance(tracks, dict) or set(tracks) != {"scientific_sequence_local", "competition_target_domain"}:
                raise BundleContractError(f"HAI promotion tracks mismatch: {task}")
            for track, evidence in tracks.items():
                if not isinstance(evidence, dict) or not isinstance(evidence.get("passed"), bool):
                    raise BundleContractError(f"HAI promotion flag missing: {task}/{track}")

        neighbors = metrics.get("challenge_only_sequence_neighbors")
        if not isinstance(neighbors, list) or len(neighbors) != 3:
            raise BundleContractError("HAI challenge-only strain neighbor contract mismatch")
        if any(not item.get("sequence_available") for item in neighbors):
            raise BundleContractError("HAI challenge-only strain lacks sequence reference")

        serialized = json.dumps(metrics, sort_keys=True, ensure_ascii=False)
        if '"participant_id"' in serialized or '"prediction"' in serialized:
            raise BundleContractError("HAI aggregate output leaked participant or row-level prediction fields")

        final_metrics = output_dir / "metrics.json"
        final_summary = output_dir / "summary.md"
        final_metrics.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        lines = [
            "# Phase A HAI strain/domain-transfer diagnostics",
            "",
            f"- request_id: `{REQUEST_ID}`",
            f"- science_source_commit: `{SOURCE_COMMIT}`",
            f"- sequence_reference_rows: `{metrics['sequence_reference_rows']}`",
            f"- sequence_lookup_strains: `{metrics['sequence_lookup_strains']}`",
            f"- vaccine_reference_rows: `{metrics['vaccine_reference_rows']}`",
            "- Competition submission: none",
            "",
        ]
        for condition in (
            "b21_reference",
            "ontology_metadata",
            "ontology_sequence_local",
            "ontology_sequence_target_domain",
        ):
            lines.extend([f"## {condition}", ""])
            for task in ("Task2.1", "Task2.2", "Task2.3"):
                item = conditions[condition][task]
                held = item["metrics"]["panel_proxy_fold_summary"]
                strain = item["stress"].get("purged_leave_one_strain_out", {})
                strain_fold = strain.get("panel_proxy_fold_summary", {}) if isinstance(strain, dict) else {}
                lines.append(
                    f"- {task}: model=`{item['selected_model']['name']}`, "
                    f"held_mean=`{held.get('spearman_mean')}`, held_min=`{held.get('spearman_min')}`, "
                    f"purged_strain_mean=`{strain_fold.get('spearman_mean')}`"
                )
            lines.append("")
        lines.extend(["## Promotion", ""])
        for task in ("Task2.1", "Task2.2", "Task2.3"):
            local = promotion[task]["scientific_sequence_local"]
            target = promotion[task]["competition_target_domain"]
            lines.append(
                f"- {task}: local_sequence=`{local['passed']}`, target_domain=`{target['passed']}`"
            )
        lines.extend(
            [
                "",
                "Output is aggregate-only; public strain names are retained, but no participant IDs or row-level predictions are written.",
            ]
        )
        final_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

        stage = "finalize_manifest"
        payload = {
            "schema_version": 1,
            "request_id": REQUEST_ID,
            "competition": COMPETITION,
            "kernel_stage": TARGET_STAGE,
            "source_commit": SOURCE_COMMIT,
            "base_package_zip_sha256": PACKAGE_ZIP_SHA256,
            "hai_transfer_source_blob_sha": HAI_TRANSFER_SOURCE_BLOB_SHA,
            "hai_transfer_source_sha256": HAI_TRANSFER_SOURCE_SHA256,
            "b21_runtime_adapter_sha256": B21_ADAPTER_SHA256,
            "b21_runtime_adapter_blob_sha": B21_ADAPTER_BLOB_SHA,
            "sequence_reference_sha256": SEQUENCE_REFERENCE_SHA256,
            "vaccine_reference_sha256": VACCINE_REFERENCE_SHA256,
            "python_version": platform.python_version(),
            "metrics_sha256": sha256_file(final_metrics),
            "derived_panel_sizes": {"vaccine": len(vaccine_panel), "challenge": len(challenge_panel)},
            "sequence_reference_rows": metrics["sequence_reference_rows"],
            "sequence_lookup_strains": metrics["sequence_lookup_strains"],
            "vaccine_reference_rows": metrics["vaccine_reference_rows"],
            "promotion": promotion,
            "challenge_rank_agreements": agreements,
            "challenge_only_sequence_neighbors": neighbors,
            "condition_summaries": {
                condition: {
                    task: {
                        "selected_model": item["selected_model"],
                        "metrics": item["metrics"],
                        "stress": item["stress"],
                    }
                    for task, item in tasks.items()
                }
                for condition, tasks in conditions.items()
            },
            "checksum": metrics.get("checksum"),
            "competition_submission_attempted": False,
        }
        (output_dir / "bridge-result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(runtime_root, ignore_errors=True)
        promoted = [
            f"{task}:{track}"
            for task, tracks in promotion.items()
            for track, evidence in tracks.items()
            if evidence.get("passed") is True
        ]
        print(
            "CMI_FLU_HAI_TRANSFER_COMPLETE "
            f"request_id={REQUEST_ID} promoted={','.join(promoted) if promoted else 'none'}"
        )
        return payload
    except Exception as error:
        safe_failure(output_dir, stage=stage, error=error)
        shutil.rmtree(runtime_root, ignore_errors=True)
        traceback.print_exc()
        raise
'''.lstrip("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=pathlib.Path, required=True)
    parser.add_argument("--sequence-reference", type=pathlib.Path, required=True)
    parser.add_argument("--vaccine-reference", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    validate_request(root)
    science_source, science_sha256 = load_science(root)
    sequence_bytes = args.sequence_reference.read_bytes()
    vaccine_bytes = args.vaccine_reference.read_bytes()
    if sha256_bytes(sequence_bytes) != SEQUENCE_SHA256:
        raise SystemExit("strain_sequences.csv SHA-256 differs from locked organizer reference")
    if sha256_bytes(vaccine_bytes) != VACCINE_SHA256:
        raise SystemExit("vaccine_strains_per_season.txt SHA-256 differs from locked organizer reference")

    adapter_path = root / "scripts/cmi_flu_b21_runtime_adapter.py"
    if git_blob_sha(adapter_path.read_bytes()) != B21_ADAPTER_BLOB:
        raise SystemExit("B2.1 runtime adapter blob mismatch")

    base_dir = root / "payloads/cmi-flu-b2-broad-004"
    base = b"".join((base_dir / f"part-{index:02d}").read_bytes() for index in range(20))
    if len(base) != BASE_SIZE or sha256_bytes(base) != BASE_SHA256:
        raise SystemExit("B2 004 base payload mismatch")

    work = output.parent
    base_source = work / "base-source.py"
    base004 = work / "base004.py"
    b21 = work / "b21.py"
    base_source.write_bytes(base)
    run(
        sys.executable,
        str(root / "scripts/cmi_flu_b2_patch_v4.py"),
        "--source",
        str(base_source),
        "--request",
        str(root / "requests/cmi-flu-b2-launch-v4.json"),
        "--output",
        str(base004),
    )
    run(
        sys.executable,
        str(root / "scripts/cmi_flu_b21_patch.py"),
        "--source",
        str(base004),
        "--adapter",
        str(adapter_path),
        "--request",
        str(root / "requests/cmi-flu-b21-launch-v1.json"),
        "--output",
        str(b21),
    )

    text = b21.read_text(encoding="utf-8")
    if "B21_ADAPTER_SOURCE" not in text:
        raise SystemExit("HAI source must be materialized B2.1 runtime")
    text = replace_once(
        text,
        f'REQUEST_ID = "{B21_BASE_REQUEST_ID}"',
        f'REQUEST_ID = "{REQUEST_ID}"',
        label="request id",
    )
    text = replace_once(
        text,
        f'SOURCE_COMMIT = "{B21_BASE_SOURCE_COMMIT}"',
        f'SOURCE_COMMIT = "{SCIENCE_COMMIT}"',
        label="science source commit",
    )
    text = replace_once(
        text,
        'default=Path("/kaggle/working/cmi-flu-b2")',
        'default=Path("/kaggle/working/cmi-flu-hai-transfer")',
        label="output directory",
    )
    text = text.replace("CMI_FLU_B2_FAILED ", "CMI_FLU_HAI_TRANSFER_FAILED ")

    marker = "\ndef execute(input_dir: Path, output_dir: Path) -> Mapping[str, Any]:\n"
    text = replace_once(
        text,
        marker,
        injected_helpers(science_source, science_sha256, sequence_bytes, vaccine_bytes) + marker,
        label="HAI source/reference insertion",
    )
    execute_start = text.index(marker.lstrip("\n"))
    main_marker = "\ndef main() -> int:\n"
    main_start = text.index(main_marker, execute_start)
    text = text[:execute_start] + execute_source() + text[main_start:]

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)

    namespace: dict[str, Any] = {"__name__": "hai_transfer_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("generated HAI runtime request identity mismatch")
    if namespace.get("SOURCE_COMMIT") != SCIENCE_COMMIT:
        raise SystemExit("generated HAI runtime science commit mismatch")
    if namespace.get("TARGET_STAGE") != TARGET_STAGE:
        raise SystemExit("generated HAI runtime stage mismatch")
    if namespace.get("HAI_TRANSFER_SOURCE_BLOB_SHA") != SCIENCE_BLOB:
        raise SystemExit("generated HAI runtime science blob mismatch")
    if namespace.get("SEQUENCE_REFERENCE_SHA256") != SEQUENCE_SHA256:
        raise SystemExit("generated HAI runtime sequence reference mismatch")
    if namespace.get("VACCINE_REFERENCE_SHA256") != VACCINE_SHA256:
        raise SystemExit("generated HAI runtime vaccine reference mismatch")
    if "kaggle competitions submit" in text or "api.competition_submit" in text:
        raise SystemExit("generated HAI runtime contains forbidden Competition submission path")
    if text.count("shutil.rmtree(runtime_root, ignore_errors=True)") != 2:
        raise SystemExit("runtime scratch cleanup must cover success and failure paths")

    run(sys.executable, str(output), "--self-test")
    print(
        "CMI_FLU_HAI_TRANSFER_PREPARE PASS "
        f"science_commit={SCIENCE_COMMIT} science_blob={SCIENCE_BLOB} "
        f"science_sha256={science_sha256} sequence_sha256={SEQUENCE_SHA256} "
        f"vaccine_sha256={VACCINE_SHA256} target_kernel={TARGET_KERNEL} expected_version=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
