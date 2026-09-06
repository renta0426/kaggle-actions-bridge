#!/usr/bin/env python3
'Build the one-shot aggregate-only Kaggle runtime for CMI-Flu strategy E00.'
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

REQUEST_ID = "20260907-cmi-flu-strategy-e00-readiness-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
SCIENCE_COMMIT = "2529b45249d6bf528593c6f4f6a445678dd3e7c2"
EXPECTED_CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"
EXPECTED_AUDIT_BLOB = "f6f6d91b7d76838f7c1623b07c8d6daa8b71e64b"

REQUIRED_PACKAGE = {
    "cmi_flu/__init__.py",
    "cmi_flu/aliases.py",
    "cmi_flu/configuration.py",
    "cmi_flu/datasets.py",
    "cmi_flu/features/flow.py",
    "cmi_flu/runner.py",
    "cmi_flu/task11_prior_immunity.py",
}
PLACEHOLDER_REFERENCES = {
    "cytokine_name_map.csv": "source,target\n",
    "flow_name_revised.csv": "source,target\n",
    "hai_map.csv": "source,target\n",
    "strain_sequences.csv": "virus_strain,sequence\n",
    "vaccine_strains_per_season.txt": "# Unused by E00 readiness audit.\n",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science-root", type=Path, required=True)
    parser.add_argument("--science-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def deterministic_package(source_root: Path) -> bytes:
    package_root = source_root / "src" / "cmi_flu"
    if not package_root.is_dir():
        raise SystemExit("science package root is missing")
    files = sorted(path for path in package_root.rglob("*.py") if path.is_file())
    names = {path.relative_to(source_root / "src").as_posix() for path in files}
    if REQUIRED_PACKAGE - names:
        raise SystemExit("science package is missing required E00 modules")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            rel = path.relative_to(source_root / "src").as_posix()
            data = path.read_bytes()
            compile(data.decode("utf-8"), rel, "exec")
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def chunk64(data: bytes, width: int = 96) -> str:
    text = base64.b64encode(data).decode("ascii")
    return "\n".join(text[i:i + width] for i in range(0, len(text), width))


def build_runtime(package: bytes, config_text: str, audit_text: str, *, config_sha: str, audit_sha: str) -> str:
    template = r'''#!/usr/bin/env python3
'CMI-Flu strategy E00 aggregate readiness audit. No fitting or submission.'
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

REQUEST_ID = "__REQUEST_ID__"
COMPETITION = "__COMPETITION__"
SCIENCE_COMMIT = "__SCIENCE_COMMIT__"
PACKAGE_SHA256 = "__PACKAGE_SHA256__"
CONFIG_SHA256 = "__CONFIG_SHA256__"
AUDIT_SHA256 = "__AUDIT_SHA256__"
PACKAGE_B64 = __PACKAGE_B64_REPR__
CONFIG_TEXT = __CONFIG_TEXT__
AUDIT_TEXT = __AUDIT_TEXT__

CORE_FILES = (
    "participants.tsv",
    "investigations_260821.tsv",
    "publicData_cytokine.tsv",
    "publicData_ex_vivo_flow.tsv",
    "publicData_serology_260821.tsv",
    "2025LJI_aim.tsv",
    "2025LJI_cytokine.tsv",
    "2025LJI_ex_vivo_flow.tsv",
    "2025LJI_serology.tsv",
    "sample_submission_part1.csv",
    "md5sum",
)
REFERENCE_PLACEHOLDERS = __REFERENCE_PLACEHOLDERS__


class BridgeContractError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def package_bytes() -> bytes:
    try:
        data = base64.b64decode("".join(PACKAGE_B64.split()), validate=True)
    except Exception as exc:
        raise BridgeContractError("embedded_package_base64_invalid") from exc
    if hashlib.sha256(data).hexdigest() != PACKAGE_SHA256:
        raise BridgeContractError("embedded_package_sha_mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        bad = archive.testzip()
        names = set(archive.namelist())
    if bad is not None:
        raise BridgeContractError("embedded_package_corrupt")
    required = {
        "cmi_flu/runner.py",
        "cmi_flu/datasets.py",
        "cmi_flu/features/flow.py",
        "cmi_flu/task11_prior_immunity.py",
    }
    if required - names:
        raise BridgeContractError("embedded_package_missing_modules")
    return data


def self_test() -> int:
    package = package_bytes()
    if hashlib.sha256(CONFIG_TEXT.encode("utf-8")).hexdigest() != CONFIG_SHA256:
        raise BridgeContractError("embedded_config_sha_mismatch")
    if hashlib.sha256(AUDIT_TEXT.encode("utf-8")).hexdigest() != AUDIT_SHA256:
        raise BridgeContractError("embedded_audit_sha_mismatch")
    compile(AUDIT_TEXT, "audit_strategy_readiness.py", "exec")
    print(
        "CMI_FLU_E00_RUNTIME_SELF_TEST PASS "
        f"request_id={REQUEST_ID} package_bytes={len(package)} "
        f"package_sha256={PACKAGE_SHA256} science_commit={SCIENCE_COMMIT}"
    )
    return 0


def locate_competition_data(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    exact = Path("/kaggle/input") / COMPETITION
    candidates.append(exact)
    root = Path("/kaggle/input")
    if root.is_dir():
        try:
            candidates.extend(sorted(p.parent for p in root.rglob("sample_submission_part1.csv")))
        except OSError:
            pass
    seen: set[Path] = set()
    valid: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        if all((resolved / name).is_file() for name in CORE_FILES):
            valid.append(resolved)
    if not valid:
        raise BridgeContractError("competition_mount_not_found")
    try:
        exact_resolved = exact.resolve()
    except OSError:
        exact_resolved = exact
    if exact_resolved in valid:
        return exact_resolved
    if explicit is not None:
        chosen = explicit.expanduser().resolve()
        if chosen in valid:
            return chosen
    unique = list(dict.fromkeys(valid))
    if len(unique) != 1:
        raise BridgeContractError("competition_mount_ambiguous")
    return unique[0]


def _vaccine_flag(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().casefold()
    return text in {"1", "1.0", "true", "yes", "y"}


def _canonical_strain(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    raw = re.sub(r"(?i)_cell$", "_MDCK", raw)
    raw = re.sub(r"(?i)\s+cell$", "_MDCK", raw)
    return raw


def derive_reference_files(input_dir: Path, external_root: Path) -> tuple[int, int]:
    import pandas as pd
    ref = external_root / "google-drive" / "challenge-resources" / "reference_files"
    ref.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(
        input_dir / "2025LJI_serology.tsv",
        sep="\t",
        usecols=["assay", "virus_strain", "virus_in_vaccine"],
        low_memory=False,
    )
    hai = source.loc[source["assay"].fillna("").astype(str).str.strip().str.casefold().eq("hai")].copy()
    hai["virus_strain"] = hai["virus_strain"].map(_canonical_strain)
    challenge = tuple(sorted(v for v in hai["virus_strain"].dropna().unique() if v))
    vaccine = tuple(sorted(v for v in hai.loc[hai["virus_in_vaccine"].map(_vaccine_flag), "virus_strain"].dropna().unique() if v))
    if len(challenge) != 12 or len(vaccine) != 3 or not set(vaccine).issubset(challenge):
        raise BridgeContractError("derived_hai_panel_contract_changed")
    (ref / "all_challenge_virus_strains.txt").write_text("\n".join(challenge) + "\n", encoding="utf-8")
    (ref / "vaccine_strains_2025.txt").write_text("\n".join(vaccine) + "\n", encoding="utf-8")
    for name, content in REFERENCE_PLACEHOLDERS.items():
        (ref / name).write_text(content, encoding="utf-8")
    return len(vaccine), len(challenge)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_summary(report: dict) -> str:
    lines = [
        "# CMI-Flu strategy E00 readiness audit",
        "",
        "Aggregate-only readiness audit. No fitting and no Kaggle submission were performed.",
        "",
        f"- science commit: `{SCIENCE_COMMIT}`",
        f"- audit: `{report.get('audit')}`",
        f"- MD5 loader completed: `{report.get('md5_loader_completed')}`",
        f"- science ready: `{report.get('ready_for_science_launch')}`",
        f"- captured warnings: `{report.get('captured_warning_count')}`",
        "",
        "## Feature-view coverage",
        "",
    ]
    for name, item in sorted(report.get("views", {}).items()):
        if item.get("status") == "no_candidate_columns":
            lines.append(f"- {name}: no candidate columns")
            continue
        train = item["train"]
        challenge = item["challenge"]
        lines.append(
            f"- {name}: features={train['feature_count']}; "
            f"train rows={train['rows_with_view']}/{train['rows']}; "
            f"challenge rows={challenge['rows_with_view']}/{challenge['rows']}"
        )
    lines.extend(["", "## Measurement-key audit", ""])
    for name, item in sorted(report.get("measurement_ambiguity", {}).items()):
        lines.append(
            f"- {name}: status={item['status']}; rows={item['rows']}; "
            f"multi-row keys={item['keys_with_multiple_rows']}; "
            f"metadata-conflict keys={item['keys_with_any_metadata_conflict']}"
        )
    lines.extend(["", "## Remaining gates", ""])
    for gate in report.get("remaining_gates", []):
        lines.append(f"- {gate}")
    lines.append("")
    return "\n".join(lines)


def safe_failure(stage: str, exc: BaseException) -> None:
    code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode("utf-8", errors="replace")).hexdigest()[:20]
    print(
        "CMI_FLU_E00_FAILED "
        f"stage={stage} exception_type={type(exc).__name__} error_code={code}",
        file=sys.stderr,
    )


def execute(input_dir: Path) -> int:
    stage = "initialize"
    working = Path("/kaggle/working")
    e00_path = working / "e00-readiness.json"
    bridge_path = working / "bridge-result.json"
    summary_path = working / "summary.md"
    for path in (e00_path, bridge_path, summary_path):
        if path.exists():
            raise BridgeContractError("approved_output_already_exists")
    try:
        package = package_bytes()
        stage = "runtime_tree"
        with tempfile.TemporaryDirectory(prefix="cmi-flu-e00-") as tmp:
            runtime = Path(tmp)
            package_path = runtime / "cmi_flu_bundle.zip"
            package_path.write_bytes(package)
            sys.path.insert(0, str(package_path))
            config_dir = runtime / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "baseline_b021_robust.yaml"
            config_path.write_text(CONFIG_TEXT, encoding="utf-8")
            helper_dir = runtime / "scripts"
            helper_dir.mkdir(parents=True)
            helper_path = helper_dir / "audit_strategy_readiness.py"
            helper_path.write_text(AUDIT_TEXT, encoding="utf-8")
            external_root = runtime / "external"
            vaccine_count, challenge_count = derive_reference_files(input_dir, external_root)

            stage = "dependency_preflight"
            import joblib, numpy, pandas, scipy, sklearn, yaml
            _ = (joblib.__version__, numpy.__version__, pandas.__version__, scipy.__version__, sklearn.__version__, yaml.__version__)

            stage = "run_science_e00"
            spec = importlib.util.spec_from_file_location("cmi_flu_strategy_e00", helper_path)
            if spec is None or spec.loader is None:
                raise BridgeContractError("audit_helper_import_failed")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            rc = int(module.main([
                "--config", str(config_path),
                "--data-dir", str(input_dir),
                "--external-dir", str(external_root),
                "--output", str(e00_path),
            ]))
            if rc != 0 or not e00_path.is_file():
                raise BridgeContractError("science_e00_failed")

            stage = "validate_science_e00"
            report = json.loads(e00_path.read_text(encoding="utf-8"))
            checks = {
                "audit": report.get("audit") == "strategy_20260907_E00_partial",
                "md5": report.get("md5_loader_completed") is True,
                "ids": report.get("contains_participant_identifiers") is False,
                "fit": report.get("fitting_performed") is False,
                "submit": report.get("submission_created") is False,
                "ready": report.get("ready_for_science_launch") is False,
                "config_sha": report.get("config_sha256") == CONFIG_SHA256,
                "helper_sha": report.get("helper_sha256") == AUDIT_SHA256,
            }
            if not all(checks.values()):
                raise BridgeContractError("science_e00_contract_mismatch")
            if set(report.get("views", {})) != {
                "all_shared_flow", "innate_name_candidates_not_final_panel", "historical_hai_prior"
            }:
                raise BridgeContractError("science_e00_view_contract_mismatch")

            stage = "write_bridge_outputs"
            summary_path.write_text(render_summary(report), encoding="utf-8")
            bridge = {
                "schema_version": 1,
                "request_id": REQUEST_ID,
                "competition": COMPETITION,
                "science_commit": SCIENCE_COMMIT,
                "accelerator": "cpu",
                "internet_enabled": False,
                "submission_created": False,
                "submission_attempted": False,
                "automatic_compute_retries": 0,
                "vaccine_strain_count": vaccine_count,
                "challenge_strain_count": challenge_count,
                "e00_sha256": sha256_file(e00_path),
                "summary_sha256": sha256_file(summary_path),
                "science_ready": False,
            }
            bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "CMI_FLU_E00_PASS "
            f"request_id={REQUEST_ID} accelerator=cpu internet=false "
            "submission=false science_ready=false"
        )
        return 0
    except Exception as exc:
        for path in (bridge_path, summary_path):
            path.unlink(missing_ok=True)
        e00_path.unlink(missing_ok=True)
        safe_failure(stage, exc)
        return 2


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    try:
        input_dir = locate_competition_data(args.input_dir)
    except Exception as exc:
        safe_failure("locate_competition_data", exc)
        return 2
    return execute(input_dir)


if __name__ == "__main__":
    raise SystemExit(main())
'''
    replacements = {
        "__REQUEST_ID__": REQUEST_ID,
        "__COMPETITION__": COMPETITION,
        "__SCIENCE_COMMIT__": SCIENCE_COMMIT,
        "__PACKAGE_SHA256__": sha256(package),
        "__CONFIG_SHA256__": config_sha,
        "__AUDIT_SHA256__": audit_sha,
        "__PACKAGE_B64_REPR__": repr(chunk64(package)),
        "__CONFIG_TEXT__": repr(config_text),
        "__AUDIT_TEXT__": repr(audit_text),
        "__REFERENCE_PLACEHOLDERS__": repr(PLACEHOLDER_REFERENCES),
    }
    for old, new in replacements.items():
        if old not in template:
            raise SystemExit(f"missing runtime placeholder: {old}")
        template = template.replace(old, new)
    compile(template, "generated_cmi_flu_e00.py", "exec")
    return template


def main() -> int:
    args = parse_args()
    root = args.science_root.expanduser().resolve()
    if args.science_commit != SCIENCE_COMMIT:
        raise SystemExit("science commit argument differs from approved E00 commit")
    config_path = root / "configs" / "baseline_b021_robust.yaml"
    audit_path = root / "scripts" / "audit_strategy_readiness.py"
    config_data = config_path.read_bytes()
    audit_data = audit_path.read_bytes()
    if git_blob_sha(config_data) != EXPECTED_CONFIG_BLOB:
        raise SystemExit("science config blob mismatch")
    if git_blob_sha(audit_data) != EXPECTED_AUDIT_BLOB:
        raise SystemExit("science E00 helper blob mismatch")
    config_text = config_data.decode("utf-8")
    audit_text = audit_data.decode("utf-8")
    compile(audit_text, str(audit_path), "exec")
    if "baseline: b021_taskwise_robust" not in config_text or "verify_md5: true" not in config_text:
        raise SystemExit("science E00 config contract mismatch")
    package = deterministic_package(root)
    runtime = build_runtime(
        package,
        config_text,
        audit_text,
        config_sha=sha256(config_data),
        audit_sha=sha256(audit_data),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(runtime, encoding="utf-8")
    print(
        "CMI_FLU_E00_BUILD PASS "
        f"science_commit={SCIENCE_COMMIT} package_bytes={len(package)} "
        f"package_sha256={sha256(package)} runtime_bytes={len(runtime.encode('utf-8'))} "
        f"runtime_sha256={sha256(runtime.encode('utf-8'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
