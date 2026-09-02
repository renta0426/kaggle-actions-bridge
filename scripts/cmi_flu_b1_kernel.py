#!/usr/bin/env python3
"""CMI-Flu B1 Same-readout Anchor runner for a private Kaggle CPU kernel."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

COMPETITION = "cmi-flu-first-prediction-challenge"
CMI_REPOSITORY = "https://github.com/renta0426/CMI-Flu-Invited-Prediction-Challenge.git"
CMI_COMMIT = "1e075cfa565698f708e872d22be629f97704a24f"
REFERENCE_FOLDER_URL = "https://drive.google.com/drive/u/0/folders/1TFDWZCxGmPauW_CQW4yy94lR9opWz4mO"
REFERENCE_SHA256 = {
    "GRCh38.113_gene_annotations.tsv": "f940650262d77aa1a4019b5b7f65f52828b2b682e8348560840f94ecc97886d7",
    "all_challenge_virus_strains.txt": "5fff74a3787fe9db1e17e569fa8a677cbbbf981c5a834edfcd7c3a30a42dfd0e",
    "cytokine_name_map.csv": "453af19d588a4c1d0c0c313f0bdf7e413e61cdd1c43bc2cb94d204968ef5b79f",
    "flow_name_revised.csv": "aadafb8cfd6de388361b676d8082648154fa8b168b8b31a19847a729f4ebe21b",
    "hai_map.csv": "e94a73b9837f3187eb5086e20fdddedba4e596a1293b1539e36b8b251a40c255",
    "strain_sequences.csv": "63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887",
    "vaccine_strains_2025.txt": "f2022ec9f51d1ffc34d4c12d4a738125f8e8704b667a4adcd7f058a254bff2ad",
    "vaccine_strains_per_season.txt": "8f6c7116f37f29df0bb21d6049d82fa28b4e42b2d10ed9394a1ae6f926bd9f35",
}
MAX_REFERENCE_BYTES = 5_000_000
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_SUBMISSION = OUTPUT_DIR / "submission.csv"
OUTPUT_METADATA = OUTPUT_DIR / "bridge-result.json"
OUTPUT_SUMMARY = OUTPUT_DIR / "summary.md"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=600,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed at fixed stage: {command[0]} rc={completed.returncode}")


def find_competition_input() -> Path:
    root = Path("/kaggle/input")
    matches = [path.parent for path in root.rglob("sample_submission_part1.csv")]
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one CMI-Flu competition mount, found {len(matches)}")
    return matches[0]


def materialize_repo(work: Path) -> Path:
    repo = work / "repo"
    run(["git", "init", "-q", str(repo)])
    run(["git", "-C", str(repo), "remote", "add", "origin", CMI_REPOSITORY])
    run([
        "git", "-C", str(repo), "-c", "protocol.version=2",
        "fetch", "-q", "--depth=1", "origin", CMI_COMMIT,
    ])
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "FETCH_HEAD"],
        text=True,
        timeout=30,
    ).strip()
    if actual != CMI_COMMIT:
        raise RuntimeError("CMI-Flu repository commit mismatch")
    run([
        "git", "-C", str(repo), "-c", "core.hooksPath=/dev/null",
        "checkout", "-q", "--detach", "FETCH_HEAD",
    ])
    return repo


def install_dependencies(repo: Path) -> None:
    run([
        sys.executable, "-m", "pip", "install",
        "--disable-pip-version-check", "--no-input", "--quiet",
        "-r", str(repo / "requirements-modeling.txt"),
    ])
    run([
        sys.executable, "-m", "pip", "install",
        "--disable-pip-version-check", "--no-input", "--quiet",
        "gdown==6.1.0",
    ])


def attach_competition_data(repo: Path) -> None:
    source = find_competition_input()
    destination = repo / "data" / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    required = {
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
    }
    available = {path.name for path in source.iterdir() if path.is_file()}
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"required competition files missing: {missing}")
    for src in source.iterdir():
        if src.is_file():
            (destination / src.name).symlink_to(src)


def download_reference_files(repo: Path) -> None:
    destination = repo / "external" / "google-drive" / "challenge-resources" / "reference_files"
    destination.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "-m", "gdown", "--continue", "--folder",
        REFERENCE_FOLDER_URL, "--output", str(destination),
    ])
    found: dict[str, Path] = {}
    for path in destination.rglob("*"):
        if path.is_file() and path.name in REFERENCE_SHA256:
            found[path.name] = path
    if set(found) != set(REFERENCE_SHA256):
        missing = sorted(set(REFERENCE_SHA256) - set(found))
        raise RuntimeError(f"reference file set mismatch; missing={missing}")
    total = sum(path.stat().st_size for path in found.values())
    if total > MAX_REFERENCE_BYTES:
        raise RuntimeError(f"reference download exceeded fixed byte budget: {total}")
    for name, source in found.items():
        if source.parent != destination:
            target = destination / name
            if target.exists():
                target.unlink()
            shutil.move(str(source), target)
    for name, expected in REFERENCE_SHA256.items():
        actual = sha256(destination / name)
        if actual != expected:
            raise RuntimeError(f"reference checksum mismatch: {name}")


def run_baseline(repo: Path) -> dict:
    sys.path.insert(0, str(repo / "src"))
    from cmi_flu.audit import audit_summary, audit_training_targets, require_audit_pass
    from cmi_flu.configuration import load_baseline_config
    from cmi_flu.runner import load_inputs, run_b01

    config = load_baseline_config(
        repo / "configs" / "baseline_b01_anchor.yaml",
        repository_root=repo,
    )
    inputs = load_inputs(config)
    audits = audit_training_targets(
        public_cytokine=inputs.tables["public_cytokine"],
        public_flow=inputs.tables["public_flow"],
        public_serology=inputs.tables["public_serology"],
        vaccine_strains=inputs.vaccine_strains,
        challenge_strains=inputs.challenge_strains,
    )
    require_audit_pass(audits)
    audit_payload = audit_summary(audits)
    result = run_b01(config, inputs)
    shutil.copy2(result.submission_path, OUTPUT_SUBMISSION)
    shutil.copy2(result.summary_path, OUTPUT_SUMMARY)
    return {
        "schema_version": 1,
        "request_id": "20260902-cmi-flu-b1-001",
        "competition": COMPETITION,
        "kernel_stage": "b1_anchor",
        "cmi_commit": CMI_COMMIT,
        "python_version": sys.version.split()[0],
        "audit_passed": bool(audit_payload["passed"]),
        "audit_checks": {
            name: {
                "rows": item["rows"],
                "participants": item["participants"],
                "studies": item["studies"],
                "passed": item["passed"],
            }
            for name, item in audit_payload["checks"].items()
        },
        "run_id": result.run_id,
        "submission_rows": result.validation_report.rows,
        "submission_sha256": sha256(OUTPUT_SUBMISSION),
        "task_unique_counts": dict(result.validation_report.task_unique_counts),
        "minus99_tasks": list(result.validation_report.minus99_tasks),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="cmi-flu-b1-", dir="/tmp"))
    try:
        repo = materialize_repo(temp_root)
        install_dependencies(repo)
        attach_competition_data(repo)
        download_reference_files(repo)
        payload = run_baseline(repo)
        OUTPUT_METADATA.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "CMI_FLU_B1_COMPLETE "
            f"request_id={payload['request_id']} "
            f"rows={payload['submission_rows']} "
            f"audit_passed={payload['audit_passed']}"
        )
        return 0
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
