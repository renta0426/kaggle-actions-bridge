#!/usr/bin/env python3
"""CMI-Flu B1 Same-readout Anchor runner for a private Kaggle CPU kernel."""

from __future__ import annotations

import hashlib
import html.parser
import http.cookiejar
import json
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

COMPETITION = "cmi-flu-first-prediction-challenge"
CMI_REPOSITORY = "https://github.com/renta0426/CMI-Flu-Invited-Prediction-Challenge.git"
CMI_COMMIT = "1e075cfa565698f708e872d22be629f97704a24f"
REFERENCE_FOLDER_ID = "1TFDWZCxGmPauW_CQW4yy94lR9opWz4mO"
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
MAX_REFERENCE_FILE_BYTES = 4_000_000
MAX_REFERENCE_TOTAL_BYTES = 5_000_000
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


def run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
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


def require_kaggle_runtime_packages() -> dict[str, str]:
    """Use Kaggle's preinstalled DS stack; do not install packages dynamically."""

    import importlib
    import importlib.metadata

    modules = {
        "joblib": "joblib",
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "yaml": "PyYAML",
    }
    versions: dict[str, str] = {}
    for module, distribution in modules.items():
        importlib.import_module(module)
        versions[distribution] = importlib.metadata.version(distribution)
    return versions


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


class _DriveFolderParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._href: str | None = None
        self._text: list[str] = []
        self.files: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._href is None:
            return
        match = re.fullmatch(
            r"https://drive\.google\.com/file/d/([-\w]{25,})/view(?:\?.*)?",
            self._href,
        )
        if match:
            name = "".join(self._text).strip()
            if name:
                self.files.append((match.group(1), name))
        self._href = None
        self._text = []


def _drive_opener() -> urllib.request.OpenerDirector:
    cookies = http.cookiejar.CookieJar()
    context = ssl.create_default_context()
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cookies),
        urllib.request.HTTPSHandler(context=context),
    )


def download_reference_files(repo: Path) -> None:
    """Download the fixed small organizer reference folder with stdlib only."""

    destination = repo / "external" / "google-drive" / "challenge-resources" / "reference_files"
    destination.mkdir(parents=True, exist_ok=True)
    opener = _drive_opener()
    user_agent = "Mozilla/5.0 kaggle-actions-bridge-cmi-flu-b1/1"
    folder_url = (
        "https://drive.google.com/embeddedfolderview?"
        + urllib.parse.urlencode({"id": REFERENCE_FOLDER_ID})
    )
    request = urllib.request.Request(folder_url, headers={"User-Agent": user_agent})
    with opener.open(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"reference folder status={response.status}")
        html_bytes = response.read(1_000_001)
    if len(html_bytes) > 1_000_000:
        raise RuntimeError("reference folder HTML exceeded fixed byte budget")
    parser = _DriveFolderParser()
    parser.feed(html_bytes.decode("utf-8", errors="strict"))
    by_name = {name: file_id for file_id, name in parser.files if name in REFERENCE_SHA256}
    if set(by_name) != set(REFERENCE_SHA256):
        missing = sorted(set(REFERENCE_SHA256) - set(by_name))
        raise RuntimeError(f"reference folder listing mismatch; missing={missing}")

    total = 0
    for name in sorted(REFERENCE_SHA256):
        file_id = by_name[name]
        query = urllib.parse.urlencode({"export": "download", "id": file_id, "confirm": "t"})
        download_url = f"https://drive.google.com/uc?{query}"
        request = urllib.request.Request(download_url, headers={"User-Agent": user_agent})
        with opener.open(request, timeout=60) as response:
            data = response.read(MAX_REFERENCE_FILE_BYTES + 1)
        if len(data) > MAX_REFERENCE_FILE_BYTES:
            raise RuntimeError(f"reference file exceeded fixed byte budget: {name}")
        target = destination / name
        target.write_bytes(data)
        actual = hashlib.sha256(data).hexdigest()
        if actual != REFERENCE_SHA256[name]:
            raise RuntimeError(f"reference checksum mismatch: {name}")
        total += len(data)
    if total > MAX_REFERENCE_TOTAL_BYTES:
        raise RuntimeError(f"reference downloads exceeded fixed total byte budget: {total}")


def run_baseline(repo: Path, package_versions: dict[str, str]) -> dict:
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
        "package_versions": package_versions,
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
        package_versions = require_kaggle_runtime_packages()
        repo = materialize_repo(temp_root)
        attach_competition_data(repo)
        download_reference_files(repo)
        payload = run_baseline(repo, package_versions)
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
