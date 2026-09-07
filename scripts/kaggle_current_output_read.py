"""Read allowlisted successful current output; never substitute historical/latest.

Exact metadata is the identity authority. Search is intentionally not used.
A second metadata check prevents accepting a different current version after
the CLI download. Failed-run salvage is a separate approved operation.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from kaggle_exact_identity import REF_RE as KERNEL_RE, exact_metadata, validate_metadata, verify_current

FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MAX_ALLOWED_FILES = 32
MAX_SINGLE_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_DOWNLOAD_BYTES = 128 * 1024 * 1024


def _parse_allow(values: list[str]) -> dict[str, int]:
    if not values or len(values) > MAX_ALLOWED_FILES:
        raise ValueError("allow-file count is outside the bounded contract")
    result: dict[str, int] = {}
    for value in values:
        name, sep, raw_limit = value.partition(":")
        if not sep or not FILE_RE.fullmatch(name):
            raise ValueError("allow-file must be NAME:MAX_BYTES with a basename only")
        limit = int(raw_limit)
        if limit <= 0 or limit > MAX_SINGLE_FILE_BYTES:
            raise ValueError("allow-file byte limit is outside the bounded contract")
        if name in result:
            raise ValueError("duplicate allow-file entry")
        result[name] = limit
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_current_kernel(api, kernel: str, expected_version: int) -> str:
    return verify_current(api, kernel, expected_version)


def _is_transport_log(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return len(relative.parts) == 1 and path.suffix == ".log"


def read_current_output(*, kernel: str, expected_version: int, allow: dict[str, int], output_dir: Path) -> None:
    if not KERNEL_RE.fullmatch(kernel):
        raise ValueError("invalid kernel ref")
    if type(expected_version) is not int or not 0 < expected_version <= 100_000:
        raise ValueError("expected version is outside the bounded contract")
    if output_dir.exists():
        raise FileExistsError("output directory already exists")
    # Validate programmatic callers too, not just CLI arguments.
    allow = _parse_allow([f"{name}:{limit}" for name, limit in allow.items()])
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    status = _verify_current_kernel(api, kernel, expected_version)
    kaggle_cli = shutil.which("kaggle")
    if not kaggle_cli:
        raise RuntimeError("official Kaggle CLI is not on PATH")
    with tempfile.TemporaryDirectory(prefix="kaggle-current-output-") as tmp:
        download = Path(tmp) / "download"
        download.mkdir()
        completed = subprocess.run(
            [kaggle_cli, "kernels", "output", kernel, "-p", str(download)],
            check=False, capture_output=True, text=True, timeout=180, env=os.environ.copy(),
        )
        if completed.returncode != 0:
            digest = hashlib.sha256((completed.stdout + completed.stderr).encode()).hexdigest()
            raise RuntimeError(f"output_cli_failed rc={completed.returncode} diagnostic_sha256={digest}")
        validate_metadata(exact_metadata(api, kernel), kernel, expected_version)
        paths = list(download.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise RuntimeError("symlink_in_saved_output")
        files = [path for path in paths if path.is_file()]
        total_bytes = sum(path.stat().st_size for path in files)
        if total_bytes <= 0 or total_bytes > MAX_TOTAL_DOWNLOAD_BYTES:
            raise RuntimeError("download_byte_contract_violated")
        unexpected = [p for p in files if p.name not in allow and not _is_transport_log(p, download)]
        if unexpected:
            raise RuntimeError(f"unexpected_saved_output_count={len(unexpected)}")
        selected = {}
        for name, limit in allow.items():
            matches = [path for path in files if path.name == name]
            if len(matches) != 1:
                raise RuntimeError("allowlisted_output_missing_or_ambiguous")
            if not 0 < matches[0].stat().st_size <= limit:
                raise RuntimeError("allowlisted_output_size_invalid")
            selected[name] = matches[0]
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            for name, source in selected.items():
                shutil.copyfile(source, output_dir / name)
        except BaseException:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise
        summary = ",".join(f"{name}:{(output_dir / name).stat().st_size}:{_sha256(output_dir / name)}" for name in sorted(selected))
        print(f"KAGGLE_CURRENT_OUTPUT_READ PASS status={status} version={expected_version} files={len(selected)} download_bytes={total_bytes} outputs={summary}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--expected-version", type=int, required=True)
    parser.add_argument("--allow-file", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    read_current_output(kernel=args.kernel, expected_version=args.expected_version, allow=_parse_allow(args.allow_file), output_dir=args.output_dir)


if __name__ == "__main__":
    main()
