"""Strict read-only helper for a Kaggle Notebook's *current* output.

This helper deliberately does not support historical versions.  It first proves
that the target kernel's current version is exactly the approved version, then
performs one official ``kaggle kernels output`` call with captured diagnostics.
Only explicitly allowlisted final files are copied out of the temporary download.

New Notebook workflows are expected to keep transient material outside
``/kaggle/working``.  If the saved output contains unexpected files, this helper
fails closed without printing their names.
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

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest


KERNEL_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ACTIVE = ("RUNNING", "QUEUED", "PENDING")
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


def _verify_current_kernel(api: KaggleApi, kernel: str, expected_version: int) -> str:
    owner, slug = kernel.split("/", 1)
    discovered = api.kernels_list(user=owner, search=slug, page_size=10) or []
    refs = [str(getattr(item, "ref", "")) for item in discovered]
    if refs.count(kernel) != 1:
        raise RuntimeError(f"exact kernel discoverability mismatch count={refs.count(kernel)}")

    status = str(getattr(api.kernels_status(kernel), "status", "")).upper()
    if not status or any(token in status for token in ACTIVE):
        raise RuntimeError(f"kernel is not terminal: status={status or 'UNKNOWN'}")

    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name = owner
        request.kernel_slug = slug
        metadata = client.kernels.kernels_api_client.get_kernel(request).metadata

    if str(getattr(metadata, "ref", "")) != kernel:
        raise RuntimeError("kernel metadata identity mismatch")
    if not bool(getattr(metadata, "is_private", False)):
        raise RuntimeError("current-output helper requires the approved private-kernel contract")
    current_version = int(getattr(metadata, "current_version_number", 0) or 0)
    if current_version != expected_version:
        raise RuntimeError(
            f"current version mismatch: observed={current_version} expected={expected_version}; "
            "historical output substitution is forbidden"
        )
    return status


def _is_transport_log(path: Path, root: Path) -> bool:
    # The official CLI emits one root-level kernel log alongside session output.
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return len(relative.parts) == 1 and path.suffix == ".log"


def read_current_output(
    *,
    kernel: str,
    expected_version: int,
    allow: dict[str, int],
    output_dir: Path,
) -> None:
    if not KERNEL_RE.fullmatch(kernel):
        raise ValueError("invalid kernel ref")
    if expected_version <= 0 or expected_version > 100_000:
        raise ValueError("expected version is outside the bounded contract")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

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
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            # Do not echo CLI stdout/stderr: it can enumerate private output paths.
            raise RuntimeError(f"kaggle kernels output failed rc={completed.returncode}")

        files = [path for path in download.rglob("*") if path.is_file()]
        total_bytes = sum(path.stat().st_size for path in files)
        if total_bytes <= 0 or total_bytes > MAX_TOTAL_DOWNLOAD_BYTES:
            raise RuntimeError(f"download byte contract violated: bytes={total_bytes}")

        unexpected = [
            path
            for path in files
            if path.name not in allow and not _is_transport_log(path, download)
        ]
        if unexpected:
            # Count only.  Do not expose unexpected private file names in public logs.
            raise RuntimeError(f"unexpected saved output files count={len(unexpected)}")

        selected: dict[str, Path] = {}
        for name, limit in allow.items():
            matches = [path for path in files if path.name == name]
            if len(matches) != 1:
                raise RuntimeError(f"allowlisted output missing/ambiguous: {name} count={len(matches)}")
            size = matches[0].stat().st_size
            if size <= 0 or size > limit:
                raise RuntimeError(f"allowlisted output size invalid: {name} bytes={size}")
            selected[name] = matches[0]

        output_dir.mkdir(parents=True, exist_ok=False)
        for name, source in selected.items():
            shutil.copyfile(source, output_dir / name)

        summary = ",".join(
            f"{name}:{(output_dir / name).stat().st_size}:{_sha256(output_dir / name)}"
            for name in sorted(selected)
        )
        print(
            "KAGGLE_CURRENT_OUTPUT_READ PASS "
            f"status={status} version={expected_version} files={len(selected)} "
            f"download_bytes={total_bytes} outputs={summary}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--expected-version", type=int, required=True)
    parser.add_argument("--allow-file", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    allow = _parse_allow(args.allow_file)
    read_current_output(
        kernel=args.kernel,
        expected_version=args.expected_version,
        allow=allow,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
