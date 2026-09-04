#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath

import gdown

FOLDER_ID = "1D1kwYPO0tGAJOGlRVNwObT__CxIHH3cP"
EXPECTED = {
    "strain_sequences.csv": "63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887",
    "vaccine_strains_per_season.txt": "8f6c7116f37f29df0bb21d6049d82fa28b4e42b2d10ed9394a1ae6f926bd9f35",
}
MAX_BYTES = {
    "strain_sequences.csv": 1_000_000,
    "vaccine_strains_per_season.txt": 100_000,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    listing = gdown.download_folder(
        id=FOLDER_ID,
        output=str(output_dir / "listing-root"),
        quiet=True,
        use_cookies=False,
        remaining_ok=True,
        skip_download=True,
    )
    if listing is None:
        raise SystemExit("organizer Google Drive folder metadata could not be listed")

    selected = {}
    for item in listing:
        path = PurePosixPath(str(item.path))
        name = path.name
        if name not in EXPECTED:
            continue
        if "reference_files" not in {part.casefold() for part in path.parts[:-1]}:
            continue
        if name in selected:
            raise SystemExit(f"duplicate organizer reference path discovered: {name}")
        selected[name] = item
    if set(selected) != set(EXPECTED):
        raise SystemExit(f"organizer reference files not uniquely discovered: {sorted(selected)}")

    for name in sorted(EXPECTED):
        destination = output_dir / name
        result = gdown.download(
            id=str(selected[name].id),
            output=str(destination),
            quiet=True,
            use_cookies=False,
        )
        if result is None or not destination.is_file():
            raise SystemExit(f"organizer reference download failed: {name}")
        size = destination.stat().st_size
        if size <= 0 or size > MAX_BYTES[name]:
            raise SystemExit(f"organizer reference byte budget failed: {name} size={size}")
        digest = sha256_file(destination)
        if digest != EXPECTED[name]:
            raise SystemExit(f"organizer reference SHA-256 mismatch: {name} digest={digest}")
        print(f"LOCKED_HAI_REFERENCE PASS name={name} bytes={size} sha256={digest}")

    listing_root = output_dir / "listing-root"
    if listing_root.exists():
        raise SystemExit("skip_download unexpectedly materialized organizer folder content")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
