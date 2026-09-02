"""Validate the cached Stage 1 feature-shard inventory from Kaggle CLI CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import re


EXPECTED_TRAIN = {f"features.part{index:03d}.parquet" for index in range(40)}
EXPECTED_VALIDATION = {f"features.part{index:03d}.parquet" for index in range(20)}
PATH_PATTERN = re.compile(
    r"(?P<path>(?:[^,\s]+/)*(?:train_10k|validation_5k)/parts/"
    r"features\.part\d{3}\.parquet)$"
)


def extract_feature_paths(rows: list[list[str]]) -> set[str]:
    paths: set[str] = set()
    for row in rows:
        for raw_cell in row:
            cell = raw_cell.strip().strip('"').replace("\\", "/")
            match = PATH_PATTERN.search(cell)
            if match:
                paths.add(match.group("path").lstrip("./"))
    return paths


def validate_paths(paths: set[str]) -> dict[str, object]:
    train_paths = {
        path for path in paths
        if re.search(r"(?:^|/)train_10k/parts/features\.part\d{3}\.parquet$", path)
    }
    validation_paths = {
        path for path in paths
        if re.search(r"(?:^|/)validation_5k/parts/features\.part\d{3}\.parquet$", path)
    }
    train_names = {PurePosixPath(path).name for path in train_paths}
    validation_names = {PurePosixPath(path).name for path in validation_paths}

    if len(paths) != 60:
        raise RuntimeError(f"expected 60 unique full shard paths, found {len(paths)}")
    if train_names != EXPECTED_TRAIN:
        missing = sorted(EXPECTED_TRAIN - train_names)
        extra = sorted(train_names - EXPECTED_TRAIN)
        raise RuntimeError(
            f"train shard inventory mismatch: count={len(train_names)} "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    if validation_names != EXPECTED_VALIDATION:
        missing = sorted(EXPECTED_VALIDATION - validation_names)
        extra = sorted(validation_names - EXPECTED_VALIDATION)
        raise RuntimeError(
            f"validation shard inventory mismatch: count={len(validation_names)} "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    if train_paths & validation_paths:
        raise RuntimeError("train and validation full shard paths overlap")

    result = {
        "status": "pass",
        "full_paths": len(paths),
        "train_shards": len(train_paths),
        "validation_shards": len(validation_paths),
        "basename_collisions_expected": len(train_names & validation_names),
    }
    if result["basename_collisions_expected"] != 20:
        raise RuntimeError("expected 20 train/validation basename collisions")
    return result


def validate_csv(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    result = validate_paths(extract_feature_paths(rows))
    print(json.dumps(result, sort_keys=True))
    return result


def self_test() -> None:
    synthetic = {
        *(f"stage1_raw_fim_v1/train_10k/parts/features.part{i:03d}.parquet" for i in range(40)),
        *(f"stage1_raw_fim_v1/validation_5k/parts/features.part{i:03d}.parquet" for i in range(20)),
    }
    result = validate_paths(synthetic)
    if result != {
        "status": "pass",
        "full_paths": 60,
        "train_shards": 40,
        "validation_shards": 20,
        "basename_collisions_expected": 20,
    }:
        raise RuntimeError("full-path shard validator self-test changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.csv is not None:
        validate_csv(args.csv)
    else:
        parser.error("provide --csv or --self-test")


if __name__ == "__main__":
    main()
