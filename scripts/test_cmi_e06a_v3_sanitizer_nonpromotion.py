#!/usr/bin/env python3
"""Regression for aggregate sanitizer handling of rank-failed non-promoted candidates."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(source: Path, target: Path, *, inconsistent: bool) -> None:
    target.mkdir()
    for name in ("bridge-result.json", "metrics.json", "summary.md"):
        shutil.copyfile(source / name, target / name)
    metrics_path = target / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    condition = metrics["conditions"]["b21_reference"]
    candidate = condition["historical_nested_study_out"]["calibrated"]["positive_affine"]
    candidate["rank_preservation"]["preserved_within_tolerance"] = False
    candidate["rank_preservation"]["max_absolute_percentile_difference"] = 0.02564102564102564
    decision = condition["promotion"]["positive_affine"]
    decision["checks"]["held_study_rank_preserved"] = False
    decision["passed"] = bool(inconsistent)
    if metrics.get("selected_promoted_condition") == "b21_reference__positive_affine":
        metrics["selected_promoted_condition"] = None
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bridge_path = target / "bridge-result.json"
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    bridge["metrics_sha256"] = file_sha(metrics_path)
    bridge["summary_sha256"] = file_sha(target / "summary.md")
    bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--sanitizer", type=Path, required=True)
    args = p.parse_args()
    source = args.source_dir.resolve()
    sanitizer = args.sanitizer.resolve()
    with tempfile.TemporaryDirectory(prefix="e06a-v3-sanitize-") as tmp:
        root = Path(tmp)
        good = root / "good"
        bad = root / "bad"
        materialize(source, good, inconsistent=False)
        materialize(source, bad, inconsistent=True)
        accepted = subprocess.run(
            [sys.executable, str(sanitizer), "--input-dir", str(good)],
            capture_output=True,
            text=True,
            check=False,
        )
        if accepted.returncode != 0:
            raise AssertionError(accepted.stdout + accepted.stderr)
        rejected = subprocess.run(
            [sys.executable, str(sanitizer), "--input-dir", str(bad)],
            capture_output=True,
            text=True,
            check=False,
        )
        if rejected.returncode == 0:
            raise AssertionError("E06a v3 sanitizer accepted inconsistent promotion=true")
    print(
        "CMI_FLU_E06A_V3_SANITIZER_NONPROMOTION_PASS "
        "rank_failed_candidate_accepted=true promotion_false_required=true inconsistent_result_rejected=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
