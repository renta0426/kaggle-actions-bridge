#!/usr/bin/env python3
"""Validate E00 Kaggle outputs and emit only aggregate-safe public results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUEST_ID = "20260907-cmi-flu-strategy-e00-readiness-001"
SCIENCE_COMMIT = "2529b45249d6bf528593c6f4f6a445678dd3e7c2"
COMPETITION = "cmi-flu-first-prediction-challenge"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compact_split(source: dict) -> dict:
    count_fields = (
        "rows", "subjects", "rows_with_view", "rows_without_view",
        "feature_count", "rows_with_complete_view", "subjects_with_view",
    )
    result = {key: source[key] for key in count_fields if key in source}
    result["studies"] = [
        {
            key: row[key]
            for key in (
                "study", "rows", "subjects", "rows_with_view",
                "subjects_with_view", "rows_with_complete_view",
            )
        }
        for row in source.get("studies", [])
    ]
    return result


def main() -> int:
    args = parse_args()
    root = args.readout.expanduser().resolve()
    e00_path = root / "e00-readiness.json"
    bridge_path = root / "bridge-result.json"
    summary_path = root / "summary.md"
    if {p.name for p in root.iterdir() if p.is_file()} != {
        "e00-readiness.json", "bridge-result.json", "summary.md"
    }:
        raise SystemExit("readout file set differs from approved aggregate contract")
    report = json.loads(e00_path.read_text(encoding="utf-8"))
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    if bridge.get("request_id") != REQUEST_ID or bridge.get("science_commit") != SCIENCE_COMMIT:
        raise SystemExit("bridge provenance mismatch")
    if bridge.get("e00_sha256") != sha(e00_path) or bridge.get("summary_sha256") != sha(summary_path):
        raise SystemExit("bridge aggregate hash mismatch")
    if any((
        bridge.get("accelerator") != "cpu",
        bridge.get("internet_enabled") is not False,
        bridge.get("submission_created") is not False,
        bridge.get("submission_attempted") is not False,
        bridge.get("automatic_compute_retries") != 0,
    )):
        raise SystemExit("bridge execution contract mismatch")
    if any((
        report.get("audit") != "strategy_20260907_E00_partial",
        report.get("md5_loader_completed") is not True,
        report.get("contains_participant_identifiers") is not False,
        report.get("fitting_performed") is not False,
        report.get("submission_created") is not False,
        report.get("ready_for_science_launch") is not False,
    )):
        raise SystemExit("E00 privacy/science contract mismatch")

    safe_views = {}
    for name, item in report["views"].items():
        if item.get("status") == "no_candidate_columns":
            safe_views[name] = {"status": "no_candidate_columns"}
            continue
        safe_views[name] = {
            "train": compact_split(item["train"]),
            "challenge": compact_split(item["challenge"]),
            "purged_support": [
                {
                    key: row[key]
                    for key in (
                        "held_study", "held_rows", "held_rows_with_view",
                        "source_rows_before_purge", "source_rows_after_purge",
                        "source_rows_with_view_after_purge",
                        "source_subjects_with_view_after_purge",
                        "source_studies_with_view_after_purge",
                    )
                }
                for row in item.get("purged_support", [])
            ],
        }
    safe = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": COMPETITION,
        "science_commit": SCIENCE_COMMIT,
        "audit": report["audit"],
        "execution": {
            "accelerator": "cpu",
            "internet_enabled": False,
            "submission_created": False,
            "submission_attempted": False,
            "automatic_compute_retries": 0,
        },
        "md5_loader_completed": True,
        "captured_warning_count": int(report.get("captured_warning_count", 0)),
        "science_ready": False,
        "views": safe_views,
        "measurement_ambiguity": report["measurement_ambiguity"],
        "remaining_gates": list(report["remaining_gates"]),
        "source_hashes": {
            "config_sha256": report["config_sha256"],
            "helper_sha256": report["helper_sha256"],
            "e00_sha256": sha(e00_path),
            "summary_sha256": sha(summary_path),
        },
    }
    payload = json.dumps(safe, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    encoded = payload.encode("utf-8")
    if len(encoded) > 250_000:
        raise SystemExit("sanitized aggregate output exceeds byte budget")
    if args.output.exists():
        raise SystemExit("sanitized output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(
        "CMI_FLU_E00_PRIVACY_GATE PASS "
        f"safe_bytes={len(encoded)} safe_sha256={hashlib.sha256(encoded).hexdigest()}"
    )
    for name, item in sorted(safe_views.items()):
        if item.get("status") == "no_candidate_columns":
            print(f"E00_VIEW name={name} status=no_candidate_columns")
        else:
            print(
                f"E00_VIEW name={name} features={item['train'].get('feature_count')} "
                f"train_rows={item['train'].get('rows_with_view')}/{item['train'].get('rows')} "
                f"challenge_rows={item['challenge'].get('rows_with_view')}/{item['challenge'].get('rows')}"
            )
    for name, item in sorted(safe["measurement_ambiguity"].items()):
        print(
            f"E00_MEASUREMENT name={name} status={item.get('status')} rows={item.get('rows')} "
            f"multi_keys={item.get('keys_with_multiple_rows')} conflicts={item.get('keys_with_any_metadata_conflict')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
