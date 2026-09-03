"""Verify the frozen Stage 1 length-interaction output before submission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

REQUEST_ID = "20260903-poisoned-chalice-stage1-length-interactions-run-001"
METHOD = "plus_length_interactions_v1"
SOURCE_DATASET = "renta0426/stage1-raw-fim-submission-v1-output"
BUILDER_BLOB = "a10c7a76be1a4d700d7b1d9846e42348c5711651"
EXPECTED_OOF_AUC = 0.68013844
OOF_AUC_TOLERANCE = 0.002
EXPECTED_TPR_1PCT = 0.0682
TPR_1PCT_TOLERANCE = 0.01
EXPECTED_CDF_FEATURES = [
    "cdf_score_loss_mean__max",
    "cdf_min_k_10__max",
    "cdf_min_kpp_10__max",
]
EXPECTED_LOG_LENGTH = ["audit_log_token_count", "audit_log_window_count"]


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def one_file(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name}, found {len(matches)}")
    return matches[0]


def finite_metric(mapping: dict[str, Any], name: str) -> float:
    value = float(mapping.get(name, float("nan")))
    if not math.isfinite(value):
        raise RuntimeError(f"metric is missing or non-finite: {name}")
    return value


def validate_manifest(manifest: dict[str, Any]) -> dict[str, float]:
    exact = {
        "status": "complete",
        "request_id": REQUEST_ID,
        "method": METHOD,
        "seed": 2027,
        "bridge_builder_blob_sha": BUILDER_BLOB,
        "source_dataset": SOURCE_DATASET,
        "source_dataset_version": 1,
        "source_extraction_reused": True,
        "model_forward_pass": False,
        "accelerator": "cpu",
        "train_rows": 10_000,
        "validation_rows": 5_000,
        "train_shards": 40,
        "validation_shards": 20,
        "submission_file": "submission.csv",
        "submission_rows": 5_000,
        "validation_labels_read": False,
        "validation_membership_read": False,
        "validation_labels_used_for_fit_or_feature_selection": False,
        "hidden_validation_labels_used": False,
        "public_leaderboard_tuning_used": False,
        "mellum_artifacts_read": False,
        "mellum_labels_used": False,
        "submission_created": True,
        "competition_submission_performed_inside_notebook": False,
    }
    for key, value in exact.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"run manifest mismatch: {key}")
    expected_counts = {
        "source": 174,
        "cdf": 3,
        "log_length": 2,
        "interaction_sources": 172,
        "interactions": 344,
        "model_total": 523,
    }
    if manifest.get("feature_counts") != expected_counts:
        raise RuntimeError("run manifest feature-count contract changed")
    if float(manifest.get("expected_candidate_oof_auc", float("nan"))) != EXPECTED_OOF_AUC:
        raise RuntimeError("expected candidate OOF AUC changed")
    schema_sha = str(manifest.get("feature_schema_sha256") or "")
    submission_sha = str(manifest.get("submission_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", schema_sha):
        raise RuntimeError("feature schema SHA-256 is malformed")
    if not re.fullmatch(r"[0-9a-f]{64}", submission_sha):
        raise RuntimeError("submission SHA-256 is malformed")
    for center in ("full_train_log_token_center", "full_train_log_window_center"):
        if not math.isfinite(float(manifest.get(center, float("nan")))):
            raise RuntimeError(f"length center is missing or non-finite: {center}")
    runtime = float(manifest.get("runtime_seconds", float("nan")))
    if not math.isfinite(runtime) or runtime <= 0:
        raise RuntimeError("runtime_seconds is invalid")

    oof = manifest.get("reproduced_candidate_oof") or {}
    auc = finite_metric(oof, "auc")
    pauc = finite_metric(oof, "pauc_01")
    tpr_1pct = finite_metric(oof, "tpr_at_0.01_fpr")
    for key in ("tpr_at_0.001_fpr", "tpr_at_0.005_fpr", "tpr_at_0.02_fpr"):
        finite_metric(oof, key)
    if abs(auc - EXPECTED_OOF_AUC) > OOF_AUC_TOLERANCE:
        raise RuntimeError(f"candidate OOF AUC gate failed: {auc}")
    if abs(tpr_1pct - EXPECTED_TPR_1PCT) > TPR_1PCT_TOLERANCE:
        raise RuntimeError(f"candidate OOF TPR@1% gate failed: {tpr_1pct}")
    return {"oof_auc": auc, "oof_pauc_1pct": pauc, "oof_tpr_1pct": tpr_1pct}


def validate_schema(path: Path, manifest: dict[str, Any]) -> None:
    schema = json.loads(path.read_text(encoding="utf-8"))
    families = schema.get("source_families") or {}
    expected_counts = {"base": 113, "structure_raw": 50, "fim": 11}
    if set(families) != set(expected_counts):
        raise RuntimeError("feature schema family set changed")
    if not all(isinstance(families[name], list) for name in expected_counts):
        raise RuntimeError("feature schema families must be lists")
    counts = {name: len(families[name]) for name in expected_counts}
    if counts != expected_counts or schema.get("source_counts") != expected_counts:
        raise RuntimeError(f"feature schema family counts changed: {counts}")
    if schema.get("source_total") != 174:
        raise RuntimeError("feature schema source total changed")
    if schema.get("cdf_features") != EXPECTED_CDF_FEATURES:
        raise RuntimeError("feature schema CDF list changed")
    if schema.get("log_length_features") != EXPECTED_LOG_LENGTH:
        raise RuntimeError("feature schema log-length list changed")
    if schema.get("interaction_sources") != 172 or schema.get("interaction_features") != 344:
        raise RuntimeError("feature schema interaction layout changed")
    if schema.get("model_total") != 523:
        raise RuntimeError("feature schema model total changed")

    ordered = families["base"] + families["structure_raw"] + families["fim"]
    if len(ordered) != 174 or len(set(ordered)) != 174:
        raise RuntimeError("feature schema contains duplicate or missing source features")
    if "token_count" not in families["base"] or "window_count" not in families["base"]:
        raise RuntimeError("length features are absent from base family")
    digest = hashlib.sha256(
        json.dumps(ordered, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if digest != schema.get("ordered_source_feature_sha256"):
        raise RuntimeError("feature schema self-digest mismatch")
    if digest != manifest.get("feature_schema_sha256"):
        raise RuntimeError("feature schema differs from run manifest")
    missing = schema.get("missing_validation_columns_filled_with_nan")
    if not isinstance(missing, list) or any(name not in ordered for name in missing):
        raise RuntimeError("missing-validation feature record is malformed")


def validate_source_shards(path: Path) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 60:
        raise RuntimeError("source shard manifest must contain 60 rows")
    relative_paths: set[str] = set()
    split_counts = {"train": 0, "validation": 0}
    for row in rows:
        if set(row) != {"relative_path", "split", "rows", "bytes", "sha256", "columns"}:
            raise RuntimeError("source shard manifest schema changed")
        relative = str(row["relative_path"])
        split = str(row["split"])
        if relative in relative_paths:
            raise RuntimeError("duplicate source shard path")
        relative_paths.add(relative)
        if split not in split_counts:
            raise RuntimeError("unknown source shard split")
        split_counts[split] += 1
        if int(row["rows"]) != 250 or int(row["bytes"]) <= 0 or int(row["columns"]) <= 0:
            raise RuntimeError("source shard size/schema record is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])):
            raise RuntimeError("source shard SHA-256 is malformed")
        if not re.search(r"features\.part\d{3}\.parquet$", relative):
            raise RuntimeError("source shard filename is unexpected")
    if split_counts != {"train": 40, "validation": 20}:
        raise RuntimeError(f"source shard split counts changed: {split_counts}")


def validate_submission(path: Path, expected_sha256: str | None) -> str:
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError("submission differs from approved SHA-256")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sample_id", "membership_score"]:
            raise RuntimeError("submission header changed")
        rows = list(reader)
    if len(rows) != 5_000:
        raise RuntimeError(f"submission row count changed: {len(rows)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != 5_000 or any(not value for value in sample_ids):
        raise RuntimeError("submission sample_id coverage is invalid")
    scores = [float(row["membership_score"]) for row in rows]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in scores):
        raise RuntimeError("submission scores are not finite probabilities")
    return digest


def validate_output(
    root: Path,
    *,
    expected_sha256: str | None = None,
    github_output: Path | None = None,
) -> dict[str, Any]:
    manifest_path = one_file(root, "run_manifest.json")
    submission_path = one_file(root, "submission.csv")
    schema_path = one_file(root, "feature_schema.json")
    shards_path = one_file(root, "source_shards.json")
    allowed = {
        manifest_path.resolve(),
        submission_path.resolve(),
        schema_path.resolve(),
        shards_path.resolve(),
    }
    unexpected = [
        path for path in root.rglob("*")
        if path.is_file() and path.resolve() not in allowed
    ]
    if unexpected:
        raise RuntimeError(
            f"downloaded output contains non-allowlisted files: {[path.name for path in unexpected]}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = validate_manifest(manifest)
    validate_schema(schema_path, manifest)
    validate_source_shards(shards_path)
    digest = validate_submission(submission_path, expected_sha256)
    if digest != manifest.get("submission_sha256"):
        raise RuntimeError("submission SHA-256 differs from run manifest")

    result = {
        "status": "verified",
        "request_id": REQUEST_ID,
        "submission_path": str(submission_path.resolve()),
        "submission_sha256": digest,
        "submission_rows": 5_000,
        "model_features": 523,
        **metrics,
        "hidden_validation_labels_used": False,
        "public_leaderboard_tuning_used": False,
    }
    if github_output is not None:
        with github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"submission_sha256={digest}\n")
            handle.write(f"submission_path={submission_path.resolve()}\n")
            handle.write(f"oof_auc={metrics['oof_auc']:.12f}\n")
            handle.write(f"oof_tpr_1pct={metrics['oof_tpr_1pct']:.12f}\n")
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    expected = args.expected_sha256
    if expected is not None and not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise SystemExit("expected SHA-256 is malformed")
    validate_output(
        args.output_dir.resolve(),
        expected_sha256=expected,
        github_output=args.github_output,
    )


if __name__ == "__main__":
    main()
