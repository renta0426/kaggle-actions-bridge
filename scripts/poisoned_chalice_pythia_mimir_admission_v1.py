#!/usr/bin/env python3
"""Credential-free admission audit for a Pythia/MIMIR development environment.

This audit performs no model inference, downloads no model weights, calls no
Kaggle API, and never evaluates membership performance.  It resolves the exact
Pythia checkpoint, validates lightweight model/tokenizer metadata, verifies the
pinned MIMIR GitHub member/nonmember files, and checks the canonical MIMIR code
that defines train/member and test/nonmember provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any
import urllib.parse
import urllib.request

SHA40 = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_AUDIT_ID = "20260912-poisoned-chalice-pythia-mimir-development-admission-v1"
MAX_MODEL_METADATA_BYTES = 12_000_000
MAX_DATA_FILE_BYTES = 8_000_000
MAX_CANONICAL_FILE_BYTES = 262_144
USER_AGENT = "kaggle-actions-bridge-poisoned-chalice-admission/1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(encoded)


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("admission config must be an object")
    validate_config(value)
    return value


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1 or config.get("audit_id") != EXPECTED_AUDIT_ID:
        raise ValueError("admission identity changed")
    if config.get("role") != "model_transfer_development":
        raise ValueError("environment role must remain development")
    if config.get("status") != "pre_label_method_selection_admission":
        raise ValueError("admission phase changed")

    model = config.get("target_model") or {}
    expected_model = {
        "repository": "EleutherAI/pythia-2.8b",
        "revision_ref": "step143000",
        "resolved_commit": None,
        "license_expected": "apache-2.0",
        "architecture_expected": "GPTNeoXForCausalLM",
        "num_hidden_layers_expected": 32,
        "hidden_size_expected": 2560,
        "max_position_embeddings_expected": 2048,
        "tokenizer_policy": "load_from_same_repository_and_exact_resolved_commit",
    }
    if model != expected_model:
        raise ValueError("target-model admission contract changed")

    benchmark = config.get("benchmark") or {}
    required_benchmark = {
        "mirror_repository": "Al-not-AI/mimir",
        "mirror_commit": "47e348e97cae2ab8c1d2278d4ffd6db2f1d043f8",
        "source": "github",
        "split": "ngram_13_0.8",
        "member_path": "cache_100_200_1000_512/train/github_ngram_13_0.8.jsonl",
        "nonmember_path": "cache_100_200_1000_512/test/github_ngram_13_0.8.jsonl",
        "expected_rows_per_class": 1000,
        "expected_total_rows": 2000,
        "unit": "MIMIR truncated GitHub snippet",
        "expected_min_whitespace_words": 100,
        "expected_max_whitespace_words": 200,
        "construction_max_mask_tokens": 512,
        "exact_cross_split_duplicates_allowed": 0,
    }
    if benchmark != required_benchmark:
        raise ValueError("benchmark admission contract changed")

    provenance = config.get("canonical_provenance") or {}
    if provenance != {
        "repository": "iamgroot42/mimir",
        "commit": "1b6fd649eeeecc887275a2336c7da808ee58757d",
        "custom_datasets_path": "mimir/custom_datasets.py",
        "custom_datasets_blob": "f7174c444c78df389e6c43f2c56d8ff5b34a76ae",
        "create_datasets_path": "data/create_datasets.py",
        "create_datasets_blob": "df1a29c04c46bbdc16613a8a5a4c760a6328ce54",
        "create_pile_subsets_path": "data/create_pile_subsets.sh",
        "create_pile_subsets_blob": "9a0b48626dcfc3b9d4bddb22abd271b0e333479e",
        "license_blob": "03e7000da6731658bc7d739995f7d4d17ab73c59",
    }:
        raise ValueError("canonical provenance contract changed")

    rules = config.get("development_rules") or {}
    forbidden_true = (
        "target_labels_may_be_used_for_method_fit",
        "target_labels_may_be_used_for_weight_search",
        "target_labels_may_be_used_for_sign_selection",
        "target_labels_may_be_used_for_layer_selection",
        "target_labels_may_be_used_for_post_hoc_sample_selection",
        "sealed_final_environment_consumed",
    )
    if any(rules.get(key) is not False for key in forbidden_true):
        raise ValueError("development clean-room boundary changed")
    required_true = (
        "labels_may_be_joined_only_after_prediction_artifacts_are_frozen",
        "content_only_control_required",
        "actual_fp_tp_required_for_low_fpr",
        "bootstrap_uncertainty_required",
    )
    if any(rules.get(key) is not True for key in required_true):
        raise ValueError("development evaluation safeguards changed")

    execution = config.get("audit_execution") or {}
    if execution != {
        "kaggle_calls": 0,
        "model_inference": False,
        "model_weight_download": False,
        "github_actions_expected_runner_minutes": 3,
        "credentials_required": False,
        "side_effects": [],
    }:
        raise ValueError("audit execution contract changed")


def fetch(url: str, maximum: int) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("only HTTPS fetches are allowed")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=90) as response:
        data = response.read(maximum + 1)
    if not data or len(data) > maximum:
        raise RuntimeError(f"fetch byte budget failed: {url} bytes={len(data)} max={maximum}")
    return data


def resolve_hf_branch(repository: str, revision_ref: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", repository) or not re.fullmatch(r"[A-Za-z0-9._-]+", revision_ref):
        raise ValueError("unsafe Hugging Face repository/ref")
    env = os.environ.copy()
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"})
    completed = subprocess.run(
        ["git", "-c", "credential.helper=", "ls-remote", "--refs", f"https://huggingface.co/{repository}.git", f"refs/heads/{revision_ref}"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("Hugging Face revision resolution failed")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected exactly one Hugging Face ref, found {len(lines)}")
    fields = lines[0].split()
    if len(fields) != 2 or fields[1] != f"refs/heads/{revision_ref}" or not SHA40.fullmatch(fields[0]):
        raise RuntimeError("unexpected Hugging Face ls-remote result")
    return fields[0]


def audit_model(config: dict[str, Any]) -> dict[str, Any]:
    model = config["target_model"]
    repository = model["repository"]
    commit = resolve_hf_branch(repository, model["revision_ref"])
    base = f"https://huggingface.co/{repository}/resolve/{commit}"
    files: dict[str, bytes] = {}
    for name, budget in (
        ("config.json", 262_144),
        ("tokenizer_config.json", 262_144),
        ("tokenizer.json", MAX_MODEL_METADATA_BYTES),
        ("README.md", 524_288),
    ):
        files[name] = fetch(f"{base}/{name}", budget)

    model_config = json.loads(files["config.json"])
    architectures = model_config.get("architectures") or []
    if model["architecture_expected"] not in architectures:
        raise RuntimeError(f"unexpected model architecture: {architectures}")
    checks = {
        "num_hidden_layers": model["num_hidden_layers_expected"],
        "hidden_size": model["hidden_size_expected"],
        "max_position_embeddings": model["max_position_embeddings_expected"],
    }
    for key, expected in checks.items():
        if int(model_config.get(key, -1)) != int(expected):
            raise RuntimeError(f"model config changed: {key}={model_config.get(key)} expected={expected}")
    if str(model_config.get("model_type")) != "gpt_neox":
        raise RuntimeError("Pythia model_type changed")

    readme = files["README.md"].decode("utf-8", errors="strict")
    license_match = re.search(r"(?m)^license:\s*([^\s]+)\s*$", readme)
    if not license_match or license_match.group(1).casefold() != model["license_expected"]:
        raise RuntimeError("Pythia model-card license metadata changed")

    tokenizer_config = json.loads(files["tokenizer_config.json"])
    if not isinstance(tokenizer_config, dict):
        raise RuntimeError("tokenizer config must be an object")
    return {
        "repository": repository,
        "requested_revision": model["revision_ref"],
        "resolved_commit": commit,
        "license": license_match.group(1),
        "architecture": model["architecture_expected"],
        "num_hidden_layers": int(model_config["num_hidden_layers"]),
        "hidden_size": int(model_config["hidden_size"]),
        "max_position_embeddings": int(model_config["max_position_embeddings"]),
        "model_config_sha256": sha256_bytes(files["config.json"]),
        "tokenizer_config_sha256": sha256_bytes(files["tokenizer_config.json"]),
        "tokenizer_json_sha256": sha256_bytes(files["tokenizer.json"]),
        "tokenizer_files_resolved_at_same_commit": True,
        "model_weights_downloaded": False,
    }


def extract_text(record: Any) -> tuple[str, str]:
    if isinstance(record, str):
        return record, "json_string"
    if isinstance(record, dict):
        for key in ("text", "input", "original"):
            value = record.get(key)
            if isinstance(value, str):
                return value, f"object:{key}"
    raise RuntimeError("unsupported MIMIR JSONL record schema")


def audit_jsonl(data: bytes, *, expected_rows: int, min_words: int, max_words: int) -> dict[str, Any]:
    texts: list[str] = []
    schemas: set[str] = set()
    lines = data.decode("utf-8", errors="strict").splitlines()
    if any(not line.strip() for line in lines):
        raise RuntimeError("blank line present in MIMIR JSONL")
    for line in lines:
        record = json.loads(line)
        text, schema = extract_text(record)
        if not text:
            raise RuntimeError("empty MIMIR sample")
        texts.append(text)
        schemas.add(schema)
    if len(texts) != expected_rows:
        raise RuntimeError(f"MIMIR row count changed: {len(texts)} expected={expected_rows}")
    if len(schemas) != 1:
        raise RuntimeError(f"mixed MIMIR record schemas: {sorted(schemas)}")
    word_counts = [len(text.split()) for text in texts]
    outside = sum(count < min_words or count > max_words for count in word_counts)
    if outside:
        raise RuntimeError(f"MIMIR word-length contract changed: outside={outside}")
    text_hashes = [sha256_bytes(text.encode("utf-8")) for text in texts]
    return {
        "rows": len(texts),
        "schema": next(iter(schemas)),
        "raw_sha256": sha256_bytes(data),
        "text_set_sha256": sha256_bytes("\n".join(sorted(text_hashes)).encode("ascii")),
        "unique_texts": len(set(text_hashes)),
        "duplicate_rows": len(text_hashes) - len(set(text_hashes)),
        "min_whitespace_words": min(word_counts),
        "max_whitespace_words": max(word_counts),
        "texts": texts,
        "text_hashes": text_hashes,
    }


def audit_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    benchmark = config["benchmark"]
    base = f"https://huggingface.co/datasets/{benchmark['mirror_repository']}/resolve/{benchmark['mirror_commit']}"
    results: dict[str, dict[str, Any]] = {}
    for role, path_key in (("member", "member_path"), ("nonmember", "nonmember_path")):
        path = benchmark[path_key]
        quoted = urllib.parse.quote(path, safe="/")
        data = fetch(f"{base}/{quoted}", MAX_DATA_FILE_BYTES)
        results[role] = audit_jsonl(
            data,
            expected_rows=int(benchmark["expected_rows_per_class"]),
            min_words=int(benchmark["expected_min_whitespace_words"]),
            max_words=int(benchmark["expected_max_whitespace_words"]),
        )
    overlap = len(set(results["member"]["text_hashes"]) & set(results["nonmember"]["text_hashes"]))
    if overlap != int(benchmark["exact_cross_split_duplicates_allowed"]):
        raise RuntimeError(f"exact member/nonmember overlap changed: {overlap}")
    total = int(results["member"]["rows"]) + int(results["nonmember"]["rows"])
    if total != int(benchmark["expected_total_rows"]):
        raise RuntimeError("MIMIR total row count changed")
    public = {
        role: {key: value for key, value in record.items() if key not in {"texts", "text_hashes"}}
        for role, record in results.items()
    }
    public["exact_cross_split_duplicates"] = overlap
    public["total_rows"] = total
    public["balanced_classes"] = results["member"]["rows"] == results["nonmember"]["rows"]
    public["mirror_repository"] = benchmark["mirror_repository"]
    public["mirror_commit"] = benchmark["mirror_commit"]
    public["source"] = benchmark["source"]
    public["split"] = benchmark["split"]
    return public


def fetch_canonical(repository: str, commit: str, path: str) -> bytes:
    quoted = urllib.parse.quote(path, safe="/")
    return fetch(f"https://raw.githubusercontent.com/{repository}/{commit}/{quoted}", MAX_CANONICAL_FILE_BYTES)


def audit_canonical_provenance(config: dict[str, Any]) -> dict[str, Any]:
    provenance = config["canonical_provenance"]
    repository = provenance["repository"]
    commit = provenance["commit"]
    files = {
        "custom_datasets": (provenance["custom_datasets_path"], provenance["custom_datasets_blob"]),
        "create_datasets": (provenance["create_datasets_path"], provenance["create_datasets_blob"]),
        "create_pile_subsets": (provenance["create_pile_subsets_path"], provenance["create_pile_subsets_blob"]),
        "license": ("LICENSE", provenance["license_blob"]),
    }
    loaded: dict[str, bytes] = {}
    for name, (path, expected_blob) in files.items():
        data = fetch_canonical(repository, commit, path)
        if git_blob_sha(data) != expected_blob:
            raise RuntimeError(f"canonical MIMIR blob changed: {path}")
        loaded[name] = data

    custom = loaded["custom_datasets"].decode("utf-8")
    if 'data_split = data_split.replace("train", "member")' not in custom:
        raise RuntimeError("MIMIR train/member mapping marker missing")
    if 'data_split = data_split.replace("test", "nonmember")' not in custom:
        raise RuntimeError("MIMIR test/nonmember mapping marker missing")
    if 'datasets.load_dataset("iamgroot42/mimir"' not in custom:
        raise RuntimeError("MIMIR canonical HF loader marker missing")

    create = loaded["create_datasets"].decode("utf-8")
    for marker in (
        "parser.add_argument('--min_len', type=int, default=100)",
        "parser.add_argument('--max_len', type=int, default=200)",
        "len(mask_tokenized) <= 512",
        "min_ngram_overlap",
        "max_ngram_overlap",
    ):
        if marker not in create:
            raise RuntimeError(f"MIMIR construction marker missing: {marker}")

    pile = loaded["create_pile_subsets"].decode("utf-8")
    if "[test]=/data/pile/test.jsonl" not in pile or "/data/pile/train/00.jsonl /data/pile/train/01.jsonl" not in pile:
        raise RuntimeError("MIMIR Pile train/test source mapping changed")
    if not loaded["license"].decode("utf-8").startswith("MIT License\n"):
        raise RuntimeError("MIMIR license changed")

    return {
        "repository": repository,
        "commit": commit,
        "license": "MIT",
        "train_maps_to_member": True,
        "test_maps_to_nonmember": True,
        "construction_min_words": 100,
        "construction_max_words": 200,
        "construction_max_mask_tokens": 512,
        "canonical_git_blobs": {name: expected for name, (_, expected) in files.items()},
    }


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    model = audit_model(config)
    benchmark = audit_benchmark(config)
    canonical = audit_canonical_provenance(config)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "status": "pass",
        "role": config["role"],
        "target_model": model,
        "benchmark": benchmark,
        "canonical_provenance": canonical,
        "label_semantics": config["label_semantics"],
        "development_rules": config["development_rules"],
        "execution": {
            "kaggle_calls": 0,
            "model_inference": False,
            "model_weights_downloaded": False,
            "credentials_used": False,
            "side_effects": [],
        },
        "decision": {
            "development_environment_admitted": True,
            "sealed_confirmation_environment_consumed": False,
            "performance_labels_observed": False,
            "next_step": "freeze method comparison before any membership-label evaluation",
        },
    }
    payload["audit_manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def self_test(config: dict[str, Any]) -> None:
    assert SHA40.fullmatch("0" * 40)
    assert git_blob_sha(b"test\n") == "9daeafb9864cf43055ae93beb0afd6c7d144bfa4"
    fixture = b'"' + (b"word " * 99) + b'word"\n'
    record = audit_jsonl(fixture, expected_rows=1, min_words=100, max_words=200)
    assert record["rows"] == 1 and record["min_whitespace_words"] == 100
    assert config["development_rules"]["sealed_final_environment_consumed"] is False
    print("PYTHIA_MIMIR_ADMISSION_SELF_TEST PASS kaggle_calls=0 model_inference=0")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.self_test:
        self_test(config)
        return 0
    if args.output is None:
        raise ValueError("--output is required for --audit")
    result = run_audit(config)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "PYTHIA_MIMIR_ADMISSION PASS "
        f"model_commit={result['target_model']['resolved_commit']} "
        f"member_sha256={result['benchmark']['member']['raw_sha256']} "
        f"nonmember_sha256={result['benchmark']['nonmember']['raw_sha256']} "
        f"rows={result['benchmark']['total_rows']} exact_cross_split_duplicates=0 "
        f"manifest_sha256={result['audit_manifest_sha256']} labels_observed=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
