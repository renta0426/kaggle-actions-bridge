#!/usr/bin/env python3
"""Frozen PYTHIA-MIMIR-DEVELOPMENT-V1 runner.

Scientific boundary:
- target candidate scores are computed without target membership labels;
- predictions are persisted and SHA-256 sealed before labels are read;
- target labels are used only for post-seal evaluation and a separately named
  OOF content-confounding diagnostic;
- no target-label fit/sign/layer/fusion/sample selection occurs;
- the sealed final model-transfer environment is not touched.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable
import urllib.parse
import urllib.request

EXPERIMENT_ID = "PYTHIA-MIMIR-DEVELOPMENT-V1"
OUTPUT = Path("/kaggle/working/pythia_mimir_development_v1")
PREDICTIONS = OUTPUT / "predictions_label_free.csv"
LABELS_TMP = Path("/tmp/pythia_mimir_development_v1_labels.jsonl")
USER_AGENT = "poisoned-chalice-pythia-mimir-development-v1/1"
PORTABLE_FEATURE_COUNT = 96
PORTABLE_GRID_POINTS = 16
PORTABLE_SUMMARIES = ("mean", "std", "min", "max", "q25", "q50", "q75", "slope")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config, runtime=False)
    return config


def validate_config(config: dict[str, Any], *, runtime: bool) -> None:
    if config.get("schema_version") != 1 or config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment identity changed")
    if config.get("role") != "model_transfer_development":
        raise ValueError("environment role changed")
    if config.get("execution_policy") != "kaggle_native_capacity_v2":
        raise ValueError("execution policy changed")
    if config.get("automatic_compute_retries") != 0:
        raise ValueError("automatic retry must remain zero")
    if config.get("competition_submission") is not False or config.get("select_as_final") is not False:
        raise ValueError("submission/final-selection is forbidden")

    target = config.get("target_model") or {}
    if target.get("repository") != "EleutherAI/pythia-2.8b":
        raise ValueError("target repository changed")
    if target.get("revision") != "dbe7ae300a54abcdc475a33907b3dff81d25709f":
        raise ValueError("target revision changed")
    if (target.get("num_hidden_layers"), target.get("hidden_size"), target.get("max_position_embeddings")) != (32, 2560, 2048):
        raise ValueError("target architecture contract changed")

    benchmark = config.get("benchmark") or {}
    if benchmark.get("repository") != "Al-not-AI/mimir" or benchmark.get("revision") != "47e348e97cae2ab8c1d2278d4ffd6db2f1d043f8":
        raise ValueError("benchmark identity changed")
    if benchmark.get("member_sha256") != "41b785ca8bbc1051596900df3d16737a92079e9fcc1ae97e13d967ff9a07b065":
        raise ValueError("member file identity changed")
    if benchmark.get("nonmember_sha256") != "d382222bd0a723e47da9dfd3afc99d0890e532a39440179e1e295b932792f8a8":
        raise ValueError("nonmember file identity changed")
    if (benchmark.get("rows_per_class"), benchmark.get("total_rows")) != (1000, 2000):
        raise ValueError("benchmark row contract changed")
    if benchmark.get("domain_label_for_label_free_calibration") != "GitHub":
        raise ValueError("domain calibration policy changed")

    source = config.get("source_bundle") or {}
    if source.get("upstream_kernel") != "renta0426/poisoned-chalice-stage2-deployment-validation-v1":
        raise ValueError("source bundle kernel changed")
    if source.get("expected_version") != 1 or source.get("expected_task_id") != "STAGE2-DEPLOYMENT-VALIDATION-V1":
        raise ValueError("source bundle version/task changed")
    if source.get("reference_model_id") != "bigcode/starcoder2-3b" or source.get("reference_model_revision") != "733247c55e3f73af49ce8e9c7949bf14af205928":
        raise ValueError("reference model changed")
    if source.get("source_cache_manifest_sha256") != "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa":
        raise ValueError("source cache identity changed")
    bundle_sha = source.get("bundle_manifest_sha256")
    if bundle_sha is not None and not re.fullmatch(r"[0-9a-f]{64}", str(bundle_sha)):
        raise ValueError("invalid source bundle SHA-256")
    if runtime and source.get("require_exact_bundle_manifest_sha256_before_launch") is True and bundle_sha is None:
        raise RuntimeError("source bundle exact SHA-256 is not frozen; runtime launch is blocked")

    output = config.get("target_output") or {}
    expected_output = {
        "method_version": "stage2-model-independent-v1",
        "max_length": 768,
        "max_batch_tokens": 2048,
        "vocab_chunk_tokens": 64,
        "rank_vocab_block_size": 8192,
        "min_k_percents": [1, 2, 5, 10, 20, 30],
        "local_widths": [32, 64, 128],
        "language_calibration_min_rows": 20,
        "length_calibration_min_rows": 12,
        "fidelity_gate": True,
        "fidelity_tokens": 24,
        "fidelity_atol": 1e-5,
        "domain_policy": "all rows use the single predeclared GitHub domain label; no inferred programming-language labels",
    }
    if output != expected_output:
        raise ValueError("target output-scoring contract changed")

    clean = config.get("clean_room") or {}
    for key in (
        "target_labels_used_for_candidate_fit",
        "target_labels_used_for_candidate_sign",
        "target_labels_used_for_candidate_layer_selection",
        "target_labels_used_for_fusion_weights",
        "target_labels_used_for_sample_selection",
        "sealed_confirmation_environment_consumed",
        "competition_submission",
    ):
        if clean.get(key) is not False:
            raise ValueError(f"clean-room boundary changed: {key}")
    if clean.get("candidate_predictions_hashed_before_target_label_load") is not True:
        raise ValueError("prediction-seal boundary changed")


def fetch(url: str, maximum: int) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("only HTTPS fetches are allowed")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read(maximum + 1)
    if not data or len(data) > maximum:
        raise RuntimeError(f"fetch byte budget failed: {url} bytes={len(data)} max={maximum}")
    return data


def parse_public_json_strings(data: bytes, expected_rows: int) -> list[str]:
    lines = data.decode("utf-8", errors="strict").splitlines()
    if len(lines) != expected_rows or any(not line.strip() for line in lines):
        raise RuntimeError("MIMIR JSONL row/blank-line contract changed")
    rows = [json.loads(line) for line in lines]
    if any(not isinstance(row, str) or not row for row in rows):
        raise RuntimeError("MIMIR admitted json_string schema changed")
    return rows


def prepare_target_inputs(config: dict[str, Any]):
    import pandas as pd

    benchmark = config["benchmark"]
    base = f"https://huggingface.co/datasets/{benchmark['repository']}/resolve/{benchmark['revision']}"
    role_rows: dict[str, list[str]] = {}
    for role, path_key, sha_key in (
        ("member", "member_path", "member_sha256"),
        ("nonmember", "nonmember_path", "nonmember_sha256"),
    ):
        quoted = urllib.parse.quote(benchmark[path_key], safe="/")
        data = fetch(f"{base}/{quoted}", 8_000_000)
        if sha256_bytes(data) != benchmark[sha_key]:
            raise RuntimeError(f"MIMIR {role} raw SHA-256 changed")
        role_rows[role] = parse_public_json_strings(data, int(benchmark["rows_per_class"]))

    records: list[dict[str, Any]] = []
    label_lines: list[str] = []
    seen: set[str] = set()
    for role, label in (("member", 1), ("nonmember", 0)):
        for text in role_rows[role]:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            sample_id = f"mimir-github-{digest}"
            if sample_id in seen:
                raise RuntimeError("exact cross-split/within-split content duplicate detected")
            seen.add(sample_id)
            records.append({"sample_id": sample_id, "content": text})
            label_lines.append(json.dumps({"sample_id": sample_id, "label": label, "role": role}, sort_keys=True))
    if len(records) != int(benchmark["total_rows"]) or len(seen) != len(records):
        raise RuntimeError("target cohort coverage changed")
    records.sort(key=lambda row: row["sample_id"])
    LABELS_TMP.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    frame = pd.DataFrame(records)
    if list(frame.columns) != ["sample_id", "content"] or frame.sample_id.duplicated().any():
        raise RuntimeError("label-free target frame contract failed")
    input_manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "rows": len(frame),
        "execution_order": "ascending_sample_id",
        "sample_set_sha256": sha256_bytes("\n".join(frame.sample_id).encode("utf-8")),
        "member_file_sha256": benchmark["member_sha256"],
        "nonmember_file_sha256": benchmark["nonmember_sha256"],
        "labels_materialized_separately": True,
        "labels_loaded_for_candidate_scoring": False,
        "performance_metrics_computed": False,
    }
    return frame, input_manifest


def locate_source_bundle(config: dict[str, Any]) -> tuple[Path, dict[str, Any], str]:
    source = config["source_bundle"]
    candidates: list[tuple[Path, dict[str, Any], str]] = []
    for path in Path("/kaggle/input").rglob("bundle_manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("task_id") != source["expected_task_id"]:
            continue
        digest = sha256_file(path)
        candidates.append((path.parent, manifest, digest))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one Stage2 source bundle, found {len(candidates)}")
    root, manifest, digest = candidates[0]
    expected_digest = source["bundle_manifest_sha256"]
    if digest != expected_digest:
        raise RuntimeError(f"source bundle manifest identity mismatch: {digest} != {expected_digest}")
    seal = (root / "SEALED.sha256").read_text(encoding="utf-8").strip().split()
    if not seal or seal[0] != digest:
        raise RuntimeError("source bundle SEALED.sha256 mismatch")
    if manifest.get("model_id") != source["reference_model_id"] or manifest.get("model_revision") != source["reference_model_revision"]:
        raise RuntimeError("source bundle reference model changed")
    if int(manifest.get("source_rows", -1)) != int(source["source_rows"]):
        raise RuntimeError("source bundle row contract changed")
    if manifest.get("source_cache_manifest_sha256") != source["source_cache_manifest_sha256"]:
        raise RuntimeError("source bundle cache provenance changed")
    if manifest.get("labels_used") != "source labels only" or int(manifest.get("holdout_rows_seen_during_fit", -1)) != 0:
        raise RuntimeError("source bundle clean-room contract changed")
    fidelity = manifest.get("reload_fidelity") or {}
    if fidelity.get("passed") is not True:
        raise RuntimeError("source bundle reload fidelity did not pass")
    required = set(source["allowed_bundle_files"])
    present = {path.name for path in root.iterdir() if path.is_file()}
    missing = required.difference(present)
    if missing:
        raise RuntimeError(f"source bundle required files missing: {sorted(missing)}")
    return root, manifest, digest


def load_gr_bundle_minimal(root: Path) -> dict[str, Any]:
    import numpy as np

    bundle = json.loads((root / "gr_metadata.json").read_text(encoding="utf-8"))
    with np.load(root / "gr_parameters.npz", allow_pickle=False) as archive:
        for name in ("imputer_statistics", "scaler_mean", "scaler_scale", "coef"):
            bundle[name] = np.asarray(archive[name], dtype=np.float64)
    return bundle


def profile_features(values, prefix: str):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("portable profile shape invalid")
    depth = values.shape[1]
    x = np.linspace(0.0, 1.0, depth)
    grid = np.linspace(0.0, 1.0, PORTABLE_GRID_POINTS)
    interp = np.vstack([np.interp(grid, x, row) for row in values])
    q = np.quantile(values, [0.25, 0.50, 0.75], axis=1).T
    xm = x - x.mean()
    slope = ((values - values.mean(axis=1, keepdims=True)) @ xm) / float(np.dot(xm, xm))
    stats = np.column_stack([
        values.mean(axis=1), values.std(axis=1), values.min(axis=1), values.max(axis=1), q, slope,
    ])
    names = [f"{prefix}_depth_{i:02d}" for i in range(PORTABLE_GRID_POINTS)]
    names += [f"{prefix}_{name}" for name in PORTABLE_SUMMARIES]
    return np.column_stack([interp, stats]), names


def scalarize_hidden_mean_96(array):
    import numpy as np

    x = np.asarray(array, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 3 or x.shape[2] < 1:
        raise ValueError("hidden array must be [rows,layers,hidden]")
    eps = 1e-12
    rms = np.sqrt(np.mean(x * x, axis=2))
    left, right = x[:, :-1, :], x[:, 1:, :]
    left_norm = np.linalg.norm(left, axis=2)
    right_norm = np.linalg.norm(right, axis=2)
    adjacent_cos = np.sum(left * right, axis=2) / np.maximum(left_norm * right_norm, eps)
    delta = right - left
    rel_delta = np.linalg.norm(delta, axis=2) / np.maximum(right_norm, eps)
    dleft, dright = delta[:, :-1, :], delta[:, 1:, :]
    delta_direction_cos = np.sum(dleft * dright, axis=2) / np.maximum(
        np.linalg.norm(dleft, axis=2) * np.linalg.norm(dright, axis=2), eps
    )
    blocks, names = [], []
    for values, prefix in (
        (rms, "rms"),
        (adjacent_cos, "adjacent_cos"),
        (rel_delta, "relative_delta_norm"),
        (delta_direction_cos, "delta_direction_cos"),
    ):
        block, block_names = profile_features(values, prefix)
        blocks.append(block)
        names.extend(block_names)
    result = np.column_stack(blocks)
    if result.shape != (x.shape[0], PORTABLE_FEATURE_COUNT) or len(names) != PORTABLE_FEATURE_COUNT:
        raise RuntimeError("portable 96-feature schema changed")
    if not np.isfinite(result).all() or len(set(names)) != PORTABLE_FEATURE_COUNT:
        raise RuntimeError("portable 96-feature finite/identity check failed")
    return result, names


def predict_gr_minimal(bundle: dict[str, Any], mean_hidden):
    import numpy as np

    X, names = scalarize_hidden_mean_96(mean_hidden)
    if names != list(bundle["feature_names"]):
        raise RuntimeError("GR feature ordering changed")
    stats = np.asarray(bundle["imputer_statistics"], dtype=np.float64)
    rows, cols = np.where(~np.isfinite(X))
    if len(rows):
        X[rows, cols] = stats[cols]
    mean = np.asarray(bundle["scaler_mean"], dtype=np.float64)
    scale = np.asarray(bundle["scaler_scale"], dtype=np.float64)
    coef = np.asarray(bundle["coef"], dtype=np.float64)
    decision = ((X - mean) / scale) @ coef + float(bundle["intercept"])
    probability = np.empty_like(decision, dtype=np.float64)
    positive = decision >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-decision[positive]))
    exp_value = np.exp(decision[~positive])
    probability[~positive] = exp_value / (1.0 + exp_value)
    if not np.isfinite(probability).all():
        raise RuntimeError("GR prediction nonfinite")
    return probability


def load_hr_bundle_minimal(root: Path) -> dict[str, Any]:
    import numpy as np

    bundle = json.loads((root / "hr_metadata.json").read_text(encoding="utf-8"))
    states: dict[str, dict[str, Any]] = {str(layer): {} for layer in bundle["top_layers"]}
    with np.load(root / "hr_parameters.npz", allow_pickle=False) as archive:
        for layer in bundle["top_layers"]:
            prefix = f"layer_{int(layer):02d}__"
            matched = [key for key in archive.files if key.startswith(prefix)]
            if not matched:
                raise RuntimeError(f"HR state missing layer {layer}")
            for key in matched:
                states[str(layer)][key[len(prefix):]] = np.asarray(archive[key])
    bundle["state_dicts"] = states
    return bundle


def make_probe_from_state(state: dict[str, Any], input_dim: int, hidden_dim: int):
    import torch

    probe = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(hidden_dim, 1),
        torch.nn.Sigmoid(),
    )
    tensor_state = {name: torch.as_tensor(value) for name, value in state.items()}
    probe.load_state_dict(tensor_state)
    probe.eval()
    return probe


def predict_hr_from_selected(bundle: dict[str, Any], selected_means: dict[int, Any]):
    import numpy as np
    import torch

    predictions = []
    batch_size = max(1, int((bundle.get("config") or {}).get("batch_size", 32)))
    for layer in bundle["top_layers"]:
        layer = int(layer)
        X = np.asarray(selected_means[layer], dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != int(bundle["input_dim"]):
            raise RuntimeError("HR selected hidden shape changed")
        probe = make_probe_from_state(bundle["state_dicts"][str(layer)], int(bundle["input_dim"]), int(bundle["hidden_dim"]))
        values = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32)
                values.append(probe(batch).squeeze(-1).numpy())
        predictions.append(np.concatenate(values).astype(np.float64, copy=False))
        del probe
    result = np.mean(np.stack(predictions, axis=0), axis=0)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise RuntimeError("HR prediction invalid")
    return result


def rank01(values):
    import pandas as pd
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def equal_rank_fusion(*arrays):
    import numpy as np
    if not arrays:
        raise ValueError("fusion requires at least one score")
    ranked = [rank01(values) for values in arrays]
    lengths = {len(values) for values in ranked}
    if len(lengths) != 1:
        raise ValueError("fusion lengths differ")
    return np.mean(np.column_stack(ranked), axis=1)


def embedding_device(model):
    import torch
    return torch.device(model.get_input_embeddings().weight.device)


def target_layers(model, kind: str):
    if kind == "pythia" and hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers), model.gpt_neox
    if kind == "starcoder2" and hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers), model.model
    raise RuntimeError(f"cannot resolve raw transformer blocks for {kind}")


class LayerMeanCollector:
    def __init__(self, layers: list[Any], selected: Iterable[int] | None = None):
        self.layers = layers
        self.selected = list(range(len(layers))) if selected is None else sorted({int(value) for value in selected})
        if any(index < 0 or index >= len(layers) for index in self.selected):
            raise ValueError("selected layer outside architecture")
        self.mask = None
        self.values: dict[int, Any] = {}
        self.handles = [layers[index].register_forward_hook(self._make_hook(index)) for index in self.selected]

    def _make_hook(self, index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            import torch
            value = output[0] if isinstance(output, (tuple, list)) else output
            if value.ndim != 3 or self.mask is None:
                raise RuntimeError("unexpected layer output/mask state")
            mask = self.mask.to(value.device, dtype=value.dtype).unsqueeze(-1)
            denominator = mask.sum(dim=1).clamp_min(1.0)
            mean = (value * mask).sum(dim=1) / denominator
            self.values[index] = mean.detach().float().cpu().numpy()
        return hook

    def begin(self, attention_mask: Any) -> None:
        self.mask = attention_mask.detach()
        self.values = {}

    def finish(self) -> dict[int, Any]:
        if set(self.values) != set(self.selected):
            raise RuntimeError("not every selected transformer block produced a mean activation")
        return dict(self.values)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def full_token_counts(tokenizer: Any, texts: list[str]) -> list[int]:
    counts = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=True, truncation=False, return_attention_mask=False)
        values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        counts.append(len(values))
    return counts


def dynamic_batches(counts: list[int], max_batch_tokens: int, max_rows: int = 8):
    ordered = sorted(range(len(counts)), key=lambda index: (counts[index], index))
    batch: list[int] = []
    width = 0
    for index in ordered:
        candidate_width = max(width, counts[index])
        if batch and (len(batch) >= max_rows or candidate_width * (len(batch) + 1) > max_batch_tokens):
            yield batch
            batch = []
            width = 0
        batch.append(index)
        width = max(width, counts[index])
    if batch:
        yield batch


def collect_target_gr(model: Any, tokenizer: Any, texts: list[str], gr_bundle: dict[str, Any], config: dict[str, Any]):
    import numpy as np
    import torch

    layers, base_model = target_layers(model, "pythia")
    expected = config["target_model"]
    if len(layers) != int(expected["num_hidden_layers"]) or int(model.config.hidden_size) != int(expected["hidden_size"]):
        raise RuntimeError("Pythia raw-block architecture changed")
    counts = full_token_counts(tokenizer, texts)
    if max(counts) > int(expected["max_position_embeddings"]):
        raise RuntimeError(f"Pythia target snippet exceeds context: max={max(counts)}")
    output = np.empty(len(texts), dtype=np.float64)
    collector = LayerMeanCollector(layers)
    device = embedding_device(model)
    started = time.perf_counter()
    batches = 0
    try:
        for indices in dynamic_batches(counts, int(config["hidden_geometry"]["target_max_batch_tokens"]), max_rows=8):
            batch_texts = [texts[index] for index in indices]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(config["hidden_geometry"]["target_max_length"]),
            )
            input_ids = encoded["input_ids"].to(device)
            attention = encoded["attention_mask"].to(device)
            collector.begin(attention)
            with torch.inference_mode():
                base_model(input_ids=input_ids, attention_mask=attention, use_cache=False, return_dict=True)
            values = collector.finish()
            mean_hidden = np.stack([values[layer] for layer in range(len(layers))], axis=1).astype(np.float32, copy=False)
            output[np.asarray(indices, dtype=int)] = predict_gr_minimal(gr_bundle, mean_hidden)
            batches += 1
            del encoded, input_ids, attention, values, mean_hidden
    finally:
        collector.close()
    return output, counts, {"batches": batches, "seconds": time.perf_counter() - started, "layers": len(layers), "hidden_size": int(model.config.hidden_size)}


def collect_reference_hr(model: Any, tokenizer: Any, texts: list[str], hr_bundle: dict[str, Any], config: dict[str, Any]):
    import numpy as np
    import torch

    layers, base_model = target_layers(model, "starcoder2")
    top_layers = [int(value) for value in hr_bundle["top_layers"]]
    if len(layers) != 30 or int(model.config.hidden_size) != int(hr_bundle["input_dim"]):
        raise RuntimeError("StarCoder2 reference architecture/bundle changed")
    counts = full_token_counts(tokenizer, texts)
    if max(counts) > int(config["reference_hidden"]["max_length"]):
        raise RuntimeError(f"reference target snippet exceeds frozen max length: max={max(counts)}")
    selected = {layer: np.empty((len(texts), int(hr_bundle["input_dim"])), dtype=np.float32) for layer in top_layers}
    collector = LayerMeanCollector(layers, selected=top_layers)
    device = embedding_device(model)
    started = time.perf_counter()
    batches = 0
    try:
        for indices in dynamic_batches(counts, int(config["reference_hidden"]["max_batch_tokens"]), max_rows=8):
            batch_texts = [texts[index] for index in indices]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(config["reference_hidden"]["max_length"]),
            )
            input_ids = encoded["input_ids"].to(device)
            attention = encoded["attention_mask"].to(device)
            collector.begin(attention)
            with torch.inference_mode():
                base_model(input_ids=input_ids, attention_mask=attention, use_cache=False, return_dict=True)
            values = collector.finish()
            for layer in top_layers:
                selected[layer][np.asarray(indices, dtype=int)] = values[layer]
            batches += 1
            del encoded, input_ids, attention, values
    finally:
        collector.close()
    result = predict_hr_from_selected(hr_bundle, selected)
    selected.clear()
    return result, counts, {"batches": batches, "seconds": time.perf_counter() - started, "layers": len(layers), "hidden_size": int(model.config.hidden_size), "top_layers": top_layers}


def load_model_and_tokenizer(repository: str, revision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repository, revision=revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad nor eos token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        repository,
        revision=revision,
        trust_remote_code=False,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def unload_model(model: Any, tokenizer: Any) -> None:
    import torch
    del tokenizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def score_target_output(model: Any, tokenizer: Any, texts: list[str], config: dict[str, Any]):
    import numpy as np
    from poisoned_chalice.stage2_api import Stage2RuntimeConfig, STAGE2_METHOD_VERSION, score_samples_detailed

    spec = config["target_output"]
    if STAGE2_METHOD_VERSION != spec["method_version"]:
        raise RuntimeError("stage2 output method version changed")
    runtime = Stage2RuntimeConfig(
        max_length=int(spec["max_length"]),
        max_batch_tokens=int(spec["max_batch_tokens"]),
        vocab_chunk_tokens=int(spec["vocab_chunk_tokens"]),
        rank_vocab_block_size=int(spec["rank_vocab_block_size"]),
        min_k_percents=tuple(int(value) for value in spec["min_k_percents"]),
        local_widths=tuple(int(value) for value in spec["local_widths"]),
        language_calibration_min_rows=int(spec["language_calibration_min_rows"]),
        length_calibration_min_rows=int(spec["length_calibration_min_rows"]),
        fidelity_gate=bool(spec["fidelity_gate"]),
        fidelity_tokens=int(spec["fidelity_tokens"]),
        fidelity_atol=float(spec["fidelity_atol"]),
        device="auto",
        move_model=False,
    )
    domains = [config["benchmark"]["domain_label_for_label_free_calibration"]] * len(texts)
    result = score_samples_detailed(model, tokenizer, texts, domains, runtime)
    features = result.features.sort_values("sample_index").reset_index(drop=True)
    if features.sample_index.tolist() != list(range(len(texts))):
        raise RuntimeError("target output feature order changed")
    required = ("best_local_64__max", "score_logp_mean__mean", "min_kpp_zselect_10__max", "mean_log_rank__mean")
    missing = [name for name in required if name not in features]
    if missing:
        raise RuntimeError(f"target output feature schema missing: {missing}")
    scores = {
        "T_TARGET": np.asarray(result.scores, dtype=np.float64),
        "LOCAL64_TARGET": features["best_local_64__max"].to_numpy(float),
        "LOSS_TARGET": features["score_logp_mean__mean"].to_numpy(float),
        "MINKPP10_TARGET": features["min_kpp_zselect_10__max"].to_numpy(float),
        "NEG_LOGRANK_TARGET": -features["mean_log_rank__mean"].to_numpy(float),
    }
    return scores, dict(result.manifest)


def conservative_counts(y, score, target_fpr: float):
    import numpy as np
    y = np.asarray(y, dtype=int)
    values = np.asarray(score, dtype=float)
    negatives = np.sort(values[y == 0])[::-1]
    allowed = max(1, int(math.floor(target_fpr * len(negatives))))
    threshold = float(negatives[allowed - 1])
    detected = values > threshold
    return {
        "target_fpr": float(target_fpr),
        "allowed_boundary_rank": int(allowed),
        "threshold": threshold,
        "fp": int(np.sum((y == 0) & detected)),
        "tp": int(np.sum((y == 1) & detected)),
        "detected": detected,
    }


def metric_block(y, score, fpr_targets):
    import numpy as np
    from poisoned_chalice.evaluation import low_fpr_metrics

    values = np.asarray(score, dtype=float)
    result = dict(low_fpr_metrics(y, values))
    result["conservative"] = {}
    for rate in fpr_targets:
        counts = conservative_counts(y, values, float(rate))
        result["conservative"][str(rate)] = {key: value for key, value in counts.items() if key != "detected"}
    return result


def content_oof_diagnostic(texts: list[str], y, config: dict[str, Any]):
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import FeatureUnion

    y = np.asarray(y, dtype=int)
    evaluation = config["evaluation"]
    splitter = StratifiedKFold(n_splits=int(evaluation["content_oof_folds"]), shuffle=True, random_state=int(evaluation["content_oof_seed"]))
    prediction = np.full(len(texts), np.nan, dtype=np.float64)
    union = FeatureUnion([
        ("char", HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=2**16, alternate_sign=True, norm="l2", lowercase=False)),
        ("token", HashingVectorizer(analyzer="word", ngram_range=(1, 2), n_features=2**16, alternate_sign=True, norm="l2", lowercase=False, token_pattern=r"(?u)\b\w+\b")),
    ])
    X = union.transform(texts)
    for fold, (fit, hold) in enumerate(splitter.split(np.arange(len(texts)), y)):
        model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=int(evaluation["content_oof_seed"]) + fold)
        model.fit(X[fit], y[fit])
        prediction[hold] = model.predict_proba(X[hold])[:, 1]
    if not np.isfinite(prediction).all():
        raise RuntimeError("content OOF coverage failed")
    return prediction


def paired_bootstrap(frame, score_names: list[str], baseline: str, config: dict[str, Any]):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y = frame.label.to_numpy(int)
    scores = {name: frame[name].to_numpy(float) for name in score_names}
    evaluation = config["evaluation"]
    rng = np.random.default_rng(int(evaluation["paired_bootstrap_seed"]))
    strata = [np.where(y == label)[0] for label in (0, 1)]
    comparisons = [name for name in ("LOCAL64_TARGET", "GR_TARGET", "HR_REF", "F_T_HR", "F_T_GR", "F_T_HR_GR") if name in scores and name != baseline]
    records = {name: {"delta_auc": [], "delta_tpr_01": [], "delta_tp_01": []} for name in comparisons}
    for _ in range(int(evaluation["paired_bootstrap_replicates"])):
        sampled = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in strata])
        ys = y[sampled]
        base_values = scores[baseline][sampled]
        base_auc = float(roc_auc_score(ys, base_values))
        base_counts = conservative_counts(ys, base_values, 0.01)
        base_tp = int(base_counts["tp"])
        member_count = int(np.sum(ys == 1))
        base_tpr = base_tp / member_count
        for name in comparisons:
            values = scores[name][sampled]
            counts = conservative_counts(ys, values, 0.01)
            tp = int(counts["tp"])
            records[name]["delta_auc"].append(float(roc_auc_score(ys, values)) - base_auc)
            records[name]["delta_tpr_01"].append(tp / member_count - base_tpr)
            records[name]["delta_tp_01"].append(tp - base_tp)
    output = {}
    for name, metrics in records.items():
        output[name + "_minus_" + baseline] = {}
        for metric, values in metrics.items():
            array = np.asarray(values, dtype=float)
            output[name + "_minus_" + baseline][metric] = {
                "mean": float(array.mean()),
                "lower_95": float(np.quantile(array, 0.025)),
                "upper_95": float(np.quantile(array, 0.975)),
            }
    return output


def evaluate_after_seal(target_frame, predictions_path: Path, prediction_sha: str, config: dict[str, Any]):
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    if sha256_file(predictions_path) != prediction_sha:
        raise RuntimeError("prediction artifact changed before label join")
    labels = [json.loads(line) for line in LABELS_TMP.read_text(encoding="utf-8").splitlines() if line.strip()]
    label_frame = pd.DataFrame(labels)[["sample_id", "label"]]
    if len(label_frame) != 2000 or label_frame.sample_id.duplicated().any() or label_frame.label.value_counts().to_dict() != {0: 1000, 1: 1000}:
        raise RuntimeError("post-seal label identity/balance changed")
    predictions = pd.read_csv(predictions_path)
    frame = predictions.merge(label_frame, on="sample_id", how="inner", validate="one_to_one")
    if len(frame) != len(predictions):
        raise RuntimeError("post-seal label join coverage failed")
    text_lookup = target_frame.set_index("sample_id").content
    texts = [str(text_lookup.loc[sample_id]) for sample_id in frame.sample_id]
    y = frame.label.to_numpy(int)
    score_names = list(config["evaluation"]["score_names"])
    metrics = {name: metric_block(y, frame[name].to_numpy(float), config["evaluation"]["fpr_targets"]) for name in score_names}

    content_oof = content_oof_diagnostic(texts, y, config)
    diagnostics = {
        "C_TARGET_OOF": metric_block(y, content_oof, config["evaluation"]["fpr_targets"]),
        "C_TARGET_OOF_role": "target-label-fit post-seal confounding diagnostic only; excluded from candidates and all fusions",
        "signed_length_auc_direction": "higher length is member; fixed before labels",
        "PYTHIA_TOKEN_COUNT": metric_block(y, frame["PYTHIA_TOKEN_COUNT"].to_numpy(float), config["evaluation"]["fpr_targets"]),
        "CHAR_COUNT": metric_block(y, frame["CHAR_COUNT"].to_numpy(float), config["evaluation"]["fpr_targets"]),
        "WORD_COUNT": metric_block(y, frame["WORD_COUNT"].to_numpy(float), config["evaluation"]["fpr_targets"]),
        "score_length_spearman": {},
    }
    for name in score_names:
        rho, pvalue = spearmanr(frame[name].to_numpy(float), frame["PYTHIA_TOKEN_COUNT"].to_numpy(float))
        diagnostics["score_length_spearman"][name] = {"rho": float(rho), "pvalue": float(pvalue)}

    baseline = config["evaluation"]["primary_baseline"]
    candidate = config["evaluation"]["primary_candidate"]
    base_mask = conservative_counts(y, frame[baseline].to_numpy(float), 0.01)["detected"]
    candidate_mask = conservative_counts(y, frame[candidate].to_numpy(float), 0.01)["detected"]
    member = y == 1
    bootstrap = paired_bootstrap(frame, score_names, baseline, config)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "prediction_sha256_verified_before_label_load": prediction_sha,
        "labels_loaded_after_prediction_seal": True,
        "candidate_metrics": metrics,
        "post_seal_confounding_diagnostics": diagnostics,
        "paired_bootstrap": bootstrap,
        "detection_delta_at_1pct": {
            "primary_candidate": candidate,
            "primary_baseline": baseline,
            "new_tp": int(np.sum(member & candidate_mask & ~base_mask)),
            "lost_tp": int(np.sum(member & base_mask & ~candidate_mask)),
        },
        "bootstrap_scope": "paired label-stratified sampling uncertainty on this Pythia/MIMIR development cohort; not a model-transfer generalization CI",
        "threshold_note": "all conservative low-FPR thresholds are recomputed from sampled negatives and use strict score>threshold",
        "official_score_claimed": False,
        "s_proxy_computed": False,
        "target_label_fit_used_by_candidate": False,
        "content_oof_is_candidate": False,
    }


def run(config: dict[str, Any]) -> None:
    import joblib
    import numpy as np
    import pandas as pd
    import torch

    validate_config(config, runtime=True)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    target_frame, input_manifest = prepare_target_inputs(config)
    write_json(OUTPUT / "input_manifest.json", input_manifest)
    texts = target_frame.content.astype(str).tolist()

    bundle_root, bundle_manifest, bundle_sha = locate_source_bundle(config)
    hr_bundle = load_hr_bundle_minimal(bundle_root)
    gr_bundle = load_gr_bundle_minimal(bundle_root)
    c1_union, c1_model = joblib.load(bundle_root / "c1_source_model.joblib")

    pythia_repo = config["target_model"]["repository"]
    pythia_revision = config["target_model"]["revision"]
    model, tokenizer = load_model_and_tokenizer(pythia_repo, pythia_revision)
    if int(model.config.num_hidden_layers) != 32 or int(model.config.hidden_size) != 2560 or int(model.config.max_position_embeddings) != 2048:
        raise RuntimeError("loaded Pythia exact revision architecture changed")
    target_output_scores, target_output_manifest = score_target_output(model, tokenizer, texts, config)
    gr_target, pythia_token_counts, gr_manifest = collect_target_gr(model, tokenizer, texts, gr_bundle, config)
    unload_model(model, tokenizer)

    reference_repo = config["reference_hidden"]["reference_model_id"]
    reference_revision = config["reference_hidden"]["reference_model_revision"]
    ref_model, ref_tokenizer = load_model_and_tokenizer(reference_repo, reference_revision)
    hr_ref, reference_token_counts, hr_manifest = collect_reference_hr(ref_model, ref_tokenizer, texts, hr_bundle, config)
    unload_model(ref_model, ref_tokenizer)

    c1_ref = c1_model.predict_proba(c1_union.transform(texts))[:, 1].astype(np.float64)
    scores = dict(target_output_scores)
    scores["GR_TARGET"] = np.asarray(gr_target, dtype=np.float64)
    scores["HR_REF"] = np.asarray(hr_ref, dtype=np.float64)
    scores["C1_REF"] = np.asarray(c1_ref, dtype=np.float64)
    scores["F_T_HR"] = equal_rank_fusion(scores["T_TARGET"], scores["HR_REF"])
    scores["F_T_GR"] = equal_rank_fusion(scores["T_TARGET"], scores["GR_TARGET"])
    scores["F_T_HR_GR"] = equal_rank_fusion(scores["T_TARGET"], scores["HR_REF"], scores["GR_TARGET"])

    predictions = pd.DataFrame({"sample_id": target_frame.sample_id.astype(str)})
    for name in config["prediction_columns"][1:]:
        if name == "PYTHIA_TOKEN_COUNT":
            values = np.asarray(pythia_token_counts, dtype=np.float64)
        elif name == "CHAR_COUNT":
            values = np.asarray([len(text) for text in texts], dtype=np.float64)
        elif name == "WORD_COUNT":
            values = np.asarray([len(text.split()) for text in texts], dtype=np.float64)
        else:
            values = np.asarray(scores[name], dtype=np.float64)
        if values.shape != (len(predictions),) or not np.isfinite(values).all():
            raise RuntimeError(f"prediction column invalid: {name}")
        predictions[name] = values
    if predictions.columns.tolist() != config["prediction_columns"]:
        raise RuntimeError("prediction schema changed")
    predictions.to_csv(PREDICTIONS, index=False)
    prediction_sha = sha256_file(PREDICTIONS)
    (OUTPUT / "predictions_label_free.sha256").write_text(prediction_sha + "  predictions_label_free.csv\n", encoding="utf-8")

    pre_eval_manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "predictions_sealed_labels_not_loaded",
        "source_bundle_manifest_sha256": bundle_sha,
        "source_bundle_top_layers": hr_bundle["top_layers"],
        "source_bundle_reload_fidelity": bundle_manifest.get("reload_fidelity"),
        "target_model": {"repository": pythia_repo, "revision": pythia_revision},
        "reference_model": {"repository": reference_repo, "revision": reference_revision},
        "target_output_manifest": target_output_manifest,
        "target_gr_manifest": gr_manifest,
        "reference_hr_manifest": hr_manifest,
        "pythia_token_count": {"min": int(min(pythia_token_counts)), "median": float(np.median(pythia_token_counts)), "max": int(max(pythia_token_counts))},
        "reference_token_count": {"min": int(min(reference_token_counts)), "median": float(np.median(reference_token_counts)), "max": int(max(reference_token_counts))},
        "prediction_sha256": prediction_sha,
        "target_labels_loaded": False,
        "target_label_candidate_fit": False,
        "target_label_sign_selection": False,
        "target_label_layer_selection": False,
        "target_label_fusion_weight_search": False,
        "competition_submission": False,
    }
    write_json(OUTPUT / "pre_evaluation_manifest.json", pre_eval_manifest)

    evaluation = evaluate_after_seal(target_frame, PREDICTIONS, prediction_sha, config)
    write_json(OUTPUT / "evaluation.json", evaluation)
    final_manifest = {
        **pre_eval_manifest,
        "status": "complete",
        "target_labels_loaded": True,
        "labels_loaded_only_after_prediction_sha256": True,
        "evaluation_sha256": sha256_file(OUTPUT / "evaluation.json"),
        "runtime_seconds": time.perf_counter() - started,
        "sealed_confirmation_environment_consumed": False,
        "automatic_compute_retries": 0,
        "competition_submission": False,
    }
    write_json(OUTPUT / "run_manifest.json", final_manifest)
    LABELS_TMP.unlink(missing_ok=True)
    print(
        "PYTHIA_MIMIR_DEVELOPMENT_V1 COMPLETE "
        f"rows={len(predictions)} prediction_sha256={prediction_sha} "
        f"bundle_sha256={bundle_sha} labels_after_seal=true retries=0 submissions=0"
    )


def self_test(config: dict[str, Any]) -> None:
    validate_config(config, runtime=False)
    if config["source_bundle"]["bundle_manifest_sha256"] is not None:
        raise RuntimeError("pre-upstream static config unexpectedly contains a source bundle SHA")
    if config["evaluation"]["primary_baseline"] != "T_TARGET" or config["evaluation"]["primary_candidate"] != "F_T_HR_GR":
        raise RuntimeError("primary comparison changed")
    if config["candidate_scores"]["F_T_HR"] != "0.5 global midrank(T_TARGET) + 0.5 global midrank(HR_REF)":
        raise RuntimeError("fixed fusion contract changed")
    print("PYTHIA_MIMIR_DEVELOPMENT_V1_SELF_TEST PASS runtime_blocked_on_bundle_sha=true labels_used=0 submissions=0")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.self_test:
        self_test(config)
        return 0
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
