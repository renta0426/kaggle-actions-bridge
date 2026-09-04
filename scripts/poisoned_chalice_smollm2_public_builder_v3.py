"""Build the PAT-free frozen SmolLM2 transfer notebook from public sources only.

This is a clean-room bridge implementation of the already-frozen experiment contract.
It reconstructs the exact 2,000-row transfer cohort from two pinned public sources,
then emits only sample_id/content hash/language/index into the Kaggle GPU notebook.
No private research repository content is fetched or embedded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import textwrap

import nbformat
import numpy as np
import pandas as pd


DATASET_ID = "Poisoned-Chalice/ICSE-2027-public"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
UPSTREAM_REPOSITORY = "https://github.com/Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026"
UPSTREAM_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
UPSTREAM_EVAL_PATH = "data/7b_train_test/eval_results.parquet"
UPSTREAM_EVAL_SHA256 = "6e8559279e8ebd94ec8c25561c91d38c454a4e109ea0813b07864ff6a66d4068"
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"
MODEL_REVISION = "effd688a12921b4cc83e3312b6feb579f70f9c71"
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
SEED = 2027
ROWS_PER_LANGUAGE_LABEL = 200
EXPECTED_ROWS = 2000
EXPECTED_CURRENT_OVERLAP = 23
EXPECTED_UPSTREAM_NULL_CONTENT = 5
SAMPLES_PER_SHARD = 100
TARGET = "renta0426/smollm2-transfer-v1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_current_10k_hashes() -> set[str]:
    from datasets import load_dataset

    hashes: set[str] = set()
    total = 0
    for language in LANGUAGES:
        frame = load_dataset(
            DATASET_ID,
            language,
            revision=DATASET_REVISION,
            split="train",
        ).to_pandas()
        frame = frame.copy()
        frame["language"] = language
        normalized = frame["membership"].astype("string").str.lower()
        frame["label"] = normalized.map({"member": 1, "non-member": 0, "non_member": 0})
        if frame["label"].isna().any():
            raise RuntimeError(f"current public membership encoding changed: {language}")
        pieces = []
        for label in (0, 1):
            pool = frame[frame.label == label]
            if len(pool) < 1000:
                raise RuntimeError(f"insufficient current public rows: {language}/{label}")
            pieces.append(pool.sample(n=1000, random_state=SEED + label))
        sampled = pd.concat(pieces, ignore_index=True)
        if len(sampled) != 2000:
            raise RuntimeError(f"current 10k reconstruction changed: {language}")
        values = sampled["content"].map(sha256_text)
        if values.duplicated().any():
            raise RuntimeError(f"duplicate current content in reconstructed 10k: {language}")
        hashes.update(values.tolist())
        total += len(values)
    if total != 10000 or len(hashes) != 10000:
        raise RuntimeError(f"current 10k hash-set contract changed: total={total} unique={len(hashes)}")
    return hashes


def fetch_upstream_eval(root: Path) -> Path:
    repo = root / "sersem"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", UPSTREAM_REPOSITORY + ".git"],
        check=True,
    )
    subprocess.run(
        [
            "git", "-C", str(repo), "-c", "credential.helper=", "fetch", "-q",
            "--depth=1", "origin", UPSTREAM_COMMIT,
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "sparse-checkout", "init", "--cone"], check=True)
    subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "data"], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {actual}")
    path = repo / UPSTREAM_EVAL_PATH
    if not path.is_file():
        raise RuntimeError("pinned upstream evaluation artifact is missing")
    observed = sha256_file(path)
    if observed != UPSTREAM_EVAL_SHA256:
        raise RuntimeError(f"upstream evaluation SHA256 mismatch: {observed}")
    return path


def reconstruct_prediction_manifest() -> pd.DataFrame:
    excluded = load_current_10k_hashes()
    with tempfile.TemporaryDirectory(prefix="pc-smollm2-public-") as tmp:
        source_path = fetch_upstream_eval(Path(tmp))
        source = pd.read_parquet(source_path)
    required = {"content", "language", "is_member", "membership"}
    missing = required.difference(source.columns)
    if missing:
        raise RuntimeError(f"upstream columns changed: {sorted(missing)}")
    null_count = int(source["content"].isna().sum())
    if null_count != EXPECTED_UPSTREAM_NULL_CONTENT:
        raise RuntimeError(f"upstream null-content count changed: {null_count}")
    frame = source.dropna(subset=["content"]).copy()
    frame["content_sha256"] = frame["content"].map(sha256_text)
    if frame["content_sha256"].duplicated().any():
        raise RuntimeError("upstream non-null content is no longer unique")
    overlap = int(frame["content_sha256"].isin(excluded).sum())
    if overlap != EXPECTED_CURRENT_OVERLAP:
        raise RuntimeError(f"current/upstream overlap contract changed: {overlap}")
    frame = frame[~frame["content_sha256"].isin(excluded)].copy()
    frame["label"] = frame["is_member"].astype(int)
    expected = frame["membership"].map({"member": 1, "non-member": 0})
    if expected.isna().any() or not np.array_equal(expected.to_numpy(), frame["label"].to_numpy()):
        raise RuntimeError("upstream membership encoding is inconsistent")

    pieces = []
    for language in LANGUAGES:
        for label in (0, 1):
            pool = frame[(frame.language == language) & (frame.label == label)]
            if len(pool) < ROWS_PER_LANGUAGE_LABEL:
                raise RuntimeError(f"insufficient transfer rows: {language}/{label}")
            pieces.append(pool.sort_values("content_sha256").head(ROWS_PER_LANGUAGE_LABEL))
    selected = pd.concat(pieces, ignore_index=True)
    selected["sample_id"] = (
        "previous-" + selected["language"].str.lower() + "-" + selected["content_sha256"]
    )
    if len(selected) != EXPECTED_ROWS or not selected["sample_id"].is_unique:
        raise RuntimeError("selected transfer cohort identity changed")
    balance = selected.groupby(["language", "label"]).size()
    if not balance.eq(ROWS_PER_LANGUAGE_LABEL).all() or len(balance) != 10:
        raise RuntimeError("selected transfer cohort balance changed")

    prediction = (
        selected[["sample_id", "content_sha256", "language"]]
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    prediction["sample_index"] = np.arange(len(prediction), dtype=np.int64)
    if prediction["content_sha256"].duplicated().any():
        raise RuntimeError("prediction hash identity changed")
    return prediction


def stage2_runtime_source() -> str:
    # Fresh minimal implementation of the frozen scientific contract.
    return r'''
from dataclasses import dataclass, asdict
import math
import time

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrozenRuntime:
    max_length: int = 768
    max_batch_tokens: int = 1536
    sequence_chunk_tokens: int = 32
    rank_vocab_block_size: int = 8192
    min_k_fraction: float = 0.10
    local_span_width: int = 64
    language_rank_min_rows: int = 20
    fidelity_token_limit: int = 32
    fidelity_atol: float = 1e-6
    deadline_seconds: float = 10800.0


def _windows(n, width):
    if n <= width:
        return [("whole", 0)]
    candidates = [
        ("prefix", 0),
        ("middle", (n - width) // 2),
        ("suffix", n - width),
    ]
    seen = set()
    out = []
    for name, start in candidates:
        if start not in seen:
            seen.add(start)
            out.append((name, start))
    return out


def _records(tokenizer, texts, languages, cfg):
    out = []
    for sample_index, (text, language) in enumerate(zip(texts, languages)):
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise RuntimeError("unexpected batched tokenizer result")
            ids = ids[0]
        ids = [int(x) for x in ids]
        if len(ids) < 2:
            raise RuntimeError(f"sample {sample_index} has fewer than two tokens")
        for position, start in _windows(len(ids), cfg.max_length):
            chunk = ids[start:start + cfg.max_length]
            out.append({
                "sample_index": sample_index,
                "language": language,
                "position": position,
                "window_start": start,
                "file_token_count": len(ids),
                "window_token_count": len(chunk),
                "input_ids": chunk,
            })
    return out


def _batches(records, max_tokens):
    ordered = sorted(
        records,
        key=lambda x: (x["window_token_count"], x["sample_index"], x["window_start"]),
    )
    batch = []
    width = 0
    for record in ordered:
        candidate = max(width, int(record["window_token_count"]))
        if batch and candidate * (len(batch) + 1) > max_tokens:
            yield batch
            batch = []
            width = 0
        batch.append(record)
        width = max(width, int(record["window_token_count"]))
    if batch:
        yield batch


def _blocked(logits, targets, block):
    import torch
    values = logits.float()
    correct = values.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    logp = correct - torch.logsumexp(values, dim=-1)
    z = (correct - values.mean(dim=-1)) / values.std(
        dim=-1, correction=0
    ).clamp_min(1e-6)
    rank = torch.ones_like(correct, dtype=torch.int64)
    threshold = correct.unsqueeze(-1)
    for left in range(0, values.shape[-1], block):
        right = min(left + block, values.shape[-1])
        rank += (values[..., left:right] > threshold).sum(dim=-1)
    return logp, z, rank


def _exact(logits, targets):
    import torch
    values = logits.float()
    correct = values.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    logp = correct - torch.logsumexp(values, dim=-1)
    z = (correct - values.mean(dim=-1)) / values.std(
        dim=-1, correction=0
    ).clamp_min(1e-6)
    rank = 1 + (values > correct.unsqueeze(-1)).sum(dim=-1)
    return logp, z, rank


def _local_best(values, width):
    if len(values) <= width:
        return float(values.mean())
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return float(((prefix[width:] - prefix[:-width]) / width).max())


def _window_features(logp, z, rank, cfg):
    k = max(1, math.ceil(len(z) * cfg.min_k_fraction))
    chosen = np.argpartition(z, k - 1)[:k]
    return {
        "window_logp_mean": float(logp.mean()),
        "window_minkpp": float(z[chosen].mean()),
        "window_local": _local_best(logp, cfg.local_span_width),
        "window_mean_log_rank": float(np.log1p(rank).mean()),
    }


def _sync(torch):
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


def extract_frozen_features(model, tokenizer, texts, languages, cfg):
    import torch
    if len(texts) != len(languages) or not texts:
        raise RuntimeError("invalid frozen scoring inputs")
    records = _records(tokenizer, texts, languages, cfg)
    batches = list(_batches(records, cfg.max_batch_tokens))
    device = model.get_input_embeddings().weight.device
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise RuntimeError("tokenizer has no padding/EOS token")
    started = time.perf_counter()
    rows = []
    fidelity = {
        "passed": None,
        "token_count": 0,
        "max_logp_absolute_difference": None,
        "max_z_absolute_difference": None,
        "rank_exact_match": None,
        "atol": cfg.fidelity_atol,
    }
    profile = {
        "samples": len(texts),
        "windows": len(records),
        "batches": len(batches),
        "target_tokens": 0,
        "forward_seconds": 0.0,
        "reduction_seconds": 0.0,
        "transfer_seconds": 0.0,
        "wall_seconds": None,
        "peak_allocated_bytes": 0,
    }
    fidelity_done = False
    with torch.inference_mode():
        for batch_index, batch in enumerate(batches):
            if time.perf_counter() - started >= cfg.deadline_seconds:
                raise TimeoutError(f"deadline exceeded before batch {batch_index}")
            width = max(int(x["window_token_count"]) for x in batch)
            input_ids = torch.full(
                (len(batch), width), int(pad), dtype=torch.long, device=device
            )
            attention = torch.zeros_like(input_ids)
            lengths = []
            for i, record in enumerate(batch):
                ids = torch.as_tensor(record["input_ids"], dtype=torch.long, device=device)
                input_ids[i, :len(ids)] = ids
                attention[i, :len(ids)] = 1
                lengths.append(len(ids) - 1)

            _sync(torch)
            phase = time.perf_counter()
            logits = model(
                input_ids=input_ids, attention_mask=attention, use_cache=False
            ).logits[:, :-1]
            targets = input_ids[:, 1:].to(logits.device)
            _sync(torch)
            profile["forward_seconds"] += time.perf_counter() - phase

            stats = torch.empty(
                (len(batch), logits.shape[1], 3), dtype=torch.float32, device=logits.device
            )
            phase = time.perf_counter()
            for left in range(0, logits.shape[1], cfg.sequence_chunk_tokens):
                right = min(left + cfg.sequence_chunk_tokens, logits.shape[1])
                logp, z, rank = _blocked(
                    logits[:, left:right], targets[:, left:right], cfg.rank_vocab_block_size
                )
                stats[:, left:right, 0] = logp
                stats[:, left:right, 1] = z
                stats[:, left:right, 2] = rank.float()

            if not fidelity_done:
                n = min(lengths[0], cfg.fidelity_token_limit)
                a_logp, a_z, a_rank = _exact(logits[0:1, :n], targets[0:1, :n])
                b_logp, b_z, b_rank = _blocked(
                    logits[0:1, :n], targets[0:1, :n], cfg.rank_vocab_block_size
                )
                d_logp = float((a_logp - b_logp).abs().max().detach().cpu())
                d_z = float((a_z - b_z).abs().max().detach().cpu())
                rank_ok = bool(torch.equal(a_rank, b_rank))
                passed = d_logp <= cfg.fidelity_atol and d_z <= cfg.fidelity_atol and rank_ok
                fidelity.update({
                    "passed": bool(passed),
                    "token_count": int(n),
                    "max_logp_absolute_difference": d_logp,
                    "max_z_absolute_difference": d_z,
                    "rank_exact_match": rank_ok,
                })
                if not passed:
                    raise RuntimeError(f"fidelity gate failed: {fidelity}")
                fidelity_done = True
            _sync(torch)
            profile["reduction_seconds"] += time.perf_counter() - phase

            phase = time.perf_counter()
            host = stats.detach().cpu().numpy()
            _sync(torch)
            profile["transfer_seconds"] += time.perf_counter() - phase
            for i, (record, n) in enumerate(zip(batch, lengths)):
                values = host[i, :n]
                row = {k: v for k, v in record.items() if k != "input_ids"}
                row.update(
                    _window_features(
                        values[:, 0],
                        values[:, 1],
                        values[:, 2].astype(np.int64, copy=False),
                        cfg,
                    )
                )
                rows.append(row)
            profile["target_tokens"] += int(sum(lengths))
            del logits, targets, stats, host, input_ids, attention
            if time.perf_counter() - started > cfg.deadline_seconds:
                raise TimeoutError(f"deadline exceeded after batch {batch_index}")

    windows = pd.DataFrame(rows)
    aggregate = []
    for sample_index, group in windows.groupby("sample_index", sort=True):
        aggregate.append({
            "sample_index": int(sample_index),
            "language": str(group["language"].iloc[0]),
            "token_count": int(group["file_token_count"].max()),
            "window_count": int(len(group)),
            "loss_multiwindow": float(group["window_logp_mean"].max()),
            "standard_minkpp": float(group["window_minkpp"].max()),
            "best_local_span": float(group["window_local"].max()),
            "neg_mean_log_rank": float(-group["window_mean_log_rank"].mean()),
        })
    frame = pd.DataFrame(aggregate).sort_values("sample_index").reset_index(drop=True)
    if len(frame) != len(texts) or frame.sample_index.tolist() != list(range(len(texts))):
        raise RuntimeError("frozen raw feature coverage/order failure")
    if not np.isfinite(
        frame.select_dtypes(include=[np.number]).drop(columns=["sample_index"]).to_numpy(float)
    ).all():
        raise RuntimeError("non-finite frozen raw feature")
    profile["wall_seconds"] = time.perf_counter() - started
    if torch.cuda.is_available():
        profile["peak_allocated_bytes"] = int(
            max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
        )
    return frame, fidelity, profile


def _pct(frame, values, min_rows):
    series = pd.Series(np.asarray(values, dtype=float), index=frame.index)
    result = series.rank(method="average", pct=True)
    for _, indices in frame.groupby("language", sort=False).groups.items():
        indices = list(indices)
        if len(indices) >= min_rows:
            result.loc[indices] = series.loc[indices].rank(method="average", pct=True)
    return result.to_numpy(float)


def frozen_v1(features, cfg):
    work = features.sort_values("sample_index").reset_index(drop=True).copy()
    for name, source in (
        ("rank_loss", "loss_multiwindow"),
        ("rank_minkpp", "standard_minkpp"),
        ("rank_local_span", "best_local_span"),
    ):
        work[name] = _pct(work, work[source].to_numpy(float), cfg.language_rank_min_rows)
    score = work[["rank_loss", "rank_minkpp", "rank_local_span"]].to_numpy(float).mean(axis=1)
    work["membership_score_v1"] = score
    return score, work


def _residual(work, column):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    design = pd.DataFrame({
        "language": work.language.astype(str),
        "log_token_count": np.log1p(work.token_count.to_numpy(float)),
        "log_window_count": np.log1p(work.window_count.to_numpy(float)),
    })
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), ["log_token_count", "log_window_count"]),
        ("language", OneHotEncoder(handle_unknown="ignore"), ["language"]),
    ])
    model = make_pipeline(prep, LinearRegression())
    target = work[column].to_numpy(float)
    model.fit(design, target)
    return target - model.predict(design)


def frozen_v2(features, cfg):
    forbidden = {
        "label", "membership", "is_member", "lumia_score", "sersem_score",
        "previous_model_score", "prior_model_score", "hidden_label",
    }
    leaked = sorted(forbidden.intersection(features.columns))
    if leaked:
        raise RuntimeError(f"forbidden target fields crossed scorer boundary: {leaked}")
    work = features.sort_values("sample_index").reset_index(drop=True).copy()
    if work.sample_index.tolist() != list(range(len(work))):
        raise RuntimeError("v2 sample order changed")
    work["minkpp_length_residual"] = _residual(work, "standard_minkpp")
    work["logrank_length_residual"] = _residual(work, "neg_mean_log_rank")
    work["rank_minkpp_length_residual"] = _pct(
        work, work["minkpp_length_residual"].to_numpy(float), cfg.language_rank_min_rows
    )
    work["rank_logrank_length_residual"] = _pct(
        work, work["logrank_length_residual"].to_numpy(float), cfg.language_rank_min_rows
    )
    score = 0.5 * (
        work["rank_minkpp_length_residual"].to_numpy(float)
        + work["rank_logrank_length_residual"].to_numpy(float)
    )
    work["membership_score_v2"] = score
    return score, work
'''


def build_notebook(prediction: pd.DataFrame) -> nbformat.NotebookNode:
    prediction_records = json.loads(prediction.to_json(orient="records"))
    runtime_source = stage2_runtime_source()

    setup = f'''
from dataclasses import asdict
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import subprocess
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERIMENT = "smollm2-transfer-v1"
UPSTREAM_REPOSITORY = {UPSTREAM_REPOSITORY!r}
UPSTREAM_COMMIT = {UPSTREAM_COMMIT!r}
UPSTREAM_EVAL_PATH = {UPSTREAM_EVAL_PATH!r}
UPSTREAM_SHA256 = {UPSTREAM_EVAL_SHA256!r}
MODEL_ID = {MODEL_ID!r}
MODEL_REVISION = {MODEL_REVISION!r}
EXPECTED_ROWS = {EXPECTED_ROWS}
SAMPLES_PER_SHARD = {SAMPLES_PER_SHARD}
PREDICTION_MANIFEST = pd.DataFrame({prediction_records!r})
OUTPUT = Path("/kaggle/working/smollm2_transfer_v1")
PARTS = OUTPUT / "parts"
PARTS.mkdir(parents=True, exist_ok=True)
RUNTIME = FrozenRuntime()

if set(PREDICTION_MANIFEST.columns) != {{"sample_id", "content_sha256", "language", "sample_index"}}:
    raise RuntimeError("prediction manifest schema changed")
if len(PREDICTION_MANIFEST) != EXPECTED_ROWS:
    raise RuntimeError("prediction manifest row count changed")
if PREDICTION_MANIFEST.sample_index.tolist() != list(range(EXPECTED_ROWS)):
    raise RuntimeError("prediction manifest order changed")
print({{"experiment": EXPERIMENT, "rows": EXPECTED_ROWS, "model": MODEL_ID, "revision": MODEL_REVISION}})
'''

    source = r'''
upstream_root = Path("/kaggle/working/sersem_upstream")
subprocess.run(["git", "init", "-q", str(upstream_root)], check=True)
subprocess.run(
    ["git", "-C", str(upstream_root), "remote", "add", "origin", UPSTREAM_REPOSITORY + ".git"],
    check=True,
)
subprocess.run(
    ["git", "-C", str(upstream_root), "-c", "credential.helper=", "fetch", "-q", "--depth=1", "origin", UPSTREAM_COMMIT],
    check=True,
)
subprocess.run(["git", "-C", str(upstream_root), "sparse-checkout", "init", "--cone"], check=True)
subprocess.run(["git", "-C", str(upstream_root), "sparse-checkout", "set", "data"], check=True)
subprocess.run(["git", "-C", str(upstream_root), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
actual_commit = subprocess.run(
    ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
if actual_commit != UPSTREAM_COMMIT:
    raise RuntimeError(f"upstream commit mismatch: {actual_commit}")
source_path = upstream_root / UPSTREAM_EVAL_PATH
actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
if actual_sha != UPSTREAM_SHA256:
    raise RuntimeError(f"upstream artifact mismatch: {actual_sha}")

source = pd.read_parquet(source_path, columns=["content", "language"]).dropna(subset=["content"]).copy()
source["content_sha256"] = source.content.map(
    lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
)
selected = source[source.content_sha256.isin(set(PREDICTION_MANIFEST.content_sha256))].copy()
if len(selected) != EXPECTED_ROWS or selected.content_sha256.duplicated().any():
    raise RuntimeError("public transfer content coverage changed")
sample = PREDICTION_MANIFEST.merge(
    selected[["content_sha256", "content", "language"]],
    on="content_sha256",
    how="left",
    validate="one_to_one",
    suffixes=("", "_source"),
)
if sample.content.isna().any() or not (sample.language == sample.language_source).all():
    raise RuntimeError("public transfer content/language mismatch")
sample = sample[["sample_index", "sample_id", "content", "language"]].sort_values(
    "sample_index"
).reset_index(drop=True)
print({"source_commit": actual_commit, "source_sha256": actual_sha, "rows": len(sample)})
'''

    model = r'''
if not torch.cuda.is_available():
    raise RuntimeError("frozen SmolLM2 transfer requires CUDA; TPU is not fidelity-equivalent")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is None:
        raise RuntimeError("SmolLM2 tokenizer has no pad/EOS token")
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    dtype=dtype,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
).to("cuda").eval()
resolved_model = getattr(model.config, "_commit_hash", None)
resolved_tokenizer = tokenizer.init_kwargs.get("_commit_hash")
if resolved_model and resolved_model != MODEL_REVISION:
    raise RuntimeError(f"resolved model revision mismatch: {resolved_model}")
if resolved_tokenizer and resolved_tokenizer != MODEL_REVISION:
    raise RuntimeError(f"resolved tokenizer revision mismatch: {resolved_tokenizer}")
runtime = {
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "dtype": str(dtype),
    "resolved_model_revision": resolved_model,
    "resolved_tokenizer_revision": resolved_tokenizer,
}
print(json.dumps(runtime, indent=2, sort_keys=True))
'''

    run = r'''
run_started = time.perf_counter()
profiles = []
fidelity = None
shard_count = math.ceil(EXPECTED_ROWS / SAMPLES_PER_SHARD)
for shard_index in range(shard_count):
    feature_path = PARTS / f"features.part{shard_index:03d}.parquet"
    profile_path = PARTS / f"profile.part{shard_index:03d}.json"
    fidelity_path = PARTS / f"fidelity.part{shard_index:03d}.json"
    start = shard_index * SAMPLES_PER_SHARD
    stop = min(EXPECTED_ROWS, start + SAMPLES_PER_SHARD)
    shard = sample.iloc[start:stop].copy()
    remaining = RUNTIME.deadline_seconds - (time.perf_counter() - run_started)
    if remaining <= 0:
        raise TimeoutError("global frozen SmolLM2 deadline exceeded")
    shard_runtime = FrozenRuntime(deadline_seconds=remaining)
    raw, shard_fidelity, profile = extract_frozen_features(
        model,
        tokenizer,
        shard.content.tolist(),
        shard.language.tolist(),
        shard_runtime,
    )
    raw["sample_index"] += start
    raw = raw.merge(
        shard[["sample_index", "sample_id"]],
        on="sample_index",
        validate="one_to_one",
    )
    if len(raw) != stop - start or raw.sample_index.tolist() != list(range(start, stop)):
        raise RuntimeError(f"shard coverage changed: {shard_index}")
    tmp = feature_path.with_suffix(".tmp.parquet")
    raw.to_parquet(tmp, index=False)
    tmp.replace(feature_path)
    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if shard_index == 0:
        fidelity = shard_fidelity
        fidelity_path.write_text(
            json.dumps(fidelity, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    profiles.append(profile)
    print({"shard": shard_index, "rows": len(raw), "wall_seconds": profile["wall_seconds"]})
    del raw
    gc.collect()
    torch.cuda.empty_cache()

parts = [pd.read_parquet(path) for path in sorted(PARTS.glob("features.part*.parquet"))]
if len(parts) != shard_count:
    raise RuntimeError(f"expected {shard_count} shards, got {len(parts)}")
raw_features = pd.concat(parts, ignore_index=True).sort_values("sample_index").reset_index(drop=True)
if (
    len(raw_features) != EXPECTED_ROWS
    or not raw_features.sample_id.is_unique
    or raw_features.sample_index.tolist() != list(range(EXPECTED_ROWS))
):
    raise RuntimeError("completed frozen feature coverage changed")
if fidelity is None or fidelity.get("passed") is not True:
    raise RuntimeError("exact/blocked rank fidelity gate did not pass")

v1_scores, v1_frame = frozen_v1(raw_features, RUNTIME)
v2_scores, output = frozen_v2(raw_features, RUNTIME)
output["membership_score_v1"] = v1_scores
for column in ("rank_loss", "rank_minkpp", "rank_local_span"):
    output[column] = v1_frame[column].to_numpy(float)
for forbidden in (
    "label", "membership", "is_member", "lumia_score", "sersem_score",
    "previous_model_score", "prior_model_score", "hidden_label",
):
    if forbidden in output.columns:
        raise RuntimeError(f"forbidden field crossed completed prediction boundary: {forbidden}")
output.to_parquet(OUTPUT / "sample_features.parquet", index=False)

manifest = {
    "status": "complete",
    "experiment": EXPERIMENT,
    "method": "stage2_v1_and_deconfounded_v2_frozen_public_reconstruction",
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "rows": EXPECTED_ROWS,
    "source_repository": UPSTREAM_REPOSITORY,
    "source_commit": UPSTREAM_COMMIT,
    "source_eval_sha256": UPSTREAM_SHA256,
    "runtime": runtime,
    "runtime_config": asdict(RUNTIME),
    "fidelity": fidelity,
    "stage2_v2": {
        "formula": "0.5*within_language_rank(minkpp_length_residual)+0.5*within_language_rank(logrank_length_residual)",
        "weights_searched": False,
        "nuisance_labels_used": False,
    },
    "v1_score_min": float(v1_scores.min()),
    "v1_score_max": float(v1_scores.max()),
    "v2_score_min": float(v2_scores.min()),
    "v2_score_max": float(v2_scores.max()),
    "target_labels_embedded_in_gpu_notebook": False,
    "target_labels_used_for_training_or_normalization": False,
    "previous_model_scores_used": False,
    "hidden_stage1_validation_labels_used": False,
    "public_leaderboard_tuning_used": False,
    "submission_created": False,
    "private_repository_content_used": False,
    "cohort_reconstructed_from_pinned_public_sources": True,
    "wall_seconds": time.perf_counter() - run_started,
}
(OUTPUT / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
'''

    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Frozen SmolLM2-1.7B transfer v1\n\n"
            "PAT-free public-source reconstruction of the already-frozen Stage2-v2 experiment. "
            "Target labels and previous-model scores are absent from the GPU prediction boundary."
        ),
        nbformat.v4.new_code_cell(
            "%pip install -q transformers==5.0.0 accelerate pyarrow scikit-learn"
        ),
        nbformat.v4.new_code_cell(runtime_source),
        nbformat.v4.new_code_cell(textwrap.dedent(setup)),
        nbformat.v4.new_code_cell(textwrap.dedent(source)),
        nbformat.v4.new_code_cell(textwrap.dedent(model)),
        nbformat.v4.new_code_cell(textwrap.dedent(run)),
    ]
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"pc-smollm2-public-v3-{index:02d}"
    nbformat.validate(notebook)
    return notebook


def validate_generated(notebook: nbformat.NotebookNode, prediction: pd.DataFrame) -> None:
    if len(notebook.cells) != 7:
        raise RuntimeError("generated notebook cell count changed")
    joined = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    required = (
        MODEL_ID,
        MODEL_REVISION,
        UPSTREAM_COMMIT,
        UPSTREAM_EVAL_SHA256,
        "standard_minkpp",
        "neg_mean_log_rank",
        "minkpp_length_residual",
        "logrank_length_residual",
        "membership_score_v2",
        "private_repository_content_used",
    )
    for marker in required:
        if marker not in joined:
            raise RuntimeError(f"generated notebook marker missing: {marker}")
    forbidden = (
        "RESEARCH_REPO_READ_TOKEN",
        "KAGGLE_API_TOKEN",
        "competitions submit",
        '"label":',
        '"membership":',
        '"is_member":',
    )
    for marker in forbidden:
        if marker in joined:
            raise RuntimeError(f"generated notebook gained forbidden content: {marker}")
    if len(prediction) != EXPECTED_ROWS:
        raise RuntimeError("prediction manifest row count changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    prediction = reconstruct_prediction_manifest()
    notebook = build_notebook(prediction)
    validate_generated(notebook, prediction)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    notebook_path = args.output_dir / "smollm2-transfer-v1.ipynb"
    nbformat.write(notebook, notebook_path)
    metadata = {
        "id": TARGET,
        "title": "SmolLM2 Transfer V1",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu", "membership-inference", "transfer", "smollm2", "stage2-v2"],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (args.output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_digest = hashlib.sha256(
        prediction[["sample_id", "content_sha256", "language", "sample_index"]]
        .to_json(orient="records")
        .encode("utf-8")
    ).hexdigest()
    print(json.dumps({
        "status": "built",
        "rows": len(prediction),
        "languages": prediction.language.value_counts().sort_index().to_dict(),
        "prediction_manifest_sha256": manifest_digest,
        "private_repository_content_used": False,
        "target_labels_embedded": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
