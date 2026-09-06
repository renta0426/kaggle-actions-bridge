"""Build the fixed 500-row StarCoder2-3B Min-K++ definition pilot."""

from __future__ import annotations

from pathlib import Path
import json
import textwrap

import nbformat


ROOT = Path(__file__).resolve().parents[1]
MINKPP_SOURCE = (ROOT / "src/poisoned_chalice/minkpp_paper.py").read_text(encoding="utf-8")
OUT = ROOT / (
    "notebooks/experiments/stage1-minkpp-paper-pilot-500/"
    "stage1-minkpp-paper-pilot-500.ipynb"
)

config_cell = r"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import gc
import json
import math
import os
import random
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")

@dataclass(frozen=True)
class PilotConfig:
    experiment_id: str = "stage1-minkpp-paper-pilot-500"
    dataset_id: str = "Poisoned-Chalice/ICSE-2027-public"
    dataset_revision: str = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
    model_id: str = "bigcode/starcoder2-3b"
    model_revision: str = "733247c55e3f73af49ce8e9c7949bf14af205928"
    seed: int = 2027
    samples_per_language: int = 100
    max_length: int = 768
    max_source_chars_per_region: int = 9_216
    max_batch_tokens: int = 4_096
    token_reduction_chunk: int = 32
    paper_vocab_block_size: int = 8_192
    paper_variance_floor: float = 1e-12
    output_dir: str = "/kaggle/working/stage1-minkpp-paper-pilot-500"

CONFIG = PilotConfig()
OUTPUT = Path(CONFIG.output_dir)
OUTPUT.mkdir(parents=True, exist_ok=True)
random.seed(CONFIG.seed)
np.random.seed(CONFIG.seed)
torch.manual_seed(CONFIG.seed)
torch.cuda.manual_seed_all(CONFIG.seed)
"""

data_cell = r'''
def membership_to_label(series):
    return series.astype("string").str.lower().map(
        {"member": 1, "non-member": 0, "non_member": 0}
    )


def load_fixed_sample(config):
    pieces = []
    for language in LANGUAGES:
        dataset = load_dataset(
            config.dataset_id,
            language,
            revision=config.dataset_revision,
            split="train",
        )
        frame = dataset.to_pandas().copy()
        frame["language"] = language
        frame["label"] = membership_to_label(frame["membership"])
        if frame["label"].isna().any():
            raise RuntimeError(f"unexpected membership label for {language}")
        for label in (0, 1):
            pool = frame.loc[frame.label == label]
            pieces.append(
                pool.sample(
                    n=config.samples_per_language // 2,
                    random_state=config.seed + label,
                )
            )
        del frame, dataset
        gc.collect()
    sample = pd.concat(pieces, ignore_index=True)
    sample = sample.sort_values("sample_id").reset_index(drop=True)
    if len(sample) != 500 or sample.sample_id.duplicated().any():
        raise RuntimeError("fixed pilot sample invariant failed")
    counts = sample.groupby(["language", "label"]).size()
    if not counts.eq(50).all() or len(counts) != 10:
        raise RuntimeError(f"pilot balance invariant failed: {counts.to_dict()}")
    expected_hash = sample["content"].map(
        lambda value: sha256(value.encode("utf-8")).hexdigest()
    )
    actual_hash = sample["sample_id"].str.rsplit("-", n=1).str[-1]
    if not np.array_equal(expected_hash.to_numpy(), actual_hash.to_numpy()):
        raise RuntimeError("sample_id/content SHA-256 mismatch in pilot sample")
    return sample


def bounded_regions(content, config):
    """Return bounded prefix/middle/suffix Unicode-safe character regions."""
    length = len(content)
    budget = config.max_source_chars_per_region
    if length <= budget:
        return [("whole", 0, length)]
    candidates = [
        ("prefix", 0, min(length, budget)),
        (
            "middle",
            max(0, length // 2 - budget // 2),
            min(length, length // 2 - budget // 2 + budget),
        ),
        ("suffix", max(0, length - budget), length),
    ]
    result, seen = [], set()
    for name, start, stop in candidates:
        key = (start, stop)
        if key not in seen:
            seen.add(key)
            result.append((name, start, stop))
    return result


def bounded_window_records(frame, tokenizer, config):
    records = []
    for row in frame.itertuples(index=False):
        for position, char_start, char_stop in bounded_regions(row.content, config):
            segment = row.content[char_start:char_stop]
            region_ids = tokenizer(
                segment,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            if len(region_ids) < 2:
                continue
            region_token_count = len(region_ids)
            if region_token_count > config.max_length:
                if position == "suffix":
                    token_start = region_token_count - config.max_length
                elif position == "middle":
                    token_start = (region_token_count - config.max_length) // 2
                else:
                    token_start = 0
                ids = region_ids[token_start : token_start + config.max_length]
            else:
                token_start = 0
                ids = region_ids
            records.append(
                {
                    "sample_id": row.sample_id,
                    "language": row.language,
                    "position": position,
                    "char_start": int(char_start),
                    "char_stop": int(char_stop),
                    "source_char_count": int(len(row.content)),
                    "region_token_start": int(token_start),
                    "region_token_count_before_window": int(region_token_count),
                    "window_token_count": int(len(ids)),
                    "input_ids": ids,
                }
            )
    represented = {record["sample_id"] for record in records}
    missing = set(frame["sample_id"]) - represented
    if missing:
        raise RuntimeError(f"{len(missing)} pilot samples have no scorable window")
    return records


def dynamic_batches(records, max_batch_tokens):
    ordered = sorted(
        records,
        key=lambda row: (row["window_token_count"], row["sample_id"], row["position"]),
    )
    batch, max_len = [], 0
    for record in ordered:
        candidate_max = max(max_len, record["window_token_count"])
        if batch and candidate_max * (len(batch) + 1) > max_batch_tokens:
            yield batch
            batch, max_len = [], 0
        batch.append(record)
        max_len = max(max_len, record["window_token_count"])
    if batch:
        yield batch

load_started = time.perf_counter()
sample = load_fixed_sample(CONFIG)
sample[["sample_id", "language", "membership", "label"]].to_parquet(
    OUTPUT / "sample_manifest.parquet", index=False
)
data_load_seconds = time.perf_counter() - load_started
'''

model_cell = r"""
tokenizer = AutoTokenizer.from_pretrained(
    CONFIG.model_id,
    revision=CONFIG.model_revision,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenize_started = time.perf_counter()
records = bounded_window_records(sample, tokenizer, CONFIG)
tokenize_seconds = time.perf_counter() - tokenize_started
window_counts = pd.Series([r["position"] for r in records]).value_counts().to_dict()

model_load_started = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    CONFIG.model_id,
    revision=CONFIG.model_revision,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
).to("cuda").eval()
model_load_seconds = time.perf_counter() - model_load_started

# Exact CPU oracle frozen in scripts/minkpp_definition_reference.py.
oracle_logits = torch.tensor([[[3.0, 1.0, -2.0]]], device="cuda")
oracle_target = torch.tensor([[1]], device="cuda")
oracle = probability_weighted_token_statistics_blocked(
    oracle_logits, oracle_target, vocab_block_size=2,
)
oracle_expected = -2.336452981844921
oracle_abs_error = abs(float(oracle.standardized_log_prob.item()) - oracle_expected)
if oracle_abs_error > 2e-5:
    raise RuntimeError(
        f"paper Min-K++ GPU scorer disagrees with CPU oracle: {oracle_abs_error}"
    )

generator = torch.Generator(device="cuda").manual_seed(CONFIG.seed)
synthetic_logits = torch.randn(2, 5, 97, generator=generator, device="cuda")
synthetic_targets = torch.randint(0, 97, (2, 5), generator=generator, device="cuda")
synthetic_dense = probability_weighted_token_statistics_dense(
    synthetic_logits, synthetic_targets
)
synthetic_blocked = probability_weighted_token_statistics_blocked(
    synthetic_logits, synthetic_targets, vocab_block_size=17,
)
synthetic_z_diff = float(
    (synthetic_dense.standardized_log_prob - synthetic_blocked.standardized_log_prob)
    .abs().max().item()
)
synthetic_var_diff = float(
    (synthetic_dense.probability_weighted_variance -
     synthetic_blocked.probability_weighted_variance).abs().max().item()
)
if synthetic_z_diff > 3e-5 or synthetic_var_diff > 3e-5:
    raise RuntimeError(
        f"dense/blocked Min-K++ mismatch: z={synthetic_z_diff} var={synthetic_var_diff}"
    )
del synthetic_logits, synthetic_targets, synthetic_dense, synthetic_blocked
gc.collect()
torch.cuda.empty_cache()
"""

run_cell = r"""
def best_local(values, width=64):
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= width:
        return float(values.mean())
    sums = np.convolve(values, np.ones(width, dtype=np.float64), mode="valid")
    return float((sums / width).max())


def summarize_window(logp, legacy_z, paper_z):
    result = {
        "mean_logp": float(np.mean(logp)),
        "best_local_64": best_local(logp, 64),
    }
    for fraction in MINK_FRACTIONS:
        suffix = f"{int(round(fraction * 100)):02d}"
        result[f"legacy_uniform_vocab_minkpp_{suffix}"] = bottom_fraction_mean(
            legacy_z, fraction
        )
        result[f"paper_prob_weighted_minkpp_{suffix}"] = bottom_fraction_mean(
            paper_z, fraction
        )
    return result


def synchronize():
    torch.cuda.synchronize()


torch.cuda.reset_peak_memory_stats()
run_started = time.perf_counter()
window_rows = []
token_rows = []
target_tokens = 0
real_dense_blocked_max_z_diff = 0.0
real_dense_blocked_max_var_diff = 0.0
real_parity_checked = False
diag_valid_tokens = 0
diag_variance_clamps = 0
diag_min_variance = math.inf
diag_max_abs_z = 0.0

pad_id = tokenizer.pad_token_id
device = model.get_input_embeddings().weight.device

with torch.inference_mode():
    for batch_index, batch in enumerate(dynamic_batches(records, CONFIG.max_batch_tokens)):
        width = max(record["window_token_count"] for record in batch)
        input_ids = torch.full(
            (len(batch), width), pad_id, dtype=torch.long, device=device
        )
        attention = torch.zeros_like(input_ids)
        valid_lengths = []
        for index, record in enumerate(batch):
            ids = torch.as_tensor(record["input_ids"], dtype=torch.long, device=device)
            input_ids[index, : len(ids)] = ids
            attention[index, : len(ids)] = 1
            valid_lengths.append(len(ids) - 1)

        logits = model(
            input_ids=input_ids, attention_mask=attention, use_cache=False,
        ).logits[:, :-1]
        targets = input_ids[:, 1:].to(logits.device)
        valid = attention[:, 1:].bool().to(logits.device)

        if not real_parity_checked:
            real_logits = logits[:1, : min(4, logits.shape[1])]
            real_targets = targets[:1, : min(4, targets.shape[1])]
            real_valid = valid[:1, : min(4, valid.shape[1])]
            dense = probability_weighted_token_statistics_dense(
                real_logits, real_targets, valid_mask=real_valid
            )
            blocked = probability_weighted_token_statistics_blocked(
                real_logits,
                real_targets,
                valid_mask=real_valid,
                vocab_block_size=CONFIG.paper_vocab_block_size,
            )
            real_dense_blocked_max_z_diff = float(
                (dense.standardized_log_prob - blocked.standardized_log_prob)
                .abs().max().item()
            )
            real_dense_blocked_max_var_diff = float(
                (dense.probability_weighted_variance -
                 blocked.probability_weighted_variance).abs().max().item()
            )
            if (
                real_dense_blocked_max_z_diff > 5e-5
                or real_dense_blocked_max_var_diff > 5e-5
            ):
                raise RuntimeError(
                    "real-logit dense/blocked parity gate failed: "
                    f"z={real_dense_blocked_max_z_diff} "
                    f"var={real_dense_blocked_max_var_diff}"
                )
            real_parity_checked = True
            del dense, blocked, real_logits, real_targets, real_valid

        accum = [
            {"logp": [], "legacy_z": [], "paper_z": [], "paper_variance": []}
            for _ in batch
        ]
        for start in range(0, logits.shape[1], CONFIG.token_reduction_chunk):
            stop = min(start + CONFIG.token_reduction_chunk, logits.shape[1])
            logit_slice = logits[:, start:stop]
            target_slice = targets[:, start:stop]
            valid_slice = valid[:, start:stop]

            dense_values = logit_slice.float()
            correct = dense_values.gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)
            logp = correct - torch.logsumexp(dense_values, dim=-1)
            legacy_z = (
                (correct - dense_values.mean(dim=-1))
                / dense_values.std(dim=-1, correction=0).clamp_min(1e-6)
            )
            del dense_values, correct

            paper = probability_weighted_token_statistics_blocked(
                logit_slice,
                target_slice,
                valid_mask=valid_slice,
                variance_floor=CONFIG.paper_variance_floor,
                vocab_block_size=CONFIG.paper_vocab_block_size,
            )
            diag = paper.diagnostics
            diag_valid_tokens += diag.valid_token_count
            diag_variance_clamps += diag.variance_clamp_count
            if math.isfinite(diag.min_variance_before_clamp):
                diag_min_variance = min(diag_min_variance, diag.min_variance_before_clamp)
            diag_max_abs_z = max(diag_max_abs_z, diag.max_abs_z)

            host_logp = logp.detach().cpu().numpy()
            host_legacy = legacy_z.detach().cpu().numpy()
            host_paper = paper.standardized_log_prob.detach().cpu().numpy()
            host_variance = paper.probability_weighted_variance.detach().cpu().numpy()
            host_valid = valid_slice.detach().cpu().numpy()

            for index in range(len(batch)):
                mask = host_valid[index]
                accum[index]["logp"].append(host_logp[index][mask])
                accum[index]["legacy_z"].append(host_legacy[index][mask])
                accum[index]["paper_z"].append(host_paper[index][mask])
                accum[index]["paper_variance"].append(host_variance[index][mask])

            del (
                logit_slice, target_slice, valid_slice, logp, legacy_z, paper,
                host_logp, host_legacy, host_paper, host_variance, host_valid,
            )

        del logits, targets, valid, attention
        for record, parts, valid_length in zip(batch, accum, valid_lengths):
            arrays = {
                key: np.concatenate(value).astype(np.float32, copy=False)
                for key, value in parts.items()
            }
            if any(len(value) != valid_length for value in arrays.values()):
                raise RuntimeError("token-statistic length mismatch")
            target_tokens += valid_length
            summary = summarize_window(
                arrays["logp"], arrays["legacy_z"], arrays["paper_z"]
            )
            metadata = {key: value for key, value in record.items() if key != "input_ids"}
            window_rows.append(metadata | summary)
            token_rows.append(
                metadata
                | {
                    "target_token_ids": np.asarray(record["input_ids"][1:], dtype=np.int32),
                    "target_positions_in_window": np.arange(
                        1, len(record["input_ids"]), dtype=np.int32
                    ),
                    "target_logp": arrays["logp"],
                    "legacy_uniform_vocab_z": arrays["legacy_z"],
                    "paper_prob_weighted_z": arrays["paper_z"],
                    "paper_probability_weighted_variance": arrays["paper_variance"],
                }
            )

        del input_ids, accum
        if (batch_index + 1) % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()

synchronize()
run_seconds = time.perf_counter() - run_started
peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
window_features = pd.DataFrame(window_rows)
token_statistics = pd.DataFrame(token_rows)
"""

aggregate_cell = r"""
score_rows = []
for sample_id in sample["sample_id"]:
    subset = token_statistics.loc[token_statistics.sample_id == sample_id]
    if subset.empty:
        raise RuntimeError(f"missing token statistics for {sample_id}")
    logp = np.concatenate(subset["target_logp"].tolist())
    legacy = np.concatenate(subset["legacy_uniform_vocab_z"].tolist())
    paper = np.concatenate(subset["paper_prob_weighted_z"].tolist())
    row = {
        "sample_id": sample_id,
        "score_mean_logp": float(logp.mean()),
        "score_local64": float(
            window_features.loc[
                window_features.sample_id == sample_id, "best_local_64"
            ].max()
        ),
        "scored_target_tokens": int(len(logp)),
        "window_count": int(len(subset)),
    }
    for fraction in MINK_FRACTIONS:
        suffix = f"{int(round(fraction * 100)):02d}"
        row[f"score_legacy_uniform_vocab_minkpp_{suffix}"] = bottom_fraction_mean(
            legacy, fraction
        )
        row[f"score_paper_prob_weighted_minkpp_{suffix}"] = bottom_fraction_mean(
            paper, fraction
        )
    score_rows.append(row)

sample_scores = sample[["sample_id", "language", "membership", "label"]].merge(
    pd.DataFrame(score_rows),
    on="sample_id",
    how="left",
    validate="one_to_one",
    sort=False,
)
if not np.array_equal(sample_scores["sample_id"].to_numpy(), sample["sample_id"].to_numpy()):
    raise RuntimeError("sample score order changed")
score_columns = [column for column in sample_scores if column.startswith("score_")]
if sample_scores[score_columns].isna().any().any():
    raise RuntimeError("NaN score found")
if not np.isfinite(sample_scores[score_columns].to_numpy(float)).all():
    raise RuntimeError("non-finite score found")

auc = {
    column: float(roc_auc_score(sample_scores["label"], sample_scores[column]))
    for column in score_columns
}
correlation = sample_scores[score_columns].corr(method="spearman")
metrics = {
    "warning": (
        "N=500 is a definition/runtime pilot. Do not use TPR@1% or small "
        "low-FPR count differences to select/reject a method."
    ),
    "rows": int(len(sample_scores)),
    "member_rows": int(sample_scores["label"].sum()),
    "auc_roc_exploratory": auc,
    "paper_legacy_spearman": {
        suffix: float(
            correlation.loc[
                f"score_legacy_uniform_vocab_minkpp_{suffix}",
                f"score_paper_prob_weighted_minkpp_{suffix}",
            ]
        )
        for suffix in ("01", "05", "10", "20")
    },
}

diagnostics = {
    "schema_version": MINKPP_PAPER_SCHEMA_VERSION,
    "cpu_oracle_abs_error": oracle_abs_error,
    "synthetic_dense_blocked_max_z_diff": synthetic_z_diff,
    "synthetic_dense_blocked_max_variance_diff": synthetic_var_diff,
    "real_dense_blocked_max_z_diff": real_dense_blocked_max_z_diff,
    "real_dense_blocked_max_variance_diff": real_dense_blocked_max_var_diff,
    "valid_target_tokens_from_reducer": int(diag_valid_tokens),
    "target_tokens_from_windows": int(target_tokens),
    "variance_clamp_count": int(diag_variance_clamps),
    "min_variance_before_clamp": (
        float(diag_min_variance) if math.isfinite(diag_min_variance) else None
    ),
    "max_abs_paper_z": float(diag_max_abs_z),
}
if diagnostics["valid_target_tokens_from_reducer"] != diagnostics["target_tokens_from_windows"]:
    raise RuntimeError("paper reducer valid-token accounting mismatch")

runtime = {
    "data_load_seconds": data_load_seconds,
    "tokenize_seconds": tokenize_seconds,
    "model_load_seconds": model_load_seconds,
    "inference_and_reduction_seconds": run_seconds,
    "windows": int(len(window_features)),
    "target_tokens": int(target_tokens),
    "peak_gpu_memory_bytes": peak_gpu_bytes,
    "gpu": torch.cuda.get_device_name(0),
    "torch_version": torch.__version__,
}
manifest = {
    "status": "complete",
    "config": asdict(CONFIG),
    "schema": {
        "legacy": "legacy_uniform_vocab_z (frozen historical definition)",
        "paper": MINKPP_PAPER_SCHEMA_VERSION,
    },
    "bounded_tokenization": {
        "strategy": "Unicode-safe bounded prefix/middle/suffix character regions",
        "max_source_chars_per_region": CONFIG.max_source_chars_per_region,
        "max_model_tokens_per_window": CONFIG.max_length,
        "positions": window_counts,
    },
    "diagnostics": diagnostics,
    "runtime": runtime,
    "metrics": metrics,
}

window_features.to_parquet(OUTPUT / "window_scores.parquet", index=False)
token_statistics.to_parquet(OUTPUT / "token_statistics.parquet", index=False)
sample_scores.to_parquet(OUTPUT / "sample_scores.parquet", index=False)
correlation.to_csv(OUTPUT / "score_spearman.csv")
(OUTPUT / "metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(OUTPUT / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
report = f'''# Stage 1 paper-faithful Min-K++ pilot — 500 rows

Status: complete.

This is a definition/runtime pilot, not a low-FPR model-selection experiment.
All compared scores use the same StarCoder2-3B forward passes and exactly the
same target tokens/windows.

- rows: {len(sample_scores)}
- windows: {len(window_features)}
- target tokens: {target_tokens}
- CPU-oracle absolute error: {oracle_abs_error:.3g}
- real dense/blocked max z difference: {real_dense_blocked_max_z_diff:.3g}
- variance clamps: {diag_variance_clamps}
- peak GPU memory: {peak_gpu_bytes / 2**30:.3f} GiB
- inference + reduction: {run_seconds:.1f} s

Exploratory ROC-AUC (not a promotion gate):

```json
{json.dumps(auc, indent=2, sort_keys=True)}
```
'''
(OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")

# Aggregate-only public log. No sample IDs or source text are printed.
print(
    json.dumps(
        {
            "status": manifest["status"],
            "rows": metrics["rows"],
            "windows": runtime["windows"],
            "target_tokens": runtime["target_tokens"],
            "diagnostics": diagnostics,
            "runtime": runtime,
            "auc_roc_exploratory": auc,
            "paper_legacy_spearman": metrics["paper_legacy_spearman"],
        },
        indent=2,
        sort_keys=True,
    )
)
"""

notebook = nbformat.v4.new_notebook()
notebook.metadata.kernelspec = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
notebook.metadata.language_info = {"name": "python", "version": "3.12"}
notebook.cells = [
    nbformat.v4.new_markdown_cell(
        "# Stage 1 — paper-faithful Min-K++ pilot (500 fixed rows)\n\n"
        "Definition/runtime validation. This notebook does not create or submit "
        "a competition prediction."
    ),
    nbformat.v4.new_code_cell(
        "%pip install -q datasets transformers==5.0.0 accelerate pyarrow scikit-learn"
    ),
    nbformat.v4.new_code_cell(MINKPP_SOURCE),
    nbformat.v4.new_code_cell(textwrap.dedent(config_cell)),
    nbformat.v4.new_code_cell(textwrap.dedent(data_cell)),
    nbformat.v4.new_code_cell(textwrap.dedent(model_cell)),
    nbformat.v4.new_code_cell(textwrap.dedent(run_cell)),
    nbformat.v4.new_code_cell(textwrap.dedent(aggregate_cell)),
]
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, OUT)

metadata = {
    "id": "renta0426/poisoned-chalice-minkpp-paper-pilot-500-v1",
    "title": "Poisoned Chalice MinKPP Paper Pilot 500 V1",
    "code_file": OUT.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": ["gpu", "membership-inference", "minkpp", "definition-pilot"],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4",
}
(OUT.parent / "kernel-metadata.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(OUT)
