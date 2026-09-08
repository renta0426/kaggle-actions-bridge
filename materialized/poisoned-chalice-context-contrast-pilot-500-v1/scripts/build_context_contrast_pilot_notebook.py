"""Build the fixed P2-01 matched-target context-contrast 500-row runtime/fidelity pilot."""

from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
PAPER_SOURCE = (ROOT / "src/poisoned_chalice/minkpp_paper.py").read_text(encoding="utf-8")
CONTEXT_SOURCE = (ROOT / "src/poisoned_chalice/context_contrast.py").read_text(encoding="utf-8")
OUT_DIR = ROOT / "notebooks/experiments/matched-target-context-contrast-v1-pilot-500"
OUT = OUT_DIR / "matched-target-context-contrast-v1-pilot-500.ipynb"

CONFIG_CELL = r'''
from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import gc, json, math, os, random, time
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")

@dataclass(frozen=True)
class PilotConfig:
    experiment_id: str = "matched-target-context-contrast-v1-pilot-500"
    dataset_id: str = "Poisoned-Chalice/ICSE-2027-public"
    dataset_revision: str = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
    model_id: str = "bigcode/starcoder2-3b"
    model_revision: str = "733247c55e3f73af49ce8e9c7949bf14af205928"
    seed: int = 2027
    samples_per_language: int = 100
    max_batch_tokens: int = 4096
    paper_vocab_block_size: int = 8192
    paper_variance_floor: float = 1e-12
    cpu_gpu_reduction_tolerance: float = 2e-5
    output_dir: str = "/kaggle/working/matched-target-context-contrast-v1-pilot-500"

CONFIG = PilotConfig()
CONTEXT_CONFIG = ContextContrastConfig(
    context_budgets=(32, 128, 512), target_tokens=256, min_target_tokens=64,
    spans_per_file=3, anchor_fractions=(0.25, 0.50, 0.75),
    local_width=64, paper_minkpp_fraction=0.10,
)
OUTPUT = Path(CONFIG.output_dir)
OUTPUT.mkdir(parents=True, exist_ok=True)
random.seed(CONFIG.seed)
np.random.seed(CONFIG.seed)
torch.manual_seed(CONFIG.seed)
'''

DATA_CELL = r'''
def _membership_label(series):
    return series.astype("string").str.lower().map(
        {"member": 1, "non-member": 0, "non_member": 0}
    )


def load_fixed_scoring_sample(config):
    selected = []
    for language in LANGUAGES:
        dataset = load_dataset(
            config.dataset_id, language, revision=config.dataset_revision, split="train"
        )
        frame = dataset.to_pandas().copy()
        frame["language"] = language
        labels = _membership_label(frame["membership"])
        if labels.isna().any():
            raise RuntimeError(f"unexpected membership label for {language}")
        frame["_selection_label"] = labels.astype(int)
        for label in (0, 1):
            pool = frame.loc[frame["_selection_label"] == label]
            chosen = pool.sample(
                n=config.samples_per_language // 2,
                random_state=config.seed + label,
            )
            selected.append(chosen[["sample_id", "language", "content"]].copy())
        del frame, dataset, labels
        gc.collect()
    scoring = pd.concat(selected, ignore_index=True)
    del selected
    scoring = scoring.sort_values("sample_id").reset_index(drop=True)
    if len(scoring) != 500 or scoring["sample_id"].duplicated().any():
        raise RuntimeError("fixed P2-01 pilot sample invariant failed")
    expected_hash = scoring["content"].map(
        lambda value: sha256(value.encode("utf-8")).hexdigest()
    )
    actual_hash = scoring["sample_id"].str.rsplit("-", n=1).str[-1]
    if not np.array_equal(expected_hash.to_numpy(), actual_hash.to_numpy()):
        raise RuntimeError("sample_id/content SHA-256 mismatch")
    if {"membership", "label", "_selection_label"}.intersection(scoring.columns):
        raise RuntimeError("selection labels leaked into scoring frame")
    return scoring

load_started = time.perf_counter()
scoring_sample = load_fixed_scoring_sample(CONFIG)
data_load_seconds = time.perf_counter() - load_started
sample_manifest = scoring_sample[["sample_id", "language"]].copy()
sample_manifest["content_sha256"] = scoring_sample["content"].map(
    lambda value: sha256(value.encode("utf-8")).hexdigest()
)
sample_manifest.to_parquet(OUTPUT / "sample_manifest.parquet", index=False)
if list(sample_manifest.columns) != ["sample_id", "language", "content_sha256"]:
    raise RuntimeError("sample manifest schema changed")
gc.collect()
'''

PLANNING_CELL = r'''
tokenizer = AutoTokenizer.from_pretrained(
    CONFIG.model_id, revision=CONFIG.model_revision, use_fast=True
)
if not getattr(tokenizer, "is_fast", False):
    raise RuntimeError("P2-01 requires a fast tokenizer with offset mapping")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

planning_started = time.perf_counter()
condition_records = []
span_manifest_rows = []
scoreable_samples = set()
for row in scoring_sample.itertuples(index=False):
    document = tokenize_with_byte_offsets(tokenizer, row.content)
    spans = select_target_spans(document, CONTEXT_CONFIG)
    windows = build_context_windows(document, spans)
    if len(spans) > 3 or len(windows) > 9:
        raise RuntimeError("bounded context planning invariant failed")
    if windows:
        scoreable_samples.add(row.sample_id)
    target_hash_by_span = {}
    for span in spans:
        target_ids = tuple(document.input_ids[span.target_start_token:span.target_end_token])
        target_sha = sha256(
            json.dumps(target_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        target_hash_by_span[span.span_index] = target_sha
        span_manifest_rows.append({
            "sample_id": row.sample_id,
            "language": row.language,
            "span_index": int(span.span_index),
            "anchor_fraction": float(span.anchor_fraction),
            "anchor_byte": int(span.anchor_byte),
            "target_start_token": int(span.target_start_token),
            "target_end_token": int(span.target_end_token),
            "target_start_byte": int(span.target_start_byte),
            "target_end_byte": int(span.target_end_byte),
            "target_tokens": int(span.target_token_count),
            "available_contexts": ",".join(map(str, span.available_context_budgets)),
            "target_token_sha256": target_sha,
        })
    for window in windows:
        if len(window.input_ids) > 768:
            raise RuntimeError("context window exceeds frozen 768-token maximum")
        target_sha = sha256(
            json.dumps(window.target_token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if target_sha != target_hash_by_span[window.span_index]:
            raise RuntimeError("shared-target identity changed during planning")
        condition_records.append({
            "sample_id": row.sample_id,
            "language": row.language,
            "span_index": int(window.span_index),
            "context_budget": int(window.context_budget),
            "target_start_in_window": int(window.target_start_in_window),
            "target_end_in_window": int(window.target_end_in_window),
            "window_token_count": int(len(window.input_ids)),
            "target_token_count": int(len(window.target_token_ids)),
            "target_token_sha256": target_sha,
            "input_ids": list(window.input_ids),
            "target_token_ids": list(window.target_token_ids),
        })
planning_seconds = time.perf_counter() - planning_started

span_manifest = pd.DataFrame(span_manifest_rows)
if span_manifest.empty:
    span_manifest = pd.DataFrame(columns=[
        "sample_id", "language", "span_index", "anchor_fraction", "anchor_byte",
        "target_start_token", "target_end_token", "target_start_byte",
        "target_end_byte", "target_tokens", "available_contexts", "target_token_sha256",
    ])
span_manifest.to_parquet(OUTPUT / "span_manifest.parquet", index=False)
del span_manifest_rows
gc.collect()

def dynamic_batches(records, max_batch_tokens):
    ordered = sorted(records, key=lambda item: (
        item["window_token_count"], item["sample_id"],
        item["span_index"], item["context_budget"],
    ))
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
'''

MODEL_CELL = r'''
# Label-bearing selection frames are gone before CUDA/model scoring begins.
if set(scoring_sample.columns) != {"sample_id", "language", "content"}:
    raise RuntimeError("scoring frame contains unexpected columns")
gc.collect()
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this separately approved pilot")
visible_gpu_count = torch.cuda.device_count()
if visible_gpu_count != 2:
    raise RuntimeError(f"expected exactly two visible T4 GPUs, got {visible_gpu_count}")
gpu_names = [torch.cuda.get_device_name(i) for i in range(visible_gpu_count)]
if not all("T4" in name for name in gpu_names):
    raise RuntimeError(f"unexpected GPU class: {gpu_names}")
torch.cuda.manual_seed_all(CONFIG.seed)
model_load_started = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    CONFIG.model_id, revision=CONFIG.model_revision, dtype=torch.float16,
    low_cpu_mem_usage=True, attn_implementation="sdpa",
).to("cuda").eval()
model_load_seconds = time.perf_counter() - model_load_started

oracle_logits = torch.tensor([[[3.0, 1.0, -2.0]]], device="cuda")
oracle_target = torch.tensor([[1]], device="cuda")
oracle = probability_weighted_token_statistics_blocked(
    oracle_logits, oracle_target, vocab_block_size=2
)
oracle_expected = -2.336452981844921
oracle_abs_error = abs(float(oracle.standardized_log_prob.item()) - oracle_expected)
if oracle_abs_error > 2e-5:
    raise RuntimeError(f"paper Min-K++ GPU oracle mismatch: {oracle_abs_error}")

generator = torch.Generator(device="cuda").manual_seed(CONFIG.seed)
synthetic_logits = torch.randn(2, 5, 97, generator=generator, device="cuda")
synthetic_targets = torch.randint(0, 97, (2, 5), generator=generator, device="cuda")
synthetic_dense = probability_weighted_token_statistics_dense(
    synthetic_logits, synthetic_targets
)
synthetic_blocked = probability_weighted_token_statistics_blocked(
    synthetic_logits, synthetic_targets, vocab_block_size=17
)
synthetic_z_diff = float((
    synthetic_dense.standardized_log_prob - synthetic_blocked.standardized_log_prob
).abs().max().item())
synthetic_var_diff = float((
    synthetic_dense.probability_weighted_variance - synthetic_blocked.probability_weighted_variance
).abs().max().item())
if synthetic_z_diff > 3e-5 or synthetic_var_diff > 3e-5:
    raise RuntimeError(
        f"synthetic dense/blocked parity failed: z={synthetic_z_diff} var={synthetic_var_diff}"
    )
del synthetic_logits, synthetic_targets, synthetic_dense, synthetic_blocked
gc.collect(); torch.cuda.empty_cache()
'''

RUN_CELL = r'''
def gpu_summary(logp_tensor, z_tensor, config):
    if logp_tensor.ndim != 1 or z_tensor.ndim != 1 or len(logp_tensor) != len(z_tensor):
        raise RuntimeError("GPU reduction received mismatched target vectors")
    count = int(logp_tensor.numel())
    if count < 1:
        raise RuntimeError("GPU reduction received empty target")
    k = max(1, int(math.ceil(count * config.paper_minkpp_fraction)))
    paper = torch.topk(z_tensor, k=k, largest=False, sorted=False).values.mean()
    width = min(config.local_width, count)
    local = logp_tensor.mean() if width == count else logp_tensor.unfold(0, width, 1).mean(-1).max()
    return {
        "mean_logp": float(logp_tensor.mean().item()),
        "paper_minkpp10": float(paper.item()),
        "local64": float(local.item()),
        "target_tokens": float(count),
    }

torch.cuda.reset_peak_memory_stats()
run_started = time.perf_counter()
condition_rows = []
shared_target_identity_failures = 0
valid_target_tokens = 0
paper_valid_target_tokens = 0
paper_variance_clamps = 0
paper_min_variance = math.inf
paper_max_abs_z = 0.0
cpu_gpu_reduction_max_abs_error = 0.0
real_dense_blocked_max_z_diff = 0.0
real_dense_blocked_max_var_diff = 0.0
real_parity_checked = False
pad_id = tokenizer.pad_token_id
device = model.get_input_embeddings().weight.device

with torch.inference_mode():
    for batch in dynamic_batches(condition_records, CONFIG.max_batch_tokens):
        width = max(record["window_token_count"] for record in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        for index, record in enumerate(batch):
            ids = torch.as_tensor(record["input_ids"], dtype=torch.long, device=device)
            input_ids[index, :len(ids)] = ids
            attention[index, :len(ids)] = 1
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
        for index, record in enumerate(batch):
            target_start = record["target_start_in_window"]
            target_end = record["target_end_in_window"]
            target_logits = logits[index, target_start - 1:target_end - 1]
            target_ids = input_ids[index, target_start:target_end]
            expected_ids = torch.as_tensor(record["target_token_ids"], dtype=torch.long, device=device)
            if target_ids.shape != expected_ids.shape or not torch.equal(target_ids, expected_ids):
                shared_target_identity_failures += 1
                raise RuntimeError("GPU target slice identity failure")
            if target_logits.shape[0] != target_ids.shape[0]:
                raise RuntimeError("causal target/logit alignment failure")
            if not real_parity_checked:
                parity_logits = target_logits[:min(4, target_logits.shape[0])]
                parity_targets = target_ids[:parity_logits.shape[0]]
                dense = probability_weighted_token_statistics_dense(parity_logits, parity_targets)
                blocked = probability_weighted_token_statistics_blocked(
                    parity_logits, parity_targets, vocab_block_size=CONFIG.paper_vocab_block_size
                )
                real_dense_blocked_max_z_diff = float((
                    dense.standardized_log_prob - blocked.standardized_log_prob
                ).abs().max().item())
                real_dense_blocked_max_var_diff = float((
                    dense.probability_weighted_variance - blocked.probability_weighted_variance
                ).abs().max().item())
                if real_dense_blocked_max_z_diff > 5e-5 or real_dense_blocked_max_var_diff > 5e-5:
                    raise RuntimeError(
                        "real dense/blocked parity failed: "
                        f"z={real_dense_blocked_max_z_diff} var={real_dense_blocked_max_var_diff}"
                    )
                real_parity_checked = True
                del dense, blocked, parity_logits, parity_targets
            values = target_logits.float()
            correct = values.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            logp = correct - torch.logsumexp(values, dim=-1)
            del values, correct
            paper = probability_weighted_token_statistics_blocked(
                target_logits, target_ids,
                variance_floor=CONFIG.paper_variance_floor,
                vocab_block_size=CONFIG.paper_vocab_block_size,
            )
            z = paper.standardized_log_prob
            diag = paper.diagnostics
            paper_valid_target_tokens += diag.valid_token_count
            paper_variance_clamps += diag.variance_clamp_count
            if math.isfinite(diag.min_variance_before_clamp):
                paper_min_variance = min(paper_min_variance, diag.min_variance_before_clamp)
            paper_max_abs_z = max(paper_max_abs_z, diag.max_abs_z)
            cpu = summarize_shared_target_scores(
                logp.detach().cpu().numpy(), z.detach().cpu().numpy(), CONTEXT_CONFIG
            )
            gpu = gpu_summary(logp, z, CONTEXT_CONFIG)
            for metric in ("mean_logp", "paper_minkpp10", "local64", "target_tokens"):
                error = abs(float(cpu[metric]) - float(gpu[metric]))
                cpu_gpu_reduction_max_abs_error = max(cpu_gpu_reduction_max_abs_error, error)
                if error > CONFIG.cpu_gpu_reduction_tolerance:
                    raise RuntimeError(f"CPU/GPU reduction parity failed for {metric}: {error}")
            valid_target_tokens += int(len(target_ids))
            condition_rows.append({
                "sample_id": record["sample_id"], "language": record["language"],
                "span_index": record["span_index"], "context_budget": record["context_budget"],
                "window_token_count": record["window_token_count"],
                "target_token_count": record["target_token_count"],
                "target_token_sha256": record["target_token_sha256"],
                "mean_logp": float(cpu["mean_logp"]),
                "paper_minkpp10": float(cpu["paper_minkpp10"]),
                "local64": float(cpu["local64"]),
            })
            del target_logits, target_ids, expected_ids, logp, paper, z
        del logits, input_ids, attention
        torch.cuda.empty_cache()
torch.cuda.synchronize()
inference_and_reduction_seconds = time.perf_counter() - run_started
peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
peak_gpu_reserved_bytes = int(torch.cuda.max_memory_reserved())
'''

OUTPUT_CELL = r'''
condition_scores = pd.DataFrame(condition_rows)
metric_columns = ["mean_logp", "paper_minkpp10", "local64"]
if condition_scores.empty:
    raise RuntimeError("runtime pilot produced no scoreable conditions")
if not np.isfinite(condition_scores[metric_columns].to_numpy(dtype=float)).all():
    raise RuntimeError("non-finite condition score")
condition_scores.to_parquet(OUTPUT / "condition_scores.parquet", index=False)

span_contrast_rows = []
for (sample_id, language, span_index), group in condition_scores.groupby(
    ["sample_id", "language", "span_index"], sort=True
):
    if len(set(group["target_token_sha256"])) != 1 or len(set(group["target_token_count"])) != 1:
        raise RuntimeError("shared target changed across context conditions")
    summaries = {
        int(row.context_budget): {
            "mean_logp": float(row.mean_logp),
            "paper_minkpp10": float(row.paper_minkpp10),
            "local64": float(row.local64),
            "target_tokens": float(row.target_token_count),
        }
        for row in group.itertuples(index=False)
    }
    span_contrast_rows.append({
        "sample_id": sample_id, "language": language, "span_index": int(span_index),
        **contrast_deltas(summaries),
    })
span_contrasts = pd.DataFrame(span_contrast_rows)
span_contrasts.to_parquet(OUTPUT / "span_contrasts.parquet", index=False)

file_feature_rows = []
for (sample_id, language), group in span_contrasts.groupby(["sample_id", "language"], sort=True):
    rows = []
    for row in group.to_dict(orient="records"):
        rows.append({
            key: float(value) for key, value in row.items()
            if key.startswith("delta_") and pd.notna(value)
        })
    file_feature_rows.append({
        "sample_id": sample_id, "language": language, **aggregate_span_contrasts(rows)
    })
file_features = pd.DataFrame(file_feature_rows)
file_features.to_parquet(OUTPUT / "file_features.parquet", index=False)

budget_counts = {
    str(int(key)): int(value)
    for key, value in condition_scores["context_budget"].value_counts().sort_index().items()
}
span_budget_sets = condition_scores.groupby(["sample_id", "span_index"])["context_budget"].agg(
    lambda values: frozenset(map(int, values))
)
rows_primary = {
    sample_id for (sample_id, _), budgets in span_budget_sets.items()
    if {32, 512}.issubset(budgets)
}
rows_fallback_only = {
    sample_id for (sample_id, _), budgets in span_budget_sets.items()
    if {32, 128}.issubset(budgets) and 512 not in budgets
} - rows_primary
expected_valid_target_tokens = int(condition_scores["target_token_count"].sum())

fidelity = {
    "experiment_id": CONFIG.experiment_id,
    "shared_target_identity_failures": int(shared_target_identity_failures),
    "cpu_gpu_reduction_max_abs_error": float(cpu_gpu_reduction_max_abs_error),
    "cpu_gpu_reduction_tolerance": float(CONFIG.cpu_gpu_reduction_tolerance),
    "paper_cpu_oracle_abs_error": float(oracle_abs_error),
    "synthetic_dense_blocked_max_z_diff": float(synthetic_z_diff),
    "synthetic_dense_blocked_max_variance_diff": float(synthetic_var_diff),
    "real_dense_blocked_max_z_diff": float(real_dense_blocked_max_z_diff),
    "real_dense_blocked_max_variance_diff": float(real_dense_blocked_max_var_diff),
    "paper_variance_clamp_count": int(paper_variance_clamps),
    "paper_min_variance_before_clamp": float(paper_min_variance) if math.isfinite(paper_min_variance) else None,
    "paper_max_abs_z": float(paper_max_abs_z),
    "valid_target_tokens_from_conditions": int(expected_valid_target_tokens),
    "valid_target_tokens_from_scorer": int(valid_target_tokens),
    "valid_target_tokens_from_paper_reducer": int(paper_valid_target_tokens),
    "all_condition_scores_finite": True,
    "max_window_tokens": int(max(record["window_token_count"] for record in condition_records)),
    "performance_metrics_computed": False,
}
if shared_target_identity_failures != 0:
    raise RuntimeError("shared-target identity gate failed")
if not (expected_valid_target_tokens == valid_target_tokens == paper_valid_target_tokens):
    raise RuntimeError("valid-target accounting gate failed")
if fidelity["max_window_tokens"] > 768:
    raise RuntimeError("window bound gate failed")
if cpu_gpu_reduction_max_abs_error > CONFIG.cpu_gpu_reduction_tolerance:
    raise RuntimeError("CPU/GPU reduction gate failed")

runtime = {
    "experiment_id": CONFIG.experiment_id,
    "gpu_names": gpu_names, "visible_gpu_count": int(visible_gpu_count),
    "data_load_seconds": float(data_load_seconds),
    "tokenize_and_plan_seconds": float(planning_seconds),
    "model_load_seconds": float(model_load_seconds),
    "inference_and_reduction_seconds": float(inference_and_reduction_seconds),
    "total_measured_seconds": float(data_load_seconds + planning_seconds + model_load_seconds + inference_and_reduction_seconds),
    "peak_gpu_memory_bytes": int(peak_gpu_memory_bytes),
    "peak_gpu_reserved_bytes": int(peak_gpu_reserved_bytes),
    "condition_windows": int(len(condition_scores)), "spans": int(len(span_contrasts)),
    "rows_total": 500, "rows_scoreable": int(len(scoreable_samples)),
    "rows_with_primary_512_minus_32": int(len(rows_primary)),
    "rows_with_fallback_128_minus_32_only": int(len(rows_fallback_only)),
    "condition_count_by_context_budget": budget_counts,
    "torch_version": torch.__version__,
}
run_manifest = {
    "experiment_id": CONFIG.experiment_id, "task_id": "P2-01",
    "status": "runtime_fidelity_pilot_complete",
    "model": {"id": CONFIG.model_id, "revision": CONFIG.model_revision},
    "dataset": {"id": CONFIG.dataset_id, "revision": CONFIG.dataset_revision, "split": "train", "rows": 500},
    "context_contract": asdict(CONTEXT_CONFIG), "selection_seed": CONFIG.seed,
    "selection": "50 rows per language x membership stratum; labels discarded before model/CUDA scoring",
    "outputs_contain_membership_labels": False, "performance_metrics_computed": False,
    "validation_rows_used": False, "codeparrot_rows_or_labels_used": False,
    "public_lb_feedback_used": False, "competition_submission": False,
    "automatic_compute_retries": 0, "same_target_identity_required": True,
    "fidelity_passed": True,
}
for name, payload in (("fidelity.json", fidelity), ("runtime.json", runtime), ("run_manifest.json", run_manifest)):
    (OUTPUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

report = f"""# P2-01 matched-target context contrast — 500-row runtime/fidelity pilot

Status: **runtime/fidelity pilot complete**.

This run intentionally computed no AUC, TPR, pAUC, threshold, or promotion decision.

## Contract
- Model: `{CONFIG.model_id}` @ `{CONFIG.model_revision}`
- Dataset: `{CONFIG.dataset_id}` train @ `{CONFIG.dataset_revision}`
- Rows: 500, selected as 50 per language × membership stratum with seed {CONFIG.seed}
- Membership labels were used only for deterministic CPU-side selection and were removed before model/CUDA scoring.
- Context budgets: 32 / 128 / 512
- Target: up to 256 exact shared tokens; minimum 64
- Byte anchors: 25% / 50% / 75%, maximum 3 spans per file
- No validation rows, CodeParrot rows/labels, Public-LB feedback, or competition submission.

## Fidelity
- Shared-target identity failures: {shared_target_identity_failures}
- CPU/GPU reduction max abs error: {cpu_gpu_reduction_max_abs_error:.9g}
- Paper CPU oracle abs error: {oracle_abs_error:.9g}
- Synthetic dense/blocked max z diff: {synthetic_z_diff:.9g}
- Real dense/blocked max z diff: {real_dense_blocked_max_z_diff:.9g}
- Valid target tokens: {valid_target_tokens}
- Probability-weighted variance clamps: {paper_variance_clamps}

## Coverage and runtime
- Scoreable rows: {len(scoreable_samples)} / 500
- Rows with at least one primary 512-vs-32 span: {len(rows_primary)}
- Rows with 128-vs-32 fallback but no primary span: {len(rows_fallback_only)}
- Spans: {len(span_contrasts)}
- Condition windows: {len(condition_scores)}
- Condition counts by budget: {budget_counts}
- Inference/reduction seconds: {inference_and_reduction_seconds:.3f}
- Peak allocated GPU bytes: {peak_gpu_memory_bytes}
- Peak reserved GPU bytes: {peak_gpu_reserved_bytes}

Performance evaluation remains blocked until this runtime/fidelity result is reviewed.
"""
(OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")
print(
    "P2_CONTEXT_RUNTIME_FIDELITY PASS "
    f"rows=500 scoreable={len(scoreable_samples)} spans={len(span_contrasts)} "
    f"windows={len(condition_scores)} valid_targets={valid_target_tokens} "
    f"cpu_gpu_max_abs={cpu_gpu_reduction_max_abs_error:.9g} "
    f"peak_gpu_bytes={peak_gpu_memory_bytes} performance_metrics=0 submissions=0"
)
'''

def _code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": cell_id,
        "metadata": {}, "outputs": [], "source": source,
    }

notebook = {
    "cells": [
        {"cell_type": "markdown", "id": "p2-context-intro", "metadata": {},
         "source": "# P2-01 matched-target context contrast — 500-row runtime/fidelity pilot\n\nNo performance metrics are computed in this notebook."},
        _code("exec(" + repr(PAPER_SOURCE) + ", globals())", "paper-source"),
        _code("exec(" + repr(CONTEXT_SOURCE) + ", globals())", "context-source"),
        _code(CONFIG_CELL, "config"), _code(DATA_CELL, "data-selection"),
        _code(PLANNING_CELL, "context-planning"), _code(MODEL_CELL, "model-fidelity"),
        _code(RUN_CELL, "gpu-scoring"), _code(OUTPUT_CELL, "outputs"),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
metadata = {
    "id": "renta0426/poisoned-chalice-context-contrast-pilot-500-v1",
    "title": "Poisoned Chalice Context Contrast Pilot 500 V1",
    "code_file": OUT.name, "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_tpu": False,
    "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
}
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
(OUT_DIR / "kernel-metadata.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(OUT)
