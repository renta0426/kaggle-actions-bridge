"""Build the frozen P2-01 fixed-10k context-contrast development evaluation notebook."""
from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
PAPER_SOURCE = (ROOT / "src/poisoned_chalice/minkpp_paper.py").read_text(encoding="utf-8")
CONTEXT_SOURCE = (ROOT / "src/poisoned_chalice/context_contrast.py").read_text(encoding="utf-8")
EVALUATION_SOURCE = (ROOT / "src/poisoned_chalice/evaluation.py").read_text(encoding="utf-8")
OUT_DIR = ROOT / "notebooks/experiments/matched-target-context-contrast-v1-10k"
OUT = OUT_DIR / "matched-target-context-contrast-v1-10k.ipynb"

CONFIG_CELL = r'''
from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import gc, json, math, os, random, shutil, time
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")

@dataclass(frozen=True)
class RunConfig:
    experiment_id: str = "matched-target-context-contrast-v1-10k"
    dataset_id: str = "Poisoned-Chalice/ICSE-2027-public"
    dataset_revision: str = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
    model_id: str = "bigcode/starcoder2-3b"
    model_revision: str = "733247c55e3f73af49ce8e9c7949bf14af205928"
    seed: int = 2027
    samples_per_language: int = 2000
    samples_per_shard: int = 250
    max_batch_tokens: int = 12288
    max_length: int = 768
    paper_vocab_block_size: int = 8192
    paper_variance_floor: float = 1e-12
    cpu_gpu_reduction_tolerance: float = 2e-5
    historical_first8_tolerance: float = 2e-5
    bootstrap_replicates: int = 1000
    output_dir: str = "/kaggle/working/matched-target-context-contrast-v1-10k"
    scratch_dir: str = "/tmp/matched-target-context-contrast-v1-10k-parts"

CONFIG = RunConfig()
CONTEXT_CONFIG = ContextContrastConfig(
    context_budgets=(32, 128, 512), target_tokens=256, min_target_tokens=64,
    spans_per_file=3, anchor_fractions=(0.25, 0.50, 0.75),
    local_width=64, paper_minkpp_fraction=0.10,
)
OUTPUT = Path(CONFIG.output_dir)
PARTS = Path(CONFIG.scratch_dir)
if PARTS.exists(): shutil.rmtree(PARTS)
PARTS.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
random.seed(CONFIG.seed); np.random.seed(CONFIG.seed); torch.manual_seed(CONFIG.seed)
'''

DATA_CELL = r'''
def _membership_label(series):
    return series.astype("string").str.lower().map(
        {"member": 1, "non-member": 0, "non_member": 0}
    )

selected_score = []
selected_eval = []
load_started = time.perf_counter()
for language in LANGUAGES:
    dataset = load_dataset(CONFIG.dataset_id, language, revision=CONFIG.dataset_revision, split="train")
    frame = dataset.to_pandas().copy()
    frame["language"] = language
    labels = _membership_label(frame["membership"])
    if labels.isna().any(): raise RuntimeError(f"unexpected membership label for {language}")
    frame["_selection_label"] = labels.astype(int)
    for label in (0, 1):
        pool = frame.loc[frame["_selection_label"] == label]
        chosen = pool.sample(n=CONFIG.samples_per_language // 2, random_state=CONFIG.seed + label)
        selected_score.append(chosen[["sample_id", "language", "content"]].copy())
        selected_eval.append(chosen[["sample_id", "language", "content", "_selection_label"]].rename(columns={"_selection_label":"label"}).copy())
    del frame, dataset, labels
    gc.collect()
scoring_sample = pd.concat(selected_score, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
evaluation_labels = pd.concat(selected_eval, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
del selected_score, selected_eval
if len(scoring_sample) != 10000 or scoring_sample.sample_id.duplicated().any(): raise RuntimeError("fixed 10k cohort invariant failed")
counts = evaluation_labels.groupby(["language", "label"]).size()
if len(counts) != 10 or not counts.eq(1000).all(): raise RuntimeError("fixed 10k balance invariant failed")
if not np.array_equal(scoring_sample.sample_id.to_numpy(), evaluation_labels.sample_id.to_numpy()): raise RuntimeError("score/evaluation sample identity mismatch")
expected_hash = scoring_sample.content.map(lambda x: sha256(x.encode("utf-8")).hexdigest())
actual_hash = scoring_sample.sample_id.str.rsplit("-", n=1).str[-1]
if not np.array_equal(expected_hash.to_numpy(), actual_hash.to_numpy()): raise RuntimeError("sample_id/content SHA-256 mismatch")
if {"membership", "label", "_selection_label"}.intersection(scoring_sample.columns): raise RuntimeError("selection labels leaked into scoring frame")
cohort_sha256 = sha256("\n".join(scoring_sample.sample_id).encode("utf-8")).hexdigest()
data_load_seconds = time.perf_counter() - load_started
'''

MODEL_CELL = r'''
tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_id, revision=CONFIG.model_revision, use_fast=True)
if not getattr(tokenizer, "is_fast", False): raise RuntimeError("fast tokenizer with offsets required")
if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
if set(scoring_sample.columns) != {"sample_id", "language", "content"}: raise RuntimeError("scoring frame schema changed")
if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
visible_gpu_count = torch.cuda.device_count()
if visible_gpu_count != 2: raise RuntimeError(f"expected two visible T4 GPUs, got {visible_gpu_count}")
gpu_names = [torch.cuda.get_device_name(i) for i in range(visible_gpu_count)]
if not all("T4" in x for x in gpu_names): raise RuntimeError(f"unexpected GPU class: {gpu_names}")
torch.cuda.manual_seed_all(CONFIG.seed)
model_load_started = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    CONFIG.model_id, revision=CONFIG.model_revision, dtype=torch.float16,
    low_cpu_mem_usage=True, attn_implementation="sdpa",
).to("cuda").eval()
model_load_seconds = time.perf_counter() - model_load_started

oracle_logits = torch.tensor([[[3.0, 1.0, -2.0]]], device="cuda")
oracle_target = torch.tensor([[1]], device="cuda")
oracle = probability_weighted_token_statistics_blocked(oracle_logits, oracle_target, vocab_block_size=2)
oracle_expected = -2.336452981844921
oracle_abs_error = abs(float(oracle.standardized_log_prob.item()) - oracle_expected)
if oracle_abs_error > 2e-5: raise RuntimeError(f"paper oracle mismatch: {oracle_abs_error}")

generator = torch.Generator(device="cuda").manual_seed(CONFIG.seed)
synthetic_logits = torch.randn(2, 5, 97, generator=generator, device="cuda")
synthetic_targets = torch.randint(0, 97, (2, 5), generator=generator, device="cuda")
dense = probability_weighted_token_statistics_dense(synthetic_logits, synthetic_targets)
blocked = probability_weighted_token_statistics_blocked(synthetic_logits, synthetic_targets, vocab_block_size=17)
synthetic_z_diff = float((dense.standardized_log_prob-blocked.standardized_log_prob).abs().max().item())
synthetic_var_diff = float((dense.probability_weighted_variance-blocked.probability_weighted_variance).abs().max().item())
if synthetic_z_diff > 3e-5 or synthetic_var_diff > 3e-5: raise RuntimeError("synthetic dense/blocked parity failed")
del synthetic_logits, synthetic_targets, dense, blocked

def historical_starts(token_count):
    if token_count <= CONFIG.max_length: return [("whole", 0)]
    candidates = [("prefix",0),("middle",(token_count-CONFIG.max_length)//2),("suffix",token_count-CONFIG.max_length)]
    out=[]; seen=set()
    for name,start in candidates:
        if start not in seen: out.append((name,start)); seen.add(start)
    return out

def dynamic_batches(records):
    ordered = sorted(records, key=lambda r: (r["window_token_count"], r["sample_id"], r["kind"], r.get("span_index",-1), r.get("context_budget",-1), r.get("position","")))
    batch=[]; max_len=0
    for record in ordered:
        candidate=max(max_len, record["window_token_count"])
        if batch and candidate*(len(batch)+1) > CONFIG.max_batch_tokens:
            yield batch; batch=[]; max_len=0
        batch.append(record); max_len=max(max_len, record["window_token_count"])
    if batch: yield batch

def gpu_summary(logp, z):
    count=int(logp.numel()); k=max(1,int(math.ceil(count*CONTEXT_CONFIG.paper_minkpp_fraction)))
    paper=torch.topk(z,k=k,largest=False,sorted=False).values.mean()
    width=min(CONTEXT_CONFIG.local_width,count)
    local=logp.mean() if width==count else logp.unfold(0,width,1).mean(-1).max()
    return {"mean_logp":float(logp.mean().item()),"paper_minkpp10":float(paper.item()),"local64":float(local.item()),"target_tokens":float(count)}
'''

RUN_CELL = r'''
EXPECTED_FIRST8 = {
"go-00098042c091d0cb85839c684ea25f02a88d3e1314773472e7999b483eff4ff5":(-1.3622978925704956,-0.5027317106723785),
"go-006e4bdbc50c3a49256ab44903bf20c0eb70c827951562fc5b4a85b518b83caa":(-0.4617495834827423,-0.0016315579414367676),
"go-006ff7bd5b5fd0b26fd009b24ec505e016d47b2f89f24e662229f43b466b66dc":(-1.6286673545837402,-0.8128157332539558),
"go-0070b3cba9df39d083bab543cb724cf1a0f2e9ea257650474b6f57f9ae7891f5":(-3.1724891662597656,-2.4475733041763306),
"go-0073f8e1f65319af2872013ce2b4e536125798e09395f398c4fb43d40f9c37c8":(-0.26913198828697205,-0.020467832684516907),
"go-0079f01a89369077b865146953771822f91e3ef12be1f88fe4b0cf96447b1768":(-0.7122843861579895,-0.23582403361797333),
"go-008ddb117bab91bf3912ecc76f86c92ac6aea219f60d4488e17b77492f7e2f6c":(-0.5732815265655518,-0.00491669774055481),
"go-00c0901c0f19f8a2de19f1aa6455b684001be74fcbe3879e1a4cdcca7c15b3af":(-0.8655945658683777,-0.36794745922088623),
}

context_valid_targets=0; baseline_valid_targets=0; paper_valid_targets=0
shared_target_identity_failures=0; cpu_gpu_reduction_max_abs_error=0.0
paper_variance_clamps=0; paper_min_variance=math.inf; paper_max_abs_z=0.0
real_dense_blocked_max_z_diff=0.0; real_dense_blocked_max_var_diff=0.0; real_parity_checked=False
condition_budget_counts={32:0,128:0,512:0}; total_spans=0; total_context_windows=0; total_baseline_windows=0
planning_seconds=0.0; scoring_started=time.perf_counter()
torch.cuda.reset_peak_memory_stats()
pad_id=tokenizer.pad_token_id; device=model.get_input_embeddings().weight.device

for shard_index, start in enumerate(range(0, len(scoring_sample), CONFIG.samples_per_shard)):
    shard=scoring_sample.iloc[start:start+CONFIG.samples_per_shard]
    plan_start=time.perf_counter(); records=[]; span_meta=[]; token_rows=[]
    for row in shard.itertuples(index=False):
        document=tokenize_with_byte_offsets(tokenizer,row.content)
        token_rows.append({"sample_id":row.sample_id,"language":row.language,"content_sha256":sha256(row.content.encode("utf-8")).hexdigest(),"token_count":len(document.input_ids)})
        spans=select_target_spans(document,CONTEXT_CONFIG); windows=build_context_windows(document,spans)
        total_spans += len(spans); total_context_windows += len(windows)
        target_hash_by_span={}
        for span in spans:
            target_ids=tuple(document.input_ids[span.target_start_token:span.target_end_token])
            target_sha=sha256(json.dumps(target_ids,separators=(",",":")).encode("utf-8")).hexdigest(); target_hash_by_span[span.span_index]=target_sha
            span_meta.append({"sample_id":row.sample_id,"language":row.language,"span_index":int(span.span_index),"target_token_sha256":target_sha})
        for window in windows:
            target_sha=sha256(json.dumps(window.target_token_ids,separators=(",",":")).encode("utf-8")).hexdigest()
            if target_sha != target_hash_by_span[window.span_index]: shared_target_identity_failures += 1; raise RuntimeError("context target identity failure")
            records.append({"kind":"context","sample_id":row.sample_id,"language":row.language,"span_index":int(window.span_index),"context_budget":int(window.context_budget),"window_token_count":len(window.input_ids),"input_ids":list(window.input_ids),"target_ids":list(window.target_token_ids),"target_start":int(window.target_start_in_window),"target_end":int(window.target_end_in_window),"target_token_sha256":target_sha})
            condition_budget_counts[int(window.context_budget)] += 1
        ids=document.input_ids
        if len(ids) < 2: raise RuntimeError(f"baseline sample shorter than 2 tokens: {row.sample_id}")
        for position,window_start in historical_starts(len(ids)):
            window_ids=ids[window_start:window_start+CONFIG.max_length]
            records.append({"kind":"baseline","sample_id":row.sample_id,"language":row.language,"position":position,"window_token_count":len(window_ids),"input_ids":list(window_ids),"target_ids":list(window_ids[1:]),"target_start":1,"target_end":len(window_ids)})
            total_baseline_windows += 1
    planning_seconds += time.perf_counter()-plan_start

    context_rows=[]; baseline_rows=[]
    with torch.inference_mode():
        for batch in dynamic_batches(records):
            width=max(r["window_token_count"] for r in batch)
            input_ids=torch.full((len(batch),width),pad_id,dtype=torch.long,device=device); attention=torch.zeros_like(input_ids)
            for i,r in enumerate(batch):
                ids=torch.as_tensor(r["input_ids"],dtype=torch.long,device=device); input_ids[i,:len(ids)]=ids; attention[i,:len(ids)]=1
            logits=model(input_ids=input_ids,attention_mask=attention,use_cache=False).logits
            for i,r in enumerate(batch):
                ts,te=r["target_start"],r["target_end"]
                target_logits=logits[i,ts-1:te-1]; target_ids=input_ids[i,ts:te]; expected=torch.as_tensor(r["target_ids"],dtype=torch.long,device=device)
                if target_ids.shape != expected.shape or not torch.equal(target_ids,expected): raise RuntimeError("GPU target identity failure")
                if not real_parity_checked:
                    p_logits=target_logits[:min(4,target_logits.shape[0])]; p_targets=target_ids[:p_logits.shape[0]]
                    dense=probability_weighted_token_statistics_dense(p_logits,p_targets); blocked=probability_weighted_token_statistics_blocked(p_logits,p_targets,vocab_block_size=CONFIG.paper_vocab_block_size)
                    real_dense_blocked_max_z_diff=float((dense.standardized_log_prob-blocked.standardized_log_prob).abs().max().item())
                    real_dense_blocked_max_var_diff=float((dense.probability_weighted_variance-blocked.probability_weighted_variance).abs().max().item())
                    if real_dense_blocked_max_z_diff>5e-5 or real_dense_blocked_max_var_diff>5e-5: raise RuntimeError("real dense/blocked parity failed")
                    real_parity_checked=True; del dense,blocked,p_logits,p_targets
                values=target_logits.float(); correct=values.gather(-1,target_ids.unsqueeze(-1)).squeeze(-1); logp=correct-torch.logsumexp(values,dim=-1); del values,correct
                paper=probability_weighted_token_statistics_blocked(target_logits,target_ids,variance_floor=CONFIG.paper_variance_floor,vocab_block_size=CONFIG.paper_vocab_block_size); z=paper.standardized_log_prob; diag=paper.diagnostics
                paper_valid_targets += int(diag.valid_token_count); paper_variance_clamps += int(diag.variance_clamp_count)
                if math.isfinite(diag.min_variance_before_clamp): paper_min_variance=min(paper_min_variance,diag.min_variance_before_clamp)
                paper_max_abs_z=max(paper_max_abs_z,diag.max_abs_z)
                cpu=summarize_shared_target_scores(logp.detach().cpu().numpy(),z.detach().cpu().numpy(),CONTEXT_CONFIG); gpu=gpu_summary(logp,z)
                for metric in ("mean_logp","paper_minkpp10","local64","target_tokens"):
                    err=abs(float(cpu[metric])-float(gpu[metric])); cpu_gpu_reduction_max_abs_error=max(cpu_gpu_reduction_max_abs_error,err)
                    if err>CONFIG.cpu_gpu_reduction_tolerance: raise RuntimeError(f"CPU/GPU reduction mismatch {metric}: {err}")
                base={"sample_id":r["sample_id"],"language":r["language"],"mean_logp":float(cpu["mean_logp"]),"paper_minkpp10":float(cpu["paper_minkpp10"]),"local64":float(cpu["local64"]),"target_token_count":int(cpu["target_tokens"])}
                if r["kind"]=="context":
                    context_valid_targets += len(target_ids); base.update({"span_index":r["span_index"],"context_budget":r["context_budget"],"target_token_sha256":r["target_token_sha256"]}); context_rows.append(base)
                else:
                    baseline_valid_targets += len(target_ids); base.update({"position":r["position"]}); baseline_rows.append(base)
                del target_logits,target_ids,expected,logp,paper,z
            del logits,input_ids,attention
            torch.cuda.empty_cache()

    context_df=pd.DataFrame(context_rows)
    span_rows=[]
    if not context_df.empty:
        for (sample_id,language,span_index),group in context_df.groupby(["sample_id","language","span_index"],sort=True):
            if len(set(group.target_token_sha256))!=1 or len(set(group.target_token_count))!=1: raise RuntimeError("shared target changed across conditions")
            summaries={int(r.context_budget):{"mean_logp":float(r.mean_logp),"paper_minkpp10":float(r.paper_minkpp10),"local64":float(r.local64),"target_tokens":float(r.target_token_count)} for r in group.itertuples(index=False)}
            span_rows.append({"sample_id":sample_id,"language":language,"span_index":int(span_index),**contrast_deltas(summaries)})
    span_df=pd.DataFrame(span_rows)
    file_rows=[]
    if not span_df.empty:
        for (sample_id,language),group in span_df.groupby(["sample_id","language"],sort=True):
            rows=[{k:float(v) for k,v in row.items() if k.startswith("delta_") and pd.notna(v)} for row in group.to_dict(orient="records")]
            file_rows.append({"sample_id":sample_id,"language":language,**aggregate_span_contrasts(rows)})
    file_df=pd.DataFrame(file_rows)
    baseline_df=pd.DataFrame(baseline_rows)
    baseline_agg=baseline_df.groupby(["sample_id","language"],sort=True).agg(
        baseline_loss=("mean_logp","max"), baseline_paper_minkpp10=("paper_minkpp10","max"), baseline_local64=("local64","max"), baseline_window_count=("position","count")
    ).reset_index()
    token_df=pd.DataFrame(token_rows)
    file_df.to_parquet(PARTS/f"file_features.part{shard_index:03d}.parquet",index=False)
    span_df.to_parquet(PARTS/f"span_contrasts.part{shard_index:03d}.parquet",index=False)
    baseline_agg.to_parquet(PARTS/f"baseline_features.part{shard_index:03d}.parquet",index=False)
    token_df.to_parquet(PARTS/f"sample_manifest.part{shard_index:03d}.parquet",index=False)
    print({"shard":shard_index,"samples":len(shard),"records":len(records),"context":len(context_df),"baseline":len(baseline_df),"scoreable":len(file_df)})
    del records,span_meta,token_rows,context_rows,baseline_rows,context_df,span_rows,span_df,file_rows,file_df,baseline_df,baseline_agg,token_df
    gc.collect(); torch.cuda.empty_cache()

torch.cuda.synchronize(); scoring_seconds=time.perf_counter()-scoring_started
peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated()); peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved())
file_features=pd.concat([pd.read_parquet(p) for p in sorted(PARTS.glob("file_features.part*.parquet"))],ignore_index=True).sort_values("sample_id").reset_index(drop=True)
span_contrasts=pd.concat([pd.read_parquet(p) for p in sorted(PARTS.glob("span_contrasts.part*.parquet")) if p.stat().st_size>0],ignore_index=True).sort_values(["sample_id","span_index"]).reset_index(drop=True)
baseline_features=pd.concat([pd.read_parquet(p) for p in sorted(PARTS.glob("baseline_features.part*.parquet"))],ignore_index=True).sort_values("sample_id").reset_index(drop=True)
sample_manifest=pd.concat([pd.read_parquet(p) for p in sorted(PARTS.glob("sample_manifest.part*.parquet"))],ignore_index=True).sort_values("sample_id").reset_index(drop=True)
if len(baseline_features)!=10000 or baseline_features.sample_id.duplicated().any(): raise RuntimeError("baseline final coverage failure")
if len(sample_manifest)!=10000 or sample_manifest.sample_id.duplicated().any(): raise RuntimeError("sample manifest final coverage failure")

historical_first8_max_abs_diff=0.0
for sample_id,(expected_loss,expected_local) in EXPECTED_FIRST8.items():
    row=baseline_features.loc[baseline_features.sample_id==sample_id]
    if len(row)!=1: raise RuntimeError(f"historical first8 sample missing: {sample_id}")
    historical_first8_max_abs_diff=max(historical_first8_max_abs_diff,abs(float(row.iloc[0].baseline_loss)-expected_loss),abs(float(row.iloc[0].baseline_local64)-expected_local))
if historical_first8_max_abs_diff > CONFIG.historical_first8_tolerance: raise RuntimeError(f"historical baseline fidelity failed: {historical_first8_max_abs_diff}")

primary_cols=["delta_512_32_mean_logp__max_span","delta_512_32_paper_minkpp10__max_span","delta_512_32_local64__max_span"]
fallback_cols=["delta_128_32_mean_logp__max_span","delta_128_32_paper_minkpp10__max_span","delta_128_32_local64__max_span"]
manifest=sample_manifest.merge(file_features[["sample_id"]+primary_cols+fallback_cols],on="sample_id",how="left",validate="one_to_one")
manifest["primary_eligible"]=manifest[primary_cols].notna().all(axis=1)
manifest["fallback_eligible"]=manifest[fallback_cols].notna().all(axis=1)
manifest=manifest.drop(columns=primary_cols+fallback_cols)
manifest.to_parquet(OUTPUT/"sample_manifest.parquet",index=False)
file_features.to_parquet(OUTPUT/"file_features.parquet",index=False)
span_contrasts.to_parquet(OUTPUT/"span_contrasts.parquet",index=False)
baseline_features.to_parquet(OUTPUT/"baseline_features.parquet",index=False)
shutil.rmtree(PARTS)
'''

EVAL_CELL = r'''
eval_frame=evaluation_labels[["sample_id","language","label","content"]].merge(sample_manifest[["sample_id","token_count"]],on="sample_id",validate="one_to_one").merge(baseline_features,on=["sample_id","language"],validate="one_to_one").merge(file_features,on=["sample_id","language"],how="left",validate="one_to_one")
primary=eval_frame.loc[eval_frame[primary_cols].notna().all(axis=1)].copy().reset_index(drop=True)
fallback=eval_frame.loc[~eval_frame[primary_cols].notna().all(axis=1) & eval_frame[fallback_cols].notna().all(axis=1)].copy().reset_index(drop=True)
if len(primary)<500 or primary.label.nunique()!=2: raise RuntimeError("primary eligible population unexpectedly small/degenerate")
SCORES={
 "context_mean_delta":primary[primary_cols[0]].to_numpy(float),
 "context_paper_delta":primary[primary_cols[1]].to_numpy(float),
 "context_local_delta":primary[primary_cols[2]].to_numpy(float),
 "baseline_loss":primary.baseline_loss.to_numpy(float),
 "baseline_paper":primary.baseline_paper_minkpp10.to_numpy(float),
 "baseline_local":primary.baseline_local64.to_numpy(float),
}
predictor_rows=[]
for name,score in SCORES.items(): predictor_rows.append({"predictor":name,"population":"primary",**low_fpr_metrics(primary.label,score)})
predictor_metrics=pd.DataFrame(predictor_rows)

fallback_rows=[]
if len(fallback)>=100 and fallback.label.nunique()==2:
    fallback_scores={"context_mean_delta":fallback[fallback_cols[0]],"context_paper_delta":fallback[fallback_cols[1]],"context_local_delta":fallback[fallback_cols[2]],"baseline_loss":fallback.baseline_loss,"baseline_paper":fallback.baseline_paper_minkpp10,"baseline_local":fallback.baseline_local64}
    for name,score in fallback_scores.items(): fallback_rows.append({"predictor":name,"population":"fallback_only",**low_fpr_metrics(fallback.label,score)})
    predictor_metrics=pd.concat([predictor_metrics,pd.DataFrame(fallback_rows)],ignore_index=True)

language_rows=[]
for language,group in primary.groupby("language",sort=True):
    for name in SCORES:
        score = group[{"context_mean_delta":primary_cols[0],"context_paper_delta":primary_cols[1],"context_local_delta":primary_cols[2],"baseline_loss":"baseline_loss","baseline_paper":"baseline_paper_minkpp10","baseline_local":"baseline_local64"}[name]]
        language_rows.append({"language":language,"predictor":name,"rows":len(group),**low_fpr_metrics(group.label,score)})
language_metrics=pd.DataFrame(language_rows)

primary["length_quartile"]=pd.qcut(primary.token_count,4,labels=False,duplicates="drop")
length_rows=[]
for quartile,group in primary.groupby("length_quartile",sort=True):
    if group.label.nunique()<2: continue
    for name in SCORES:
        col={"context_mean_delta":primary_cols[0],"context_paper_delta":primary_cols[1],"context_local_delta":primary_cols[2],"baseline_loss":"baseline_loss","baseline_paper":"baseline_paper_minkpp10","baseline_local":"baseline_local64"}[name]
        length_rows.append({"length_quartile":int(quartile),"predictor":name,"rows":len(group),"min_tokens":int(group.token_count.min()),"max_tokens":int(group.token_count.max()),**low_fpr_metrics(group.label,group[col])})
length_metrics=pd.DataFrame(length_rows)

strict_rows=[]
def append_split_metrics(view,splits):
    for fold,(_,hold) in enumerate(splits):
        group=primary.iloc[hold]
        if group.label.nunique()<2: continue
        for name,score_all in SCORES.items(): strict_rows.append({"view":view,"fold":fold,"predictor":name,"rows":len(group),**low_fpr_metrics(group.label,np.asarray(score_all)[hold])})
append_split_metrics("stratified_5fold",stratified_splits(primary,5,CONFIG.seed))
groups,minhash_diag=minhash_groups(primary.content.tolist())
append_split_metrics("minhash_group_5fold",grouped_splits(primary,groups,5,CONFIG.seed))
append_split_metrics("leave_one_language_out",leave_one_language_out_splits(primary))
length_splits,_=length_holdout_splits(primary,5); append_split_metrics("length_holdout",length_splits)
strict_metrics=pd.DataFrame(strict_rows)

jaccard,unique,_=overlap_tables(primary.sample_id,primary.label,SCORES,target_fpr=0.01)
bootstrap={name:stratified_bootstrap_metrics(primary,score,CONFIG.bootstrap_replicates,CONFIG.seed) for name,score in SCORES.items()}

coverage=evaluation_labels[["sample_id","language","label"]].merge(manifest[["sample_id","primary_eligible","fallback_eligible"]],on="sample_id",validate="one_to_one")
coverage_rows=[]
for (language,label),group in coverage.groupby(["language","label"],sort=True):
    coverage_rows.append({"language":language,"label":int(label),"rows":len(group),"primary_eligible":int(group.primary_eligible.sum()),"primary_rate":float(group.primary_eligible.mean()),"fallback_eligible":int(group.fallback_eligible.sum()),"fallback_rate":float(group.fallback_eligible.mean())})
coverage_by_language_label=pd.DataFrame(coverage_rows)

if context_valid_targets + baseline_valid_targets != paper_valid_targets: raise RuntimeError("paper valid-token accounting mismatch")
fidelity={"experiment_id":CONFIG.experiment_id,"shared_target_identity_failures":shared_target_identity_failures,"cpu_gpu_reduction_max_abs_error":cpu_gpu_reduction_max_abs_error,"cpu_gpu_reduction_tolerance":CONFIG.cpu_gpu_reduction_tolerance,"paper_cpu_oracle_abs_error":oracle_abs_error,"synthetic_dense_blocked_max_z_diff":synthetic_z_diff,"synthetic_dense_blocked_max_variance_diff":synthetic_var_diff,"real_dense_blocked_max_z_diff":real_dense_blocked_max_z_diff,"real_dense_blocked_max_variance_diff":real_dense_blocked_max_var_diff,"historical_loss_local_first8_max_abs_diff":historical_first8_max_abs_diff,"historical_first8_tolerance":CONFIG.historical_first8_tolerance,"context_valid_target_tokens":context_valid_targets,"baseline_valid_target_tokens":baseline_valid_targets,"paper_valid_target_tokens":paper_valid_targets,"paper_variance_clamp_count":paper_variance_clamps,"paper_min_variance_before_clamp":paper_min_variance,"paper_max_abs_z":paper_max_abs_z,"all_primary_scores_finite":bool(np.isfinite(np.column_stack(list(SCORES.values()))).all()),"performance_metrics_computed":True}
if shared_target_identity_failures!=0 or cpu_gpu_reduction_max_abs_error>CONFIG.cpu_gpu_reduction_tolerance or historical_first8_max_abs_diff>CONFIG.historical_first8_tolerance: raise RuntimeError("fidelity gate failed")
runtime={"experiment_id":CONFIG.experiment_id,"gpu_names":gpu_names,"visible_gpu_count":visible_gpu_count,"model_devices_used":1,"max_batch_tokens":CONFIG.max_batch_tokens,"data_load_seconds":data_load_seconds,"model_load_seconds":model_load_seconds,"planning_seconds":planning_seconds,"scoring_seconds":scoring_seconds,"peak_gpu_memory_bytes":peak_gpu_memory_bytes,"peak_gpu_reserved_bytes":peak_gpu_reserved_bytes,"rows_total":10000,"rows_scoreable":len(file_features),"rows_primary":len(primary),"rows_fallback_only":len(fallback),"spans":total_spans,"context_windows":total_context_windows,"baseline_windows":total_baseline_windows,"context_budget_counts":{str(k):int(v) for k,v in condition_budget_counts.items()},"torch_version":torch.__version__}
evaluation_summary={"experiment_id":CONFIG.experiment_id,"status":"development_metrics_complete_review_required","cohort_sha256":cohort_sha256,"rows_total":10000,"rows_primary":len(primary),"rows_fallback_only":len(fallback),"predictor_metrics":predictor_metrics.to_dict(orient="records"),"minhash_diagnostics":minhash_diag,"automatic_promotion":False,"independent_holdout_opened":False,"competition_submission":False,"public_lb_feedback_used":False}
run_manifest={"experiment_id":CONFIG.experiment_id,"task_id":"P2-01","status":"fixed_10k_development_complete_review_required","model":{"id":CONFIG.model_id,"revision":CONFIG.model_revision},"dataset":{"id":CONFIG.dataset_id,"revision":CONFIG.dataset_revision,"split":"train","rows":10000},"selection_seed":CONFIG.seed,"cohort_sha256":cohort_sha256,"row_level_outputs_contain_labels":False,"validation_rows_used":False,"codeparrot_rows_or_labels_used":False,"public_lb_feedback_used":False,"competition_submission":False,"automatic_compute_retries":0,"automatic_promotion":False}

predictor_metrics.to_csv(OUTPUT/"predictor_metrics.csv",index=False); language_metrics.to_csv(OUTPUT/"language_metrics.csv",index=False); length_metrics.to_csv(OUTPUT/"length_metrics.csv",index=False); strict_metrics.to_csv(OUTPUT/"strict_metrics.csv",index=False); jaccard.to_csv(OUTPUT/"detection_jaccard.csv",index=False); unique.to_csv(OUTPUT/"unique_true_positives.csv",index=False); coverage_by_language_label.to_csv(OUTPUT/"coverage_by_language_label.csv",index=False)
for name,payload in (("bootstrap.json",bootstrap),("fidelity.json",fidelity),("runtime.json",runtime),("evaluation_summary.json",evaluation_summary),("run_manifest.json",run_manifest)):
    (OUTPUT/name).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")

report_lines=["# P2-01 matched-target context contrast — fixed 10k development evaluation","","Status: **development metrics complete; review required before any candidate freeze or independent holdout.**","",f"- rows total: 10000",f"- primary 512-vs-32 eligible: {len(primary)}",f"- fallback-only 128-vs-32: {len(fallback)}",f"- scoreable context rows: {len(file_features)}",f"- context windows: {total_context_windows}",f"- baseline windows: {total_baseline_windows}",f"- shared-target identity failures: {shared_target_identity_failures}",f"- CPU/GPU reduction max abs error: {cpu_gpu_reduction_max_abs_error:.9g}",f"- historical first8 loss/local max abs diff: {historical_first8_max_abs_diff:.9g}",f"- scoring seconds: {scoring_seconds:.3f}",f"- peak allocated GPU bytes: {peak_gpu_memory_bytes}","","## Primary predictor metrics",""]
for row in predictor_metrics.loc[predictor_metrics.population=="primary"].to_dict(orient="records"):
    report_lines.append(f"- {row['predictor']}: AUC {row['auc']:.6f}, pAUC@1% {row['pauc_01']:.6f}, TPR@1% {row['tpr_at_0.01_fpr']:.6f}")
report_lines += ["","No variant is automatically promoted by this notebook. Review strict views, language/length stability, overlap, bootstrap and coverage before freezing a candidate."]
(OUTPUT/"REPORT.md").write_text("\n".join(report_lines)+"\n",encoding="utf-8")
print("P2_CONTEXT_10K DEVELOPMENT_COMPLETE",json.dumps({"rows_primary":len(primary),"rows_fallback_only":len(fallback),"context_windows":total_context_windows,"baseline_windows":total_baseline_windows,"cpu_gpu_max_abs":cpu_gpu_reduction_max_abs_error,"historical_first8_max_abs":historical_first8_max_abs_diff,"submissions":0},sort_keys=True))
'''

def _code(source: str, cell_id: str) -> dict:
    return {"cell_type":"code","execution_count":None,"id":cell_id,"metadata":{},"outputs":[],"source":source}

notebook={
 "cells":[
  {"cell_type":"markdown","id":"intro","metadata":{},"source":"# P2-01 matched-target context contrast — fixed 10k development evaluation\n\nFrozen source-development performance evaluation; no competition submission."},
  _code("%pip install -q datasets transformers==5.0.0 accelerate pyarrow scikit-learn","deps"),
  _code("exec("+repr(PAPER_SOURCE)+", globals())","paper-source"),
  _code("exec("+repr(CONTEXT_SOURCE)+", globals())","context-source"),
  _code("exec("+repr(EVALUATION_SOURCE)+", globals())","evaluation-source"),
  _code(CONFIG_CELL,"config"),_code(DATA_CELL,"data"),_code(MODEL_CELL,"model"),_code(RUN_CELL,"extract"),_code(EVAL_CELL,"evaluate")],
 "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
metadata={"id":"renta0426/poisoned-chalice-context-contrast-10k-v1","title":"Poisoned Chalice Context Contrast 10K V1","code_file":OUT.name,"language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":True,"machine_shape":"NvidiaTeslaT4","dataset_sources":[],"kernel_sources":[],"competition_sources":[],"model_sources":[]}
OUT_DIR.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+"\n",encoding="utf-8")
(OUT_DIR/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(OUT)
