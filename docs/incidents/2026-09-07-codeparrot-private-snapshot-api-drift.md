# 2026-09-07 CodeParrot private scorer snapshot API drift

## Scope

Target: `renta0426/codeparrot-fresh-pilot-v1`, version 1.

The notebook was created successfully and executed on a T4, but failed before model scoring at the configuration cell.

## Observed failure

Kaggle log:

```text
TypeError: FeatureCacheV2Config.__init__() got an unexpected keyword argument 'rank_vocab_block_size'
```

The protected reconciliation later confirmed the exact target exists as current version 1 with GPU enabled, TPU disabled, and status `ERROR`.

## Root cause

The public bridge composer assumed the current research-repository `FeatureCacheV2Config` API. The private Kaggle scorer source is a frozen historical notebook snapshot whose embedded `FeatureCacheV2Config` predates `rank_vocab_block_size`. The same historical source family can also expose the old two-argument `_chunk_statistics(logits, targets)` signature.

This is an orchestration/source-snapshot compatibility defect, not a CodeParrot model failure and not a result about membership-inference performance.

## Repair contract

The repair must not replace the frozen scientific definitions. It:

- removes `rank_vocab_block_size` from the historical dataclass constructor;
- fixes `CONFIG.rank_vocab_block_size = 8192` after construction using `object.__setattr__` because the dataclass is frozen;
- wraps old/new `_chunk_statistics` signatures behind a compatibility implementation;
- uses vocabulary-blocked strict rank for the old signature, which is mathematically identical to the original strict `>` rank count while bounding temporary memory;
- retains max length 768, max batch tokens 1536, sequence chunk 32, local span 64, Min-K++ fraction 0.10, paper block size 8192, and all label-blind boundaries;
- overwrites only the existing failed target as version 2, after verifying version 1 remains `ERROR` and the T4/GPU metadata is unchanged;
- performs exactly one push, with zero automatic compute retries and no competition submission.

## Operational lesson

A private Kaggle notebook used as a source is a versioned code snapshot. Current repository APIs must not be assumed to match it. Source-derived composers must either pin and validate the embedded API contract or provide an explicitly tested compatibility adapter. Public credential-free self-tests should include the historical API shape that is actually expected at protected runtime.
