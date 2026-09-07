# CodeParrot pilot v2: missing `mean_log_rank` after successful forward

## Observed failure

The approved `renta0426/codeparrot-fresh-pilot-v1` version 2 run loaded the pinned CodeParrot model, completed the 200-row same-forward scorer, and then failed during sample aggregation with:

`AttributeError: 'DataFrame' object has no attribute 'mean_log_rank'`

The bridge write itself succeeded and created version 2. Membership labels or evaluation manifests were not attached, and no membership performance metric was computed.

## Root cause

The private `poisoned-chalice-minkpp-paper-10k-v1` source ultimately inherits the older 500-row Min-K++ pilot runner. That runner accumulates token `logp`, legacy normalized-logit Z, paper probability-weighted Z, and paper variance, but it does not materialize token rank. The CodeParrot adapter incorrectly assumed the resulting window table already contained `mean_log_rank`.

The frozen science definition is explicit: per-window `mean_log_rank = mean(log1p(rank))`, where `rank = 1 + count(logit > target_logit)`, and the sample feature is the negative mean across windows.

## Repair rule

Do not impute, zero-fill, derive rank from Z, or drop the feature. Version 3 must compute strict rank from the same logits already used for loss and both Min-K++ variants, with vocabulary blocking only as a memory-equivalent implementation detail. It must add a first-8 comparison against the historical `mean_log_rank` extractor before accepting the repaired feature path.

The repair is performed by pulling the exact failed private version 2 source under the protected runner, patching it runner-locally, and pushing exactly one version 3. It does not reconstruct the notebook from a different source snapshot.

## Operational consequences

- failure occurred after GPU forward work, so resource time was consumed;
- version 2 remains a failed experimental artifact and must not be overwritten blindly by rerunning its workflow;
- version 3 requires a fresh protected approval;
- no automatic compute retry is allowed;
- Public-LB submission and membership performance evaluation remain out of scope.
