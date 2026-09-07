# 2026-09-07 CodeParrot pilot post-write visibility incident

## Summary

Protected run `34113474602` successfully passed the policy/resource preflight, composed the approved private 200-row CodeParrot pilot, and `kaggle kernels push` returned exit code 0. The immediately following verification queried `kernels_list(user=..., search=...)` roughly 0.6 seconds later and did not see the target, so the GitHub Actions job was marked failed.

## Confirmed facts

- The write command reported success: `CODEPARROT_PILOT_REPAIR_KERNEL_PUSH PASS count=1 retry=0 submission=0`.
- The failure occurred only in the post-write visibility check.
- No retry was performed.
- The target must be reconciled read-only before any further write because a successful push followed by a stale listing is an ambiguous-write state.
- Private source material was cleaned from the runner.

## Operational concern

`kernels_list(..., search=...)` may be eventually consistent immediately after `kernels push`, or otherwise may not be suitable as the sole synchronous proof of successful creation. This is recorded as an observed integration concern, not yet asserted as a Kaggle CLI/API defect.

## Rule going forward

1. A successful write followed by an immediate list miss is **not** evidence that the write did not occur.
2. Never blind-retry the write in that state.
3. Reconcile using read-only direct target status / metadata lookup plus list visibility under a fresh protected approval.
4. Only if all direct and listing paths establish absence may a later, separately approved write be considered.
5. Future write workflows should avoid treating one immediate `kernels_list` miss as definitive failure; use bounded read-only reconciliation semantics instead.

This incident did not consume CodeParrot membership labels or performance metrics; it concerns only private Notebook creation/visibility.
