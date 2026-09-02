# CMI-Flu B1 admission policy

This note defines how an immutable CMI-Flu B1 launch request is handled when another Kaggle kernel is already active on the account.

## Admission defer

A live account-wide active-kernel check occurs immediately before any `kernels push` call. If another Kaggle kernel is active, the CMI-Flu request is deferred, not executed.

A deferred admission has all of the following properties:

- no Kaggle notebook version is created or updated;
- no compute run is started;
- the request target must still be absent;
- the immutable request ID remains unexecuted;
- no automatic retry or polling is performed;
- the same immutable request may be manually dispatched again only after a fresh protected-Environment approval.

This distinction prevents routine account contention from being treated as a model/runtime failure while preserving the bridge rule that resource-consuming execution requires a fresh human approval.

## Why GitHub concurrency is insufficient by itself

The GitHub Actions `kaggle-resource-global` concurrency group serializes bridge jobs, but the lock ends when the launcher workflow exits. A Kaggle kernel can continue running remotely after that point. Therefore remote Kaggle activity remains the authoritative admission check.

The bridge uses `queue: max` so protected/pending GitHub runs are not evicted by newer bridge operations. It still performs the live Kaggle active-kernel check immediately before the write.

## Manual redispatch

The CMI-Flu request-005 workflow exposes a no-input `workflow_dispatch` entry point. Redispatch is permitted only for the exact committed request `20260902-cmi-flu-b1-005`; the workflow revalidates source blobs, rules, target nonexistence, and active Kaggle state on every attempt. If the target already exists, the operation fails closed rather than creating another version.
