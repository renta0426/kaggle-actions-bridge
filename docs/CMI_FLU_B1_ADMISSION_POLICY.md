# CMI-Flu B1 admission policy

This note defines how the immutable CMI-Flu B1 launch request is handled when another Kaggle kernel is already active on the account.

## Resource-class admission

The bridge serializes admission and write operations through the account-wide `kaggle-resource-global` GitHub concurrency group. A remote Kaggle kernel may continue after the launcher exits, so the workflow also checks live Kaggle state immediately before `kernels push`.

Remote compute is not treated as one account-wide slot. Kaggle exposes separate CPU, GPU, and TPU session classes, so an approved CPU request may run while a GPU or TPU kernel is active. For request `20260902-cmi-flu-b1-005`, admission is blocked only by:

- another active CPU kernel; or
- an active kernel whose accelerator class cannot be determined from its Kaggle metadata.

An active GPU or TPU kernel is recorded as non-blocking for this CPU request. The workflow still creates at most one target version and makes exactly one `kernels push` call.

`resource.max_active_runs` is interpreted within the requested accelerator class, not across all accelerator classes on the account. This is a bridge admission limit and does not override stricter live Kaggle or Competition limits.

## Deferred admission

If a same-class or unclassified active kernel blocks admission, the CMI-Flu request is deferred, not executed.

A deferred admission has all of the following properties:

- no Kaggle notebook version is created or updated;
- no compute run is started;
- the request target must still be absent;
- the immutable request ID remains unexecuted;
- no automatic retry or polling is performed;
- the same immutable request may be manually dispatched again only after a fresh protected-Environment approval.

This distinction prevents routine contention from being reported as a model/runtime failure while preserving explicit approval and duplicate protection.

## Why GitHub concurrency is insufficient by itself

The GitHub Actions `kaggle-resource-global` concurrency group serializes bridge admission/write jobs, but the lock ends when the launcher workflow exits. It does not represent all remote Kaggle compute. The live resource-class check therefore remains authoritative for bridge admission.

The bridge uses `queue: max` so protected/pending GitHub runs are not evicted by newer bridge operations.

## Manual redispatch

The CMI-Flu request-005 workflow exposes a no-input `workflow_dispatch` entry point. Redispatch is permitted only for the exact committed request `20260902-cmi-flu-b1-005`; the workflow revalidates source blobs, rules, target nonexistence, and live resource-class state on every attempt. If the target already exists, the operation fails closed rather than creating another version.
