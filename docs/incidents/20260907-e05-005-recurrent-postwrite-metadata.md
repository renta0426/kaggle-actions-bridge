# E05 005 recurrent post-write metadata incident

## Classification

This is a **recurrent bridge transport failure**, not a new E05 scientific failure.

The approved E05 005 diagnostic `kaggle kernels push` returned exit code 0. The immediately following SDK `GetKernel` call returned HTTP 403, causing the Actions job to fail before it could reconcile the already-created target. This repeats the post-write visibility/metadata failure class previously documented in the bridge.

Separately, the Kaggle Notebook itself executed far enough to emit the intended safe traceback-site marker. That marker proves the persistent E05 scientific `DataContractError` occurs at `hai_transfer.build_sequence_lookup -> contracts.require_columns`; this is distinct from the Actions-side HTTP 403.

## Prevention rule

1. A successful `kaggle kernels push` followed by a metadata/list/status failure is an ambiguous-write state, never proof of absence.
2. No executor may blind-retry the write from that state.
3. Immediate post-write `GetKernel` must not be a single mandatory success criterion. Reconciliation must tolerate endpoint-specific 403/eventual-consistency failures and use multiple read-only signals under bounded rules.
4. New HAI runtimes must validate the exact hash-locked organizer reference schema before protected Kaggle execution.
5. CI must include a regression that fails if an E05 executor delegates post-write identity exclusively to the legacy `kernel_meta`/`GetKernel` path.
