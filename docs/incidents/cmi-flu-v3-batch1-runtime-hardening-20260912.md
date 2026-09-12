# CMI-Flu V3 batch1 runtime hardening — 2026-09-12

This incident record covers the repeated deterministic failures encountered while preparing Strategy-v3 batch1 V3-01/V3-03. None of the failures below is a scientific negative.

## Failure sequence

| Generation | Failure class | Side effect | Proven cause |
|---|---|---|---|
| initial precheck | bridge implementation / prewrite | write=0, compute=0 | existing `TemporaryDirectory` was passed to a reader requiring an absent output child |
| ambiguous title/id attempt | bridge implementation / ambiguous write | one write call, no recoverable target after reconciliation | Kaggle title-derived slug did not match the approved target id |
| `20260911-001` v1 | resource-consumed runtime compatibility | write=1, compute=1 | missing `cmi_flu.strategy_e04_contract` in the frozen runtime bundle |
| `20260912-002` v1 | resource-consumed science implementation | write=1, compute=1 | audit incorrectly treated Task1.2 zero strict baseline support as a model-dataset hard error instead of recording `data_limited` |
| `20260912-003` v1 | resource-consumed runtime compatibility | write=1, compute=1 | `audit_measurements()` lazily imported missing `cmi_flu.task13_harmonization` |
| `20260912-004` first main run | bridge implementation / prewrite | write=0, compute=0 | same shell step wrote `WORKDIR` to `$GITHUB_ENV` and then expanded `${WORKDIR}`; `$GITHUB_ENV` is visible only to later steps |

`20260912-004` did **not** reach the Kaggle write step. The immutable request and fresh target therefore remain eligible for a later manually approved execution after the bridge repair; no new successor slug is needed for this prewrite-only failure.

## Durable corrections

### 1. Execute the generated runtime in Stage A

Import/compile smoke tests are insufficient for a runtime assembled from a frozen package plus relayed extensions. Lazy imports, schema expectations and output-boundary defects may only appear after entry-point execution.

For V3 batch1, credential-free CI now executes the exact generated runtime with real numerical libraries and fully synthetic files through:

1. all seven no-fit audit branches;
2. the Task1.2 `data_limited` path;
3. the Task1.3 measurement audit and lazy dependency;
4. all six exact-token singleton CSVs;
5. five aggregate JSON outputs plus diagnostic manifest and runtime receipt;
6. private-output hash validation;
7. aggregate sanitizer;
8. injected late failure and cleanup of partial staged outputs.

The CI also reproduces the consumed `003` lazy-import fingerprint before proving the repaired runtime succeeds. A profiler rejects model-fit and Kaggle write/submit calls. Synthetic CI success is not reported as a real Competition-data result.

### 2. Treat runtime dependency closure as executable, not import-only

Every science module used by the generated runtime, including function-local/lazy imports on the exercised path, must be available from the frozen package or exact reviewed relay. The relayed `task13_harmonization.py` is pinned by its science Git blob and validated before runtime construction.

### 3. Stage output before publication

The Notebook writes to a private temporary staging directory first. The complete allowlist and CSV hashes are validated before any file is published to `/kaggle/working`. A later failure removes partial published files. This prevents a failed Notebook from leaving output that can be mistaken for a successful V3-03 diagnostic set.

### 4. Freeze the exact generated runtime digest

The generated production runtime is deterministic and has an approved SHA-256. PR/main validation and the protected executor must agree on that digest. CI-only synthetic fixtures must not appear in the production Notebook source.

### 5. Lint GitHub Actions shell-state propagation

Appending `NAME=value` to `$GITHUB_ENV` does not assign `NAME` in the current shell. Same-step consumers must use a local shell variable or explicitly assign/export `NAME`; `$GITHUB_ENV` is for later steps.

The bridge now includes `scripts/kaggle_workflow_shell_v2.py` and a repository-wide workflow that rejects changed Actions workflows which publish a variable to `$GITHUB_ENV` and then expand the unpublished shell variable later in the same `run` block. Parameter expansions such as `${NAME-}` used only for safe cleanup remain allowed.

## Execution boundary retained

These corrections do not change the Strategy-v3 scientific expectations, source-B identity, predictions, task definitions, model-selection rules or Competition submission policy. The next real operation remains:

- private CPU Notebook only;
- exact E12c-v2 source B version 1 / 5,926 bytes / approved SHA-256;
- one `save_kernel_once` call after fresh `kaggle-readonry` approval;
- automatic compute retries = 0;
- model fit/refit/HPO = 0;
- Competition submission = 0;
- six diagnostic CSVs remain private until the operator manually submits them.
