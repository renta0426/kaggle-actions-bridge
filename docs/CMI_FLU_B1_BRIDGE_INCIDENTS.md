# CMI-Flu B1 bridge incidents — 2026-09-02

This note records bridge/runtime failures observed while bringing up the first CMI-Flu Part 1 baseline. It contains no Competition Data, predictions, credentials, or private notebook content.

## Request 001 — private source materialization

- Target: `renta0426/cmi-flu-b1-anchor-20260902-001`
- Result: Kaggle kernel `error`; zero verified outputs.
- Cause: runtime `git` source materialization failed before Competition Data processing.
- Action: replaced runtime repository checkout with a self-contained runner.

## Request 002 — anonymous private-repository fetch

- Target: `renta0426/cmi-flu-b1-anchor-20260902-002`
- Result: no Kaggle kernel launched.
- Cause: secret-free bridge validation attempted to fetch the self-contained runner anonymously from the private CMI-Flu repository and received HTTP 404.
- Action: moved the executable runtime source into the public bridge while retaining the private CMI commit/blob as provenance only.

## Request 003 — Kaggle script packaging

- Target: `renta0426/cmi-flu-b1-anchor-20260902-003`
- Result: Kaggle kernel `error` before Competition Data processing.
- Cause: Kaggle executed the configured `code_file` as `/kaggle/src/script.py`; a sibling helper Python module was not importable. The primary exception was `ModuleNotFoundError` for `cmi_flu_b1_kernel_impl`.
- Non-cause: later `mistune` / `nbconvert` `SyntaxWarning` messages came from notebook HTML rendering.
- Action: package the runtime as exactly one `script.py` and reject any sibling-Python dependency before push.

## Request 004 — GitHub concurrency pending eviction

- Target: `renta0426/cmi-flu-b1-anchor-single-file-20260902-004`
- Result: no Kaggle kernel launched.
- Cause: the workflow used workflow-level `concurrency.group: kaggle-resource-global` with the default single pending slot. While request 004 was pending around the protected Environment gate, a separate Kaggle resource workflow entered the same group and GitHub cancelled/replaced the older pending run. This behavior is independent of `cancel-in-progress: false`.
- Evidence: request-004 launch run `33639452159` ended `cancelled` without jobs; no request-004 Kaggle target was created.
- Action: request 005 keeps the same account-wide global group but opts into `concurrency.queue: max`, which allows pending operations to queue rather than replace each other. The live Kaggle active-run guard remains in place immediately before `kernels push`.

## Durable bridge rule

Resource-consuming workflows that share `kaggle-resource-global` must use a queued concurrency policy rather than the default single pending slot. Read-only status/output checks should not occupy the compute concurrency group unless they themselves need exclusive resource mutation.

GitHub reference: `https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency`.
