# CMI-Flu E02: incident, recovery and future operating procedure

## Status and evidence boundary

This is an engineering incident, not a negative model result. The affected production run is [34079658937](https://github.com/renta0426/kaggle-actions-bridge/actions/runs/34079658937), protected job `101612426080`, bridge commit `2010e1cec8f9bfa4810a97bb7e35b76319abef96`. At 2026-09-07 04:38:59 UTC (13:38:59 JST), the job failed with `pushed E02 kernel exact discoverability mismatch`.

Observed from the original log and exact source:

1. The protected boundary, original runtime reconstruction, CLI installation and live Competition-rule preflight succeeded.
2. `kaggle kernels push` returned exit code 0. The error occurred afterwards, in a `kernels_list(user=owner, search=slug)` lookup.
3. The executor therefore reached a write call. It is incorrect to report this as a pre-write failure, a private science repository authentication failure, or proof that no Notebook exists.
4. The original CLI response was captured but not preserved as a safe receipt. The log alone does not establish the returned kernel identity, remote terminal status, or resource consumption. Classify this as **push acknowledged / post-write identity unconfirmed**. CPU execution may already have happened.
5. The approved runtime SHA-256 is `ad36802efe2af9d4678b5d216b4afbb904a74746ec971818e75fdb2692709f37` and expected kernel version is 1.

The exact reason the search result omitted the target remains unconfirmed: title/slug search semantics, private visibility and search indexing can all differ from direct resource lookup. Do not claim an upstream Kaggle defect or eventual-consistency incident without a reproduction. The bridge defect is already established: **it made search discoverability a mandatory proof of exact identity, before and after a side-effecting operation**. The shared `kaggle_current_output_read.py` had the same defect, so fixing only the executor would have failed again during output retrieval.

## A second, independent failure found before another launch

The old E02 builder reused E01's output writer. E02 returns `conditions`, `fixed_conditions` and `task`, while the inherited writer reads `result['frozen_incumbent']` and later `result['tasks']`.

The order is important: `metrics.json` and `summary.md` are written **before** the first invalid key lookup. Consequently, a failed Notebook may contain complete scientific metrics even when no `bridge-result.json` exists. This is a source-derived failure mechanism until checked against the remote output, not an assertion that it already happened in Kaggle.

The regression suite executes the original output lifecycle with a synthetic E02 result and requires this failure to reproduce; it then executes the repaired lifecycle and requires all three original final files and a zero exit code. The release builder changes only the AST-bounded `execute` function, verifies the original runtime SHA-256, and proves all seven embedded source/config/package strings are unchanged. Scratch material is moved outside the export directory.

## Why repeated green/failed CI cycles did not converge

There were multiple independent failure classes, not one continuously failing statistical method:

- Hand-relayed source chunks lost line boundaries; byte-identity checks correctly rejected them. A single exact Git blob removes this entire class.
- Broad string slicing between function names crossed the `sha256_file` and `execute` functions between `render_summary` and `main`, deleting the execution body. Changing whitespace/token patterns repeatedly could not repair that structural mistake.
- Compile-only self-tests verified syntax and blob identity, not actual imported signatures, scientific entry execution, or output serialization.
- Frozen B2.1 package regeneration included later experiment modules; package provenance and current source development were not separated sufficiently.
- An E01 fixture had fewer observations than the existing Spearman minimum; testing the intended identity property requires a fixture within that pre-existing metric contract, not lowering the metric requirement.
- No durable post-write receipt/state distinguished successful upload from successful computation and successful result export.

The failure cannot be solved solely by telling an operator to click approval more carefully. It requires tested runtime and bridge changes. Conversely, the available evidence does not establish that GitHub Actions, pandas or Kaggle itself is generally defective.

## Recovery selected now: read existing version, do not rerun

The new workflow is `175-cmi-flu-e02-current-recovery-001.yml`. It requests **zero new Kaggle writes, zero Notebook executions and zero submissions**. The old 174 launch entry is retired; its historical implementation remains available in Git history.

A fresh `kaggle-readonry` approval authorizes only this recovery:

1. Verify exact metadata and current version 1 directly, without any search or latest-version substitution.
2. Read current code once using official `kernels pull`, hash it against the approved runtime, and never execute the downloaded code.
3. Recheck the version, read current output once with official `kernels output`, and recheck version again.
4. Keep all downloads in temporary runner storage. Reject unexpected outputs, symlinks, duplicates and oversized results. CLI stdout/stderr and the raw transport log are not public output.
5. Validate the full E02 metrics with the approved public runtime validator. Emit only a typed numerical/study aggregate projection for scientific analysis.
6. If the original bridge receipt is absent, require the exact known `write_outputs` / `KeyError:'frozen_incumbent'` error signature as well as source-hash and version proofs. Produce an explicitly named **recovery receipt**, never a forged original bridge receipt.
7. If outputs are absent, source/version differs, the run is still active, or an unknown failure occurred, stop. No retry, alternate resource or compute branch is hidden in this recovery.

The original Notebook may remain marked failed even when its scientific metrics are recoverable. Report those states separately. No E02 performance claims can be made until these checks succeed.

## Canonical procedure for the next experiment

### A. Source transport: the agent is the private-repository bridge

The competition science repository is private. **CI must not read it**, with or without checkout, raw URLs, API calls, PATs, installation tokens or SSH. No additional research-repository credential is requested.

The agent reads the authorized science commit through the user's `@GitHub` tool, relays only reviewed publication-safe experiment code/config into the public bridge, and verifies each Git blob SHA plus the generated runtime SHA-256. Competition data, participant identifiers, sequences, private output and secrets are never relayed into the public repository. Reuse already-relayed identical blobs rather than copying them again. CI materializes only its own approved public bridge commit; actual Competition Data are supplied to a private Kaggle Notebook by Kaggle.

### B. One test ladder before one approval

Use a single PR with the following gates, not a new production run for each discovered integration issue:

- Exact payload/config/runtime hash and model/fold/seed contract checks.
- Actual installed, locked SDK request-shape checks, without authentication.
- Negative tests for empty search results, wrong ref/version, missing privacy, unknown status, failed-run versus successful-run read permissions and secret-free diagnostics.
- Frozen package + adapter + exact scientific entry on synthetic data, including the intended callable namespaces. This is a smoke test, not real-data CV or Kaggle-image equivalence.
- Full `execute` lifecycle through metric validation, serialization, receipt generation, final message and cleanup. A `--self-test` that only compiles is insufficient.
- For source transformations, edit a named top-level AST node; assert neighboring functions and embedded source literals are untouched. Never add another speculative regex/substring patch layer to make a check pass.

When a local container has the files/dependencies, run this same ladder locally first. In a network-restricted agent environment, use the secret-free CI job to materialize the **public** bridge; do not diagnose by creating a Kaggle compute run. Read the first complete failing traceback once, reproduce it, then make one supported repair. Neither repeated status polling nor splitting tests arbitrarily is a substitute for obtaining the error.

### C. Write and read states are different

For a future, separately approved compute launcher, validate title/slug consistency and resource admission before the write and recheck admission immediately before the one permitted write. Emit a safe receipt before/after the call: request/commit/runtime hash, write-attempt flag, CLI return code, stdout/stderr byte counts and digests. Obtain exact metadata directly after push; do not require list/search visibility. Classify an interrupted or unconfirmed write as ambiguous and reconcile it read-only before considering another push.

Keep these states distinct: prepared, write attempted, upload acknowledged, exact identity confirmed, remote terminal, scientific metrics validated, output receipt complete. A green bridge build is not a model result. A non-green orchestration job is not proof the Notebook never ran.

### D. One concrete completion report

Record the experiment and engineering result separately. Include fixed source hashes, task/model/fold contract, metrics and study-level comparisons, negative controls, adoption decision, actual resource use when known, and any unresolved boundary. Request operator action only at an actual approval/access boundary. Do not claim to continue after ending a response.

## Files implementing these changes

- `scripts/kaggle_exact_identity.py`: direct metadata and exact status contracts.
- `scripts/kaggle_current_output_read.py`: successful-current-output guard; version recheck after download.
- `scripts/cmi_flu_e02_recover.py`: fixed read-only recovery; bounded typed aggregate export.
- `scripts/build_cmi_e02_release.py`: provenance-locked, AST-scoped output-sink repair; no remote operation.
- `scripts/test_cmi_e02_contracts.py`: unit, original-failure, repaired-lifecycle and synthetic scientific-entry tests.
- `requests/cmi-flu-e02-current-recovery-001.json`: fresh, bounded read-only approval contract.

These tests prevent the reproduced classes of failure; they cannot guarantee zero failures from future SDK/server changes, an untested Kaggle image, or genuine data-contract problems. Unknown conditions continue to fail closed rather than silently changing the experiment.
