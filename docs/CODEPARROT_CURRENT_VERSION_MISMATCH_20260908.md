# CodeParrot current-version guard stop — 2026-09-08

This supersedes the operational status of run 34177475292 as approval-waiting. It does not alter any scientific result or frozen contract.

## Evidence

Run 34177475292, attempt 1, commit 14e24c90fb33ca296cfb62328b1836dc3bb83816 was approved and its protected job 101909753915 ran at 2026-09-08 01:45:47 UTC. The preparation and both exact-filename and mathematical self-tests passed.

At 01:46:24 UTC, the output-reader's pre-download exact metadata validation raised `IdentityError("current_version_mismatch")`. The call chain is `read_current_output` -> `_verify_current_kernel` -> `verify_current_eventually` -> `validate_metadata`.

The response satisfied ref/private checks but did not satisfy the expected current-version-1 check. The actual observed version value/type was not included in that diagnostic. Therefore it is not yet established whether the kernel advanced, the SDK field was missing, or a metadata representation issue occurred. Do not invent the observed version or silently set expected version to 2/latest.

The filename repair is effective; this is a different identity boundary. No output CLI invocation was reached. No five-file hash comparison, membership source reconstruction, label join, metric evaluation or bootstrap ran. Kaggle writes/compute/submissions were zero. GitHub CPU/setup and a temporary local credential were used. The final cleanup explicitly reports raw_prediction_retained=0, row_label_join_retained=0, credentials_retained=0.

## Stop and diagnose, not retry

Do not rerun 34177475292. Do not change request 214, its expected version, hashes, source model, formulas or gates. Do not download whichever output is current; no historical-output fallback is authorized.

A separate metadata-only diagnostic is defined by request `20260908-codeparrot-current-identity-after-mismatch-001` and workflow `215-codeparrot-current-identity-diagnostic.yml`. It has its own secret-free PR regression and protected human `kaggle-readonry` approval.

It invokes exact GetKernel once, projects only fixed expected ref, ref/private check results, safe numeric version value/type and frozen-version comparison, and then stops. Output download, Kaggle write/compute, membership source access and performance evaluation are absent. It never updates an expected version or launches a recovery. SDK exceptions are summarized without raw bodies or credentials. No private science source is copied to public bridge.

Regression covers observed 1, 2, missing/None, bool, numeric string, unprintable canary string, ref/private mismatch and failed read without retry. Local Python self-test, JSON, YAML and embedded-Python compilation passed. A successful diagnostic workflow means the observation completed, not that the original v1 output passed identity or that membership evaluation succeeded.

After the observation, choose a separately authorized recovery from the proven identity state. If original v1 is no longer current, keep frozen evaluation blocked rather than overriding the contract. An explicitly supplied original result bundle with the existing five hashes can be considered under a separately reviewed input path, but must not be silently substituted by this diagnostic.
