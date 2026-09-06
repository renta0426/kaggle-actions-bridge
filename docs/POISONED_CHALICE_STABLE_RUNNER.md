# Poisoned Chalice stable kernel runner

## Why this exists

The Min-K++ pilot exposed two avoidable workflow-authoring failure patterns:

- workflows 197 and 198 failed before a job could start;
- workflow 199 then started correctly, rendered the current Competition Rules successfully, but its credential-free preflight used the nonexistent URL `https://www.kaggle.com/legal/acceptable-use` and failed with HTTP 404;
- the first PR validation of workflow 200 proved the corrected centralized policy preflight worked, then correctly failed before merge because it attempted anonymous `raw.githubusercontent.com` reads from the private research repository;
- all of these failures happened before any Kaggle kernel push, so they consumed no Kaggle GPU run and created no Kaggle Notebook version.

The correct Kaggle Acceptable Use Policy URL is `https://www.kaggle.com/aup`. More importantly, policy URLs and launch mechanics must not be copied into every experiment workflow, and the protected/credential-free path must not depend on runtime access to the private research repository.

## Canonical workflow

`.github/workflows/200-poisoned-chalice-stable-kernel-runner.yml` is the canonical launch path for supported Poisoned Chalice T4 experiments.

Routine experiments MUST NOT add another workflow YAML. They update:

1. `requests/poisoned-chalice-kernel-run-active.json`;
2. one request-specific launcher under `runners/poisoned_chalice/`;
3. exact public bridge snapshots under `materialized/<target-slug>/` whose reconstructed Git blobs are proven identical to the approved private-research files.

Changing the stable workflow itself is reserved for a capability change to the bridge.

## Merge gate

A new request is first opened as a pull request. The stable workflow's credential-free `validate` job must finish successfully before merge. Validation:

- verifies repository/actor boundaries and absence of credentials;
- checks the centralized current Kaggle Terms, AUP, Community Guidelines, API docs, and Notebook docs;
- validates the active request schema and one-shot side-effect contract;
- verifies the request-specific launcher Git blob;
- reads only the approved public bridge snapshots and verifies that their reconstructed Git blobs match the research provenance recorded by the request;
- compiles the launcher/research sources;
- builds the final Kaggle Notebook package without executing model inference;
- validates target id, title-derived slug, privacy, accelerator, Internet setting, attached sources, and scientific markers.

**Do not merge a request while this validation is red or absent.** Repair the branch without merging. If several preparatory commits are required, temporarily close the PR and reopen it only after the bundle is internally complete so the validation history stays useful rather than noisy.

## Protected launch

After a green PR is merged to `main`, the `launch` job waits on the existing `kaggle-readonry` Environment. Approval authorizes only that merged commit and active request.

After approval the runner:

1. reconstructs the already validated bundle from the exact merged public bridge snapshots;
2. installs the SHA-256 locked Kaggle CLI 2.2.4;
3. rechecks the centralized public Kaggle policy pages;
4. authenticates with `KAGGLE_API_TOKEN` only in the final write step;
5. obtains current Competition Rules/Evaluation/Data pages through the Kaggle API and verifies the Poisoned Chalice evaluation contract;
6. verifies the target does not already exist;
7. checks GPU quota and active-run admission;
8. performs exactly one `kaggle kernels push` and never automatically retries compute;
9. cleans all runner-local material.

A nonzero push is classified as an ambiguous write with only return code, byte counts, and hashes logged. Raw CLI output is not published and the request is not automatically retried.

## Shared policy preflight

`scripts/kaggle_policy_preflight.py` is the single source of truth for the public Kaggle policy URLs. Experiment launchers and workflows must not hard-code alternative Terms/AUP/Guidelines URLs.

Competition-specific Rules are deliberately checked through the authenticated Kaggle API immediately before the write, using the same pattern already proven by the successful Gold start-boundary workflow.

## Private research provenance

The research repository remains private and is never made a runtime dependency of the bridge. For an approved experiment, only the minimal required files are copied into `materialized/<target-slug>/` on the public bridge. Each request records the original private-research Git blob SHA and a byte bound. The controller reads the public snapshot (or a bounded, explicitly declared number of snapshot parts), reconstructs the original bytes, and refuses to continue unless the reconstructed Git blob SHA is identical.

No `RESEARCH_REPO_READ_TOKEN`, PAT, deploy key, or broader GitHub credential is introduced into the protected Kaggle job.

## Current Min-K++ request

The first active request on this runner is `20260907-poisoned-chalice-minkpp-paper-pilot-500-001`, targeting `renta0426/poisoned-chalice-minkpp-paper-pilot-500-v1` on Kaggle T4. It is a 500-row definition/runtime pilot and explicitly forbids using the sample to choose a method from TPR@1% or small low-FPR count differences.

The three required private-research files are now materialized as exact public snapshots. Their bridge Git blob SHAs equal the recorded research Git blob SHAs, including the 25,283-byte Notebook builder.

## Deprecated failed workflows

Workflows 197, 198, and 199 are historical failed launch attempts. Do not rerun them. Once workflow 200 has completed its first real launch successfully, they can be removed in a separate cleanup change without changing the canonical runner.
