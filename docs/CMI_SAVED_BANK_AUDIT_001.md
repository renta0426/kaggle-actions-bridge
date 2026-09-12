# CMI-Flu saved-bank audit 001

## Reason and boundaries

V3-05 repair run 34683870785 / protected job 103527356923 completed successfully. All four scale heads worsened source RMSE relative to raw baseline; no candidate was promoted. However the rebuilt PLS reference differs from the V3-02 reference and S3 becomes constant on all 40 Challenge rows despite its positive coefficient. Investigate the saved artifacts without fitting, rerunning either Notebook, changing predictions, or inferring hidden labels.

Science closeout and auditor: PR #83, main `c822a0ca0bfb8406e4ea293a2d9706855bf6e181`. Exact source blob `aace13086964535ffc6c3ef5a0dcfe976b54289b`; synthetic test blob `1b833cda295ba0f60d61cdaf747ba72fdd1cdad3`. The auditor is standalone; it does not import historical layered runtime bundles.

Request: `requests/cmi-flu-saved-bank-audit-001.json`. Driver: `scripts/cmi_flu_saved_bank_audit_v1.py`.

Read only the current version 1 output of:

- `renta0426/cmi-flu-v3-v02-validation-bank-20260912-001`
- `renta0426/cmi-flu-v3-v05-task13-scale-20260912-002`

The eight file sizes and SHA-256 identities are pinned in both the request and exact science source. `kaggle_current_output_read.py` requires current == expected before and after download. If either version or any file hash changes, stop; do not silently substitute latest, create a successor Notebook, or refit a reference.

Output operations: two private output reads. New Kaggle writes: 0. Compute launches: 0. Fits: 0. Competition submissions and Final Submission selections: 0. No automatic compute or readout retry; the established reader retains only its bounded read-only metadata reconciliation.

## Export and validation

Private CSVs exist only in the protected job's temporary directory and are removed unconditionally. The CSV parsers, pairing checks, formula verification and diagnostics run before the safe file is published. Only `saved_bank_audit.json` may be emitted, after the aggregate schema and no-row-field sanitizer pass. No Actions artifact, cache, Notebook source or raw log upload is added.

The audit compares independent subject/teacher identities, repeated-OOF aggregation, old/new saved PLS predictions and actual Challenge weak orders. It checks whether S3 ECDF inputs are outside or inside source support. It does not authorize a new mapping, unit conversion or candidate. The V02 bank omitted split assignments; the audit explicitly leaves historical split identity unresolved rather than fabricating it.

Local tests: 14 exact science regressions and 8 driver integrations, all successful. They exercise synthetic CSV round trips, full parser->science->sanitizer->cleanup success, positive-slope ECDF collapse, version/hash/allowlist/identity rejection, captured private transport stdout, zero fit/write calls and failure cleanup. These tests are not the real two-bank audit. YAML, all shell steps and embedded Python were checked locally without starting an Actions runner.

## Manual execution and human approval

Workflow: `.github/workflows/cmi-flu-saved-bank-audit-001.yml`.

Trigger is **workflow_dispatch only**, on main, with no operator-supplied target, URL, command, version or content inputs. First the credential-free validation job runs the exact relayed tests. The readout job then waits for human approval of `kaggle-readonry`. The service connection exposed in the authoring session had GET workflow/run reads and rerun actions, but no workflow-dispatch action; a rerun of the old writer is not a substitute for dispatching this reader. Start this workflow manually and approve only its readout job.

Do not approve the stale consumed V05-001 writer run 34683870989 / job 103527359090. The new reader paths intentionally do not match old `scripts/cmi_flu_v3_v05_*.py` launch globs.

## Policies

See [Execution Policy v2](EXECUTION_POLICY_V2.md) and [Operational lessons](OPERATIONAL_LESSONS.md). No remote-capacity heuristic or automatic launch event is introduced. The science repository's private baseline workflow was made manual-only before PR #83, without changing the test job body or action pins. No private CI minutes are requested by this readout, which runs only in the existing public bridge after explicit manual dispatch and Environment approval.
