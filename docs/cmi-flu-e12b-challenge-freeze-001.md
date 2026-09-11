# CMI-Flu E12b Challenge prediction freeze 001

Date: 2026-09-11

## Purpose

Run one private CPU Notebook that regenerates the already-reconciled E12a-v2 seven-task Challenge candidate and freezes aggregate content fingerprints. This request performs no new model selection, no Public query, and no Competition submission.

## Exact science

- science repository: `renta0426/CMI-Flu-Invited-Prediction-Challenge`
- science main: `a411bf85a4a79fec2e9a2b9c2bc8cbe186adee5f`
- E12b blob: `af5df92ca81abc34a7f7046dba5bc284c98302d4`
- E12b synthetic blob: `6b36d94299ee3656f9faed623a2ac030a97a714d`
- E12b contract blob: `7e844b2d56bd6739793888bf6306946a0028d966`
- E12a-v2 parent science blob: `0c4c970c8bacfed61bfbb9587e0a0bfdec7903d9`
- B2.1 config blob: `170d3211e2795c0730e481056c7bb068accf97c9`

Final portfolio:

- Task1.1 `b21_pls_2`
- Task1.2 `task12_anchor_residual_et_d5_l5_sqrt_lambda0.5`
- Task1.3 `strict_asc_anchor`
- Task1.4 `raw_pre_vacc_conserved_anchor`
- Task2.1 `b21_et_subtype_d3_l5`
- Task2.2 `b21_et_subtype_d5_l10`
- Task2.3 `b21_ridge_exact_a100`

## Parent E12a-v2

Accepted E12a-v2 request:
`20260911-cmi-flu-strategy-e12a-v2-portfolio-reconciliation-001`

Target/version:
`renta0426/cmi-flu-e12a-v2-portfolio-reconcile-20260911-001` / `1`

Actions run/job:
`34547852282` / `103104288165`

The Kaggle Notebook reached terminal `COMPLETE`, all six supervised final components reproduced, and the accepted aggregate ZIP SHA-256 is:

`7dca4546caa96f3c8fedeba3e4e758c677cb9cd7e7627ab67a79570d66dc0b64`

The Actions job failed afterward only because the installed Kaggle CLI was not on the current-output reader's `PATH`. The parent target/version is consumed and must not be replayed.

## E12b execution identity

- request: `20260911-cmi-flu-strategy-e12b-challenge-freeze-001`
- fresh target: `renta0426/cmi-flu-e12b-challenge-freeze-20260911-001`
- expected version: `1`
- private CPU
- Internet disabled
- automatic compute retries: `0`
- output allowlist: `bridge-result.json`, `metrics.json`, `summary.md`

The runtime constructs the row-level 40×7 candidate only transiently in process memory. It does not persist the candidate CSV, participant identifiers, prediction vectors, fitted models, or OOF rows. Persistent output is aggregate fingerprints and diagnostics only.

## Structural controls

The runtime regenerates three transient candidates:

1. frozen B2.1;
2. frozen Task1.2-only portfolio;
3. final E12b portfolio.

PASS requires:

- final vs B2.1 changes exactly Task1.2 and Task1.3;
- final vs Task1.2-only changes exactly Task1.3;
- Task1.3 strict ASC has 40 Challenge rows and 36 unique prediction values;
- all Public-scored columns are nonconstant;
- exact seven-task portfolio identity is preserved.

The historical Public `0.218` value is not used by this run and byte identity with an old CSV is not assumed.

## Fingerprints

E12b emits:

- semantic full-candidate SHA-256 over row order, participant identity bytes, task names, and float64 prediction bits;
- canonical CSV SHA-256 using LF and `%.17g`;
- one semantic prediction SHA-256 per task;
- aggregate min/max/unique/tie statistics and rank-movement diagnostics.

These hashes are the future submission gate. A separately authorized submission must regenerate the candidate and match the accepted E12b fingerprints before any submission API call.

## E12a-v2 PATH recurrence prevention

The protected job installs locked `kaggle==2.2.4` in a fresh venv. The E12b executor prepends the directory containing its own Python executable to `PATH`, then requires `shutil.which("kaggle")` to resolve to that same venv directory.

Credential-free CI installs the same locked client and runs only `--path-self-test`. It performs no authentication, write, or compute.

The schema column name `participant_id` is allowed only as part of the validated submission-schema metadata. A row-level dictionary key `participant_id` is still rejected by the sanitizer regression.

## Approval boundary

PR validation is secret-free. After merge, only the main-push protected job under `kaggle-readonry` may authenticate and perform the one SaveKernel operation. Approval authorizes only this exact request/commit/run.

E12b does **not** authorize Competition submission. Successful next state is:

`ready_for_separate_submission_authorization`
