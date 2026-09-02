# CMI-Flu B2 reference baseline

This bridge request runs the Part 1 `B2 Taskwise Compact` reference baseline for `cmi-flu-first-prediction-challenge`.

## Request

- Request ID: `20260903-cmi-flu-b2-001`
- Target: `renta0426/cmi-flu-b2-taskwise-compact-20260903-001`
- Accelerator: CPU
- Internet: disabled
- Automatic compute retries: zero
- Competition submission: not performed by the kernel launch
- CMI source contract: `342c9c1208f1dba28e58c1698fc399f7eca4b5be`

The generated Kaggle package contains exactly one `script.py`. The readable implementation is compressed only to keep the bridge payload below connector limits; the request pins every segment, the compressed stream, and the reconstructed script by SHA-256.

## Work performed inside Kaggle

1. Verify the organizer MD5 manifest.
2. Check the six published aggregate target counts.
3. Build B1 OOF anchor diagnostics.
4. Run demographics/subtype negative controls.
5. Fit task-specific Ridge, Elastic Net, PLS, and conservative ExtraTrees candidates.
6. Use subject-purged leave-one-study-out validation for Tasks 1.1, 1.2, and HAI.
7. Use repeated subject GroupKFold for Task 1.3, which has one public study.
8. Run HAI leave-one-vaccine-season-out and leave-one-strain-out stress tests.
9. Save OOF predictions, fold metrics, candidate summaries, fitted models, aggregate metrics, and a validated 40-row submission as private Kaggle outputs.

Task 1.4 remains an unlabeled, hypothesis-driven Pre-vacc Conserved AIM anchor and receives no fabricated local CV score.

## Local preflight

Before committing the payload, the reconstructed script was checked in a local synthetic workspace with Python warnings promoted to errors.

- Fast single-Ridge integration path: passed.
- Full candidate grid: passed.
- HAI season and strain stress tests: passed.
- Output contract: 40 rows, seven nonconstant task columns, no `-99` values.
- Full synthetic runtime: approximately 23 seconds and 182 MB maximum RSS.

Synthetic scores are intentionally not interpreted as evidence of biological generalization.

## Safety boundary

The Kaggle script has no Git, HTTP client, socket, subprocess, package-install, Kaggle submission, or kernel-launch capability. It reads the mounted Competition Data and writes only private Notebook outputs. Launch and Competition submission remain separate protected operations.
