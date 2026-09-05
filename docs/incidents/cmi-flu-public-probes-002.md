# CMI-Flu controlled Public probes generator 002 incident

Date: 2026-09-06

## Classification

- Request: `20260906-cmi-flu-public-probes-002`
- Actions run: `33980093685`
- Failed stage: read the frozen B2.1 current-v1 backbone
- New Kaggle Notebook write: **not reached**
- New Kaggle compute consumed: **none**
- Competition submission: **none**

## Established root cause

The repaired legacy reader attempted `kernels_status()` for the frozen private B2.1 Notebook `renta0426/cmi-flu-b21-robust-cv-20260903-001`. The currently authorized Kaggle token returned HTTP 403 with `Permission 'kernels.get' was denied`.

The original successful B2.1 GitHub Actions run (`33759100292`) has no retained Actions artifact, and its job log has expired (HTTP 410), so the exact historical `submission.csv` cannot be recovered through GitHub Actions. A File Library search also did not locate the exact row-level B2.1 submission/archive.

## Repair decision

Do not make further attempts to read the old private B2.1 Notebook. Request 003 deterministically regenerates the B2.1 backbone inside the same new CPU Notebook using the frozen B2.1 package, frozen runtime adapter, original Competition Data, and MD5 verification, then applies the already-fixed three-probe family.

The scientific probe definition, exact science blob, and probe family are unchanged. Request 003 remains a generator-only operation and does not submit to the Competition.
