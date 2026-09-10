# CMI-Flu E11c robust TabPFN 001 bridge record

## Purpose

This bridge request executes the predeclared CMI-Flu Strategy-v2 E11c Task1.1 bounded robustness/fusion experiment from science commit `0815f4517345e127dbcbcae0f380a54f3a3d15bd` on a fresh private CPU Kaggle target.

## E11b-005 consumed boundary

The prior target `renta0426/cmi-flu-e11b-tabpfn3-20260910-005` version 1 is consumed. It is not rerun by this request. The E11b scientific result is already valid and closed; this request changes no E11b scientific setting or result.

## Read-only metadata audit evidence

The first protected metadata audit, Actions run `34458275946` / job `102809839733`, failed locally before a metadata HTTP read because locked `kaggle==2.2.4` exposes no `KaggleApi.kernel_metadata` method. It performed no write, compute, submission, or output download.

The repaired read-only audit 002, Actions run `34460007561` / job `102815418161`, made one exact GetKernel read and succeeded. It observed:

- requested model source: `prior-labsai/tabpfn-3/pytorch/default/1`
- GetKernel model source: `prior-labsai/tabpfn-3/PyTorch/default/1`
- Competition source: `cmi-flu-first-prediction-challenge`
- model source count: 1
- Competition source count: 1
- write: false
- compute: false
- submission: false
- output download: false

Therefore the validator repair permits case normalization only for the five-segment model source's framework segment. Owner, model, variant, and version remain exact; the Competition source remains exact; both source counts must be exactly one.

## Frozen E11c execution

The fresh target is `renta0426/cmi-flu-e11c-robust-tabpfn-20260910-001`, expected version 1. The exact TabPFN package/checkpoint identity, CPU resource, seed, estimator count, C0–C4 definitions, 0.75/0.25 fusion, independent C2/C4 promotion gates, all-three source-drop C3 diagnostic, outcome-independent audits, 28-subject sensitivity, and maximum 21 real TabPFN fits are frozen by the request manifest and science source.

There is no automatic compute retry, Public/Leaderboard selection, Competition submission, or automatic incumbent change. PR validation is secret-free. A merged `main` commit requires a new protected `kaggle-readonry` Environment approval for the one fresh CPU operation. Submission authorization remains separate.
