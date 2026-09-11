# CMI-Flu E12c manual-submission Notebook 001

Date: 2026-09-11

## Purpose

Create one private Kaggle Notebook that independently regenerates the frozen E12b 40×7 candidate on Competition data, verifies the exact E12c submission gates, and leaves `submission.csv` in the private Notebook Output for operator-managed manual upload.

This operation does **not** call the Competition submission API and does not read or optimize against the Public leaderboard.

## Exact science

- science main: `bb6f41ed81b0bbd4ecdc397c6abb9df671c6ebf8`
- E12c source blob: `a495584a7e461478bd1a41d01d2536c4435a8f7f`
- parent E12b source blob: `af5df92ca81abc34a7f7046dba5bc284c98302d4`
- final semantic SHA-256: `5faf93bee4b68aba23c53e09adbb251bdd07d287dd48fb6d94d6fce53ef9a60f`
- final canonical CSV SHA-256: `983aaf097d04477c4ccf7bf817fdf66e552937e69bceaa48cfb260cb84413f1b`
- final canonical CSV bytes: `5933`

## Historical 0.218 control

The Notebook attaches the existing private source:

- `renta0426/cmi-flu-manual-probe-files-20260906-001`
- required current version: `1`
- required terminal state: `COMPLETE`
- required Task1.2-only CSV SHA-256: `365607d59cd530656b929a1c1c57412cc6d375265a8d1ba10d304c64e012f387`

The source version/state are rechecked before the single Kaggle write. The attached CSV itself is accepted only by exact hash inside the Notebook.

## Fail-closed runtime gates

The Notebook does not write `submission.csv` until all of the following have passed:

1. frozen B2.1 input MD5 verification;
2. exact E12b final portfolio generation;
3. E12b aggregate contract validation;
4. E12c semantic hash equality;
5. E12c canonical CSV hash and byte-count equality;
6. exact historical Task1.2-only hash equality;
7. exact participant alignment with the historical control;
8. proof that `Task1.3` is the only changed task versus the known Public-0.218 candidate.

On any exception the runtime removes both declared outputs before failing.

## Private Output

Successful execution leaves exactly:

- `submission.csv` — 40 rows, eight columns, 5,933 bytes, SHA-256 `983aaf097d04477c4ccf7bf817fdf66e552937e69bceaa48cfb260cb84413f1b`;
- `manual-submission-manifest.json` — aggregate provenance and hash contract only.

The bridge may recover these two files transiently to the GitHub-hosted runner solely to verify the fixed hash/schema. Row-level content is never printed, uploaded as a GitHub artifact, committed, or retained after the runner cleanup step.

## Execution policy

The request uses `kaggle_native_capacity_v2`.

- private CPU Notebook;
- Internet disabled;
- no bridge-local session/quota/capacity admission;
- exactly one Kaggle Notebook write/run-start call;
- no automatic compute retry;
- fresh exact target only;
- protected `kaggle-readonry` approval required;
- Kaggle is authoritative for platform capacity.

Target:

`renta0426/cmi-flu-e12c-manual-submission-20260911-001`

After successful execution, the operator downloads `submission.csv` from the Notebook Output and manually uploads it to the Competition. Competition submission and Final Submission selection remain outside this workflow.
