# CMI-Flu E12c-v2 historical-backbone manual submission Notebook 002

Date: 2026-09-11

## Purpose

Create one private Kaggle Notebook that leaves a manually downloadable `submission.csv` for the fixed E12 portfolio without calling the Competition submission API.

## Parent E12c-v1 incident

Request `20260911-cmi-flu-e12c-manual-submission-notebook-001`, target `renta0426/cmi-flu-e12c-manual-submission-20260911-001`, version 1 was written once and ended in `ERROR` after model construction. The Kaggle log reported `CMI_FLU_E12C_MANUAL_FAILED stage=verify_frozen_candidate exception_type=DataContractError error_code=f6d5ab53cc8a8f25fb0d`. No `submission.csv` was persisted. Version 1 is consumed and is not retried.

The failure was the E12c-v1 exact whole-candidate fingerprint gate after a fresh refit of unchanged task components. The lower-level source of the bit-level drift is not inferred from the hashed exception.

## E12c-v2 repair

E12c-v2 removes the unnecessary refit of unchanged columns.

- exact historical Public-0.218 Task1.2-only CSV is the immutable backbone;
- required backbone SHA-256: `365607d59cd530656b929a1c1c57412cc6d375265a8d1ba10d304c64e012f387`;
- source private Notebook: `renta0426/cmi-flu-manual-probe-files-20260906-001`, current version 1, terminal `COMPLETE`;
- only Task1.3 is regenerated from the frozen strict ASC anchor;
- strict Task1.3 expected prediction SHA-256: `de8bc3b6bbd3e3aad4099b83eb63a1ebbd81c4f4eeb60f77808858f4674be5da`;
- Challenge Task1.3 must have 40 rows and 36 unique values;
- final candidate must differ from the known 0.218 backbone in exactly `Task1.3`;
- unchanged Task1.1, Task1.2, Task1.4, Task2.1, Task2.2 and Task2.3 are not refit.

The final whole-file SHA-256 is not predeclared. It is computed after exact content assembly, written to the private manifest, and independently rechecked against the persisted `submission.csv` by the bridge sanitizer.

## Execution boundary

- execution policy: `kaggle_native_capacity_v2`
- fresh target: `renta0426/cmi-flu-e12c-manual-submission-20260911-002`
- private CPU, Internet disabled
- exactly one Notebook write/run-start call
- automatic compute retries: 0
- no bridge-side remote-capacity admission
- no Competition submission call
- no Public query
- `/kaggle/working` contains only `submission.csv` and `manual-submission-manifest.json` after success
- recovered runner-local files are used only for non-row-level hash/schema validation and are then deleted
- no GitHub artifact upload of `submission.csv`

## Successful operator handoff

On success the sanitizer prints an aggregate-safe line containing the final `submission_sha256`. The user should open the private Kaggle Notebook output, download `submission.csv`, verify that SHA-256 locally against the printed value, and manually upload that file to the Competition.
