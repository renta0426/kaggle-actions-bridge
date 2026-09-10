# CMI-Flu E11c robust TabPFN 002

This request is the manual successor to E11c-001 after Actions run `34474957373`, job `102863653370` stopped at the CPU admission guard before any SaveKernel call.

## Parent outcome

- request: `20260910-cmi-flu-strategy-e11c-robust-tabpfn-001`
- target: `renta0426/cmi-flu-e11c-robust-tabpfn-20260910-001`
- failure class: `resource_concurrency_admission_prewrite`
- model access preflight: passed
- fresh target absence: confirmed
- SaveKernel attempted: no
- kernel version consumed: no
- E11c compute started: no
- scientific result produced: no

The competing CPU Notebook that caused the block was external to CMI-Flu. The admission guard therefore behaved as designed; no scientific or bridge-runtime repair is justified.

## 002 contract

002 changes only request lineage and Kaggle target identity. The science commit `0815f4517345e127dbcbcae0f380a54f3a3d15bd`, exact science Git blobs, C0-C4 definitions, fixed 0.75/0.25 rank fusion, TabPFN 8.5.0/checkpoint identity, source-attachment normalization, promotion gates, 28-subject sensitivity, CPU-only resource class, maximum 21 TabPFN fits, output allowlist, and no-Public/no-submission boundary are unchanged.

New target: `renta0426/cmi-flu-e11c-robust-tabpfn-20260910-002` version 1.

Execution remains one-shot and approval-gated. Before the only SaveKernel call, the executor must re-read live Competition rules, verify model access, prove the fresh target absent, and require the CPU class to have no active or unclassifiable run under `max_active_runs=1`. If admission is closed, it fails before write again; it does not wait, retry, or launch compute automatically.
