# CMI-Flu HAI transfer request 002 failure

Date: 2026-09-04

## Classification

- request: `20260904-cmi-flu-hai-transfer-002`
- workflow run: `33874545675`
- intended Kaggle target: `renta0426/cmi-flu-phase-a-hai-strain-transfer-20260904-002`
- failed step: `Push exactly one private CPU repair run`
- write/resource status: `ambiguous-write`; the CLI write call was attempted and returned non-zero in about 0.55 seconds, the status-monitor step never ran, and there is no workflow evidence that a CPU session started
- Competition submission: none
- automatic compute retry: none

## Static root-cause evidence

The approved metadata had an inconsistent title and id:

- id/target slug: `cmi-flu-phase-a-hai-strain-transfer-20260904-002`
- title: `CMI Flu Phase A HAI Strain Transfer Repair 20260904 002`
- title-derived slug: `cmi-flu-phase-a-hai-strain-transfer-repair-20260904-002`

Kaggle's documented kernel metadata contract links the title and slug: for a new kernel, the slug is the lowercase title with separators normalized to hyphens. The request-002 metadata therefore violated that contract before the API call. The immediate push failure is consistent with metadata rejection. Because the workflow deleted captured stdout/stderr before emitting a bounded diagnostic, the exact server error text was not retained; this incident records the title/slug mismatch as the established static defect rather than claiming an unavailable raw API message.

## Repair 003

Repair 003 preserves the request-002 scientific runtime byte-for-byte and changes only execution identity/diagnostics:

- new target: `renta0426/cmi-flu-phase-a-hai-strain-transfer-20260904-003`
- title: `CMI Flu Phase A HAI Strain Transfer 20260904 003`
- a local pre-write slug derivation must match `TARGET_SLUG` and `TARGET_KERNEL`
- on push failure, raw stdout/stderr remain private and are deleted, but the public log records only an allowlisted failure category, return code, byte counts, and SHA-256 digests
- the responsive bounded status polling from repair 002 remains unchanged
- no scientific source, B2.1 compatibility shim, organizer reference, model-selection rule, stress test, promotion rule, accelerator class, or submission behavior changes
