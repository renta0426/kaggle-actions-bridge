# Bridge incident reports

This directory contains bounded, non-sensitive records of resource-consuming or operationally significant bridge failures and their approved repair class.

- `cmi-flu-hai-transfer-001.md`: CMI-Flu Phase A HAI transfer request 001 frozen-runtime API mismatch and delayed terminal-status detection.
- `cmi-flu-hai-transfer-001-monitoring.md`: focused note explaining why the bounded 900-second first sleep looked like a stuck workflow and how repair 002 preserves a finite watch while detecting short failures sooner.
- `cmi-flu-hai-transfer-002.md`: request 002 pre-execution kernel-push failure caused by inconsistent title/slug metadata, plus repair 003's local identity check and bounded push-failure diagnostics.
- `cmi-flu-hai-transfer-003.md`: request 003 resource-consuming HAI failure caused by the locked organizer strain-sequence CSV using raw headers `Virus`, `Sequence`, and `Status_of_sequence` rather than the science layer's canonical internal names.
- `cmi-flu-hai-strain-reference-schema-004.md`: credential-free structural diagnosis of the locked organizer strain reference used to define repair 004 without exposing sequence values.
- `cmi-flu-public-probes-002.md`: controlled Public-probe repair 002 pre-write failure because the current Kaggle token could not read the historical private B2.1 Notebook.
- `cmi-flu-public-probes-003.md`: request 003 resource-consuming failure caused by a bridge-side-only `sha256_bytes` helper leaking into the generated Kaggle runtime before frozen B2.1 regeneration began.
- `cmi-flu-public-probes-004.md`: request 004 resource-consuming failure caused by stale bridge provenance assertions for B2.1 selected models; repair 005 locks the authoritative seven-task model map and exact original B2.1 submission hash.
- `cmi-flu-public-probes-005.md`: request 005 resource-consuming failure because byte-for-byte equality with the historical B2.1 submission was used as a cross-worker identity gate; repair 006 exports the regenerated B2.1 as an explicit control alongside the unchanged three-probe family.
