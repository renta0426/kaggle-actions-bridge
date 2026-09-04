# CMI-Flu HAI transfer request 003 failure

Date: 2026-09-04

## Classification

- request: `20260904-cmi-flu-hai-transfer-003`
- workflow run: `33876159784`
- Kaggle target: `renta0426/cmi-flu-phase-a-hai-strain-transfer-20260904-003`
- write/resource status: resource-consumed; one private CPU Notebook version was created and started
- Competition submission: none
- automatic compute retry: none

## Runtime failure

The protected push succeeded and the Kaggle CPU run reached the HAI experiment. It failed in `build_sequence_lookup` with bounded error code `74bf25be40ca17429601` because the HAI science source expects canonical columns `virus_strain`, `sequence`, and `sequence_status`, while the locked organizer `strain_sequences.csv` does not use those literal headers.

The science source, B2.1 runtime adapter, and HAI/B2.1 compatibility shim remained the approved versions. No model-family, CV, stress-test, promotion, or leaderboard-selection change caused the failure.

## Secret-free locked-reference diagnosis

A separate GitHub-hosted, credential-free diagnostic (`33880982921`) downloaded only the public organizer file already locked by SHA-256 and inspected schema metadata without logging sequence contents.

The exact file identity remained:

- bytes: `206797`
- SHA-256: `63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887`
- data rows: `352`
- columns per row: `3`
- empty required cells: none

The exact raw header is:

`Virus, Sequence, Status_of_sequence`

Aggregate column diagnostics showed all 352 `Virus` and `Sequence` entries non-empty; `Virus` was unique for all 352 rows. Sequence contents were not emitted.

## Repair 004

Repair `20260904-cmi-flu-hai-transfer-004` keeps the science source byte-identical and changes only the locked-reference transport boundary. Before the unchanged science entry point it requires the exact raw header above and renames:

- `Virus` -> `virus_strain`
- `Sequence` -> `sequence`
- `Status_of_sequence` -> `sequence_status`

The raw file checksum, row count, and row width are revalidated before protected execution. The normalized and raw column names are included in aggregate provenance. A fresh Environment approval is required before any new Kaggle CPU run.
