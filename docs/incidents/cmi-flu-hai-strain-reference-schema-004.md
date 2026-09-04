# CMI-Flu HAI strain-reference schema diagnostic

Date: 2026-09-04

- diagnostic workflow run: `33880982921`
- environment: GitHub-hosted Ubuntu runner, no Kaggle credentials, no protected Environment, no Kaggle compute
- source: public organizer Google Drive folder already approved for the HAI experiment
- file: `reference_files/strain_sequences.csv`
- bytes: `206797`
- SHA-256: `63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887`
- data rows: `352`
- columns: `3`
- row width: exactly `3`
- exact header: `Virus, Sequence, Status_of_sequence`
- required-field empties: none

Only schema metadata, counts, length ranges, and checksums were emitted. No sequence values were logged or committed. This diagnostic established the parser boundary needed by HAI repair 004 after request 003 failed on canonical-column assumptions.
