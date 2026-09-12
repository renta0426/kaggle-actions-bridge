# CMI-Flu Strategy v3 V3-07 execution contract

This bridge materializes the exact Science source at commit `9bfe05664e4d96f09631a3210f7951bf9dfefe71`, blob `4cf804e0c10889d0e037457ac096711b77200ea9`.

Before any Kaggle result is observed, V3-07 is split into exactly two independent one-shot requests: `task22_panel_mean` (H1) and `task23_retention` (H2). The split is operational, not a parameter search: H1 depends on E05 panel/reference logic, whereas H2 depends on paired D28/D365 longitudinal logic and a different nested teacher. A failure or scientific negative in one request must not be used to mutate the other request.

Both operations are CPU-only, expected version 1, automatic retry 0, internet disabled in Kaggle, and protected by `kaggle-readonry`. The bridge may create one private Notebook version per authorized request and then read only that exact current version's declared outputs. It may not submit to the Competition, select a Final Submission, use Public-LB results for selection, or reopen hyperparameter tuning.

Row-level OOF and Challenge banks are private Kaggle outputs. GitHub/Actions may emit only aggregate diagnostics: fit count, provenance, RMSE/Spearman summaries, study/panel strata, paired error diagnostics, rank/tie shifts, support counts, and the H1 aggregate correction-SD range. Participant identifiers and row predictions must never be printed or committed.

The credential-free validation workflow builds each generated runtime twice and requires byte identity, executes frozen helper science under the real numerical stack, validates the private-bank writer and manifest, exercises failure cleanup, runs the aggregate sanitizer, verifies the one-write executor path, and statically rejects submission paths. Protected execution workflows are added only after these runtime digests are frozen.
