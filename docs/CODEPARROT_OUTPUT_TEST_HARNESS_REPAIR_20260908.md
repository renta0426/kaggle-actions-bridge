# Output-contract test harness repair — 2026-09-08

Distinct failure discovered during CodeParrot basename-repair PR #171:

- run `34177175055`, job `101908787496`, head `35501d103406d6b86c84bb45c4735e63704fc821`;
- failing step: `Exercise exact output-reader safety behavior`;
- secret-free job; GitHub CPU and synthetic temporary files only; Kaggle credentials/API calls/writes/compute/labels: 0; cleanup succeeded.

The downloaded job log proves that eight scenario errors came from `patch.object(reader, "exact_metadata", ...)`: the reader no longer exports that function after adopting `exact_metadata_eventually`. This fails before the scenario reaches its assertions. The current main test blob is `7705b4c2cfa024f38c7df6fa8f925da7d004b753`; the identity helper blob is `d4b9ab1199aa21977bbc14541a6b7456b5984331`. Both were reconstructed locally with Git blob checks; the same AttributeError was reproduced.

This is a test-harness API drift, separate from the leading-underscore production parser failure. Do not restore an unused production import, add `create=True`, skip the safety test, suppress errors, or weaken production identity validation merely to make CI green.

The repair removes only the obsolete reader-level mock. The still-existing `identity.exact_metadata` primitive remains mocked, so both real bounded-wrapper call paths and real version validation execute with synthetic metadata. New assertions require exactly two exact metadata reads after a successful CLI call and one when the CLI fails, while all previous assertions still require one captured download, no search, version-change rejection, unexpected/duplicate/symlink/missing/oversize rejection and cleanup. Safe-underscore/path rejection fixtures are also added to this shared suite.

Local result: all 7 unittest methods, including their scenarios, passed without SDK/network credentials. This is not the remote PR CI result; merge still requires the fresh head's secret-free checks to pass. The scientific evaluator and protected workflow are not changed by this test repair. Do not rerun the failed workflow attempt.
