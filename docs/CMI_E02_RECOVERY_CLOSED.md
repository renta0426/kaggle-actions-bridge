# E02 recovery closed — 2026-09-07

## Verified outcome

Approved recovery run [34085988392](https://github.com/renta0426/kaggle-actions-bridge/actions/runs/34085988392), job `101630101632`, completed successfully at approximately 05:30:45 UTC / 14:30:45 JST. Cleanup succeeded. This was a read-only operation: no new Notebook push, compute run, or submission.

The exact existing version is 1. Its code hash matched `ad36802efe2af9d4678b5d216b4afbb904a74746ec971818e75fdb2692709f37`. The original Notebook remains `ERROR`; the recovered log contains precisely the anticipated `write_outputs` / `KeyError` / `abaf4a59d478c511eb39` signature. Original metrics and summary passed validation. Original `bridge-result.json` was absent and was not invented.

Original output hashes from the verified recovery receipt:

- metrics: `99d0869567ff301995556160b8c9efbc1045b3624f37f0261bf6859efe783b52`
- summary: `938301fe316bfb26e47025cb01f353f10bf72b290910596e007506f16e6d7b31`

Scientific disposition: the fixed structured Ridge is not promoted. Scientific values are recorded by the authorized agent in the science repository; bridge CI does not fetch or write that private repository. The typed recovery projection is sufficient for the primary paired decision; it is not a copy of the original full metrics file. There is no justification for rerunning unchanged training merely to change the original Notebook's status badge.

## The independently approved obsolete run

Old Public-probe run `34085988457` was also approved. Its protected job `101630035484` failed at the existing B2.1 status/read step with HTTP 403, before the new-Notebook push step. All push, wait, output and submission-related follow-on steps were skipped. This run did not create a new Public-probe Notebook. Its old-kernel access problem is not bypassed or retried. Run `34085988483` was already cancelled.

A workflow-file change does not retroactively cancel a previously created run. The old run is now terminal, so it no longer blocks E02's global lock. This is an operational side incident, not an E02 scientific result.

## Retire consumed requests, not just compute launches

Requests 163, 164 and 174 were already retired. Request 175 is now also retired after its one successful read. Shared identity/output helper changes must not create another authenticated recovery request. The retirement workflow has only a manual refusal stub, no protected environment, no secrets, no compute lock, no network/read/write action. The structural regression in workflow 176 now covers all four retired paths.

Do not mutate historical request 175 or rerun historical Actions job 101630101632. A genuinely new need requires a new reviewed manifest and a fresh environment approval. All earlier run, source, code and output hashes remain preserved in history.

## Operating baseline for the next experiment

Use [the incident/runbook](CMI_E02_INCIDENT_AND_RUNBOOK.md) and [the trigger audit addendum](CMI_E02_TRIGGER_AUDIT_ADDENDUM.md). The verified short path is: agent-mediated exact reviewed source blobs; local/secret-free complete scientific-entry and output-lifecycle tests; separate audit of PR and main-push trigger effects; one protected run; exact current-version and source checks; a typed aggregate record; explicit request retirement after closeout.

No private research repository clone/API/raw read is permitted from CI. No PAT, SSH key, or expanded GitHub token is introduced. A model-quality negative result does not authorize threshold changes, an automatic rerun, or Public-LB tuning. The tests cover observed recurrence classes, not every future Kaggle service or image change.
