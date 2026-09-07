# CMI-Flu E05 007 watcher-lifetime incident

Date: 2026-09-08

E05 007 exposed an orchestration-state bug rather than a scientific failure. The one-shot Kaggle write returned code 0 and exact target/version metadata was confirmed. The remote CPU kernel continued to run, but the inherited executor watched for only 60 minutes (`120 seconds × 30 polls`) and then raised `polling bound exceeded`. The user later supplied the completed version-1 outputs, which passed scientific/provenance validation.

The prior error classification was therefore too coarse: **watcher expiry while the exact remote version is still RUNNING/QUEUED/PENDING is not a remote scientific failure**. It also cannot authorize a replay, because a write has already occurred and the remote resource may still complete.

E06a prevention contract:

- hard requested runtime: 120 minutes;
- watcher horizon: 130 minutes (`120 seconds × 65 polls`);
- protected Actions job timeout: 150 minutes;
- write retries remain zero;
- exact metadata/version remains authoritative after write;
- explicit remote `ERROR`/`CANCEL`/`FAIL` is a remote failure;
- watcher expiry with unresolved active remote state requires current-version read-only recovery, never another write;
- `scripts/test_cmi_e06a_watcher_contract.py` locks the horizon and recovery semantics in secret-free CI.

This incident should be treated as a reusable bridge lesson for future long-running CPU/GPU/TPU notebooks: request runtime, watcher lifetime, CI job timeout, remote terminal state, and write state are separate parts of the state machine and must not be collapsed into one success/failure flag.
