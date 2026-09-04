# CMI-Flu rank-transfer request 003

This request retries the Phase A Task1.1/Task1.2 rank-transfer diagnostic after request 002 failed before model execution.

The embedded runtime package is the audited B2 package, whose configuration loader predates the `b021_taskwise_robust` baseline identifier. Request 003 therefore keeps `baseline: b02_taskwise_compact` in the serialized config until the legacy loader returns successfully, verifies `selection.policy=robust_v1`, then promotes the in-memory config identity to `b021_taskwise_robust` before the exact relayed `rank_transfer.py` science code runs.

The B2.1 runtime adapter, model grids, split logic, relayed science blob, CPU-only resource contract, Internet-off setting, and no-submission policy are unchanged.

GitHub Actions does not access the private science repository. The exact science source needed for this run was relayed by the agent and is pinned by Git blob SHA.
