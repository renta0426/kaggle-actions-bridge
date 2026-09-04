# Kaggle remote-run concurrency policy

## Effective bridge limit

For automated Kaggle batch Notebook launches, the bridge permits up to **2 active remote runs** at once unless a Competition rule, Kaggle live restriction, quota condition, or request manifest is stricter.

This is intentionally separate from GitHub Actions launch concurrency. Admission/write jobs remain serialized so two workflows cannot pass preflight and write to Kaggle simultaneously. After a launch job exits, the remote Kaggle run may continue while a second approved launch is admitted.

## Admission rule

For a request with `resource.max_active_runs: 2`:

- `active == 0`: launch may proceed.
- `active == 1`: launch may proceed.
- `active >= 2`: do not write; defer/refuse the launch.
- any active-status lookup that cannot be resolved: fail closed and do not write.

A request may declare a lower `max_active_runs`; it may not silently exceed the bridge/platform limit.

The active-run count is checked immediately before the single allowed `kaggle kernels push`. No automatic compute retry or polling-based relaunch is permitted.

## Why this exists

The earlier controlled-shadow GPU workflow used a zero-active admission gate: any `RUNNING`, `QUEUED`, or `PENDING` Notebook among the recent account runs blocked all new GPU work. That was unnecessarily strict for a two-run concurrency allowance and caused `shadow-gpu-t4x2-pilot-v1` to stop before any Kaggle write with `active=1`.

The repaired policy therefore distinguishes bridge-side write serialization from Kaggle-side remote concurrency and allows the second approved batch run to start.
