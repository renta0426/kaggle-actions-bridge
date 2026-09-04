# CMI-Flu HAI transfer request 001 failure

Date: 2026-09-04

## Classification

- request: `20260904-cmi-flu-hai-transfer-001`
- workflow run: `33871307603`
- Kaggle target: `renta0426/cmi-flu-phase-a-hai-strain-transfer-20260904-001`
- write/resource status: resource-consumed; one private CPU Notebook version was created and started
- Competition submission: none
- automatic compute retry: none

## Scientific runtime failure

The Kaggle run reached the HAI experiment and failed after about 104 seconds with:

`TypeError: run_hai_compact_for_panels() got an unexpected keyword argument 'selection_policy'`

The HAI extension was pinned to science commit `d1ebf13dc0dc7e5d5a2798b29c288265cbf56618`, blob `b671d8bf7f10bebbd65aca2a5bad42e267ee78d5`. The frozen B2 package exposes an older `cmi_flu.evaluation.run_hai_compact_for_panels` signature without `selection_policy`. B2.1 robust HAI selection is supplied later by the frozen runtime adapter in the `cmi_flu.runner` namespace. The extension imported the evaluation namespace directly, so the robust adapter was not visible at that call site.

Repair 002 therefore preserves the science source byte-for-byte and adds a provenance-locked runtime compatibility shim that exposes the already-approved B2.1 robust HAI implementation through the evaluation namespace. The shim rejects any selection policy other than `robust_v1`. It also reconstructs the deterministic panel-proxy fold aggregates expected by the new stress-test code from the frozen enriched OOF object.

## Monitoring delay

The workflow monitor made an immediate status request while the Kaggle run was queued, then slept for 900 seconds. Kaggle failed within roughly two minutes, but GitHub Actions did not observe the terminal error until the next poll about fifteen minutes later. The monitor was finite, not an infinite loop.

Repair 002 uses short bounded startup polling (`30`, `60`, `120`, `300` seconds) and then backs off to `900` seconds, while retaining a hard 210-minute deadline and a maximum of 24 status calls.
