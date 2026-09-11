# Kaggle execution policy v2

Effective: 2026-09-11

This policy replaces the bridge-local remote-session admission model for new or modified Kaggle execution workflows.

## Decision

Kaggle is the authority for Kaggle-side CPU/GPU/TPU capacity, account quota, concurrent Notebook availability, and transient platform availability.

The bridge MUST NOT independently refuse an otherwise approved Notebook launch because of locally inferred remote-capacity state. In particular, a launch MUST NOT be blocked by:

- `quota_view()` remaining-time thresholds;
- counting `RUNNING` / `QUEUED` / `PENDING` kernels;
- CPU/GPU/TPU classification of unrelated active kernels;
- `max_active_runs` limits invented by the bridge;
- `min_remaining_quota_hours` limits invented by the bridge;
- failure to classify an unrelated active kernel's accelerator;
- a bridge-side prediction that Kaggle will reject the requested accelerator.

If Kaggle has no free GPU/TPU/CPU slot or quota, the one approved Kaggle operation is allowed to reach Kaggle and Kaggle may reject it. That rejection is a platform-capacity outcome, not a reason to manufacture another scientific repair.

This change does **not** relax Competition Rules, Kaggle Terms/AUP, explicit user constraints, security boundaries, exact target identity, or the one-write/no-automatic-compute-retry contract.

## Resource fields under policy v2

For policy-v2 requests, `resource` describes the requested execution environment only. Typical fields are:

```json
{
  "execution_policy": "kaggle_native_capacity_v2",
  "resource": {
    "accelerator": "gpu",
    "machine_shape": "NvidiaTeslaT4",
    "expected_runtime_minutes": 100,
    "hard_timeout_minutes": 180
  },
  "automatic_compute_retries": 0
}
```

The following bridge-local capacity fields are deprecated and forbidden in new policy-v2 requests:

- `resource.max_active_runs`
- `resource.min_remaining_quota_hours`
- `resource.min_remaining_cpu_quota_hours`
- `resource.min_remaining_gpu_quota_hours`
- `resource.min_remaining_tpu_quota_hours`

Legacy requests are historical records and do not need to be rewritten. When a legacy launch workflow/request is materially modified, migrate its launch behavior to this policy unless the user or an official Competition rule explicitly requires a stricter constraint.

## What may still block before a write

A protected job should block before the write only when the condition is necessary to prevent the wrong or unauthorized operation, or when the approved scientific input would be wrong.

Allowed blocking checks include:

1. repository / actor / event / workflow / Environment identity;
2. Kaggle credential presence and expected credential form;
3. exact approved target/ref/title/slug identity;
4. exact current target version/state when version semantics are part of the approved operation;
5. exact required input Dataset/Notebook/Model version and terminal status when that identity is part of the scientific contract;
6. privacy, accelerator selection, Internet setting, and other metadata that define the requested Notebook itself;
7. explicit Competition submission authorization and official Competition constraints that the bridge must enforce to avoid an unauthorized action;
8. immutable source/blob/hash checks that prove the approved payload is the payload being sent.

The following are observation-only by default and MUST NOT become launch gates merely because they are available from an API:

- remaining account CPU/GPU/TPU quota;
- unrelated active sessions;
- unrelated active-kernel accelerator metadata;
- estimated queue length;
- current platform capacity;
- best-effort status of unrelated work.

If observation-only metadata is queried, failure to retrieve it must not stop the approved launch. Prefer not querying it at all when it does not change the authorized operation.

## Two-stage validation model

Repeated failures after Environment approval often came from checks that could have been completed before any secret was exposed. Policy v2 therefore separates static correctness from runtime authorization.

### Stage A: credential-free PR validation

Complete as much deterministic validation as possible before merge:

- parse request JSON;
- compile Python payloads;
- materialize the exact Notebook deterministically;
- verify Notebook SHA-256 and source/blob provenance;
- verify title/slug/ref consistency;
- validate required static input names and expected versions;
- validate frozen formula/config contracts;
- exercise exact callable/signature/schema boundaries using synthetic fixtures where practical;
- validate dependency locks and serializer/tool versions;
- validate that no bridge-local capacity gate was introduced.

This is where implementation defects should fail. A failure here costs no Kaggle write, no Kaggle compute, and no Environment approval.

### Stage B: protected execution

After Stage A is green, the protected job should be intentionally small. Recheck only facts that can change between PR validation and launch or facts that must be proven under the credential:

- exact event/repository/actor/workflow identity;
- exact approved payload hash;
- credential availability;
- exact current target version/state if relevant;
- exact current required Kaggle input versions/status if relevant;
- then perform the single approved Kaggle write/run-start call.

Do not duplicate large blocks of static assertions in both Stage A and Stage B. Do not re-validate unrelated live platform state merely because an API exposes it.

## Live Competition pages

Competition Rules and official requirements must be reviewed when authoring the request. However, a Notebook launch should not normally depend on reparsing mutable HTML/page text immediately before SaveKernel if the page contents do not change the authorized operation.

A live page/API check may remain blocking only when the current official value is required to prevent an unauthorized submission or rule violation. A generic scientific Notebook run should not fail because a page is temporarily unavailable or because presentation text changed.

## One-shot write and reconciliation

The bridge still performs at most one resource-starting/write call for one approved request. Automatic compute retries remain disabled.

After the call:

- if the expected target version/resource is observed, treat the write as observed even if the client returned an ambiguous transport error;
- if Kaggle explicitly rejects the call and read-only reconciliation proves no new version/resource was created, classify it as `platform_rejected_no_side_effect`;
- if the write outcome cannot be proven, classify it as `ambiguous_write` and reconcile before any later write;
- if a new version/run exists, never send the same write again merely because the client returned an error.

## Retry policy after a capacity/platform rejection

A confirmed `platform_rejected_no_side_effect` caused by Kaggle capacity/quota/temporary availability does **not** require a code change, repair PR, new slug, or modified scientific request.

The same immutable request may be run again later with a fresh Environment approval, provided read-only reconciliation has proved that the prior attempt created no new version/resource and consumed no compute.

This is a manual re-execution of the same approved operation, not an automatic retry.

By contrast:

- bridge implementation defects require a code repair;
- scientific Notebook runtime failures after compute starts require diagnosis before another compute run;
- ambiguous writes require exact reconciliation first.

## Standard failure classes

Use one of these classes in new operational records:

- `static_validation_failure`: deterministic defect found before protected execution;
- `prewrite_identity_or_authorization_failure`: target/input/authorization no longer matches the approved operation;
- `platform_rejected_no_side_effect`: Kaggle rejected the one call and no side effect was created;
- `ambiguous_write`: the call outcome cannot yet be proven;
- `resource_consumed_runtime_failure`: Kaggle created/ran the Notebook but execution failed;
- `readout_failure`: compute result exists but output/status retrieval failed.

Do not classify bridge-local quota/session heuristics as a prewrite failure under policy v2 because those heuristics should not exist.

## Workflow-authoring anti-patterns

New or modified launch workflows should not contain hard-gating logic based on patterns such as:

- `api.quota_view()` followed by a minimum remaining-time check;
- `api.kernels_list(...)` followed by active-session counting for launch admission;
- errors such as `GPU concurrency refused`, `GPU admission unknown`, `CPU admission`, `TPU admission`;
- request fields named `max_active_runs` or `min_remaining_quota_hours`;
- repeated static Notebook/science assertions in the protected job that already ran in credential-free PR validation.

The repository linter in `scripts/kaggle_launch_policy_v2.py` checks the capacity-gate portion of this policy for newly added policy-v2 requests and changed resource-launch workflows.

## Incident that motivated the change

Poisoned Chalice P1-03 run `34554141826` reached the protected preflight but stopped before SaveKernel with `GPU admission unknown=3`. Version 2 was not written and no new compute started. At least one overlapping Kaggle run was independently known to be CPU-only, demonstrating that unrelated active-session classification was an unnecessary source of false refusal.

The earlier bounded metadata-resolution successor was a correct repair under the old policy, but this policy supersedes that design for future work: unrelated session classification is no longer a launch prerequisite at all.

## Security invariants retained

The following remain unchanged:

- protected `kaggle-readonry` Environment for authenticated operations;
- GitHub-hosted runner only;
- `permissions: {}` by default;
- no arbitrary shell/URL/package input from requests;
- no credential/private content in public logs/artifacts/cache;
- exact target/input identity where scientifically or operationally required;
- one approved write/run-start call per request execution;
- no automatic compute retry;
- reconcile ambiguous writes before any later write;
- Competition Rules, Kaggle Terms/AUP, and explicit user constraints remain authoritative.
