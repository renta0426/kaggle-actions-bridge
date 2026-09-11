# Operational lessons

This file records failures already observed in bridge runs and converts them into durable operating rules.

## 2026-09-11 policy supersession: remote capacity belongs to Kaggle

Earlier bridge versions treated CPU/GPU/TPU active-session classification, remaining quota, and locally chosen concurrency thresholds as pre-write admission requirements. That policy is now superseded by [`EXECUTION_POLICY_V2.md`](EXECUTION_POLICY_V2.md).

Poisoned Chalice P1-03 Actions run `34554141826` stopped before SaveKernel with `GPU admission unknown=3`. The approved target remained version 1 in `ERROR`; no version-2 write and no new GPU compute occurred. At least one overlapping bridge-launched Kaggle operation was independently known to be CPU-only. The bridge had converted incomplete metadata about unrelated active work into a false launch blocker.

New rule: Kaggle is authoritative for remote CPU/GPU/TPU capacity, account quota, and concurrent Notebook availability. New or modified launch workflows must not block on `quota_view()` thresholds, active-kernel counting, unrelated accelerator classification, bridge-local `max_active_runs`, bridge-local minimum-quota thresholds, or unknown metadata for unrelated sessions.

If Kaggle has no slot/quota, let the one approved operation reach Kaggle and classify Kaggle's response. If read-only reconciliation proves no new side effect, use `platform_rejected_no_side_effect`; do not create a scientific repair merely to predict platform capacity better.

The bounded active-metadata resolution successor created immediately after run `34554141826` remains useful historical evidence, but its admission design is not the template for future workflows.

## Static validation must finish before protected execution

A recurring source of wasted approvals was deterministic materialization, request-contract, title/slug, dependency, callable/signature, or schema validation happening too late. Other failures were caused by the same invariant being hand-coded in multiple places and drifting between request, PR validation, and protected launch.

New rule: use a two-stage design.

1. Credential-free PR CI must parse requests, compile payloads, deterministically materialize the exact Notebook, verify hashes/title/slug/source provenance, test frozen callable/schema boundaries where practical, and reject policy-v2 capacity gates.
2. The protected job should recheck only facts that can change after merge or that require the Kaggle credential: event/actor/workflow identity, exact approved payload hash, current exact target/input version/status where required, then the single approved Kaggle call.

Do not copy large blocks of static assertions into both stages. A deterministic defect should fail before merge, not after Environment approval.

The policy checker is `scripts/kaggle_launch_policy_v2.py`.

## Private research inputs

Two SmolLM2 launch attempts failed before any Kaggle write because the public bridge expected material from a private research repository that was not available in the protected execution context.

Rule: protected Kaggle execution must be self-contained from the approved bridge commit plus pinned public/Kaggle inputs. Prefer deterministic reconstruction from public sources. Do not solve this class of failure by broadening the bridge's GitHub access.

This reconstruction must itself be exercised in credential-free PR CI so that a missing source does not first appear after Environment approval.

## Accelerator admission (historical rule, superseded)

A SmolLM2 request was previously deferred because a GPU run was already active. The old rule classified active sessions as CPU/GPU/TPU, applied `max_active_runs`, treated unknown classification as blocking, and rechecked immediately before write.

**Superseded on 2026-09-11.** Do not use this as the current launch template. See the policy-supersession section above and [`EXECUTION_POLICY_V2.md`](EXECUTION_POLICY_V2.md).

Resource fidelity remains relevant: if the approved scientific experiment requires CUDA/GPU, the Notebook metadata must still request the approved accelerator. What is removed is bridge-side prediction of whether Kaggle currently has room to run it.

## Historical versus current Notebook output

Kaggle SDK `ListKernelSessionOutput` with explicit version `1` returned HTTP 404 for a completed private Notebook even though its current output existed. Earlier attempts to retrieve historical `scriptVersionId` output were also not reliable.

Rule: historical-version output is not a production capability. For current-version reads, first require `current_version_number == expected_version`. If it differs, stop rather than silently substituting latest output.

## Broad CLI output downloads

The successful SmolLM2 readout used official `kaggle kernels output` after proving current version == 1. It retrieved the entire saved working directory, not only the desired final files, and normal stdout enumerated downloaded paths.

Rule for Notebook authors: treat `/kaggle/working` as an export surface. Put source clones, scratch data, temporary caches, and intermediate files under `/tmp`; leave only declared final outputs in `/kaggle/working` when the Notebook succeeds.

Rule for bridge readouts: capture rather than stream output command stdout/stderr, enforce output allowlists and byte limits, reject unexpected files for new workflows, copy only declared outputs to the evaluator, and delete the temporary download tree unconditionally. Never retain broad private output as an Actions artifact/cache.

## Prediction/label boundary

The successful SmolLM2 pattern scored a deterministic cohort with no target membership labels in the GPU Notebook, then joined labels only in a separate evaluation step.

Rule: when a transfer experiment is declared frozen, keep target labels and prior-model scores physically outside the prediction boundary. Freeze fitting, normalization, selection, and fusion before label join. Negative results do not authorize target-label retuning.

## Failure repair and re-execution

The old blanket practice was: every pre-write failure gets a new repair request, PR, merge, and approval. That was too coarse because platform-capacity rejection and bridge defects are different classes.

Current rules:

- `static_validation_failure`: fix in PR CI; no Kaggle execution was attempted.
- `prewrite_identity_or_authorization_failure`: investigate the target/input/authorization mismatch and change only the affected contract.
- `platform_rejected_no_side_effect`: if Kaggle rejects due capacity/quota/temporary availability and exact reconciliation proves no side effect, the same immutable request may be manually run later with fresh approval. No repair PR/new slug is required.
- `ambiguous_write`: reconcile exact target/submission/resource state before any later write.
- `resource_consumed_runtime_failure`: diagnose the Notebook/runtime/science failure before another compute run.
- `readout_failure`: repair readout without rerunning compute.

Automatic compute retries remain disabled. The relaxation is specifically that a confirmed no-side-effect platform rejection is not mislabeled as an implementation defect.

## Frozen runtime API surface

CMI-Flu HAI transfer request `20260904-cmi-flu-hai-transfer-001` consumed one CPU Notebook run and failed inside the scientific entry point because the HAI extension imported `run_hai_compact_for_panels` from the frozen package's `cmi_flu.evaluation` namespace and passed `selection_policy=...`, while the frozen B2 API did not expose that keyword. The B2.1 robust implementation existed only as a runtime-adapter replacement in the `cmi_flu.runner` namespace.

Rule: an extension layered on a frozen runtime must validate the exact callable namespace and signature it will import at execution time, not merely a current-source API or compile-only contract. If approved behavior is supplied by a runtime adapter, expose it through an explicit provenance-locked compatibility shim. Never silently fall back from an approved robust selector to an older selector.

The same HAI extension expected per-fold panel-proxy aggregates absent from the older frozen result object. A compatibility shim may reconstruct only deterministic aggregate fields from the frozen enriched OOF result using pre-registered split/panel definitions; underlying fitting, labels, panel membership, and selection rules remain unchanged.

Under policy v2, these compatibility checks belong in credential-free PR CI whenever practical.

## Polling responsiveness

A HAI request failed in Kaggle after roughly two minutes, but its Actions monitor slept for 900 seconds after an immediate queued-state read. The next read detected the terminal error much later.

Rule: long-running watches should use a small bounded startup schedule for early queue/startup failures, then back off. Total deadline and API-call budget remain bounded; adaptive polling must not become keep-alive or automatic retry.

Capacity prediction is not a reason to poll active sessions before launch.

## Kernel metadata identity and push diagnostics

CMI-Flu HAI transfer request `20260904-cmi-flu-hai-transfer-002` failed in `kaggle kernels push` before the status monitor. Its metadata id slug and title-derived slug differed (`-repair-` was present only in the title-derived slug). The CLI returned non-zero almost immediately, and raw diagnostics were deleted before a bounded category/hash was emitted.

Rule: title/slug/target consistency is deterministic and must be proven in credential-free PR CI before approval. A title/slug mismatch should never first appear in the protected job.

Rule: a failed write must not dump raw CLI stdout/stderr into public logs, but it must retain enough sanitized diagnosis: allowlisted error category, return code, byte counts, SHA-256 digests, then unconditional deletion of raw files.

After any failed write call, reconcile target state. If no new side effect is proven and the rejection is platform capacity/availability, use `platform_rejected_no_side_effect`; if outcome is unknown, use `ambiguous_write`.

## Locked external-reference schema

CMI-Flu HAI transfer request `20260904-cmi-flu-hai-transfer-003` consumed a CPU Notebook run and failed only after reaching the science entry point. The locked organizer `strain_sequences.csv` had the expected checksum/size/row count but literal header `Virus, Sequence, Status_of_sequence`; the extension expected `virus_strain, sequence, sequence_status`.

Rule: for small external references that influence a resource-consuming experiment, credential-free preflight must validate the exact structural contract the parser depends on: header/order where meaningful, row count or bounded shape, required-field non-emptiness, delimiter/width assumptions, and explicit raw-to-canonical mappings. Checksum identity alone is insufficient parser validation.

Schema diagnostics for sequence or other potentially large biological fields should publish only structural metadata such as column names, counts, length ranges, and checksums, not sequence values or participant-level content.

## Accelerator fidelity

Accelerator **selection** remains part of the scientific contract even though accelerator **availability** is no longer a bridge admission decision.

A CUDA logits/rank experiment must not be moved to TPU merely to save GPU quota unless numerical equivalence has been established separately. CPU-only aggregation/readout work should not request an unnecessary accelerator.

If the approved request asks for GPU and Kaggle currently has no GPU slot, let Kaggle reject/queue according to its own platform behavior rather than changing the experiment or inventing a bridge quota threshold.

## Live Competition page parsing

Some workflows reparsed mutable Competition page text immediately before SaveKernel even when that text could not change the already-approved scientific Notebook operation. This creates another transient pre-Kaggle failure surface.

Rule: review Rules/Code Requirements during request authoring and record the relevant contract. A live page check may be blocking only when its current value is necessary to prevent an unauthorized submission or explicit rule violation. Generic Notebook launches should not fail because non-operative page text is temporarily unavailable or reformatted.

## Public log discipline

Public logs should contain only bounded operational metadata such as request IDs, counts, status classes, hashes, versions, byte counts, resource selection, and success/failure markers. Capture verbose CLI output instead of dumping private file lists/content. Hash private operational identities when diagnosis requires correlation.

## Checklist for new/modified workflows

- [ ] Read `EXECUTION_POLICY_V2.md` and this file first.
- [ ] New resource-write request declares `execution_policy: kaggle_native_capacity_v2`.
- [ ] No bridge-local active-session/quota/concurrency launch gate exists.
- [ ] Credential-free PR CI materializes and validates the exact payload/Notebook.
- [ ] Protected job is limited to live identity/authorization/input checks plus the one approved operation.
- [ ] Exact target/version semantics are explicit where needed.
- [ ] New-kernel title/slug/target id are proven before protected execution.
- [ ] Frozen extension call sites are tested against the actual runtime namespace/signature.
- [ ] Locked external references are tested for checksum and parser schema where relevant.
- [ ] There is at most one write/run-start call and no automatic compute retry.
- [ ] Failed write calls are followed by exact read-only reconciliation.
- [ ] `platform_rejected_no_side_effect` does not generate an unnecessary repair PR.
- [ ] Current-versus-historical output semantics are explicit.
- [ ] Polling is bounded and is not used to predict capacity before launch.
- [ ] Transient Notebook material stays outside `/kaggle/working`.
- [ ] Final outputs are allowlisted/size-bounded where read back through the bridge.
- [ ] Verbose diagnostics are sanitized rather than streamed or discarded.
- [ ] Cleanup is unconditional.
