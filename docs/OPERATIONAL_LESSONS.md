# Operational lessons

This file records failures already observed in bridge runs and converts them into durable operating rules.

## Private research inputs

Two SmolLM2 launch attempts failed before any Kaggle write because the public bridge expected material from a private research repository that was not available in the protected execution context.

Rule: protected Kaggle execution must be self-contained from the approved bridge commit plus pinned public/Kaggle inputs. Prefer deterministic reconstruction from public sources. Do not solve this class of failure by broadening the bridge's GitHub access.

## Accelerator admission

A later SmolLM2 request was correctly deferred because a GPU run was already active.

Rule: classify active sessions as CPU/GPU/TPU from exact metadata, apply `max_active_runs` within the requested class, treat unknown classification as blocking, and check again immediately before the one allowed write. For diagnosis, log counts and hashed identities rather than private refs. A defer is pre-write and requires a fresh approved run; do not auto-poll or auto-retry.

## Historical versus current Notebook output

Kaggle SDK `ListKernelSessionOutput` with explicit version `1` returned HTTP 404 for a completed private Notebook even though its current output existed. Earlier attempts to retrieve historical `scriptVersionId` output were also not reliable.

Rule: historical-version output is not a production capability. For current-version reads, first require `current_version_number == expected_version`. If it differs, stop rather than silently substituting latest output.

## Broad CLI output downloads

The successful SmolLM2 readout used official `kaggle kernels output` after proving current version == 1. It retrieved the entire saved working directory, not only the two desired final files, and its normal stdout enumerated downloaded paths.

Rule for Notebook authors: treat `/kaggle/working` as an export surface. Put source clones, scratch data, temporary caches, and intermediate files under `/tmp`; leave only declared final outputs in `/kaggle/working` when the Notebook succeeds.

Rule for bridge readouts: the CLI output command is a bounded current-version fallback. Invoke it once, capture rather than stream its stdout/stderr, enforce output allowlists and byte limits, reject unexpected saved files for new workflows, copy only declared outputs to the evaluator, and delete the full temporary download tree unconditionally. Never retain the broad download as an Actions artifact/cache.

## Prediction/label boundary

The successful SmolLM2 pattern scored a deterministic cohort with no target membership labels in the GPU Notebook, then joined labels only in a separate evaluation step.

Rule: when a transfer experiment is declared frozen, keep target labels and prior-model scores physically outside the prediction boundary. Freeze fitting, normalization, selection, and fusion before label join. Negative results do not authorize target-label retuning.

## Failure repair

Distinct SmolLM2 failures had distinct causes: unavailable private-repo materialization, GPU admission defer, and output-read API 404.

Rule: after a failure, identify the exact step and whether a write/resource consumption occurred; classify it as pre-write, ambiguous-write, resource-consumed, or read-only; record the prior run; change only the mechanism required by the established cause; validate without credentials first; then require a fresh approval. Never stack speculative repairs or blindly rerun an ambiguous write.

## Accelerator fidelity

Accelerator choice is part of the scientific contract. A CUDA logits/rank experiment must not be moved to TPU merely to save GPU quota unless numerical equivalence has been established separately. CPU-only aggregation/readout work must not reserve an accelerator.

## Public log discipline

Public logs should contain only bounded operational metadata such as request IDs, counts, status classes, hashes, versions, byte counts, resource classes, and success/failure markers. Capture verbose CLI output instead of dumping file lists or private content. Hash sensitive operational identities when useful for diagnosis.

## Checklist for new workflows

- [ ] Preflight and reconstruction can run before protected execution.
- [ ] Exact target and version semantics are explicit.
- [ ] Compute resource class is fixed and admission is checked immediately before write.
- [ ] There is at most one bounded write call and no automatic compute retry.
- [ ] Current-versus-historical output semantics are explicit.
- [ ] Transient Notebook material stays outside `/kaggle/working`.
- [ ] Final outputs are allowlisted and size-bounded.
- [ ] Verbose CLI/API diagnostics are captured and sanitized.
- [ ] Cleanup is unconditional.
- [ ] Failures emit enough non-sensitive metadata to avoid another blind probe.
