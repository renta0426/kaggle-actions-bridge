# CodeParrot full-run pre-write failures — 2026-09-08

## Scope

This incident covers the two failed attempts to launch `renta0426/codeparrot-fresh-full-v1` from the sealed 200-row CodeParrot pilot. Neither failure reached `kaggle kernels push`; therefore no full-run Kaggle kernel version was created and no 4,000-row GPU compute was started.

## Failure 1 — stale source-version precondition

Actions run `34140663623` failed in the live admission step with:

`sealed pilot version changed: 3`

The full-run workflow had hard-coded pilot version 2, while the canonical successful pilot had already advanced to version 3 through PR #160, which repaired the missing same-forward mean-log-rank feature.

### Root cause

The downstream workflow pinned an intermediate implementation version instead of the final consumed source identity.

### Prevention

Downstream workflows that depend on a repaired Kaggle kernel must pin the final version and verify the repair's scientific markers, not only the numeric version. For this source those markers include:

- `CODEPARROT_PRIVATE_SNAPSHOT_COMPAT_V3`
- `CODEPARROT_MEAN_LOG_RANK_REPAIR_V2`
- `summary["mean_log_rank"] = float(np.log1p(rank_integer).mean())`
- first-8 same-forward fidelity marker
- audited cohort SHA

## Failure 2 — formatting-sensitive source transformer

Actions run `34143250952` passed live admission against exact pilot version 3 and successfully pulled/validated the sealed private source. It then failed before any Kaggle write with:

`RuntimeError: manifest selection: expected one replacement anchor, found 0`

The scientific manifest value was valid, but `scripts/patch_codeparrot_full_v1.py` required one exact multi-line source formatting for the `selection` field. Version 3 used a semantically equivalent source shape that did not match that literal formatting anchor.

### Root cause

A private-notebook transformation treated whitespace/quoting layout as part of the scientific contract. The existing self-test covered only the original synthetic formatting and therefore could not detect the incompatibility.

### Prevention

1. Private notebook compatibility transformations must identify contract fields semantically where practical. For Python dict fields, AST identity is preferred to whitespace-sensitive multi-line matching.
2. Formatting normalization and scientific transformation are separated. `normalize_codeparrot_pilot_v3_for_full.py` only canonicalizes the already-verified `manifest["selection"]` source span; it does not change scoring, row selection, labels, metrics, model identity, or authorization.
3. The normalizer self-test covers inline, multi-line, and single-quoted dict forms.
4. The full transformer self-test still runs after normalization, so both compatibility and scientific-boundary checks must pass before protected execution is exposed.
5. Every follow-up execution remains a new one-shot workflow. Failed workflows are not re-run automatically.
6. The target-absence check remains mandatory immediately before any write. A later attempt must stop if `renta0426/codeparrot-fresh-full-v1` already exists.
7. No performance labels or metrics are read during these repair steps.

## Execution rule after this incident

A CodeParrot full-run launch is permitted only when all of the following are true in the same protected run:

- live competition policy preflight passes;
- source pilot is COMPLETE and exact version 3;
- source scientific/repair markers pass;
- cohort is COMPLETE;
- target is absent;
- GPU quota/concurrency admission passes;
- AST manifest normalization passes;
- existing 200->4,000 transformer passes;
- transformed full-bundle markers pass;
- only then may one `kaggle kernels push` be attempted.

Any failure before the final write is classified as a non-compute pre-write failure and must not be retried automatically.
