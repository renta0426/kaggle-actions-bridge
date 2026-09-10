# CMI-Flu E12a-001 pre-write materialization incident

Date: 2026-09-11 JST

## Scope

This incident covers the first protected execution attempt for Strategy-v2 E12a final-system reproduction.

- workflow run: `34494784507`
- protected job: `102930444158`
- bridge commit: `fd06724073c9bb60a45a27e3a7feb6a8c6bd1d57`
- request: `20260910-cmi-flu-strategy-e12a-final-reproduction-001`
- target: `renta0426/cmi-flu-e12a-final-reproduction-20260910-001`

## Observed failure

The secret-free validation job succeeded. In the protected job, the repository/actor/workflow boundary succeeded, then the next step failed while materializing the exact approved bridge commit and runtime:

`WORKDIR: unbound variable`

The step created a shell-local variable `workdir`, appended `WORKDIR=<path>` to `$GITHUB_ENV`, and then incorrectly referenced `${WORKDIR}` later in the same shell step. Values written to `$GITHUB_ENV` become available to subsequent steps, not to the currently executing shell.

## Failure classification

`bridge_implementation`

This is not a Kaggle transport/API failure, Kaggle compute failure, Competition input staging failure, output-hygiene failure, or scientific negative.

## Side-effect boundary

The failure occurred before:

- locked Kaggle client installation,
- Kaggle authentication,
- live Competition preflight,
- target existence check,
- CPU admission check,
- `KaggleApi.kernels_push`,
- any Kaggle compute,
- any output recovery,
- any Competition submission.

Therefore the E12a target was not consumed by this failed workflow attempt. The existing request/science/runtime contracts remain valid.

## Single repair

Use the shell-local `${workdir}` for runtime construction and self-test inside the materialization step, while still exporting `WORKDIR` through `$GITHUB_ENV` for subsequent steps. Do not alter E12a science, frozen incumbent identities, reproduction tolerance, data/CV contract, target, resource class, or submission policy.

The repaired workflow must pin and verify the existing E12a request/payload/prepare/execute/sanitize blobs, exercise the corrected same-step materialization path in secret-free PR CI, and require a fresh `kaggle-readonry` approval before the protected Kaggle operation.