# CMI-Flu Strategy v3 batch1 source-B precheck incident — 2026-09-11

## Classification

`bridge_implementation`

This was not a GitHub capability/permission failure, science implementation failure, Competition input staging failure, Kaggle transport/API failure, Kaggle compute failure, output sanitizer failure, or scientific negative.

## Failed execution

- Actions run: `34601047087`
- protected job: `103268207917`
- bridge commit: `14bc323a8945d5bd233fec281a3c09eba7a223b5`
- request: `20260911-cmi-flu-strategy-v3-batch1-001`
- intended target: `renta0426/cmi-flu-v3-batch1-audit-diagnostics-20260911-001`
- automatic compute retries: `0`
- Competition submissions: `0`

The protected boundary, exact runtime materialization, locked Kaggle client installation, and static no-fit/no-submit checks succeeded. The executor then failed in the read-only source-B identity precheck before `require_fresh_target` and before `api.kernels_push`.

Observed exception:

`FileExistsError: output directory already exists`

The caller passed the root returned by `TemporaryDirectory` directly to `kaggle_current_output_read.read_current_output`. That reader intentionally owns creation of its output directory and rejects an existing path.

Therefore:

- Kaggle Notebook write calls: `0`
- Kaggle target versions created by this failed run: `0`
- Kaggle compute launched by this failed run: `0`
- model fits: `0`
- Competition submissions: `0`
- six diagnostic CSVs generated: `0`

The cleanup message saying that six diagnostic CSVs remained in private Kaggle output was generic and did not describe this failed path; no target Notebook had been created.

## Repair

Use a deliberately absent child path below the temporary parent for source-B current-output recovery. The executor now has an explicit `fresh_source_output_dir` contract and its locked-client `--path-self-test` verifies that the selected child does not yet exist. `verify_source_B` also refuses an existing read destination.

The repair does not change the science payload, source-B identity, requested target, Competition submission boundary, retry policy, or compute resource.
