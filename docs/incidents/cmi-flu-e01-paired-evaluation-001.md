# CMI-Flu E01 paired evaluation runtime failure 001

Date: 2026-09-07

## Scope

This incident concerns the first Kaggle execution of strategy-v2 E01 paired evaluation.

- GitHub Actions run: `34068082388`
- launch job: `101580307819`
- bridge merge commit: `a3f410089ae82a31df88a9b7e5c1973b9935b0c9`
- science commit: `0b2ecb47eaa09f22450424c9c06dc88cf44bc1fb`
- E01 source blob: `dd27aea0cf97d41bad3cec64819c4c4269d94cbd`
- E01 v2 source blob: `8cc64dc5ab9483d5957cfada18d445188566c56c`
- failed private kernel: `renta0426/cmi-flu-e01-paired-eval-20260907-001`, version 1
- accelerator: CPU
- Internet: disabled
- Competition submission: not attempted

One Kaggle CPU Notebook execution was consumed. The run reached Kaggle successfully, transitioned through queued/running, and terminated with `KERNELWORKERSTATUS.ERROR`. Current-version aggregate output read and sanitizer steps did not execute.

## Failure evidence

The Notebook emitted only the privacy-safe failure identity:

```text
CMI_FLU_E01_FAILED stage=run_e01 exception_type=TypeError error_code=a117b10075001dd7887e
```

The runtime constructs this code as the first 20 hexadecimal characters of SHA-256 over `ExceptionType:message`. The code reconstructs exactly to:

```text
TypeError:run_compact_task() got an unexpected keyword argument 'selection_policy'
```

## Root cause

This is a bridge/frozen-runtime API compatibility defect, not an E01 scientific-result failure.

The frozen B2.1 package exposes an older `cmi_flu.evaluation.run_compact_task` call surface that does not accept `selection_policy`. The already-audited B2.1 runtime adapter installs the intended robust-v1 implementations on `cmi_flu.runner.run_compact_task` and `cmi_flu.runner.run_hai_compact_for_panels`, but E01 imports those functions from `cmi_flu.evaluation`. Consequently E01 called the stale evaluation symbol before any E01 result could be produced.

The E01 science source, frozen incumbent identities, splits, metrics, negative controls, random seed, and n=28 sensitivity contract are not changed by the repair.

## Repair 002

A single narrowly scoped compatibility repair is prepared under a fresh request and kernel slug:

- request: `20260907-cmi-flu-strategy-e01-paired-evaluation-repair-002`
- kernel: `renta0426/cmi-flu-e01-paired-eval-repair-20260907-002`
- repair marker: `e01_frozen_selection_policy_api_compat_v1`

After the existing B2.1 adapter installs its robust implementations, the repair binds wrappers into `cmi_flu.evaluation` that:

1. accept the E01 `selection_policy` argument,
2. require it to be exactly `robust_v1`, and
3. delegate to the already-audited robust implementation installed on `cmi_flu.runner`.

The same compatibility is applied to HAI panel evaluation so E01 does not fail later on the analogous `selection_policy` argument.

Per bridge policy, this is not an automatic retry. It uses a fresh kernel slug, has `automatic_compute_retries: 0`, and requires a new protected-environment approval before any Kaggle write.
