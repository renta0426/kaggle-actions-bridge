# CMI-Flu controlled Public probes request 004

Date: 2026-09-06

## Classification

Resource-consumed failure. GitHub Actions run `34001421161` successfully pushed one private CPU Notebook at `renta0426/cmi-flu-public-probes-20260906-004`. The Notebook ran for about 20 minutes and terminated with error. No Competition submission was attempted and there was no automatic compute retry.

## Failure

The supplied Kaggle log reported:

- stage: `regenerate_frozen_b21`
- exception: `BundleContractError`
- error code: `2daf3d1f7f807ac8f8b4`
- message: `regenerated B2.1 model mismatch for Task2.2: et_subtype_d5_l10`

The regenerated B2.1 run had already passed the frozen-package SHA-256 boundary repaired in request 004 and had completed enough of the frozen B2.1 pipeline to produce selected-model metadata.

## Root cause

The failure was in the bridge provenance assertion, not in the regenerated B2.1 science result.

Request 004 incorrectly hardcoded the older HAI expectations:

- Task2.2: `et_subtype_d3_l5`
- Task2.3: `pls_exact_5`

The authoritative B2.1 result report `reports/b2-1-results.md` records the actual frozen B2.1 selected models as:

- Task1.1: `pls_2`
- Task1.2: `enet_a0.001_l0.5`
- Task1.3: `pls_1`
- Task1.4: `raw_pre_vacc_conserved_anchor`
- Task2.1: `et_subtype_d3_l5`
- Task2.2: `et_subtype_d5_l10`
- Task2.3: `ridge_exact_a100`

The same report records the original B2.1 submission SHA-256 as `46f187ba85957ef1815f8b89d6f7aec53fa0b935d37225f05d140e309105dd38`.

The request-004 regenerated Task2.2 selection therefore matched the authoritative B2.1 record and was rejected only because the bridge guard was stale.

## Repair class

Request 004 is not rerun. Repair 005 uses a fresh target and changes only the provenance guard:

1. validate the exact seven-task selected-model map from the authoritative B2.1 report;
2. require the regenerated 40x7 B2.1 submission SHA-256 to be byte-identical to the original recorded B2.1 submission before generating any controlled probes;
3. preserve the exact Public-probe science blob, three-probe family, CPU resource, Internet-off setting, and no-submission contract.

This makes the final submission hash, rather than a partial remembered model list, the strongest backbone identity check.
