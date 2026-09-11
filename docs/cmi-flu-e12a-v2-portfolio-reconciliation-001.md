# CMI-Flu E12a-v2 portfolio reconciliation 001

Date: 2026-09-11

## Purpose

Run one fresh, private, CPU-only E12a-v2 reproduction audit after E12a-v1 exposed a finalization-layer Task1.3 manifest error.

E12a-v1 successfully reproduced its six embedded E01 references, but its final portfolio used the historical E01 Task1.3 control `b21_pls_1`. Science records created before E12a-v1 already retained `strict_asc_anchor` as the competition-facing Task1.3 predictor. Science PR #64 therefore fixes only the portfolio reconciliation contract; it performs no new model selection.

## Exact science relay

- science repository: `renta0426/CMI-Flu-Invited-Prediction-Challenge`
- science main: `e6e05e578f172ce710bc0f1da55acdf290e83dd9`
- `src/cmi_flu/strategy_e12a_v2.py`: Git blob `0c4c970c8bacfed61bfbb9587e0a0bfdec7903d9`
- `src/cmi_flu/strategy_e12a_v2_synthetic.py`: Git blob `6477076d9c5aba7d37d816147e4443b5c94ad2d4`
- `configs/strategy_e12a_v2_portfolio_reconciliation.json`: Git blob `6b40ac35a3b43c3bafcdd08fcc49fc9fe22960c1`
- frozen E01: `dd27aea0cf97d41bad3cec64819c4c4269d94cbd`
- frozen E01-v2: `8cc64dc5ab9483d5957cfada18d445188566c56c`
- frozen B2.1 config: `170d3211e2795c0730e481056c7bb068accf97c9`

The three E12a-v2 science/contract blobs are copied byte-for-byte into the public bridge and revalidated before runtime construction. Protected CI never reads the private science repository.

## Corrected final portfolio

- Task1.1: `b21_pls_2`
- Task1.2: `task12_anchor_residual_et_d5_l5_sqrt_lambda0.5`
- Task1.3: `strict_asc_anchor`, predictor `flow_rank__Antibody-secreting_cells_(ASC)`
- Task1.4: `raw_pre_vacc_conserved_anchor`
- Task2.1: `b21_et_subtype_d3_l5`
- Task2.2: `b21_et_subtype_d5_l10`
- Task2.3: `b21_ridge_exact_a100`

The old E01 Task1.3 `b21_pls_1` condition is reproduced only as a historical control and is explicitly marked `final_portfolio_member=false`.

## E12a-v1 parent is consumed

The prior target `renta0426/cmi-flu-e12a-final-reproduction-20260910-001`, version 1, was created by Actions run `34536723038` / protected job `103070014918`. SaveKernel succeeded and the version is consumed. It is never rerun.

The supplied E12a-v1 artifacts were complete, but the Notebook terminated `ERROR` after writing them because the generated E12a runtime retained an inherited E01 success print that accessed `result['tasks']` after the result schema had changed. The E12a-v2 builder explicitly replaces this terminal path.

## Terminal-path regression

The E12a-v2 generated runtime defines `terminal_success_line`. Runtime self-test invokes that exact helper using an E12a-v2 fixture that has no `tasks` key. PR validation also imports the generated runtime, calls its validator, renderer, and terminal helper, and rejects a fixture that tries to put `b21_pls_1` back into the final Task1.3 slot.

This directly covers the E12a-v1 post-artifact terminal defect rather than relying only on syntax/build checks.

## One-shot Kaggle contract

- request: `20260911-cmi-flu-strategy-e12a-v2-portfolio-reconciliation-001`
- target: `renta0426/cmi-flu-e12a-v2-portfolio-reconcile-20260911-001`
- expected version: 1
- private Notebook
- CPU only
- Internet disabled
- maximum active CPU Notebook: 1
- automatic compute retries: 0
- Competition submission: not authorized
- Public Leaderboard selection: not authorized
- output allowlist: `bridge-result.json`, `metrics.json`, `summary.md`

The protected executor reuses the audited E12a one-shot transport for live Rules/Data/Evaluation checks, exact fresh-target checks, resource-class admission, immediate pre-write rechecks, exactly one SaveKernel call, terminal-state monitoring, and current-version-only output recovery. A wrapper changes only the fresh request/target/title and runtime identity validation.

Any failure after a write is treated as a consumed version until proven otherwise. No automatic rerun is permitted.
