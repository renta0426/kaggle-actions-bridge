# Kaggle Write and Download Integration Test

## Authorization

This workflow exists for an explicitly authorized integration test. The repository owner explicitly expanded the Kaggle token capability beyond read-only and requested all operations below to be executed.

The legacy GitHub Environment name remains `kaggle-readonry`; its name no longer describes the full Kaggle token capability. The Environment remains manually approved before every run.

## Operations

`50-kaggle-write-download-integration.yml` performs exactly four requested tests:

1. Retrieve the exact private notebook source for `renta0426/rsna-knee-public0033-meniscus-10-percent-residual` with `scriptVersionId=346645650`.
2. Create one uniquely named **private** Kaggle dataset containing `bridge-smoke.txt`. The created private dataset is intentionally retained on Kaggle after the run so the owner can inspect it.
3. List and download `sample_submission.csv` from `biohub-cell-tracking-during-development`.
4. List the competition discussion sorted by active/recent activity and retrieve the returned discussion threads/comments.

No competition submission is made. No existing Kaggle notebook, dataset, competition object, or discussion is modified.

## Security boundaries

- GitHub-hosted `ubuntu-24.04` only.
- `permissions: {}`; no GitHub repository write token is exposed to the runner.
- Owner/repository/event/ref/workflow identities are pinned.
- `KAGGLE_API_TOKEN` is mapped only to the integration execution step and is protected by the existing Environment approval.
- Official Kaggle CLI `2.2.4` and its 32 dependencies are installed from the committed 33-package SHA-256 lock using `--require-hashes`.
- No external GitHub Action, cache, or artifact is used.
- Private notebook source, competition file, and discussion contents remain only in the ephemeral runner working directory and are deleted in an `always()` cleanup step.
- Public workflow logs contain only success/failure and non-sensitive metadata such as byte counts, hashes, version number, topic counts, and the private dummy dataset handle.

## Private notebook version resolution

Kaggle CLI can pull a notebook by owner/slug and optional ordinal version, but the supplied URL uses Kaggle's global `scriptVersionId`. The test therefore uses Kaggle's authenticated internal JSON service to:

1. resolve the private kernel id;
2. list its versions;
3. locate the row whose kernel session id equals `346645650`;
4. retrieve source for that exact session id.

The approach mirrors the version-resolution mechanism used by the pinned NVIDIA `nvidia-kaggle` skill, while using the already hash-locked `requests` dependency.

## Private dataset side effect

The workflow creates one dataset handle of this form:

```text
renta0426/kaggle-actions-bridge-smoke-<workflow-run-id>-<attempt>
```

`kaggle datasets create` is invoked **without** `-u`/`--public`; Kaggle CLI 2.2.4 documents private as the default. The workflow verifies that `bridge-smoke.txt` is observable through the authenticated dataset-files API before reporting success.

## Competition data and discussions

Before downloading the requested competition file, the workflow first checks the authenticated competition file listing for `sample_submission.csv`.

For discussions, it requests up to 100 topics using the activity-oriented sort (`active`, with `recent` fallback), then downloads each returned topic with comments using the competition topic API. The text itself is never printed or uploaded as an Actions artifact.
