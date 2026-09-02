# Kaggle Read-only Authentication Smoke Test

## Purpose

`10-kaggle-readonly-smoke.yml` verifies that the protected `kaggle-readonly` Environment exposes a valid `KAGGLE_API_TOKEN` and that a GitHub-hosted runner can make one authenticated, read-only Kaggle API request.

The test does not install the Kaggle CLI and does not perform a write operation.

## Trigger and approval

The workflow runs only when its own workflow file is pushed to `main` through the protected branch flow.

Before the job starts, GitHub must apply the `kaggle-readonly` Environment protection rules. The owner reviews the pending deployment in GitHub Actions and explicitly approves it. Until approval, the Environment Secret is unavailable to the runner.

## Fixed security boundary

The job validates all of the following before the authenticated step:

- repository: `renta0426/kaggle-actions-bridge`
- repository ID: `1354356687`
- actor: `renta0426`
- actor ID and sender ID: `71638068`
- event: `push`
- ref: `refs/heads/main`
- workflow source: the triggering commit on `main`
- runner environment: `github-hosted`
- runner image family: Linux X64 on `ubuntu-24.04`

The workflow declares `permissions: {}`. No checkout, external Action, package installation, cache, artifact, PAT, SSH key, cloud credential, or self-hosted runner is used.

## Secret exposure boundary

`KAGGLE_API_TOKEN` is mapped only to the single Python process that performs the API request. It is not mapped at workflow or job scope.

The process checks only whether the value:

- exists;
- has no surrounding whitespace;
- starts with `KGAT_`;
- falls within a broad length boundary.

The token, its prefix beyond the fixed format name, its length, request headers, and environment-variable listing are not printed.

## API request

The request is hard-coded to:

```text
GET https://www.kaggle.com/api/v1/competitions/list?page=1
```

The request uses `Authorization: Bearer <token>` and accepts JSON only.

Additional controls:

- TLS certificate verification uses Python's default trusted CA store.
- Environment-configured HTTP proxies are ignored.
- HTTP redirects are rejected so the Authorization header cannot be forwarded to another origin.
- The response body is capped at 2 MB.
- The body must be valid JSON.
- Response records are not logged.

A successful log contains only the HTTP status, response media type, top-level JSON type, and an aggregate item count when available.

## Interpretation

| Result | Meaning |
|---|---|
| Environment approval is requested | Environment protection is active and the workflow reached the protected job gate. |
| Secret unavailable | The Environment name or Environment Secret configuration is incorrect. |
| HTTP 401 or 403 | Kaggle was reached, but the token was rejected or lacks applicable access. |
| HTTP 200 with valid JSON | The token and authenticated Kaggle API path are operational. |
| Redirect rejected | Kaggle changed the endpoint behavior or an unexpected network path was encountered. |
| DNS/TLS/timeout error | GitHub-hosted runner could not complete the network request. |

## Non-goals

This test does not:

- reveal or persist the Kaggle account profile;
- download competition data;
- accept competition rules;
- push or run a Notebook;
- create or update a Dataset or Model;
- make a competition submission;
- install or validate the Kaggle CLI;
- install or validate NVIDIA `nvidia-kaggle`.

Those capabilities require separate review and separate workflows after this smoke test passes.
