# Kaggle Read-only Smoke Test Result

## Result

- Workflow: `Kaggle read-only authentication smoke test #2`
- Run ID: `33611396605`
- Commit: `350f55d3b3488555191ceb897ebd202124e8979b`
- Environment: `kaggle-readonry`
- Conclusion: **PASS**
- Approved execution completed: 2026-09-02 09:12 UTC / 18:12 JST

## Verified controls

- GitHub-hosted Ubuntu 24.04 runner
- `permissions: {}`; effective GitHub token permission was `Metadata: read`
- repository ID, owner actor ID, sender ID, event, branch, workflow ref and workflow SHA checks passed
- `KAGGLE_API_TOKEN` was obtained from the protected Environment only after deployment approval
- no checkout, third-party Action, package installation, cache, artifact, submission, upload or download was used

## Kaggle result

The fixed authenticated request to `GET https://www.kaggle.com/api/v1/competitions/list?page=1` returned:

- HTTP `200`
- media type `application/json`
- top-level type `list`
- 19 records

Response records were deliberately not logged.

## Credential/log audit

The Actions runner masked the secret as `***` in the environment display and in the rendered Authorization expression. The raw token, token length, Authorization header value, cookies and response records did not appear in the fetched job log.

No workflow artifacts were created.

## Next gate

Before executing the official Kaggle CLI with the Environment Secret, resolve the complete `kaggle==2.2.4` dependency set in a secret-free job, record exact package versions and SHA-256 hashes, and then install only that locked set in the protected job.
