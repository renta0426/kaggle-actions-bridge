# Diagnostic workflow context-scope defect — 2026-09-08

Run `34178537337`, head `94e3d8e466cdcddf3555e475fb1ec4ed3ab15c89`, failed workflow validation before creating any jobs. The Actions jobs API returned an empty job list. No credential step, exact metadata request, output download, Kaggle write/compute or membership access occurred.

The authored workflow placed `${{ runner.environment }}` in `jobs.diagnose.env`. GitHub's official context-availability table excludes `runner` from job-level env and permits it in step-level env: https://docs.github.com/en/actions/reference/workflows-and-actions/contexts . The invalid placement is directly visible in the source and reproduced by the new targeted context-scope regression. The server's annotation endpoint could not be read with the available connector, so this record does not pretend to quote an unseen server annotation.

The repair moves this one binding into the guarded step's env. The hosted-runner check itself, human approval, main-only protected job condition, identity/actor checks and no-output/no-compute contract are unchanged. A focused, standard-library-only regression rejects runner/steps/job in job.env and accepts runner in step.env. It tests the actual exact-commit workflow during secret-free validation. It is explicitly not a complete YAML/Actions validator; GitHub validation remains required.

The earlier empty `fetch_commit_workflow_runs` response was not evidence that no run existed: that connector wrapper filters pull-request-triggered runs. The direct repository Actions runs API identified the failed push run. Branch registration alone did not fix the syntax/context problem. Neither that failed run nor the full evaluation run is rerun; the causal repair produces a fresh validation commit.

Local checks: exact pre-fix source matched Git blob `67ab274f80fe2246edd5160b18c22ff8bc1a06b4`; baseline context violation reproduced; patched YAML parsed; embedded Python compiled; metadata diagnostic fixtures and context negative/positive fixtures passed. These are local checks, not a claim of remote CI success.
