# CodeParrot current-output basename repair — 2026-09-08

## Established failure

Prior run: `34175131465`, attempt 1, workflow `214-codeparrot-full-eval-public-v2.yml`, commit `20452003cc99fb687f0fb7ba2281e9b9c5ce7236`.

The secret-free validation job succeeded. Protected job `101903003025` failed at `Read exact current output once and verify uploaded identity`. Its membership evaluation and aggregate emission steps were skipped, and unconditional cleanup succeeded.

The user-supplied exact traceback identifies `_parse_allow` rejecting `__huggingface_repos__.json:428` with `ValueError: allow-file must be NAME:MAX_BYTES with a basename only`. Reading the actual main parser confirms that its first-character class was `[A-Za-z0-9]`. A local, credential-free extraction of that exact parser reproduced the rejection; the reconstructed source Git blob matched `89192fdcc9386667e8af42fd7e1b1c45f7f2e92c` before modification.

This is a bridge validator defect, not a Kaggle API/network failure, not invalid output identity, not an upstream Kaggle bug, and not a scientific negative result.

## Side effects and resources

- Kaggle write calls: 0; Kaggle CPU/GPU/TPU runs started: 0; submissions: 0.
- `kaggle kernels output` was not reached; membership sources and label join were not reached.
- GitHub-hosted CPU was consumed for setup, dependency installation and validation. Do not call this zero total compute.
- The workflow wrote a temporary runner-local Kaggle credential before invoking the parser. Its cleanup step succeeded; the handoff reports credentials retained 0 and raw predictions retained 0. Do not describe this as zero local writes.
- No raw Kaggle output, sample code, joined labels, or private science source is copied into this public repository.

## One causal repair and regression

Permit safe leading underscores in basename allowlists. Continue to reject slash, backslash, leading dot/hyphen, whitespace/control characters, overlong names, and any `..` substring. Count, per-file size, total-download size, duplicate, symlink, exact-version, post-download-identity, unexpected-file and cleanup checks are unchanged.

The authoritative public v2 `--self-test` now invokes `output_allowlist_self_test`, exercising the real reader parser. Fixtures include the exact five declared filenames and byte limits; the generated `__huggingface_repos__.json` is not dropped. Rejection fixtures include `../secret`, `/tmp/x`, `a/b`, `a\\b`, `a..b`, duplicate entries, 33 entries and a file limit above 64 MiB. Boundary values of 32 entries and 64 MiB remain valid. This is run by the existing secret-free PR job and again before credential use on main. Existing low-FPR, tie-block, auxiliary and Stage2 fusion fixtures remain unchanged.

The evaluator's scientific core, all predictor formulas, matching contract, promotion gates, model/cohort/artifact hashes and workflow are unchanged. Only the request ID is replaced with `20260908-poisoned-chalice-codeparrot-full-eval-basename-repair-001`.

## Fresh-run protocol

Do not rerun `34175131465`. Require secret-free regression success on a fresh repair PR, merge that PR, and require the human `kaggle-readonry` approval for the resulting new main run and request.

The protected operation remains: verify exact target current version 1; one captured official output download; verify all five uploaded byte sizes/SHA-256 values; remove Kaggle credential; seal label-free predictors; reconstruct the pinned public membership manifest; prove complete ID/content agreement; join once; execute frozen CPU metrics/bootstrap; emit aggregates only; unconditional cleanup.

Opening labels consumes CodeParrot even if a later step fails. Never interpret a post-join failure as an unused holdout; establish the exposure boundary before any recovery. No GPU rerun, prediction retuning, competition rows, hidden labels, Public leaderboard read, or competition submission is authorized by this repair.
