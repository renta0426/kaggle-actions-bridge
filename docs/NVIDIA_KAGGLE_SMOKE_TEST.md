# NVIDIA Kaggle Skill Smoke Test

## Purpose

This test verifies that NVIDIA's `nvidia-kaggle-skill` can be materialized from a fixed upstream commit and can execute one bundled, read-only Kaggle workflow through the protected GitHub Actions bridge.

## Upstream pin

Repository: `NVIDIA/nvidia-kaggle`

Commit:

```text
2b78cf29f5f30680764292a6592de8d53d4147a8
```

GitHub reports this commit as signature-verified.

The tested files are additionally checked against their Git blob IDs:

```text
SKILL.md                         33a3e641da5af24edcee1398a26f1a1b058c3dc5
scripts/runtime.py               7b46f7a702b2f82d5c1777011c1369f0ba88e00b
scripts/fetch_competition_info.py 8a75ea865e51636426c0efaaae44c3b3c96ec171
```

## Installation interpretation

`nvidia-kaggle` is an Agent Skill/plugin, not a standalone Kaggle CLI. NVIDIA documents installation for generic Agent Skills harnesses as copying the whole `skills/nvidia-kaggle-skill` directory into the harness skills directory.

The GitHub-hosted runner has no Codex or Claude plugin runtime, so this smoke test follows that generic installation model: it copies the entire pinned skill directory into a temporary `installed-skills` / `skills` directory, verifies the selected bundled scripts, then executes a bundled script directly.

No local PC or self-hosted runner is used.

## Runtime dependencies

The tested competition-overview path uses the already audited official Kaggle CLI/runtime dependency lock:

```text
requirements/kaggle-2.2.4.lock
```

It contains 33 wheels pinned by version and SHA-256. Installation uses:

```text
--no-deps
--only-binary=:all:
--require-hashes
```

No dependency resolution occurs while the Kaggle secret is exposed.

## Test stages

### Stage 1: secret-free source verification

- owner/repository/event/ref/runner boundary is checked;
- credential environment variables must be absent;
- only the exact NVIDIA commit is fetched;
- commit SHA and selected Git blob IDs are verified;
- the whole Agent Skill directory is copied into a temporary skills directory;
- `runtime.py` and `fetch_competition_info.py` are bytecode-compiled;
- `fetch_competition_info.py --help` must succeed;
- all temporary files are deleted.

### Stage 2: protected read-only execution

The second job requires the existing `kaggle-readonry` Environment approval.

Before the secret is mapped to any step, the job:

- fetches and verifies the same fixed NVIDIA commit again;
- materializes the whole skill directory;
- fetches the 33-package lock from the exact triggering bridge commit;
- installs only hash-pinned wheels;
- runs `pip check`, Python compilation, and script help.

Only the final execution step receives `KAGGLE_API_TOKEN`.

It executes:

```text
python scripts/fetch_competition_info.py titanic
```

This is a read-only competition-overview workflow. Standard output and standard error are redirected to temporary files and never printed. The test checks only successful exit status and a broad output-size boundary, then deletes the files.

## Explicit non-goals

This smoke test does not:

- submit to a competition;
- accept competition rules;
- download competition data;
- push or run a notebook;
- create/update a dataset or model;
- ingest discussions or kernels;
- use NVIDIA's submission workflow;
- use a self-hosted runner;
- expose PAT, SSH, cloud, or private-repository credentials.

Any write-capable NVIDIA workflow requires a separate Environment and explicit per-operation approval before it is enabled.
