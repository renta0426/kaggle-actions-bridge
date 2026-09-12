# Poisoned Chalice Pythia/MIMIR development admission v1

This bridge audit admits a repeatedly usable **development** environment for Stage 2 model-transfer work. It does not consume a sealed final-confirmation environment and does not evaluate membership performance.

The target is `EleutherAI/pythia-2.8b` at the final `step143000` branch. The audit resolves that branch to an immutable commit before any later experiment is frozen. The non-deduplicated target is intentional: the benchmark member/nonmember construction is derived from The Pile train/test split, so using the non-deduplicated Pythia reduces one avoidable ambiguity between benchmark membership and the model's training corpus. This is still not claimed to be a bit-level audit of Pythia's exact training sequence.

The public development cohort is pinned to `Al-not-AI/mimir@47e348e97cae2ab8c1d2278d4ffd6db2f1d043f8`, GitHub source, `ngram_13_0.8`, with 1,000 member and 1,000 nonmember snippets. Canonical benchmark semantics are checked against `iamgroot42/mimir@1b6fd649eeeecc887275a2336c7da808ee58757d`: train maps to member, test maps to nonmember, samples are constructed at 100–200 whitespace words with the documented 512-token masking constraint, and the source code is MIT licensed.

The audit is intentionally lightweight and credential-free. It downloads no model weights, runs no model inference, calls no Kaggle API, and uses no membership labels for method selection. It resolves model/tokenizer metadata, hashes the two public JSONL cohort files, verifies counts/schema/word-length bounds and exact cross-split duplicates, and emits only aggregate identity metadata.

Passing this admission audit authorizes only the next design step: freeze a label-clean comparison of target-only output scores, target `local_64`, fixed StarCoder2 reference hidden probe, target 96-D geometry probe, fixed 0.5 rank fusions, and a content-only control **before** evaluating the Pythia/MIMIR membership labels. Target labels must not be used for coefficient, layer, sign, weight, or sample selection.
