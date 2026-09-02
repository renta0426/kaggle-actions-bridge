# Poisoned Chalice Stage1 request 003 failure

- Target: `renta0426/stage1-raw-fim-submission-v1`
- Launch request: `20260902-poisoned-chalice-stage1-raw-fim-003`
- Result: Kaggle worker `ERROR`; no submission was attempted.
- Failure location: `load_canonical_data()` -> `validate_public_data()` -> `unexpected train language counts`.
- Root cause: the bridge assembled separate research modules into one notebook namespace. `starter_plus.py` defines `EXPECTED_TRAIN_ROWS` as the canonical per-language row-count dictionary, while the flattened Stage1 runner rebound the same global name to integer `10000`. `validate_public_data()` therefore compared the observed language-count dictionary with integer `10000` and failed before model loading/feature extraction.
- Scope: bridge self-contained assembly bug, not a change to the frozen `raw_plus_fim` algorithm and not a Kaggle/Hugging Face/library defect.
- Corrective action for request 004: namespace all runner constants under `STAGE1_*`, verify there are no runner/source uppercase constant collisions before push, add stable nbformat cell IDs, require the existing target to still be private version 1 in an error state, and permit exactly one corrected T4 version push with zero automatic retries.
