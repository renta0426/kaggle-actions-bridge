# Diagnostic notes

## CMI-Flu B1 run 20260902

The first CMI-Flu B1 kernel launch succeeded at the Kaggle API level but the kernel itself entered an error state and produced no output files. The first bounded diagnostic confirmed a non-empty 2,040-byte execution log but its exact-line traceback parser did not match Kaggle's prefixed log format. Diagnostic v2 strips stream/timestamp prefixes, redacts long identifiers, URLs, paths and numeric values, and emits only error-related lines plus exception/frame metadata. Raw Kaggle logs remain ephemeral and are not committed or uploaded as artifacts.
