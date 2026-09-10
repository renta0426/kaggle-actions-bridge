# CMI-Flu E11b 005 prewrite-guard repair

E11b-004 completed model-access preflight and exact runtime reconstruction, then stopped before any SaveKernel call while checking the failed 003 target. The direct `GetKernel` endpoint returned HTTP 500 for the nonexistent target, so the fail-closed guard treated absence as uncertain and aborted.

This is a bridge runtime compatibility / prewrite-guard failure, not a Kaggle compute failure and not a scientific result. The 004 write path was not reached; no kernel version or compute was consumed.

005 preserves the exact E11b science, TabPFN 8.5.0 package/checkpoint, CPU settings, LOSO folds, promotion gate, online exact-wheel transport, aggregate-only output, and no-submission boundary. The sole behavior change is exact-target absence checking: `kernels_list(user=owner, search=slug)` followed by exact `ref` equality for failed 003, failed 004, and fresh 005. Any exact match fails closed before write.
