# E02 merge-trigger audit addendum: push-only legacy dependencies

After PR #141 merged as `4f63faccd8129ac0366bafd50ed0e8129fac1f37`, main-push runs also appeared for historical Public-probe workflows 163 and 164. Their **push** path filters include `scripts/kaggle_current_output_read.py`, but their **pull_request** path filters do not. The final PR validation list therefore did not reveal these two additional production triggers. The earlier HAI shared-document trigger had been avoided, but that was not a complete push-trigger audit.

Observed runs: 163 = `34085988483` (cancelled while pending); 164 = `34085988457` (protected approval waiting at the time of inspection). The intended E02 read-only recovery is `34085988392`; its validation passed, but its recovery job is pending behind the shared compute lock. No approval has been given by the agent to either historical request.

This follow-up retires only those two immutable historical one-shot launch entry points. Their original code is preserved in Git history at `2010e1cec8f9bfa4810a97bb7e35b76319abef96`; scientific code, data, results and models are unchanged. The entry points now have no automatic trigger, no protected job, no credentials and no compute lock. Manual dispatch refuses replay instead of creating a new version of a historical target.

A secret-free YAML-structure regression checks workflows 163, 164 and previously retired 174. It validates that there is no push/PR trigger, protected environment, compute lock, action invocation or nontrivial command path. The new regression itself performs no Kaggle call and never reads the private science repository.

Changing a workflow file does not cancel runs already created from its historical commit. The existing unintended run 34085988457 must be cancelled/rejected, NOT approved. Then approve only E02 recovery 34085988392. The available GitHub connector has no cancel-run action; no token or permissions expansion is used to work around that limitation.

Operating rule: before merging shared-helper changes, evaluate **main-push filters of all workflows independently of pull-request filters**. Completed one-shot launch workflows should be retired. Further launches require a fresh immutable request, reviewed payload and explicit approval, rather than a push to a shared helper. This fixes the observed 163/164/174 cases; it is not a claim that every historical workflow in the repository has been migrated.
