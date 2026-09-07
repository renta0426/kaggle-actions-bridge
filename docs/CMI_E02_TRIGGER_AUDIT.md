# E02 merge-trigger audit

The E02 recovery implementation and canonical operating procedure are in
[CMI_E02_INCIDENT_AND_RUNBOOK.md](CMI_E02_INCIDENT_AND_RUNBOOK.md).

During PR #141, a proposed edit to `docs/OPERATIONAL_LESSONS.md` also triggered
the secret-free PR job of old HAI workflow 154. Inspection established that
workflow 154 includes that shared document in **both** pull-request and main-push
paths and uses the global Kaggle concurrency group. Merging such an incidental
document change could create an unrelated protected HAI launch request.

This PR therefore leaves `docs/OPERATIONAL_LESSONS.md` byte-identical to its base.
The new E02 runbook remains a separate document referenced by PR #141 and the
science incident report. No unrelated historical compute workflow is modified
or relaunched. The retired E02 workflow 174 has no push trigger; the intended
new protected operation is workflow 175, which is read-only.

Future workflow reviews must inspect trigger paths as well as job code. Shared
runbooks/helper changes should trigger credential-free regression tests, not
replay historical one-shot compute requests. A future generalized launcher
should use an explicit immutable request ID/manifest and a durable executed
request ledger, rather than treating every push to shared documentation as a
new compute instruction. That broader migration is not silently applied to
unrelated experiments in this repair.
