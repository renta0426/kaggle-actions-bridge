# CMI-Flu controlled Public probes request 003 incident

Date: 2026-09-06

## Classification

- request: `20260906-cmi-flu-public-probes-003`
- workflow run: `33999234203`
- target: `renta0426/cmi-flu-public-probes-20260906-003`
- failure class: `resource-consumed`
- Competition submission attempted: `false`
- automatic compute retry: `0`

The protected workflow successfully pushed exactly one private CPU Notebook. The status monitor observed `KERNELWORKERSTATUS.QUEUED` and then `KERNELWORKERSTATUS.ERROR` 30 seconds later. Output readout and probe validation were skipped.

## Root cause

The supplied Kaggle log reports the first runtime failure at `stage=materialize_frozen_b21_package`:

```text
NameError: name 'sha256_bytes' is not defined. Did you mean: 'sha256_file'?
```

The request-003 generator used `sha256_bytes(bundle)` when verifying the frozen B2.1 package. `sha256_bytes` exists in the bridge-side prepare script but was not injected into the generated Kaggle runtime. The generated runtime already imports `hashlib`, so the required repair is to use `hashlib.sha256(bundle).hexdigest()` directly at this boundary.

No B2.1 fitting, Public-probe fitting, Competition submission, or Public-score tuning occurred before the failure.

## Repair boundary

Request 003 must not be rerun. Repair 004 uses a fresh target slug and fresh approval. It changes only the missing runtime-symbol mechanism at the frozen-package SHA-256 check and adds a credential-free assertion that reconstructs the frozen package and computes its SHA-256 through the exact generated-runtime `hashlib` binding before any protected run.

The Public-probe science blob, frozen B2.1 package, runtime adapter, three-probe family, resource class, and no-submission contract remain unchanged.
