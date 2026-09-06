# CMI-Flu controlled Public-probe generator 005 incident

Request `20260906-cmi-flu-public-probes-005` consumed one private CPU Notebook and failed after the frozen B2.1 regeneration completed. The authoritative seven-task selected-model map passed. The failing guard was the exact byte-level SHA-256 equality check against the historical B2.1 `submission.csv`.

The regenerated submission did not byte-match the original recorded B2.1 submission SHA-256 `46f187ba85957ef1815f8b89d6f7aec53fa0b935d37225f05d140e309105dd38`.

The frozen model code fixes model random states, but byte-for-byte equality of a floating-point CSV across separate Kaggle workers is stronger than the scientific reproducibility contract. The pipeline includes parallel ExtraTrees (`n_jobs=-1`) and numerical linear-algebra operations, and CSV serialization records floating-point output directly. The historical row-level submission bytes are no longer available through the current protected read path, so the bridge cannot establish whether the mismatch is only least-significant numerical/serialization drift or a larger prediction difference from the historical file.

Repair class for a fresh request: do not claim byte identity to the historical file. Preserve the exact frozen B2.1 package, runtime adapter, authoritative selected-model map, Competition-data MD5 verification, fixed Public-probe science blob, and fixed three-probe family. Export the regenerated B2.1 submission as an explicit control CSV alongside the three probes. The three probes must copy every non-target task column exactly from that regenerated control. A later Competition-submission workflow may submit the control and the three predeclared probes together so the three probes are interpreted against the same regenerated backbone without adapting candidates after observing Public scores.

Request 005 must not be rerun automatically. No Competition submission was attempted by the generator.
