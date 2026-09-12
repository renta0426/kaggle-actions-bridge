#!/usr/bin/env python3
"""Runtime hardening entrypoint for PYTHIA-MIMIR-DEVELOPMENT-V1.

The scientific protocol lives in ``pythia_mimir_development_v1.py``.  This
entrypoint patches two runtime-only compatibility details without changing any
candidate score, fit, sign, layer, fusion weight, sample set, or label boundary:

1. move the completed model to CPU before cache cleanup so the caller's still-
   live local reference cannot retain T4 VRAM while the next model is loaded;
2. call ``FeatureUnion.fit_transform`` for the post-seal HashingVectorizer-only
   content diagnostic, avoiding sklearn fitted-state ambiguity.  HashingVectorizer
   is stateless and this diagnostic remains post-seal and excluded from candidates.
"""
from __future__ import annotations

import gc

import pythia_mimir_development_v1 as core


def _unload_model_hardened(model, tokenizer) -> None:
    import torch

    # ``core.run`` still owns a local reference after this call.  Moving the
    # completed model off CUDA is therefore required; merely ``del``-ing this
    # function's parameter would not release the caller's GPU tensors.
    try:
        if not getattr(model, "hf_device_map", None):
            model.to("cpu")
        else:
            # Accelerate-dispatched models may reject a blanket .to().  Move
            # parameters/buffers module-by-module only when they currently live
            # on CUDA; tied tensors are handled by PyTorch storage semantics.
            for module in model.modules():
                local_parameters = list(module.parameters(recurse=False))
                local_buffers = list(module.buffers(recurse=False))
                if any(getattr(value, "is_cuda", False) for value in [*local_parameters, *local_buffers]):
                    try:
                        module.to("cpu")
                    except Exception:
                        pass
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _content_oof_diagnostic_hardened(texts, y, config):
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import FeatureUnion

    y = np.asarray(y, dtype=int)
    evaluation = config["evaluation"]
    splitter = StratifiedKFold(
        n_splits=int(evaluation["content_oof_folds"]),
        shuffle=True,
        random_state=int(evaluation["content_oof_seed"]),
    )
    prediction = np.full(len(texts), np.nan, dtype=np.float64)
    union = FeatureUnion([
        ("char", HashingVectorizer(
            analyzer="char", ngram_range=(3, 5), n_features=2**16,
            alternate_sign=True, norm="l2", lowercase=False,
        )),
        ("token", HashingVectorizer(
            analyzer="word", ngram_range=(1, 2), n_features=2**16,
            alternate_sign=True, norm="l2", lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
        )),
    ])
    # Stateless transforms; fitting sets only FeatureUnion compatibility state.
    X = union.fit_transform(texts)
    for fold, (fit, hold) in enumerate(splitter.split(np.arange(len(texts)), y)):
        model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1000,
            random_state=int(evaluation["content_oof_seed"]) + fold,
        )
        model.fit(X[fit], y[fit])
        prediction[hold] = model.predict_proba(X[hold])[:, 1]
    if not np.isfinite(prediction).all():
        raise RuntimeError("content OOF coverage failed")
    return prediction


core.unload_model = _unload_model_hardened
core.content_oof_diagnostic = _content_oof_diagnostic_hardened


if __name__ == "__main__":
    raise SystemExit(core.main())
