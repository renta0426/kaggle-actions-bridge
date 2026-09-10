#!/usr/bin/env python3
"""Fail-closed source-attachment identity checks for CMI-Flu TabPFN kernels.

Kaggle 2.2.4 exact GetKernel was observed to canonicalize only the model
framework segment from ``pytorch`` in SaveKernel metadata to ``PyTorch`` in
``model_data_sources``.  This helper permits that representation-only case
change and nothing else.  Competition source identity remains exact.
"""
from __future__ import annotations

from typing import Any

AUDIT_RUN_ID = 34460007561
AUDIT_JOB_ID = 102815418161


class SourceAttachmentIdentityError(RuntimeError):
    """Fixed local error categories; never include remote metadata values."""


def _model_parts(value: str) -> tuple[str, str, str, str, str]:
    if type(value) is not str:
        raise SourceAttachmentIdentityError("model_source_not_string")
    parts = value.split("/")
    if len(parts) != 5 or any(part == "" for part in parts):
        raise SourceAttachmentIdentityError("model_source_shape_mismatch")
    owner, model, framework, variant, version = parts
    return owner, model, framework, variant, version


def model_source_equivalent(expected: str, observed: str) -> bool:
    """Compare exact model identity with framework-segment case normalization.

    owner/model/variant/version are byte-for-byte strict.  Only framework is
    casefolded because that is the representation change measured by the
    E11b-005 read-only metadata audit 002.
    """
    try:
        exp = _model_parts(expected)
        obs = _model_parts(observed)
    except SourceAttachmentIdentityError:
        return False
    return bool(
        exp[0] == obs[0]
        and exp[1] == obs[1]
        and exp[2].casefold() == obs[2].casefold()
        and exp[3] == obs[3]
        and exp[4] == obs[4]
    )


def require_exact_source_attachments(
    metadata: Any,
    *,
    expected_model_source: str,
    expected_competition_source: str,
) -> None:
    """Require exactly one audited model source and one exact competition source."""
    model_sources = list(getattr(metadata, "model_data_sources", []) or [])
    competition_sources = list(getattr(metadata, "competition_data_sources", []) or [])
    if len(model_sources) != 1:
        raise SourceAttachmentIdentityError("model_source_count_mismatch")
    if len(competition_sources) != 1:
        raise SourceAttachmentIdentityError("competition_source_count_mismatch")
    observed_model = model_sources[0]
    observed_competition = competition_sources[0]
    if type(observed_model) is not str or not model_source_equivalent(
        expected_model_source, observed_model
    ):
        raise SourceAttachmentIdentityError("model_source_identity_mismatch")
    if type(observed_competition) is not str or observed_competition != expected_competition_source:
        raise SourceAttachmentIdentityError("competition_source_identity_mismatch")


def self_test() -> None:
    assert AUDIT_RUN_ID == 34460007561
    assert AUDIT_JOB_ID == 102815418161
    expected = "prior-labsai/tabpfn-3/pytorch/default/1"
    observed = "prior-labsai/tabpfn-3/PyTorch/default/1"
    assert model_source_equivalent(expected, expected)
    assert model_source_equivalent(expected, observed)
    assert not model_source_equivalent(expected, "other/tabpfn-3/PyTorch/default/1")
    assert not model_source_equivalent(expected, "prior-labsai/other/PyTorch/default/1")
    assert not model_source_equivalent(expected, "prior-labsai/tabpfn-3/PyTorch/other/1")
    assert not model_source_equivalent(expected, "prior-labsai/tabpfn-3/PyTorch/default/2")
    assert not model_source_equivalent(expected, "prior-labsai/tabpfn-3/PyTorch/default")

    class Metadata:
        model_data_sources = [observed]
        competition_data_sources = ["cmi-flu-first-prediction-challenge"]

    require_exact_source_attachments(
        Metadata(),
        expected_model_source=expected,
        expected_competition_source="cmi-flu-first-prediction-challenge",
    )
    Metadata.competition_data_sources = ["other-competition"]
    try:
        require_exact_source_attachments(
            Metadata(),
            expected_model_source=expected,
            expected_competition_source="cmi-flu-first-prediction-challenge",
        )
    except SourceAttachmentIdentityError as exc:
        assert str(exc) == "competition_source_identity_mismatch"
    else:
        raise AssertionError("competition source drift did not fail closed")
    print(
        "CMI_FLU_SOURCE_ATTACHMENT_IDENTITY_SELF_TEST PASS "
        "framework_case_only=true owner_model_variant_version_strict=true competition_strict=true "
        f"audit_run={AUDIT_RUN_ID} audit_job={AUDIT_JOB_ID}"
    )


if __name__ == "__main__":
    self_test()
