#!/usr/bin/env python3
"""One-shot private CPU executor for the frozen CMI-Flu E11c experiment."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

import cmi_flu_strategy_e11b_execute_v5 as e11b005
from cmi_flu_source_attachment_identity import require_exact_source_attachments

REQUEST_ID = "20260910-cmi-flu-strategy-e11c-robust-tabpfn-001"
TARGET = "renta0426/cmi-flu-e11c-robust-tabpfn-20260910-001"
TARGET_SLUG = TARGET.split("/", 1)[1]
TITLE = "CMI Flu E11c Robust TabPFN 20260910 001"
CONSUMED_E11B_TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-005"
EXPECTED_VERSION = 1

if e11b005.REQUEST_ID != "20260910-cmi-flu-strategy-e11b-tabpfn3-005":
    raise SystemExit("E11c executor E11b-005 request ancestry changed")
if e11b005.TARGET != CONSUMED_E11B_TARGET:
    raise SystemExit("E11c executor E11b-005 target ancestry changed")
if e11b005.TITLE != "CMI Flu E11b TabPFN3 20260910 005":
    raise SystemExit("E11c executor E11b-005 title ancestry changed")

v4 = e11b005.v4
prior = v4.prior
if v4.MAX_RUNTIME_BYTES != 900_000:
    raise SystemExit("E11c executor runtime budget ancestry changed")
if prior.MODEL_SOURCE != "prior-labsai/tabpfn-3/pytorch/default/1":
    raise SystemExit("E11c executor model source ancestry changed")
if prior.COMPETITION != "cmi-flu-first-prediction-challenge":
    raise SystemExit("E11c executor competition ancestry changed")
if prior.CHECKPOINT_FILENAME != "tabpfn-v3-regressor-v3_default.ckpt" or prior.CHECKPOINT_BYTES != 233_289_807:
    raise SystemExit("E11c executor checkpoint ancestry changed")
if prior.EXPECTED_VERSION != EXPECTED_VERSION:
    raise SystemExit("E11c executor version ancestry changed")

# Retarget only runtime orchestration. The frozen package/model/CV science is in
# the generated E11c runtime, while this module preserves the proven E11b-005
# Kaggle transport and admission/watcher/output-reader path.
v4.REQUEST_ID = REQUEST_ID
v4.TARGET = TARGET
v4.TARGET_SLUG = TARGET_SLUG
v4.TITLE = TITLE
prior.REQUEST_ID = REQUEST_ID
prior.TARGET = TARGET
prior.TARGET_SLUG = TARGET_SLUG
prior.TITLE = TITLE


def model_access_preflight() -> None:
    """Preserve the proven read-only exact model-version file preflight."""
    v4.model_access_preflight()
    print(
        "CMI_FLU_E11C_MODEL_ACCESS_PASS "
        "source=frozen_e11b_model_input write=false compute=false submission=false"
    )


def _require_fresh_target_by_list(api: KaggleApi) -> None:
    owner, slug = TARGET.split("/", 1)
    discovered = api.kernels_list(user=owner, search=slug, page_size=20) or []
    exact = [item for item in discovered if str(getattr(item, "ref", "")) == TARGET]
    if exact:
        raise RuntimeError("E11c exact fresh target exists; write refused")
    print(
        "CMI_FLU_E11C_EXACT_ABSENT "
        "label=fresh_e11c_001 confirmed=true method=kernels_list_exact_ref"
    )


def prewrite_guard(api: KaggleApi) -> None:
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != TARGET_SLUG:
        raise RuntimeError("E11c target identity contract changed")
    if TITLE != "CMI Flu E11c Robust TabPFN 20260910 001":
        raise RuntimeError("E11c title identity contract changed")
    if TARGET == CONSUMED_E11B_TARGET or TARGET_SLUG.endswith("tabpfn3-20260910-005"):
        raise RuntimeError("E11c refused consumed E11b-005 identity")
    _require_fresh_target_by_list(api)


def push(api: KaggleApi, runtime: Path, work: Path) -> None:
    raw = runtime.read_bytes()
    if len(raw) >= v4.MAX_RUNTIME_BYTES:
        raise RuntimeError(f"E11c runtime exceeds source budget:{len(raw)}")
    text = raw.decode("utf-8")
    required_runtime_anchors = (
        'TABPFN_WHEEL_B64 = ""',
        'E11C_BLOB = "acf876996b5849f7ba794999d45ecb862d1be507"',
        'E11C_SYNTH_BLOB = "9b94633192b5254a7d1debe8db10606b48104076"',
        'SCIENCE_COMMIT = "0815f4517345e127dbcbcae0f380a54f3a3d15bd"',
    )
    if any(anchor not in text for anchor in required_runtime_anchors) or '"pip","download"' not in text:
        raise RuntimeError("E11c generated runtime identity/transport contract changed")

    kernel_dir = work / "kernel"
    kernel_dir.mkdir(parents=True)
    shutil.copyfile(runtime, kernel_dir / "script.py")
    metadata = {
        "id": TARGET,
        "title": TITLE,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "competition_sources": [prior.COMPETITION],
        "model_sources": [prior.MODEL_SOURCE],
    }
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if {path.name for path in kernel_dir.iterdir()} != {"script.py", "kernel-metadata.json"}:
        raise RuntimeError("E11c kernel payload file set changed")
    if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
        raise RuntimeError("E11c materialized metadata changed")

    print(
        "CMI_FLU_E11C_API_WRITE_RECEIPT "
        + json.dumps(
            {
                "request_id": REQUEST_ID,
                "phase": "before_write",
                "transport": "KaggleApi.kernels_push",
                "target_sha256": hashlib.sha256(TARGET.encode()).hexdigest(),
                "runtime_sha256": hashlib.sha256(raw).hexdigest(),
                "runtime_bytes": len(raw),
                "internet_enabled_for_exact_package_retrieval": True,
                "write_attempted": False,
            },
            sort_keys=True,
        )
    )
    try:
        result = api.kernels_push(str(kernel_dir))
    except BaseException as exc:
        receipt = v4._error_receipt(exc)
        print("CMI_FLU_E11C_API_WRITE_RECEIPT " + json.dumps(receipt, sort_keys=True))
        response = getattr(exc, "response", None)
        body = str(getattr(response, "text", "") or "")
        safe = v4._safe_server_message(body) or v4._safe_server_message(str(exc))
        if safe:
            print("CMI_FLU_E11C_API_SAFE_ERROR message=" + safe)
        raise RuntimeError("E11c direct SaveKernel failed; no automatic retry") from exc

    error = str(getattr(result, "error", "") or "")
    print(
        "CMI_FLU_E11C_API_WRITE_RECEIPT "
        + json.dumps(
            {
                "request_id": REQUEST_ID,
                "phase": "after_api_write",
                "transport": "KaggleApi.kernels_push",
                "write_attempted": True,
                "response_type": type(result).__name__,
                "response_error_present": bool(error),
                "response_error_bytes": len(error.encode("utf-8", errors="replace")),
                "response_error_sha256": prior._digest_text(error),
                "raw_server_error_persisted": False,
            },
            sort_keys=True,
        )
    )
    if error:
        safe = v4._safe_server_message(error)
        if safe:
            print("CMI_FLU_E11C_API_SAFE_ERROR message=" + safe)
        raise RuntimeError("E11c SaveKernel response contained error; no automatic retry")

    direct = prior.base.kernel_meta(api, TARGET)
    if (
        str(getattr(direct, "ref", "")) != TARGET
        or not bool(getattr(direct, "is_private", False))
        or bool(getattr(direct, "enable_gpu", False))
        or bool(getattr(direct, "enable_tpu", False))
        or not bool(getattr(direct, "enable_internet", False))
        or int(getattr(direct, "current_version_number", 0) or 0) != EXPECTED_VERSION
    ):
        raise RuntimeError("E11c direct metadata identity mismatch")
    require_exact_source_attachments(
        direct,
        expected_model_source=prior.MODEL_SOURCE,
        expected_competition_source=prior.COMPETITION,
    )
    print(
        "CMI_FLU_E11C_API_IDENTITY_CONFIRMED "
        "version=1 private=true cpu=true internet=true model_attached=true "
        "competition_attached=true framework_case_normalization_only=true"
    )


prior.model_access_preflight = model_access_preflight
prior.prewrite_guard = prewrite_guard
prior.push = push


def main() -> int:
    return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
