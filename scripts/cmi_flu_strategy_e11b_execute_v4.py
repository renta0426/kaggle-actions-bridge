#!/usr/bin/env python3
"""E11b 004 executor: exact online-wheel transport, same frozen science."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

import cmi_flu_strategy_e11b_execute as prior

REQUEST_ID = "20260910-cmi-flu-strategy-e11b-tabpfn3-004"
TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-004"
TARGET_SLUG = TARGET.split("/", 1)[1]
TITLE = "CMI Flu E11b TabPFN3 20260910 004"
FAILED_003_TARGET = "renta0426/cmi-flu-e11b-tabpfn3-20260910-003"
MAX_RUNTIME_BYTES = 900_000

_EXPECTED_PRIOR = {
    "REQUEST_ID": "20260910-cmi-flu-strategy-e11b-tabpfn3-001",
    "TARGET": "renta0426/cmi-flu-e11b-tabpfn3-20260910-001",
    "TITLE": "CMI Flu E11b TabPFN3 20260910 001",
    "MODEL_SOURCE": "prior-labsai/tabpfn-3/pytorch/default/1",
    "CHECKPOINT_FILENAME": "tabpfn-v3-regressor-v3_default.ckpt",
    "CHECKPOINT_BYTES": 233_289_807,
    "EXPECTED_VERSION": 1,
}
for key, value in _EXPECTED_PRIOR.items():
    if getattr(prior, key, None) != value:
        raise SystemExit(f"E11b executor-v4 ancestry changed:{key}")

prior.REQUEST_ID = REQUEST_ID
prior.TARGET = TARGET
prior.TARGET_SLUG = TARGET_SLUG
prior.TITLE = TITLE


def _safe_server_message(value: str) -> str | None:
    if not value:
        return None
    lowered = value.casefold()
    banned = (
        "kgat_", "authorization", "cookie", "access_token", "api_token", "password",
        "participant_id", "subject_group", "oof_predictions", "challenge_predictions",
    )
    if any(token in lowered for token in banned):
        return None
    compact = re.sub(r"\s+", " ", value).strip()
    compact = re.sub(r"[^A-Za-z0-9_.,:;=+\-\[\] ()/'?&.%]", "?", compact)
    return compact[:400] if compact else None


def _error_receipt(exc: BaseException) -> dict:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    body = str(getattr(response, "text", "") or "")
    message = str(exc)
    return {
        "request_id": REQUEST_ID,
        "transport": "KaggleApi.kernels_push",
        "write_attempted": True,
        "exception_type": type(exc).__name__,
        "http_status": int(status) if isinstance(status, int) else None,
        "exception_message_bytes": len(message.encode("utf-8", errors="replace")),
        "exception_message_sha256": prior._digest_text(message),
        "response_body_bytes": len(body.encode("utf-8", errors="replace")),
        "response_body_sha256": prior._digest_text(body),
        "raw_server_error_persisted": False,
    }


def model_access_preflight() -> None:
    completed = subprocess.run(
        ["kaggle", "models", "instances", "versions", "files", prior.MODEL_SOURCE,
         "--format", "json", "--page-size", "20"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    stdout, stderr = completed.stdout, completed.stderr
    marker = {
        "request_id": REQUEST_ID,
        "operation": "model_instance_version_files",
        "model_source_sha256": hashlib.sha256(prior.MODEL_SOURCE.encode()).hexdigest(),
        "return_code": int(completed.returncode),
        "stdout_bytes": len(stdout.encode("utf-8", errors="replace")),
        "stdout_sha256": prior._digest_text(stdout),
        "stderr_bytes": len(stderr.encode("utf-8", errors="replace")),
        "stderr_sha256": prior._digest_text(stderr),
        "write_attempted": False,
        "cli_output_format": "--format json",
    }
    print("CMI_FLU_E11B_MODEL_ACCESS_RECEIPT " + json.dumps(marker, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("E11b 004 model access preflight failed; no kernel write attempted")
    try:
        payload = json.loads(stdout)
    except Exception as exc:
        raise RuntimeError("E11b 004 model access preflight returned non-JSON") from exc
    serialized = json.dumps(payload, sort_keys=True)
    if prior.CHECKPOINT_FILENAME not in serialized:
        raise RuntimeError("E11b 004 model access checkpoint not visible")
    if str(prior.CHECKPOINT_BYTES) not in serialized and f"{prior.CHECKPOINT_BYTES}.0" not in serialized:
        raise RuntimeError("E11b 004 checkpoint byte identity changed")
    print(
        "CMI_FLU_E11B_MODEL_ACCESS_PASS "
        f"checkpoint={prior.CHECKPOINT_FILENAME} bytes={prior.CHECKPOINT_BYTES} write=false compute=false"
    )


def _require_exact_absent(api: KaggleApi, target: str, label: str) -> None:
    owner, slug = target.split("/", 1)
    try:
        with api.build_kaggle_client() as client:
            request = ApiGetKernelRequest()
            request.user_name = owner
            request.kernel_slug = slug
            client.kernels.kernels_api_client.get_kernel(request)
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status == 404:
            print(f"CMI_FLU_E11B_EXACT_ABSENT label={label} confirmed=true")
            return
        safe = hashlib.sha256(f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")).hexdigest()[:20]
        raise RuntimeError(f"E11b 004 exact absence uncertain:{label}:{safe}") from exc
    raise RuntimeError(f"E11b 004 exact target exists; write refused:{label}")


def prewrite_guard(api: KaggleApi) -> None:
    owner, slug = TARGET.split("/", 1)
    if owner != "renta0426" or slug != TARGET_SLUG or TITLE != "CMI Flu E11b TabPFN3 20260910 004":
        raise RuntimeError("E11b 004 target identity contract changed")
    _require_exact_absent(api, FAILED_003_TARGET, "failed_003")
    _require_exact_absent(api, TARGET, "fresh_004")


def push(api: KaggleApi, runtime: Path, work: Path) -> None:
    raw = runtime.read_bytes()
    if len(raw) >= MAX_RUNTIME_BYTES:
        raise RuntimeError(f"E11b 004 runtime exceeds source budget:{len(raw)}")
    text = raw.decode("utf-8")
    if 'TABPFN_WHEEL_B64 = ""' not in text or 'pip","download"' not in text:
        raise RuntimeError("E11b 004 dependency transport runtime contract changed")

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
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if {p.name for p in kernel_dir.iterdir()} != {"script.py", "kernel-metadata.json"}:
        raise RuntimeError("E11b 004 kernel payload file set changed")
    if json.loads((kernel_dir / "kernel-metadata.json").read_text()) != metadata:
        raise RuntimeError("E11b 004 materialized metadata changed")

    before = {
        "request_id": REQUEST_ID,
        "phase": "before_write",
        "transport": "KaggleApi.kernels_push",
        "target_sha256": hashlib.sha256(TARGET.encode()).hexdigest(),
        "runtime_sha256": hashlib.sha256(raw).hexdigest(),
        "runtime_bytes": len(raw),
        "internet_enabled_for_exact_package_retrieval": True,
        "write_attempted": False,
    }
    print("CMI_FLU_E11B_API_WRITE_RECEIPT " + json.dumps(before, sort_keys=True))

    try:
        result = api.kernels_push(str(kernel_dir))
    except BaseException as exc:
        receipt = _error_receipt(exc)
        print("CMI_FLU_E11B_API_WRITE_RECEIPT " + json.dumps(receipt, sort_keys=True))
        response = getattr(exc, "response", None)
        body = str(getattr(response, "text", "") or "")
        safe = _safe_server_message(body) or _safe_server_message(str(exc))
        if safe:
            print("CMI_FLU_E11B_API_SAFE_ERROR message=" + safe)
        raise RuntimeError("E11b 004 direct SaveKernel failed; no automatic retry") from exc

    error = str(getattr(result, "error", "") or "")
    print("CMI_FLU_E11B_API_WRITE_RECEIPT " + json.dumps({
        "request_id": REQUEST_ID,
        "phase": "after_api_write",
        "transport": "KaggleApi.kernels_push",
        "write_attempted": True,
        "response_type": type(result).__name__,
        "response_error_present": bool(error),
        "response_error_bytes": len(error.encode("utf-8", errors="replace")),
        "response_error_sha256": prior._digest_text(error),
        "raw_server_error_persisted": False,
    }, sort_keys=True))
    if error:
        safe = _safe_server_message(error)
        if safe:
            print("CMI_FLU_E11B_API_SAFE_ERROR message=" + safe)
        raise RuntimeError("E11b 004 SaveKernel response contained error; no automatic retry")

    direct = prior.base.kernel_meta(api, TARGET)
    if (
        str(getattr(direct, "ref", "")) != TARGET
        or not bool(getattr(direct, "is_private", False))
        or bool(getattr(direct, "enable_gpu", False))
        or bool(getattr(direct, "enable_tpu", False))
        or not bool(getattr(direct, "enable_internet", False))
        or int(getattr(direct, "current_version_number", 0) or 0) != prior.EXPECTED_VERSION
    ):
        raise RuntimeError("E11b 004 direct metadata identity mismatch")
    model_sources = list(getattr(direct, "model_data_sources", []) or [])
    competition_sources = list(getattr(direct, "competition_data_sources", []) or [])
    if prior.MODEL_SOURCE not in model_sources or prior.COMPETITION not in competition_sources:
        raise RuntimeError("E11b 004 source attachment mismatch")
    print(
        "CMI_FLU_E11B_API_IDENTITY_CONFIRMED "
        "version=1 private=true cpu=true internet=true model_attached=true competition_attached=true"
    )


prior.model_access_preflight = model_access_preflight
prior.prewrite_guard = prewrite_guard
prior.push = push


def main() -> int:
    return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
