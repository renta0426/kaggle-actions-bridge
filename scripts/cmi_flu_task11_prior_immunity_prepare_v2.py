#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import py_compile
import subprocess
import sys
from typing import Any

REQUEST_ID = "20260905-cmi-flu-task11-prior-immunity-001"
SCIENCE_COMMIT = "06c85e6263b59cd6ac97b7087779e7a9fb1cbdae"
SCIENCE_BLOB = "50d9a43604d2b75479b8f873a86a8daf9d5bd7a9"
SCIENCE_TRANSPORT = "agent_relay_exact_blob"
RELAYED_SCIENCE_PATH = "payloads/cmi-flu-task11-prior-immunity-001/task11_prior_immunity.py"
TARGET_STAGE = "phase_b_task11_prior_immunity_late_fusion"
BASE_CONDITIONS = ("b1", "b21", "anchor_residual")
FUSION_CONDITIONS = (
    "b1_plus_prior_w0.25",
    "b1_plus_prior_w0.5",
    "b21_plus_prior_w0.25",
    "b21_plus_prior_w0.5",
    "anchor_residual_plus_prior_w0.25",
    "anchor_residual_plus_prior_w0.5",
)


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def validate_relay_contract(root: pathlib.Path, science: pathlib.Path) -> None:
    request = json.loads(
        (root / "requests/cmi-flu-task11-prior-immunity-001.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        "request_id": REQUEST_ID,
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": SCIENCE_TRANSPORT,
        "relayed_science_path": RELAYED_SCIENCE_PATH,
        "task11_prior_immunity_blob_sha": SCIENCE_BLOB,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise SystemExit(f"Task1.1 prior-immunity relay manifest mismatch: {key}")
    expected_path = (root / RELAYED_SCIENCE_PATH).resolve()
    if science != expected_path:
        raise SystemExit("Task1.1 prior-immunity science source is not the approved relay path")
    data = science.read_bytes()
    if git_blob_sha(data) != SCIENCE_BLOB:
        raise SystemExit("Task1.1 prior-immunity relayed science blob mismatch")
    compile(data.decode("utf-8"), RELAYED_SCIENCE_PATH, "exec")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=pathlib.Path, required=True)
    parser.add_argument("--science-source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    science = args.science_source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    validate_relay_contract(root, science)

    # v1 was authored with an anonymous-public-source assumption.  Repair only
    # that transport-validation layer in a runner-local copy.  The science
    # commit/blob and all generated scientific/runtime logic stay unchanged.
    v1_path = root / "scripts/cmi_flu_task11_prior_immunity_prepare.py"
    preparer_text = v1_path.read_text(encoding="utf-8")
    preparer_text = replace_once(
        preparer_text,
        'SCIENCE_TRANSPORT = "pinned_public_raw_presecret"',
        'SCIENCE_TRANSPORT = "agent_relay_exact_blob"',
        label="legacy transport constant",
    )
    preparer_text = replace_once(
        preparer_text,
        '        "science_source_raw_url": SCIENCE_URL,\n',
        f'        "relayed_science_path": "{RELAYED_SCIENCE_PATH}",\n',
        label="legacy request transport field",
    )
    patched_preparer = output.parent / "task11-prior-immunity-prepare-relay.py"
    compile(preparer_text, str(patched_preparer), "exec")
    patched_preparer.write_text(preparer_text, encoding="utf-8")

    base_output = output.parent / "task11-prior-immunity-v1.py"
    subprocess.run(
        [
            sys.executable,
            str(patched_preparer),
            "--repository-root",
            str(root),
            "--science-source",
            str(science),
            "--output",
            str(base_output),
        ],
        check=True,
    )
    text = base_output.read_text(encoding="utf-8")
    anchor = f'TARGET_STAGE = "{TARGET_STAGE}"\n'
    injected = (
        anchor
        + f"BASE_CONDITIONS = {BASE_CONDITIONS!r}\n"
        + f"FUSION_CONDITIONS = {FUSION_CONDITIONS!r}\n"
    )
    text = replace_once(text, anchor, injected, label="runtime condition constants")

    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'SOURCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'TASK11_PRIOR_IMMUNITY_SOURCE_BLOB_SHA = "{SCIENCE_BLOB}"',
        "BASE_CONDITIONS = ('b1', 'b21', 'anchor_residual')",
        "anchor_residual_plus_prior_w0.5",
        "CMI_FLU_TASK11_PRIOR_IMMUNITY_COMPLETE",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"Task1.1 prior-immunity v2 runtime missing tokens: {missing}")
    if "kaggle competitions submit" in text or "api.competition_submit" in text:
        raise SystemExit("Task1.1 prior-immunity v2 runtime contains submission path")

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)
    namespace: dict[str, Any] = {"__name__": "task11_prior_immunity_v2_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("Task1.1 prior-immunity v2 request identity mismatch")
    if namespace.get("SOURCE_COMMIT") != SCIENCE_COMMIT:
        raise SystemExit("Task1.1 prior-immunity v2 science commit mismatch")
    if namespace.get("TASK11_PRIOR_IMMUNITY_SOURCE_BLOB_SHA") != SCIENCE_BLOB:
        raise SystemExit("Task1.1 prior-immunity v2 science blob mismatch")
    if tuple(namespace.get("BASE_CONDITIONS", ())) != BASE_CONDITIONS:
        raise SystemExit("Task1.1 prior-immunity v2 base conditions mismatch")
    if tuple(namespace.get("FUSION_CONDITIONS", ())) != FUSION_CONDITIONS:
        raise SystemExit("Task1.1 prior-immunity v2 fusion conditions mismatch")

    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_TASK11_PRIOR_IMMUNITY_PREPARE_V2 PASS "
        f"science_commit={SCIENCE_COMMIT} science_blob={SCIENCE_BLOB} "
        "science_transport=agent_relay_exact_blob runtime_condition_constants=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
