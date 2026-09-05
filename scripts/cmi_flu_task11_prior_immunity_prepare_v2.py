#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import py_compile
import subprocess
import sys
from typing import Any

REQUEST_ID = "20260905-cmi-flu-task11-prior-immunity-001"
SCIENCE_COMMIT = "06c85e6263b59cd6ac97b7087779e7a9fb1cbdae"
SCIENCE_BLOB = "50d9a43604d2b75479b8f873a86a8daf9d5bd7a9"
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


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


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

    base_output = output.parent / "task11-prior-immunity-v1.py"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/cmi_flu_task11_prior_immunity_prepare.py"),
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
    if tuple(namespace.get("BASE_CONDITIONS", ())) != BASE_CONDITIONS:
        raise SystemExit("Task1.1 prior-immunity v2 base conditions mismatch")
    if tuple(namespace.get("FUSION_CONDITIONS", ())) != FUSION_CONDITIONS:
        raise SystemExit("Task1.1 prior-immunity v2 fusion conditions mismatch")

    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_TASK11_PRIOR_IMMUNITY_PREPARE_V2 PASS "
        f"science_commit={SCIENCE_COMMIT} science_blob={SCIENCE_BLOB} "
        "runtime_condition_constants=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
