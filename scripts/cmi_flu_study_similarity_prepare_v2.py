#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import py_compile
import subprocess
import sys

REQUEST_ID = "20260905-cmi-flu-study-similarity-001"
SCIENCE_COMMIT = "f47ee36ce8933895b522de0ec402c75c7fe517a7"
SCIENCE_BLOB = "27351df3d9187899c4bce2ff1a24b06efc160185"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    base = output.parent / "study-similarity-v1.py"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/cmi_flu_study_similarity_prepare.py"),
            "--repository-root",
            str(root),
            "--output",
            str(base),
        ],
        check=True,
    )
    text = base.read_text(encoding="utf-8")

    old = '''            for condition in ("reference", "uniform", "similarity_weighted"):\n                if int((summary[condition] or {}).get("count", -1)) != expected_folds[task]:\n                    raise BundleContractError(f"summary count mismatch: {task}/{condition}")\n'''
    new = '''            for condition in ("reference", "uniform", "similarity_weighted"):\n                finite_count = int((summary[condition] or {}).get("count", -1))\n                if finite_count < 1 or finite_count > expected_folds[task]:\n                    raise BundleContractError(\n                        f"summary finite-count mismatch: {task}/{condition} count={finite_count}"\n                    )\n'''
    text = replace_once(
        text,
        old,
        new,
        label="usable-fold summary validation",
    )

    required = (
        f'REQUEST_ID = "{REQUEST_ID}"',
        f'SOURCE_COMMIT = "{SCIENCE_COMMIT}"',
        f'STUDY_SIMILARITY_SOURCE_BLOB_SHA = "{SCIENCE_BLOB}"',
        "finite_count < 1 or finite_count > expected_folds[task]",
        "CMI_FLU_STUDY_SIMILARITY_COMPLETE",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(f"study-similarity v2 runtime missing tokens: {missing}")
    if "kaggle competitions submit" in text or "api.competition_submit" in text:
        raise SystemExit("study-similarity v2 runtime contains submission path")

    compile(text, str(output), "exec")
    output.write_text(text, encoding="utf-8")
    py_compile.compile(str(output), doraise=True)
    namespace: dict[str, object] = {"__name__": "study_similarity_v2_static_preflight"}
    exec(compile(text, str(output), "exec"), namespace, namespace)
    if namespace.get("REQUEST_ID") != REQUEST_ID:
        raise SystemExit("study-similarity v2 request mismatch")
    if namespace.get("SOURCE_COMMIT") != SCIENCE_COMMIT:
        raise SystemExit("study-similarity v2 science commit mismatch")
    if namespace.get("STUDY_SIMILARITY_SOURCE_BLOB_SHA") != SCIENCE_BLOB:
        raise SystemExit("study-similarity v2 science blob mismatch")

    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_STUDY_SIMILARITY_PREPARE_V2 PASS "
        f"science_commit={SCIENCE_COMMIT} science_blob={SCIENCE_BLOB} "
        "usable_fold_validation=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
