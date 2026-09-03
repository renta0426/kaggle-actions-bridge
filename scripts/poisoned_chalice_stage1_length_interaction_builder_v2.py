"""Build the frozen Stage 1 length-interaction notebook with current Kaggle mount compatibility.

This is a packaging-only repair of the v1 frozen candidate.  The model, features,
OOF gates, source Dataset, seed, and fit procedure are inherited byte-for-byte from
the pinned v1 builder except for locating an attached Dataset below modern nested
/kaggle/input/datasets/<owner>/<slug> mounts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

BASE_BUILDER = Path(__file__).with_name("poisoned_chalice_stage1_length_interaction_builder.py")
BASE_BUILDER_BLOB_SHA = "a10c7a76be1a4d700d7b1d9846e42348c5711651"
TARGET = "renta0426/stage1-length-interactions-v2"
REQUEST_ID = "20260904-poisoned-chalice-stage1-length-interactions-run-002"
SOURCE_DATASET = "renta0426/stage1-raw-fim-submission-v1-output"
SOURCE_SLUG = "stage1-raw-fim-submission-v1-output"

OLD_LOCATOR = '''def locate_input_root():
    exact = Path("/kaggle/input/stage1-raw-fim-submission-v1-output")
    if exact.is_dir():
        return exact
    root = Path("/kaggle/input")
    candidates = sorted(
        path for path in root.iterdir()
        if path.is_dir() and "stage1-raw-fim-submission-v1-output" in path.name
    )
    if len(candidates) != 1:
        raise RuntimeError(f"unable to identify the one frozen source dataset: {candidates}")
    return candidates[0]
'''

NEW_LOCATOR = '''def locate_input_root():
    root = Path("/kaggle/input")
    if not root.is_dir():
        raise RuntimeError("/kaggle/input is unavailable")

    # Kaggle currently uses both the legacy flat mount and nested dataset mounts
    # such as /kaggle/input/datasets/<owner>/<dataset-slug>.  Resolve by the exact
    # immutable Dataset slug rather than assuming one layout.
    preferred = (
        root / "stage1-raw-fim-submission-v1-output",
        root / "datasets" / "renta0426" / "stage1-raw-fim-submission-v1-output",
        root / "renta0426" / "stage1-raw-fim-submission-v1-output",
    )
    direct = [path for path in preferred if path.is_dir()]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        raise RuntimeError(f"ambiguous frozen source dataset mounts: {direct}")

    candidates = sorted({
        path for path in root.rglob("stage1-raw-fim-submission-v1-output")
        if path.is_dir()
    })
    if len(candidates) != 1:
        immediate = sorted(path.name for path in root.iterdir() if path.is_dir())
        raise RuntimeError(
            "unable to identify the one frozen source dataset: "
            f"candidates={candidates}, immediate_input_dirs={immediate}"
        )
    return candidates[0]
'''


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def load_base():
    data = BASE_BUILDER.read_bytes()
    observed = git_blob_sha(data)
    if observed != BASE_BUILDER_BLOB_SHA:
        raise RuntimeError(
            f"pinned v1 builder changed: {observed} != {BASE_BUILDER_BLOB_SHA}"
        )
    spec = importlib.util.spec_from_file_location("pc_stage1_length_v1", BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load pinned v1 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(output_dir: Path, builder_blob_sha: str) -> dict:
    if len(builder_blob_sha) != 40 or any(c not in "0123456789abcdef" for c in builder_blob_sha):
        raise ValueError("builder_blob_sha must be a 40-character Git blob SHA")

    base = load_base()
    runtime = str(base.RUNTIME_SOURCE)
    if runtime.count(OLD_LOCATOR) != 1:
        raise RuntimeError("pinned v1 input locator no longer matches exactly once")
    runtime = runtime.replace(OLD_LOCATOR, NEW_LOCATOR)

    # Keep all scientific logic inherited from the pinned v1 builder.  Only the
    # request/target identity and mount locator change for this repaired run.
    base.TARGET = TARGET
    base.REQUEST_ID = REQUEST_ID
    base.SOURCE_DATASET = SOURCE_DATASET
    base.RUNTIME_SOURCE = runtime
    result = base.build(output_dir, builder_blob_sha)

    old_notebook = output_dir / "stage1-length-interactions-v1.ipynb"
    new_notebook = output_dir / "stage1-length-interactions-v2.ipynb"
    if not old_notebook.is_file():
        raise RuntimeError("pinned v1 builder did not emit the expected notebook")
    old_notebook.replace(new_notebook)

    metadata_path = output_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["id"] = TARGET
    metadata["title"] = "Stage1 Length Interactions V2"
    metadata["code_file"] = new_notebook.name
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result.update({
        "target": TARGET,
        "request_id": REQUEST_ID,
        "notebook": str(new_notebook),
        "metadata": str(metadata_path),
        "mount_compatibility_fix": True,
        "base_builder_blob_sha": BASE_BUILDER_BLOB_SHA,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--builder-blob-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.builder_blob_sha), sort_keys=True))


if __name__ == "__main__":
    main()
