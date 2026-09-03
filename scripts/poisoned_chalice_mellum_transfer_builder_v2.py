"""Apply one audited source-contract correction to the frozen Mellum transfer builder.

The request-002 builder expected the generated feature name
``min_kpp_zselect_10`` to occur literally in the FeatureCacheV2 source.  The
actual frozen source correctly creates that name from the f-string template
``min_kpp_zselect_{percent:02d}``, so the old check rejected the valid source
Notebook before any Kaggle write.  This wrapper verifies the exact request-002
builder blob, changes only that static marker, compiles it, and invokes its
existing build() implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

BASE_BUILDER_BLOB = "0b8b48b2547454affbde6e730d4e43bde39421cf"
OLD_CHECK = '"standard_minkpp": "min_kpp_zselect_10" in feature_cache,'
NEW_CHECK = '"standard_minkpp": "min_kpp_zselect_{percent:02d}" in feature_cache,'


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def load_patched_builder(base_path: Path):
    raw = base_path.read_bytes()
    if git_blob_sha(raw) != BASE_BUILDER_BLOB:
        raise RuntimeError("BASE_BUILDER_BLOB_MISMATCH")
    source = raw.decode("utf-8")
    if source.count(OLD_CHECK) != 1:
        raise RuntimeError("BASE_BUILDER_CHECK_SITE_CHANGED")
    if NEW_CHECK in source:
        raise RuntimeError("BASE_BUILDER_ALREADY_PATCHED")
    patched = source.replace(OLD_CHECK, NEW_CHECK, 1)
    if patched.count(NEW_CHECK) != 1 or OLD_CHECK in patched:
        raise RuntimeError("BUILDER_PATCH_CARDINALITY_FAILED")
    compile(patched, str(base_path), "exec")
    namespace = {
        "__name__": "poisoned_chalice_mellum_transfer_builder_patched",
        "__file__": str(base_path),
    }
    exec(compile(patched, str(base_path), "exec"), namespace)
    build = namespace.get("build")
    if not callable(build):
        raise RuntimeError("BASE_BUILDER_BUILD_MISSING")
    return build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-notebook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-builder",
        type=Path,
        default=None,
        help="Exact request-002 builder; defaults to MELLUM_BASE_BUILDER.",
    )
    args = parser.parse_args()
    base = args.base_builder
    if base is None:
        value = os.environ.get("MELLUM_BASE_BUILDER", "")
        if not value:
            raise RuntimeError("BASE_BUILDER_PATH_UNSET")
        base = Path(value)
    build = load_patched_builder(base)
    build(args.source_notebook, args.output_dir)
    print("MELLUM_BUILDER_V2 PASS correction=generated_fstring_marker_only")


if __name__ == "__main__":
    main()
