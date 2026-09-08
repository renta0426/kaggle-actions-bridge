#!/usr/bin/env python3
"""Regression for E06b saved-output pollution by locked reference staging."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "cmi_flu_strategy_e05_prepare_v3.py"
READER = ROOT / "scripts" / "kaggle_current_output_read.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("cmi_e05_v3_hygiene", BUILDER)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load E05 v3 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    builder = load_builder()
    marker = f"LOCKED_REFERENCE_SHA256 = {builder.REF_SHA256!r}\n"
    runtime = f'''from pathlib import Path\nimport base64, hashlib\nREQUEST = "{builder.OLD_REQUEST_ID}"\nTARGET = "{builder.OLD_TARGET_KERNEL}"\n{marker}\ndef json_safe(value):\n    return value\n\ndef locate_locked_reference(name):\n    return Path(name)\n'''
    references = {name: b"fixture" for name in builder.REF_SHA256}
    patched = builder.patch_runtime(runtime, references)

    if 'Path("/tmp")' not in patched:
        raise SystemExit("locked references are not staged under /tmp")
    if "/kaggle/working/.e05-locked-references" in patched:
        raise SystemExit("legacy saved-output staging path remains")
    if 'base / ".e05-locked-references"' in patched:
        raise SystemExit("legacy hidden working-directory staging remains")
    if 'root = base / "cmi-flu-e05-locked-references"' not in patched:
        raise SystemExit("new transient staging directory contract missing")

    reader = READER.read_text(encoding="utf-8")
    if "unexpected_saved_output_count" not in reader:
        raise SystemExit("strict current-output allowlist was weakened")

    print(
        "CMI_FLU_LOCKED_REFERENCE_OUTPUT_HYGIENE_PASS "
        "staging=/tmp saved_working=false reader_allowlist=strict"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
