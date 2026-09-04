"""Exact one-line repair wrapper for controlled-shadow TPU pilot v4 builder.

The approved v5 launcher has one generator bug: while iterating ``lines`` it
inserts the Kaggle TPU environment initializer immediately before the same
matched line without guarding against a second match.  The original line moves
forward and is matched forever.  This wrapper verifies the exact v5 Git blob,
applies exactly one source replacement adding ``not init_injected``, compiles
the repaired launcher in memory, and delegates to its existing CLI.

No Kaggle operation is implemented here.  The only write path remains the one
literal ``kaggle kernels push`` in the pinned v5 launcher.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types

V5_BLOB_SHA = "ea0ee793fb5afd500ab9341332dea534fa771e96"
OLD = "        if line.strip() == '\"from .shadow_xla_compat import install_shadow_xla_compat\\\\n\"':\n"
NEW = "        if (not init_injected) and line.strip() == '\"from .shadow_xla_compat import install_shadow_xla_compat\\\\n\"':\n"


def _blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _load_repaired_v5(path: Path):
    data = path.read_bytes()
    if _blob_sha(data) != V5_BLOB_SHA:
        raise RuntimeError("pinned v5 launcher Git blob mismatch")
    source = data.decode("utf-8")
    if source.count(OLD) != 1 or NEW in source:
        raise RuntimeError("pinned v5 loop-repair anchor changed")
    repaired = source.replace(OLD, NEW, 1)
    if repaired.count(NEW) != 1 or repaired.count(OLD) != 0:
        raise RuntimeError("v5 loop repair was not exactly one replacement")
    compile(repaired, str(path), "exec")
    module = types.ModuleType("shadow_tpu_pilot_v5_repaired")
    module.__file__ = str(path)
    module.__package__ = None
    exec(compile(repaired, str(path), "exec"), module.__dict__)
    return module


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("v5 launcher path is required as first argument")
    v5_path = Path(sys.argv[1]).resolve()
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    module = _load_repaired_v5(v5_path)
    module.main()


if __name__ == "__main__":
    main()
