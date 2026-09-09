#!/usr/bin/env python3
"""E07 builder v2: adapt final output creation to Kaggle's existing working directory."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = "scripts/cmi_flu_strategy_e07_prepare.py"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    a = args()
    root = a.repository_root.resolve()
    output = a.output.resolve()
    with tempfile.TemporaryDirectory(prefix="cmi-e07-v2-") as tmp:
        base = Path(tmp) / "runtime.py"
        subprocess.run(
            [sys.executable, str(root / BASE), "--repository-root", str(root), "--output", str(base)],
            check=True,
        )
        runtime = base.read_text(encoding="utf-8")
    old = '    output_dir.mkdir(parents=True, exist_ok=False)\n    metrics = output_dir / "metrics.json"\n'
    new = '''    if synthetic:
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        for _name in ("bridge-result.json", "metrics.json", "summary.md"):
            if (output_dir / _name).exists():
                raise RuntimeError("E07 declared output already exists")
    metrics = output_dir / "metrics.json"
'''
    if runtime.count(old) != 1:
        raise SystemExit("E07 v2 output creation anchor changed")
    runtime = runtime.replace(old, new, 1)
    compile(runtime, "generated_e07_runtime_v2.py", "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print("CMI_FLU_E07_PREPARE_V2 PASS kaggle_working_existing_dir=true declared_outputs_must_not_preexist=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
