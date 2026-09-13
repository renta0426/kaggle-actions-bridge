#!/usr/bin/env python3
"""Build the frozen PYTHIA-MIMIR-DEVELOPMENT-V1 Kaggle notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
TARGET = "renta0426/poisoned-chalice-pythia-mimir-development-v1"
TITLE = "Poisoned Chalice Pythia MIMIR Development V1"
NOTEBOOK_NAME = "poisoned-chalice-pythia-mimir-development-v1.ipynb"
OUT_DIR = ROOT / "notebooks/experiments/poisoned-chalice-pythia-mimir-development-v1"
CONFIG_PATH = ROOT / "configs/poisoned-chalice-pythia-mimir-development-v1.json"
RUNNER_PATH = ROOT / "runners/poisoned_chalice/pythia_mimir_development_v1.py"
ENTRY_PATH = ROOT / "runners/poisoned_chalice/pythia_mimir_development_v1_entry.py"
SCIENCE_SOURCES = {
    "stage2_api.py": ROOT / "materialized/poisoned-chalice-stage2-deployment-validation-v1/src/poisoned_chalice/stage2_api.py",
    "evaluation.py": ROOT / "materialized/poisoned-chalice-stage2-deployment-validation-v1/src/poisoned_chalice/evaluation.py",
}
EXPECTED_SCIENCE_BLOBS = {
    "stage2_api.py": "16578d3e01b6471e5bfb2b04cebb99e1b3126c7c",
    "evaluation.py": "cb3afd4f3e3ecafe0e0677eda4fda30e93b04ba8",
}


def git_blob_sha(data: bytes) -> str:
    import hashlib
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def cell(cell_type: str, source: str) -> dict:
    base = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def load_sources() -> tuple[dict[str, str], str, str, str]:
    sources: dict[str, str] = {}
    for name, path in SCIENCE_SOURCES.items():
        data = path.read_bytes()
        observed = git_blob_sha(data)
        if observed != EXPECTED_SCIENCE_BLOBS[name]:
            raise RuntimeError(f"science source blob mismatch: {name} {observed}")
        text = data.decode("utf-8")
        compile(text, name, "exec")
        sources[name] = text
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    entry = ENTRY_PATH.read_text(encoding="utf-8")
    config = CONFIG_PATH.read_text(encoding="utf-8")
    compile(runner, str(RUNNER_PATH), "exec")
    compile(entry, str(ENTRY_PATH), "exec")
    parsed = json.loads(config)
    if parsed["experiment_id"] != "PYTHIA-MIMIR-DEVELOPMENT-V1":
        raise RuntimeError("experiment config identity changed")
    if parsed["source_bundle"]["bundle_manifest_sha256"] is None:
        # Static implementation is intentionally allowed before the upstream
        # result identity exists; runtime self-test/launch remains blocked.
        pass
    return sources, runner, entry, config


def build() -> Path:
    sources, runner, entry, config_text = load_sources()
    install_source = r'''
from pathlib import Path
import os, shutil, subprocess, sys
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
SITE_ROOT = Path("/tmp/pythia-mimir-dev-v1-site")
SRC_ROOT = Path("/tmp/pythia-mimir-dev-v1-src")
BIN_ROOT = Path("/tmp/pythia-mimir-dev-v1-bin")
CORE_RUNNER = Path("/tmp/pythia_mimir_development_v1.py")
ENTRY_RUNNER = Path("/tmp/pythia_mimir_development_v1_entry.py")
CONFIG = Path("/tmp/pythia-mimir-development-v1.json")
for path in (SITE_ROOT, SRC_ROOT, BIN_ROOT):
    if path.exists(): shutil.rmtree(path)
    path.mkdir(parents=True)
for path in (CORE_RUNNER, ENTRY_RUNNER, CONFIG):
    path.unlink(missing_ok=True)
packages = [
    "accelerate==1.13.0",
    "numpy==2.4.2",
    "pandas==3.0.1",
    "scikit-learn==1.8.0",
    "scipy==1.17.1",
    "joblib==1.5.3",
    "threadpoolctl==3.6.0",
    "transformers==4.52.0",
    "tokenizers==0.21.0",
    "huggingface-hub==0.30.0",
]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps", "--force-reinstall", "--target", str(SITE_ROOT), *packages],
    check=True,
)
print("PYTHIA_MIMIR_DEV_INSTALL PASS")
'''

    materialize_source = '''
package_root = SRC_ROOT / "poisoned_chalice"
package_root.mkdir(parents=True, exist_ok=True)
(package_root / "__init__.py").write_text("", encoding="utf-8")
SCIENCE_SOURCES = __SCIENCE_SOURCES__
for name, text in SCIENCE_SOURCES.items():
    path = package_root / name
    path.write_text(text, encoding="utf-8")
    compile(text, str(path), "exec")
CORE_SOURCE = __CORE_SOURCE__
ENTRY_SOURCE = __ENTRY_SOURCE__
CONFIG_SOURCE = __CONFIG_SOURCE__
CORE_RUNNER.write_text(CORE_SOURCE, encoding="utf-8")
ENTRY_RUNNER.write_text(ENTRY_SOURCE, encoding="utf-8")
CONFIG.write_text(CONFIG_SOURCE, encoding="utf-8")
compile(CORE_SOURCE, str(CORE_RUNNER), "exec")
compile(ENTRY_SOURCE, str(ENTRY_RUNNER), "exec")
import json
cfg = json.loads(CONFIG_SOURCE)
assert cfg["experiment_id"] == "PYTHIA-MIMIR-DEVELOPMENT-V1"
assert cfg["automatic_compute_retries"] == 0
assert cfg["competition_submission"] is False
assert cfg["select_as_final"] is False
assert cfg["clean_room"]["candidate_predictions_hashed_before_target_label_load"] is True
print("PYTHIA_MIMIR_DEV_MATERIALIZE PASS sources=" + str(len(SCIENCE_SOURCES)))
'''.replace("__SCIENCE_SOURCES__", repr(sources)).replace("__CORE_SOURCE__", repr(runner)).replace("__ENTRY_SOURCE__", repr(entry)).replace("__CONFIG_SOURCE__", repr(config_text))

    run_source = r'''
import os, subprocess, sys
child_env = os.environ.copy()
child_env["PYTHONPATH"] = str(SITE_ROOT) + os.pathsep + str(SRC_ROOT)
child_env["PATH"] = str(BIN_ROOT) + os.pathsep + child_env.get("PATH", "")
completed = subprocess.run(
    [sys.executable, str(ENTRY_RUNNER), "--config", str(CONFIG)],
    env=child_env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    timeout=21600,
)
print(completed.stdout)
if completed.returncode != 0:
    raise RuntimeError(f"PYTHIA_MIMIR_DEVELOPMENT_V1_FAILED rc={completed.returncode}")
print("PYTHIA_MIMIR_DEVELOPMENT_V1_NOTEBOOK COMPLETE")
'''

    notebook = {
        "cells": [
            cell("markdown", "# PYTHIA-MIMIR-DEVELOPMENT-V1\n\nFrozen reusable model-transfer development environment. Candidate predictions are sealed before membership-label evaluation. No competition submission."),
            cell("code", textwrap.dedent(install_source).strip()),
            cell("code", textwrap.dedent(materialize_source).strip()),
            cell("code", textwrap.dedent(run_source).strip()),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": TARGET,
        "title": TITLE,
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "kernel_sources": ["renta0426/poisoned-chalice-stage2-deployment-validation-v1"],
        "competition_sources": [],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / NOTEBOOK_NAME
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-output", action="store_true")
    args = parser.parse_args()
    output = build()
    if args.print_output:
        print(output)
    else:
        print("PYTHIA_MIMIR_DEVELOPMENT_V1_BUILD PASS " + str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
