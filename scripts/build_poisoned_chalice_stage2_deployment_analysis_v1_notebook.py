from __future__ import annotations

import json
import sys
from pathlib import Path

TARGET = "renta0426/poisoned-chalice-stage2-deployment-analysis-v1"
TITLE = "Poisoned Chalice Stage2 Deployment Analysis V1"
NOTEBOOK_NAME = "poisoned-chalice-stage2-deployment-analysis-v1.ipynb"
SOURCE_KERNEL = "renta0426/poisoned-chalice-stage2-deployment-validation-v1"
SOURCE_SCRIPT_VERSION_ID = 349274389
OUT_DIR = Path("notebooks/poisoned-chalice-stage2-deployment-analysis-v1")


def code_cell(source: str, cell_id: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cell_id, "metadata": {}, "outputs": [], "source": source}


def markdown_cell(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source}


def main() -> int:
    analyzer_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scripts/poisoned_chalice_stage2_deployment_analysis_v2.py")
    analysis_source = analyzer_path.read_text(encoding="utf-8")
    compile(analysis_source, str(analyzer_path), "exec")
    install = '''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "datasets==4.6.1"], check=True)
print("STAGE2_ANALYSIS_V2_INSTALL PASS")'''
    materialize = (
        "from pathlib import Path\n"
        "ANALYZER = Path('/tmp/poisoned_chalice_stage2_deployment_analysis_v2.py')\n"
        "ANALYZER.write_text(__SOURCE__, encoding='utf-8')\n"
        "compile(ANALYZER.read_text(encoding='utf-8'), str(ANALYZER), 'exec')\n"
        "print('STAGE2_ANALYSIS_V2_MATERIALIZE PASS')\n"
    ).replace("__SOURCE__", repr(analysis_source))
    run = '''import subprocess, sys
completed = subprocess.run([sys.executable, "/tmp/poisoned_chalice_stage2_deployment_analysis_v2.py"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=3600)
print(completed.stdout)
if completed.returncode != 0:
    raise RuntimeError(f"STAGE2_DEPLOYMENT_ANALYSIS_V1_V2_FAILED rc={completed.returncode}")
print("STAGE2_DEPLOYMENT_ANALYSIS_V1_NOTEBOOK COMPLETE")'''
    notebook = {
        "cells": [
            markdown_cell(
                "# STAGE2-DEPLOYMENT-ANALYSIS-V1 — repair version 2\n\n"
                "CPU-only analysis of the exact successful Stage2 deployment output. "
                "This version closes the analysis-v1 fusion-reconstruction bug and evaluates every frozen candidate from its persisted score column.",
                "intro",
            ),
            code_cell(install, "install"),
            code_cell(materialize, "materialize"),
            code_cell(run, "run"),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
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
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [SOURCE_KERNEL],
        "competition_sources": [],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / NOTEBOOK_NAME).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
