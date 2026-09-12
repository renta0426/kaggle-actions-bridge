"""Build the frozen STAGE2-DEPLOYMENT-VALIDATION-V1 Kaggle notebook."""
from __future__ import annotations

from pathlib import Path
import json
import textwrap

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TARGET = "renta0426/poisoned-chalice-stage2-deployment-validation-v1"
TITLE = "Poisoned Chalice Stage2 Deployment Validation V1"
NOTEBOOK_NAME = "poisoned-chalice-stage2-deployment-validation-v1.ipynb"
OUT_DIR = ROOT / "notebooks/experiments/poisoned-chalice-stage2-deployment-validation-v1"
OUT = OUT_DIR / NOTEBOOK_NAME

SOURCE_FILES = [
    "src/poisoned_chalice/evaluation.py",
    "src/poisoned_chalice/sersem_author_faithful.py",
    "src/poisoned_chalice/sersem_author_runtime.py",
    "src/poisoned_chalice/lumia_hidden_state_pilot.py",
    "src/poisoned_chalice/lumia_hidden_state_cache.py",
    "src/poisoned_chalice/lumia_hidden_state_scale_cache.py",
    "src/poisoned_chalice/lumia_hidden_state_probe.py",
    "src/poisoned_chalice/stage2_api.py",
    "src/poisoned_chalice/stage2_deployment_validation.py",
    "src/poisoned_chalice/stage2_deployment_evaluation.py",
]
SOURCES = {Path(path).name: (ROOT / path).read_text(encoding="utf-8") for path in SOURCE_FILES}
RUNNER_SOURCE = (ROOT / "scripts/run_stage2_deployment_validation_v1.py").read_text(encoding="utf-8")

install_cell = r'''
from pathlib import Path
import os, shutil, subprocess, sys
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
SITE_ROOT = Path("/tmp/stage2-dv-v1-site")
SRC_ROOT = Path("/tmp/stage2-dv-v1-src")
BIN_ROOT = Path("/tmp/stage2-dv-v1-bin")
RUNNER = Path("/tmp/stage2-dv-v1-runner.py")
for path in (SITE_ROOT, SRC_ROOT, BIN_ROOT):
    if path.exists(): shutil.rmtree(path)
    path.mkdir(parents=True)
RUNNER.unlink(missing_ok=True)
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "libenchant-2-2", "hunspell-en-us"], check=True, stdout=subprocess.DEVNULL)
packages = [
    "datasets==4.6.1", "accelerate==1.13.0", "numpy==2.4.2", "pandas==3.0.1",
    "pyarrow==23.0.1", "scikit-learn==1.8.0", "scipy==1.17.1", "joblib==1.5.3",
    "threadpoolctl==3.6.0", "transformers==4.52.0", "tokenizers==0.21.0",
    "huggingface-hub==0.30.0", "tree-sitter==0.25.2", "tree-sitter-go==0.25.0",
    "tree-sitter-java==0.23.5", "tree-sitter-python==0.25.0", "tree-sitter-ruby==0.23.1",
    "tree-sitter-rust==0.24.0", "pyenchant==3.3.0", "flake8==7.3.0",
    "pycodestyle==2.14.0", "pyflakes==3.4.0", "mccabe==0.7.0",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps", "--force-reinstall", "--target", str(SITE_ROOT), *packages], check=True)
flake8_wrapper = BIN_ROOT / "flake8"
flake8_wrapper.write_text("#!/bin/sh\nexec " + sys.executable + " -m flake8 \"$@\"\n", encoding="utf-8")
flake8_wrapper.chmod(0o755)
print("STAGE2_DV_INSTALL PASS")
'''

materialize_cell = '''
package_root = SRC_ROOT / "poisoned_chalice"
package_root.mkdir(parents=True, exist_ok=True)
(package_root / "__init__.py").write_text("", encoding="utf-8")
SOURCES = __SOURCES__
for name, text in SOURCES.items():
    (package_root / name).write_text(text, encoding="utf-8")
RUNNER_SOURCE = __RUNNER__
RUNNER.write_text(RUNNER_SOURCE, encoding="utf-8")
for path in package_root.glob("*.py"):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
compile(RUNNER_SOURCE, str(RUNNER), "exec")
print("STAGE2_DV_MATERIALIZE PASS files=" + str(len(SOURCES)))
'''.replace("__SOURCES__", repr(SOURCES)).replace("__RUNNER__", repr(RUNNER_SOURCE))

run_cell = r'''
import os, subprocess, sys
child_env = os.environ.copy()
child_env["PYTHONPATH"] = str(SITE_ROOT) + os.pathsep + str(SRC_ROOT)
child_env["PATH"] = str(BIN_ROOT) + os.pathsep + child_env.get("PATH", "")
completed = subprocess.run([sys.executable, str(RUNNER)], env=child_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=18000)
print(completed.stdout)
if completed.returncode != 0:
    raise RuntimeError(f"STAGE2_DEPLOYMENT_VALIDATION_V1_FAILED rc={completed.returncode}")
print("STAGE2_DEPLOYMENT_VALIDATION_V1_NOTEBOOK COMPLETE")
'''

nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell("# STAGE2-DEPLOYMENT-VALIDATION-V1\n\nFrozen source-fit -> label-free deployment -> post-hash evaluation. No competition submission."),
    nbformat.v4.new_code_cell(textwrap.dedent(install_cell).strip()),
    nbformat.v4.new_code_cell(textwrap.dedent(materialize_cell).strip()),
    nbformat.v4.new_code_cell(textwrap.dedent(run_cell).strip()),
]
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
OUT_DIR.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, OUT)
metadata = {
    "id": TARGET,
    "title": TITLE,
    "code_file": NOTEBOOK_NAME,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": ["renta0426/stage1-raw-fim-submission-v1-output"],
    "kernel_sources": ["renta0426/poisoned-chalice-lumia-hidden-state-cache-5000-v1"],
    "competition_sources": [],
}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(str(OUT))
