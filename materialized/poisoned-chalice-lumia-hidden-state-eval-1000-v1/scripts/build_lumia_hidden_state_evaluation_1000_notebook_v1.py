"""Build the frozen P1-03 1,000-row LUMIA hidden-state OOF evaluation Kaggle notebook."""
from pathlib import Path
import textwrap
import json

import nbformat

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SOURCE = (ROOT / "src/poisoned_chalice/evaluation.py").read_text(encoding="utf-8")
FAITHFUL_SOURCE = (ROOT / "src/poisoned_chalice/sersem_author_faithful.py").read_text(encoding="utf-8")
PILOT_SOURCE = (ROOT / "src/poisoned_chalice/lumia_hidden_state_pilot.py").read_text(encoding="utf-8")
CACHE_SOURCE = (ROOT / "src/poisoned_chalice/lumia_hidden_state_cache.py").read_text(encoding="utf-8")
PROBE_SOURCE = (ROOT / "src/poisoned_chalice/lumia_hidden_state_probe.py").read_text(encoding="utf-8")
RUN_SCRIPT_SOURCE = (ROOT / "scripts/run_lumia_hidden_state_probe_evaluation.py").read_text(encoding="utf-8")
EVAL_CONFIG_TEXT = (ROOT / "configs/p1_03_lumia_hidden_state_evaluation_v1_20260910.json").read_text(encoding="utf-8")
SCIENCE_CONFIG_TEXT = (ROOT / "configs/p1_03_lumia_hidden_state_pilot_v1_20260909.json").read_text(encoding="utf-8")
EXPECTED_CACHE_MANIFEST_TEXT = (
    ROOT
    / "experiments/p1-03-lumia-hidden-state-cache-1000-v1"
    / "lumia_hidden_state_cache_1000_v1"
    / "cache_manifest.json"
).read_text(encoding="utf-8")

TARGET = "renta0426/poisoned-chalice-lumia-hidden-state-eval-1000-v1"
TITLE = "Poisoned Chalice Lumia Hidden State Eval 1000 V1"
CACHE_KERNEL = "renta0426/poisoned-chalice-lumia-hidden-state-cache-1000-v1"
CACHE_INPUT_SLUG = "poisoned-chalice-lumia-hidden-state-cache-1000-v1"
NOTEBOOK_NAME = "poisoned-chalice-lumia-hidden-state-eval-1000-v1.ipynb"
OUT_DIR = ROOT / "notebooks/experiments/poisoned-chalice-lumia-hidden-state-eval-1000-v1"
OUT = OUT_DIR / NOTEBOOK_NAME

install_cell = r'''
from pathlib import Path
import os
import shutil
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SITE_ROOT = Path("/tmp/lumia-eval-1000-v1-site")
SRC_ROOT = Path("/tmp/lumia-eval-1000-v1-src")
RUNNER = SRC_ROOT / "scripts/run_lumia_hidden_state_probe_evaluation.py"
OUTPUT = Path("/kaggle/working/lumia_hidden_state_evaluation_1000_v1")
CACHE_ROOT = Path("/kaggle/input/poisoned-chalice-lumia-hidden-state-cache-1000-v1/lumia_hidden_state_cache_1000_v1")
for path in (SITE_ROOT, SRC_ROOT):
    if path.exists():
        shutil.rmtree(path)
if OUTPUT.exists():
    raise RuntimeError("fresh evaluation target must not already contain output")
SITE_ROOT.mkdir(parents=True)
SRC_ROOT.mkdir(parents=True)

packages = [
    "datasets==4.6.1",
    "numpy==2.4.2",
    "pandas==3.0.1",
    "pyarrow==23.0.1",
    "scikit-learn==1.8.0",
    "scipy==1.17.1",
    "joblib==1.5.3",
    "threadpoolctl==3.6.0",
    "huggingface-hub==0.30.0",
]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--target", str(SITE_ROOT), *packages],
    check=True,
)
probe = r"""
import importlib.metadata
import pathlib
import numpy, pandas, pyarrow, scipy, sklearn, datasets, torch
site = pathlib.Path(__import__('os').environ['LUMIA_SITE_ROOT']).resolve()
expected = {
    'numpy':'2.4.2','pandas':'3.0.1','pyarrow':'23.0.1','scipy':'1.17.1',
    'scikit-learn':'1.8.0','datasets':'4.6.1','huggingface-hub':'0.30.0'
}
for dist, version in expected.items():
    assert importlib.metadata.version(dist) == version, (dist, importlib.metadata.version(dist))
for module in (numpy, pandas, pyarrow, scipy, sklearn, datasets):
    assert site in pathlib.Path(module.__file__).resolve().parents, (module.__name__, module.__file__)
assert site not in pathlib.Path(torch.__file__).resolve().parents
print('LUMIA_EVAL_1000_IMPORT_GATE PASS')
"""
env = os.environ.copy()
env["PYTHONPATH"] = str(SITE_ROOT)
env["LUMIA_SITE_ROOT"] = str(SITE_ROOT)
completed = subprocess.run([sys.executable, "-c", probe], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
print(completed.stdout)
if completed.returncode != 0:
    raise RuntimeError(f"LUMIA_EVAL_1000_IMPORT_GATE_FAILED rc={completed.returncode}")
# Keep the notebook-side label-materialization imports on the same pinned stack
# as the evaluation child. System Torch intentionally remains outside SITE_ROOT.
sys.path.insert(0, str(SITE_ROOT))
'''

write_sources_cell = r'''
package_root = SRC_ROOT / "poisoned_chalice"
config_root = SRC_ROOT / "configs"
data_root = SRC_ROOT / "data/public"
scripts_root = SRC_ROOT / "scripts"
package_root.mkdir(parents=True, exist_ok=True)
config_root.mkdir(parents=True, exist_ok=True)
scripts_root.mkdir(parents=True, exist_ok=True)
(package_root / "__init__.py").write_text("", encoding="utf-8")
(package_root / "evaluation.py").write_text(EVALUATION_SOURCE_TEXT, encoding="utf-8")
(package_root / "sersem_author_faithful.py").write_text(FAITHFUL_SOURCE_TEXT, encoding="utf-8")
(package_root / "lumia_hidden_state_pilot.py").write_text(PILOT_SOURCE_TEXT, encoding="utf-8")
(package_root / "lumia_hidden_state_cache.py").write_text(CACHE_SOURCE_TEXT, encoding="utf-8")
(package_root / "lumia_hidden_state_probe.py").write_text(PROBE_SOURCE_TEXT, encoding="utf-8")
RUNNER.write_text(RUN_SCRIPT_SOURCE_TEXT, encoding="utf-8")
(config_root / "p1_03_lumia_hidden_state_evaluation_v1_20260910.json").write_text(EVAL_CONFIG_TEXT_VALUE, encoding="utf-8")
(config_root / "p1_03_lumia_hidden_state_pilot_v1_20260909.json").write_text(SCIENCE_CONFIG_TEXT_VALUE, encoding="utf-8")
'''

execute_cell = r'''
import hashlib
import json
import time

EXPECTED_CACHE_MANIFEST_SHA256 = hashlib.sha256(EXPECTED_CACHE_MANIFEST_TEXT_VALUE.encode("utf-8")).hexdigest()
if not CACHE_ROOT.is_dir():
    raise RuntimeError(f"expected kernel input is missing: {CACHE_ROOT}")
observed_manifest = CACHE_ROOT / "cache_manifest.json"
if not observed_manifest.is_file():
    raise RuntimeError("mounted cache manifest missing")
observed_manifest_sha256 = hashlib.sha256(observed_manifest.read_bytes()).hexdigest()
if observed_manifest_sha256 != EXPECTED_CACHE_MANIFEST_SHA256:
    raise RuntimeError(
        "mounted cache is not the exact sealed cache recorded in science Git: "
        f"observed={observed_manifest_sha256} expected={EXPECTED_CACHE_MANIFEST_SHA256}"
    )
manifest = json.loads(observed_manifest.read_text(encoding="utf-8"))
if manifest.get("rows") != 1000 or len(manifest.get("shards") or []) != 20:
    raise RuntimeError("mounted cache row/shard contract changed")
if manifest.get("labels_present_during_model_scoring") is not False or manifest.get("performance_metrics_computed") is not False:
    raise RuntimeError("mounted cache clean-room boundary changed")

# Reconstruct only the public training label table from the same pinned public HF revision.
from datasets import load_dataset
import pandas as pd
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
EXPECTED_ROWS = {"Go":10000,"Java":10000,"Python":10000,"Ruby":10000,"Rust":9040}
DATASET_ID = "Poisoned-Chalice/ICSE-2027-public"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
for language in LANGUAGES:
    frame = load_dataset(DATASET_ID, language, split="train", revision=DATASET_REVISION).to_pandas()
    if len(frame) != EXPECTED_ROWS[language]:
        raise RuntimeError(f"public train row count changed for {language}: {len(frame)}")
    if not {"sample_id", "membership"}.issubset(frame.columns):
        raise RuntimeError(f"public label columns missing for {language}")
    label_dir = data_root / language
    label_dir.mkdir(parents=True, exist_ok=True)
    frame[["sample_id", "membership"]].to_parquet(label_dir / "train-00000-of-00001.parquet", index=False)
    del frame

import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the frozen 1k probe evaluation run")
if "T4" not in torch.cuda.get_device_name(0):
    raise RuntimeError(f"expected T4 on cuda:0, got {torch.cuda.get_device_name(0)}")

env = os.environ.copy()
env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + str(SITE_ROOT)
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
started = time.perf_counter()
completed = subprocess.run(
    [sys.executable, str(RUNNER), "--cache-dir", str(CACHE_ROOT), "--output-dir", str(OUTPUT), "--device", "cuda:0"],
    cwd=str(SRC_ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    timeout=6900,
)
print(completed.stdout)
if completed.returncode != 0:
    raise RuntimeError(f"LUMIA_EVAL_1000_CHILD_FAILED rc={completed.returncode}")
expected_outputs = ["decision.json", "fold_selection.json", "metrics.json", "oof_predictions.csv", "run_manifest.json"]
observed_outputs = sorted(path.name for path in OUTPUT.iterdir())
if observed_outputs != expected_outputs:
    raise RuntimeError(f"unexpected persistent output set: {observed_outputs}")
run_manifest = json.loads((OUTPUT / "run_manifest.json").read_text(encoding="utf-8"))
if run_manifest.get("status") != "complete" or run_manifest.get("rows") != 1000 or run_manifest.get("outer_oof_coverage") is not True:
    raise RuntimeError("frozen OOF evaluation did not complete with full coverage")
print(
    "LUMIA_HIDDEN_STATE_EVAL_1000_V1 COMPLETE "
    f"rows=1000 cache_sha256={observed_manifest_sha256} wall_seconds={time.perf_counter()-started:.3f}"
)
'''

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.12"}
nb.cells = [
    nbformat.v4.new_markdown_cell(
        "# P1-03 LUMIA hidden-state — frozen 1,000-row OOF evaluation\n\n"
        "This notebook consumes the exact completed cache notebook as a Kaggle kernel input. "
        "Cache integrity is verified before public training membership labels are reconstructed from the pinned public HF revision. "
        "The already-frozen 5-fold outer-OOF evaluator is then executed without changing probe hyperparameters, layer selection, metrics, bootstrap, or scale gate. "
        "No validation rows or Public-LB feedback are used and no competition submission is created."
    ),
    nbformat.v4.new_code_cell(textwrap.dedent(install_cell)),
    nbformat.v4.new_code_cell(
        "EVALUATION_SOURCE_TEXT = " + repr(EVALUATION_SOURCE) + "\n"
        "FAITHFUL_SOURCE_TEXT = " + repr(FAITHFUL_SOURCE) + "\n"
        "PILOT_SOURCE_TEXT = " + repr(PILOT_SOURCE) + "\n"
        "CACHE_SOURCE_TEXT = " + repr(CACHE_SOURCE) + "\n"
        "PROBE_SOURCE_TEXT = " + repr(PROBE_SOURCE) + "\n"
        "RUN_SCRIPT_SOURCE_TEXT = " + repr(RUN_SCRIPT_SOURCE) + "\n"
        "EVAL_CONFIG_TEXT_VALUE = " + repr(EVAL_CONFIG_TEXT) + "\n"
        "SCIENCE_CONFIG_TEXT_VALUE = " + repr(SCIENCE_CONFIG_TEXT) + "\n"
        "EXPECTED_CACHE_MANIFEST_TEXT_VALUE = " + repr(EXPECTED_CACHE_MANIFEST_TEXT) + "\n"
        + textwrap.dedent(write_sources_cell)
    ),
    nbformat.v4.new_code_cell(textwrap.dedent(execute_cell)),
]
for index, cell in enumerate(nb.cells):
    cell["id"] = f"lumia-eval-1000-v1-{index:02d}"

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
    "enable_tpu": False,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [],
    "kernel_sources": [CACHE_KERNEL],
    "competition_sources": [],
    "model_sources": [],
}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(OUT)
